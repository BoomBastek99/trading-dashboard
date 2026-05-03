"""
Regime-Based Market Screener.

Scans a universe of tickers, runs HMM regime detection on each (forward-only,
no look-ahead, BIC-selected K in 3..5), and surfaces the cleanest setups -
where regime, trend filter, and volume trend all align.

Run with:
    py -3.13 -m streamlit run regime_market_screener.py --server.port=8508
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from data_provider import clear_cache as _clear_data_cache, get_history_recent
from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Regime Market Screener",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def render_html(s: str) -> None:
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)


# ── Universe ──────────────────────────────────────────────────────────────

LARGE_CAP = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]
ETFS      = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "TLT", "HYG"]
CRYPTO    = ["BTC-USD", "ETH-USD", "SOL-USD"]

CATEGORY = {
    **{t: "Large Cap" for t in LARGE_CAP},
    **{t: "ETF" for t in ETFS},
    **{t: "Crypto" for t in CRYPTO},
}


def categorize(ticker: str) -> str:
    return CATEGORY.get(ticker, "Other")


# ── Data + features ──────────────────────────────────────────────────────

FEATURE_COLS = ["log_ret", "realized_vol", "volume_ratio", "hl_range_pct"]


def fetch_history(ticker: str, lookback_days: int = 504) -> pd.DataFrame:
    """Thin wrapper around the unified data provider (FMP → TwelveData → yfinance)."""
    return get_history_recent(ticker, lookback_days)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    out["log_ret"] = np.log(close).diff()
    out["realized_vol"] = out["log_ret"].rolling(20).std()

    if vol.sum() > 0:
        avg_vol = vol.rolling(20).mean()
        out["volume_ratio"] = (vol / avg_vol).replace([np.inf, -np.inf], np.nan)
    else:
        abs_ret = out["log_ret"].abs()
        out["volume_ratio"] = abs_ret.rolling(5).mean() / (abs_ret.rolling(60).mean() + 1e-9)

    out["hl_range_pct"] = (high - low) / close
    out["close"] = close
    out["volume"] = vol
    return out.dropna()


# ── HMM (forward-only) ───────────────────────────────────────────────────

def fit_hmm(X: np.ndarray, n_components: int, seed: int = 42) -> GaussianHMM:
    m = GaussianHMM(
        n_components=n_components, covariance_type="diag",
        n_iter=120, tol=1e-3, random_state=seed,
    )
    m.fit(X)
    return m


def bic(model: GaussianHMM, X: np.ndarray) -> float:
    n, d = X.shape
    k = model.n_components
    n_params = (k - 1) + k * (k - 1) + 2 * k * d
    return -2.0 * model.score(X) + n_params * np.log(n)


def select_best_model(X: np.ndarray, candidates=range(3, 6)):
    best, best_n, best_bic = None, None, np.inf
    for n in candidates:
        try:
            m = fit_hmm(X, n)
            b = bic(m, X)
            if b < best_bic:
                best, best_n, best_bic = m, n, b
        except Exception:
            pass
    return best, best_n


def emission_logprob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    T, K = X.shape[0], model.n_components
    out = np.empty((T, K))
    for k in range(K):
        out[:, k] = multivariate_normal.logpdf(X, mean=model.means_[k], cov=model.covars_[k])
    return out


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    T = X.shape[0]
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    log_b = emission_logprob(model, X)
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def vol_label_for(rank: int, K: int) -> str:
    """Map a vol-sorted rank into one of three buckets regardless of K."""
    frac = rank / max(K - 1, 1)
    if frac < 0.34:
        return "Low Vol"
    if frac < 0.67:
        return "Medium Vol"
    return "High Vol"


@st.cache_data(ttl=1800, show_spinner=False)
def scan_ticker(ticker: str, lookback_days: int = 504) -> dict:
    df = fetch_history(ticker, lookback_days)
    if df.empty:
        return {"ok": False, "ticker": ticker, "error": "no data"}
    feat = build_features(df)
    if len(feat) < 100:
        return {"ok": False, "ticker": ticker, "error": f"only {len(feat)} bars"}

    X = feat[FEATURE_COLS].values
    model, K = select_best_model(X)
    if model is None:
        return {"ok": False, "ticker": ticker, "error": "HMM failed"}

    filtered = forward_filter(model, X)
    vol_idx = FEATURE_COLS.index("realized_vol")
    order = np.argsort(model.means_[:, vol_idx])
    raw_to_label = {int(raw): vol_label_for(r, K) for r, raw in enumerate(order)}

    raw_states = filtered.argmax(axis=1)
    labels = np.array([raw_to_label[int(s)] for s in raw_states])
    confidence = filtered.max(axis=1)

    cur_label = str(labels[-1])
    days = 1
    for i in range(len(labels) - 2, -1, -1):
        if labels[i] == cur_label:
            days += 1
        else:
            break

    close = feat["close"].values
    sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
    above_sma = bool(close[-1] > sma50) if not np.isnan(sma50) else None

    vol = feat["volume"].values
    if vol.sum() > 0 and len(vol) > 25:
        recent = vol[-5:].mean()
        baseline = vol[-25:-5].mean()
        vol_trend_up = bool(recent > baseline) if baseline > 0 else None
    else:
        vol_trend_up = None

    history = pd.Series(labels).value_counts(normalize=True).to_dict()

    return {
        "ok": True,
        "ticker": ticker,
        "current_price": float(close[-1]),
        "label": cur_label,
        "confidence": float(confidence[-1]),
        "days_in_regime": int(days),
        "above_sma": above_sma,
        "vol_trend_up": vol_trend_up,
        "n_regimes": int(K),
        "labels": labels.tolist(),
        "dates": feat.index,
        "close": close,
        "history": history,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Universe"), unsafe_allow_html=True)
    default_str = ",".join(LARGE_CAP + ETFS + CRYPTO)
    universe_str = st.text_area("Tickers (comma separated)", value=default_str, height=110)
    universe = [t.strip().upper() for t in universe_str.split(",") if t.strip()]
    universe = list(dict.fromkeys(universe))

    st.markdown(section_header("Filters"), unsafe_allow_html=True)
    regime_filter = st.multiselect(
        "Show regimes",
        options=["Low Vol", "Medium Vol", "High Vol"],
        default=["Low Vol", "Medium Vol", "High Vol"],
    )
    min_conf = st.slider("Min confidence (%)", 0, 100, 50, step=5)
    sma_filter = st.selectbox("vs 50-day SMA", ["Any", "Above only", "Below only"])
    vol_filter = st.selectbox("Volume trend", ["Any", "Increasing only", "Decreasing only"])
    sort_by = st.selectbox(
        "Sort by",
        ["Confidence (desc)", "Days in regime (desc)", "Ticker (A-Z)"],
    )

    scan = st.button("Scan Market", type="primary", use_container_width=True)


# ── Run scan ─────────────────────────────────────────────────────────────

if "screener_results" not in st.session_state:
    st.session_state.screener_results = None

if scan:
    _clear_data_cache()
    scan_ticker.clear()
    st.session_state.screener_results = None

if st.session_state.screener_results is None and universe:
    with st.spinner("Scanning the universe..."):
        progress = st.progress(0.0, text="Starting...")
        results: list[dict] = []
        for i, t in enumerate(universe):
            progress.progress(i / len(universe), text=f"Scanning {t}...  ({i+1}/{len(universe)})")
            results.append(scan_ticker(t))
        progress.empty()
        st.session_state.screener_results = results

results: list[dict] = st.session_state.screener_results or []
ok_results = [r for r in results if r.get("ok")]
fail_results = [r for r in results if not r.get("ok")]


# ── Source banner ────────────────────────────────────────────────────────

render_html(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{TEXT_MUTED};
                letter-spacing:2px;margin-bottom:14px;'>
      {status_dot('connected')}{len(ok_results)} / {len(universe)} TICKERS SCANNED &nbsp;·&nbsp;
      FORWARD-ONLY HMM (BIC K=3..5) &nbsp;·&nbsp;
      2-YEAR LOOKBACK
    </div>
""")

if fail_results:
    failed_summary = ", ".join(f"{r['ticker']} ({r.get('error','?')})" for r in fail_results[:6])
    if len(fail_results) > 6:
        failed_summary += f" + {len(fail_results) - 6} more"
    render_html(f"""
        <div style='background:rgba(255,193,7,0.08);border:1px solid {ACCENT_AMBER};
                    border-left:3px solid {ACCENT_AMBER};border-radius:10px;
                    padding:10px 16px;margin-bottom:14px;
                    font-family:DM Sans;font-size:0.78rem;color:{TEXT_SECONDARY};'>
          <b style='color:{ACCENT_AMBER};text-transform:uppercase;letter-spacing:2px;font-size:0.7rem;'>
            Skipped tickers</b> &nbsp;·&nbsp; {failed_summary}
        </div>
    """)

if not ok_results:
    render_html(f"<div style='color:{TEXT_MUTED};padding:48px;text-align:center;'>"
                f"No data. Click <b style='color:{ACCENT_CYAN}'>Scan Market</b> "
                f"or wait for yfinance rate-limit to clear, then refresh.</div>")
    st.stop()


# ── Top summary cards ────────────────────────────────────────────────────

count_low = sum(1 for r in ok_results if r["label"] == "Low Vol")
count_med = sum(1 for r in ok_results if r["label"] == "Medium Vol")
count_high = sum(1 for r in ok_results if r["label"] == "High Vol")
avg_conf = float(np.mean([r["confidence"] for r in ok_results]))
counts = {"Low Vol": count_low, "Medium Vol": count_med, "High Vol": count_high}
strongest_label = max(counts, key=counts.get)
strongest_color = REGIME_COLORS.get(strongest_label, ACCENT_VIOLET)

# Best category by % in Low Vol
by_cat: dict[str, list[dict]] = {}
for r in ok_results:
    by_cat.setdefault(categorize(r["ticker"]), []).append(r)
cat_scores = {
    cat: sum(1 for r in rs if r["label"] == "Low Vol") / max(len(rs), 1)
    for cat, rs in by_cat.items()
}
best_cat = max(cat_scores, key=cat_scores.get) if cat_scores else "—"
best_cat_pct = cat_scores.get(best_cat, 0.0) * 100

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("In Low Vol",      str(count_low),  ACCENT_GREEN), unsafe_allow_html=True)
c2.markdown(metric_card("In Medium Vol",   str(count_med),  ACCENT_AMBER), unsafe_allow_html=True)
c3.markdown(metric_card("In High Vol",     str(count_high), ACCENT_RED),   unsafe_allow_html=True)
c4.markdown(metric_card("Avg Confidence",  f"{avg_conf*100:.0f}%", ACCENT_CYAN), unsafe_allow_html=True)

render_html(f"""
    <div style='font-family:DM Sans;font-size:0.85rem;color:{TEXT_MUTED};margin:12px 4px 0 4px;'>
      Strongest regime: <b style='color:{strongest_color}'>{strongest_label}</b> ({counts[strongest_label]} tickers)
      &nbsp;·&nbsp;
      Most favorable category: <b style='color:{ACCENT_CYAN}'>{best_cat}</b> ({best_cat_pct:.0f}% in Low Vol)
    </div>
""")


# ── Apply filters & sort ────────────────────────────────────────────────

def passes_filters(r: dict) -> bool:
    if r["label"] not in regime_filter:
        return False
    if r["confidence"] * 100 < min_conf:
        return False
    if sma_filter == "Above only" and not r["above_sma"]:
        return False
    if sma_filter == "Below only" and r["above_sma"]:
        return False
    if vol_filter == "Increasing only" and r["vol_trend_up"] is not True:
        return False
    if vol_filter == "Decreasing only" and r["vol_trend_up"] is not False:
        return False
    return True


filtered = [r for r in ok_results if passes_filters(r)]
if sort_by.startswith("Confidence"):
    filtered.sort(key=lambda r: r["confidence"], reverse=True)
elif sort_by.startswith("Days"):
    filtered.sort(key=lambda r: r["days_in_regime"], reverse=True)
else:
    filtered.sort(key=lambda r: r["ticker"])


def is_clean_setup(r: dict) -> bool:
    return r["label"] == "Low Vol" and r["above_sma"] is True and r["vol_trend_up"] is True


# ── Screener table ───────────────────────────────────────────────────────

st.markdown(section_header(f"Screener · {len(filtered)} matches"), unsafe_allow_html=True)

# Header row
render_html(f"""
    <div style='display:grid;grid-template-columns:90px 60px 110px 200px 1fr 80px 60px 60px;
                gap:14px;align-items:center;padding:6px 16px;
                font-family:DM Sans;font-size:0.65rem;color:{TEXT_MUTED};
                text-transform:uppercase;letter-spacing:2px;
                border-bottom:1px solid {BORDER};margin-bottom:8px;'>
      <div>Ticker</div>
      <div>Cat</div>
      <div style='text-align:right;'>Price</div>
      <div>Regime</div>
      <div>Confidence</div>
      <div style='text-align:right;'>Days</div>
      <div style='text-align:center;'>50-SMA</div>
      <div style='text-align:center;'>Vol</div>
    </div>
""")

if not filtered:
    render_html(f"<div style='color:{TEXT_MUTED};padding:32px;text-align:center;'>"
                f"No tickers pass the current filters.</div>")
else:
    for r in filtered:
        clean = is_clean_setup(r)
        regime_col = REGIME_COLORS.get(r["label"], ACCENT_VIOLET)
        cat = categorize(r["ticker"])

        # SMA arrow
        if r["above_sma"] is True:
            sma_arrow, sma_color = "▲", ACCENT_GREEN
        elif r["above_sma"] is False:
            sma_arrow, sma_color = "▼", ACCENT_RED
        else:
            sma_arrow, sma_color = "—", TEXT_MUTED

        # Volume arrow
        if r["vol_trend_up"] is True:
            vol_arrow, vol_color = "▲", ACCENT_GREEN
        elif r["vol_trend_up"] is False:
            vol_arrow, vol_color = "▼", ACCENT_RED
        else:
            vol_arrow, vol_color = "—", TEXT_MUTED

        border_left = (
            f"border-left:3px solid {ACCENT_CYAN};box-shadow:inset 6px 0 18px rgba(0,212,255,0.10);"
            if clean else f"border-left:3px solid transparent;"
        )

        conf_pct = r["confidence"] * 100
        badge = regime_badge(r["label"], conf_pct)

        render_html(f"""
            <div style='display:grid;grid-template-columns:90px 60px 110px 200px 1fr 80px 60px 60px;
                        gap:14px;align-items:center;padding:10px 14px;
                        background:{BG_CARD};border:1px solid {BORDER};{border_left}
                        border-radius:10px;margin-bottom:6px;'>
              <div style='font-family:DM Sans;font-size:1.0rem;color:{TEXT_PRIMARY};font-weight:700;'>{r['ticker']}</div>
              <div style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};
                          text-transform:uppercase;letter-spacing:1.5px;'>{cat}</div>
              <div style='font-family:JetBrains Mono;font-size:0.85rem;color:{TEXT_SECONDARY};text-align:right;'>${r['current_price']:,.2f}</div>
              <div>{badge}</div>
              <div>
                <div style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};
                            margin-bottom:3px;'>{conf_pct:.0f}%</div>
                <div style='height:5px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;'>
                  <div style='width:{conf_pct}%;height:100%;background:{regime_col};
                              box-shadow:0 0 8px {regime_col}80;border-radius:3px;'></div>
                </div>
              </div>
              <div style='font-family:JetBrains Mono;font-size:0.85rem;color:{TEXT_SECONDARY};text-align:right;'>{r['days_in_regime']}d</div>
              <div style='font-family:JetBrains Mono;font-size:1.1rem;color:{sma_color};text-align:center;'>{sma_arrow}</div>
              <div style='font-family:JetBrains Mono;font-size:1.1rem;color:{vol_color};text-align:center;'>{vol_arrow}</div>
            </div>
        """)

# Legend for clean setups
n_clean = sum(1 for r in filtered if is_clean_setup(r))
if n_clean:
    render_html(f"""
        <div style='font-family:DM Sans;font-size:0.78rem;color:{TEXT_MUTED};
                    margin-top:10px;padding:0 4px;'>
          <span style='display:inline-block;width:10px;height:10px;background:{ACCENT_CYAN};
                       border-radius:2px;vertical-align:middle;margin-right:8px;'></span>
          <b style='color:{ACCENT_CYAN}'>{n_clean}</b> clean setup{'s' if n_clean != 1 else ''}
          (Low Vol + above 50-SMA + rising volume)
        </div>
    """)


# ── Quick chart for selected ticker ──────────────────────────────────────

st.markdown(section_header("Quick Look"), unsafe_allow_html=True)

ticker_choices = [r["ticker"] for r in filtered] or [r["ticker"] for r in ok_results]
if ticker_choices:
    pick = st.selectbox("Inspect ticker", ticker_choices, index=0)
    pr = next(r for r in ok_results if r["ticker"] == pick)

    # Mini price chart with regime bands
    dates = pr["dates"]
    close = pr["close"]
    labels = pr["labels"]

    qfig = go.Figure()

    # Build runs
    if labels:
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[start]:
                col = REGIME_COLORS.get(labels[start], ACCENT_VIOLET)
                qfig.add_vrect(
                    x0=dates[start], x1=dates[i],
                    fillcolor=col, opacity=0.13, line_width=0, layer="below",
                )
                start = i
        col = REGIME_COLORS.get(labels[start], ACCENT_VIOLET)
        qfig.add_vrect(
            x0=dates[start], x1=dates[-1],
            fillcolor=col, opacity=0.13, line_width=0, layer="below",
        )

    qfig.add_trace(go.Scatter(
        x=dates, y=close, mode="lines",
        line=dict(color=TEXT_PRIMARY, width=1.6),
        name=pick,
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    ql = get_plotly_layout()
    ql["height"] = 320
    ql["margin"] = dict(l=50, r=20, t=20, b=40)
    ql["showlegend"] = False
    qfig.update_layout(**ql)

    left, right = st.columns([2.2, 1.0])
    with left:
        st.plotly_chart(qfig, use_container_width=True)

    # Regime history (% time in each regime over the lookback)
    with right:
        history = pr["history"]
        order = ["Low Vol", "Medium Vol", "High Vol"]
        hist_pairs = [(lab, history.get(lab, 0.0) * 100) for lab in order]
        hfig = go.Figure(go.Bar(
            x=[lab for lab, _ in hist_pairs],
            y=[pct for _, pct in hist_pairs],
            marker=dict(color=[REGIME_COLORS.get(lab, ACCENT_VIOLET) for lab, _ in hist_pairs]),
            text=[f"{pct:.0f}%" for _, pct in hist_pairs],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", color=TEXT_PRIMARY, size=12),
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            showlegend=False,
        ))
        hl = get_plotly_layout()
        hl["height"] = 320
        hl["margin"] = dict(l=40, r=20, t=40, b=40)
        hl["yaxis"]["range"] = [0, 100]
        hl["yaxis"]["ticksuffix"] = "%"
        hl["title"] = dict(
            text=f"<span style='font-family:DM Sans;color:{TEXT_MUTED};font-size:11px;"
                 f"text-transform:uppercase;letter-spacing:2px;'>{pick} regime time-share</span>",
            x=0.02, xanchor="left", y=0.97, yanchor="top",
        )
        hfig.update_layout(**hl)
        st.plotly_chart(hfig, use_container_width=True)


# ── Universe regime distribution (bottom stacked bar) ───────────────────

st.markdown(section_header("Universe Regime Distribution"), unsafe_allow_html=True)

total = len(ok_results)
buckets = [
    ("Low Vol",    count_low),
    ("Medium Vol", count_med),
    ("High Vol",   count_high),
]

dfig = go.Figure()
cum = 0.0
for lab, cnt in buckets:
    pct = cnt / total * 100 if total else 0
    if cnt == 0:
        continue
    col = REGIME_COLORS.get(lab, ACCENT_VIOLET)
    dfig.add_trace(go.Bar(
        y=["Universe"], x=[pct], base=[cum],
        orientation="h",
        marker=dict(color=col, line=dict(color=BG_PRIMARY, width=2)),
        text=[f"<b>{lab}</b><br>{cnt} ({pct:.0f}%)"],
        textposition="inside",
        insidetextfont=dict(color=BG_PRIMARY, family="JetBrains Mono", size=12),
        hovertemplate=f"{lab}<br>{cnt} tickers ({pct:.1f}%)<extra></extra>",
        showlegend=False, name=lab,
    ))
    cum += pct

dl = get_plotly_layout()
dl["height"] = 130
dl["margin"] = dict(l=20, r=20, t=10, b=30)
dl["barmode"] = "relative"
dl["xaxis"]["range"] = [0, 100]
dl["xaxis"]["ticksuffix"] = "%"
dl["yaxis"]["showticklabels"] = False
dl["yaxis"]["showgrid"] = False
dfig.update_layout(**dl)
st.plotly_chart(dfig, use_container_width=True)
