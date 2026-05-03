"""
Portfolio Risk Dashboard.

DEMO MODE by default - sample positions are loaded so the dashboard runs
without an Alpaca account. Replace via the sidebar editor or by uploading a
positions CSV. Per-position HMM regime detection (forward-only), 60-day
rolling correlation matrix, and historical stress-test scenarios.

Run with:
    py -3.13 -m streamlit run portfolio_risk_dashboard.py --server.port=8506
"""

from __future__ import annotations

import textwrap
import warnings
from zoneinfo import ZoneInfo

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
    page_title="Portfolio Risk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def render_html(s: str) -> None:
    """Streamlit's markdown processor closes HTML blocks when it sees an
    indented line. Dedent + strip avoids that, so multi-line f-strings
    render as HTML instead of leaking source into the page."""
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)


# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_POSITIONS = pd.DataFrame([
    {"ticker": "SPY",  "shares": 100, "entry": 540.0, "current": 558.0},
    {"ticker": "QQQ",  "shares":  50, "entry": 480.0, "current": 495.0},
    {"ticker": "AAPL", "shares":  75, "entry": 210.0, "current": 218.0},
    {"ticker": "GLD",  "shares":  40, "entry": 235.0, "current": 242.0},
    {"ticker": "TLT",  "shares":  60, "entry":  88.0, "current":  85.0},
])

STRESS_DRAWDOWNS = {
    "2008 GFC":   {"SPY": -0.56, "QQQ": -0.54, "AAPL": -0.61, "GLD": +0.21, "TLT": +0.33},
    "2020 COVID": {"SPY": -0.34, "QQQ": -0.28, "AAPL": -0.31, "GLD": -0.03, "TLT": +0.21},
    "2022 Hikes": {"SPY": -0.25, "QQQ": -0.33, "AAPL": -0.30, "GLD": -0.04, "TLT": -0.31},
}
STRESS_PROXY = "SPY"  # fallback for unknown tickers

VOL_RANK_LABELS = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
FAVORABLE_RANKS = {0, 1}  # Low + Medium = favorable; High = unfavorable

FEATURE_COLS = ["log_ret", "realized_vol", "volume_ratio", "hl_range_pct"]


# ── Data + features ────────────────────────────────────────────────────────

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
    return out.dropna()


# ── HMM (forward-only, no look-ahead) ─────────────────────────────────────

def fit_hmm(X: np.ndarray, n_components: int = 3, seed: int = 42) -> GaussianHMM:
    m = GaussianHMM(
        n_components=n_components, covariance_type="diag",
        n_iter=150, tol=1e-3, random_state=seed,
    )
    m.fit(X)
    return m


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


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_ticker(ticker: str, lookback_days: int = 504) -> dict:
    """Returns regime label, confidence, days-in-regime, latest close, and the
    full close series for downstream correlation work."""
    df = fetch_history(ticker, lookback_days)
    if df.empty:
        return {"ok": False, "ticker": ticker, "error": "no data from yfinance"}
    feat = build_features(df)
    if len(feat) < 100:
        return {"ok": False, "ticker": ticker, "error": f"only {len(feat)} bars after warmup"}

    X = feat[FEATURE_COLS].values
    try:
        model = fit_hmm(X, n_components=3)
        filtered = forward_filter(model, X)
    except Exception as exc:
        return {"ok": False, "ticker": ticker, "error": f"HMM failed: {exc}"}

    vol_idx = FEATURE_COLS.index("realized_vol")
    order = np.argsort(model.means_[:, vol_idx])
    raw_to_label = {int(raw): VOL_RANK_LABELS[r] for r, raw in enumerate(order)}
    raw_to_rank = {int(raw): r for r, raw in enumerate(order)}

    raw_states = filtered.argmax(axis=1)
    labels = np.array([raw_to_label[s] for s in raw_states])
    confidence = filtered.max(axis=1)

    cur_label = str(labels[-1])
    cur_conf = float(confidence[-1])
    cur_rank = int(raw_to_rank[int(raw_states[-1])])

    # Days in current regime: count consecutive matching bars from the end
    days = 1
    for i in range(len(labels) - 2, -1, -1):
        if labels[i] == cur_label:
            days += 1
        else:
            break

    return {
        "ok": True,
        "ticker": ticker,
        "label": cur_label,
        "rank": cur_rank,
        "confidence": cur_conf,
        "days_in_regime": int(days),
        "current_price": float(feat["close"].iloc[-1]),
        "close": feat["close"].rename(ticker),
    }


# ── Correlation ────────────────────────────────────────────────────────────

def correlation_matrix(closes: list[pd.Series], lookback: int = 60) -> pd.DataFrame:
    if not closes:
        return pd.DataFrame()
    df = pd.concat(closes, axis=1).dropna()
    if len(df) < lookback + 5:
        return pd.DataFrame()
    rets = df.pct_change().dropna()
    return rets.tail(lookback).corr()


# ── Stress test ────────────────────────────────────────────────────────────

def stress_drawdown(ticker: str, scenario: str) -> float:
    table = STRESS_DRAWDOWNS[scenario]
    return table.get(ticker, table[STRESS_PROXY])


def portfolio_stress(positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pv = (positions["shares"] * positions["current"]).sum()
    for scenario in STRESS_DRAWDOWNS:
        loss_dollars = float(sum(
            row["shares"] * row["current"] * stress_drawdown(row["ticker"], scenario)
            for _, row in positions.iterrows()
        ))
        loss_pct = (loss_dollars / pv) if pv > 0 else 0.0
        rows.append({"scenario": scenario, "loss_dollars": loss_dollars, "loss_pct": loss_pct})
    return pd.DataFrame(rows)


# ── Market status ──────────────────────────────────────────────────────────

def us_market_status() -> tuple[bool, str]:
    try:
        now_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    except Exception:
        return False, "Unknown"
    is_weekday = now_et.weekday() < 5
    minutes = now_et.hour * 60 + now_et.minute
    is_hours = (9 * 60 + 30) <= minutes < (16 * 60)
    if is_weekday and is_hours:
        return True, f"Open · {now_et.strftime('%H:%M')} ET"
    return False, f"Closed · {now_et.strftime('%a %H:%M')} ET"


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Positions"), unsafe_allow_html=True)

    upload = st.file_uploader("Upload positions CSV", type=["csv"])
    csv_positions = None
    if upload is not None:
        try:
            csv_positions = pd.read_csv(upload)
            csv_positions.columns = [c.strip().lower() for c in csv_positions.columns]
            for col in ["ticker", "shares", "entry", "current"]:
                if col not in csv_positions.columns:
                    raise ValueError(f"missing column: {col}")
            csv_positions["ticker"] = csv_positions["ticker"].str.upper().str.strip()
        except Exception as exc:
            st.error(f"CSV error: {exc}")
            csv_positions = None

    initial = csv_positions if csv_positions is not None else DEFAULT_POSITIONS
    positions = st.data_editor(
        initial,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker":  st.column_config.TextColumn("Ticker", required=True),
            "shares":  st.column_config.NumberColumn("Shares", min_value=0, step=1),
            "entry":   st.column_config.NumberColumn("Entry $", min_value=0.0, format="%.2f"),
            "current": st.column_config.NumberColumn("Current $", min_value=0.0, format="%.2f"),
        },
        key="positions_editor",
    )

    st.markdown(section_header("Watchlist"), unsafe_allow_html=True)
    watch_str = st.text_input("Extra tickers (comma separated)", value="")

    st.markdown(section_header("Alpaca (optional)"), unsafe_allow_html=True)
    st.caption("Demo mode: keys are not transmitted. Live integration is a future enhancement.")
    st.text_input("API key", value="", type="password", key="alpaca_key", disabled=True)
    st.text_input("API secret", value="", type="password", key="alpaca_secret", disabled=True)

    refresh = st.button("Refresh Analysis", type="primary", use_container_width=True)


# ── Run ─────────────────────────────────────────────────────────────────────

# Clean positions: drop rows with no ticker / zero shares
positions = positions.dropna(subset=["ticker"]).copy()
positions["ticker"] = positions["ticker"].astype(str).str.upper().str.strip()
positions = positions[positions["ticker"] != ""]
positions = positions[positions["shares"].fillna(0) > 0].reset_index(drop=True)

if len(positions) == 0:
    st.markdown(
        f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;'>"
        f"Add at least one position in the sidebar editor.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Always analyse on initial load (cached) - "Refresh" just busts the cache
if refresh:
    _clear_data_cache()
    analyze_ticker.clear()

with st.spinner("Analysing positions..."):
    progress = st.progress(0.0, text="Starting...")
    analyses: dict[str, dict] = {}
    for i, t in enumerate(positions["ticker"].tolist()):
        progress.progress(i / len(positions), text=f"Analysing {t}...")
        analyses[t] = analyze_ticker(t)
    progress.empty()

# Watchlist
watchlist_tickers = [
    t.strip().upper() for t in watch_str.split(",") if t.strip().upper() and t.strip().upper() not in analyses
]
watchlist: dict[str, dict] = {}
if watchlist_tickers:
    with st.spinner("Loading watchlist..."):
        for t in watchlist_tickers:
            watchlist[t] = analyze_ticker(t)


# ── Rate-limit / failure banner ────────────────────────────────────────────

failed = [t for t, a in analyses.items() if not a.get("ok")]
if failed:
    rate_limited = any("rate" in (a.get("error") or "").lower() or "too many" in (a.get("error") or "").lower()
                       for a in analyses.values() if not a.get("ok"))
    if rate_limited or len(failed) == len(analyses):
        render_html(f"""
            <div style='background:rgba(255,193,7,0.10);border:1px solid {ACCENT_AMBER};border-left:3px solid {ACCENT_AMBER};border-radius:10px;padding:12px 18px;margin-bottom:12px;font-family:DM Sans;color:{TEXT_PRIMARY};font-size:0.85rem;'>
              <b style='color:{ACCENT_AMBER};text-transform:uppercase;letter-spacing:2px;font-size:0.7rem;'>yfinance rate-limited</b>
              &nbsp;·&nbsp; {len(failed)} of {len(analyses)} ticker(s) couldn't fetch price history. Wait a few minutes, then click <b style='color:{ACCENT_CYAN}'>Refresh Analysis</b>. Stress test and P&L still work from your editor values.
            </div>
        """)
    else:
        render_html(f"""
            <div style='background:rgba(255,193,7,0.06);border:1px solid {BORDER};border-radius:10px;padding:10px 16px;margin-bottom:12px;font-family:DM Sans;color:{TEXT_MUTED};font-size:0.8rem;'>
              {len(failed)} ticker(s) had no regime data: {', '.join(failed)}
            </div>
        """)


# ── Top bar ────────────────────────────────────────────────────────────────

market_open, market_text = us_market_status()
positions["value"] = positions["shares"] * positions["current"]
positions["cost"] = positions["shares"] * positions["entry"]
positions["pnl_dollars"] = positions["value"] - positions["cost"]
positions["pnl_pct"] = (positions["current"] / positions["entry"] - 1.0) * 100

total_value = float(positions["value"].sum())
total_cost = float(positions["cost"].sum())
total_pnl = total_value - total_cost
total_pnl_pct = (total_value / total_cost - 1.0) * 100 if total_cost > 0 else 0.0

n_positions = len(positions)
n_favorable = sum(
    1 for t in positions["ticker"]
    if analyses[t].get("ok") and analyses[t]["rank"] in FAVORABLE_RANKS
)

arrow = "▲" if total_pnl >= 0 else "▼"
pnl_col = pnl_color(total_pnl)

cols = st.columns([2.0, 1.4, 0.9, 1.3, 1.1])
with cols[0]:
    st.markdown(
        f"""
        <div style='padding:6px 0;'>
            <div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};
                        text-transform:uppercase;letter-spacing:2px;'>Portfolio Value</div>
            <div style='font-family:JetBrains Mono;font-size:3.2rem;font-weight:700;
                        color:{TEXT_PRIMARY};line-height:1.05;
                        text-shadow:0 0 22px rgba(0,212,255,0.35);'>
                ${total_value:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f"""
        <div style='padding:6px 0;'>
            <div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};
                        text-transform:uppercase;letter-spacing:2px;'>Total P&L</div>
            <div style='font-family:JetBrains Mono;font-size:1.9rem;font-weight:700;
                        color:{pnl_col};line-height:1.1;'>
                {arrow} ${abs(total_pnl):,.0f}</div>
            <div style='font-family:JetBrains Mono;font-size:1.0rem;color:{pnl_col};'>
                {total_pnl_pct:+.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(metric_card("Positions", str(n_positions), ACCENT_VIOLET), unsafe_allow_html=True)
with cols[3]:
    fav_color = ACCENT_GREEN if n_favorable >= n_positions * 0.6 else (ACCENT_AMBER if n_favorable > 0 else ACCENT_RED)
    st.markdown(
        metric_card("Regime Health", f"{n_favorable} / {n_positions}", fav_color),
        unsafe_allow_html=True,
    )
with cols[4]:
    dot = status_dot("connected" if market_open else "warning")
    color = ACCENT_GREEN if market_open else ACCENT_AMBER
    st.markdown(
        f"""
        <div style='padding:18px 14px;background:{BG_CARD};border:1px solid {BORDER};
                    border-radius:12px;text-align:center;'>
            <div style='font-family:DM Sans;font-size:0.65rem;color:{TEXT_MUTED};
                        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>
                Market</div>
            <div style='font-family:JetBrains Mono;font-size:1.0rem;color:{color};'>
                {dot}{'OPEN' if market_open else 'CLOSED'}</div>
            <div style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};
                        margin-top:4px;'>{market_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Two-column body ────────────────────────────────────────────────────────

left, right = st.columns([1.5, 1.0])


# ─ Position cards (left, ~60%) ─────────────────────────────────────────────

def pnl_bar_html(pnl_pct: float) -> str:
    """Center-anchored P&L bar. Right-fill (green) for gains, left-fill (red) for losses.
    Full extension (50% of container width) at +/-20% P&L."""
    abs_pct = min(abs(pnl_pct) / 20.0 * 50.0, 50.0)
    if pnl_pct >= 0:
        side, color = "left", ACCENT_GREEN
    else:
        side, color = "right", ACCENT_RED
    return (
        f"<div style='position:relative;height:8px;background:rgba(255,255,255,0.04);"
        f"border-radius:4px;overflow:hidden;margin:6px 0;'>"
        f"<div style='position:absolute;left:50%;top:0;width:1px;height:100%;"
        f"background:rgba(255,255,255,0.25);'></div>"
        f"<div style='position:absolute;{side}:50%;top:0;width:{abs_pct}%;height:100%;"
        f"background:{color};border-radius:3px;box-shadow:0 0 12px {color}66;'></div>"
        f"</div>"
    )


with left:
    st.markdown(section_header("Positions"), unsafe_allow_html=True)

    for _, row in positions.iterrows():
        t = row["ticker"]
        a = analyses.get(t, {})
        pos_color = pnl_color(row["pnl_pct"])

        if a.get("ok"):
            badge_html = regime_badge(a["label"], a["confidence"] * 100)
            days_html = (
                f"<span style='font-family:JetBrains Mono;color:{TEXT_MUTED};font-size:0.72rem;'>"
                f"{a['days_in_regime']}d in regime</span>"
            )
        else:
            badge_html = (
                f"<span style='color:{TEXT_MUTED};font-family:DM Sans;font-size:0.75rem;'>"
                f"regime unavailable</span>"
            )
            days_html = ""

        render_html(f"""
            <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:12px;'>
              <div style='display:flex;align-items:center;justify-content:space-between;gap:14px;'>
                <div style='display:flex;align-items:center;gap:14px;flex:1;'>
                  <div style='font-family:DM Sans;font-size:1.15rem;color:{TEXT_PRIMARY};font-weight:700;min-width:70px;'>{t}</div>
                  <div>{badge_html}</div>
                  {days_html}
                </div>
                <div style='text-align:right;font-family:JetBrains Mono;font-size:0.78rem;color:{TEXT_MUTED};'>{int(row['shares'])} sh &nbsp;·&nbsp; ${row['entry']:.2f} → <span style='color:{TEXT_PRIMARY}'>${row['current']:.2f}</span></div>
              </div>
              {pnl_bar_html(row['pnl_pct'])}
              <div style='display:flex;justify-content:space-between;align-items:center;font-family:JetBrains Mono;font-size:0.85rem;'>
                <span style='color:{TEXT_MUTED};'>P&L</span>
                <span style='color:{pos_color};font-weight:600;'>{'+' if row['pnl_dollars'] >= 0 else ''}${row['pnl_dollars']:,.0f} &nbsp;·&nbsp; {row['pnl_pct']:+.2f}%</span>
              </div>
            </div>
        """)


# ─ Right column panels ─────────────────────────────────────────────────────

with right:
    # Correlation
    st.markdown(section_header("Correlation Risk"), unsafe_allow_html=True)
    closes = [a["close"] for a in analyses.values() if a.get("ok") and "close" in a]
    corr = correlation_matrix(closes, lookback=60)

    if corr.empty:
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-family:DM Sans;font-size:0.85rem;"
            f"padding:18px 0;'>Need at least 2 positions with overlapping history.</div>",
            unsafe_allow_html=True,
        )
        high_pairs = []
    else:
        n = len(corr)
        # Discrete-ish blue-to-white scale
        scale = [
            [0.00, "#0a0f2a"], [0.30, "#0a3a5a"], [0.55, "#0a6a8a"],
            [0.75, ACCENT_CYAN], [0.90, "#bfeaff"], [1.00, "#ffffff"],
        ]
        z = corr.values
        text = [[f"{v:.2f}" for v in row] for row in z]

        cfig = go.Figure(go.Heatmap(
            z=z, x=corr.columns, y=corr.index,
            zmin=-1, zmax=1, colorscale=scale,
            showscale=False, xgap=2, ygap=2,
            text=text, texttemplate="%{text}",
            textfont={"family": "JetBrains Mono", "size": 11, "color": BG_PRIMARY},
            hovertemplate="%{y} / %{x}<br>r=%{z:.3f}<extra></extra>",
        ))

        # Red border for high-correlation off-diagonal cells
        shapes = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(z[i, j]) > 0.85:
                    shapes.append(dict(
                        type="rect",
                        x0=j - 0.48, x1=j + 0.48,
                        y0=i - 0.48, y1=i + 0.48,
                        line=dict(color=ACCENT_RED, width=2),
                        fillcolor="rgba(0,0,0,0)",
                    ))

        cl = get_plotly_layout()
        cl["height"] = 60 + 50 * n
        cl["margin"] = dict(l=70, r=10, t=10, b=40)
        cl["xaxis"]["showgrid"] = False
        cl["yaxis"]["showgrid"] = False
        cl["yaxis"]["autorange"] = "reversed"
        cl["shapes"] = shapes
        cfig.update_layout(**cl)
        st.plotly_chart(cfig, use_container_width=True)

        # List of high-corr pairs
        high_pairs = []
        seen = set()
        for i in range(n):
            for j in range(n):
                if i != j and (j, i) not in seen and abs(z[i, j]) > 0.85:
                    high_pairs.append((corr.index[i], corr.columns[j], float(z[i, j])))
                    seen.add((i, j))

        if high_pairs:
            warns = "".join(
                f"<li><b style='color:{ACCENT_RED}'>{a}</b> ↔ "
                f"<b style='color:{ACCENT_RED}'>{b}</b> &nbsp;r = {r:+.2f}</li>"
                for a, b, r in high_pairs
            )
            st.markdown(
                f"""
                <div style='background:rgba(255,23,68,0.08);border:1px solid {ACCENT_RED};
                            border-left:3px solid {ACCENT_RED};border-radius:10px;
                            padding:12px 16px;margin-top:10px;
                            font-family:DM Sans;font-size:0.82rem;color:{TEXT_SECONDARY};'>
                    <div style='font-family:DM Sans;font-size:0.7rem;color:{ACCENT_RED};
                                text-transform:uppercase;letter-spacing:2px;font-weight:700;
                                margin-bottom:6px;'>High correlation (|r| &gt; 0.85)</div>
                    <ul style='margin:0;padding-left:18px;line-height:1.6;'>{warns}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Stress test
    st.markdown(section_header("Stress Test"), unsafe_allow_html=True)
    stress_df = portfolio_stress(positions)
    for _, row in stress_df.iterrows():
        loss_pct = row["loss_pct"]
        loss_dollars = row["loss_dollars"]
        abs_pct = abs(loss_pct) * 100

        if abs_pct < 10:
            tile_color = ACCENT_GREEN
        elif abs_pct < 20:
            tile_color = ACCENT_AMBER
        else:
            tile_color = ACCENT_RED

        bar_pct = min(abs_pct / 50.0 * 100.0, 100.0)
        bar_dir = pnl_color(loss_dollars)  # red if loss, green if gain

        st.markdown(
            f"""
            <div style='background:{BG_CARD};border:1px solid {BORDER};
                        border-left:3px solid {tile_color};border-radius:10px;
                        padding:14px 18px;margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;align-items:baseline;'>
                    <div style='font-family:DM Sans;font-size:0.85rem;color:{TEXT_PRIMARY};
                                font-weight:600;'>{row['scenario']}</div>
                    <div style='font-family:JetBrains Mono;font-size:1.2rem;font-weight:700;
                                color:{bar_dir};'>
                        {'+' if loss_dollars >= 0 else '−'}${abs(loss_dollars):,.0f}
                    </div>
                </div>
                <div style='display:flex;justify-content:space-between;align-items:center;
                            margin-top:4px;'>
                    <div style='flex:1;height:6px;background:rgba(255,255,255,0.04);
                                border-radius:3px;margin-right:10px;overflow:hidden;'>
                        <div style='width:{bar_pct}%;height:100%;background:{bar_dir};
                                    border-radius:3px;'></div>
                    </div>
                    <div style='font-family:JetBrains Mono;font-size:0.85rem;color:{bar_dir};
                                min-width:60px;text-align:right;'>
                        {loss_pct*100:+.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Watchlist
    st.markdown(section_header("Watchlist"), unsafe_allow_html=True)
    if not watchlist:
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-family:DM Sans;font-size:0.85rem;"
            f"padding:8px 0;'>Add tickers in the sidebar to monitor regimes.</div>",
            unsafe_allow_html=True,
        )
    else:
        ranked = sorted(
            [(t, a) for t, a in watchlist.items() if a.get("ok")],
            key=lambda kv: kv[1]["confidence"],
            reverse=True,
        )
        for t, a in ranked:
            badge = regime_badge(a["label"], a["confidence"] * 100)
            conf_pct = a["confidence"] * 100
            bar_color = REGIME_COLORS.get(a["label"], ACCENT_VIOLET)
            st.markdown(
                f"""
                <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
                            padding:12px 16px;margin-bottom:8px;'>
                    <div style='display:flex;align-items:center;justify-content:space-between;
                                gap:12px;margin-bottom:8px;'>
                        <div>
                            <span style='font-family:DM Sans;font-size:1.0rem;font-weight:700;
                                         color:{TEXT_PRIMARY};'>{t}</span>
                            <span style='font-family:JetBrains Mono;font-size:0.78rem;
                                         color:{TEXT_MUTED};margin-left:10px;'>
                                ${a['current_price']:,.2f}</span>
                        </div>
                        <div>{badge}</div>
                    </div>
                    <div style='height:4px;background:rgba(255,255,255,0.04);border-radius:2px;
                                overflow:hidden;'>
                        <div style='width:{conf_pct}%;height:100%;background:{bar_color};
                                    border-radius:2px;'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Note any failures
        for t, a in watchlist.items():
            if not a.get("ok"):
                st.caption(f"⚠ {t}: {a.get('error','unknown error')}")
