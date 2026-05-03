"""
Multi-Asset Regime Backtester.

Walk-forward HMM regime detection per asset (no look-ahead), with a
volatility-tier allocation rule benchmarked against buy-and-hold and a
200-day SMA trend filter. Includes stress-test breakdowns for 2008, 2020,
and 2022 crisis windows.

Run with:
    py -3.13 -m streamlit run multi_asset_regime_backtester.py --server.port=8505
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from data_provider import get_history
from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Multi-Asset Regime Backtester",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Asset palette ──────────────────────────────────────────────────────────

ASSET_COLORS = {
    "SPY":     ACCENT_CYAN,
    "BTC-USD": ACCENT_AMBER,
    "GLD":     "#FFD700",
    "TLT":     ACCENT_VIOLET,
}
DEFAULT_TICKERS = ["SPY", "BTC-USD", "GLD", "TLT"]


def color_for(ticker: str, idx: int = 0) -> str:
    if ticker in ASSET_COLORS:
        return ASSET_COLORS[ticker]
    fallback = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_VIOLET, "#FFD700"]
    return fallback[idx % len(fallback)]


# ── Data + features ────────────────────────────────────────────────────────

def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around the unified data provider (FMP → TwelveData → yfinance)."""
    return get_history(ticker, start, end)


FEATURE_COLS = ["log_ret", "realized_vol", "volume_ratio", "hl_range_pct"]


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


# ── HMM helpers (forward-only, no look-ahead) ──────────────────────────────

def fit_hmm(X: np.ndarray, n_components: int = 3, seed: int = 42) -> GaussianHMM:
    m = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=150,
        tol=1e-3,
        random_state=seed,
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
    """Filtered posterior P(state_t | obs_{1..t}) - row t depends only on X[:t+1]."""
    T = X.shape[0]
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    log_b = emission_logprob(model, X)
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


# ── Walk-forward backtest ──────────────────────────────────────────────────

VOL_RANK_LABELS = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
RANK_TO_ALLOC = {0: 0.95, 1: 0.775, 2: 0.60}  # linear scale 95% -> 60%


@dataclass
class AssetResult:
    ticker: str
    feat: pd.DataFrame
    daily_ret: np.ndarray              # asset's daily return series, aligned to feat.index
    regime_rank: np.ndarray            # int per bar; -1 = no regime assigned (warmup)
    alloc: np.ndarray                  # allocation per bar in [0, 1]
    strat_equity: np.ndarray           # strategy equity curve from active_start
    bh_equity: np.ndarray              # buy-and-hold equity curve from active_start
    trend_equity: np.ndarray           # 200-SMA trend equity curve from active_start
    active_idx: pd.DatetimeIndex       # dates over which the equity curves run
    days_per_year: float
    stats_strat: dict
    stats_bh: dict
    stats_trend: dict


def stats_from_returns(rets: np.ndarray, days_per_year: float) -> dict:
    if len(rets) == 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    equity = np.cumprod(1.0 + rets)
    total_return = equity[-1] - 1.0
    years = len(rets) / days_per_year
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(days_per_year)) if sd > 0 else 0.0
    rmax = np.maximum.accumulate(equity)
    max_dd = float(((equity - rmax) / rmax).min())
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def backtest_asset(ticker: str, start: str, end: str,
                   train_days: int, test_days: int) -> AssetResult | None:
    prices = load_prices(ticker, start, end)
    if prices.empty:
        return None
    feat = build_features(prices)
    if len(feat) < train_days + max(30, test_days):
        return None

    n = len(feat)
    X = feat[FEATURE_COLS].values
    close = feat["close"].values
    daily_ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()

    regime_rank = np.full(n, -1, dtype=int)
    alloc = np.zeros(n)

    vol_idx = FEATURE_COLS.index("realized_vol")
    cur = train_days
    while cur < n:
        train_end = cur
        test_end = min(cur + test_days, n)
        try:
            model = fit_hmm(X[:train_end], n_components=3)
            filtered = forward_filter(model, X[:test_end])
            order = np.argsort(model.means_[:, vol_idx])  # ascending vol
            raw_to_rank = {int(raw): r for r, raw in enumerate(order)}
            for t in range(train_end, test_end):
                raw = int(filtered[t].argmax())
                rank = raw_to_rank[raw]
                regime_rank[t] = rank
                alloc[t] = RANK_TO_ALLOC[rank]
        except Exception:
            # leave regime_rank=-1 / alloc=0 for this window
            pass
        cur = test_end

    active_mask = regime_rank >= 0
    if not active_mask.any():
        return None
    active_start = int(np.argmax(active_mask))

    # Use yesterday's allocation against today's return
    alloc_lag = np.concatenate([[0.0], alloc[:-1]])
    strat_rets = (alloc_lag * daily_ret)[active_start:]
    bh_rets = daily_ret[active_start:]

    sma200 = pd.Series(close).rolling(200).mean().to_numpy()
    trend_alloc = np.where(close > sma200, 1.0, 0.0)
    trend_lag = np.concatenate([[0.0], trend_alloc[:-1]])
    trend_rets = (trend_lag * daily_ret)[active_start:]

    days_per_year = len(feat) / max((feat.index[-1] - feat.index[0]).days / 365.25, 1e-6)

    return AssetResult(
        ticker=ticker,
        feat=feat,
        daily_ret=daily_ret,
        regime_rank=regime_rank,
        alloc=alloc,
        strat_equity=np.cumprod(1.0 + strat_rets),
        bh_equity=np.cumprod(1.0 + bh_rets),
        trend_equity=np.cumprod(1.0 + trend_rets),
        active_idx=feat.index[active_start:],
        days_per_year=days_per_year,
        stats_strat=stats_from_returns(strat_rets, days_per_year),
        stats_bh=stats_from_returns(bh_rets, days_per_year),
        stats_trend=stats_from_returns(trend_rets, days_per_year),
    )


# ── Stress tests ───────────────────────────────────────────────────────────

CRISIS_WINDOWS = {
    "2008 GFC":      (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31")),
    "2020 COVID":    (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")),
    "2022 Hikes":    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
}


def stress_for_window(r: AssetResult, lo: pd.Timestamp, hi: pd.Timestamp) -> dict | None:
    """Compute strat & B&H max drawdown over a date window. None if no overlap."""
    idx = r.active_idx
    mask = (idx >= lo) & (idx <= hi)
    if not mask.any():
        return None
    # Equity curves are anchored at active_idx[0]; slice and re-base to 1.0 at window start
    sl = np.where(mask)[0]
    s = sl[0]; e = sl[-1] + 1

    def window_dd(equity_full: np.ndarray) -> float:
        seg = equity_full[s:e]
        if len(seg) == 0: return 0.0
        seg = seg / seg[0]
        rmax = np.maximum.accumulate(seg)
        return float(((seg - rmax) / rmax).min())

    return {
        "strat_dd": window_dd(r.strat_equity),
        "bh_dd":    window_dd(r.bh_equity),
        "n_days":   int(mask.sum()),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Inputs"), unsafe_allow_html=True)
    tickers_str = st.text_input(
        "Tickers (comma separated)",
        value=",".join(DEFAULT_TICKERS),
    )
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    today = pd.Timestamp.today().normalize()
    default_start = (today - pd.DateOffset(years=5)).date()
    date_range = st.date_input(
        "Date range",
        value=(default_start, today.date()),
        max_value=today.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = default_start, today.date()

    train_days = st.slider("Train window (trading days)", 100, 504, 252, step=10)
    test_days = st.slider("Test window (trading days)", 20, 252, 126, step=10)

    include_stress = st.checkbox("Include crisis stress tests", value=True)

    run = st.button("Run Backtest", type="primary", use_container_width=True)


# ── Run ─────────────────────────────────────────────────────────────────────

if "ma_results" not in st.session_state:
    st.session_state.ma_results = None

if run:
    with st.spinner(f"Backtesting {len(tickers)} assets..."):
        results: dict[str, AssetResult] = {}
        progress = st.progress(0.0, text="Starting...")
        for i, t in enumerate(tickers):
            progress.progress(i / len(tickers), text=f"Backtesting {t}...")
            try:
                r = backtest_asset(t, str(start_d), str(end_d), train_days, test_days)
                if r is not None:
                    results[t] = r
            except Exception as exc:
                st.warning(f"Skipping {t}: {exc}")
        progress.empty()

        if not results:
            st.error("No assets produced backtest results. Check tickers and date range.")
            st.stop()

        # Stress tests per crisis per asset
        stress: dict[str, dict[str, dict | None]] = {}
        if include_stress:
            for crisis, (lo, hi) in CRISIS_WINDOWS.items():
                stress[crisis] = {t: stress_for_window(r, lo, hi) for t, r in results.items()}

        st.session_state.ma_results = {
            "results": results,
            "stress": stress,
            "tickers": list(results.keys()),
            "train_days": train_days,
            "test_days": test_days,
            "include_stress": include_stress,
        }


# ── Render ──────────────────────────────────────────────────────────────────

R = st.session_state.ma_results

if R is None:
    st.markdown(
        f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;"
        f"font-family:DM Sans,sans-serif;font-size:0.95rem;'>"
        f"Pick tickers and date range, then click "
        f"<b style='color:{ACCENT_CYAN}'>Run Backtest</b>."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

results = R["results"]
tickers = R["tickers"]
stress = R["stress"]

# Source banner
st.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    f"color:{TEXT_MUTED};letter-spacing:2px;margin-bottom:14px;'>"
    f"{status_dot('connected')}{len(tickers)} ASSETS &nbsp;·&nbsp; "
    f"WALK-FORWARD train={R['train_days']}d / test={R['test_days']}d &nbsp;·&nbsp; "
    f"FORWARD-ONLY HMM"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Asset tabs (hero equity curves) ────────────────────────────────────────

tab_labels = []
for t in tickers:
    r = results[t]
    sharpe_imp = r.stats_strat["sharpe"] - r.stats_bh["sharpe"]
    arrow = "▲" if sharpe_imp > 0 else "▼"
    tab_labels.append(f"{t}  {arrow}{abs(sharpe_imp):.2f}")

tabs = st.tabs(tab_labels)
for i, t in enumerate(tickers):
    r = results[t]
    color = color_for(t, i)
    with tabs[i]:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card(f"{t} Strategy CAGR", f"{r.stats_strat['cagr']*100:+.1f}%", color), unsafe_allow_html=True)
        c2.markdown(metric_card("B&H CAGR", f"{r.stats_bh['cagr']*100:+.1f}%", TEXT_SECONDARY), unsafe_allow_html=True)
        c3.markdown(metric_card("Sharpe Improvement",
                                f"{r.stats_strat['sharpe'] - r.stats_bh['sharpe']:+.2f}",
                                pnl_color(r.stats_strat['sharpe'] - r.stats_bh['sharpe'])),
                                unsafe_allow_html=True)
        c4.markdown(metric_card("Max DD (strat)", f"{r.stats_strat['max_dd']*100:.1f}%", ACCENT_RED), unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r.active_idx, y=r.bh_equity,
            mode="lines",
            line=dict(color=TEXT_MUTED, width=1.5, dash="dash"),
            name="Buy & Hold",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra>B&H</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=r.active_idx, y=r.trend_equity,
            mode="lines",
            line=dict(color=TEXT_SECONDARY, width=1.2, dash="dot"),
            name="200-SMA Trend",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra>Trend</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=r.active_idx, y=r.strat_equity,
            mode="lines",
            line=dict(color=color, width=2.4),
            name=f"{t} Regime Strategy",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}x<extra>strat</extra>",
        ))
        layout = get_plotly_layout()
        layout["height"] = 460
        layout["margin"] = dict(l=50, r=20, t=20, b=40)
        layout["showlegend"] = True
        layout["legend"] = dict(
            bgcolor="rgba(10,10,15,0.6)", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT_SECONDARY, size=11),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        )
        layout["yaxis"]["title"] = dict(text="Equity (norm.)", font=dict(color=TEXT_MUTED, size=11))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)


# ── Regime timeline strips ─────────────────────────────────────────────────

st.markdown(section_header("Regime Timeline · All Assets"), unsafe_allow_html=True)

# Align all assets to a common date axis
all_dates = sorted(set().union(*(set(r.feat.index) for r in results.values())))
date_index = pd.DatetimeIndex(all_dates)
z = np.full((len(tickers), len(date_index)), np.nan)
for i, t in enumerate(tickers):
    r = results[t]
    s = pd.Series(r.regime_rank, index=r.feat.index)
    s = s.reindex(date_index)
    s_arr = s.to_numpy()
    s_arr = np.where(s_arr >= 0, s_arr, np.nan)
    z[i] = s_arr

# Discrete colorscale: 0=Low (green), 1=Medium (amber), 2=High (red)
discrete_scale = [
    [0.00, ACCENT_GREEN], [0.33, ACCENT_GREEN],
    [0.34, ACCENT_AMBER], [0.66, ACCENT_AMBER],
    [0.67, ACCENT_RED],   [1.00, ACCENT_RED],
]

hfig = go.Figure(go.Heatmap(
    z=z,
    x=date_index,
    y=tickers,
    zmin=0, zmax=2,
    colorscale=discrete_scale,
    showscale=False,
    hovertemplate="%{y} · %{x|%Y-%m-%d}<br>regime rank %{z}<extra></extra>",
    xgap=0, ygap=4,
))
hl = get_plotly_layout()
hl["height"] = 60 + 60 * len(tickers)
hl["margin"] = dict(l=80, r=20, t=20, b=40)
hl["yaxis"]["tickfont"] = dict(family="JetBrains Mono", color=TEXT_PRIMARY, size=12)
hl["xaxis"]["showgrid"] = False
hl["yaxis"]["showgrid"] = False
hfig.update_layout(**hl)
st.plotly_chart(hfig, use_container_width=True)

# Legend strip
legend_html = (
    "<div style='display:flex;gap:18px;justify-content:center;font-family:DM Sans;"
    f"font-size:0.75rem;color:{TEXT_MUTED};margin-top:-8px;'>"
)
for label, c in [("Low Vol", ACCENT_GREEN), ("Medium Vol", ACCENT_AMBER), ("High Vol", ACCENT_RED)]:
    legend_html += (
        f"<span style='display:inline-flex;align-items:center;gap:6px;'>"
        f"<span style='width:14px;height:14px;background:{c};border-radius:3px;'></span>"
        f"{label}</span>"
    )
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)


# ── Comparison table ───────────────────────────────────────────────────────

st.markdown(section_header("Comparison · Strategy vs Buy & Hold"), unsafe_allow_html=True)

rows = []
for t in tickers:
    r = results[t]
    rows.append({
        "Asset": t,
        "Strat CAGR":  r.stats_strat["cagr"] * 100,
        "B&H CAGR":    r.stats_bh["cagr"]    * 100,
        "Strat Max DD": r.stats_strat["max_dd"] * 100,
        "B&H Max DD":   r.stats_bh["max_dd"]   * 100,
        "Strat Sharpe": r.stats_strat["sharpe"],
        "B&H Sharpe":   r.stats_bh["sharpe"],
        "Sharpe Improvement": r.stats_strat["sharpe"] - r.stats_bh["sharpe"],
    })
comp_df = pd.DataFrame(rows).set_index("Asset")
best_asset = comp_df["Sharpe Improvement"].idxmax()

styled = (
    comp_df.style
    .format({
        "Strat CAGR":  "{:+.2f}%",
        "B&H CAGR":    "{:+.2f}%",
        "Strat Max DD": "{:.2f}%",
        "B&H Max DD":   "{:.2f}%",
        "Strat Sharpe": "{:.2f}",
        "B&H Sharpe":   "{:.2f}",
        "Sharpe Improvement": "{:+.2f}",
    })
    .map(lambda v: f"color:{ACCENT_GREEN}" if isinstance(v, (int, float)) and v > 0 else f"color:{ACCENT_RED}",
         subset=["Sharpe Improvement"])
    .apply(lambda row: [f"box-shadow:inset 0 0 0 1px {ACCENT_CYAN}; background-color:rgba(0,212,255,0.06)"] * len(row)
           if row.name == best_asset else [""] * len(row), axis=1)
    .set_properties(**{
        "background-color": BG_PRIMARY,
        "color": TEXT_SECONDARY,
        "border": f"1px solid {BORDER}",
        "font-family": "JetBrains Mono, monospace",
        "font-size": "13px",
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", BG_CARD),
            ("color", TEXT_PRIMARY),
            ("border", f"1px solid {BORDER}"),
            ("font-family", "DM Sans, sans-serif"),
            ("font-size", "11px"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "1px"),
            ("padding", "10px 8px"),
        ]},
        {"selector": "td", "props": [("padding", "8px")]},
    ])
)
st.dataframe(styled, use_container_width=True)


# ── Stress test ────────────────────────────────────────────────────────────

if R["include_stress"] and stress:
    st.markdown(section_header("Crisis Stress Tests · Strategy vs Buy & Hold Drawdowns"), unsafe_allow_html=True)
    cols = st.columns(len(CRISIS_WINDOWS))
    for ci, (crisis, (lo, hi)) in enumerate(CRISIS_WINDOWS.items()):
        bucket = stress.get(crisis, {})
        with cols[ci]:
            st.markdown(
                f"<div style='font-family:DM Sans;color:{TEXT_PRIMARY};font-size:0.9rem;"
                f"font-weight:600;margin-bottom:4px;'>{crisis}</div>"
                f"<div style='font-family:JetBrains Mono;color:{TEXT_MUTED};font-size:0.7rem;'>"
                f"{lo.date()} - {hi.date()}</div>",
                unsafe_allow_html=True,
            )
            xs, strat, bh, bar_colors = [], [], [], []
            for t in tickers:
                w = bucket.get(t)
                if w is None or w["n_days"] < 5:
                    continue
                xs.append(t)
                strat.append(w["strat_dd"] * 100)
                bh.append(w["bh_dd"] * 100)
                bar_colors.append(color_for(t, tickers.index(t)))

            if not xs:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-family:DM Sans;"
                    f"font-size:0.85rem;padding:30px 0;text-align:center;'>"
                    f"No data in this window for any selected asset.</div>",
                    unsafe_allow_html=True,
                )
                continue

            sf = go.Figure()
            sf.add_trace(go.Bar(
                x=xs, y=strat, name="Strategy",
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="%{x}<br>Strat DD %{y:.1f}%<extra></extra>",
            ))
            sf.add_trace(go.Bar(
                x=xs, y=bh, name="Buy & Hold",
                marker=dict(color=TEXT_MUTED, line=dict(width=0)),
                hovertemplate="%{x}<br>B&H DD %{y:.1f}%<extra></extra>",
            ))
            sl = get_plotly_layout()
            sl["height"] = 280
            sl["margin"] = dict(l=40, r=15, t=15, b=40)
            sl["barmode"] = "group"
            sl["showlegend"] = (ci == 0)
            sl["legend"] = dict(
                bgcolor="rgba(10,10,15,0.6)", bordercolor=BORDER, borderwidth=1,
                font=dict(color=TEXT_SECONDARY, size=10),
                x=0.01, y=0.01, xanchor="left", yanchor="bottom",
            )
            sl["yaxis"]["ticksuffix"] = "%"
            sl["yaxis"]["title"] = dict(text="Max DD", font=dict(color=TEXT_MUTED, size=10))
            sf.update_layout(**sl)
            st.plotly_chart(sf, use_container_width=True)


# ── Summary text ───────────────────────────────────────────────────────────

st.markdown(section_header("Summary"), unsafe_allow_html=True)

improvements = {t: results[t].stats_strat["sharpe"] - results[t].stats_bh["sharpe"] for t in tickers}
best_t = max(improvements, key=improvements.get)
worst_t = min(improvements, key=improvements.get)
best_imp = improvements[best_t]
worst_imp = improvements[worst_t]

summary = (
    f"Across <b style='color:{ACCENT_CYAN}'>{len(tickers)}</b> assets, regime detection added the "
    f"most value for <b style='color:{color_for(best_t, tickers.index(best_t))}'>{best_t}</b> with a "
    f"Sharpe improvement of <b style='color:{pnl_color(best_imp)}'>{best_imp:+.2f}</b> "
    f"({results[best_t].stats_strat['sharpe']:.2f} vs B&H {results[best_t].stats_bh['sharpe']:.2f}). "
    f"It struggled most with <b style='color:{color_for(worst_t, tickers.index(worst_t))}'>{worst_t}</b> "
    f"(<b style='color:{pnl_color(worst_imp)}'>{worst_imp:+.2f}</b>), suggesting regime detection may not "
    f"suit this asset class as well - or that the chosen window sizes don't match its volatility cycle."
)
st.markdown(
    f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;
                padding:22px 26px;font-family:DM Sans,sans-serif;color:{TEXT_SECONDARY};
                line-height:1.7;font-size:0.95rem;">{summary}</div>
    """,
    unsafe_allow_html=True,
)
