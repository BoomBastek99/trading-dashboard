"""
Sensitivity Analysis Dashboard.

One-at-a-time parameter sweeps over an SMA-crossover strategy on SPY, with a
coefficient-of-variation-based robustness score per parameter. Helps detect
overfitting: a strategy whose performance collapses when any single parameter
is perturbed is fragile, regardless of how good the base backtest looks.

Run with:
    py -3.13 -m streamlit run sensitivity_analysis_dashboard.py --server.port=8504
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_provider import get_history
from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sensitivity Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Data ────────────────────────────────────────────────────────────────────

def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around the unified data provider (FMP → TwelveData → yfinance)."""
    return get_history(ticker, start, end)


# ── Backtest ────────────────────────────────────────────────────────────────

def backtest_sma_crossover(
    close: np.ndarray,
    fast_ma: int,
    slow_ma: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict:
    """SMA crossover with stop-loss and take-profit. Long-only.

    Entry: fast crosses above slow.
    Exit:  fast crosses below slow, OR price hits stop-loss / take-profit.

    Returns total_return, sharpe (annualized from daily equity), max_drawdown,
    win_rate, n_trades.
    """
    EMPTY = {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
             "win_rate": 0.0, "n_trades": 0}

    n = len(close)
    fast_ma = int(fast_ma); slow_ma = int(slow_ma)
    if fast_ma >= slow_ma or n < slow_ma + 5:
        return EMPTY

    fast = pd.Series(close).rolling(fast_ma).mean().to_numpy()
    slow = pd.Series(close).rolling(slow_ma).mean().to_numpy()

    sl = -stop_loss_pct / 100.0
    tp = take_profit_pct / 100.0

    equity = np.ones(n)
    in_pos = False
    entry_price = 0.0
    trade_returns: list[float] = []

    for i in range(slow_ma, n):
        if in_pos:
            day_ret = close[i] / close[i - 1] - 1.0
            equity[i] = equity[i - 1] * (1.0 + day_ret)

            ret = close[i] / entry_price - 1.0
            cross_below = fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]
            if ret <= sl or ret >= tp or cross_below:
                trade_returns.append(ret)
                in_pos = False
        else:
            equity[i] = equity[i - 1]
            cross_above = fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]
            if cross_above:
                in_pos = True
                entry_price = close[i]

    if not trade_returns:
        return EMPTY

    daily = np.diff(equity) / equity[:-1]
    daily = daily[np.isfinite(daily)]
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    max_dd = float(((equity - running_max) / running_max).min())

    return {
        "total_return": float(equity[-1] - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": float(np.mean(np.array(trade_returns) > 0.0)),
        "n_trades": len(trade_returns),
    }


# ── Sensitivity sweep ───────────────────────────────────────────────────────

PARAM_META = {
    "fast_ma":         {"base": 10,  "range": np.arange(5, 31, 1).tolist(),                 "label": "Fast MA"},
    "slow_ma":         {"base": 50,  "range": np.arange(20, 101, 5).tolist(),               "label": "Slow MA"},
    "stop_loss_pct":   {"base": 2.0, "range": np.round(np.arange(0.5, 5.01, 0.5), 2).tolist(), "label": "Stop-Loss %"},
    "take_profit_pct": {"base": 4.0, "range": np.round(np.arange(1.0, 10.01, 0.5), 2).tolist(), "label": "Take-Profit %"},
}

METRICS = ["total_return", "sharpe", "max_drawdown", "win_rate"]
METRIC_LABELS = {
    "total_return":  "Total Return",
    "sharpe":        "Sharpe",
    "max_drawdown":  "Max DD",
    "win_rate":      "Win Rate",
}


def sweep_parameter(close: np.ndarray, base_params: dict, name: str, values: list) -> pd.DataFrame:
    rows = []
    for v in values:
        p = dict(base_params)
        p[name] = v
        m = backtest_sma_crossover(close, **p)
        rows.append({"value": v, **m})
    return pd.DataFrame(rows)


def coefficient_of_variation(series: pd.Series) -> float:
    """CV = std / |mean|. Robust to sign; returns inf when mean == 0 and std > 0."""
    s = series.dropna().to_numpy()
    if len(s) < 2:
        return 0.0
    m = abs(s.mean())
    sd = s.std()
    if m < 1e-9:
        return float("inf") if sd > 1e-9 else 0.0
    return float(sd / m)


def robustness_score(sweep_df: pd.DataFrame, metric: str = "sharpe") -> float:
    """0-100 based on coefficient of variation of `metric` across the sweep.
    CV=0 -> 100, CV>=1 -> 0, linear in between."""
    cv = coefficient_of_variation(sweep_df[metric])
    return float(np.clip(100.0 * (1.0 - cv), 0.0, 100.0))


def deviation_pct(value: float, base: float) -> float:
    """|value - base| / |base| as a percentage. Returns inf if base ~ 0."""
    if abs(base) < 1e-9:
        return 0.0 if abs(value) < 1e-9 else float("inf")
    return abs(value - base) / abs(base) * 100.0


def classify_robustness(score: float) -> tuple[str, str]:
    if score >= 70:
        return "Robust", ACCENT_GREEN
    if score >= 40:
        return "Moderate", ACCENT_AMBER
    return "Fragile", ACCENT_RED


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Inputs"), unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value="SPY").upper().strip()

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

    with st.expander("Custom parameter ranges (advanced)"):
        st.caption("Comma-separated lists override the defaults.")
        fast_str = st.text_input("fast_ma values", value="")
        slow_str = st.text_input("slow_ma values", value="")
        sl_str = st.text_input("stop_loss_pct values", value="")
        tp_str = st.text_input("take_profit_pct values", value="")

    run = st.button("Run Analysis", type="primary", use_container_width=True)


def parse_csv_numbers(s: str, default: list, as_int: bool = False) -> list:
    s = s.strip()
    if not s:
        return default
    try:
        vals = [float(x) for x in s.split(",") if x.strip()]
        return [int(v) for v in vals] if as_int else vals
    except Exception:
        return default


# ── Run ─────────────────────────────────────────────────────────────────────

if "sa_results" not in st.session_state:
    st.session_state.sa_results = None

if run:
    with st.spinner("Downloading prices and running parameter sweeps..."):
        prices = load_prices(ticker, str(start_d), str(end_d))
        if prices.empty:
            st.error(f"No data returned for {ticker}.")
            st.stop()
        close = prices["Close"].astype(float).to_numpy()

        params = {k: PARAM_META[k]["base"] for k in PARAM_META}
        ranges = {
            "fast_ma":         parse_csv_numbers(fast_str, PARAM_META["fast_ma"]["range"], as_int=True),
            "slow_ma":         parse_csv_numbers(slow_str, PARAM_META["slow_ma"]["range"], as_int=True),
            "stop_loss_pct":   parse_csv_numbers(sl_str,   PARAM_META["stop_loss_pct"]["range"]),
            "take_profit_pct": parse_csv_numbers(tp_str,   PARAM_META["take_profit_pct"]["range"]),
        }

        base_metrics = backtest_sma_crossover(close, **params)
        sweeps = {name: sweep_parameter(close, params, name, ranges[name]) for name in PARAM_META}

        # Per-parameter robustness scores (CV of Sharpe across the sweep)
        param_scores = {name: robustness_score(df, "sharpe") for name, df in sweeps.items()}
        overall_score = float(np.mean(list(param_scores.values())))

        # Heatmap: mean |deviation| % per (parameter, metric) across the sweep
        heatmap = pd.DataFrame(index=list(PARAM_META.keys()), columns=METRICS, dtype=float)
        for name, df in sweeps.items():
            for metric in METRICS:
                base_val = base_metrics[metric]
                devs = [deviation_pct(v, base_val) for v in df[metric].dropna()]
                heatmap.loc[name, metric] = float(np.nanmean([d for d in devs if np.isfinite(d)])) if devs else 0.0

        st.session_state.sa_results = {
            "ticker": ticker,
            "prices": prices,
            "params": params,
            "ranges": ranges,
            "base_metrics": base_metrics,
            "sweeps": sweeps,
            "param_scores": param_scores,
            "overall_score": overall_score,
            "heatmap": heatmap,
        }


# ── Render ──────────────────────────────────────────────────────────────────

R = st.session_state.sa_results

if R is None:
    st.markdown(
        f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;"
        f"font-family:DM Sans,sans-serif;font-size:0.95rem;'>"
        f"Configure inputs in the sidebar and click "
        f"<b style='color:{ACCENT_CYAN}'>Run Analysis</b>."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

base = R["base_metrics"]
overall = R["overall_score"]
overall_label, overall_color = classify_robustness(overall)

# Source banner
st.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    f"color:{TEXT_MUTED};letter-spacing:2px;margin-bottom:14px;'>"
    f"{status_dot('connected')}{R['ticker']} &nbsp;·&nbsp; "
    f"BASE SHARPE {base['sharpe']:.2f} &nbsp;·&nbsp; "
    f"BASE RETURN {base['total_return']*100:+.1f}% &nbsp;·&nbsp; "
    f"BASE MAX-DD {base['max_drawdown']*100:.1f}% &nbsp;·&nbsp; "
    f"BASE TRADES {base['n_trades']}"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Robustness gauge ────────────────────────────────────────────────────────

g_col, summary_col = st.columns([1.0, 1.6])

with g_col:
    gfig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall,
        number={"font": {"color": overall_color, "size": 64, "family": "JetBrains Mono"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED,
                     "tickfont": {"color": TEXT_MUTED, "size": 10},
                     "tickwidth": 1},
            "bar": {"color": overall_color, "thickness": 0.30},
            "bgcolor": BG_CARD,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "rgba(255,23,68,0.15)"},
                {"range": [40, 70],  "color": "rgba(255,193,7,0.15)"},
                {"range": [70, 100], "color": "rgba(0,230,118,0.15)"},
            ],
            "threshold": {
                "line": {"color": TEXT_PRIMARY, "width": 2},
                "thickness": 0.75,
                "value": overall,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    gl = get_plotly_layout()
    gl["height"] = 320
    gl["margin"] = dict(l=20, r=20, t=20, b=20)
    gl["paper_bgcolor"] = BG_PRIMARY
    gfig.update_layout(**gl)
    st.markdown(
        f"<div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};"
        f"text-transform:uppercase;letter-spacing:3px;text-align:center;"
        f"margin-top:16px;'>Strategy Robustness</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(gfig, use_container_width=True)
    st.markdown(
        f"<div style='font-family:DM Sans;font-size:1.1rem;color:{overall_color};"
        f"text-align:center;font-weight:700;margin-top:-20px;letter-spacing:2px;'>"
        f"{overall_label.upper()}</div>",
        unsafe_allow_html=True,
    )

with summary_col:
    st.markdown(section_header("Per-Parameter Scores"), unsafe_allow_html=True)
    for name, score in R["param_scores"].items():
        label, color = classify_robustness(score)
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:16px;
                        background:{BG_CARD};border:1px solid {BORDER};
                        border-left:3px solid {color};border-radius:10px;
                        padding:14px 18px;margin-bottom:10px;'>
                <div style='flex:1;'>
                    <div style='font-family:JetBrains Mono;color:{TEXT_PRIMARY};
                                font-size:0.95rem;font-weight:600;'>{name}</div>
                    <div style='font-family:DM Sans;color:{TEXT_MUTED};font-size:0.75rem;'>
                        base = {R['params'][name]} &nbsp;·&nbsp;
                        sweep over {len(R['ranges'][name])} values
                        ({min(R['ranges'][name])} to {max(R['ranges'][name])})
                    </div>
                </div>
                <div style='font-family:JetBrains Mono;color:{color};
                            font-size:1.6rem;font-weight:700;'>{score:.0f}</div>
                <div style='font-family:DM Sans;color:{color};font-size:0.7rem;
                            text-transform:uppercase;letter-spacing:2px;
                            min-width:75px;text-align:right;'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Heatmap ────────────────────────────────────────────────────────────────

st.markdown(section_header("Sensitivity Heatmap · Mean |Deviation| from Base"), unsafe_allow_html=True)


def deviation_color(dev_pct: float) -> tuple[str, float]:
    if not np.isfinite(dev_pct):
        return ACCENT_RED, 0.55
    if dev_pct <= 10:
        return ACCENT_GREEN, 0.55
    if dev_pct <= 20:
        return ACCENT_CYAN, 0.45
    if dev_pct <= 40:
        return ACCENT_AMBER, 0.45
    return ACCENT_RED, 0.55


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# Header row
header_cols = "120px " + " ".join(["1fr"] * len(METRICS))
hm = f"<div style='display:grid;grid-template-columns:{header_cols};gap:6px;'>"
hm += f"<div></div>"
for m in METRICS:
    hm += (
        f"<div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};"
        f"text-transform:uppercase;letter-spacing:2px;text-align:center;"
        f"padding:6px 0;'>{METRIC_LABELS[m]}</div>"
    )

for param in R["heatmap"].index:
    hm += (
        f"<div style='font-family:JetBrains Mono;color:{TEXT_PRIMARY};font-size:0.85rem;"
        f"display:flex;align-items:center;'>{param}</div>"
    )
    for metric in METRICS:
        dev = float(R["heatmap"].loc[param, metric])
        color, alpha = deviation_color(dev)
        bg = hex_to_rgba(color, alpha)
        text = f"{dev:.1f}%" if np.isfinite(dev) else "N/A"
        hm += (
            f"<div style='background:{bg};border:1px solid {hex_to_rgba(color, 0.35)};"
            f"border-radius:8px;padding:18px 8px;text-align:center;"
            f"font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;"
            f"color:{TEXT_PRIMARY};'>{text}</div>"
        )
hm += "</div>"
st.markdown(hm, unsafe_allow_html=True)

st.markdown(
    f"<div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};"
    f"margin-top:10px;'>Cell shows mean |deviation| of the metric from its "
    f"base value across the sweep range. Green = within 10%, cyan = within 20%, "
    f"amber = within 40%, red = beyond 40%.</div>",
    unsafe_allow_html=True,
)


# ── Per-parameter detail charts ────────────────────────────────────────────

st.markdown(section_header("Per-Parameter Detail · Sharpe vs. Value"), unsafe_allow_html=True)

detail_cols = st.columns(2)
chart_idx = 0
for name, df in R["sweeps"].items():
    score = R["param_scores"][name]
    label, color = classify_robustness(score)
    base_val = R["params"][name]
    base_sharpe = R["base_metrics"]["sharpe"]

    # Stable zone: within 20% of base sharpe (use min/max for sign safety)
    band_lo = min(base_sharpe * 0.8, base_sharpe * 1.2)
    band_hi = max(base_sharpe * 0.8, base_sharpe * 1.2)

    fig = go.Figure()

    # Stable zone band
    if base_sharpe != 0:
        fig.add_hrect(
            y0=band_lo, y1=band_hi,
            fillcolor="rgba(0,230,118,0.08)",
            line_width=0, layer="below",
        )

    # Performance curve
    fig.add_trace(go.Scatter(
        x=df["value"], y=df["sharpe"],
        mode="lines+markers",
        line=dict(color=TEXT_PRIMARY, width=2),
        marker=dict(size=5, color=TEXT_PRIMARY),
        name="Sharpe",
        hovertemplate=f"{name}=%{{x}}<br>Sharpe %{{y:.3f}}<extra></extra>",
    ))

    # Base value vertical line
    fig.add_vline(
        x=base_val,
        line=dict(color=ACCENT_CYAN, width=1.5, dash="dash"),
        annotation_text=f"base={base_val}",
        annotation_position="top",
        annotation_font=dict(color=ACCENT_CYAN, family="JetBrains Mono", size=10),
    )

    layout = get_plotly_layout()
    layout["height"] = 280
    layout["margin"] = dict(l=50, r=20, t=40, b=40)
    layout["showlegend"] = False
    layout["title"] = dict(
        text=f"<b style='color:{TEXT_PRIMARY}'>{name}</b> "
             f"<span style='color:{color};font-size:11px'>· {label} {score:.0f}/100</span>",
        font=dict(family="DM Sans", size=12),
        x=0.02, xanchor="left", y=0.98, yanchor="top",
    )
    layout["xaxis"]["title"] = dict(text=name, font=dict(color=TEXT_MUTED, size=10))
    layout["yaxis"]["title"] = dict(text="Sharpe", font=dict(color=TEXT_MUTED, size=10))
    fig.update_layout(**layout)

    detail_cols[chart_idx % 2].plotly_chart(fig, use_container_width=True)
    chart_idx += 1


# ── Per-parameter interpretation cards ─────────────────────────────────────

st.markdown(section_header("Interpretation"), unsafe_allow_html=True)

interp_cols = st.columns(2)
for i, (name, score) in enumerate(R["param_scores"].items()):
    label, color = classify_robustness(score)
    rng = R["ranges"][name]
    rmin, rmax = min(rng), max(rng)
    df = R["sweeps"][name]

    if score >= 70:
        msg = (
            f"<b>{name}</b> is <b style='color:{color}'>Robust ({score:.0f}/100)</b> - "
            f"Sharpe stays stable across the {rmin}-{rmax} range. Choice within this "
            f"window has little effect on performance."
        )
    elif score >= 40:
        msg = (
            f"<b>{name}</b> is <b style='color:{color}'>Moderate ({score:.0f}/100)</b> - "
            f"Sharpe varies meaningfully across {rmin}-{rmax}. Stick close to the base "
            f"value or revalidate before deploying alternatives."
        )
    else:
        msg = (
            f"<b>{name}</b> is <b style='color:{color}'>Fragile ({score:.0f}/100)</b> - "
            f"small changes across {rmin}-{rmax} produce large Sharpe swings. "
            f"This is a strong overfitting signal: the base value may have been "
            f"tuned to a sweet spot that won't generalize."
        )

    sharpe_min = float(df["sharpe"].min()) if len(df) else 0
    sharpe_max = float(df["sharpe"].max()) if len(df) else 0

    interp_cols[i % 2].markdown(
        f"""
        <div style='background:{BG_CARD};border:1px solid {BORDER};
                    border-left:4px solid {color};border-radius:12px;
                    padding:18px 22px;margin-bottom:12px;'>
            <div style='font-family:DM Sans;color:{TEXT_SECONDARY};
                        font-size:0.92rem;line-height:1.6;'>{msg}</div>
            <div style='font-family:JetBrains Mono;color:{TEXT_MUTED};
                        font-size:0.72rem;margin-top:10px;'>
                Sharpe range: <span style='color:{TEXT_PRIMARY}'>{sharpe_min:.2f} to {sharpe_max:.2f}</span>
                &nbsp;·&nbsp; sweep size: <span style='color:{TEXT_PRIMARY}'>{len(rng)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Raw sweep data ─────────────────────────────────────────────────────────

with st.expander("Raw sweep data"):
    for name, df in R["sweeps"].items():
        st.markdown(
            f"<div style='font-family:DM Sans;color:{ACCENT_CYAN};font-size:0.85rem;"
            f"margin:14px 0 6px 0;font-weight:600;'>{name}</div>",
            unsafe_allow_html=True,
        )
        out = df.copy()
        out["total_return"] = (out["total_return"] * 100).round(2)
        out["max_drawdown"] = (out["max_drawdown"] * 100).round(2)
        out["win_rate"] = (out["win_rate"] * 100).round(1)
        out["sharpe"] = out["sharpe"].round(3)
        st.dataframe(style_dataframe(out), use_container_width=True, hide_index=True)
