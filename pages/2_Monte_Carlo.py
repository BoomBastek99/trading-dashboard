"""
Monte Carlo Simulation Dashboard.

Bootstrap-resamples a backtest's trade returns (with random noise) to
estimate the distribution of outcomes. The fan chart of 1,000 overlapping
equity curves at very low opacity creates a glowing probability cloud.

Run with:
    py -3.13 -m streamlit run monte_carlo_dashboard.py --server.port=8502
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from design_system import *  # noqa: F401,F403

st.set_page_config(
    page_title="Monte Carlo Simulation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Sample backtest generator ───────────────────────────────────────────────

def generate_sample_backtest(n_trades: int = 200, seed: int = 42) -> pd.DataFrame:
    """Two-state Markov mixture: a 'normal' regime with positive edge and
    a 'bad' regime with negative edge. Sticky transitions create loss clusters."""
    rng = np.random.default_rng(seed)
    states = np.zeros(n_trades, dtype=int)
    # Sticky transitions: normal regime ~80% of time, bad regime ~20%, with
    # avg run lengths of ~20 / ~5 trades to create visible loss clusters.
    for i in range(1, n_trades):
        if states[i - 1] == 0:
            states[i] = 0 if rng.random() < 0.95 else 1
        else:
            states[i] = 1 if rng.random() < 0.80 else 0
    rets = np.where(
        states == 0,
        rng.normal(0.005, 0.011, n_trades),
        rng.normal(-0.004, 0.020, n_trades),
    )
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_trades)
    return pd.DataFrame({"date": dates, "trade_return": rets})


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "trade_return" not in df.columns:
        raise ValueError("CSV must have columns: date, trade_return")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["trade_return"] = pd.to_numeric(df["trade_return"], errors="coerce")
    df = df.dropna(subset=["trade_return"])
    return df.reset_index(drop=True)


# ── Monte Carlo engine ──────────────────────────────────────────────────────

def equity_curve(returns: np.ndarray, capital: float) -> np.ndarray:
    """Equity curve including the starting capital point. Length n+1."""
    growth = np.cumprod(1.0 + returns)
    return np.concatenate([[capital], capital * growth])


def max_drawdown(curve: np.ndarray) -> float:
    """Maximum peak-to-trough fractional drawdown (a negative number)."""
    running_max = np.maximum.accumulate(curve)
    return float(((curve - running_max) / running_max).min())


def run_monte_carlo(
    returns: np.ndarray,
    capital: float,
    n_sims: int,
    noise_pct: float = 0.003,
    method: str = "bootstrap",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: build (n_sims, n) resampled+noised return matrix, integrate
    to equity curves, compute per-curve max drawdown.

    method:
        "bootstrap" - sample n returns WITH replacement (standard MC; fans out)
        "shuffle"   - permute the n returns (preserves total return; pinches)

    Returns:
        equity:  shape (n_sims, n+1) - includes starting capital column
        max_dd:  shape (n_sims,)     - most-negative drawdown per curve
    """
    rng = np.random.default_rng(seed)
    n = len(returns)

    if method == "shuffle":
        idx = np.argsort(rng.random((n_sims, n)), axis=1)
    else:  # bootstrap with replacement
        idx = rng.integers(0, n, size=(n_sims, n))
    resampled = returns[idx]
    noise = rng.uniform(-noise_pct, noise_pct, size=(n_sims, n))
    sims = resampled + noise

    growth = np.cumprod(1.0 + sims, axis=1)
    equity = capital * growth
    equity = np.column_stack([np.full(n_sims, capital), equity])

    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min(axis=1)
    return equity, max_dd


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Inputs"), unsafe_allow_html=True)
    upload = st.file_uploader("Backtest CSV (date, trade_return)", type=["csv"])

    # Offer the bundled sample CSV so users can see the expected schema
    # without leaving the app.
    import os as _os
    _sample_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "samples", "sample_backtest.csv",
    )
    if _os.path.exists(_sample_path):
        with open(_sample_path, "rb") as _f:
            st.download_button(
                "Download sample CSV",
                data=_f.read(),
                file_name="sample_backtest.csv",
                mime="text/csv",
                use_container_width=True,
            )

    capital = st.number_input(
        "Starting capital ($)",
        min_value=1000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=1000.0,
        format="%.0f",
    )
    n_sims = st.slider("Number of simulations", 100, 5000, 1000, step=100)
    noise_bps = st.slider("Per-trade noise (basis points, +/-)", 0, 100, 30, step=5)
    method = st.selectbox(
        "Resample method",
        options=["bootstrap", "shuffle"],
        index=0,
        help="bootstrap = sample with replacement (fans out, standard MC); "
             "shuffle = permute the historical trades (preserves total return)",
    )
    run = st.button("Run Simulation", type="primary", use_container_width=True)


# ── Run ─────────────────────────────────────────────────────────────────────

if "mc_results" not in st.session_state:
    st.session_state.mc_results = None

if run:
    with st.spinner("Resampling and integrating curves..."):
        try:
            if upload is not None:
                bt = load_csv(upload)
                source = f"CSV ({len(bt)} trades)"
            else:
                bt = generate_sample_backtest()
                source = f"Sample data ({len(bt)} trades)"
        except Exception as exc:
            st.error(f"Failed to load CSV: {exc}")
            st.stop()

        if len(bt) < 20:
            st.error(f"Need at least 20 trades; got {len(bt)}.")
            st.stop()

        returns = bt["trade_return"].to_numpy()
        original_curve = equity_curve(returns, capital)
        original_final = float(original_curve[-1])
        original_dd = max_drawdown(original_curve)

        equity, max_dd = run_monte_carlo(
            returns,
            capital,
            n_sims=int(n_sims),
            noise_pct=noise_bps / 10_000.0,
            method=method,
        )

        finals = equity[:, -1]
        median_final = float(np.median(finals))
        p5_final = float(np.percentile(finals, 5))
        p95_final = float(np.percentile(finals, 95))

        prob_loss = float((finals < capital).mean())
        prob_dd_20 = float((max_dd <= -0.20).mean())
        prob_dd_30 = float((max_dd <= -0.30).mean())

        dd_pcts = np.percentile(max_dd, [5, 25, 50, 75, 95])

        # Where the original sits in the simulated final-value distribution
        original_pct = float((finals < original_final).mean() * 100)

        st.session_state.mc_results = {
            "source": source,
            "bt": bt,
            "capital": capital,
            "n_sims": int(n_sims),
            "noise_pct": noise_bps / 10_000.0,
            "method": method,
            "equity": equity,
            "max_dd": max_dd,
            "finals": finals,
            "original_curve": original_curve,
            "original_final": original_final,
            "original_dd": original_dd,
            "median_final": median_final,
            "p5_final": p5_final,
            "p95_final": p95_final,
            "prob_loss": prob_loss,
            "prob_dd_20": prob_dd_20,
            "prob_dd_30": prob_dd_30,
            "dd_pcts": dd_pcts,
            "original_pct": original_pct,
        }


# ── Render ──────────────────────────────────────────────────────────────────

R = st.session_state.mc_results

if R is None:
    st.markdown(
        f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;"
        f"font-family:DM Sans,sans-serif;font-size:0.95rem;'>"
        f"Upload a backtest CSV (or leave empty for sample data) and click "
        f"<b style='color:{ACCENT_CYAN}'>Run Simulation</b>."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Source banner
st.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    f"color:{TEXT_MUTED};letter-spacing:2px;margin-bottom:14px;'>"
    f"{status_dot('connected')}{R['source'].upper()} &nbsp;·&nbsp; "
    f"{R['n_sims']:,} SIMULATIONS &nbsp;·&nbsp; "
    f"METHOD: {R['method'].upper()} &nbsp;·&nbsp; "
    f"NOISE +/- {R['noise_pct']*10000:.0f} BPS</div>",
    unsafe_allow_html=True,
)


# ── Top metric cards ────────────────────────────────────────────────────────

prob_loss_pct = R["prob_loss"] * 100
if prob_loss_pct >= 30:
    loss_color = ACCENT_RED
elif prob_loss_pct >= 10:
    loss_color = ACCENT_AMBER
else:
    loss_color = ACCENT_GREEN

median_ret_pct = (R["median_final"] / R["capital"] - 1.0) * 100
ret_color = pnl_color(median_ret_pct)

worst_dd_pct = R["dd_pcts"][0] * 100  # 5th percentile (most negative)

orig_pct = R["original_pct"]
if orig_pct >= 90:
    overfit_text, overfit_color = "HIGH", ACCENT_RED
elif orig_pct >= 75:
    overfit_text, overfit_color = "MEDIUM", ACCENT_AMBER
else:
    overfit_text, overfit_color = "LOW", ACCENT_GREEN

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Probability of Loss", f"{prob_loss_pct:.1f}%", loss_color), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Median Return", f"{median_ret_pct:+.1f}%", ret_color), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Worst-5% Drawdown", f"{worst_dd_pct:.1f}%", ACCENT_RED), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Overfitting Risk", overfit_text, overfit_color), unsafe_allow_html=True)


# ── Hero fan chart ─────────────────────────────────────────────────────────

st.markdown(section_header("Equity Curve Fan · 1,000 Bootstrap Paths"), unsafe_allow_html=True)

equity = R["equity"]
n_sims, T = equity.shape
trade_idx = np.arange(T)

# Single-trace cloud: concatenate all paths separated by None for fast WebGL render.
xs = np.empty((n_sims, T + 1), dtype=float)
xs[:, :T] = trade_idx
xs[:, T] = np.nan
ys = np.empty((n_sims, T + 1), dtype=float)
ys[:, :T] = equity
ys[:, T] = np.nan

p5 = np.percentile(equity, 5, axis=0)
p25 = np.percentile(equity, 25, axis=0)
p50 = np.percentile(equity, 50, axis=0)
p75 = np.percentile(equity, 75, axis=0)
p95 = np.percentile(equity, 95, axis=0)

fig = go.Figure()

# Cloud of all paths (single Scattergl trace, low opacity = glow)
fig.add_trace(go.Scattergl(
    x=xs.ravel(),
    y=ys.ravel(),
    mode="lines",
    line=dict(color=ACCENT_CYAN, width=0.6),
    opacity=0.02,
    hoverinfo="skip",
    showlegend=False,
    name="paths",
))

# 5-95 band: upper invisible, lower fills to it
fig.add_trace(go.Scatter(
    x=trade_idx, y=p95, mode="lines",
    line=dict(color="rgba(0,0,0,0)", width=0),
    hoverinfo="skip", showlegend=False, name="p95",
))
fig.add_trace(go.Scatter(
    x=trade_idx, y=p5, mode="lines",
    line=dict(color="rgba(0,0,0,0)", width=0),
    fill="tonexty", fillcolor="rgba(0,212,255,0.06)",
    hoverinfo="skip", showlegend=False, name="p5",
))

# 25-75 inner band
fig.add_trace(go.Scatter(
    x=trade_idx, y=p75, mode="lines",
    line=dict(color="rgba(0,0,0,0)", width=0),
    hoverinfo="skip", showlegend=False, name="p75",
))
fig.add_trace(go.Scatter(
    x=trade_idx, y=p25, mode="lines",
    line=dict(color="rgba(0,0,0,0)", width=0),
    fill="tonexty", fillcolor="rgba(0,212,255,0.10)",
    hoverinfo="skip", showlegend=False, name="p25",
))

# Median
fig.add_trace(go.Scatter(
    x=trade_idx, y=p50, mode="lines",
    line=dict(color=ACCENT_CYAN, width=2.2),
    name="Median",
    hovertemplate="Trade %{x}<br>$%{y:,.0f}<extra>median</extra>",
))

# Original backtest
fig.add_trace(go.Scatter(
    x=trade_idx, y=R["original_curve"],
    mode="lines",
    line=dict(color=TEXT_PRIMARY, width=2.5),
    name="Original backtest",
    hovertemplate="Trade %{x}<br>$%{y:,.0f}<extra>original</extra>",
))

layout = get_plotly_layout()
layout["height"] = 600
layout["margin"] = dict(l=50, r=20, t=20, b=40)
layout["showlegend"] = True
layout["legend"] = dict(
    bgcolor="rgba(10,10,15,0.6)",
    bordercolor=BORDER,
    borderwidth=1,
    font=dict(color=TEXT_SECONDARY, size=11),
    x=0.01, y=0.99, xanchor="left", yanchor="top",
)
layout["xaxis"]["title"] = dict(text="Trade #", font=dict(color=TEXT_MUTED, size=11))
layout["yaxis"]["tickprefix"] = "$"
layout["yaxis"]["tickformat"] = ",.0f"
fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)


# ── Side-by-side histograms ────────────────────────────────────────────────

h1, h2 = st.columns(2)

with h1:
    st.markdown(section_header("Final Portfolio Value Distribution"), unsafe_allow_html=True)
    hfig = go.Figure()
    hfig.add_trace(go.Histogram(
        x=R["finals"],
        nbinsx=60,
        marker=dict(
            color="rgba(0,212,255,0.18)",
            line=dict(color=ACCENT_CYAN, width=1),
        ),
        hovertemplate="$%{x:,.0f}<br>%{y} sims<extra></extra>",
        showlegend=False,
    ))
    hfig.add_vline(
        x=R["original_final"],
        line=dict(color=TEXT_PRIMARY, width=2, dash="dash"),
        annotation_text=f"Original  ${R['original_final']:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color=TEXT_PRIMARY, family="JetBrains Mono"),
    )
    hfig.add_vline(
        x=R["capital"],
        line=dict(color=TEXT_MUTED, width=1, dash="dot"),
    )
    hl = get_plotly_layout()
    hl["height"] = 320
    hl["margin"] = dict(l=50, r=20, t=20, b=40)
    hl["xaxis"]["tickprefix"] = "$"
    hl["xaxis"]["tickformat"] = ",.0f"
    hl["bargap"] = 0.05
    hfig.update_layout(**hl)
    st.plotly_chart(hfig, use_container_width=True)

with h2:
    st.markdown(section_header("Max Drawdown Distribution"), unsafe_allow_html=True)
    dfig = go.Figure()
    dfig.add_trace(go.Histogram(
        x=R["max_dd"] * 100,
        nbinsx=60,
        marker=dict(
            color="rgba(255,23,68,0.18)",
            line=dict(color=ACCENT_RED, width=1),
        ),
        hovertemplate="%{x:.1f}%<br>%{y} sims<extra></extra>",
        showlegend=False,
    ))
    dfig.add_vline(
        x=R["original_dd"] * 100,
        line=dict(color=TEXT_PRIMARY, width=2, dash="dash"),
        annotation_text=f"Original  {R['original_dd']*100:.1f}%",
        annotation_position="top left",
        annotation_font=dict(color=TEXT_PRIMARY, family="JetBrains Mono"),
    )
    dl = get_plotly_layout()
    dl["height"] = 320
    dl["margin"] = dict(l=50, r=20, t=20, b=40)
    dl["xaxis"]["ticksuffix"] = "%"
    dl["bargap"] = 0.05
    dfig.update_layout(**dl)
    st.plotly_chart(dfig, use_container_width=True)


# ── Interpretation card ────────────────────────────────────────────────────

st.markdown(section_header("Analysis"), unsafe_allow_html=True)

original_ret_pct = (R["original_final"] / R["capital"] - 1.0) * 100
median_ret_pct = (R["median_final"] / R["capital"] - 1.0) * 100

interp = (
    f"Based on <b style='color:{ACCENT_CYAN}'>{R['n_sims']:,}</b> simulations, "
    f"there is a <b style='color:{loss_color}'>{R['prob_loss']*100:.1f}%</b> chance of "
    f"losing money, a <b style='color:{ACCENT_AMBER}'>{R['prob_dd_20']*100:.1f}%</b> chance "
    f"of a 20%+ drawdown, and a <b style='color:{ACCENT_RED}'>{R['prob_dd_30']*100:.1f}%</b> "
    f"chance of a 30%+ drawdown. "
    f"The median outcome is <b style='color:{pnl_color(median_ret_pct)}'>"
    f"{median_ret_pct:+.1f}%</b> "
    f"(5-95% range: <b style='color:{TEXT_PRIMARY}'>"
    f"${R['p5_final']:,.0f} - ${R['p95_final']:,.0f}</b>). "
    f"Your original backtest returned <b style='color:{pnl_color(original_ret_pct)}'>"
    f"{original_ret_pct:+.1f}%</b>, which falls in the "
    f"<b style='color:{ACCENT_CYAN}'>{R['original_pct']:.0f}th</b> percentile."
)

st.markdown(
    f"""
    <div style="
        background:{BG_CARD};
        border:1px solid {BORDER};
        border-radius:12px;
        padding:22px 26px;
        font-family:DM Sans,sans-serif;
        color:{TEXT_SECONDARY};
        line-height:1.7;
        font-size:0.95rem;
    ">{interp}</div>
    """,
    unsafe_allow_html=True,
)

if R["original_pct"] >= 90:
    st.markdown(
        f"""
        <div style="
            background:rgba(255,193,7,0.08);
            border:1px solid {ACCENT_AMBER};
            border-left:4px solid {ACCENT_AMBER};
            border-radius:12px;
            padding:18px 22px;
            margin-top:14px;
            color:{TEXT_PRIMARY};
            font-family:DM Sans,sans-serif;
        ">
            <div style="font-family:DM Sans;font-size:0.78rem;color:{ACCENT_AMBER};
                        text-transform:uppercase;letter-spacing:2.5px;font-weight:700;
                        margin-bottom:8px;">Overfitting Warning</div>
            Your backtest sits in the top {100 - R['original_pct']:.0f}% of bootstrap
            outcomes. The historical trade ordering may have been unusually favorable -
            re-evaluate edge assumptions before sizing capital to this strategy.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Drawdown percentile reference
with st.expander("Max drawdown percentiles"):
    dd_df = pd.DataFrame({
        "Percentile": ["5th (worst)", "25th", "50th (median)", "75th", "95th (best)"],
        "Max Drawdown": [f"{x*100:.2f}%" for x in R["dd_pcts"]],
    }).set_index("Percentile")
    st.dataframe(style_dataframe(dd_df), use_container_width=True)
