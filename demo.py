"""
Design-system demo — exercises every component so we can visually verify
the theme before building the real dashboards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from design_system import *

# ── Theme ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Design System Demo", layout="wide")
apply_theme()

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{TEXT_PRIMARY};margin-bottom:0'>Trading Design System</h1>"
    f"<p style='color:{TEXT_MUTED};margin-top:4px'>Component showcase</p>",
    unsafe_allow_html=True,
)

# ── Section: Metric Cards ────────────────────────────────────────────────────
st.markdown(section_header("METRIC CARDS"), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Total P&L", "+$12,345", pnl_color(12345)), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Win Rate", "68.5%", ACCENT_CYAN), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Sharpe Ratio", "1.23", ACCENT_AMBER), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Max Drawdown", "-5.2%", pnl_color(-5.2)), unsafe_allow_html=True)

# ── Section: Regime Badges ───────────────────────────────────────────────────
st.markdown(section_header("REGIME BADGES"), unsafe_allow_html=True)

badges = "".join([
    regime_badge("Bull", 85),
    "&nbsp;&nbsp;",
    regime_badge("Low Vol"),
    "&nbsp;&nbsp;",
    regime_badge("High Vol", 62),
    "&nbsp;&nbsp;",
    regime_badge("Uncertain", 45),
])
st.markdown(badges, unsafe_allow_html=True)

# ── Section: Status Dots ────────────────────────────────────────────────────
st.markdown(section_header("STATUS INDICATORS"), unsafe_allow_html=True)

dots_html = (
    f"<p>{status_dot('connected')} <span style='color:{TEXT_SECONDARY}'>Broker Connected</span></p>"
    f"<p>{status_dot('warning')} <span style='color:{TEXT_SECONDARY}'>Low Margin Warning</span></p>"
    f"<p>{status_dot('error')} <span style='color:{TEXT_SECONDARY}'>API Disconnected</span></p>"
)
st.markdown(dots_html, unsafe_allow_html=True)

# ── Section: Plotly Chart ────────────────────────────────────────────────────
st.markdown(section_header("SAMPLE PLOTLY CHART"), unsafe_allow_html=True)

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=120, freq="B")
prices = 100 + np.cumsum(np.random.randn(120) * 0.8)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates, y=prices,
    mode="lines",
    line=dict(color=ACCENT_CYAN, width=2),
    name="Price",
))
fig.update_layout(
    **get_plotly_layout(),
    title=dict(text="Sample Equity Curve", font=dict(color=TEXT_PRIMARY)),
    height=420,
    xaxis_title="Date",
    yaxis_title="Price",
)
st.plotly_chart(fig, width="stretch")

# ── Section: Styled Dataframe ────────────────────────────────────────────────
st.markdown(section_header("STYLED DATAFRAME"), unsafe_allow_html=True)

df = pd.DataFrame({
    "Symbol": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
    "Price":  [218.50, 178.30, 425.10, 172.60, 950.20],
    "Change%": [2.5, -1.2, 0.8, -3.4, 5.1],
    "Volume":  ["45 M", "12 M", "28 M", "35 M", "25 M"],
})

st.dataframe(style_dataframe(df), width="stretch", hide_index=True)
