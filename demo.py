import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from design_system import *

# Apply the theme
apply_theme()

st.title("Trading Dashboard Design System Demo")

# Section Header
st.markdown(section_header("METRIC CARDS"), unsafe_allow_html=True)

# Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card("Total P&L", "+$12,345", pnl_color(12345)), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Win Rate", "68.5%", ACCENT_CYAN), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("Sharpe Ratio", "1.23", ACCENT_AMBER), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("Max Drawdown", "-5.2%", pnl_color(-5.2)), unsafe_allow_html=True)

# Section Header
st.markdown(section_header("REGIME BADGES"), unsafe_allow_html=True)

# Regime Badges
st.markdown("Current Market Regime:", unsafe_allow_html=True)
st.markdown(regime_badge("Bull", 85), unsafe_allow_html=True)
st.markdown(regime_badge("High Vol"), unsafe_allow_html=True)
st.markdown(regime_badge("Uncertain", 45), unsafe_allow_html=True)

# Section Header
st.markdown(section_header("STATUS DOTS"), unsafe_allow_html=True)

# Status Dots
st.markdown(status_dot("connected") + " Connected to Broker", unsafe_allow_html=True)
st.markdown(status_dot("warning") + " Low Margin Warning", unsafe_allow_html=True)
st.markdown(status_dot("error") + " API Disconnected", unsafe_allow_html=True)

# Section Header
st.markdown(section_header("SAMPLE PLOTLY CHART"), unsafe_allow_html=True)

# Sample Plotly Chart
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
df_chart = pd.DataFrame({'Date': dates, 'Price': prices})

fig = px.line(df_chart, x='Date', y='Price', title='Sample Price Chart')
fig.update_layout(**get_plotly_layout())
st.plotly_chart(fig, use_container_width=True)

# Section Header
st.markdown(section_header("STYLED DATAFRAME"), unsafe_allow_html=True)

# Sample DataFrame
sample_data = {
    'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
    'Price': [150.25, 2800.50, 305.75, 245.80, 450.20],
    'Change': [2.5, -1.2, 0.8, -3.4, 5.1],
    'Volume': [45000000, 1200000, 28000000, 35000000, 25000000]
}
df = pd.DataFrame(sample_data)
df['Change Color'] = df['Change'].apply(pnl_color)

st.dataframe(style_dataframe(df))