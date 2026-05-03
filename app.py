import streamlit as st
from design_system import *

apply_theme()

st.markdown(f"""
<div style="text-align: center; padding: 60px 20px 30px 20px;">
    <h1 style="color: {TEXT_PRIMARY}; font-size: 42px; margin: 0;">Trading Dashboard Suite</h1>
    <p style="color: {TEXT_SECONDARY}; font-size: 16px; margin-top: 8px;">8 regime-aware tools for quantitative trading analysis</p>
</div>
""", unsafe_allow_html=True)

dashboards = [
    ("Regime Detection", "pages/1_Regime_Detection.py", "HMM-based market regime classification with forward-only filtering", ACCENT_CYAN),
    ("Monte Carlo Simulation", "pages/2_Monte_Carlo.py", "Equity curve simulation with fan chart and overfitting detection", ACCENT_GREEN),
    ("Sensitivity Analysis", "pages/3_Sensitivity_Analysis.py", "Parameter robustness scoring for moving average strategies", ACCENT_AMBER),
    ("Multi-Asset Backtester", "pages/4_Multi_Asset_Backtester.py", "Walk-forward regime backtesting across SPY, BTC, GLD, TLT", ACCENT_VIOLET),
    ("Portfolio Risk", "pages/5_Portfolio_Risk.py", "Position monitoring with regime overlays and stress testing", ACCENT_CYAN),
    ("Sentiment Analysis", "pages/6_Sentiment_Analysis.py", "News sentiment scoring via VADER with cockpit-style gauges", ACCENT_GREEN),
    ("Market Screener", "pages/7_Market_Screener.py", "Scan 30+ tickers for regime, confidence, and technical alignment", ACCENT_AMBER),
    ("Correlation Break Detector", "pages/8_Correlation_Breaks.py", "Monitor pair correlations and detect structural breaks", ACCENT_RED),
]

cols = st.columns(2)
for i, (name, path, desc, color) in enumerate(dashboards):
    with cols[i % 2]:
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-left: 4px solid {color};
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
        ">
            <h3 style="color: {color}; margin: 0 0 6px 0; font-size: 18px;">{name}</h3>
            <p style="color: {TEXT_SECONDARY}; margin: 0; font-size: 14px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<p style="color: {TEXT_MUTED}; text-align: center; margin-top: 40px; font-size: 13px;">
    Use the sidebar ◀ to navigate between dashboards
</p>
""", unsafe_allow_html=True)
