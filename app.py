"""
Trading Dashboard Suite — landing page.

8 regime-aware tools for quantitative trading analysis. Pages are auto-loaded
from the pages/ directory by Streamlit's multipage mechanism — use the sidebar
to navigate. Shared modules: design_system.py (theme + helpers) and
data_provider.py (FMP → TwelveData → yfinance fallback chain).

Run with:
    py -3.13 -m streamlit run app.py
"""

import streamlit as st

from data_provider import available_providers
from design_system import *  # noqa: F401,F403

st.set_page_config(
    page_title="Trading Dashboard Suite",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.markdown(
    f"""
    <div style='text-align:center;padding:50px 20px 20px 20px;'>
      <div style='font-family:DM Sans;font-size:0.7rem;color:{ACCENT_CYAN};
                  text-transform:uppercase;letter-spacing:5px;margin-bottom:8px;'>
        {status_dot('connected')}Trading Suite
      </div>
      <h1 style='color:{TEXT_PRIMARY};font-family:DM Sans;font-size:2.6rem;margin:0;
                 font-weight:700;letter-spacing:0.5px;'>
        Trading Dashboard Suite
      </h1>
      <p style='color:{TEXT_SECONDARY};font-size:1.0rem;margin-top:6px;
                font-family:DM Sans;'>
        8 regime-aware tools for quantitative analysis
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data provider status banner
providers = available_providers()
provider_str = " → ".join(providers)
st.markdown(
    f"<div style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};"
    f"letter-spacing:2px;text-align:center;margin-bottom:24px;'>"
    f"DATA CHAIN: {provider_str}</div>",
    unsafe_allow_html=True,
)

DASHBOARDS = [
    ("Regime Detection",          "pages/1_Regime_Detection.py",
     "HMM-based market regime classification with forward-only filtering",  ACCENT_CYAN),
    ("Monte Carlo Simulation",    "pages/2_Monte_Carlo.py",
     "Equity curve fan chart from bootstrap-resampled trade returns",       ACCENT_GREEN),
    ("Sensitivity Analysis",      "pages/3_Sensitivity_Analysis.py",
     "Parameter robustness scoring for SMA-crossover strategies",           ACCENT_AMBER),
    ("Multi-Asset Backtester",    "pages/4_Multi_Asset_Backtester.py",
     "Walk-forward regime backtesting across SPY, BTC, GLD, TLT",           ACCENT_VIOLET),
    ("Portfolio Risk",            "pages/5_Portfolio_Risk.py",
     "Position monitoring with regime overlays and stress testing",          ACCENT_CYAN),
    ("Sentiment Analysis",        "pages/6_Sentiment_Analysis.py",
     "News sentiment via VADER + finance lexicon, with cockpit gauges",     ACCENT_GREEN),
    ("Market Screener",           "pages/7_Market_Screener.py",
     "Scan a universe for regime, confidence and technical alignment",      ACCENT_AMBER),
    ("Correlation Break Detector","pages/8_Correlation_Breaks.py",
     "Monitor pair correlations and surface structural breaks",              ACCENT_RED),
]

cols = st.columns(2)
for i, (name, path, desc, color) in enumerate(DASHBOARDS):
    with cols[i % 2]:
        st.markdown(
            f"""
            <div style='background:{BG_CARD};border:1px solid {BORDER};
                        border-left:4px solid {color};border-radius:12px;
                        padding:18px 22px;margin:8px 0;'>
              <h3 style='color:{color};margin:0 0 6px 0;font-family:DM Sans;
                         font-size:1.05rem;letter-spacing:0.5px;'>{name}</h3>
              <p style='color:{TEXT_SECONDARY};margin:0;font-family:DM Sans;
                        font-size:0.85rem;line-height:1.5;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"<p style='color:{TEXT_MUTED};text-align:center;margin-top:36px;"
    f"font-size:0.75rem;font-family:DM Sans;'>"
    f"Use the sidebar to navigate ◀</p>",
    unsafe_allow_html=True,
)
