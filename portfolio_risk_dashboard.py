import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from design_system import *

# Apply theme
apply_theme()

# Demo positions
demo_positions = [
    {"ticker": "SPY", "shares": 100, "entry_price": 540, "current_price": 558},
    {"ticker": "QQQ", "shares": 50, "entry_price": 480, "current_price": 495},
    {"ticker": "AAPL", "shares": 75, "entry_price": 210, "current_price": 218},
    {"ticker": "GLD", "shares": 40, "entry_price": 235, "current_price": 242},
    {"ticker": "TLT", "shares": 60, "entry_price": 88, "current_price": 85}
]

# Sidebar
st.sidebar.header("Portfolio Management")

use_demo = st.sidebar.checkbox("Use Demo Portfolio", value=True)

if not use_demo:
    # Allow editing positions
    positions = []
    for pos in demo_positions:
        with st.sidebar.expander(f"Edit {pos['ticker']}"):
            shares = st.number_input(f"Shares {pos['ticker']}", value=pos['shares'], key=f"shares_{pos['ticker']}")
            entry = st.number_input(f"Entry Price {pos['ticker']}", value=pos['entry_price'], key=f"entry_{pos['ticker']}")
            current = st.number_input(f"Current Price {pos['ticker']}", value=pos['current_price'], key=f"current_{pos['ticker']}")
            positions.append({"ticker": pos['ticker'], "shares": shares, "entry_price": entry, "current_price": current})
else:
    positions = demo_positions

# Optional Alpaca
alpaca_key = st.sidebar.text_input("Alpaca API Key (optional)", type="password")
alpaca_secret = st.sidebar.text_input("Alpaca Secret (optional)", type="password")

watchlist = st.sidebar.multiselect("Watchlist Tickers", ["SPY", "BTC-USD", "ETH-USD", "NVDA", "TSLA"], default=["BTC-USD"])

refresh = st.sidebar.button("Refresh Data")

@st.cache_data
def download_price_data(tickers, period="2y"):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period)
            data[ticker] = df['Close']
        except:
            data[ticker] = pd.Series()
    return pd.DataFrame(data)

@st.cache_data
def run_regime_detection(df):
    if df.empty or len(df) < 50:
        return "Unknown", 0, 0
    
    returns = np.log(df / df.shift(1)).dropna()
    vol = returns.rolling(20).std().dropna()
    features = np.column_stack([returns.values[-len(vol):], vol.values])
    
    if len(features) < 50:
        return "Unknown", 0, 0
    
    model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=1000, random_state=42)
    model.fit(features)
    
    # Forward filtering
    log_likelihoods = model._compute_log_likelihood(features)
    log_alpha = np.zeros((len(features), 3))
    log_alpha[0] = model.startprob_ + log_likelihoods[0]
    for t in range(1, len(features)):
        log_alpha[t] = log_likelihoods[t] + np.log(np.sum(np.exp(log_alpha[t-1] + model.transmat_.T), axis=1))
    alpha = np.exp(log_alpha)
    posterior = alpha / alpha.sum(axis=1, keepdims=True)
    current_regime = np.argmax(posterior[-1])
    confidence = posterior[-1, current_regime]
    days_in_regime = np.sum(np.argmax(posterior, axis=1) == current_regime)
    
    # Label regimes by volatility
    regime_vols = [np.mean(vol.iloc[np.argmax(posterior, axis=1) == i]) for i in range(3)]
    sorted_regimes = np.argsort(regime_vols)
    regime_names = ["Low Vol", "Medium Vol", "High Vol"]
    regime_name = regime_names[sorted_regimes.tolist().index(current_regime)]
    
    return regime_name, confidence, days_in_regime

# Stress test drawdowns
stress_drawdowns = {
    "2008": {"SPY": -0.56, "QQQ": -0.54, "AAPL": -0.61, "GLD": 0.21, "TLT": 0.33},
    "2020": {"SPY": -0.34, "QQQ": -0.28, "AAPL": -0.31, "GLD": -0.03, "TLT": 0.21},
    "2022": {"SPY": -0.25, "QQQ": -0.33, "AAPL": -0.30, "GLD": -0.04, "TLT": -0.31}
}

if refresh or 'portfolio_data' not in st.session_state:
    with st.spinner("Loading portfolio data..."):
        tickers = [p['ticker'] for p in positions] + watchlist
        price_data = download_price_data(tickers)
        
        portfolio_data = []
        for pos in positions:
            ticker = pos['ticker']
            current_price = price_data[ticker].iloc[-1] if ticker in price_data.columns and not price_data[ticker].empty else pos['current_price']
            regime, conf, days = run_regime_detection(price_data[ticker] if ticker in price_data.columns else pd.Series())
            pnl = (current_price - pos['entry_price']) * pos['shares']
            pnl_pct = (current_price / pos['entry_price'] - 1) * 100
            
            portfolio_data.append({
                'ticker': ticker,
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'value': current_price * pos['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'regime': regime,
                'confidence': conf,
                'days_in_regime': days
            })
        
        # Correlation matrix
        returns_df = price_data.pct_change().dropna()
        corr_matrix = returns_df.rolling(60).corr().groupby(level=0).last()
        
        # Watchlist
        watchlist_data = []
        for ticker in watchlist:
            if ticker in price_data.columns and not price_data[ticker].empty:
                price = price_data[ticker].iloc[-1]
                regime, conf, days = run_regime_detection(price_data[ticker])
                watchlist_data.append({
                    'ticker': ticker,
                    'price': price,
                    'regime': regime,
                    'confidence': conf,
                    'days': days
                })
        
        st.session_state['portfolio_data'] = portfolio_data
        st.session_state['price_data'] = price_data
        st.session_state['corr_matrix'] = corr_matrix
        st.session_state['watchlist_data'] = watchlist_data

if 'portfolio_data' in st.session_state:
    portfolio_data = st.session_state['portfolio_data']
    corr_matrix = st.session_state['corr_matrix']
    watchlist_data = st.session_state['watchlist_data']
    
    # Top bar
    total_value = sum(p['value'] for p in portfolio_data)
    total_pnl = sum(p['pnl'] for p in portfolio_data)
    total_pnl_pct = (total_value / sum(p['entry_price'] * p['shares'] for p in portfolio_data) - 1) * 100
    n_positions = len(portfolio_data)
    favorable_regimes = sum(1 for p in portfolio_data if p['regime'] in ["Low Vol", "Medium Vol"])
    
    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
    with col1:
        st.markdown(f"<h1 style='color: {TEXT_PRIMARY}; font-size: 56px; margin: 0;'>{total_value:,.0f}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: " + TEXT_MUTED + "; margin: 0;'>Total Portfolio Value</p>", unsafe_allow_html=True)
    with col2:
        color = pnl_color(total_pnl_pct)
        st.markdown(f"<h2 style='color: {color}; margin: 0;'>{total_pnl:,.0f}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_MUTED}; margin: 0;'>{'▲' if total_pnl > 0 else '▼'} {abs(total_pnl_pct):.1f}%</p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h3 style='color: {TEXT_SECONDARY}; margin: 0;'>{n_positions}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: " + TEXT_MUTED + "; margin: 0;'>Positions</p>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<h3 style='color: {TEXT_SECONDARY}; margin: 0;'>{favorable_regimes}/{n_positions}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: " + TEXT_MUTED + "; margin: 0;'>Favorable Regimes</p>", unsafe_allow_html=True)
    with col5:
        st.markdown("<span style='color: " + ACCENT_GREEN + "; font-size: 24px;'>●</span>", unsafe_allow_html=True)
        st.markdown("<p style='color: " + TEXT_MUTED + "; margin: 0;'>Market Open</p>", unsafe_allow_html=True)
    
    # Positions
    st.markdown(section_header("Positions"), unsafe_allow_html=True)
    
    for pos in portfolio_data:
        color = REGIME_COLORS.get(pos['regime'], ACCENT_VIOLET)
        pnl_bar_width = min(abs(pos['pnl_pct']) * 2, 100)  # Scale for visualization
        
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            display: flex;
            align-items: center;
        ">
            <div style="flex: 1;">
                <h3 style="color: {TEXT_PRIMARY}; margin: 0 0 5px 0;">{pos['ticker']}</h3>
                <p style="color: {TEXT_SECONDARY}; margin: 0;">{regime_badge(pos['regime'], int(pos['confidence']*100))}</p>
                <p style="color: {TEXT_MUTED}; margin: 0; font-size: 12px;">{pos['days_in_regime']} days in regime</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <p style="color: {TEXT_SECONDARY}; margin: 0;">Entry: ${pos['entry_price']:.2f}</p>
                <p style="color: {TEXT_SECONDARY}; margin: 0;">Current: ${pos['current_price']:.2f}</p>
            </div>
            <div style="flex: 1;">
                <div style="
                    height: 20px;
                    background-color: {BG_CARD_HOVER};
                    border-radius: 10px;
                    overflow: hidden;
                    position: relative;
                ">
                    <div style="
                        height: 100%;
                        width: {pnl_bar_width}%;
                        background-color: {pnl_color(pos['pnl_pct'])};
                        position: absolute;
                        left: {0 if pos['pnl_pct'] >= 0 else 100 - pnl_bar_width}%;
                    "></div>
                </div>
                <p style="color: {pnl_color(pos['pnl_pct'])}; margin: 5px 0 0 0; text-align: center;">{pos['pnl_pct']:+.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Correlation Heatmap
    st.markdown(section_header("Correlation Risk"), unsafe_allow_html=True)
    
    corr_subset = corr_matrix.loc[:, [p['ticker'] for p in portfolio_data]].dropna()
    if not corr_subset.empty:
        fig_corr = px.imshow(
            corr_subset.corr(),
            color_continuous_scale=[BG_PRIMARY, ACCENT_CYAN, TEXT_PRIMARY],
            aspect="auto"
        )
        fig_corr.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Stress Test
    st.markdown(section_header("Stress Test"), unsafe_allow_html=True)
    
    stress_results = []
    for scenario, drawdowns in stress_drawdowns.items():
        portfolio_loss = 0
        portfolio_loss_pct = 0
        for pos in portfolio_data:
            dd = drawdowns.get(pos['ticker'], drawdowns.get('SPY', -0.25))  # Default to SPY
            loss = pos['value'] * abs(dd)
            portfolio_loss += loss
            portfolio_loss_pct += (loss / total_value) * 100
        
        severity_color = ACCENT_GREEN if portfolio_loss_pct < 10 else ACCENT_AMBER if portfolio_loss_pct < 20 else ACCENT_RED
        
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        ">
            <h4 style="color: {TEXT_PRIMARY}; margin: 0 0 10px 0;">{scenario}</h4>
            <div style="display: flex; align-items: center;">
                <div style="flex: 1;">
                    <p style="color: {severity_color}; font-size: 24px; margin: 0;">${portfolio_loss:,.0f}</p>
                    <p style="color: {TEXT_SECONDARY}; margin: 0;">{portfolio_loss_pct:.1f}% loss</p>
                </div>
                <div style="flex: 1;">
                    <div style="
                        height: 20px;
                        background-color: {BG_CARD_HOVER};
                        border-radius: 10px;
                        overflow: hidden;
                    ">
                        <div style="
                            height: 100%;
                            width: {min(portfolio_loss_pct * 5, 100)}%;
                            background-color: {severity_color};
                        "></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Regime Watchlist
    st.markdown(section_header("Watchlist"), unsafe_allow_html=True)
    
    for item in watchlist_data:
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
            display: flex;
            align-items: center;
        ">
            <div style="flex: 1;">
                <strong style="color: {TEXT_PRIMARY};">{item['ticker']}</strong>
                <p style="color: {TEXT_SECONDARY}; margin: 0;">${item['price']:.2f}</p>
            </div>
            <div style="flex: 1;">
                {regime_badge(item['regime'], int(item['confidence']*100))}
            </div>
            <div style="flex: 1; text-align: right;">
                <div style="
                    height: 8px;
                    background-color: {BG_CARD_HOVER};
                    border-radius: 4px;
                    overflow: hidden;
                ">
                    <div style="
                        height: 100%;
                        width: {item['confidence']*100}%;
                        background-color: {REGIME_COLORS.get(item['regime'], ACCENT_VIOLET)};
                    "></div>
                </div>
                <p style="color: {TEXT_MUTED}; margin: 0; font-size: 12px;">{item['days']} days</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Click 'Refresh Data' to load portfolio information.")