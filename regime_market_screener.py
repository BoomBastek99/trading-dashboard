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

# Sidebar
st.sidebar.header("Market Screener")

default_universe = {
    'Large Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH'],
    'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'GLD', 'TLT', 'HYG'],
    'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD']
}

universe = {}
for category, tickers in default_universe.items():
    universe[category] = st.sidebar.multiselect(f"{category}", options=tickers, default=tickers[:5])

custom_tickers = st.sidebar.text_input("Add Custom Tickers (comma-separated)")
if custom_tickers:
    custom_list = [t.strip() for t in custom_tickers.split(',') if t.strip()]
    universe['Custom'] = custom_list

# Filters
regime_filter = st.sidebar.multiselect("Regime Filter", ["Low Vol", "Medium Vol", "High Vol", "Uncertain"], default=[])
confidence_min = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5)
sma_filter = st.sidebar.selectbox("SMA Filter", ["All", "Above 50 SMA", "Below 50 SMA"], index=0)
volume_filter = st.sidebar.selectbox("Volume Filter", ["All", "Increasing Volume", "Decreasing Volume"], index=0)
sort_by = st.sidebar.selectbox("Sort By", ["Confidence", "Days in Regime", "Ticker"], index=0)

scan = st.sidebar.button("Scan Market")

@st.cache_data
def scan_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y")
        if df.empty or len(df) < 100:
            return None
        
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df = df.dropna()
        
        if len(df) < 100:
            return None
        
        features = df[['Returns', 'Volatility', 'Volume_Ratio', 'HL_Range']].values
        
        # Fit HMM
        best_bic = np.inf
        best_model = None
        best_n = 3
        for n in range(3, 6):
            model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=1000, random_state=42)
            model.fit(features)
            log_likelihood = model.score(features)
            n_params = n * (n - 1) + n * features.shape[1] * (features.shape[1] + 1) // 2
            bic = -2 * log_likelihood + n_params * np.log(len(features))
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n
        
        # Forward filtering
        log_likelihoods = best_model._compute_log_likelihood(features)
        log_alpha = np.zeros((len(features), best_n))
        log_alpha[0] = best_model.startprob_ + log_likelihoods[0]
        for t in range(1, len(features)):
            log_alpha[t] = log_likelihoods[t] + np.log(np.sum(np.exp(log_alpha[t-1] + best_model.transmat_.T), axis=1))
        alpha = np.exp(log_alpha)
        posterior = alpha / alpha.sum(axis=1, keepdims=True)
        regimes = np.argmax(posterior, axis=1)
        confidence = posterior[-1, regimes[-1]]
        
        # Label regimes
        unique_regimes = np.unique(regimes)
        mean_vols = [np.mean(df['Volatility'].iloc[np.where(regimes == r)[0]]) for r in unique_regimes]
        sorted_indices = np.argsort(mean_vols)
        regime_labels = ["Low Vol", "Medium Vol", "High Vol", "Very High Vol", "Extreme Vol"][:len(unique_regimes)]
        regime_map = {unique_regimes[i]: regime_labels[i] for i in sorted_indices}
        current_regime = regime_map[regimes[-1]]
        
        days_in_regime = np.sum(regimes == regimes[-1])
        
        # Technicals
        current_price = df['Close'].iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_position = "Above" if current_price > sma_50 else "Below"
        
        volume_trend = "Increasing" if df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] else "Decreasing"
        
        return {
            'ticker': ticker,
            'price': current_price,
            'regime': current_regime,
            'confidence': confidence,
            'days_in_regime': days_in_regime,
            'sma_position': sma_position,
            'volume_trend': volume_trend,
            'df': df,
            'regimes': regimes,
            'regime_map': regime_map
        }
    except Exception as e:
        return None

if scan or 'scan_results' not in st.session_state:
    with st.spinner("Scanning market..."):
        all_tickers = []
        for category, tickers in universe.items():
            all_tickers.extend(tickers)
        
        scan_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(all_tickers):
            status_text.text(f"Scanning {ticker}...")
            result = scan_ticker(ticker)
            if result:
                scan_results.append(result)
            progress_bar.progress((i + 1) / len(all_tickers))
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['scan_results'] = scan_results

if 'scan_results' in st.session_state:
    results = st.session_state['scan_results']
    
    # Apply filters
    filtered_results = results.copy()
    
    if regime_filter:
        filtered_results = [r for r in filtered_results if r['regime'] in regime_filter]
    
    filtered_results = [r for r in filtered_results if r['confidence'] >= confidence_min]
    
    if sma_filter == "Above 50 SMA":
        filtered_results = [r for r in filtered_results if r['sma_position'] == "Above"]
    elif sma_filter == "Below 50 SMA":
        filtered_results = [r for r in filtered_results if r['sma_position'] == "Below"]
    
    if volume_filter == "Increasing Volume":
        filtered_results = [r for r in filtered_results if r['volume_trend'] == "Increasing"]
    elif volume_filter == "Decreasing Volume":
        filtered_results = [r for r in filtered_results if r['volume_trend'] == "Decreasing"]
    
    # Sort
    if sort_by == "Confidence":
        filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == "Days in Regime":
        filtered_results.sort(key=lambda x: x['days_in_regime'], reverse=True)
    else:
        filtered_results.sort(key=lambda x: x['ticker'])
    
    # Top summary
    regime_counts = {}
    for r in results:
        regime_counts[r['regime']] = regime_counts.get(r['regime'], 0) + 1
    
    strongest_regime = max(regime_counts, key=regime_counts.get)
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card(f"{regime_counts.get('Low Vol', 0)}", "Low Vol Tickers", REGIME_COLORS["Low Vol"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(f"{regime_counts.get('High Vol', 0)}", "High Vol Tickers", REGIME_COLORS["High Vol"]), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card(f"{avg_confidence:.1f}%", "Avg Confidence", ACCENT_CYAN), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card(strongest_regime, "Strongest Regime", REGIME_COLORS.get(strongest_regime, ACCENT_VIOLET)), unsafe_allow_html=True)
    
    # Screener table
    st.markdown(section_header("Market Screener Results"), unsafe_allow_html=True)
    
    table_data = []
    for r in filtered_results:
        table_data.append({
            'Ticker': r['ticker'],
            'Price': f"${r['price']:.2f}",
            'Regime': r['regime'],
            'Confidence': f"{r['confidence']:.1%}",
            'Days in Regime': r['days_in_regime'],
            'vs 50 SMA': f"{'↑' if r['sma_position'] == 'Above' else '↓'} {r['sma_position']}",
            'Volume Trend': f"{'↑' if r['volume_trend'] == 'Increasing' else '↓'} {r['volume_trend']}"
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(style_dataframe(df_table))
    
    # Quick chart for selected ticker
    selected_ticker = st.selectbox("Select ticker for chart", [r['ticker'] for r in filtered_results])
    selected_result = next((r for r in filtered_results if r['ticker'] == selected_ticker), None)
    
    if selected_result:
        df = selected_result['df']
        regimes = selected_result['regimes']
        regime_map = selected_result['regime_map']
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Regime bands
        unique_regimes = list(regime_map.keys())
        for reg in unique_regimes:
            color = REGIME_COLORS.get(regime_map[reg], ACCENT_VIOLET) + "1f"
            reg_mask = regimes == reg
            if reg_mask.any():
                starts = np.where(np.diff([False] + reg_mask.tolist()))[0]
                ends = np.where(np.diff(reg_mask.tolist() + [False]))[0]
                for start, end in zip(starts, ends):
                    fig.add_shape(
                        type="rect",
                        x0=df.index[start],
                        x1=df.index[end-1],
                        y0=df['Low'].min(),
                        y1=df['High'].max(),
                        fillcolor=color,
                        line=dict(width=0),
                        layer="below"
                    )
        
        fig.update_layout(**get_plotly_layout(), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Regime distribution
    st.markdown(section_header("Regime Distribution"), unsafe_allow_html=True)
    
    regime_dist = pd.Series([r['regime'] for r in results]).value_counts()
    
    fig_dist = go.Figure()
    colors = [REGIME_COLORS.get(reg, ACCENT_VIOLET) for reg in regime_dist.index]
    fig_dist.add_trace(go.Bar(
        x=regime_dist.index,
        y=regime_dist.values,
        marker_color=colors
    ))
    fig_dist.update_layout(**get_plotly_layout(), height=300)
    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("Configure universe and filters, then click 'Scan Market' to analyze tickers.")