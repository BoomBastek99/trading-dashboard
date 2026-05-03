import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from design_system import *

# Apply theme
apply_theme()

# Sidebar
st.sidebar.header("Correlation Monitor")

default_pairs = [
    ("SPY", "QQQ"),
    ("GLD", "TLT"),
    ("SPY", "IWM"),
    ("BTC-USD", "ETH-USD"),
    ("SPY", "EEM")
]

pairs = []
for pair in default_pairs:
    if st.sidebar.checkbox(f"{pair[0]} / {pair[1]}", value=True):
        pairs.append(pair)

custom_pair = st.sidebar.text_input("Add Custom Pair (format: TICKER1,TICKER2)")
if custom_pair:
    try:
        t1, t2 = custom_pair.split(',')
        pairs.append((t1.strip(), t2.strip()))
    except:
        pass

window = st.sidebar.selectbox("Correlation Window", [20, 60, 120], index=1)
z_threshold = st.sidebar.slider("Z-Score Alert Threshold", -3.0, -1.0, -2.0, 0.1)

check = st.sidebar.button("Check Correlations")

@st.cache_data
def download_pair_data(pair, period="3y"):
    try:
        df = yf.download([pair[0], pair[1]], period=period)['Close']
        df = df.dropna()
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def calculate_correlations(df, window):
    returns = df.pct_change().dropna()
    rolling_corr = returns.iloc[:, 0].rolling(window).corr(returns.iloc[:, 1])
    rolling_corr = rolling_corr.dropna()
    
    # Historical mean and std
    hist_mean = rolling_corr.mean()
    hist_std = rolling_corr.std()
    
    # Z-score
    z_scores = (rolling_corr - hist_mean) / hist_std
    
    return rolling_corr, z_scores, hist_mean, hist_std

@st.cache_data
def detect_breaks(z_scores, threshold):
    breaks = z_scores < threshold
    break_dates = z_scores[breaks].index
    return break_dates, breaks

@st.cache_data
def historical_context(df, break_dates, window):
    returns = df.pct_change().dropna()
    context = []
    
    for break_date in break_dates:
        idx = df.index.get_loc(break_date)
        if idx + 20 < len(df):
            asset1_returns = []
            asset2_returns = []
            for days in [5, 10, 20]:
                ret1 = (df.iloc[idx + days, 0] / df.iloc[idx, 0] - 1) * 100
                ret2 = (df.iloc[idx + days, 1] / df.iloc[idx, 1] - 1) * 100
                asset1_returns.append(ret1)
                asset2_returns.append(ret2)
            
            context.append({
                'date': break_date,
                'asset1_5d': asset1_returns[0],
                'asset1_10d': asset1_returns[1],
                'asset1_20d': asset1_returns[2],
                'asset2_5d': asset2_returns[0],
                'asset2_10d': asset2_returns[1],
                'asset2_20d': asset2_returns[2]
            })
    
    return context

if check or 'correlation_data' not in st.session_state:
    with st.spinner("Analyzing correlations..."):
        correlation_data = {}
        
        for pair in pairs:
            df = download_pair_data(pair)
            if df.empty:
                continue
            
            rolling_corr, z_scores, hist_mean, hist_std = calculate_correlations(df, window)
            break_dates, breaks = detect_breaks(z_scores, z_threshold)
            
            current_corr = rolling_corr.iloc[-1]
            current_z = z_scores.iloc[-1]
            
            if current_z > -1.5:
                status = "Normal"
                color = BG_CARD
                glow = ""
            elif current_z > -2.0:
                status = "Notable"
                color = ACCENT_AMBER
                glow = "box-shadow: 0 0 10px rgba(255, 193, 7, 0.5);"
            elif current_z > -2.5:
                status = "Significant"
                color = "#FF8F00"
                glow = "box-shadow: 0 0 15px rgba(255, 143, 0, 0.7);"
            else:
                status = "Extreme"
                color = ACCENT_RED
                glow = "animation: pulse 1s infinite;"
            
            correlation_data[pair] = {
                'df': df,
                'rolling_corr': rolling_corr,
                'z_scores': z_scores,
                'hist_mean': hist_mean,
                'hist_std': hist_std,
                'current_corr': current_corr,
                'current_z': current_z,
                'status': status,
                'color': color,
                'glow': glow,
                'break_dates': break_dates,
                'context': historical_context(df, break_dates, window) if len(break_dates) > 0 else []
            }
        
        st.session_state['correlation_data'] = correlation_data

if 'correlation_data' in st.session_state:
    data = st.session_state['correlation_data']
    
    # Top status overview
    cols = st.columns(min(len(data), 5))
    selected_pair = st.selectbox("Select pair for detail", list(data.keys()))
    
    for i, (pair, info) in enumerate(data.items()):
        if i >= len(cols):
            break
        with cols[i]:
            st.markdown(f"""
            <div style="
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 15px;
                margin: 5px;
                text-align: center;
                {info['glow']}
            ">
                <h4 style="color: {TEXT_PRIMARY}; margin: 0 0 5px 0;">{pair[0]} / {pair[1]}</h4>
                <p style="color: {TEXT_SECONDARY}; font-size: 18px; margin: 0 0 5px 0;">{info['current_corr']:.2f}</p>
                <p style="color: {TEXT_MUTED}; margin: 0 0 5px 0;">Z: {info['current_z']:.2f}</p>
                <span style="
                    background-color: {info['color']};
                    color: {TEXT_PRIMARY};
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                ">{info['status']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Main chart
    if selected_pair in data:
        info = data[selected_pair]
        df = info['df']
        rolling_corr = info['rolling_corr']
        z_scores = info['z_scores']
        hist_mean = info['hist_mean']
        hist_std = info['hist_std']
        
        fig = go.Figure()
        
        # Correlation line
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode='lines',
            line=dict(color=TEXT_PRIMARY, width=2),
            name='Rolling Correlation'
        ))
        
        # Historical mean
        fig.add_hline(y=hist_mean, line_dash="dash", line_color=ACCENT_CYAN, annotation_text="Historical Mean")
        
        # Z-score threshold
        corr_threshold = hist_mean + z_threshold * hist_std
        fig.add_hline(y=corr_threshold, line_dash="dash", line_color=ACCENT_RED, annotation_text=f"Z < {z_threshold}")
        
        # Break periods
        break_mask = z_scores < z_threshold
        if break_mask.any():
            break_periods = []
            start = None
            for i, is_break in enumerate(break_mask):
                if is_break and start is None:
                    start = i
                elif not is_break and start is not None:
                    break_periods.append((start, i))
                    start = None
            if start is not None:
                break_periods.append((start, len(break_mask)))
            
            for start, end in break_periods:
                fig.add_shape(
                    type="rect",
                    x0=rolling_corr.index[start],
                    x1=rolling_corr.index[min(end, len(rolling_corr)-1)],
                    y0=rolling_corr.min(),
                    y1=rolling_corr.max(),
                    fillcolor="rgba(244, 67, 54, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
        
        # Current correlation marker
        fig.add_trace(go.Scatter(
            x=[rolling_corr.index[-1]],
            y=[rolling_corr.iloc[-1]],
            mode='markers',
            marker=dict(size=10, color=ACCENT_CYAN, symbol='circle'),
            name='Current'
        ))
        
        fig.update_layout(**get_plotly_layout(), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 20-day and 60-day comparison
    if selected_pair in data:
        info = data[selected_pair]
        df = info['df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            rolling_20, _, _, _ = calculate_correlations(df, 20)
            fig_20 = go.Figure()
            fig_20.add_trace(go.Scatter(x=rolling_20.index, y=rolling_20, mode='lines', line=dict(color=ACCENT_CYAN)))
            fig_20.update_layout(**get_plotly_layout(), title="20-Day Rolling Correlation", height=250)
            st.plotly_chart(fig_20, use_container_width=True)
        
        with col2:
            rolling_60, _, _, _ = calculate_correlations(df, 60)
            fig_60 = go.Figure()
            fig_60.add_trace(go.Scatter(x=rolling_60.index, y=rolling_60, mode='lines', line=dict(color=ACCENT_AMBER)))
            fig_60.update_layout(**get_plotly_layout(), title="60-Day Rolling Correlation", height=250)
            st.plotly_chart(fig_60, use_container_width=True)
    
    # Historical context
    if selected_pair in data and data[selected_pair]['context']:
        st.markdown(section_header("Historical Context"), unsafe_allow_html=True)
        
        context = data[selected_pair]['context']
        df_context = pd.DataFrame(context)
        df_context['Date'] = df_context['date'].dt.strftime('%Y-%m-%d')
        
        # Table
        st.dataframe(style_dataframe(df_context[['Date', 'asset1_5d', 'asset1_10d', 'asset1_20d', 'asset2_5d', 'asset2_10d', 'asset2_20d']]))
        
        # Average outcomes
        avg_5d_1 = df_context['asset1_5d'].mean()
        avg_10d_1 = df_context['asset1_10d'].mean()
        avg_20d_1 = df_context['asset1_20d'].mean()
        avg_5d_2 = df_context['asset2_5d'].mean()
        avg_10d_2 = df_context['asset2_10d'].mean()
        avg_20d_2 = df_context['asset2_20d'].mean()
        
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        ">
            <h4 style="color: {TEXT_PRIMARY}; margin: 0 0 10px 0;">Average Outcomes After Correlation Breaks</h4>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p style="color: {TEXT_SECONDARY}; margin: 0;"><strong>{selected_pair[0]}:</strong></p>
                    <p style="color: {pnl_color(avg_5d_1)}; margin: 0;">5 days: {avg_5d_1:.1f}%</p>
                    <p style="color: {pnl_color(avg_10d_1)}; margin: 0;">10 days: {avg_10d_1:.1f}%</p>
                    <p style="color: {pnl_color(avg_20d_1)}; margin: 0;">20 days: {avg_20d_1:.1f}%</p>
                </div>
                <div>
                    <p style="color: {TEXT_SECONDARY}; margin: 0;"><strong>{selected_pair[1]}:</strong></p>
                    <p style="color: {pnl_color(avg_5d_2)}; margin: 0;">5 days: {avg_5d_2:.1f}%</p>
                    <p style="color: {pnl_color(avg_10d_2)}; margin: 0;">10 days: {avg_10d_2:.1f}%</p>
                    <p style="color: {pnl_color(avg_20d_2)}; margin: 0;">20 days: {avg_20d_2:.1f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="color: {TEXT_MUTED}; font-style: italic; margin: 10px 0;">
        Past correlation breaks do not guarantee similar outcomes. Correlations can drift without signaling directional moves.
        </p>
        """, unsafe_allow_html=True)
    
    # Alert log (placeholder)
    st.markdown(section_header("Recent Alerts"), unsafe_allow_html=True)
    st.info("Alert logging would be implemented here with JSON file storage.")
else:
    st.info("Configure pairs and click 'Check Correlations' to analyze.")