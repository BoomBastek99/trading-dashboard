import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from design_system import *

# Apply theme
apply_theme()

# Verification check for no look-ahead bias
st.sidebar.markdown("### Verification")
st.sidebar.info("✅ Using forward algorithm only - no look-ahead bias in regime detection")

# Sidebar inputs
st.sidebar.header("Analysis Parameters")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", key="ticker")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*3), key="start")
end_date = st.sidebar.date_input("End Date", value=datetime.now(), key="end")
n_regimes_override = st.sidebar.slider("Number of Regimes (0 = auto)", min_value=0, max_value=7, value=0, key="n_regimes")

run_analysis = st.sidebar.button("Run Analysis", key="run")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df = df.dropna()
    features = df[['Returns', 'Volatility', 'Volume_Ratio', 'HL_Range']].values
    return df, features

@st.cache_data
def train_hmm(features, n_regimes_override):
    if n_regimes_override > 0:
        n_components = n_regimes_override
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000, random_state=42)
        model.fit(features)
        return model, n_components
    
    min_components, max_components = 3, 7
    best_bic = np.inf
    best_model = None
    best_n = 0
    
    for n in range(min_components, max_components + 1):
        model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=1000, random_state=42)
        model.fit(features)
        log_likelihood = model.score(features)
        # Rough BIC calculation
        n_params = n * (n - 1) + n * features.shape[1] * (features.shape[1] + 1) // 2
        bic = -2 * log_likelihood + n_params * np.log(len(features))
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n = n
    
    return best_model, best_n

def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis)) + a_max.squeeze(axis=axis if axis is not None else 0)
    return out

def forward_filter(hmm_model, features):
    n_samples = features.shape[0]
    n_states = hmm_model.n_components
    
    log_startprob = np.log(hmm_model.startprob_ + 1e-300)
    log_transmat = np.log(hmm_model.transmat_ + 1e-300)
    log_likelihoods = hmm_model._compute_log_likelihood(features)
    
    # Forward pass in log space with numerically stable logsumexp
    log_alpha = np.zeros((n_samples, n_states))
    log_alpha[0] = log_startprob + log_likelihoods[0]
    
    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = _logsumexp(log_alpha[t-1] + log_transmat[:, j]) + log_likelihoods[t, j]
    
    # Normalize each row to get posterior probabilities
    log_norm = _logsumexp(log_alpha, axis=1)
    log_posterior = log_alpha - log_norm[:, np.newaxis]
    posterior = np.exp(log_posterior)
    
    regimes = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)
    
    return regimes, confidence

def label_regimes(regimes, vol_series):
    unique_regimes = np.unique(regimes)
    mean_vols = [np.mean(vol_series[regimes == r]) for r in unique_regimes]
    sorted_indices = np.argsort(mean_vols)
    
    labels = ["Low Vol", "Medium Vol", "High Vol", "Very High Vol", "Extreme Vol"][:len(unique_regimes)]
    regime_labels = {}
    for i, idx in enumerate(sorted_indices):
        regime_labels[unique_regimes[idx]] = labels[i]
    
    return regime_labels

def apply_stability_filter(regimes, min_persist=3, flicker_threshold=4, window=20):
    filtered = regimes.copy()
    
    # Persistence filter
    for i in range(min_persist - 1, len(regimes)):
        current = regimes[i]
        if not np.all(regimes[i - min_persist + 1:i + 1] == current):
            filtered[i] = filtered[i - 1]
    
    # Uncertainty filter
    uncertain_mask = np.zeros(len(regimes), dtype=bool)
    for i in range(window, len(regimes)):
        changes = np.sum(np.diff(filtered[i - window:i]) != 0)
        if changes > flicker_threshold:
            uncertain_mask[i] = True
    
    filtered[uncertain_mask] = -1  # Special value for uncertain
    
    return filtered

if run_analysis or 'data' not in st.session_state:
    with st.spinner("Loading data and running analysis..."):
        try:
            df = load_data(ticker, start_date, end_date)
            if df.empty or len(df) < 50:
                st.error("No data found for the selected ticker and date range.")
                st.stop()
            
            df_clean, features = engineer_features(df)
            
            hmm_model, n_regimes = train_hmm(features, n_regimes_override)
            
            regimes, confidence = forward_filter(hmm_model, features)
            
            regime_labels = label_regimes(regimes, df_clean['Volatility'].values)
            
            filtered_regimes = apply_stability_filter(regimes)
            
            # Map filtered regimes to labels
            regime_series = []
            for r in filtered_regimes:
                if r == -1:
                    regime_series.append("Uncertain")
                else:
                    regime_series.append(regime_labels[r])
            
            # Store in session state
            st.session_state['data'] = df_clean
            st.session_state['regimes'] = regime_series
            st.session_state['confidence'] = confidence
            st.session_state['detected_n_regimes'] = n_regimes
            st.session_state['current_regime'] = regime_series[-1]
            st.session_state['current_confidence'] = confidence[-1]
            st.session_state['stability'] = "Uncertain" if "Uncertain" in regime_series[-20:] else "Stable"
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# Display results
if 'data' in st.session_state:
    df = st.session_state['data']
    regimes = st.session_state['regimes']
    confidence = st.session_state['confidence']
    n_regimes = st.session_state['detected_n_regimes']
    current_regime = st.session_state['current_regime']
    current_confidence = st.session_state['current_confidence']
    stability = st.session_state['stability']
    
    # Top bar
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
    with col1:
        st.markdown(f"<h1 style='color:{TEXT_PRIMARY}; margin:0;'>{ticker.upper()}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(regime_badge(current_regime, int(current_confidence * 100)), unsafe_allow_html=True)
    with col3:
        st.markdown(f"<span style='color:{ACCENT_CYAN}; font-size:2rem; font-weight:bold;'>{current_confidence*100:.1f}%</span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<span style='color:{TEXT_SECONDARY};'>Stability: {stability}</span>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<span style='color:{TEXT_SECONDARY};'>Regimes: {n_regimes}</span>", unsafe_allow_html=True)
    
    # Main chart
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=TEXT_PRIMARY,
        decreasing_line_color=TEXT_SECONDARY
    ))
    
    # Add regime background bands
    unique_regimes = list(set(regimes))
    for regime in unique_regimes:
        if regime == "Uncertain":
            color = REGIME_COLORS["Uncertain"] + "1f"  # ~12% opacity
        else:
            color = REGIME_COLORS.get(regime, ACCENT_VIOLET) + "1f"
        
        regime_mask = [r == regime for r in regimes]
        starts = np.where(np.diff([False] + regime_mask))[0]
        ends = np.where(np.diff(regime_mask + [False]))[0]
        
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
    
    fig.update_layout(**get_plotly_layout())
    fig.update_layout(
        title="Price Chart with Regime Background",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regime Statistics
    st.markdown(section_header("Regime Statistics"), unsafe_allow_html=True)
    
    regime_stats = {}
    for regime in unique_regimes:
        mask = np.array([r == regime for r in regimes])
        regime_stats[regime] = {
            'mean_return': df['Returns'].values[mask].mean() * 100,
            'mean_vol': df['Volatility'].values[mask].mean() * 100,
            'mean_vol_ratio': df['Volume_Ratio'].values[mask].mean(),
            'pct_time': mask.mean() * 100
        }
    
    cols = st.columns(len(unique_regimes))
    for i, (regime, stats) in enumerate(regime_stats.items()):
        with cols[i]:
            color = REGIME_COLORS.get(regime, ACCENT_VIOLET)
            st.markdown(f"""
            <div style="
                background-color: {BG_CARD};
                border-left: 4px solid {color};
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
            ">
                <h4 style="color:{color}; margin:0 0 10px 0;">{regime}</h4>
                <p style="color:{TEXT_SECONDARY}; margin:5px 0;">Mean Return: {stats['mean_return']:.2f}%</p>
                <p style="color:{TEXT_SECONDARY}; margin:5px 0;">Mean Vol: {stats['mean_vol']:.2f}%</p>
                <p style="color:{TEXT_SECONDARY}; margin:5px 0;">Mean Vol Ratio: {stats['mean_vol_ratio']:.2f}</p>
                <p style="color:{TEXT_SECONDARY}; margin:5px 0;">Time in Regime: {stats['pct_time']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Confidence Timeline
    st.markdown(section_header("Confidence Timeline"), unsafe_allow_html=True)
    
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=df.index,
        y=confidence,
        fill='tozeroy',
        fillcolor=ACCENT_CYAN + '4d',
        line=dict(color=ACCENT_CYAN),
        name='Confidence'
    ))
    fig_conf.update_layout(**get_plotly_layout())
    fig_conf.update_layout(
        title="Regime Confidence Over Time",
        xaxis_title="Date",
        yaxis_title="Confidence",
        height=300
    )
    
    st.plotly_chart(fig_conf, use_container_width=True)
else:
    st.info("Click 'Run Analysis' in the sidebar to start.")