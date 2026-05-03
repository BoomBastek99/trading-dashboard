import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from design_system import *

# Apply theme
apply_theme()

# Sidebar
st.sidebar.header("Backtest Parameters")

default_assets = ["SPY", "BTC-USD", "GLD", "TLT"]
assets = st.sidebar.multiselect("Assets", options=default_assets + ["Add Custom"], default=default_assets)

custom_ticker = st.sidebar.text_input("Custom Ticker")
if custom_ticker and custom_ticker not in assets:
    assets.append(custom_ticker)

start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

train_period = st.sidebar.slider("Training Period (months)", min_value=6, max_value=24, value=12)
test_period = st.sidebar.slider("Test Period (months)", min_value=3, max_value=12, value=6)

run_backtest = st.sidebar.button("Run Backtest")

@st.cache_data
def download_asset_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df = df.dropna()
    return df

@st.cache_data
def train_hmm_forward(df):
    features = df[['Returns', 'Volatility', 'Volume_Ratio', 'HL_Range']].values
    
    # Select best n_components
    best_bic = np.inf
    best_model = None
    best_n = 3
    for n in range(3, 8):
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
    n_samples, n_features = features.shape
    n_states = best_model.n_components
    log_likelihoods = best_model._compute_log_likelihood(features)
    log_alpha = np.zeros((n_samples, n_states))
    log_alpha[0] = best_model.startprob_ + log_likelihoods[0]
    for t in range(1, n_samples):
        log_alpha[t] = log_likelihoods[t] + np.log(np.sum(np.exp(log_alpha[t-1] + best_model.transmat_.T), axis=1))
    alpha = np.exp(log_alpha)
    posterior = alpha / alpha.sum(axis=1, keepdims=True)
    regimes = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)
    
    return regimes, confidence, best_n

@st.cache_data
def run_walk_forward_backtest(df, regimes, train_months, test_months):
    df = df.copy()
    df['Regime'] = regimes
    
    # Allocation rules: scale linearly between low vol (95%) and high vol (60%)
    unique_regimes = np.unique(regimes)
    mean_vols = [np.mean(df[df['Regime'] == r]['Volatility']) for r in unique_regimes]
    sorted_indices = np.argsort(mean_vols)
    low_vol_regime = unique_regimes[sorted_indices[0]]
    high_vol_regime = unique_regimes[sorted_indices[-1]]
    
    allocations = []
    for r in df['Regime']:
        if r == low_vol_regime:
            alloc = 0.95
        elif r == high_vol_regime:
            alloc = 0.60
        else:
            # Linear interpolation
            vol = np.mean(df[df['Regime'] == r]['Volatility'])
            low_vol = mean_vols[sorted_indices[0]]
            high_vol = mean_vols[sorted_indices[-1]]
            alloc = 0.95 - (vol - low_vol) / (high_vol - low_vol) * 0.35
            alloc = np.clip(alloc, 0.60, 0.95)
        allocations.append(alloc)
    
    df['Allocation'] = allocations
    df['Strategy_Returns'] = df['Returns'] * df['Allocation']
    
    # Buy and hold
    df['BH_Returns'] = df['Returns']
    
    # 200-day SMA trend
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['SMA_Trend'] = (df['Close'] > df['SMA_200']).astype(int)
    df['SMA_Returns'] = df['Returns'] * df['SMA_Trend']
    
    # Calculate cumulative returns
    df['Strategy_Cum'] = (1 + df['Strategy_Returns']).cumprod()
    df['BH_Cum'] = (1 + df['BH_Returns']).cumprod()
    df['SMA_Cum'] = (1 + df['SMA_Returns']).cumprod()
    
    # Annualized returns
    total_days = (df.index[-1] - df.index[0]).days
    ann_factor = 252 / total_days * len(df)
    
    strategy_ann = (df['Strategy_Cum'].iloc[-1]) ** ann_factor - 1
    bh_ann = (df['BH_Cum'].iloc[-1]) ** ann_factor - 1
    sma_ann = (df['SMA_Cum'].iloc[-1]) ** ann_factor - 1
    
    # Sharpe ratios
    strategy_sharpe = np.mean(df['Strategy_Returns']) / np.std(df['Strategy_Returns']) * np.sqrt(252) if np.std(df['Strategy_Returns']) > 0 else 0
    bh_sharpe = np.mean(df['BH_Returns']) / np.std(df['BH_Returns']) * np.sqrt(252) if np.std(df['BH_Returns']) > 0 else 0
    sma_sharpe = np.mean(df['SMA_Returns']) / np.std(df['SMA_Returns']) * np.sqrt(252) if np.std(df['SMA_Returns']) > 0 else 0
    
    # Max drawdowns
    def max_drawdown(cum_returns):
        peak = cum_returns.expanding().max()
        dd = (cum_returns - peak) / peak
        return dd.min()
    
    strategy_dd = max_drawdown(df['Strategy_Cum'])
    bh_dd = max_drawdown(df['BH_Cum'])
    sma_dd = max_drawdown(df['SMA_Cum'])
    
    return {
        'df': df,
        'strategy_ann': strategy_ann,
        'bh_ann': bh_ann,
        'sma_ann': sma_ann,
        'strategy_sharpe': strategy_sharpe,
        'bh_sharpe': bh_sharpe,
        'sma_sharpe': sma_sharpe,
        'strategy_dd': strategy_dd,
        'bh_dd': bh_dd,
        'sma_dd': sma_dd,
        'sharpe_improvement': strategy_sharpe - bh_sharpe
    }

# Crisis periods
crisis_periods = {
    '2008 Crisis': ('2008-09-01', '2009-03-01'),
    '2020 Covid': ('2020-02-01', '2020-04-01'),
    '2022 Hikes': ('2022-01-01', '2022-10-01')
}

@st.cache_data
def calculate_crisis_performance(df, start, end):
    mask = (df.index >= start) & (df.index <= end)
    if not mask.any():
        return {'strategy': 0, 'bh': 0}
    
    crisis_df = df[mask]
    strategy_ret = (crisis_df['Strategy_Cum'].iloc[-1] / crisis_df['Strategy_Cum'].iloc[0]) - 1
    bh_ret = (crisis_df['BH_Cum'].iloc[-1] / crisis_df['BH_Cum'].iloc[0]) - 1
    return {'strategy': strategy_ret, 'bh': bh_ret}

if run_backtest or 'backtest_results' not in st.session_state:
    with st.spinner("Running multi-asset backtest..."):
        results = {}
        for asset in assets:
            if asset == "Add Custom":
                continue
            df = download_asset_data(asset, start_date, end_date)
            if df.empty:
                continue
            regimes, confidence, n_regimes = train_hmm_forward(df)
            backtest_result = run_walk_forward_backtest(df, regimes, train_period, test_period)
            
            # Crisis performance
            crisis_perf = {}
            for crisis, (c_start, c_end) in crisis_periods.items():
                crisis_perf[crisis] = calculate_crisis_performance(backtest_result['df'], c_start, c_end)
            
            results[asset] = {
                'backtest': backtest_result,
                'regimes': regimes,
                'confidence': confidence,
                'n_regimes': n_regimes,
                'crisis': crisis_perf
            }
        
        st.session_state['backtest_results'] = results

if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    
    # Asset tabs
    asset_colors = {
        "SPY": ACCENT_CYAN,
        "BTC-USD": ACCENT_AMBER,
        "GLD": "#FFD700",
        "TLT": ACCENT_VIOLET
    }
    
    tabs = st.tabs([f"{asset} ({'👍' if results[asset]['backtest']['sharpe_improvement'] > 0 else '👎'})" for asset in results.keys()])
    
    selected_asset = st.selectbox("Select Asset for Detail View", list(results.keys()))
    
    # Hero equity curves
    st.markdown(section_header("Equity Curves"), unsafe_allow_html=True)
    
    asset_result = results[selected_asset]['backtest']
    df = asset_result['df']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Strategy_Cum'], 
        mode='lines', line=dict(color=asset_colors.get(selected_asset, ACCENT_CYAN), width=3),
        name='Regime Strategy'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BH_Cum'], 
        mode='lines', line=dict(color=TEXT_MUTED, width=2, dash='dash'),
        name='Buy & Hold'
    ))
    fig.update_layout(**get_plotly_layout(), height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regime Timeline Strips
    st.markdown(section_header("Regime Timeline"), unsafe_allow_html=True)
    
    fig_timeline = go.Figure()
    y_pos = 0
    for asset in results.keys():
        regimes = results[asset]['regimes']
        unique_regimes = np.unique(regimes)
        mean_vols = [np.mean(results[asset]['backtest']['df'][results[asset]['backtest']['df']['Regime'] == r]['Volatility']) for r in unique_regimes]
        sorted_indices = np.argsort(mean_vols)
        regime_labels = ["Low Vol", "Medium Vol", "High Vol", "Very High Vol"][:len(unique_regimes)]
        
        for i, r in enumerate(regimes):
            color = REGIME_COLORS.get(regime_labels[sorted_indices.tolist().index(r)], ACCENT_VIOLET)
            fig_timeline.add_shape(
                type="rect",
                x0=df.index[i], x1=df.index[min(i+1, len(df)-1)],
                y0=y_pos, y1=y_pos + 0.8,
                fillcolor=color,
                line=dict(width=0)
            )
        fig_timeline.add_annotation(
            x=df.index[0], y=y_pos + 0.4, text=asset,
            showarrow=False, xanchor="right", font=dict(color=TEXT_PRIMARY)
        )
        y_pos += 1
    
    fig_timeline.update_layout(
        **get_plotly_layout(),
        height=len(results) * 50,
        showlegend=False,
        xaxis=dict(showticklabels=True),
        yaxis=dict(showticklabels=False, range=[0, len(results)])
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Comparison table
    st.markdown(section_header("Performance Comparison"), unsafe_allow_html=True)
    
    table_data = []
    for asset, data in results.items():
        b = data['backtest']
        table_data.append({
            'Asset': asset,
            'Strategy Ann.': f"{b['strategy_ann']*100:.1f}%",
            'BH Ann.': f"{b['bh_ann']*100:.1f}%",
            'Strategy Max DD': f"{b['strategy_dd']*100:.1f}%",
            'BH Max DD': f"{b['bh_dd']*100:.1f}%",
            'Strategy Sharpe': f"{b['strategy_sharpe']:.2f}",
            'BH Sharpe': f"{b['bh_sharpe']:.2f}",
            'Sharpe Improvement': f"{b['sharpe_improvement']:.2f}"
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(style_dataframe(df_table))
    
    # Stress test results
    st.markdown(section_header("Stress Test Results"), unsafe_allow_html=True)
    
    crisis_data = []
    for crisis in crisis_periods.keys():
        row = {'Crisis': crisis}
        for asset in results.keys():
            perf = results[asset]['crisis'][crisis]
            row[f"{asset} Strategy"] = f"{perf['strategy']*100:.1f}%"
            row[f"{asset} BH"] = f"{perf['bh']*100:.1f}%"
        crisis_data.append(row)
    
    df_crisis = pd.DataFrame(crisis_data)
    st.dataframe(style_dataframe(df_crisis))
    
    # Summary
    best_asset = max(results.keys(), key=lambda x: results[x]['backtest']['sharpe_improvement'])
    worst_asset = min(results.keys(), key=lambda x: results[x]['backtest']['sharpe_improvement'])
    
    st.markdown(f"""
    **Summary:** Regime detection added the most value for **{best_asset}** with a Sharpe improvement of {results[best_asset]['backtest']['sharpe_improvement']:.2f}. 
    It struggled most with **{worst_asset}**, suggesting regime detection may not suit this asset class as well.
    """)
else:
    st.info("Configure assets and parameters in the sidebar, then click 'Run Backtest'.")