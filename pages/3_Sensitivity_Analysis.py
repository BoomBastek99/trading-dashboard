import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from design_system import *

# Apply theme
apply_theme()

# Sidebar
st.sidebar.header("Analysis Parameters")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", key="ticker")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*5), key="start")
end_date = st.sidebar.date_input("End Date", value=datetime.now(), key="end")

# Parameter ranges
fast_ma_base = 10
fast_ma_range = st.sidebar.slider("Fast MA Range", min_value=5, max_value=30, value=(5, 30), step=1)
slow_ma_base = 50
slow_ma_range = st.sidebar.slider("Slow MA Range", min_value=20, max_value=100, value=(20, 100), step=5)
stop_loss_base = 2.0
stop_loss_range = st.sidebar.slider("Stop Loss Range (%)", min_value=0.5, max_value=5.0, value=(0.5, 5.0), step=0.5)
take_profit_base = 4.0
take_profit_range = st.sidebar.slider("Take Profit Range (%)", min_value=1.0, max_value=10.0, value=(1.0, 10.0), step=0.5)

run_analysis = st.sidebar.button("Run Analysis")

@st.cache_data
def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

@st.cache_data
def backtest_ma_crossover(df, fast_ma, slow_ma, stop_loss_pct, take_profit_pct):
    df = df.copy()
    df['Fast_MA'] = df['Close'].rolling(fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(slow_ma).mean()
    
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(max(fast_ma, slow_ma), len(df)):
        if position == 0:
            if df['Fast_MA'].iloc[i] > df['Slow_MA'].iloc[i] and df['Fast_MA'].iloc[i-1] <= df['Slow_MA'].iloc[i-1]:
                position = 1
                entry_price = df['Close'].iloc[i]
        elif position == 1:
            pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * 100
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or (df['Fast_MA'].iloc[i] < df['Slow_MA'].iloc[i] and df['Fast_MA'].iloc[i-1] >= df['Slow_MA'].iloc[i-1]):
                position = 0
                exit_price = df['Close'].iloc[i]
                trades.append((entry_price, exit_price))
    
    if trades:
        returns = [(exit - entry) / entry for entry, exit in trades]
        total_return = (1 + sum(returns) / len(returns)) ** (len(returns) / len(df)) - 1  # rough annualized
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_dd = 0
        peak = 1
        cum_return = 1
        for ret in returns:
            cum_return *= (1 + ret)
            if cum_return > peak:
                peak = cum_return
            dd = (peak - cum_return) / peak
            if dd > max_dd:
                max_dd = dd
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
    else:
        total_return = sharpe = max_dd = win_rate = 0
    
    return total_return, sharpe, max_dd, win_rate

@st.cache_data
def run_sensitivity_analysis(df, param_name, param_range, base_params):
    results = []
    for val in param_range:
        params = base_params.copy()
        params[param_name] = val
        ret, sharpe, dd, win = backtest_ma_crossover(df, **params)
        results.append({
            'value': val,
            'total_return': ret,
            'sharpe': sharpe,
            'max_dd': dd,
            'win_rate': win
        })
    return pd.DataFrame(results)

@st.cache_data
def calculate_robustness(results_df, base_value):
    metrics = ['total_return', 'sharpe', 'max_dd', 'win_rate']
    scores = []
    for metric in metrics:
        values = results_df[metric].values
        base_idx = np.argmin(np.abs(results_df['value'].values - base_value))
        base_perf = values[base_idx]
        if metric == 'max_dd':
            deviations = np.abs(values - base_perf) / base_perf if base_perf != 0 else 0
        else:
            deviations = np.abs(values - base_perf) / abs(base_perf) if base_perf != 0 else 0
        cv = np.std(deviations) / np.mean(deviations) if np.mean(deviations) > 0 else 0
        score = max(0, 100 * (1 - cv))
        scores.append(score)
    return np.mean(scores)

if run_analysis or 'analysis' not in st.session_state:
    with st.spinner("Downloading data and running sensitivity analysis..."):
        df = download_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No data found for the selected ticker and date range.")
            st.stop()
        
        base_params = {
            'fast_ma': fast_ma_base,
            'slow_ma': slow_ma_base,
            'stop_loss_pct': stop_loss_base,
            'take_profit_pct': take_profit_base
        }
        
        # Run base backtest
        base_ret, base_sharpe, base_dd, base_win = backtest_ma_crossover(df, **base_params)
        
        # Sensitivity analysis
        fast_ma_results = run_sensitivity_analysis(df, 'fast_ma', range(fast_ma_range[0], fast_ma_range[1]+1), base_params)
        slow_ma_results = run_sensitivity_analysis(df, 'slow_ma', range(slow_ma_range[0], slow_ma_range[1]+1, 5), base_params)
        stop_loss_results = run_sensitivity_analysis(df, 'stop_loss_pct', np.arange(stop_loss_range[0], stop_loss_range[1]+0.5, 0.5), base_params)
        take_profit_results = run_sensitivity_analysis(df, 'take_profit_pct', np.arange(take_profit_range[0], take_profit_range[1]+0.5, 0.5), base_params)
        
        # Robustness scores
        fast_ma_robust = calculate_robustness(fast_ma_results, fast_ma_base)
        slow_ma_robust = calculate_robustness(slow_ma_results, slow_ma_base)
        stop_loss_robust = calculate_robustness(stop_loss_results, stop_loss_base)
        take_profit_robust = calculate_robustness(take_profit_results, take_profit_base)
        overall_robust = np.mean([fast_ma_robust, slow_ma_robust, stop_loss_robust, take_profit_robust])
        
        st.session_state['analysis'] = {
            'df': df,
            'base': {'ret': base_ret, 'sharpe': base_sharpe, 'dd': base_dd, 'win': base_win},
            'fast_ma': fast_ma_results,
            'slow_ma': slow_ma_results,
            'stop_loss': stop_loss_results,
            'take_profit': take_profit_results,
            'robustness': {
                'fast_ma': fast_ma_robust,
                'slow_ma': slow_ma_robust,
                'stop_loss': stop_loss_robust,
                'take_profit': take_profit_robust,
                'overall': overall_robust
            }
        }

if 'analysis' in st.session_state:
    analysis = st.session_state['analysis']
    
    # Top robustness gauge
    st.markdown("<h2 style='text-align: center; color: white;'>STRATEGY ROBUSTNESS</h2>", unsafe_allow_html=True)
    
    robustness = analysis['robustness']['overall']
    if robustness > 70:
        color = ACCENT_GREEN
        status = "Robust"
    elif robustness > 40:
        color = ACCENT_AMBER
        status = "Moderate"
    else:
        color = ACCENT_RED
        status = "Fragile"
    
    # Simple gauge using HTML/CSS
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: conic-gradient({color} 0% {robustness}%, {BG_CARD} {robustness}% 100%);
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 4px solid {BORDER};
        ">
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: {color};">{robustness:.0f}</div>
                <div style="font-size: 16px; color: {TEXT_SECONDARY};">{status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary heatmap
    st.markdown(section_header("Parameter Sensitivity Heatmap"), unsafe_allow_html=True)
    
    params = ['fast_ma', 'slow_ma', 'stop_loss_pct', 'take_profit_pct']
    metrics = ['total_return', 'sharpe', 'max_dd', 'win_rate']
    
    heatmap_data = []
    for param in params:
        results = analysis[param]
        base_val = {'fast_ma': fast_ma_base, 'slow_ma': slow_ma_base, 'stop_loss_pct': stop_loss_base, 'take_profit_pct': take_profit_base}[param]
        base_idx = np.argmin(np.abs(results['value'].values - base_val))
        
        for metric in metrics:
            base_perf = results[metric].iloc[base_idx]
            values = results[metric].values
            deviations = np.abs(values - base_perf) / abs(base_perf) if base_perf != 0 else np.abs(values)
            
            # Color based on max deviation
            max_dev = np.max(deviations)
            if max_dev <= 0.1:
                color = "#00c853cc"
            elif max_dev <= 0.2:
                color = "#00d4ff99"
            elif max_dev <= 0.4:
                color = "#ffc10799"
            else:
                color = "#ff174499"
            
            heatmap_data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{base_perf:.3f}",
                'Color': color
            })
    
    # Display as a grid of colored cells
    cols = st.columns(len(metrics))
    metric_names = [m.replace('_', ' ').title() for m in metrics]
    for i, metric in enumerate(metric_names):
        with cols[i]:
            st.markdown(f"<div style='text-align: center; font-weight: bold; color: {TEXT_SECONDARY};'>{metric}</div>", unsafe_allow_html=True)
            for param in params:
                cell = next((c for c in heatmap_data if c['Parameter'] == param.replace('_', ' ').title() and c['Metric'] == metric), None)
                if cell:
                    st.markdown(f"""
                    <div style="
                        background-color: {cell['Color']};
                        border: 1px solid {BORDER};
                        border-radius: 4px;
                        padding: 8px;
                        margin: 2px;
                        text-align: center;
                        color: white;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 12px;
                    ">{cell['Value']}</div>
                    """, unsafe_allow_html=True)
    
    # Per-parameter detail charts
    st.markdown(section_header("Parameter Detail Charts"), unsafe_allow_html=True)
    
    param_configs = [
        ('fast_ma', fast_ma_results, fast_ma_base, fast_ma_robust),
        ('slow_ma', slow_ma_results, slow_ma_base, slow_ma_robust),
        ('stop_loss_pct', stop_loss_results, stop_loss_base, stop_loss_robust),
        ('take_profit_pct', take_profit_results, take_profit_base, take_profit_robust)
    ]
    
    for param_name, results, base_val, robust_score in param_configs:
        fig = go.Figure()
        
        # Stable zone (within 20% of base)
        base_idx = np.argmin(np.abs(results['value'].values - base_val))
        base_perf = results['total_return'].iloc[base_idx]
        stable_min = base_perf * 0.8
        stable_max = base_perf * 1.2
        
        fig.add_shape(
            type="rect",
            x0=results['value'].min(),
            x1=results['value'].max(),
            y0=stable_min,
            y1=stable_max,
            fillcolor=ACCENT_GREEN + "33",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_trace(go.Scatter(
            x=results['value'],
            y=results['total_return'],
            mode='lines',
            line=dict(color=TEXT_PRIMARY, width=2),
            name='Total Return'
        ))
        
        fig.add_vline(x=base_val, line_dash="dash", line_color=ACCENT_CYAN)
        
        fig.update_layout(
            **get_plotly_layout(),
            title=f"{param_name.replace('_', ' ').title()} - Robustness: {robust_score:.0f}/100",
            xaxis_title=param_name.replace('_', ' ').title(),
            yaxis_title='Total Return'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation cards
    st.markdown(section_header("Parameter Interpretations"), unsafe_allow_html=True)
    
    for param_name, _, _, robust_score in param_configs:
        if robust_score > 70:
            border_color = ACCENT_GREEN
            text = f"{param_name.replace('_', ' ')} is Robust ({robust_score:.0f}/100) — performance stays stable across the tested range."
        elif robust_score > 40:
            border_color = ACCENT_AMBER
            text = f"{param_name.replace('_', ' ')} is Moderate ({robust_score:.0f}/100) — some sensitivity to parameter changes."
        else:
            border_color = ACCENT_RED
            text = f"{param_name.replace('_', ' ')} is Fragile ({robust_score:.0f}/100) — small changes cause large performance swings, possible overfitting."
        
        st.markdown(f"""
        <div style="
            background-color: {BG_CARD};
            border-left: 4px solid {border_color};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: {TEXT_SECONDARY};
        ">
            {text}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to start.")