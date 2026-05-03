import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from design_system import *

# Apply theme
apply_theme()

# Sidebar
st.sidebar.header("Simulation Parameters")

uploaded_file = st.sidebar.file_uploader("Upload Backtest CSV (columns: date, trade_return)", type="csv")

starting_capital = st.sidebar.number_input("Starting Capital", value=100000, min_value=1000, step=1000)

n_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)

run_simulation = st.sidebar.button("Run Simulation")

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    # Generate returns with slight positive edge
    returns = np.random.normal(0.001, 0.02, 200)
    # Add some loss clustering
    for i in range(0, 200, 50):
        returns[i:i+10] -= 0.015  # clustered losses
    df = pd.DataFrame({'date': dates, 'trade_return': returns})
    return df

@st.cache_data
def run_monte_carlo(df, start_cap, n_sim):
    n_trades = len(df)
    all_equity = np.zeros((n_sim, n_trades + 1))
    all_equity[:, 0] = start_cap
    all_max_dd = np.zeros(n_sim)
    
    for sim in range(n_sim):
        # Shuffle and add noise
        shuffled_returns = np.random.permutation(df['trade_return'].values)
        noise = np.random.uniform(-0.003, 0.003, n_trades)
        noisy_returns = shuffled_returns * (1 + noise)
        
        equity = [start_cap]
        peak = start_cap
        max_dd = 0
        
        for ret in noisy_returns:
            new_eq = equity[-1] * (1 + ret)
            equity.append(new_eq)
            
            if new_eq > peak:
                peak = new_eq
            
            dd = (peak - new_eq) / peak
            if dd > max_dd:
                max_dd = dd
        
        all_equity[sim] = equity
        all_max_dd[sim] = max_dd
    
    return all_equity, all_max_dd

if run_simulation or 'results' not in st.session_state:
    with st.spinner("Running Monte Carlo simulations..."):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'date' not in df.columns or 'trade_return' not in df.columns:
                st.error("CSV must have 'date' and 'trade_return' columns.")
                st.stop()
            df['date'] = pd.to_datetime(df['date'])
        else:
            df = generate_sample_data()
        
        all_equity, all_max_dd = run_monte_carlo(df, starting_capital, n_simulations)
        
        # Calculate statistics
        final_values = all_equity[:, -1]
        median_final = np.median(final_values)
        p5_final = np.percentile(final_values, 5)
        p95_final = np.percentile(final_values, 95)
        prob_loss = np.mean(final_values < starting_capital)
        prob_20dd = np.mean(all_max_dd >= 0.2)
        prob_30dd = np.mean(all_max_dd >= 0.3)
        dd_percentiles = np.percentile(all_max_dd, [5, 25, 50, 75, 95])
        
        # Original backtest
        original_returns = df['trade_return'].values
        original_equity = starting_capital * np.cumprod(1 + original_returns)
        original_final = original_equity[-1]
        original_percentile = np.mean(final_values <= original_final) * 100
        
        # Store results
        st.session_state['results'] = {
            'all_equity': all_equity,
            'all_max_dd': all_max_dd,
            'final_values': final_values,
            'median_final': median_final,
            'p5_final': p5_final,
            'p95_final': p95_final,
            'prob_loss': prob_loss,
            'prob_20dd': prob_20dd,
            'prob_30dd': prob_30dd,
            'dd_percentiles': dd_percentiles,
            'original_final': original_final,
            'original_percentile': original_percentile,
            'original_equity': np.concatenate([[starting_capital], original_equity]),
            'df': df
        }

if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        loss_color = ACCENT_RED if results['prob_loss'] > 0.3 else ACCENT_AMBER if results['prob_loss'] > 0.1 else ACCENT_GREEN
        st.markdown(metric_card("Probability of Loss", f"{results['prob_loss']*100:.1f}%", loss_color), unsafe_allow_html=True)
    with col2:
        med_ret = (results['median_final'] / starting_capital - 1) * 100
        st.markdown(metric_card("Median Return", f"{med_ret:.1f}%", pnl_color(med_ret)), unsafe_allow_html=True)
    with col3:
        worst_dd = results['dd_percentiles'][4] * 100
        st.markdown(metric_card("Worst 5% Max DD", f"{worst_dd:.1f}%", ACCENT_RED), unsafe_allow_html=True)
    with col4:
        if results['original_percentile'] > 90:
            risk = "HIGH"
            risk_color = ACCENT_RED
        elif results['original_percentile'] > 75:
            risk = "MEDIUM"
            risk_color = ACCENT_AMBER
        else:
            risk = "LOW"
            risk_color = ACCENT_GREEN
        st.markdown(metric_card("Overfitting Risk", risk, risk_color), unsafe_allow_html=True)
    
    # Fan chart
    n_trades = len(results['df'])
    x = np.arange(n_trades + 1)
    
    median_curve = np.median(results['all_equity'], axis=0)
    p5_curve = np.percentile(results['all_equity'], 5, axis=0)
    p95_curve = np.percentile(results['all_equity'], 95, axis=0)
    p25_curve = np.percentile(results['all_equity'], 25, axis=0)
    p75_curve = np.percentile(results['all_equity'], 75, axis=0)
    
    fig = go.Figure()
    
    # Add cloud of curves
    for sim in range(n_simulations):
        fig.add_trace(go.Scatter(
            x=x, 
            y=results['all_equity'][sim], 
            mode='lines', 
            line=dict(width=1, color='rgba(255,255,255,0.02)'), 
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=x, y=p95_curve, mode='lines', 
        line=dict(width=0, color='rgba(0,212,255,0.3)'), 
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5_curve, mode='lines', 
        fill='tonexty', fillcolor='rgba(0,212,255,0.2)', 
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p75_curve, mode='lines', 
        line=dict(width=0, color='rgba(0,212,255,0.5)'), 
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p25_curve, mode='lines', 
        fill='tonexty', fillcolor='rgba(0,212,255,0.4)', 
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    # Median curve
    fig.add_trace(go.Scatter(
        x=x, y=median_curve, mode='lines', 
        line=dict(width=3, color=ACCENT_CYAN), 
        name='Median'
    ))
    
    # Original curve
    fig.add_trace(go.Scatter(
        x=x, y=results['original_equity'], mode='lines', 
        line=dict(width=3, color=TEXT_PRIMARY), 
        name='Original Backtest'
    ))
    
    fig.update_layout(**get_plotly_layout())
    fig.update_layout(
        height=600, 
        title='Monte Carlo Equity Curves',
        xaxis_title='Trade Number',
        yaxis_title='Portfolio Value'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Histograms
    col1, col2 = st.columns(2)
    with col1:
        fig_hist_final = px.histogram(
            results['final_values'], 
            nbins=50, 
            title='Final Portfolio Values Distribution'
        )
        fig_hist_final.update_traces(
            marker_color=BG_CARD, 
            marker_line_color=ACCENT_CYAN, 
            marker_line_width=1
        )
        fig_hist_final.add_vline(
            x=results['original_final'], 
            line_dash='dash', 
            line_color=TEXT_PRIMARY,
            annotation_text="Original"
        )
        fig_hist_final.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_hist_final, use_container_width=True)
    
    with col2:
        fig_hist_dd = px.histogram(
            results['all_max_dd'] * 100, 
            nbins=50, 
            title='Max Drawdown Distribution (%)'
        )
        fig_hist_dd.update_traces(
            marker_color=BG_CARD, 
            marker_line_color=ACCENT_RED, 
            marker_line_width=1
        )
        fig_hist_dd.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_hist_dd, use_container_width=True)
    
    # Interpretation
    st.markdown(section_header("Analysis"), unsafe_allow_html=True)
    
    analysis_text = f"""
    Based on {n_simulations} simulations, there is a {results['prob_loss']*100:.1f}% chance of losing money, 
    a {results['prob_20dd']*100:.1f}% chance of a 20%+ drawdown, and the median outcome is 
    {(results['median_final']/starting_capital - 1)*100:.1f}% return. Your original backtest returned 
    {(results['original_final']/starting_capital - 1)*100:.1f}%, which falls in the {results['original_percentile']:.1f}th percentile.
    """
    
    border_color = ACCENT_RED if results['original_percentile'] > 90 else BG_CARD
    warning_text = "⚠️ High overfitting risk detected! Your backtest results are unusually good compared to random simulations." if results['original_percentile'] > 90 else ""
    
    st.markdown(f"""
    <div style="
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid {border_color};
        color: {TEXT_SECONDARY};
    ">
        <p>{analysis_text}</p>
        <p style="color: {ACCENT_RED};">{warning_text}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Upload a CSV or use sample data, then click 'Run Simulation' to start.")