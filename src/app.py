"""
PHASE 7 — Interactive Streamlit Dashboard
Deploys the backtest results into an interactive web application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

# --- Config ---
st.set_page_config(page_title="Algorithmic Trading Backtester", layout="wide")
DATA_DIR = "data_backtested"

# --- Helper Functions ---
@st.cache_data
def load_data(asset_name):
    path = os.path.join(DATA_DIR, f"{asset_name}_backtest.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df

def calculate_metrics(df):
    TRADING_DAYS = 252
    
    # Strategy
    strat_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1
    strat_ann_return = (1 + strat_return) ** (TRADING_DAYS / len(df)) - 1
    strat_vol = df["net_return"].std() * np.sqrt(TRADING_DAYS)
    strat_sharpe = strat_ann_return / strat_vol if strat_vol != 0 else 0
    strat_dd = ((df["equity"] - df["equity"].cummax()) / df["equity"].cummax()).min()
    
    # Benchmark
    bh_return = (df["buy_hold_equity"].iloc[-1] / df["buy_hold_equity"].iloc[0]) - 1
    bh_ann_return = (1 + bh_return) ** (TRADING_DAYS / len(df)) - 1
    bh_vol = df["asset_return"].std() * np.sqrt(TRADING_DAYS)
    bh_sharpe = bh_ann_return / bh_vol if bh_vol != 0 else 0
    bh_dd = ((df["buy_hold_equity"] - df["buy_hold_equity"].cummax()) / df["buy_hold_equity"].cummax()).min()

    return {
        "strat": {"return": strat_ann_return, "vol": strat_vol, "sharpe": strat_sharpe, "dd": strat_dd, "final": df["equity"].iloc[-1]},
        "bh": {"return": bh_ann_return, "vol": bh_vol, "sharpe": bh_sharpe, "dd": bh_dd, "final": df["buy_hold_equity"].iloc[-1]}
    }

# --- UI Layout ---
st.title("📈 Quantitative Backtest Engine: MA + RSI Strategy")
st.markdown("An end-to-end MLOps pipeline evaluating a risk-averse momentum strategy, accounting for transaction costs and look-ahead bias.")

# Asset Selector
assets = [f.replace("_backtest.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_backtest.csv")]
selected_asset = st.selectbox("Select Asset to Analyze", [a.upper() for a in assets])

if selected_asset:
    # Load data for selected asset
    df = load_data(selected_asset.lower())
    metrics = calculate_metrics(df)
    
    st.divider()
    
    # --- KPI Cards ---
    st.subheader(f"{selected_asset} Risk & Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Portfolio Value", f"${metrics['strat']['final']:,.2f}", f"vs ${metrics['bh']['final']:,.2f} (Hold)")
    with col2:
        st.metric("Annualized Return", f"{metrics['strat']['return']*100:.2f}%", f"{metrics['bh']['return']*100:.2f}% (Hold)")
    with col3:
        st.metric("Max Drawdown (Risk)", f"{metrics['strat']['dd']*100:.2f}%", f"{metrics['bh']['dd']*100:.2f}% (Hold)", delta_color="inverse")
    with col4:
        st.metric("Sharpe Ratio", f"{metrics['strat']['sharpe']:.2f}", f"{metrics['bh']['sharpe']:.2f} (Hold)")

    # --- Interactive Plotly Chart ---
    st.subheader("Interactive Equity Curve")
    
    fig = go.Figure()
    
    # Strategy Line
    fig.add_trace(go.Scatter(x=df.index, y=df["equity"], mode='lines', name='Strategy Portfolio', line=dict(color='#29b5e8', width=2)))
    # Benchmark Line
    fig.add_trace(go.Scatter(x=df.index, y=df["buy_hold_equity"], mode='lines', name='Buy & Hold Benchmark', line=dict(color='#ff9f36', width=2, dash='dot')))
    
    fig.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_dark" # Looks incredibly professional for finance apps
    )
    
    st.plotly_chart(fig, use_container_width=True)