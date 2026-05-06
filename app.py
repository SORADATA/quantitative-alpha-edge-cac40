import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import time
import os
from typing import Dict, Any, List

st.set_page_config(
    page_title="AlphaEdge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(interval=900000, key="datarefresh")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .kpi-container {
        background-color: #151922;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #262730;
        text-align: left;
    }
    .kpi-minimal { text-align: left; padding: 10px 0; }
    .kpi-label { font-size: 12px; color: #8b92a5; margin-bottom: 4px; }
    .kpi-value { font-size: 24px; font-weight: 700; color: #ffffff; }
    .kpi-delta-pos { color: #00cc96; font-size: 20px; font-weight: 600; }
    .kpi-delta-neg { color: #ef553b; font-size: 20px; font-weight: 600; }
    .disclaimer-box {
        background-color: #1E1E1E;
        color: #888888;
        padding: 20px;
        border-radius: 5px;
        font-size: 12px;
        border-top: 1px solid #333;
        margin-top: 50px;
        text-align: center;
    }
    .disclaimer-title {
        color: #EF553B;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 14px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

def load_market_config(config_name: str) -> Dict[str, Any]:
    target_path = CONFIG_DIR / config_name
    default_fallback = {
        "market_name": "CAC 40",
        "benchmark_ticker": "^FCHI",
        "currency": "EUR",
        "assets": []
    }
    if not target_path.exists():
        return default_fallback
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_fallback

@st.cache_data(ttl=600, show_spinner=False)


def load_all_data():
    history_file = BASE_DIR / 'portfolio_history.csv'
    signals_file = BASE_DIR / 'latest_signals.csv'
    rebalance_file = BASE_DIR / 'rebalance_history.csv'
    df_hist = pd.DataFrame()
    df_signals = pd.DataFrame()
    df_rebalance = pd.DataFrame()
    errors = []

    if history_file.exists():
        try:
            df_hist = pd.read_csv(history_file, index_col=0)
            df_hist.index = pd.to_datetime(df_hist.index, format='mixed', errors='coerce')
            df_hist = df_hist[df_hist.index.notna()].sort_index(ascending=True)
        except Exception as e:
            errors.append(f"Portfolio error: {e}") 
    if signals_file.exists():
        try:
            df_signals = pd.read_csv(signals_file)
        except Exception as e:
            errors.append(f"Signals error: {e}")       
    if rebalance_file.exists():
        try:
            df_rebalance = pd.read_csv(rebalance_file, index_col=0)
            df_rebalance.index = pd.to_datetime(df_rebalance.index)
        except Exception as e:
            errors.append(f"Rebalance error: {e}")        
    return df_hist, df_signals, df_rebalance, errors


def display_kpi_card(label, value, is_percent=True, color_code=False, prefix="", minimal=False):
    if pd.isna(value) or np.isinf(value):
        html_val = '<span class="kpi-value">N/A</span>'
    else:
        formatted_val = f"{prefix}{value:.1%}" if is_percent else f"{prefix}{value:.2f}"
        if color_code:
            color_class = "kpi-delta-pos" if value >= 0 else "kpi-delta-neg"
            arrow = "▲" if value >= 0 else "▼"
            html_val = f'<span class="{color_class}">{arrow} {formatted_val}</span>'
        else:
            html_val = f'<span class="kpi-value">{formatted_val}</span>'
    css_class = "kpi-minimal" if minimal else "kpi-container"
    st.markdown(f'<div class="{css_class}"><div class="kpi-label">{label}</div>{html_val}</div>', unsafe_allow_html=True)


def calculate_metrics(df):
    if df.empty or len(df) < 2:
        return 0, 0, 0, 0
    try:
        total_ret = (df['Strategy'].iloc[-1] / df['Strategy'].iloc[0]) - 1
        bench_ret = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) - 1
        alpha = total_ret - bench_ret 
        strategy_returns = df['Strategy'].pct_change().dropna()
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
        cum_ret = (1 + strategy_returns).cumprod()
        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        return total_ret, alpha, sharpe, max_dd
    except Exception:
        return 0, 0, 0, 0

def calculate_period_return(df, days=None, ytd=False, daily=False):
    if df.empty or 'Strategy' not in df.columns or len(df) < 2:
        return 0.0
    try:
        if daily:
            return (df['Strategy'].iloc[-1] / df['Strategy'].iloc[-2]) - 1
        last_price, last_date = df['Strategy'].iloc[-1], df.index[-1]
        if ytd:
            target_date = datetime(last_date.year, 1, 1)
        elif days:
            target_date = last_date - timedelta(days=days)
        else:
            target_date = df.index[0]
        idx = df.index.get_indexer([pd.Timestamp(target_date)], method='nearest')[0]
        start_price = df['Strategy'].iloc[idx]
        return ((last_price / start_price) - 1) if start_price != 0 else 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def get_live_ticker_data(ticker, period="1y"):
    for _ in range(2):
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=10)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                if 'adj close' not in df.columns and 'close' in df.columns:
                    df['adj close'] = df['close']
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

with st.spinner("Loading..."):
    df_hist, df_signals, df_rebalance, load_errors = load_all_data()

st.sidebar.title("AlphaEdge")
st.sidebar.caption("Quantitative Asset Allocation")

available_configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')] if CONFIG_DIR.exists() else []
selected_config = st.sidebar.selectbox("Market Selection", available_configs, index=0 if "cac40.json" not in available_configs else available_configs.index("cac40.json"))
MARKET_CONFIG = load_market_config(selected_config)
CURRENCY_SYMBOL = "€" if MARKET_CONFIG.get("currency") == "EUR" else "$"

if st.sidebar.button("Force Sync Pipeline"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GITHUB-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation)")

page = st.sidebar.radio("Navigation", ["Dashboard", "Daily Signals", "Data Explorer", "Model Details", "Rebalance History"])

if not df_hist.empty:
    last_dt = df_hist.index[-1]
    st.sidebar.info(f"Last Update: {last_dt.date()}")
    st.sidebar.markdown("🟢 System Online" if (datetime.now() - last_dt).days <= 1 else "🔴 Update Required")

if page == "Dashboard":
    st.title(f"Portfolio Overview: {MARKET_CONFIG.get('market_name')}")
    if not df_hist.empty:
        tot_ret, alpha, sharpe, max_dd = calculate_metrics(df_hist)
        c1, c2, c3, c4 = st.columns(4)
        with c1: display_kpi_card("Total Return", tot_ret, color_code=True)
        with c2: display_kpi_card("Alpha vs Bench", alpha, color_code=True)
        with c3: display_kpi_card("Sharpe Ratio", sharpe, is_percent=False)
        with c4: display_kpi_card("Max Drawdown", max_dd, color_code=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: display_kpi_card("YTD", calculate_period_return(df_hist, ytd=True), color_code=True, minimal=True)
        with k2: display_kpi_card("6 Months", calculate_period_return(df_hist, days=180), color_code=True, minimal=True)
        with k3: display_kpi_card("3 Months", calculate_period_return(df_hist, days=90), color_code=True, minimal=True)
        with k4: display_kpi_card("1 Month", calculate_period_return(df_hist, days=30), color_code=True, minimal=True)
        with k5: display_kpi_card("Daily", calculate_period_return(df_hist, daily=True), color_code=True, minimal=True)

        df_base = df_hist.apply(lambda x: x / x.iloc[0] * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['Strategy'], name='Strategy', line=dict(color='#00CC96', width=2), fill='tonexty', fillcolor='rgba(0, 204, 150, 0.1)'))
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['Benchmark'], name='Benchmark', line=dict(color='#8b92a5', width=1, dash='dash')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Daily Signals":
    st.title("Daily Trading Signals")
    if not df_signals.empty:
        st.dataframe(df_signals, use_container_width=True, height=500, hide_index=True,
            column_config={
                "Allocation": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1),
                "Proba_Hausse": st.column_config.NumberColumn("Probability ↑", format="%.1f%%")
            })

elif page == "Data Explorer":
    st.title("Market Data Explorer")
    tickers = MARKET_CONFIG.get("assets", [])
    selected_ticker = st.selectbox("Select Asset", tickers if tickers else (df_signals['Ticker'].tolist() if not df_signals.empty else []))
    df_asset = get_live_ticker_data(selected_ticker)
    if not df_asset.empty:
        m1, m2, m3, m4 = st.columns(4)
        last_price = df_asset['adj close'].iloc[-1]
        daily_var = (last_price / df_asset['adj close'].iloc[-2]) - 1
        with m1: display_kpi_card("Price", last_price, is_percent=False, prefix=f"{CURRENCY_SYMBOL} ")
        with m2: display_kpi_card("Change", daily_var, color_code=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_asset.index, open=df_asset['open'], high=df_asset['high'], low=df_asset['low'], close=df_asset['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_asset.index, y=df_asset['volume'], name='Volume'), row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Details":
    st.title("Model Configuration")
    st.info("Strategy: XGBoost Prediction + K-Means Regime Detection + Markowitz Optimization")
    metrics_path = BASE_DIR / "src" / "models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            m = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
            c2.metric("Precision", f"{m.get('precision', 0):.1%}")
            c3.metric("ROC AUC", f"{m.get('auc_score', 0):.3f}")

elif page == "Rebalance History":
    st.title("Rebalancing History")
    if not df_rebalance.empty:
        st.dataframe(df_rebalance.sort_index(ascending=False), use_container_width=True)

st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">AVIS DE NON-RESPONSABILITÉ</div>
    <p>Informations à titre éducatif uniquement. Pas de conseil en investissement.</p>
</div>
""", unsafe_allow_html=True)