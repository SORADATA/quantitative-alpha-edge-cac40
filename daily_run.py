import os
import sys
import json
import time
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any

# --- DATA SCIENCE STACK ---
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# --- TECHNICAL ANALYSIS ---
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

# --- PORTFOLIO OPTIMIZATION ---
from pypfopt import EfficientFrontier, risk_models, expected_returns

# =============================================================================
# 0. CONSTANTS & SYSTEM CONFIGURATION
# =============================================================================

warnings.filterwarnings('ignore')

# Trading Constants
TRADING_DAYS_YEAR = 252
RISK_FREE_RATE = 0.03
TARGET_CLUSTER = 3
PROBA_THRESHOLD = 0.55

# Auto-detect project root safely
BASE_DIR = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parent))

MODEL_DIR = BASE_DIR / "src" / "models"
DATA_DIR = BASE_DIR / "data" / "raw"
CONFIG_FILE = BASE_DIR / "config" / "market_config.json"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. LOGGING CONFIGURATION
# =============================================================================

def setup_logger() -> logging.Logger:
    """Configures production-grade logging."""
    logger = logging.getLogger("PortfolioPipeline")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs if executed multiple times in the same session
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s')
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

logger = setup_logger()

# =============================================================================
# 2. MARKET CONFIGURATION LOADER
# =============================================================================

def load_market_config() -> Dict[str, Any]:
    """Loads market configuration from JSON file with safe fallbacks."""
    default_config = {
        "market_name": "Default (CAC40)",
        "benchmark_ticker": "^FCHI",
        "assets": ["AI.PA", "AIR.PA", "SAN.PA", "MC.PA"]
    }

    if CONFIG_FILE.exists():
        logger.info(f"Loading configuration from: {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}. Using defaults.")
    else:
        logger.warning(f"Config file not found at {CONFIG_FILE}. Using defaults.")
        
    return default_config

MARKET_CONFIG = load_market_config()
TICKERS = MARKET_CONFIG.get('assets', [])
BENCHMARK_TICKER = MARKET_CONFIG.get('benchmark_ticker', '^FCHI')
MARKET_NAME = MARKET_CONFIG.get('market_name', 'Unknown Market')

logger.info(f"Market: {MARKET_NAME} | Assets: {len(TICKERS)} | Benchmark: {BENCHMARK_TICKER}")

# =============================================================================
# 3. TICKER MANAGEMENT & VALIDATION
# =============================================================================

def handle_ticker_changes() -> Tuple[Dict[str, str], List[str]]:
    """Defines known ticker changes and delistings."""
    TICKER_CHANGES = {} # e.g., {'OLD': 'NEW'}
    DELISTED_TICKERS = [] # e.g., ['BANKRUPT.PA']
    return TICKER_CHANGES, DELISTED_TICKERS

def validate_and_clean_tickers(df: pd.DataFrame, tickers_list: List[str], max_days_stale: int = 30) -> Tuple[pd.DataFrame, List[str], Dict[str, List]]:
    """Validates data quality, removing stale or invalid tickers."""
    logger.info("Validating ticker data integrity...")
    
    alerts = {'delisted': [], 'stale': [], 'missing': [], 'warnings': []}
    tickers_in_data = df.index.get_level_values('ticker').unique().tolist()
    
    # 1. Check missing
    missing_tickers = set(tickers_list) - set(tickers_in_data)
    if missing_tickers:
        alerts['missing'] = list(missing_tickers)
        logger.warning(f"{len(missing_tickers)} tickers missing from download: {list(missing_tickers)[:5]}...")

    # 2. Check stale data
    last_date = df.index.get_level_values('date').max()
    stale_tickers = []
    
    for ticker in tickers_in_data:
        ticker_data = df.xs(ticker, level='ticker')
        if ticker_data.empty: continue
        
        days_stale = (last_date - ticker_data.index.max()).days
        if days_stale > max_days_stale:
            stale_tickers.append({'ticker': ticker, 'days_stale': days_stale})

    if stale_tickers:
        alerts['stale'] = stale_tickers
        logger.warning(f"{len(stale_tickers)} stale tickers detected (> {max_days_stale} days).")
        
        tickers_to_remove = [t['ticker'] for t in stale_tickers]
        df = df[~df.index.get_level_values('ticker').isin(tickers_to_remove)]
        alerts['delisted'] = tickers_to_remove

    valid_tickers = df.index.get_level_values('ticker').unique().tolist()
    logger.info(f"Valid Tickers: {len(valid_tickers)} / {len(tickers_list)}")
    
    return df, valid_tickers, alerts

# =============================================================================
# 4. ETL & FEATURE ENGINEERING PIPELINE
# =============================================================================

def load_models() -> Tuple[Optional[Any], Optional[Any]]:
    """Loads pre-trained models safely."""
    logger.info(f"Loading ML models from {MODEL_DIR}...")
    try:
        with open(MODEL_DIR / 'xgboost_model.pkl', 'rb') as f:
            xgb = pickle.load(f)
        with open(MODEL_DIR / 'kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        return xgb, kmeans
    except FileNotFoundError:
        logger.error("Models not found. Ensure training notebook was executed.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI, Bollinger, MACD, ATR, and Garman-Klass Volatility."""
    logger.info("Computing Technical Indicators...")

    # Garman-Klass
    df['garman_klass_vol'] = (
        (np.log(df['high']) - np.log(df['low']))**2 / 2 - 
        (2*np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open']))**2
    )

    # TA-Lib Indicators per ticker
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = df.loc[idx, 'adj close']
        
        if len(close_series) > 20:
            df.loc[idx, 'rsi'] = RSIIndicator(close=close_series, window=20).rsi().values
            
            close_log = np.log1p(close_series)
            bb = BollingerBands(close=close_log, window=20, window_dev=2)
            df.loc[idx, 'bb_low'] = bb.bollinger_lband().values
            df.loc[idx, 'bb_mid'] = bb.bollinger_mavg().values
            df.loc[idx, 'bb_high'] = bb.bollinger_hband().values

    # ATR
    def compute_atr(stock_data):
        if len(stock_data) < 15: return pd.Series(np.nan, index=stock_data.index)
        atr = AverageTrueRange(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], window=14).average_true_range()
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    # MACD
    def compute_macd(stock_data):
        if len(stock_data) < 26: return pd.Series(np.nan, index=stock_data.index)
        macd_val = MACD(close=stock_data['adj close'], window_slow=26, window_fast=12, window_sign=9).macd()
        return macd_val.sub(macd_val.mean()).div(macd_val.std())

    df['macd'] = df.groupby(level=1, group_keys=False).apply(compute_macd)
    df['euro_volume'] = (df['adj close'] * df['volume']) / 1e6

    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates winsorized momentum returns."""
    logger.info("Computing Momentum Returns...")
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        returns_raw = df['adj close'].pct_change(lag)
        lower_bound = returns_raw.expanding(min_periods=12).quantile(outlier_cutoff)
        upper_bound = returns_raw.expanding(min_periods=12).quantile(1 - outlier_cutoff)
        df[f'return_{lag}m'] = returns_raw.clip(lower=lower_bound, upper=upper_bound)

    return df

def get_fama_french_betas(data: pd.DataFrame) -> pd.DataFrame:
    """Retrieves Fama-French factors and computes rolling betas safely."""
    logger.info("Retrieving Fama-French Factors (Europe 5 Factors)...")
    
    # Isolated import to prevent hard crash if pandas version is incompatible
    try:
        import pandas_datareader.data as web
    except ImportError as e:
        logger.error(f"Missing dependency or Pandas incompatibility: {e}. Filling FF factors with zeros.")
        return data.assign(**{f: 0.0 for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']})

    try:
        factor_data = web.DataReader('Europe_5_Factors', 'famafrench', start='2010')[0].drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample('BM').last().div(100)
        factor_data.index.name = 'date'

        data_ff = data.copy()
        if 'return_1m' not in data_ff.columns: return data

        betas_list = []
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        for ticker in data_ff.index.get_level_values(1).unique():
            y = data_ff.xs(ticker, level=1).get('return_1m')
            if y is None or y.dropna().empty: continue
            
            X = factor_data.loc[factor_data.index.intersection(y.index)]
            y = y.loc[X.index]

            if len(y) > 24:
                exog = sm.add_constant(X[factors])
                params = RollingOLS(y, exog, window=24).fit().params.drop('const', axis=1)
                params['ticker'] = ticker
                betas_list.append(params)

        if not betas_list: return data

        betas_df = pd.concat(betas_list).set_index('ticker', append=True)
        data = data.join(betas_df.groupby('ticker').shift()) 
        data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

        return data

    except Exception as e:
        logger.error(f"Fama-French retrieval failed ({e}). Filling with zeros.")
        return data.assign(**{f: 0.0 for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']})

def get_data_pipeline() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Executes the full ETL pipeline."""
    ticker_changes, delisted = handle_ticker_changes()
    active_tickers = [ticker_changes.get(t, t) for t in TICKERS if t not in delisted]
    
    end_date = (datetime.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(datetime.today()) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    
    logger.info(f"Downloading Market Data ({start_date} -> {end_date}) for {len(active_tickers)} assets...")
    
    df = None
    for attempt in range(1, 4):
        try:
            df = yf.download(active_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False, threads=True)
            if not df.empty:
                logger.info(f"Download successful (Attempt {attempt})")
                break
            logger.warning(f"Empty data received (Attempt {attempt})")
        except Exception as e:
            logger.warning(f"Download error (Attempt {attempt}): {e}")
            time.sleep(5)
    
    if df is None or df.empty:
        logger.error("FATAL: Failed to download data after retries.")
        return None, None

    df = df.stack(future_stack=True) # future_stack prevents deprecation warnings
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    
    if 'adj close' not in df.columns and 'close' in df.columns:
        df['adj close'] = df['close']

    df, valid_tickers, alerts = validate_and_clean_tickers(df, active_tickers)
    
    with open(BASE_DIR / 'ticker_validation.json', 'w') as f:
        json.dump({'date': str(datetime.now()), 'alerts': alerts, 'valid_tickers': len(valid_tickers)}, f, indent=2)

    logger.info(f"Saving raw data to {DATA_DIR}...")
    df.to_parquet(DATA_DIR / 'daily_raw.parquet', compression='gzip')

    df = compute_technical_indicators(df)

    logger.info("Resampling to Monthly Frequency...")
    last_cols = [c for c in df.columns if c not in ['euro_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    data_monthly = pd.concat([
        df.unstack('ticker')['euro_volume'].resample('BM').mean().stack('ticker').to_frame('euro_volume'),
        df.unstack()[last_cols].resample('BM').last().stack('ticker')
    ], axis=1).dropna()

    data_monthly = data_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
    data_monthly = get_fama_french_betas(data_monthly)

    vars_to_lag = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'euro_volume', 'garman_klass_vol']
    for col in vars_to_lag:
        if col in data_monthly.columns:
            data_monthly[f'{col}_lag1'] = data_monthly.groupby('ticker')[col].shift(1)

    logger.info("Saving processed monthly features...")
    data_monthly.to_parquet(DATA_DIR / 'monthly_features.parquet', compression='gzip')

    return df, data_monthly

# =============================================================================
# 5. PORTFOLIO OPTIMIZATION
# =============================================================================

def get_optimal_weights(prices_df: pd.DataFrame) -> Tuple[Dict[str, float], bool]:
    """Calculates Mean-Variance Optimization weights."""
    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_YEAR)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

        max_weight = max(0.25, 1.0 / len(prices_df.columns) * 2.0)
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
        
        weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        return ef.clean_weights(), True
    except Exception as e:
        logger.warning(f"Optimization Failed: {e}. Falling back to equal weights.")
        return {}, False

# =============================================================================
# 6. BACKTESTING ENGINE
# =============================================================================

def backtest_strategy_with_rebalancing(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, xgb_model: Any, kmeans_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulates the strategy historically with monthly rebalancing."""
    logger.info("Starting Backtest with Monthly Rebalancing...")
    
    current_portfolio_value = 100.0
    current_benchmark_value = 100.0
    portfolio_values = []
    rebalance_log = []
    
    daily_prices = df_daily['adj close'].unstack().ffill()
    
    # Download Benchmark
    try:
        start_bench = df_daily.index.get_level_values('date').min()
        end_bench = df_daily.index.get_level_values('date').max() + pd.DateOffset(days=1)
        bench_data = yf.download(BENCHMARK_TICKER, start=start_bench, end=end_bench, progress=False, auto_adjust=False)
        bench_prices = bench_data['Close'].iloc[:, 0] if isinstance(bench_data.columns, pd.MultiIndex) else bench_data['Close']
        benchmark_returns = bench_prices.reindex(daily_prices.index, method='ffill').pct_change().fillna(0)
        logger.info(f"Benchmark loaded ({len(benchmark_returns)} days)")
    except Exception as e:
        logger.warning(f"Benchmark download failed ({e}). Using average market returns.")
        benchmark_returns = daily_prices.mean(axis=1).pct_change().fillna(0)

    feature_cols = [
        'rsi', 'macd', 'bb_low', 'bb_high', 'atr', 'return_2m', 'return_3m', 'return_6m',
        'euro_volume_lag1', 'garman_klass_vol_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 
        'RMW_lag1', 'CMA_lag1', 'cluster'
    ]
    
    monthly_dates = df_monthly.index.get_level_values('date').unique().sort_values()
    
    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = df_monthly.xs(month_date, level='date').copy()
        
        if 'rsi' in month_data.columns:
            month_data['cluster'] = kmeans_model.predict(month_data[['rsi']].fillna(50))
            
        if any(c not in month_data.columns for c in feature_cols): continue
        
        month_data['proba_upside'] = xgb_model.predict_proba(month_data[feature_cols].fillna(0))[:, 1]
        
        selected = month_data[(month_data['cluster'] == TARGET_CLUSTER) & (month_data['proba_upside'] > PROBA_THRESHOLD)]
        
        new_allocation = {}
        if not selected.empty:
            tickers = selected.index.tolist()
            prices_subset = daily_prices[tickers].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1)
            if not prices_subset.empty and len(prices_subset.columns) >= 3:
                weights, success = get_optimal_weights(prices_subset)
                new_allocation = weights if success else {t: 1.0/len(tickers) for t in tickers}
                
        trading_mask = (daily_prices.index >= month_date) & (daily_prices.index < monthly_dates[i + 1])
        trading_days = daily_prices.index[trading_mask]
        
        for date in trading_days:
            bench_ret = benchmark_returns.get(date, 0.0)
            strat_ret = 0.0
            
            if new_allocation:
                daily_rets = [
                    (daily_prices[t].iloc[daily_prices[t].index.get_loc(date)] / daily_prices[t].iloc[daily_prices[t].index.get_loc(date)-1]) - 1 
                    if t in daily_prices.columns and daily_prices[t].index.get_loc(date) > 0 else 0.0 
                    for t in new_allocation.keys()
                ]
                strat_ret = np.average(pd.Series(daily_rets).fillna(0), weights=list(new_allocation.values())) if daily_rets else 0.0
                
            current_benchmark_value *= (1 + bench_ret)
            current_portfolio_value *= (1 + strat_ret)
            
            portfolio_values.append({'Date': date, 'Strategy': current_portfolio_value, 'Benchmark': current_benchmark_value})
            
        rebalance_log.append({'Date': month_date, 'N_Stocks': len(new_allocation), 'Allocation': new_allocation})

    logger.info(f"Backtest Complete. Final Value: {current_portfolio_value:.2f}")
    return pd.DataFrame(portfolio_values).set_index('Date'), pd.DataFrame(rebalance_log).set_index('Date')

# =============================================================================
# 7. MAIN ORCHESTRATOR
# =============================================================================

def run_pipeline():
    """Main execution entry point."""
    start_time = datetime.now()
    logger.info("="*60)
    logger.info(f"🚀 STARTING DAILY PIPELINE")
    logger.info("="*60)
    
    try:
        xgb_model, kmeans_model = load_models()
        if xgb_model is None: raise RuntimeError("ML Models not found.")

        df_daily, df_monthly = get_data_pipeline()
        if df_daily is None: raise RuntimeError("Data Pipeline Failed.")

        last_date = df_monthly.index.get_level_values('date').max()
        logger.info(f"Generating signals for: {last_date.date()}")
        
        today_data = df_monthly.xs(last_date, level=0).copy()
        
        if 'rsi' in today_data.columns:
            today_data['cluster'] = kmeans_model.predict(today_data[['rsi']].fillna(50))
            
        feature_cols = [
            'rsi', 'macd', 'bb_low', 'bb_high', 'atr', 'return_2m', 'return_3m', 'return_6m',
            'euro_volume_lag1', 'garman_klass_vol_lag1', 'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 
            'RMW_lag1', 'CMA_lag1', 'cluster'
        ]
        
        today_data['proba_upside'] = xgb_model.predict_proba(today_data[feature_cols].fillna(0))[:, 1]
        
        selected_stocks = today_data[(today_data['cluster'] == TARGET_CLUSTER) & (today_data['proba_upside'] > PROBA_THRESHOLD)]
        logger.info(f"Selected Assets: {selected_stocks.index.tolist()}")
        
        final_alloc = {}
        if not selected_stocks.empty:
            tickers = selected_stocks.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1)
            weights, success = get_optimal_weights(prices_subset)
            final_alloc = weights if success else {t: 1.0/len(tickers) for t in tickers}

        # Save outputs
        export_df = today_data[['cluster', 'proba_upside', 'rsi', 'return_3m']].reset_index()
        export_df.rename(columns={'ticker': 'Ticker', 'proba_upside': 'Proba_Hausse', 'return_3m': 'Return_3M', 'cluster': 'Cluster', 'rsi': 'RSI'}, inplace=True)
        export_df['Proba_Hausse'] *= 100
        export_df['Allocation'] = export_df['Ticker'].map(final_alloc).fillna(0.0)
        export_df['Signal'] = np.where(export_df['Allocation'] > 0, 'BUY', 'NEUTRAL')
        
        export_df.to_csv(BASE_DIR / 'latest_signals.csv', index=False)
        
        hist_df, rebal_df = backtest_strategy_with_rebalancing(df_daily, df_monthly, xgb_model, kmeans_model)
        hist_df.to_csv(BASE_DIR / 'portfolio_history.csv')
        rebal_df.to_csv(BASE_DIR / 'rebalance_history.csv')
        
        with open(BASE_DIR / 'data_metadata.json', 'w') as f:
            json.dump({
                'market_name': MARKET_NAME, 'last_update': datetime.now().isoformat(),
                'data_start': df_daily.index.get_level_values('date').min().isoformat(),
                'n_assets_tracked': len(TICKERS), 'current_allocation': final_alloc
            }, f, indent=2)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"🏁 PIPELINE COMPLETED SUCCESSFULLY in {duration:.1f}s")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()