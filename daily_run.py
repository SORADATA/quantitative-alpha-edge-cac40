
import json
import warnings
import os
from datetime import datetime

from const import (
    BASE_DIR,
    CONFIG_DIR,
    TARGET_CLUSTER,
    PROBA_THRESHOLD,
    FEATURE_COLS
)
from src.utils.logger import setup_logger
from src.utils.config_loader import load_market_config
from src.utils.market_utils import build_export_df
from src.pipeline.etl import get_data_pipeline, load_models
from src.pipeline.backtest import backtest_strategy_with_rebalancing, get_optimal_weights

warnings.filterwarnings('ignore')
logger = setup_logger("DailyRun")


def run_pipeline_for_config(config_file: str):
    start_time = datetime.now()
    
    config = load_market_config(config_file)
    market_name = config.get('market_name', 'Unknown')
    tickers = config.get('assets', [])
    benchmark = config.get('benchmark_ticker', '^GSPC')

    logger.info("-" * 60)
    logger.info(f"STARTING PIPELINE | MARKET: {market_name} | FILE: {config_file}")
    logger.info(f"ASSETS: {len(tickers)} | BENCHMARK: {benchmark}")
    logger.info("-" * 60)

    try:
        xgb_model, kmeans_model = load_models()
        if xgb_model is None:
            raise RuntimeError("ML Models not found.")

        df_daily, df_monthly = get_data_pipeline(config_file)
        if df_daily is None or df_monthly is None:
            raise RuntimeError(f"Data Pipeline failure for {market_name}")

        last_date = df_monthly.index.get_level_values('date').max()
        today_data = df_monthly.xs(last_date, level=0).copy()
        
        today_data['cluster'] = kmeans_model.predict(today_data[['rsi']].fillna(50))
        today_data['proba_upside'] = xgb_model.predict_proba(today_data[FEATURE_COLS].fillna(0))[:, 1]
        
        selected = today_data[
            (today_data['cluster'] == TARGET_CLUSTER) &
            (today_data['proba_upside'] > PROBA_THRESHOLD)
        ]

        final_alloc = {}
        if not selected.empty:
            sel_tickers = selected.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[sel_tickers].iloc[-252:].dropna(axis=1)
            weights, success = get_optimal_weights(prices_subset)
            final_alloc = weights if success else {t: 1.0/len(sel_tickers) for t in sel_tickers}

        suffix = config_file.replace('.json', '')
        export_df = build_export_df(today_data, final_alloc)
        export_df.to_csv(BASE_DIR / f'latest_signals_{suffix}.csv', index=False)

        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily, df_monthly, xgb_model, kmeans_model, get_optimal_weights
        )
        
        hist_df.to_csv(BASE_DIR / f'portfolio_history_{suffix}.csv')
        rebal_df.to_csv(BASE_DIR / f'rebalance_history_{suffix}.csv')

        metadata = {
            'market_name': market_name,
            'last_update': datetime.now().isoformat(),
            'n_assets_tracked': len(tickers),
            'current_allocation': final_alloc
        }
        with open(BASE_DIR / f'metadata_{suffix}.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"SUCCESS: {market_name} processed in {duration:.1f}s")

    except Exception as e:
        logger.error(f"FAILURE for {market_name}: {e}")

def run_all_pipelines():
    if not CONFIG_DIR.exists():
        logger.error(f"Config directory not found at {CONFIG_DIR}")
        return

    config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    
    if not config_files:
        logger.warning("No JSON config files found.")
        return

    logger.info(f"Found {len(config_files)} markets to process.")
    
    for config_file in config_files:
        run_pipeline_for_config(config_file)

if __name__ == "__main__":
    run_all_pipelines()