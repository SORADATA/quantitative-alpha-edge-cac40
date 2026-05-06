import io
import json
import os
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
from huggingface_hub import HfApi

from src.utils.logger import setup_logger
from src.utils.config_loader import load_market_config
from src.utils.market_utils import build_export_df
from src.pipeline.etl import get_data_pipeline, load_models
from src.backtest.backtest import backtest_strategy_with_rebalancing
from src.strategy.signals import AlphaSignal


warnings.filterwarnings("ignore")
logger = setup_logger("DailyRun")


# =============================================================================
# 1. GLOBAL CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_CONFIG_PATH = BASE_DIR / "config" / "backtest_config.yaml"


def load_global_config() -> Dict[str, Any]:
    if not BACKTEST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {BACKTEST_CONFIG_PATH}")

    with open(BACKTEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths_cfg = cfg.get("paths", {})
    strategy_cfg = cfg.get("strategy", {})
    hf_cfg = cfg.get("huggingface", {})
    backtest_cfg = cfg.get("backtest", {})

    config_dir = Path(paths_cfg.get("config_dir", "config"))
    if not config_dir.is_absolute():
        config_dir = BASE_DIR / config_dir

    end_time_raw = backtest_cfg.get("end_time")
    end_time = end_time_raw if end_time_raw else date.today().strftime("%Y-%m-%d")

    return {
        "BASE_DIR": BASE_DIR,
        "CONFIG_DIR": config_dir,
        "TARGET_CLUSTER": strategy_cfg.get("target_cluster", 1),
        "PROBA_THRESHOLD": strategy_cfg.get("proba_threshold", 0.55),
        "TOP_K": strategy_cfg.get("top_k", 5),
        "FEATURE_COLS": strategy_cfg.get("feature_cols", []),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "REPO_ID": hf_cfg.get("repo_id", os.getenv("HF_REPO_ID", "")),
        "END_TIME": end_time,
    }


GLOBAL_CFG = load_global_config()

CONFIG_DIR = GLOBAL_CFG["CONFIG_DIR"]
TARGET_CLUSTER = GLOBAL_CFG["TARGET_CLUSTER"]
PROBA_THRESHOLD = GLOBAL_CFG["PROBA_THRESHOLD"]
TOP_K = GLOBAL_CFG["TOP_K"]
FEATURE_COLS = GLOBAL_CFG["FEATURE_COLS"]
HF_TOKEN = GLOBAL_CFG["HF_TOKEN"]
REPO_ID = GLOBAL_CFG["REPO_ID"]
END_TIME = GLOBAL_CFG["END_TIME"]

logger.info(f"Pipeline running for period up to: {END_TIME}")


# =============================================================================
# 2. HUGGING FACE SYNC
# =============================================================================

def upload_to_hub(df: pd.DataFrame, filename: str) -> None:
    """Upload a dataframe to Hugging Face datasets as parquet."""
    if df is None or df.empty:
        logger.warning(f"Skip upload for {filename}: empty dataframe")
        return

    if not HF_TOKEN or not REPO_ID:
        logger.info(f"Local mode: HF upload skipped for {filename}")
        return

    try:
        api = HfApi()
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=True, compression="gzip")
        parquet_buffer.seek(0)

        api.upload_file(
            path_or_fileobj=parquet_buffer,
            path_in_repo=f"data/{filename}.parquet",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        logger.info(f"Sync successful on HF: {filename}.parquet")

    except Exception as e:
        logger.error(f"Sync failed for {filename}: {e}")


# =============================================================================
# 3. TOP-K ALLOCATION
# =============================================================================

def build_topk_allocation(today_data: pd.DataFrame) -> Dict[str, float]:
    """
    Build today's live allocation:
    1. filter on cluster
    2. filter on probability threshold
    3. keep top-k highest proba names
    4. equal weight
    """
    if today_data.empty:
        return {}

    selected = today_data[
        (today_data["cluster"] == TARGET_CLUSTER) &
        (today_data["proba_upside"] >= PROBA_THRESHOLD)
    ].copy()

    if selected.empty:
        logger.warning("No assets passed cluster + probability filters.")
        return {}

    selected = selected.sort_values("proba_upside", ascending=False).head(TOP_K)

    n_assets = len(selected)
    if n_assets == 0:
        return {}

    weight = 1.0 / n_assets
    allocation = {ticker: weight for ticker in selected.index.tolist()}

    logger.info(
        f"Top-k allocation built with {n_assets} assets | "
        f"cluster={TARGET_CLUSTER} | threshold={PROBA_THRESHOLD:.2f} | top_k={TOP_K}"
    )
    return allocation


# =============================================================================
# 4. MARKET PIPELINE
# =============================================================================

def run_pipeline_for_config(config_file: str) -> None:
    start_time = datetime.now()
    config = load_market_config(config_file)
    market_name = config.get("market_name", "Unknown")

    logger.info("-" * 60)
    logger.info(f"STARTING PIPELINE | MARKET: {market_name}")
    logger.info("-" * 60)

    try:
        # 1. Load models + data
        xgb_model, kmeans_model = load_models()
        df_daily, df_monthly = get_data_pipeline(config_file)

        if df_daily is None or df_monthly is None or df_daily.empty or df_monthly.empty:
            raise RuntimeError(f"Data pipeline failure for {market_name}")

        # 2. Build signal generator
        logger.info("Generating signal cache (vectorized)...")
        signal_generator = AlphaSignal.from_xgboost_kmeans(
            df_monthly,
            xgb_model,
            kmeans_model,
            FEATURE_COLS,
        )

        # 3. Get latest monthly snapshot
        last_date = df_monthly.index.get_level_values("date").max()
        today_data = df_monthly.xs(last_date, level="date").copy()

        today_signals = signal_generator.get_signal(last_date)
        today_data["proba_upside"] = today_signals["proba_upside"]
        today_data["cluster"] = today_signals["cluster"]

        # 4. Current live allocation
        final_alloc = build_topk_allocation(today_data)

        # 5. Backtest TOP-K only
        logger.info("Running Top-k backtest...")
        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily=df_daily,
            signal_generator=signal_generator,
            top_k=TOP_K,
            target_cluster=TARGET_CLUSTER,
            proba_threshold=PROBA_THRESHOLD,
        )

        # 6. Export dataframe for dashboard
        export_df = build_export_df(today_data, final_alloc)

        # 7. Upload one official output set
        suffix = config_file.replace(".json", "")
        upload_to_hub(export_df, f"latest_signals_{suffix}")
        upload_to_hub(hist_df, f"portfolio_history_{suffix}")
        upload_to_hub(rebal_df, f"rebalance_history_{suffix}")

        # 8. Local metadata
        metadata = {
            "market_name": market_name,
            "strategy_method": "topk",
            "top_k": TOP_K,
            "target_cluster": TARGET_CLUSTER,
            "proba_threshold": PROBA_THRESHOLD,
            "last_update": datetime.now().isoformat(),
            "current_allocation": final_alloc,
            "metrics": {
                "final_value": float(hist_df["Strategy"].iloc[-1]) if not hist_df.empty else 0.0
            },
        }

        with open(BASE_DIR / f"metadata_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"SUCCESS: {market_name} processed in {duration:.1f}s")

    except Exception as e:
        logger.error(f"FAILURE for {market_name}: {e}", exc_info=True)


# =============================================================================
# 5. GLOBAL EXECUTION
# =============================================================================

def run_all_pipelines() -> None:
    if not CONFIG_DIR.exists():
        logger.error(f"Config directory not found at {CONFIG_DIR}")
        return

    config_files = sorted([f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")])

    if not config_files:
        logger.warning("No JSON config files found.")
        return

    logger.info(f"Found {len(config_files)} markets to process.")
    for config_file in config_files:
        run_pipeline_for_config(config_file)


if __name__ == "__main__":
    run_all_pipelines()