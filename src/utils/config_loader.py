import json
from typing import Dict, Any
from const import CONFIG_FILE
from src.utils.logger import setup_logger

logger = setup_logger("ConfigLoader")


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
