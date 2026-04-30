import pandas as pd
from typing import Tuple, List, Dict

from src.utils.logger import setup_logger
# INITIALIZATION
logger = setup_logger("DataValidation")


def handle_ticker_changes() -> Tuple[Dict[str, str], List[str]]:
    """Defines known ticker changes and delistings."""
    TICKER_CHANGES = {}    # e.g., {'OLD': 'NEW'}
    DELISTED_TICKERS = []  # e.g., ['BANKRUPT.PA']
    return TICKER_CHANGES, DELISTED_TICKERS


def validate_and_clean_tickers(df: pd.DataFrame, tickers_list: List[str], max_days_stale: int = 30) -> Tuple[pd.DataFrame, List[str], Dict[str, List]]:
    """Validates data quality, removing stale or invalid tickers (Optimized Version)."""
    logger.info("Validating ticker data integrity...")
    alerts = {'delisted': [], 'stale': [], 'missing': [], 'warnings': []}
    tickers_in_data = df.index.get_level_values('ticker').unique()
    missing_tickers = list(set(tickers_list) - set(tickers_in_data))
    if missing_tickers:
        alerts['missing'] = missing_tickers
        logger.warning(
            f"{len(missing_tickers)} tickers missing from download: {missing_tickers[:5]}..."
            )
    last_global_date = df.index.get_level_values('date').max()
    last_dates_per_ticker = df.reset_index().groupby('ticker')['date'].max()
    stale_mask = (last_global_date - last_dates_per_ticker).dt.days > max_days_stale
    stale_tickers_series = last_dates_per_ticker[stale_mask]
    if not stale_tickers_series.empty:
        alerts['stale'] = [
            {'ticker': t, 'days_stale': (last_global_date - d).days}
            for t, d in stale_tickers_series.items()
        ]
        tickers_to_remove = stale_tickers_series.index.tolist()
        logger.warning(
            f"{len(tickers_to_remove)} stale tickers detected (> {max_days_stale} days)."
            )
        alerts['delisted'] = tickers_to_remove
        df = df[~df.index.get_level_values('ticker').isin(tickers_to_remove)]
    valid_tickers = df.index.get_level_values('ticker').unique().tolist()
    logger.info(f"Valid Tickers: {len(valid_tickers)} / {len(tickers_list)}")
    return df, valid_tickers, alerts

