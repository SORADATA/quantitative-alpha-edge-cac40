import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import MACD

from const import ATR_WINDOW, MACD_SLOW, MACD_FAST, MACD_SIGN, MIN_HISTORY_TA


def _safe_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalization; returns NaN series when std is zero."""
    std = series.std()
    if std == 0:
        return pd.Series(np.nan, index=series.index)
    return series.sub(series.mean()).div(std)


def compute_atr(stock_data: pd.DataFrame) -> pd.Series:
    """
    Normalized Average True Range for a single ticker slice.
    Expects columns: high, low, close.
    Returns a z-scored ATR series aligned to stock_data.index.
    """
    if len(stock_data) < ATR_WINDOW + 1:
        return pd.Series(np.nan, index=stock_data.index)

    atr = AverageTrueRange(
        high=stock_data["high"],
        low=stock_data["low"],
        close=stock_data["close"],
        window=ATR_WINDOW,
    ).average_true_range()

    return _safe_normalize(atr)


def compute_macd(stock_data: pd.DataFrame) -> pd.Series:
    """
    Normalized MACD line for a single ticker slice.
    Expects column: adj close.
    Returns a z-scored MACD series aligned to stock_data.index.
    """
    if len(stock_data) < MACD_SLOW + MIN_HISTORY_TA:
        return pd.Series(np.nan, index=stock_data.index)

    macd_val = MACD(
        close=stock_data["adj close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGN,
    ).macd()

    return _safe_normalize(macd_val)