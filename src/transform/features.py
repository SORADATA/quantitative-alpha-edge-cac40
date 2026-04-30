import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


from const import (
    RSI_WINDOW,
    BB_WINDOW,
    BB_STD,
    MIN_HISTORY_TA,
    MIN_HISTORY_FF,
    MOMENTUM_LAGS,
    WINSOR_CUTOFF,
    VARS_TO_LAG,
    FAMA_FRENCH_FACTORS,
)
from src.utils.feature_utils import compute_atr, compute_macd
from src.utils.logger import setup_logger

logger = setup_logger("features")


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Garman-Klass volatility, RSI, Bollinger Bands, ATR, MACD,
    and euro volume to a multi-index (date, ticker) DataFrame.
    """
    logger.info("Computing technical indicators...")

    df["garman_klass_vol"] = (
        (np.log(df["high"]) - np.log(df["low"])) ** 2 / 2
        - (2 * np.log(2) - 1)
        * (np.log(df["adj close"]) - np.log(df["open"])) ** 2
    )

    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close = df.loc[idx, "adj close"]

        if len(close) > MIN_HISTORY_TA:
            df.loc[idx, "rsi"] = RSIIndicator(
                close=close, window=RSI_WINDOW
            ).rsi().values

            bb = BollingerBands(
                close=np.log1p(close), window=BB_WINDOW, window_dev=BB_STD
            )
            df.loc[idx, "bb_low"] = bb.bollinger_lband().values
            df.loc[idx, "bb_mid"] = bb.bollinger_mavg().values
            df.loc[idx, "bb_high"] = bb.bollinger_hband().values

    df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    df["macd"] = df.groupby(level=1, group_keys=False).apply(compute_macd)
    df["euro_volume"] = (df["adj close"] * df["volume"]) / 1e6

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds winsorized momentum return columns for each lag in MOMENTUM_LAGS.
    Operates on a single-ticker slice (called via groupby).
    """
    for lag in MOMENTUM_LAGS:
        raw = df["adj close"].pct_change(lag)
        lower = raw.expanding(min_periods=12).quantile(WINSOR_CUTOFF)
        upper = raw.expanding(min_periods=12).quantile(1 - WINSOR_CUTOFF)
        df[f"return_{lag}m"] = raw.clip(lower=lower, upper=upper)
    return df


def get_fama_french_betas(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches Europe 5-Factor data from Kenneth French's library and computes
    rolling 24-month OLS betas for each ticker. Fills with zeros on failure.
    """
    logger.info("Retrieving Fama-French factors (Europe 5)...")

    try:
        import pandas_datareader.data as web
    except ImportError as exc:
        logger.error(f"pandas_datareader unavailable ({exc}). Filling betas with zeros.")
        return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})

    try:
        factor_data = (
            web.DataReader("Europe_5_Factors", "famafrench", start="2010")[0]
            .drop("RF", axis=1)
        )
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample("BM").last().div(100)
        factor_data.index.name = "date"

        if "return_1m" not in data.columns:
            return data

        betas_list = []
        for ticker in data.index.get_level_values(1).unique():
            y = data.xs(ticker, level=1).get("return_1m")
            if y is None or y.dropna().empty:
                continue

            X = factor_data.loc[factor_data.index.intersection(y.index)]
            y = y.loc[X.index]

            if len(y) <= MIN_HISTORY_FF:
                continue

            params = (
                RollingOLS(y, sm.add_constant(X[FAMA_FRENCH_FACTORS]), window=MIN_HISTORY_FF)
                .fit()
                .params.drop("const", axis=1)
            )
            params["ticker"] = ticker
            betas_list.append(params)

        if not betas_list:
            return data

        betas_df = pd.concat(betas_list).set_index("ticker", append=True)
        data = data.join(betas_df.groupby("ticker").shift())
        data.loc[:, FAMA_FRENCH_FACTORS] = (
            data.groupby("ticker", group_keys=False)[FAMA_FRENCH_FACTORS]
            .apply(lambda x: x.fillna(x.mean()))
        )
        return data

    except Exception as exc:
        logger.error(f"Fama-French retrieval failed ({exc}). Filling with zeros.")
        return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})
