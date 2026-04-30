from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

from const import (
    TARGET_CLUSTER,
    PROBA_THRESHOLD,
    FEATURE_COLS,
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE
)
from src.utils.logger import setup_logger
from src.utils.config_loader import BENCHMARK_TICKER
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")


def get_optimal_weights(prices_df: pd.DataFrame) -> Tuple[Dict[str, float], bool]:
    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_YEAR)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.25))
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        
        return dict(ef.clean_weights()), True
    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        return {}, False


def _generate_monthly_signals(
    month_data: pd.DataFrame,
    xgb_model: Any,
    kmeans_model: Any,
) -> pd.DataFrame:
    if "rsi" in month_data.columns:
        month_data = month_data.copy()
        month_data["cluster"] = kmeans_model.predict(month_data[["rsi"]].fillna(50))

    if not all(c in month_data.columns for c in FEATURE_COLS):
        return pd.DataFrame()

    month_data["proba_upside"] = xgb_model.predict_proba(month_data[FEATURE_COLS].fillna(0))[:, 1]
    return month_data


def _simulate_daily_returns(
    allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[list, float, float]:
    records = []
    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        strat_ret = 0.0

        if allocation:
            weights = list(allocation.values())
            rets = [
                daily_returns.loc[date, t] if t in daily_returns.columns and date in daily_returns.index else 0.0
                for t in allocation
            ]
            strat_ret = float(np.average(pd.Series(rets).fillna(0), weights=weights))

        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)
        records.append({"Date": date, "Strategy": portfolio_value, "Benchmark": benchmark_value})

    return records, portfolio_value, benchmark_value


def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    xgb_model: Any,
    kmeans_model: Any,
    get_optimal_weights_fn: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Starting backtest...")

    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        BENCHMARK_TICKER, date_min, date_max, daily_prices.index
    )

    portfolio_value, benchmark_value = 100.0, 100.0
    all_records, rebalance_log = [], []
    monthly_dates = df_monthly.index.get_level_values("date").unique().sort_values()

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = _generate_monthly_signals(
            df_monthly.xs(month_date, level="date").copy(),
            xgb_model,
            kmeans_model,
        )

        allocation: Dict[str, float] = {}
        if not month_data.empty:
            selected = month_data[
                (month_data["cluster"] == TARGET_CLUSTER) & 
                (month_data["proba_upside"] > PROBA_THRESHOLD)
            ]

            if not selected.empty:
                tickers = selected.index.tolist()
                prices_subset = daily_prices[tickers].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1)

                if not prices_subset.empty and len(prices_subset.columns) >= 3:
                    weights, success = get_optimal_weights_fn(prices_subset)
                    allocation = weights if success else {t: 1.0 / len(tickers) for t in tickers}

        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[(daily_prices.index >= month_date) & (daily_prices.index < next_month)]

        day_records, portfolio_value, benchmark_value = _simulate_daily_returns(
            allocation, trading_days, daily_returns,
            benchmark_returns, portfolio_value, benchmark_value,
        )
        all_records.extend(day_records)
        rebalance_log.append({"Date": month_date, "N_Stocks": len(allocation), "Allocation": allocation})

    logger.info(f"Backtest complete. Final value: {portfolio_value:.2f}")
    return pd.DataFrame(all_records).set_index("Date"), pd.DataFrame(rebalance_log).set_index("Date")