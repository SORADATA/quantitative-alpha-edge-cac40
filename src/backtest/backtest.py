from typing import Dict, Tuple
import numpy as np
import pandas as pd

from const import TRADING_DAYS_YEAR
from src.utils.logger import setup_logger
from src.utils.config_loader import BENCHMARK_TICKER
from src.utils.market_utils import get_benchmark_returns


logger = setup_logger("backtest")


# =============================================================================
# COSTS / TURNOVER
# =============================================================================

def calculate_turnover_friction(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    fee_bps: float = 0.0020,
) -> Tuple[float, float]:
    total_turnover = 0.0
    all_tickers = set(old_weights.keys()).union(set(new_weights.keys()))

    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_turnover += abs(new_w - old_w)

    true_turnover = total_turnover / 2.0
    friction_cost_pct = true_turnover * fee_bps
    return friction_cost_pct, true_turnover


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def get_topk_equal_weights(
    selected: pd.DataFrame,
    top_k: int = 5,
    score_col: str = "proba_upside",
) -> Dict[str, float]:
    if selected.empty:
        return {}

    top_selected = selected.sort_values(score_col, ascending=False).head(top_k)
    n = len(top_selected)

    if n == 0:
        return {}

    w = 1.0 / n
    return {ticker: w for ticker in top_selected.index.tolist()}


# =============================================================================
# DAILY PATH SIMULATION
# =============================================================================

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
                daily_returns.loc[date, ticker]
                if ticker in daily_returns.columns and date in daily_returns.index
                else 0.0
                for ticker in allocation
            ]
            strat_ret = float(np.average(pd.Series(rets).fillna(0.0), weights=weights))

        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)

        records.append(
            {
                "Date": date,
                "Strategy": portfolio_value,
                "Benchmark": benchmark_value,
            }
        )

    return records, portfolio_value, benchmark_value


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    signal_generator,
    top_k: int = 5,
    target_cluster: int = 1,
    proba_threshold: float = 0.55,
    fee_bps: float = 0.0020,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("Starting backtest | method=topk")

    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0.0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        BENCHMARK_TICKER,
        date_min,
        date_max,
        daily_prices.index,
    )

    portfolio_value = 100.0
    benchmark_value = 100.0
    all_records = []
    rebalance_log = []

    monthly_dates = (
        signal_generator.signal_cache.index.get_level_values("date")
        .unique()
        .sort_values()
    )

    previous_allocation: Dict[str, float] = {}

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = signal_generator.get_signal(month_date)

        allocation: Dict[str, float] = {}
        selected_count = 0

        if not month_data.empty:
            selected = month_data[
                (month_data["cluster"] == target_cluster) &
                (month_data["proba_upside"] >= proba_threshold)
            ].copy()

            selected_count = len(selected)

            if not selected.empty:
                allocation = get_topk_equal_weights(
                    selected=selected,
                    top_k=top_k,
                    score_col="proba_upside",
                )

        cost_pct, turnover = calculate_turnover_friction(
            previous_allocation,
            allocation,
            fee_bps=fee_bps,
        )

        portfolio_value *= (1 - cost_pct)
        previous_allocation = allocation.copy()

        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) & (daily_prices.index < next_month)
        ]

        day_records, portfolio_value, benchmark_value = _simulate_daily_returns(
            allocation=allocation,
            trading_days=trading_days,
            daily_returns=daily_returns,
            benchmark_returns=benchmark_returns,
            portfolio_value=portfolio_value,
            benchmark_value=benchmark_value,
        )
        all_records.extend(day_records)

        rebalance_log.append(
            {
                "Date": month_date,
                "Method": "topk",
                "Selected_Count": selected_count,
                "N_Stocks": len(allocation),
                "Turnover": turnover,
                "Cost_Pct": cost_pct,
                "Allocation": allocation,
            }
        )

    logger.info(f"Backtest complete. Final value: {portfolio_value:.2f}")

    hist_df = pd.DataFrame(all_records).set_index("Date") if all_records else pd.DataFrame(columns=["Strategy", "Benchmark"])
    rebal_df = pd.DataFrame(rebalance_log).set_index("Date") if rebalance_log else pd.DataFrame(
        columns=["Method", "Selected_Count", "N_Stocks", "Turnover", "Cost_Pct", "Allocation"]
    )

    return hist_df, rebal_df
