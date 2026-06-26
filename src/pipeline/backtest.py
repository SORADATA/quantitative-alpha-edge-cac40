from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# On a retiré FEATURE_COLS d'ici !
from const import (
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE,
    TRANSACTION_COST,
    MIN_STOCKS_OPTIM,
    MAX_STOCKS_SELECT,
    PROBA_MIN,
    WEIGHT_BOUNDS
)
from src.utils.logger import setup_logger
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")


def get_optimal_weights(prices_df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Tuple[Dict[str, float], str]:
    if prices_df.shape[1] < MIN_STOCKS_OPTIM:
        n = prices_df.shape[1]
        return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"

    try:
        mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=risk_free_rate)

        return dict(ef.clean_weights()), "max_sharpe"

    except Exception as e1:
        logger.warning(f"Max Sharpe failed ({e1}) -> min_volatility fallback.")
        try:
            mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
            S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

            ef2 = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
            ef2.min_volatility()

            return dict(ef2.clean_weights()), "min_vol"

        except Exception as e2:
            logger.warning(f"Min vol failed ({e2}) -> equal_weight fallback.")
            n = prices_df.shape[1]
            return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"


# ---- LA FONCTION CORRIGÉE EST ICI ----
def _generate_monthly_signals(month_data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Génère les signaux ML pour un mois donné.
    Le modèle AlphaEdgeEnsemble extrait lui-même les features dont il a besoin.
    """
    if month_data.empty:
        return pd.DataFrame()

    try:
        month_data = month_data.copy()
        # On passe directement les données brutes, le modèle sait quelles colonnes utiliser !
        month_data["proba_upside"] = model.predict_proba(month_data)[:, 1]
    except Exception as e:
        logger.error(f"predict_proba failed: {e}")
        return pd.DataFrame()

    return month_data
# --------------------------------------


def _select_tickers(month_data: pd.DataFrame, proba_min: float = PROBA_MIN, max_stocks: int = MAX_STOCKS_SELECT) -> List[str]:
    if "proba_upside" not in month_data.columns:
        return []

    selected = month_data[month_data["proba_upside"] >= proba_min]

    if selected.empty:
        logger.info("Alerte Marché : Aucun signal au-dessus du seuil. Passage en 100% Cash.")
        return []

    return selected.nlargest(max_stocks, "proba_upside").index.tolist()


def _compute_turnover(new_alloc: Dict[str, float], old_alloc: Dict[str, float]) -> float:
    all_tickers = set(new_alloc) | set(old_alloc)
    return sum(abs(new_alloc.get(t, 0.0) - old_alloc.get(t, 0.0)) for t in all_tickers) / 2.0


def _simulate_daily_returns(
    allocation: Dict[str, float],
    drifted_allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[list, float, float, Dict[str, float]]:
    records = []
    is_first_day = True
    stock_values = {t: portfolio_value * w for t, w in allocation.items()} if allocation else {}

    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        if is_first_day:
            turnover = _compute_turnover(allocation, drifted_allocation)
            transaction_fees = portfolio_value * turnover * TRANSACTION_COST
            portfolio_value -= transaction_fees
            stock_values = {t: portfolio_value * w for t, w in allocation.items()}
            is_first_day = False

        daily_portfolio_pnl = 0.0
        if stock_values:
            for t in list(stock_values.keys()):
                ret = daily_returns.loc[date, t] if (t in daily_returns.columns and date in daily_returns.index) else 0.0
                pnl = stock_values[t] * ret
                stock_values[t] += pnl
                daily_portfolio_pnl += pnl

        portfolio_value += daily_portfolio_pnl
        benchmark_value *= (1 + bench_ret)

        records.append({
            "Date":      date,
            "Strategy":  portfolio_value,
            "Benchmark": benchmark_value,
            "N_Stocks":  len(stock_values),
        })

    new_drifted_allocation = {}
    if portfolio_value > 0 and stock_values:
        new_drifted_allocation = {t: val / portfolio_value for t, val in stock_values.items()}

    return records, portfolio_value, benchmark_value, new_drifted_allocation


def compute_performance_metrics(results_df: pd.DataFrame, rebalance_log: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, float]:
    strat = results_df["Strategy"]
    bench = results_df["Benchmark"]

    strat_ret = strat.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()

    n_years = len(strat_ret) / TRADING_DAYS_YEAR
    cagr = (strat.iloc[-1] / strat.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    vol = strat_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    excess = strat_ret - risk_free_rate / TRADING_DAYS_YEAR
    sharpe = (excess.mean() / strat_ret.std()) * np.sqrt(TRADING_DAYS_YEAR) if strat_ret.std() != 0 else 0.0

    downside = strat_ret[strat_ret < 0].std() * np.sqrt(TRADING_DAYS_YEAR)
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else np.nan

    rolling_max = strat.cummax()
    drawdown = (strat - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    align_df = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    if len(align_df) > 1:
        cov_matrix = np.cov(align_df.iloc[:, 0], align_df.iloc[:, 1])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    else:
        beta = np.nan

    alpha = (cagr - risk_free_rate) - beta * (
        (bench.iloc[-1] / bench.iloc[0]) ** (1 / n_years) - 1 - risk_free_rate
    )

    monthly_strat = strat.resample("BME").last().pct_change().dropna()
    hit_rate = (monthly_strat > 0).mean()

    avg_stocks = rebalance_log["N_Stocks"].mean() if "N_Stocks" in rebalance_log.columns else np.nan

    metrics = {
        "CAGR":          round(cagr, 4),
        "Volatility":    round(vol, 4),
        "Sharpe":        round(sharpe, 4),
        "Sortino":       round(sortino, 4),
        "Calmar":        round(calmar, 4),
        "Max_Drawdown":  round(max_dd, 4),
        "Alpha":         round(alpha, 4),
        "Beta":          round(beta, 4),
        "Hit_Rate":      round(hit_rate, 4),
        "Avg_N_Stocks":  round(avg_stocks, 1),
        "Final_Value":   round(strat.iloc[-1], 2),
    }

    logger.info("=" * 50)
    logger.info("📊 PERFORMANCE METRICS")
    logger.info(f"   CAGR          : {cagr:.2%}")
    logger.info(f"   Sharpe Ratio  : {sharpe:.3f}")
    logger.info(f"   Sortino Ratio : {sortino:.3f}")
    logger.info(f"   Calmar Ratio  : {calmar:.3f}")
    logger.info(f"   Max Drawdown  : {max_dd:.2%}")
    logger.info(f"   Alpha         : {alpha:.2%}")
    logger.info(f"   Beta          : {beta:.3f}")
    logger.info(f"   Hit Rate      : {hit_rate:.1%}")
    logger.info("=" * 50)

    return metrics


def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    model: Any,
    benchmark_ticker: str = "^FCHI",
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    logger.info(f"Starting backtest | Benchmark: {benchmark_ticker}")

    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(benchmark_ticker, date_min, date_max, daily_prices.index)

    portfolio_value = 100.0
    benchmark_value = 100.0
    all_records = []
    rebalance_log = []
    drifted_allocation: Dict[str, float] = {} 
    monthly_dates = df_monthly.index.get_level_values("date").unique().sort_values()

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = _generate_monthly_signals(
            df_monthly.xs(month_date, level="date").copy(),
            model,
        )

        allocation: Dict[str, float] = {}
        optim_method = "no_signal"

        if not month_data.empty:
            tickers = _select_tickers(month_data, proba_min, max_stocks)

            if tickers:
                available_tickers = [t for t in tickers if t in daily_prices.columns]
                prices_subset = (
                    daily_prices[available_tickers]
                    .loc[:month_date]
                    .iloc[-TRADING_DAYS_YEAR:]
                    .dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
                )

                if not prices_subset.empty:
                    weights, optim_method = get_optimal_weights(prices_subset)
                    allocation = {t: w for t, w in weights.items() if w > 1e-4}

        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) &
            (daily_prices.index < next_month)
        ]

        day_records, portfolio_value, benchmark_value, drifted_allocation = _simulate_daily_returns(
            allocation, drifted_allocation,
            trading_days, daily_returns,
            benchmark_returns, portfolio_value, benchmark_value,
        )
        all_records.extend(day_records)

        rebalance_log.append({
            "Date":         month_date,
            "N_Stocks":     len(allocation),
            "Optim_Method": optim_method,
            "Allocation":   allocation,
            "Top_Ticker":   max(allocation, key=allocation.get) if allocation else None,
        })

    results_df = pd.DataFrame(all_records).set_index("Date")
    rebalance_df = pd.DataFrame(rebalance_log).set_index("Date")

    metrics = compute_performance_metrics(results_df, rebalance_df)

    logger.info(f" Backtest complete | Final value: {portfolio_value:.2f}")

    return results_df, rebalance_df, metrics


if __name__ == "__main__":
    import json
    import pandas as pd
    from const import CONFIG_DIR, DATA_DIR, MODEL_DIR
    from src.models.ensemble import AlphaEdgeEnsemble

    config_path = CONFIG_DIR / "markets" / "cac40.json"
    market = json.load(open(config_path))["market_name"] if config_path.exists() else "CAC40"

    logger.info(f"🚀 Démarrage de la simulation sur le {market}...")

    model_path = MODEL_DIR / "ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Aucun modèle trouvé. Lance 'python -m src.models.train' en premier.")
    model = AlphaEdgeEnsemble.load(model_path)

    # Chargement du fichier qui DOIT déjà contenir les features
    logger.info("Chargement des données mensuelles...")
    df_monthly = pd.read_parquet(DATA_DIR / "processed" / market / "monthly_features.parquet")
    
    # ⚠️ Vérification critique des features
    missing_features = [f for f in model.features_ if f not in df_monthly.columns]
    if missing_features:
        logger.error(f"FATAL: Il manque {len(missing_features)} features dans monthly_features.parquet.")
        logger.error(f"Exemples de features manquantes : {missing_features[:5]}")
        logger.error("Veuillez vérifier que l'ETL génère bien TOUTES les features utilisées lors de l'entraînement.")
        exit(1) # On arrête l'exécution si les données ne correspondent pas

    df_daily = pd.read_parquet(DATA_DIR / "processed" / market / "daily_raw.parquet")

    results_df, rebalance_df, metrics = backtest_strategy_with_rebalancing(
        df_daily=df_daily,
        df_monthly=df_monthly, 
        model=model,
        benchmark_ticker="^FCHI"
    )

    results_df.to_csv(DATA_DIR / "backtest_results_daily.csv")
    rebalance_df.to_csv(DATA_DIR / "backtest_rebalance_log.csv")
    logger.info("✅ Fichiers d'analyse CSV sauvegardés avec succès !")