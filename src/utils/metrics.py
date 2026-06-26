import numpy as np
import pandas as pd


def calculate_financial_metrics(df_test: pd.DataFrame, probas: np.ndarray, threshold: float = 0.5) -> dict:
    signals = (probas > threshold).astype(int)
    strategy_returns = signals * df_test["future_return"]
    portfolio_returns = strategy_returns.groupby(level="date").mean()
    if portfolio_returns.std() == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0}
    annualization_factor = np.sqrt(12)
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * annualization_factor
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    total_return = cumulative_returns.iloc[-1] - 1 if not cumulative_returns.empty else 0.0
    return {
        "sharpe": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return, 4)
    }
