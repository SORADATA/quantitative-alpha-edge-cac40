import pandas as pd
import yfinance as yf


from src.utils.logger import setup_logger

logger = setup_logger("market_utils")


def get_benchmark_returns(
    benchmark_ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reindex_to: pd.DatetimeIndex,
) -> pd.Series:
    """
    Downloads benchmark daily returns and reindexes to the strategy calendar.
    Falls back to the mean of reindex_to index returns on failure.
    """
    try:
        raw = yf.download(
            benchmark_ticker,
            start=start,
            end=end + pd.DateOffset(days=1),
            progress=False,
            auto_adjust=False,
        )

        if raw.empty:
            logger.error(f"No data found for benchmark {benchmark_ticker}")
            return pd.Series(0.0, index=reindex_to)

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].iloc[:, 0]
        else:
            prices = raw["Close"]

        prices.index = prices.index.tz_localize(None)
        reindex_to = reindex_to.tz_localize(None)

        bench_returns = (
            prices.pct_change()
            .reindex(reindex_to, method="ffill")
            .fillna(0)
        )
        
        return bench_returns

    except Exception as e:
        logger.error(f"Benchmark processing failed: {e}")
        return pd.Series(0.0, index=reindex_to)


def build_export_df(
    today_data: pd.DataFrame,
    final_alloc: dict,
) -> pd.DataFrame:
    """
    Formats the daily signal snapshot into a clean export-ready DataFrame.
    Columns: Ticker, Cluster, Proba_Hausse (%), RSI, Return_3M, Allocation, Signal.
    """
    export = (
        today_data[["cluster", "proba_upside", "rsi", "return_3m"]]
        .reset_index()
        .rename(
            columns={
                "ticker":       "Ticker",
                "cluster":      "Cluster",
                "proba_upside": "Proba_Hausse",
                "rsi":          "RSI",
                "return_3m":    "Return_3M",
            }
        )
    )
    export["Proba_Hausse"] = (export["Proba_Hausse"] * 100).round(2)
    export["Allocation"] = export["Ticker"].map(final_alloc).fillna(0.0)
    export["Signal"] = export["Allocation"].apply(
        lambda w: "BUY" if w > 0 else "NEUTRAL"
    )
    return export
