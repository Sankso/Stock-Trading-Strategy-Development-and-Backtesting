import os
import yaml
from src.logger import get_logger
from src.data.ingestion import DataIngestionEngine, DataIngestionError
from src.plotting import plot_all
from src.strategies.technical import TechnicalIndicatorStrategy
from src.backtester import (
    Backtester,
    compute_returns,
    cumulative_return,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
)

logger = get_logger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config()
    logger.info("Config loaded. Symbol=%s", cfg["trading"]["symbol"])

    # ── 1. Fetch data ─────────────────────────────────────────────────────── #
    engine = DataIngestionEngine(
        api_key     = cfg["api"]["key"],
        timeout     = cfg["api"]["timeout"],
        max_retries = cfg["api"]["max_retries"],
    )
    try:
        df = engine.fetch(cfg["trading"]["symbol"])
        logger.info("Fetched %d rows of market data.", len(df))
    except DataIngestionError as e:
        logger.error("Data ingestion failed: %s", e)
        raise SystemExit(1)

    # ── 2. Compute indicators & signals ───────────────────────────────────── #
    ind = cfg["indicators"]
    df  = TechnicalIndicatorStrategy(
        df,
        rsi_period = ind["rsi_period"],
        bb_period  = ind["bb_period"],
        bb_std     = ind["bb_std"],
        adx_period = ind["adx_period"],
    ).run()
    logger.info("Signals generated. BUY=%d  SELL=%d",
                int((df["signal"] == 1).sum()),
                int((df["signal"] == -1).sum()))

    # ── 3. Backtest ───────────────────────────────────────────────────────── #
    initial_capital = cfg["trading"]["initial_capital"]
    portfolio, trade_log = Backtester(df, initial_capital).run()
    logger.info("Backtest complete. Trades executed=%d", len(trade_log))

    # ── 4. Results ────────────────────────────────────────────────────────── #
    returns = compute_returns(portfolio)

    print("\n=== Trade Ledger ===")
    print(trade_log.to_string(index=False) if not trade_log.empty else "No trades executed.")

    print("\n=== Performance Metrics ===")
    print(f"Total Trades      : {len(trade_log)}")
    print(f"Cumulative P&L    : ${cumulative_return(portfolio, initial_capital)}")
    print(f"Annual Return     : {annualised_return(returns)}%")
    print(f"Annual Volatility : {annualised_volatility(returns)}%")
    print(f"Sharpe Ratio      : {sharpe_ratio(returns)}")

    # ── 5. Output Generation ──────────────────────────────────────────────── #
    plot_all(df, portfolio, initial_capital)
    
    # Export to CSV for Power BI
    os.makedirs("exports", exist_ok=True)
    
    # 1. Portfolio over time
    portfolio.to_csv("exports/portfolio.csv")
    
    # 2. Trade Ledger
    trade_log.to_csv("exports/trade_log.csv", index=False)
    
    # 3. Market Data & Indicators (so Power BI can plot price/RSI/ADX)
    df.to_csv("exports/market_data.csv")
    
    # 4. Final Performance Metrics
    import pandas as pd
    metrics_df = pd.DataFrame([{
        "Total Trades": len(trade_log),
        "Cumulative P&L": cumulative_return(portfolio, initial_capital),
        "Annual Return %": annualised_return(returns),
        "Annual Volatility %": annualised_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns)
    }])
    metrics_df.to_csv("exports/metrics.csv", index=False)
    
    logger.info("Exported portfolio, trade_log, market_data, and metrics to 'exports/' for Power BI.")
