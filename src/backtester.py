import numpy as np
import pandas as pd


class Backtester:
    """
    Simulates trade execution against a signal column.
    Tracks cash, position, and logs every order as a structured ledger.
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10_000):
        self.df              = df.copy()
        self.initial_capital = initial_capital

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            portfolio : daily portfolio value timeseries
            trade_log : structured ledger of every executed order
        """
        cash      = self.initial_capital
        position  = 0        # shares held
        trades    = []
        portfolio = []

        close  = self.df["close"]
        signal = self.df["signal"].shift(1).fillna(0)  # act on next-day open

        for date, price, sig in zip(self.df.index, close, signal):
            # Execute signal
            if sig == 1 and position == 0:          # BUY — go all-in
                position = cash / price
                cash     = 0.0
                trades.append({"date": date, "action": "BUY",
                                "price": price, "shares": position})

            elif sig == -1 and position > 0:        # SELL — liquidate
                cash     = position * price
                trades.append({"date": date, "action": "SELL",
                                "price": price, "shares": position,
                                "pnl": cash - self.initial_capital})
                position = 0.0

            portfolio.append({"date": date, "value": cash + position * price})

        portfolio_df = pd.DataFrame(portfolio).set_index("date")
        trade_log    = pd.DataFrame(trades)
        return portfolio_df, trade_log


# ── Performance metrics — isolated, stateless functions ───────────────────── #

def compute_returns(portfolio: pd.DataFrame) -> pd.Series:
    return portfolio["value"].pct_change().dropna()


def cumulative_return(portfolio: pd.DataFrame, initial_capital: float) -> float:
    return round(portfolio["value"].iloc[-1] - initial_capital, 2)


def annualised_return(returns: pd.Series) -> float:
    return round(returns.mean() * 252 * 100, 2)


def annualised_volatility(returns: pd.Series) -> float:
    return round(returns.std() * (252 ** 0.5) * 100, 2)


def sharpe_ratio(returns: pd.Series) -> float:
    ann_ret = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    return round(ann_ret / ann_vol, 2) if ann_vol else float("nan")
