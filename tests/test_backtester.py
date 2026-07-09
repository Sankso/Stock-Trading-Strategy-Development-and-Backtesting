"""
Unit tests for Backtester and performance metric functions.
"""
import numpy as np
import pandas as pd
import pytest
from src.backtester import (
    Backtester,
    compute_returns,
    cumulative_return,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
)


@pytest.fixture
def signal_df():
    """Synthetic price + manually crafted signals for deterministic trade testing.
    Signals are designed so every BUY is matched by a subsequent SELL.
    """
    close   = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
    signals = np.array([0,   1,   0,   0,  -1,   0,   1,   0,  -1,   0], dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "signal": signals},
        index=pd.date_range("2024-01-01", periods=10),
    )


@pytest.fixture
def backtest_result(signal_df):
    return Backtester(signal_df, initial_capital=1000).run()


# ── Trade execution ──────────────────────────────────────────────────────── #

def test_trade_log_not_empty(backtest_result):
    _, trade_log = backtest_result
    assert not trade_log.empty


def test_buy_sell_pairs(backtest_result):
    """Every BUY must be followed by a SELL (balanced ledger)."""
    _, trade_log = backtest_result
    assert (trade_log["action"] == "BUY").sum() == (trade_log["action"] == "SELL").sum()


def test_no_double_buy(backtest_result):
    """Should never have two consecutive BUY actions."""
    _, trade_log = backtest_result
    actions = trade_log["action"].tolist()
    for i in range(len(actions) - 1):
        assert not (actions[i] == "BUY" and actions[i + 1] == "BUY")


# ── Portfolio value ──────────────────────────────────────────────────────── #

def test_portfolio_always_positive(backtest_result):
    portfolio, _ = backtest_result
    assert (portfolio["value"] > 0).all()


# ── Metrics ──────────────────────────────────────────────────────────────── #

def test_sharpe_is_finite_for_nonzero_vol():
    """Sharpe should be finite when returns have variance."""
    returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.012])
    assert np.isfinite(sharpe_ratio(returns))


def test_sharpe_nan_for_zero_vol():
    """Sharpe should be NaN when returns are constant (zero volatility)."""
    returns = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns))
