"""
Unit tests for TechnicalIndicatorStrategy.
Uses synthetic price data — no API calls required.
"""
import numpy as np
import pandas as pd
import pytest
from src.strategies.technical import TechnicalIndicatorStrategy


@pytest.fixture
def sample_df():
    """100 bars of synthetic OHLCV data."""
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, 100),
    }, index=pd.date_range("2023-01-01", periods=100))
    return df


@pytest.fixture
def strategy(sample_df):
    s = TechnicalIndicatorStrategy(sample_df)
    s.compute_indicators()
    return s


# ── RSI ─────────────────────────────────────────────────────────────────── #

def test_rsi_range(strategy):
    """RSI must always be between 0 and 100."""
    rsi = strategy.df["RSI"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_rsi_length(strategy):
    """RSI should produce NaNs only for the warm-up period."""
    assert strategy.df["RSI"].notna().sum() == 100 - strategy.rsi_period


# ── Bollinger Bands ──────────────────────────────────────────────────────── #

def test_bb_upper_above_lower(strategy):
    """Upper band must always be >= lower band."""
    df = strategy.df.dropna(subset=["BB_upper", "BB_lower"])
    assert (df["BB_upper"] >= df["BB_lower"]).all()


def test_bb_ma_between_bands(strategy):
    """Moving average must sit between the two bands."""
    df = strategy.df.dropna(subset=["BB_MA", "BB_upper", "BB_lower"])
    assert (df["BB_MA"] <= df["BB_upper"]).all()
    assert (df["BB_MA"] >= df["BB_lower"]).all()


# ── ADX ─────────────────────────────────────────────────────────────────── #

def test_adx_non_negative(strategy):
    """ADX is always non-negative."""
    assert (strategy.df["ADX"].dropna() >= 0).all()


# ── Signals ─────────────────────────────────────────────────────────────── #

def test_signal_values(sample_df):
    """Signal column must only contain -1, 0, or 1."""
    df = TechnicalIndicatorStrategy(sample_df).run()
    assert set(df["signal"].unique()).issubset({-1, 0, 1})
