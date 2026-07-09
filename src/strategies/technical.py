import pandas as pd
import numpy as np
from .base import BaseStrategy


class TechnicalIndicatorStrategy(BaseStrategy):
    """
    Concrete strategy using RSI, Bollinger Bands, and ADX.
    All calculations are fully vectorized — zero row-level loops.
    """

    def __init__(self, df: pd.DataFrame, rsi_period: int = 14,
                 bb_period: int = 20, bb_std: float = 2.0, adx_period: int = 14):
        super().__init__(df)
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period

    # ------------------------------------------------------------------ #
    #  Indicator math — fully vectorized                                   #
    # ------------------------------------------------------------------ #

    def _compute_rsi(self) -> pd.Series:
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_bollinger_bands(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        ma = self.df['close'].rolling(self.bb_period).mean()
        std = self.df['close'].rolling(self.bb_period).std()
        return ma, ma + self.bb_std * std, ma - self.bb_std * std

    def _compute_adx(self) -> pd.Series:
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        n = self.adx_period

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        # Directional movement
        dm_pos = high.diff().clip(lower=0).where(high.diff() > (-low.diff()), 0)
        dm_neg = (-low.diff()).clip(lower=0).where((-low.diff()) > high.diff(), 0)

        atr   = tr.rolling(n).mean()
        di_pos = 100 * dm_pos.rolling(n).mean() / atr.replace(0, np.nan)
        di_neg = 100 * dm_neg.rolling(n).mean() / atr.replace(0, np.nan)
        dx     = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan))
        return dx.rolling(n).mean()

    # ------------------------------------------------------------------ #
    #  BaseStrategy interface                                              #
    # ------------------------------------------------------------------ #

    def compute_indicators(self) -> pd.DataFrame:
        self.df['RSI'] = self._compute_rsi()
        self.df['BB_MA'], self.df['BB_upper'], self.df['BB_lower'] = self._compute_bollinger_bands()
        self.df['ADX'] = self._compute_adx()
        return self.df

    def generate_signals(self) -> pd.DataFrame:
        """
        Buy  (+1): RSI oversold (<30) OR price below lower BB, AND trend present (ADX>25).
        Sell (-1): RSI overbought (>70) OR price above upper BB, AND trend present (ADX>25).
        """
        trending = self.df['ADX'] > 25
        buy  = trending & ((self.df['RSI'] < 30) | (self.df['close'] < self.df['BB_lower']))
        sell = trending & ((self.df['RSI'] > 70) | (self.df['close'] > self.df['BB_upper']))

        self.df['signal'] = np.where(buy, 1, np.where(sell, -1, 0))
        return self.df
