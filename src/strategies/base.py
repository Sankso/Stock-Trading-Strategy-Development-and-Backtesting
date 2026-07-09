from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class enforcing a standard strategy lifecycle."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def compute_indicators(self) -> pd.DataFrame:
        """Compute all technical indicators. Must return enriched DataFrame."""

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Generate buy/sell signals. Must return DataFrame with 'signal' column."""

    def run(self) -> pd.DataFrame:
        """Standard lifecycle: compute indicators → generate signals."""
        self.df = self.compute_indicators()
        self.df = self.generate_signals()
        return self.df
