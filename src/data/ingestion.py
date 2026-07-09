import time
import requests
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestionError(Exception):
    """Raised when market data cannot be fetched after all retries."""


class DataIngestionEngine:
    """
    Fetches OHLCV data from Alpha Vantage with:
      - connection timeout enforcement
      - automatic retry on transient failures
      - rate-limit detection (HTTP 429 or API message)
      - malformed / missing payload validation
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: int = 10, max_retries: int = 3):
        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries

    def fetch(self, symbol: str) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol":   symbol,
            "apikey":   self.api_key,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()                     # catches 4xx / 5xx
                return self._parse(response.json(), symbol)

            except requests.Timeout:
                logger.warning("Attempt %d: request timed out.", attempt)
            except requests.ConnectionError:
                logger.warning("Attempt %d: connection error.", attempt)
            except requests.HTTPError as e:
                if response.status_code == 429:
                    wait = 60
                    logger.warning("Rate limited. Waiting %ds before retry.", wait)
                    time.sleep(wait)
                else:
                    raise DataIngestionError(f"HTTP error: {e}") from e

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)   # exponential back-off: 2s, 4s

        raise DataIngestionError(
            f"Failed to fetch '{symbol}' after {self.max_retries} attempts."
        )

    def _parse(self, payload: dict, symbol: str) -> pd.DataFrame:
        # Detect API-level errors (rate limit message, bad symbol, etc.)
        if "Error Message" in payload:
            raise DataIngestionError(f"API error: {payload['Error Message']}")
        if "Note" in payload:
            raise DataIngestionError(f"API rate limit: {payload['Note']}")

        key = "Time Series (Daily)"
        if key not in payload:
            raise DataIngestionError(
                f"Unexpected payload for '{symbol}': missing '{key}' key."
            )

        df = pd.DataFrame.from_dict(payload[key], orient="index").astype(float)
        df.index   = pd.to_datetime(df.index)
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.sort_index()
