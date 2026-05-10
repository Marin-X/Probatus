"""Market data handlers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .events import MarketEvent


class DataHandler(ABC):
    @abstractmethod
    def update(self) -> Optional[MarketEvent]: ...

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame: ...

    @abstractmethod
    def get_latest_price(self, symbol: str, field: str = "Close") -> float: ...

    @property
    @abstractmethod
    def current_time(self) -> Optional[pd.Timestamp]: ...

    @property
    @abstractmethod
    def symbols(self) -> list[str]: ...


class InMemoryDataHandler(DataHandler):
    """Replay a {symbol: OHLCV DataFrame} dict in chronological order."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        if not data:
            raise ValueError("data dict must be non-empty")
        self._data: dict[str, pd.DataFrame] = {
            sym: df.copy().sort_index() for sym, df in data.items()
        }
        self._symbols: list[str] = list(self._data.keys())

        all_timestamps: set[pd.Timestamp] = set()
        for df in self._data.values():
            all_timestamps.update(df.index)
        self._timeline: list[pd.Timestamp] = sorted(all_timestamps)
        self._idx: int = -1

    def update(self) -> Optional[MarketEvent]:
        self._idx += 1
        if self._idx >= len(self._timeline):
            return None
        return MarketEvent(timestamp=self._timeline[self._idx])

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        if self._idx < 0 or symbol not in self._data:
            return pd.DataFrame()
        cutoff = self._timeline[self._idx]
        df = self._data[symbol]
        return df[df.index <= cutoff].tail(n)

    def get_latest_price(self, symbol: str, field: str = "Close") -> float:
        bars = self.get_latest_bars(symbol, n=1)
        if bars.empty or field not in bars.columns:
            return float("nan")
        return float(bars[field].iloc[-1])

    def get_symbol_history(self, symbol: str) -> pd.DataFrame:
        """Return full OHLCV history for a symbol, ignoring time cursor.

        For analytics/benchmarking only — strategies must use
        get_latest_bars() to preserve the no-look-ahead guarantee.
        """
        if symbol not in self._data:
            return pd.DataFrame()
        return self._data[symbol].copy()

    @property
    def current_time(self) -> Optional[pd.Timestamp]:
        if 0 <= self._idx < len(self._timeline):
            return self._timeline[self._idx]
        return None

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)


class YahooDataHandler(InMemoryDataHandler):
    """Daily bars from Yahoo Finance via yfinance."""

    def __init__(self, symbols: list[str], start: str, end: str,
                 auto_adjust: bool = True):
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YahooDataHandler. "
                "Install with: pip install yfinance"
            ) from e

        data: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = yf.download(sym, start=start, end=end,
                             progress=False, auto_adjust=auto_adjust)
            if df is None or df.empty:
                raise ValueError(f"No data returned for {sym}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            data[sym] = df
        super().__init__(data)