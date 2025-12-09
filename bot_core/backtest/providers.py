"""Wspólny interfejs providerów danych historycznych (OHLCV/Trades)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Protocol, Sequence, Tuple

import pandas as pd

__all__ = [
    "HistoryProvider",
    "OHLCVBar",
    "TradeTick",
    "PandasHistoryProvider",
    "ListHistoryProvider",
]


@dataclass(slots=True)
class OHLCVBar:
    """Pojedyncza świeca OHLCV w UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, float], timestamp: datetime) -> "OHLCVBar":
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return cls(
            timestamp=ts,
            open=float(payload.get("open", payload.get("close", 0.0))),
            high=float(payload.get("high", payload.get("close", 0.0))),
            low=float(payload.get("low", payload.get("close", 0.0))),
            close=float(payload["close"]),
            volume=float(payload.get("volume", 0.0)),
        )

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(slots=True)
class TradeTick:
    """Minimalna reprezentacja ticka transakcyjnego."""

    timestamp: datetime
    price: float
    size: float
    side: str


class HistoryProvider(Protocol):
    """Wspólny interfejs loaderów danych historycznych dla backtestu i runtime."""

    def iter_ohlcv(self) -> Iterable[OHLCVBar]:
        ...

    def iter_rows(self) -> Iterable[Tuple[datetime, Mapping[str, float]]]:
        ...

    def history_until(self, index: int) -> pd.DataFrame:
        ...

    @property
    def dataframe(self) -> pd.DataFrame:  # pragma: no cover - dostęp debugowy
        ...


class PandasHistoryProvider(HistoryProvider):
    """Adapter ramki OHLCV do interfejsu providerów backtest/runtime."""

    def __init__(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        if data.empty:
            raise ValueError("Backtest wymaga danych historycznych")
        if "close" not in data.columns:
            raise ValueError("Dane historyczne wymagają kolumny 'close'")
        if "volume" not in data.columns:
            data = data.copy()
            data["volume"] = 0.0
        self._data = data.sort_index()
        self.symbol = symbol
        self.timeframe = timeframe
        self._history_cache_idx = -1
        self._history_cache: pd.DataFrame = self._data.iloc[:0]

    def iter_ohlcv(self) -> Iterable[OHLCVBar]:
        for ts, row in self._data.iterrows():
            timestamp = (
                ts if isinstance(ts, datetime) else datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            yield OHLCVBar.from_mapping(row, timestamp=timestamp)

    def iter_rows(self) -> Iterable[Tuple[datetime, Mapping[str, float]]]:
        for candle in self.iter_ohlcv():
            yield candle.timestamp, candle.as_mapping()

    def history_until(self, index: int) -> pd.DataFrame:
        if index < 0:
            return self._data.iloc[:0]
        if index == self._history_cache_idx:
            return self._history_cache
        self._history_cache = self._data.head(index + 1)
        self._history_cache_idx = index
        return self._history_cache

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._data


class ListHistoryProvider(HistoryProvider):
    """Provider oparty o listę świec (np. z runtime/streamingu)."""

    def __init__(self, candles: Sequence[OHLCVBar]) -> None:
        if not candles:
            raise ValueError("Wymagane są dane OHLCV do utworzenia provider'a")
        self._candles = list(candles)

    def iter_ohlcv(self) -> Iterable[OHLCVBar]:
        return iter(self._candles)

    def iter_rows(self) -> Iterable[Tuple[datetime, Mapping[str, float]]]:
        for candle in self._candles:
            yield candle.timestamp, candle.as_mapping()

    def history_until(self, index: int) -> pd.DataFrame:
        if index < 0:
            return self.dataframe.iloc[:0]
        return self.dataframe.head(index + 1)

    @property
    def dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame([candle.as_mapping() for candle in self._candles])
        frame.index = pd.to_datetime([candle.timestamp for candle in self._candles])
        frame.index.name = "timestamp"
        return frame
