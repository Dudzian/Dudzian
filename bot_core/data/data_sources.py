"""Publiczne interfejsy źródeł danych i lokalnych magazynów cache."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable


@dataclass(slots=True)
class OHLCVRequest:
    """Parametry pobierania danych OHLCV z publicznych API."""

    symbol: str
    interval: str
    start: int
    end: int
    limit: int | None = None


@dataclass(slots=True)
class OHLCVResponse:
    """Ujednolicone dane świec w UTC."""

    columns: Sequence[str]
    rows: Sequence[Sequence[float]]


class DataSource(Protocol):
    """Bazowy kontrakt źródeł danych rynkowych."""

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Pobiera i normalizuje dane OHLCV."""

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        """Pozwala na wstępne zapełnienie cache (np. przy starcie aplikacji)."""


@runtime_checkable
class CacheStorage(Protocol):
    """Minimalny kontrakt lokalnego magazynu danych (np. Parquet/SQLite)."""

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        ...

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        ...

    def metadata(self) -> MutableMapping[str, str]:
        ...

    def latest_timestamp(self, key: str) -> float | None:
        """Zwraca znacznik czasu ostatniej świecy zapisanej w cache."""

        ...


__all__ = [
    "CacheStorage",
    "DataSource",
    "OHLCVRequest",
    "OHLCVResponse",
]
