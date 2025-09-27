"""Testy procesu backfillu OHLCV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource


class InMemoryStorage(CacheStorage):
    """Prosta pamięciowa implementacja magazynu do testów."""

    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Sequence[Sequence[float]]]] = {}
        self._metadata: dict[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._store[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        try:
            rows = self._store[key]["rows"]
        except KeyError:
            return None
        if not rows:
            return None
        return float(rows[-1][0])


@dataclass(slots=True)
class StubSource(DataSource):
    """Źródło danych generujące deterministyczne świece dla testów."""

    interval_ms: int

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        rows: list[Sequence[float]] = []
        current = request.start
        limit = request.limit or 1000
        while current <= request.end and len(rows) < limit:
            rows.append([float(current), 1.0, 2.0, 0.5, 1.5, 10.0])
            current += self.interval_ms
        return OHLCVResponse(columns=("open_time", "open", "high", "low", "close", "volume"), rows=rows)

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def test_backfill_downloads_missing_candles() -> None:
    storage = InMemoryStorage()
    cached = CachedOHLCVSource(storage=storage, upstream=StubSource(interval_ms=86_400_000))
    service = OHLCVBackfillService(cached, chunk_limit=2)

    summaries = service.synchronize(
        symbols=("BTCUSDT",),
        interval="1d",
        start=0,
        end=86_400_000 * 4,
    )

    summary = summaries[0]
    assert summary.fetched_candles == 5
    assert summary.skipped_candles == 0


def test_backfill_skips_up_to_date_symbol() -> None:
    storage = InMemoryStorage()
    # Wstępnie zapisujemy trzy świece
    storage.write(
        "BTCUSDT::1d",
        {
            "columns": ("open_time", "open", "high", "low", "close", "volume"),
            "rows": [
                [0.0, 1.0, 2.0, 0.5, 1.5, 10.0],
                [86_400_000.0, 1.0, 2.0, 0.5, 1.5, 10.0],
                [172_800_000.0, 1.0, 2.0, 0.5, 1.5, 10.0],
            ],
        },
    )

    cached = CachedOHLCVSource(storage=storage, upstream=StubSource(interval_ms=86_400_000))
    service = OHLCVBackfillService(cached, chunk_limit=2)

    summaries = service.synchronize(
        symbols=("BTCUSDT",),
        interval="1d",
        start=0,
        end=172_800_000,
    )

    summary = summaries[0]
    assert summary.fetched_candles == 0
    assert summary.skipped_candles == 3
