"""Testy procesu backfillu OHLCV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pytest

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
    service = OHLCVBackfillService(cached, chunk_limit=2, min_request_interval=0.0)

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
    service = OHLCVBackfillService(cached, chunk_limit=2, min_request_interval=0.0)

    summaries = service.synchronize(
        symbols=("BTCUSDT",),
        interval="1d",
        start=0,
        end=172_800_000,
    )

    summary = summaries[0]
    assert summary.fetched_candles == 0
    assert summary.skipped_candles == 3


class FixedSource(DataSource):
    """Źródło zwracające z góry zdefiniowane świece."""

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._rows = rows

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        del request
        return OHLCVResponse(columns=("open_time", "open", "high", "low", "close", "volume"), rows=self._rows)

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def test_cached_source_updates_metadata_and_merges_rows() -> None:
    storage = InMemoryStorage()
    storage.write(
        "BTCUSDT::1d",
        {
            "columns": ("open_time", "open", "high", "low", "close", "volume"),
            "rows": [[0.0, 1.0, 2.0, 0.5, 1.5, 10.0]],
        },
    )

    upstream = FixedSource(
        rows=[
            [0.0, 10.0, 20.0, 5.0, 15.0, 100.0],
            [86_400_000.0, 11.0, 21.0, 6.0, 16.0, 110.0],
        ]
    )
    cached = CachedOHLCVSource(storage=storage, upstream=upstream)

    cached.warm_cache(symbols=("ETHUSDT", "BTCUSDT"), intervals=("1d", "1h", "1d"))

    metadata = storage.metadata()
    assert metadata["symbols"] == "BTCUSDT,ETHUSDT"
    assert metadata["intervals"] == "1d,1h"

    response = cached.fetch_ohlcv(
        OHLCVRequest(symbol="BTCUSDT", interval="1d", start=0, end=200_000_000)
    )
    assert len(response.rows) == 2
    assert response.rows[0][1] == 10.0  # nowa świeca zastąpiła zapis z cache
    assert response.rows[1][0] == 86_400_000.0

    persisted = storage.read("BTCUSDT::1d")
    assert len(persisted["rows"]) == 2
    assert persisted["rows"][0][1] == 10.0


class FlakySource(DataSource):
    """Źródło, które dwukrotnie zwraca błąd zanim zacznie działać."""

    def __init__(self) -> None:
        self.calls = 0

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary error")
        return OHLCVResponse(
            columns=("open_time", "open", "high", "low", "close", "volume"),
            rows=[[float(request.start), 1.0, 2.0, 0.5, 1.5, 10.0]],
        )

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def test_backfill_retries_with_backoff_and_jitter() -> None:
    storage = InMemoryStorage()
    flaky = FlakySource()
    cached = CachedOHLCVSource(storage=storage, upstream=flaky)

    sleep_calls: list[float] = []
    current_time = 0.0

    def sleep_stub(delay: float) -> None:
        nonlocal current_time
        sleep_calls.append(delay)
        current_time += max(0.0, delay)

    def time_stub() -> float:
        return current_time

    random_values = iter([0.0, 0.0, 0.0])

    def random_stub() -> float:
        return next(random_values, 0.0)

    service = OHLCVBackfillService(
        cached,
        chunk_limit=2,
        min_request_interval=0.2,
        max_retries=3,
        backoff_factor=2.0,
        max_jitter_seconds=0.1,
        sleep=sleep_stub,
        time_source=time_stub,
        rng=random_stub,
    )

    summaries = service.synchronize(symbols=("BTCUSDT",), interval="1d", start=0, end=0)

    assert flaky.calls == 3  # dwa błędy + jedna udana próba
    assert summaries[0].fetched_candles == 1
    assert any(delay >= 0.2 for delay in sleep_calls)


class AlwaysFailingSource(DataSource):
    """Źródło, które zawsze rzuca błąd – testujemy limit retry."""

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        raise RuntimeError("permanent failure")

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def test_backfill_raises_after_exhausting_retries() -> None:
    storage = InMemoryStorage()
    cached = CachedOHLCVSource(storage=storage, upstream=AlwaysFailingSource())

    service = OHLCVBackfillService(
        cached,
        chunk_limit=2,
        min_request_interval=0.0,
        max_retries=1,
        backoff_factor=2.0,
        max_jitter_seconds=0.0,
        sleep=lambda _delay: None,
        time_source=lambda: 0.0,
        rng=lambda: 0.0,
    )

    with pytest.raises(RuntimeError):
        service.synchronize(symbols=("BTCUSDT",), interval="1d", start=0, end=0)
