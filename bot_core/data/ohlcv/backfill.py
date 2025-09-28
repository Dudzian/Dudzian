"""Proces backfillu OHLCV łączący publiczne API i lokalny cache."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from bot_core.data.base import CacheStorage, OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.cache import CachedOHLCVSource

_LOGGER = logging.getLogger(__name__)

_INTERVAL_TO_MILLISECONDS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


@dataclass(slots=True)
class BackfillSummary:
    """Raport końcowy z procesu synchronizacji OHLCV."""

    symbol: str
    interval: str
    requested_start: int
    requested_end: int
    fetched_candles: int
    skipped_candles: int


class OHLCVBackfillService:
    """Synchronizuje lokalny cache z publicznymi danymi giełdowymi.

    Wbudowane ograniczanie częstotliwości zapytań, losowy jitter i kontrola retry
    pozwalają respektować limity giełd i rekomendacje dotyczące „dobrego sąsiedztwa”
    podczas długich backfilli historii.
    """

    def __init__(
        self,
        source: CachedOHLCVSource,
        *,
        chunk_limit: int = 1000,
        min_request_interval: float = 0.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_jitter_seconds: float = 0.0,
        sleep: Callable[[float], None] | None = None,
        time_source: Callable[[], float] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        if chunk_limit <= 0:
            raise ValueError("chunk_limit musi być dodatni")
        self._source = source
        self._storage: CacheStorage = source.storage
        self._chunk_limit = chunk_limit
        self._sleep = sleep or time.sleep
        self._time = time_source or time.monotonic
        self._random = rng or random.random
        self._min_request_interval = max(0.0, float(min_request_interval))
        self._backoff_factor = max(1.0, float(backoff_factor))
        self._max_retries = max(0, int(max_retries))
        self._max_jitter = max(0.0, float(max_jitter_seconds))
        self._last_request_ts: float | None = None

    def _interval_milliseconds(self, interval: str) -> int:
        try:
            return _INTERVAL_TO_MILLISECONDS[interval]
        except KeyError as exc:  # pragma: no cover - walidacja w czasie runtime
            raise ValueError(f"Nieobsługiwany interwał: {interval}") from exc

    def _latest_cached_timestamp(self, symbol: str, interval: str) -> float | None:
        key = self._source._cache_key(symbol, interval)  # pylint: disable=protected-access
        try:
            return self._storage.latest_timestamp(key)
        except AttributeError:  # pragma: no cover - zabezpieczenie pod inne storage
            try:
                rows = self._storage.read(key)["rows"]
            except KeyError:
                return None
            if not rows:
                return None
            return float(rows[-1][0])
        except KeyError:
            return None

    def _count_cached_rows(self, symbol: str, interval: str) -> int:
        key = self._source._cache_key(symbol, interval)  # pylint: disable=protected-access
        try:
            metadata = self._storage.metadata()
            stored = metadata.get(f"row_count::{symbol}::{interval}")
            if stored is not None:
                return int(stored)
        except Exception:  # pragma: no cover - wspieramy różne implementacje storage
            pass
        try:
            rows = self._storage.read(key)["rows"]
        except KeyError:
            return 0
        return len(rows)

    def synchronize(
        self,
        *,
        symbols: Sequence[str],
        interval: str,
        start: int,
        end: int,
    ) -> list[BackfillSummary]:
        """Aktualizuje cache dla listy symboli i zwraca raport."""

        if start > end:
            raise ValueError("Parametr start nie może być większy niż end")

        interval_ms = self._interval_milliseconds(interval)
        summaries: list[BackfillSummary] = []

        for symbol in symbols:
            existing_rows = self._count_cached_rows(symbol, interval)
            latest_cached = self._latest_cached_timestamp(symbol, interval)
            effective_start = start
            if latest_cached is not None:
                effective_start = max(start, int(latest_cached) + interval_ms)

            if effective_start > end:
                summaries.append(
                    BackfillSummary(
                        symbol=symbol,
                        interval=interval,
                        requested_start=start,
                        requested_end=end,
                        fetched_candles=0,
                        skipped_candles=existing_rows,
                    )
                )
                _LOGGER.info(
                    "Symbol %s (%s) jest aktualny do %s – pomijam backfill.",
                    symbol,
                    interval,
                    latest_cached,
                )
                continue

            next_start = effective_start
            last_progress = -1.0

            while next_start <= end:
                window_end = min(end, next_start + interval_ms * (self._chunk_limit - 1))
                request = OHLCVRequest(
                    symbol=symbol,
                    interval=interval,
                    start=next_start,
                    end=window_end,
                    limit=self._chunk_limit,
                )
                response = self._fetch_with_retry(request)
                if not response.rows:
                    _LOGGER.warning(
                        "Brak danych OHLCV dla %s (%s) w przedziale %s-%s.",
                        symbol,
                        interval,
                        next_start,
                        window_end,
                    )
                    break

                newest_timestamp = float(response.rows[-1][0])
                if newest_timestamp <= last_progress:
                    _LOGGER.debug(
                        "Zatrzymuję backfill %s (%s) – brak postępu (ostatni=%s).",
                        symbol,
                        interval,
                        newest_timestamp,
                    )
                    break

                last_progress = newest_timestamp
                next_start = int(newest_timestamp) + interval_ms

                if newest_timestamp >= end:
                    break

            updated_rows = self._count_cached_rows(symbol, interval)
            summaries.append(
                BackfillSummary(
                    symbol=symbol,
                    interval=interval,
                    requested_start=start,
                    requested_end=end,
                    fetched_candles=max(0, updated_rows - existing_rows),
                    skipped_candles=existing_rows,
                )
            )

        return summaries

    # ----------------------------------------------------------------------------------------------
    def _throttle(self) -> None:
        if self._min_request_interval <= 0 and self._max_jitter <= 0:
            return

        now = self._time()
        delay = 0.0
        if self._last_request_ts is not None:
            elapsed = now - self._last_request_ts
            if elapsed < self._min_request_interval:
                delay = self._min_request_interval - elapsed

        jitter = self._random() * self._max_jitter if self._max_jitter > 0 else 0.0
        total_delay = delay + jitter
        if total_delay > 0:
            self._sleep(total_delay)

        self._last_request_ts = self._time()

    def _fetch_with_retry(self, request: OHLCVRequest) -> OHLCVResponse:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._source.fetch_ohlcv(request)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self._max_retries:
                    _LOGGER.error(
                        "Backfill: nie udało się pobrać danych dla %s (%s) po %s próbach.",
                        request.symbol,
                        request.interval,
                        attempt,
                        exc_info=exc,
                    )
                    raise

                base_delay = max(self._min_request_interval, 0.1)
                backoff_delay = base_delay * (self._backoff_factor ** (attempt - 1))
                jitter = self._random() * self._max_jitter if self._max_jitter > 0 else 0.0
                total_delay = backoff_delay + jitter
                _LOGGER.warning(
                    "Backfill: próba %s dla %s (%s) nieudana – retry za %.3fs.",
                    attempt,
                    request.symbol,
                    request.interval,
                    total_delay,
                    exc_info=exc,
                )
                if total_delay > 0:
                    self._sleep(total_delay)
                continue

            self._last_request_ts = self._time()
            return response


__all__ = ["OHLCVBackfillService", "BackfillSummary"]
