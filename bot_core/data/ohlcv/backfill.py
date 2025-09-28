"""Proces backfillu OHLCV łączący publiczne API i lokalny cache."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from bot_core.data.base import CacheStorage, OHLCVRequest
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
    """Synchronizuje lokalny cache z publicznymi danymi giełdowymi."""

    def __init__(self, source: CachedOHLCVSource, *, chunk_limit: int = 1000) -> None:
        if chunk_limit <= 0:
            raise ValueError("chunk_limit musi być dodatni")
        self._source = source
        self._storage: CacheStorage = source.storage
        self._chunk_limit = chunk_limit

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
                response = self._source.fetch_ohlcv(request)
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


__all__ = ["OHLCVBackfillService", "BackfillSummary"]
