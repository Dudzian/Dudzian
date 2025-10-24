"""Warstwa odpowiedzialna za backfill i lokalny cache OHLCV."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.data.intervals import interval_to_milliseconds
from bot_core.exchanges.errors import ExchangeNetworkError

_LOGGER = logging.getLogger(__name__)

_DEFAULT_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


SnapshotFetcher = Callable[[OHLCVRequest], Sequence[Sequence[float]]]


@dataclass(slots=True)
class CachedOHLCVSource(DataSource):
    """Łączy publiczne API z lokalnym cache, zapewniając minimalne hit-rate API."""

    storage: CacheStorage
    upstream: DataSource
    snapshot_fetcher: SnapshotFetcher | None = None
    snapshots_enabled: bool = False

    def _cache_key(self, symbol: str, interval: str) -> str:
        return f"{symbol}::{interval}"

    def _merge_rows(
        self,
        cached_rows: Sequence[Sequence[float]],
        upstream_rows: Sequence[Sequence[float]],
    ) -> list[Sequence[float]]:
        combined: dict[float, Sequence[float]] = {float(row[0]): row for row in cached_rows}
        for row in upstream_rows:
            if not row:
                continue
            combined[float(row[0])] = row
        return [combined[key] for key in sorted(combined)]

    def _load_cached_payload(
        self, cache_key: str
    ) -> tuple[Sequence[Sequence[float]], tuple[str, ...]]:
        """Loads rows and columns from cache, tolerating a missing entry."""

        try:
            cached_payload = self.storage.read(cache_key)
        except KeyError:
            return (), ()

        rows = cached_payload.get("rows", [])
        columns = tuple(cached_payload.get("columns", ()))
        return rows, columns

    def _fetch_upstream_response(
        self,
        request: OHLCVRequest,
        fallback_columns: Sequence[str] | tuple[str, ...],
    ) -> OHLCVResponse:
        """Always attempts to hit the upstream API, falling back to cache columns."""

        try:
            return self.upstream.fetch_ohlcv(request)
        except ExchangeNetworkError as exc:
            _LOGGER.warning(
                "Upstream OHLCV niedostępny – używam danych z cache (%s %s): %s",
                request.symbol,
                request.interval,
                exc,
            )
            return OHLCVResponse(columns=fallback_columns or _DEFAULT_COLUMNS, rows=[])

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Pobiera dane OHLCV łącząc cache oraz świeże dane z API."""

        cache_key = self._cache_key(request.symbol, request.interval)
        cached_rows, columns = self._load_cached_payload(cache_key)

        snapshot_fetcher = self.snapshot_fetcher
        if snapshot_fetcher is None and self.snapshots_enabled:
            snapshot_fetcher = self._fallback_snapshot_fetcher()
            if snapshot_fetcher is not None:
                self.snapshot_fetcher = snapshot_fetcher

        matching_cached_rows = [
            row for row in cached_rows if row and request.start <= float(row[0]) <= request.end
        ]

        cache_covers_request = False
        if matching_cached_rows:
            deduped_timestamps = sorted({float(row[0]) for row in matching_cached_rows if row})
            if request.limit is not None:
                cache_covers_request = self._cached_rows_cover_limit_window(
                    request, deduped_timestamps
                )
            else:
                cache_covers_request = self._cached_rows_cover_range(
                    request, deduped_timestamps
                )

        should_hit_upstream = not (
            cache_covers_request and snapshot_fetcher is not None
        )

        rows = cached_rows
        if should_hit_upstream:
            upstream_response = self._fetch_upstream_response(request, columns)
            if upstream_response.rows:
                rows = self._merge_rows(cached_rows, upstream_response.rows)
                # Aktualizacja cache: zapisujemy całą serię, aby kolejne zapytania były szybkie.
                selected_columns = columns or tuple(upstream_response.columns or _DEFAULT_COLUMNS)
                self.storage.write(
                    cache_key,
                    {
                        "columns": list(selected_columns),
                        "rows": rows,
                    },
                )
                columns = selected_columns
        if snapshot_fetcher is not None:
            try:
                raw_snapshot_rows = tuple(snapshot_fetcher(request))
            except ExchangeNetworkError as exc:
                _LOGGER.warning(
                    "Snapshot API niedostępne – pozostaję przy danych z cache (%s %s): %s",
                    request.symbol,
                    request.interval,
                    exc,
                )
                raw_snapshot_rows = ()
            except Exception as exc:  # pragma: no cover - logowanie diagnostyczne
                _LOGGER.exception("Nieudany snapshot OHLCV (%s %s): %s", request.symbol, request.interval, exc)
                raw_snapshot_rows = ()

            snapshot_rows = self._normalize_snapshot_rows(raw_snapshot_rows, request)
            if snapshot_rows:
                rows = self._merge_rows(rows, snapshot_rows)
                if not columns:
                    columns = _DEFAULT_COLUMNS
                self.storage.write(
                    cache_key,
                    {
                        "columns": list(columns),
                        "rows": rows,
                    },
                )
        if not columns:
            columns = _DEFAULT_COLUMNS

        filtered = [
            row
            for row in rows
            if request.start <= float(row[0]) <= request.end
        ]
        if request.limit is not None and request.limit > 0:
            filtered = filtered[-request.limit :]

        return OHLCVResponse(columns=columns, rows=filtered)

    def _cached_rows_cover_limit_window(
        self,
        request: OHLCVRequest,
        deduped_timestamps: Sequence[float],
    ) -> bool:
        if request.limit is None or request.limit <= 0:
            return False

        if len(deduped_timestamps) < request.limit:
            return False

        try:
            interval_ms = interval_to_milliseconds(request.interval)
        except (KeyError, ValueError):  # pragma: no cover - brak znanych interwałów
            interval_ms = 0

        recent_timestamps = deduped_timestamps[-request.limit :]
        latest = recent_timestamps[-1]

        if interval_ms <= 0:
            return latest >= request.end

        max_allowed_gap = interval_ms
        min_expected_start = request.end - interval_ms * max(request.limit - 1, 1)

        covers_end = request.end - max_allowed_gap <= latest <= request.end
        if not covers_end:
            return False

        if request.limit == 1:
            return True

        earliest = recent_timestamps[0]
        if earliest < min_expected_start:
            return False

        # Pozwól na drobne odchylenia (np. agregatory zwracające wartości w sekundach).
        tolerance = max(1.0, interval_ms * 0.05)
        allowed_gap = max_allowed_gap + tolerance
        for previous, current in zip(recent_timestamps, recent_timestamps[1:]):
            if current - previous > allowed_gap:
                return False

        return True

    def _cached_rows_cover_range(
        self,
        request: OHLCVRequest,
        deduped_timestamps: Sequence[float],
    ) -> bool:
        if not deduped_timestamps:
            return False

        try:
            interval_ms = interval_to_milliseconds(request.interval)
        except (KeyError, ValueError):  # pragma: no cover - brak znanych interwałów
            return (
                request.start >= deduped_timestamps[0]
                and request.end <= deduped_timestamps[-1]
            )

        tolerance = max(1.0, interval_ms * 0.05)
        allowed_gap = interval_ms + tolerance

        start_gap = deduped_timestamps[0] - request.start
        end_gap = request.end - deduped_timestamps[-1]

        if start_gap > tolerance or end_gap > tolerance:
            return False

        for previous, current in zip(deduped_timestamps, deduped_timestamps[1:]):
            if current - previous > allowed_gap:
                return False

        return True

    def _normalize_snapshot_rows(
        self,
        snapshot_rows: Sequence[Sequence[float]],
        request: OHLCVRequest,
    ) -> list[Sequence[float]]:
        normalized: list[Sequence[float]] = [tuple(row) for row in snapshot_rows if row]
        if not normalized:
            return []

        last_row = list(normalized[-1])
        if last_row and float(last_row[0]) != float(request.end):
            adjusted_row = list(last_row)
            adjusted_row[0] = float(request.end)
            normalized.append(tuple(adjusted_row))

        return normalized

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        """Aktualizuje metadane cache, ułatwiając audyt i monitoring."""

        metadata = self.storage.metadata()
        metadata["symbols"] = ",".join(sorted(set(symbols)))
        metadata["intervals"] = ",".join(sorted(set(intervals)))
        _LOGGER.info("Cache OHLCV gotowe dla %s symboli i %s interwałów.", len(set(symbols)), len(set(intervals)))

    def _fallback_snapshot_fetcher(self) -> SnapshotFetcher | None:
        """Buduje zapasowy snapshot wykorzystując adapter upstreamowy, jeśli to możliwe."""

        adapter = getattr(self.upstream, "exchange_adapter", None)
        fetch = getattr(adapter, "fetch_ohlcv", None)
        if fetch is None:
            return None

        def _snapshot(request: OHLCVRequest) -> Sequence[Sequence[float]]:
            interval_ms = interval_to_milliseconds(request.interval)
            window = max(interval_ms * 2, interval_ms)
            window_start = max(request.start, request.end - window)
            limit = request.limit if request.limit and request.limit > 0 else 1
            return fetch(
                request.symbol,
                request.interval,
                start=window_start,
                end=request.end,
                limit=limit,
            )

        return _snapshot


@dataclass(slots=True)
class PublicAPIDataSource(DataSource):
    """Adapter korzystający bezpośrednio z publicznych endpointów giełdy."""

    exchange_adapter: "ExchangeAdapter"

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        rows = self.exchange_adapter.fetch_ohlcv(
            request.symbol,
            request.interval,
            start=request.start,
            end=request.end,
            limit=request.limit,
        )
        return OHLCVResponse(columns=_DEFAULT_COLUMNS, rows=rows)

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        # Publiczne API nie wymaga przygotowania cache – metoda pozostaje dla zgodności interfejsu.
        _LOGGER.debug(
            "Pomijam warm_cache dla PublicAPIDataSource (symbole=%s, interwały=%s)",
            list(symbols),
            list(intervals),
        )


@dataclass(slots=True)
class OfflineOnlyDataSource(DataSource):
    """Źródło danych działające wyłącznie na cache, bez dostępu do sieci."""

    exchange_name: str | None = None

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        if self.exchange_name:
            _LOGGER.debug(
                "Tryb offline: pomijam upstream OHLCV dla %s (%s %s)",
                self.exchange_name,
                request.symbol,
                request.interval,
            )
        else:
            _LOGGER.debug(
                "Tryb offline: pomijam upstream OHLCV (%s %s)",
                request.symbol,
                request.interval,
            )
        return OHLCVResponse(columns=_DEFAULT_COLUMNS, rows=())

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        _LOGGER.debug(
            "Tryb offline: warm_cache bez działania (symbole=%s, interwały=%s)",
            list(symbols),
            list(intervals),
        )


# Aby uniknąć cyklicznych importów, importujemy w runtime.
from bot_core.exchanges.base import ExchangeAdapter  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["CachedOHLCVSource", "PublicAPIDataSource", "OfflineOnlyDataSource"]
