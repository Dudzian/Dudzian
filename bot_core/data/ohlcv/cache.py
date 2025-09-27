"""Warstwa odpowiedzialna za backfill i lokalny cache OHLCV."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse

_LOGGER = logging.getLogger(__name__)

_DEFAULT_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(slots=True)
class CachedOHLCVSource(DataSource):
    """Łączy publiczne API z lokalnym cache, zapewniając minimalne hit-rate API."""

    storage: CacheStorage
    upstream: DataSource

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

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Pobiera dane OHLCV łącząc cache oraz świeże dane z API."""

        cache_key = self._cache_key(request.symbol, request.interval)
        try:
            cached_payload = self.storage.read(cache_key)
        except KeyError:
            cached_payload = {"rows": [], "columns": _DEFAULT_COLUMNS}
        cached_rows: Sequence[Sequence[float]] = cached_payload.get("rows", [])
        columns = tuple(cached_payload.get("columns", ()))

        upstream_response = self.upstream.fetch_ohlcv(request)
        rows = cached_rows
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
        if not columns:
            columns = _DEFAULT_COLUMNS

        filtered = [
            row
            for row in rows
            if request.start <= float(row[0]) <= request.end
        ]
        if request.limit is not None:
            filtered = filtered[: request.limit]

        return OHLCVResponse(columns=columns, rows=filtered)

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        """Aktualizuje metadane cache, ułatwiając audyt i monitoring."""

        metadata = self.storage.metadata()
        metadata["symbols"] = ",".join(sorted(set(symbols)))
        metadata["intervals"] = ",".join(sorted(set(intervals)))
        _LOGGER.info("Cache OHLCV gotowe dla %s symboli i %s interwałów.", len(set(symbols)), len(set(intervals)))


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


# Aby uniknąć cyklicznych importów, importujemy w runtime.
from bot_core.exchanges.base import ExchangeAdapter  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["CachedOHLCVSource", "PublicAPIDataSource"]
