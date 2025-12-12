"""Wspólny harmonogram backfillu/warmupu danych OHLCV."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Sequence

from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.sources import CachedOHLCVSource
from bot_core.data.ohlcv import OHLCVBackfillService
from bot_core.exchanges.base import ExchangeAdapter

_LOGGER = logging.getLogger(__name__)

_DEFAULT_OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class BackfillScheduler:
    """Centralizuje logikę backfillu i rozgrzewania cache'u.

    Scheduler działa na dostarczonym źródle cache'ującym oraz opcjonalnym serwisie
    backfillu i adapterze giełdowym. Wykorzystywany zarówno w runtime, jak i w
    testach, aby uniknąć duplikacji ręcznych pętli.
    """

    def __init__(
        self,
        data_source: CachedOHLCVSource,
        *,
        backfill_service: OHLCVBackfillService | None = None,
        adapter: ExchangeAdapter | None = None,
        default_columns: Sequence[str] | None = None,
    ) -> None:
        self._data_source = data_source
        self._backfill_service = backfill_service
        self._adapter = adapter
        self._default_columns: tuple[str, ...] = tuple(default_columns or _DEFAULT_OHLCV_COLUMNS)

    def ensure_ohlcv_availability(
        self,
        *,
        symbols: Iterable[str],
        interval: str,
        environment: object,
        now_ms: int | None = None,
    ) -> None:
        """Zapewnia minimalną dostępność świec OHLCV dla wskazanych symboli."""

        symbols = tuple(symbols)
        offline_mode = bool(getattr(environment, "offline_mode", False))
        current_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)

        missing_symbols: list[str] = []
        for symbol in symbols:
            request = OHLCVRequest(symbol=symbol, interval=interval, start=0, end=current_ms, limit=1)
            try:
                response = self._data_source.fetch_ohlcv(request)
            except Exception:  # pragma: no cover - fallback gdy upstream cache zwróci błąd
                response = OHLCVResponse(columns=self._default_columns, rows=())
            if response.rows:
                continue
            missing_symbols.append(symbol)

        if not missing_symbols:
            return

        lookback_days = getattr(environment, "offline_backfill_days", 7 if offline_mode else 30)
        lookback_ms = max(int(lookback_days) * 86_400_000, 86_400_000)
        start_ms = max(0, current_ms - lookback_ms)

        if offline_mode and self._backfill_service is not None:
            try:
                self._backfill_service.synchronize(
                    symbols=missing_symbols,
                    interval=interval,
                    start=start_ms,
                    end=current_ms,
                )
                return
            except Exception:  # pragma: no cover - defensywne logowanie w trybie offline
                _LOGGER.debug(
                    "Offline backfill nie powiódł się dla %s – przechodzę do fallbacku adaptera",
                    missing_symbols,
                    exc_info=True,
                )

        if offline_mode:
            _LOGGER.debug(
                "Środowisko offline – pomijam dogrywanie danych OHLCV (symbole=%s)",
                missing_symbols,
            )
            return

        if self._backfill_service is not None:
            try:
                self._backfill_service.synchronize(
                    symbols=missing_symbols,
                    interval=interval,
                    start=start_ms,
                    end=current_ms,
                )
                return
            except Exception:  # pragma: no cover - upstream może być chwilowo niedostępny
                _LOGGER.exception(
                    "Nie udało się wykonać backfillu startowego (%s, %s) – spróbuję fallbacku adaptera",
                    missing_symbols,
                    interval,
                )

        if self._adapter is None:
            _LOGGER.debug(
                "Brak adaptera giełdowego – pomijam wypełnianie cache danych OHLCV (symbole=%s)",
                missing_symbols,
            )
            return

        warmup_limit = max(int(getattr(environment, "offline_warmup_candles", 180)), 1)
        fetch_start = None if offline_mode else start_ms
        fetch_end = None if offline_mode else current_ms

        for symbol in missing_symbols:
            try:
                rows = self._adapter.fetch_ohlcv(
                    symbol,
                    interval,
                    start=fetch_start,
                    end=fetch_end,
                    limit=warmup_limit,
                )
            except Exception:  # pragma: no cover - diagnostyka adaptera testowego
                _LOGGER.warning(
                    "Nie udało się pobrać danych OHLCV przez adapter w trybie offline (%s %s)",
                    symbol,
                    interval,
                    exc_info=True,
                )
                continue

            if not rows:
                _LOGGER.debug(
                    "Adapter nie zwrócił danych OHLCV (%s %s) – cache pozostaje pusty",
                    symbol,
                    interval,
                )
                continue

            cache_key = self._data_source._cache_key(symbol, interval)  # pylint: disable=protected-access
            payload_rows = [list(map(float, row)) for row in rows if row]
            self._data_source.storage.write(
                cache_key,
                {
                    "columns": list(self._default_columns),
                    "rows": payload_rows,
                },
            )


__all__ = ["BackfillScheduler"]
