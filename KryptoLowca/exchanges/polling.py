"""Pomocnicze narzędzia do odpytywania danych rynkowych przez REST."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .interfaces import MarketPayload


logger = logging.getLogger(__name__)


CallbackType = Callable[[str, MarketPayload], Awaitable[None] | None]
ErrorCallbackType = Callable[[str, Exception], Awaitable[None] | None]


class MarketDataPoller:
    """Asynchroniczny poller korzystający z metod REST adaptera giełdy."""

    def __init__(
        self,
        adapter: Any,
        *,
        symbols: Iterable[str],
        interval: float = 1.0,
        callback: CallbackType,
        error_callback: ErrorCallbackType | None = None,
    ) -> None:
        symbols = list(symbols)
        if not symbols:
            raise ValueError("MarketDataPoller wymaga co najmniej jednego symbolu")
        if interval <= 0:
            raise ValueError("Odstęp odpytywania musi być dodatni")
        self._adapter = adapter
        self._symbols = symbols
        self._interval = interval
        self._callback = callback
        self._error_callback = error_callback
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def __aenter__(self) -> "MarketDataPoller":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.stop()

    async def start(self) -> None:
        """Uruchamia pętlę odpytywania, jeśli nie jest aktywna."""

        if self._task is not None and not self._task.done():
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run(), name="market-data-poller")

    async def stop(self) -> None:
        """Zatrzymuje pętlę odpytywania i czeka na jej zakończenie."""

        self._stopped.set()
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                for symbol in self._symbols:
                    if self._stopped.is_set():
                        break
                    try:
                        payload = await self._adapter.fetch_market_data(symbol)
                    except Exception as exc:  # pragma: no cover - logowanie ścieżki błędu
                        logger.warning(
                            "Nie udało się pobrać danych rynku %s: %s", symbol, exc
                        )
                        await self._handle_error(symbol, exc)
                        continue
                    await self._dispatch_callback(symbol, payload)
                if self._stopped.is_set():
                    break
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise
        finally:
            self._task = None

    async def _dispatch_callback(self, symbol: str, payload: MarketPayload) -> None:
        try:
            result = self._callback(symbol, payload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - błędy użytkownika
            logger.exception("Callback danych rynkowych zgłosił wyjątek: %s", exc)

    async def _handle_error(self, symbol: str, error: Exception) -> None:
        if self._error_callback is None:
            return
        try:
            result = self._error_callback(symbol, error)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - błędy callbacku użytkownika
            logger.exception(
                "Callback błędu danych rynkowych zgłosił wyjątek: %s", exc
            )


__all__ = ["MarketDataPoller"]

