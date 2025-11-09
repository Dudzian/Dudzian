"""Decision cycle scheduler for the lightweight auto-trader."""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .app import AutoTrader


@dataclass(slots=True)
class AutoTraderDecisionScheduler:
    """Asynchronous scheduler driving :class:`AutoTrader` decision cycles."""

    trader: "AutoTrader"
    interval_s: float = 30.0
    loop: asyncio.AbstractEventLoop | None = None
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stop_event: asyncio.Event | None = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _thread_stop: threading.Event | None = field(init=False, default=None, repr=False)

    async def start(self) -> None:
        """Start the scheduler inside an asyncio event loop."""

        if self._task is not None and not self._task.done():
            return
        loop = self.loop or asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._task = loop.create_task(self._run_async())

    async def stop(self) -> None:
        """Request shutdown of the asynchronous scheduler and await completion."""

        task = self._task
        stop_event = self._stop_event
        if task is None or stop_event is None:
            return
        stop_event.set()
        try:
            await task
        finally:
            self._task = None
            self._stop_event = None

    def start_in_background(self) -> None:
        """Spawn a lightweight background thread executing decision cycles."""

        if self._thread is not None and self._thread.is_alive():
            return

        stop_event = threading.Event()
        self._thread_stop = stop_event

        def _worker() -> None:
            while not stop_event.is_set():
                try:
                    self.trader.run_cycle_once()
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.exception("AutoTraderDecisionScheduler cycle failed")
                else:
                    self._drain_model_change_events()
                if stop_event.wait(max(0.0, float(self.interval_s))):
                    break

        thread = threading.Thread(
            target=_worker,
            name="AutoTraderDecisionScheduler",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop_background(self) -> None:
        """Stop the background thread variant of the scheduler."""

        stop_event = self._thread_stop
        if stop_event is not None:
            stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, float(self.interval_s)))
        self._thread = None
        self._thread_stop = None

    def _drain_model_change_events(self) -> None:
        poll = getattr(self.trader, "poll_model_change_event", None)
        emitter = getattr(self.trader, "emitter", None)
        emit = getattr(emitter, "emit", None) if emitter is not None else None
        if not callable(poll) or not callable(emit):
            return
        while True:
            try:
                event = poll()
            except Exception:  # pragma: no cover - defensywne logowanie
                LOGGER.debug("Nie udało się pobrać zdarzenia model_changed", exc_info=True)
                break
            if not event:
                break
            try:
                emit("auto_trader.model_changed", **dict(event))
            except Exception:  # pragma: no cover - zdarzenie nie powinno zatrzymać pętli
                LOGGER.exception("AutoTraderDecisionScheduler nie mógł wyemitować zdarzenia model_changed")
                break

    async def _run_async(self) -> None:
        assert self._stop_event is not None
        stop_event = self._stop_event
        interval = max(0.0, float(self.interval_s))
        while not stop_event.is_set():
            try:
                await asyncio.to_thread(self.trader.run_cycle_once)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception("AutoTraderDecisionScheduler async cycle failed")
            else:
                self._drain_model_change_events()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


__all__ = ["AutoTraderDecisionScheduler"]
