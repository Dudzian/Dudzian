"""Decision cycle scheduler for the lightweight auto-trader."""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol


LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .app import AutoTrader, DecisionCycleReport


class AutoTraderSchedulerHooks(Protocol):
    """Lifecycle callbacks invoked by :class:`AutoTraderDecisionScheduler`."""

    def on_bootstrap(self, scheduler: "AutoTraderDecisionScheduler") -> None:  # pragma: no cover - interface only
        """Execute bootstrap logic right before the first cycle."""

    def on_cycle_success(self, report: "DecisionCycleReport") -> None:  # pragma: no cover - interface only
        """Handle a successful decision cycle."""

    def on_cycle_failure(self, exc: BaseException) -> float | None:  # pragma: no cover - interface only
        """Handle a failed cycle and optionally override the restart delay."""


@dataclass(slots=True)
class AutoTraderDecisionScheduler:
    """Asynchronous scheduler driving :class:`AutoTrader` decision cycles."""

    trader: "AutoTrader"
    interval_s: float = 30.0
    loop: asyncio.AbstractEventLoop | None = None
    hooks: AutoTraderSchedulerHooks | None = None
    restart_backoff_s: float = 1.0
    restart_backoff_max_s: float = 30.0
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
            self._invoke_bootstrap_hook()
            retry_delay = 0.0
            base_backoff = max(0.0, float(self.restart_backoff_s) or 0.0)
            max_backoff = max(base_backoff or 1.0, float(self.restart_backoff_max_s) or 1.0)
            while not stop_event.is_set():
                if retry_delay > 0.0:
                    if stop_event.wait(retry_delay):
                        break
                    retry_delay = 0.0
                try:
                    report = self.trader.run_cycle_once()
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.exception("AutoTraderDecisionScheduler cycle failed")
                    retry_delay = self._handle_failure(exc, max_backoff=max_backoff)
                    if retry_delay <= 0.0:
                        retry_delay = min(max_backoff, base_backoff or 1.0)
                    continue
                else:
                    self._handle_success(report)
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
        retry_delay = 0.0
        base_backoff = max(0.0, float(self.restart_backoff_s) or 0.0)
        max_backoff = max(base_backoff or 1.0, float(self.restart_backoff_max_s) or 1.0)
        self._invoke_bootstrap_hook()
        while not stop_event.is_set():
            if retry_delay > 0.0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=retry_delay)
                except asyncio.TimeoutError:
                    pass
                else:
                    break
                retry_delay = 0.0
            try:
                report = await asyncio.to_thread(self.trader.run_cycle_once)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("AutoTraderDecisionScheduler async cycle failed")
                retry_delay = self._handle_failure(exc, max_backoff=max_backoff)
                if retry_delay <= 0.0:
                    retry_delay = min(max_backoff, base_backoff or 1.0)
                continue
            else:
                self._handle_success(report)
                self._drain_model_change_events()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


    def _invoke_bootstrap_hook(self) -> None:
        hooks = self.hooks
        if hooks is None:
            return
        try:
            hooks.on_bootstrap(self)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("AutoTraderDecisionScheduler bootstrap hook failed")

    def _handle_success(self, report: "DecisionCycleReport") -> None:
        hooks = self.hooks
        if hooks is None:
            return
        try:
            hooks.on_cycle_success(report)
        except Exception:  # pragma: no cover - lifecycle hook must not break scheduling
            LOGGER.exception("AutoTraderDecisionScheduler success hook failed")

    def _handle_failure(self, exc: BaseException, *, max_backoff: float) -> float:
        hooks = self.hooks
        delay = max(0.0, float(self.restart_backoff_s) or 0.0)
        if hooks is None:
            return min(delay or max_backoff, max_backoff)
        try:
            override = hooks.on_cycle_failure(exc)
        except Exception:  # pragma: no cover - lifecycle hook must not break scheduling
            LOGGER.exception("AutoTraderDecisionScheduler failure hook failed")
            return min(delay or max_backoff, max_backoff)
        if override is None:
            return min(delay or max_backoff, max_backoff)
        try:
            resolved = max(0.0, float(override))
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            resolved = delay
        if resolved <= 0.0:
            return min(max_backoff, delay or max_backoff)
        return min(resolved, max_backoff)


__all__ = ["AutoTraderDecisionScheduler", "AutoTraderSchedulerHooks"]
