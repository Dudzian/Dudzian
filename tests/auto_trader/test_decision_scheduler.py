"""Unit tests for the lightweight auto-trader decision scheduler."""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

from bot_core.auto_trader.decision_scheduler import AutoTraderDecisionScheduler
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, TradingDecisionEvent


class _StubTrader:
    def __init__(self) -> None:
        self.invocations: list[float] = []

    def run_cycle(self, _request=None) -> None:  # noqa: ANN001 - interfejs schedulera
        self.run_cycle_once()

    def run_cycle_once(self) -> None:
        self.invocations.append(time.monotonic())


def test_scheduler_async_runs_until_stop() -> None:
    trader = _StubTrader()
    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)

    async def _runner() -> None:
        await scheduler.start()
        await asyncio.sleep(0.05)
        await scheduler.stop()

        invocation_count = len(trader.invocations)
        await asyncio.sleep(0.03)
        assert len(trader.invocations) == invocation_count

    asyncio.run(_runner())


def _wait_for_invocations(
    trader: _StubTrader,
    minimum: int,
    *,
    interval_s: float,
    timeout_s: float | None = None,
) -> None:
    if minimum <= 0:
        return

    resolved_interval_s = max(0.0, float(interval_s))
    if timeout_s is None:
        minimum_cycles_budget_s = max(0.0, float(minimum)) * resolved_interval_s * 2.0
        guard_budget_s = max(0.25, min(5.0, 20.0 * resolved_interval_s))
        resolved_timeout_s = max(minimum_cycles_budget_s, guard_budget_s)
    else:
        resolved_timeout_s = max(0.0, float(timeout_s))
    poll_sleep_s = min(0.05, max(0.001, resolved_interval_s / 4.0))
    deadline = time.monotonic() + resolved_timeout_s
    while len(trader.invocations) < minimum and time.monotonic() < deadline:
        time.sleep(poll_sleep_s)
    assert len(trader.invocations) >= minimum, (
        "Expected >= "
        f"{minimum} invocations within {resolved_timeout_s:.2f}s "
        f"(interval_s={resolved_interval_s:.3f}), got {len(trader.invocations)}"
    )


def test_scheduler_background_cycles_and_stops() -> None:
    trader = _StubTrader()
    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)

    scheduler.start_in_background()
    background_thread: threading.Thread | None = None
    try:
        _wait_for_invocations(trader, 2, interval_s=scheduler.interval_s)
        background_thread = scheduler._thread
    finally:
        scheduler.stop_background()

    invocation_count = len(trader.invocations)
    if background_thread is not None:
        background_thread.join(timeout=0.5)
    assert scheduler._thread is None
    assert background_thread is None or not background_thread.is_alive()
    time.sleep(min(0.05, max(0.01, 3.0 * scheduler.interval_s)))
    assert len(trader.invocations) == invocation_count


def test_scheduler_stop_without_start_is_noop() -> None:
    scheduler = AutoTraderDecisionScheduler(_StubTrader(), interval_s=0.01)

    async def _runner() -> None:
        await scheduler.stop()

    asyncio.run(_runner())
    scheduler.stop_background()


class _JournalTrader(_StubTrader):
    def __init__(self) -> None:
        super().__init__()
        self.journal = InMemoryTradingDecisionJournal()
        self._decision_journal = self.journal
        self.started = threading.Event()

    def run_cycle(self, _request=None) -> None:  # noqa: ANN001 - interfejs schedulera
        self.started.set()
        time.sleep(0.05)
        self.journal.record(
            TradingDecisionEvent(
                event_type="order_filled",
                timestamp=datetime.now(timezone.utc),
                environment="test",
                portfolio="demo",
                risk_profile="paper",
            )
        )


def test_scheduler_stop_waits_for_inflight_cycle() -> None:
    trader = _JournalTrader()
    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)

    scheduler.start_in_background()
    try:
        assert trader.started.wait(timeout=0.2)
        stop_started = time.perf_counter()
        scheduler.stop_background()
        stop_elapsed = time.perf_counter() - stop_started
    finally:
        scheduler.stop_background()

    events = list(trader.journal.export())
    assert any(entry.get("event") == "order_filled" for entry in events)
    assert stop_elapsed >= 0.04
