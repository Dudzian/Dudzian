"""Unit tests for the lightweight auto-trader decision scheduler."""
from __future__ import annotations

import asyncio
import time

from bot_core.auto_trader.decision_scheduler import AutoTraderDecisionScheduler


class _StubTrader:
    def __init__(self) -> None:
        self.invocations: list[float] = []

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


def test_scheduler_background_cycles_and_stops() -> None:
    trader = _StubTrader()
    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)

    scheduler.start_in_background()
    try:
        time.sleep(0.05)
        assert len(trader.invocations) >= 2
    finally:
        scheduler.stop_background()

    invocation_count = len(trader.invocations)
    time.sleep(0.03)
    assert len(trader.invocations) == invocation_count


def test_scheduler_stop_without_start_is_noop() -> None:
    scheduler = AutoTraderDecisionScheduler(_StubTrader(), interval_s=0.01)

    async def _runner() -> None:
        await scheduler.stop()

    asyncio.run(_runner())
    scheduler.stop_background()
