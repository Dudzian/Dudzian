import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Sequence

import pytest

from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.runtime.multi_strategy_scheduler import (
    MultiStrategyScheduler,
    StrategyDataFeed,
    StrategySignalSink,
)
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


class DummyStrategy(StrategyEngine):
    def __init__(self) -> None:
        self.snapshots: list[MarketSnapshot] = []

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        self.snapshots.extend(history)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        self.snapshots.append(snapshot)
        return [
            StrategySignal(
                symbol=snapshot.symbol,
                side="buy",
                confidence=0.9,
                metadata={"price": snapshot.close},
            )
        ]


class DummyFeed(StrategyDataFeed):
    def __init__(self, snapshots: Sequence[MarketSnapshot]) -> None:
        self._history = list(snapshots)
        self._queue: deque[MarketSnapshot] = deque(snapshots)

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        return self._history[:bars]

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        if not self._queue:
            return []
        return [self._queue.popleft()]


class DummySink(StrategySignalSink):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[StrategySignal]]] = []

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: Sequence[StrategySignal],
    ) -> None:
        self.calls.append((schedule_name, tuple(signals)))


class DummyCoordinator:
    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.cooldown_seconds = 5.0

    def evaluate(self, *, force: bool = False):
        self.calls.append(force)
        return None


def _snapshot(price: float, ts: int) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC_USDT",
        timestamp=ts,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000.0,
    )


def test_scheduler_dispatches_signals_and_logs_decisions() -> None:
    snapshots = [_snapshot(100.0 + i, 1000 + i) for i in range(5)]
    strategy = DummyStrategy()
    feed = DummyFeed(snapshots)
    sink = DummySink()
    journal = InMemoryTradingDecisionJournal()
    telemetry_calls: list[tuple[str, dict[str, float]]] = []

    def _telemetry(schedule: str, payload: dict[str, float]) -> None:
        telemetry_calls.append((schedule, payload))

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        telemetry_emitter=_telemetry,
        decision_journal=journal,
    )

    scheduler.register_schedule(
        name="mean_reversion_intraday",
        strategy_name="mean_reversion",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=10,
        max_drift_seconds=2,
        warmup_bars=3,
        risk_profile="balanced",
        max_signals=2,
    )

    schedule = scheduler._schedules[0]

    async def _run_once() -> None:
        await scheduler._execute_schedule(schedule, datetime(2024, 1, 1, tzinfo=timezone.utc))

    asyncio.run(_run_once())

    assert sink.calls
    schedule_name, signals = sink.calls[0]
    assert schedule_name == "mean_reversion_intraday"
    assert signals[0].metadata["price"] == pytest.approx(100.0)

    exported = list(journal.export())
    assert exported
    assert exported[0]["strategy"] == "mean_reversion"
    assert exported[0]["schedule"] == "mean_reversion_intraday"
    assert exported[0]["schedule_run_id"].startswith("mean_reversion_intraday")
    assert exported[0]["strategy_instance_id"] == "mean_reversion"
    assert exported[0]["base_asset"] == "BTC"
    assert exported[0]["quote_asset"] == "USDT"
    assert exported[0]["signal_id"].startswith("mean_reversion_intraday")
    assert exported[0]["confidence"] == "0.9"
    assert exported[0]["telemetry_namespace"].endswith("mean_reversion_intraday")
    assert telemetry_calls and telemetry_calls[0][0] == "mean_reversion_intraday"
    telemetry_payload = telemetry_calls[0][1]
    assert telemetry_payload["avg_confidence"] == pytest.approx(0.9)


def test_scheduler_invokes_portfolio_coordinator_once() -> None:
    scheduler = MultiStrategyScheduler(
        environment="paper",
        portfolio="demo",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    coordinator = DummyCoordinator()
    scheduler.attach_portfolio_coordinator(coordinator)

    asyncio.run(scheduler.run_once())

    assert coordinator.calls == [True]
