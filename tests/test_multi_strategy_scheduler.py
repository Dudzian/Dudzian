import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
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


def test_scheduler_suspension_and_resume_flow() -> None:
    snapshots = [_snapshot(150.0 + i, 2000 + i) for i in range(3)]
    strategy = DummyStrategy()
    feed = DummyFeed(snapshots)
    sink = DummySink()
    now = {"value": datetime(2024, 1, 1, tzinfo=timezone.utc)}

    def _clock() -> datetime:
        return now["value"]

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=_clock,
    )

    scheduler.register_schedule(
        name="mean_schedule",
        strategy_name="mean_engine",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    scheduler.suspend_schedule("mean_schedule", reason="maintenance", duration_seconds=600)

    asyncio.run(scheduler.run_once())

    assert sink.calls == []
    snapshot = scheduler.suspension_snapshot()
    assert "mean_schedule" in snapshot["schedules"]
    assert snapshot["schedules"]["mean_schedule"]["reason"] == "maintenance"


def test_scheduler_logs_expired_suspensions(caplog: pytest.LogCaptureFixture) -> None:
    snapshots = [_snapshot(150.0, 2000)]
    strategy = DummyStrategy()
    feed = DummyFeed(snapshots)
    sink = DummySink()
    now = {"value": datetime(2024, 1, 1, tzinfo=timezone.utc)}

    def _clock() -> datetime:
        return now["value"]

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=_clock,
    )

    scheduler.register_schedule(
        name="mean_schedule",
        strategy_name="mean_engine",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    scheduler.suspend_schedule("mean_schedule", reason="maintenance", duration_seconds=30)

    caplog.set_level(logging.INFO)
    caplog.clear()

    now["value"] = now["value"] + timedelta(minutes=5)

    snapshot = scheduler.suspension_snapshot()

    assert "mean_schedule" not in snapshot["schedules"]
    assert (
        "automatycznie wznowiony po wygaśnięciu zawieszenia" in caplog.text
    )
    assert "maintenance" in caplog.text
    assert "mean_schedule" in caplog.text


def test_scheduler_updates_allocation_interval() -> None:
    now = {"value": datetime(2024, 1, 1, tzinfo=timezone.utc)}

    def _clock() -> datetime:
        return now["value"]

    strategy = DummyStrategy()
    feed = DummyFeed([_snapshot(100.0, 1500.0)])
    sink = DummySink()

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=_clock,
    )

    scheduler.set_allocation_rebalance_seconds(180)
    assert scheduler._allocation_rebalance_seconds == pytest.approx(180.0)

    scheduler.set_allocation_rebalance_seconds(0)
    assert scheduler._allocation_rebalance_seconds is None

    scheduler.set_allocation_rebalance_seconds(None)
    assert scheduler._allocation_rebalance_seconds is None

    scheduler.set_allocation_rebalance_seconds("invalid")  # type: ignore[arg-type]
    assert scheduler._allocation_rebalance_seconds is None

    scheduler.register_schedule(
        name="mean_schedule",
        strategy_name="mean_engine",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=10,
        max_drift_seconds=2,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=2,
    )

    scheduler.suspend_schedule("mean_schedule", reason="cooldown", duration_seconds=60)
    scheduler.resume_schedule("mean_schedule")
    now["value"] = now["value"] + timedelta(minutes=20)

    asyncio.run(scheduler.run_once())

    assert sink.calls, "Strategia powinna zostać wznowiona i wysłać sygnał"
    resumed_snapshot = scheduler.suspension_snapshot()
    assert "mean_schedule" not in resumed_snapshot["schedules"]


def test_describe_schedules_returns_metadata() -> None:
    strategy = DummyStrategy()
    strategy.metadata = {"tags": ["trend"], "primary_tag": "trend"}
    feed = DummyFeed([_snapshot(200.0, 3000)])
    sink = DummySink()
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: now,
    )

    scheduler.register_schedule(
        name="trend_schedule",
        strategy_name="trend_engine",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=15,
        max_drift_seconds=3,
        warmup_bars=5,
        risk_profile="balanced",
        max_signals=4,
    )

    scheduler.configure_signal_limit("trend_engine", "balanced", 6)
    scheduler.suspend_tag("trend", reason="cooldown", duration_seconds=120)

    schedule = scheduler._schedules[0]
    schedule.portfolio_weight = 0.35
    schedule.allocator_weight = 0.55
    schedule.allocator_signal_factor = 0.9
    schedule.governor_signal_factor = 0.8
    schedule.last_run = now
    schedule.warmed_up = True
    schedule.metrics["signals"] = 2

    descriptions = scheduler.describe_schedules()

    assert "trend_schedule" in descriptions
    entry = descriptions["trend_schedule"]
    assert entry["strategy_name"] == "trend_engine"
    assert entry["risk_profile"] == "balanced"
    assert entry["cadence_seconds"] == pytest.approx(15.0)
    assert entry["base_max_signals"] == 4
    assert entry["active_max_signals"] == 4
    assert entry["signal_limit_override"] == 6
    details = entry["signal_limit_details"]
    assert details["limit"] == 6
    assert details["active"] is True
    assert entry["allocator_weight"] == pytest.approx(0.55)
    assert entry["portfolio_weight"] == pytest.approx(0.35)
    assert entry["warmed_up"] is True
    assert entry["tags"] == ["trend"]
    assert entry["active_suspension"]["reason"] == "cooldown"
    assert "metrics" in entry and entry["metrics"]["signals"] == pytest.approx(2.0)
    assert entry["last_run"] == now.isoformat()


def test_signal_limit_snapshot_returns_nested_mapping() -> None:
    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        signal_limits={
            "trend": {"balanced": 3},
        },
    )

    scheduler.configure_signal_limit("grid", "aggressive", 2)
    scheduler.configure_signal_limit("trend", "balanced", 4)
    scheduler.configure_signal_limit("grid", "aggressive", None)

    snapshot = scheduler.signal_limit_snapshot()

    assert "trend" in snapshot
    assert "balanced" in snapshot["trend"]
    entry = snapshot["trend"]["balanced"]
    assert entry["limit"] == 4
    assert entry["active"] is True
    snapshot["trend"]["balanced"]["limit"] = 10
    assert (
        scheduler.signal_limit_snapshot()["trend"]["balanced"]["limit"]
        == 4
    )


def test_signal_limit_override_expires_and_is_purged() -> None:
    now = {"value": datetime(2024, 1, 1, tzinfo=timezone.utc)}

    def _clock() -> datetime:
        return now["value"]

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=_clock,
    )

    scheduler.configure_signal_limit(
        "trend",
        "balanced",
        5,
        reason="manual_adjustment",
        duration_seconds=90.0,
    )

    first_snapshot = scheduler.signal_limit_snapshot()
    entry = first_snapshot["trend"]["balanced"]
    assert entry["limit"] == 5
    assert entry["reason"] == "manual_adjustment"
    assert entry["active"] is True
    assert entry["remaining_seconds"] == pytest.approx(90.0)

    now["value"] = now["value"] + timedelta(seconds=200)

    second_snapshot = scheduler.signal_limit_snapshot()
    assert second_snapshot == {}



def test_expired_signal_limit_restores_schedule_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    now = {"value": datetime(2024, 1, 1, tzinfo=timezone.utc)}

    def _clock() -> datetime:
        return now["value"]

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=_clock,
    )

    strategy = DummyStrategy()
    feed = DummyFeed([_snapshot(150.0, 2000)])
    sink = DummySink()

    scheduler.register_schedule(
        name="trend_schedule",
        strategy_name="trend",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=10,
        max_drift_seconds=2,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=5,
    )

    schedule = scheduler._schedules[0]
    scheduler.configure_signal_limit(
        "trend",
        "balanced",
        2,
        reason="temporary_guard",
        duration_seconds=30,
    )
    scheduler._apply_signal_limits(schedule)

    assert schedule.active_max_signals == 2

    now["value"] = now["value"] + timedelta(seconds=120)

    with caplog.at_level("INFO"):
        snapshot = scheduler.signal_limit_snapshot()

    assert snapshot == {}
    assert schedule.active_max_signals == 5
    assert any(
        "Wygasło nadpisanie limitu sygnałów trend/balanced" in record.message
        for record in caplog.records
    )

