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


def test_suspension_snapshot_includes_metadata() -> None:
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

    secondary_strategy = DummyStrategy()
    secondary_feed = DummyFeed([_snapshot(151.0, 2001)])
    secondary_sink = DummySink()
    scheduler.register_schedule(
        name="backup_schedule",
        strategy_name="backup_engine",
        strategy=secondary_strategy,
        feed=secondary_feed,
        sink=secondary_sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    scheduler.suspend_schedule("mean_schedule", reason="maintenance", duration_seconds=600)
    now["value"] = now["value"] + timedelta(seconds=1)
    scheduler.suspend_schedule("backup_schedule", reason="maintenance", duration_seconds=120)
    now["value"] = now["value"] + timedelta(seconds=1)
    scheduler.suspend_tag("trend", reason="incident")
    now["value"] = now["value"] + timedelta(seconds=1)
    scheduler.suspend_tag("momentum", reason="maintenance", duration_seconds=300)

    snapshot = scheduler.suspension_snapshot()

    assert snapshot["counts"] == {"schedules": 2, "tags": 2, "total": 4}
    assert snapshot["reasons"]["schedules"] == {
        "mean_schedule": "maintenance",
        "backup_schedule": "maintenance",
    }
    assert snapshot["reasons"]["tags"] == {"trend": "incident", "momentum": "maintenance"}

    scope_stats = snapshot["scope_stats"]
    schedule_scope = scope_stats["schedules"]
    assert schedule_scope["total"] == 2
    assert schedule_scope["expiring"] == 2
    assert schedule_scope["indefinite"] == 0
    assert schedule_scope["next_expiration"]["name"] == "backup_schedule"
    assert schedule_scope["next_expiration"]["scope"] == "schedule"
    assert schedule_scope["oldest"]["name"] in {"mean_schedule", "backup_schedule"}
    assert schedule_scope["newest"]["name"] in {"mean_schedule", "backup_schedule"}
    age_stats = schedule_scope["age_stats"]
    assert age_stats["min"] == pytest.approx(2.0)
    assert age_stats["max"] == pytest.approx(3.0)
    assert age_stats["average"] == pytest.approx(2.5)

    tag_scope = scope_stats["tags"]
    assert tag_scope["total"] == 2
    assert tag_scope["expiring"] == 1
    assert tag_scope["indefinite"] == 1
    assert tag_scope["next_expiration"]["name"] == "momentum"
    assert tag_scope["next_expiration"]["scope"] == "tag"
    assert tag_scope["oldest"]["name"] in {"trend", "momentum"}
    assert tag_scope["newest"]["name"] in {"trend", "momentum"}
    tag_age = tag_scope["age_stats"]
    assert tag_age["min"] == pytest.approx(0.0)
    assert tag_age["max"] == pytest.approx(1.0)
    assert tag_age["average"] == pytest.approx(0.5)

    next_expiration = snapshot.get("next_expiration")
    assert next_expiration is not None
    assert next_expiration["scope"] == "schedule"
    assert next_expiration["name"] == "backup_schedule"
    backup_remaining = snapshot["schedules"]["backup_schedule"]["remaining_seconds"]
    assert isinstance(backup_remaining, (int, float))
    assert next_expiration["remaining_seconds"] == pytest.approx(backup_remaining)
    assert backup_remaining < snapshot["schedules"]["mean_schedule"]["remaining_seconds"]

    expiring_entries = snapshot["expiring_entries"]
    assert snapshot["expiring_total"] == 3
    assert [entry["name"] for entry in expiring_entries] == [
        "backup_schedule",
        "momentum",
        "mean_schedule",
    ]
    assert expiring_entries[0]["remaining_seconds"] == pytest.approx(backup_remaining)
    assert expiring_entries[1]["remaining_seconds"] == pytest.approx(300.0, abs=1e-3)
    assert expiring_entries[2]["remaining_seconds"] > expiring_entries[1]["remaining_seconds"]
    assert expiring_entries[0]["scope"] == "schedule"
    assert expiring_entries[1]["scope"] == "tag"
    assert all("age_seconds" in entry for entry in expiring_entries)

    expiration_buckets = snapshot["expiration_buckets"]
    assert expiration_buckets["5m"]["count"] == 2
    assert expiration_buckets["5m"]["next"]["name"] == "backup_schedule"
    assert expiration_buckets["5m"]["scopes"] == ["schedule", "tag"]
    assert expiration_buckets["15m"]["count"] == 3
    assert expiration_buckets["15m"]["next"]["name"] == "backup_schedule"
    assert expiration_buckets["15m"]["last"]["name"] == "mean_schedule"
    assert expiration_buckets["15m"]["scopes"] == ["schedule", "tag"]

    reason_stats = snapshot["reason_stats"]
    maintenance_stats = reason_stats["maintenance"]
    assert maintenance_stats["schedules"] == 2
    assert maintenance_stats["tags"] == 1
    assert maintenance_stats["total"] == 3
    assert maintenance_stats["expiring"] == 3
    assert maintenance_stats["indefinite"] == 0
    assert maintenance_stats["next_expiration"]["name"] == "backup_schedule"
    assert maintenance_stats["next_expiration"]["scope"] == "schedule"
    assert maintenance_stats["next_expiration"]["remaining_seconds"] == pytest.approx(
        backup_remaining
    )
    assert maintenance_stats["scopes"] == ["schedules", "tags"]
    maintenance_age = maintenance_stats["age_stats"]
    assert maintenance_age["min"] == pytest.approx(0.0)
    assert maintenance_age["max"] == pytest.approx(3.0)
    assert maintenance_age["average"] == pytest.approx(5.0 / 3.0)
    entries = maintenance_stats["entries"]
    names = {entry["name"] for entry in entries}
    assert names == {"mean_schedule", "backup_schedule", "momentum"}
    assert all(entry["scope"] in {"schedule", "tag"} for entry in entries)
    assert maintenance_stats["oldest"]["name"] in {"mean_schedule", "backup_schedule"}
    assert maintenance_stats["newest"]["name"] in {"backup_schedule", "momentum"}
    maintenance_breakdown = maintenance_stats["scope_breakdown"]
    assert maintenance_breakdown["schedules"]["expiring"] == 2
    assert maintenance_breakdown["schedules"]["indefinite"] == 0
    assert "oldest" in maintenance_breakdown["schedules"]
    assert "newest" in maintenance_breakdown["schedules"]
    maintenance_schedule_age = maintenance_breakdown["schedules"]["age_stats"]
    assert maintenance_schedule_age["min"] == pytest.approx(2.0)
    assert maintenance_schedule_age["max"] == pytest.approx(3.0)
    assert maintenance_schedule_age["average"] == pytest.approx(2.5)
    assert maintenance_breakdown["tags"]["expiring"] == 1
    assert maintenance_breakdown["tags"]["indefinite"] == 0
    assert maintenance_breakdown["tags"]["next_expiration"]["name"] == "momentum"
    assert maintenance_breakdown["tags"]["next_expiration"]["scope"] == "tag"
    maintenance_tag_age = maintenance_breakdown["tags"]["age_stats"]
    assert maintenance_tag_age["min"] == pytest.approx(0.0)
    assert maintenance_tag_age["max"] == pytest.approx(0.0)
    assert maintenance_tag_age["average"] == pytest.approx(0.0)

    incident_stats = reason_stats["incident"]
    assert incident_stats["schedules"] == 0
    assert incident_stats["tags"] == 1
    assert incident_stats["total"] == 1
    assert incident_stats["expiring"] == 0
    assert incident_stats["indefinite"] == 1
    assert "next_expiration" not in incident_stats
    assert incident_stats["scopes"] == ["tags"]
    assert incident_stats["entries"][0]["name"] == "trend"
    incident_age = incident_stats["age_stats"]
    assert incident_age["min"] == pytest.approx(1.0)
    assert incident_age["max"] == pytest.approx(1.0)
    assert incident_age["average"] == pytest.approx(1.0)
    incident_breakdown = incident_stats["scope_breakdown"]
    assert incident_breakdown["tags"]["expiring"] == 0
    assert incident_breakdown["tags"]["indefinite"] == 1
    assert "next_expiration" not in incident_breakdown["tags"]
    incident_tag_age = incident_breakdown["tags"]["age_stats"]
    assert incident_tag_age["min"] == pytest.approx(1.0)
    assert incident_tag_age["max"] == pytest.approx(1.0)
    assert incident_tag_age["average"] == pytest.approx(1.0)


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

