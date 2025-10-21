"""Asynchroniczne testy integracyjne MultiStrategyScheduler."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from bot_core.runtime.multi_strategy_scheduler import (
    FixedWeightAllocation,
    MultiStrategyScheduler,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
)

from tests.test_multi_strategy_scheduler import (
    DummyFeed,
    DummySink,
    DummyStrategy,
    _snapshot,
)


class FixedAllocator:
    name = "fixed"

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights
        self.calls = 0

    def allocate(self, schedules):
        self.calls += 1
        mapping = {}
        for schedule in schedules:
            mapping[schedule.name] = self.weights.get(schedule.name, 1.0)
        return mapping


def test_scheduler_applies_capital_allocation_and_signal_limits() -> None:
    snapshots_a = [_snapshot(101.0 + i, 2000 + i) for i in range(4)]
    snapshots_b = [_snapshot(200.0 + i, 3000 + i) for i in range(4)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots_a)
    feed_b = DummyFeed(snapshots_b)
    sink = DummySink()
    allocator = FixedAllocator({"trend_schedule": 3.0, "grid_schedule": 1.0})

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=allocator,
        allocation_rebalance_seconds=0.0,
    )
    scheduler.configure_signal_limit("trend_engine", "balanced", 2)

    scheduler.register_schedule(
        name="trend_schedule",
        strategy_name="trend_engine",
        strategy=strategy_a,
        feed=feed_a,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="grid_schedule",
        strategy_name="grid_engine",
        strategy=strategy_b,
        feed=feed_b,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )

    asyncio.run(scheduler.run_once())

    assert allocator.calls >= 1
    trend_ctx, grid_ctx = scheduler._schedules
    assert trend_ctx.active_max_signals == 2
    assert grid_ctx.active_max_signals >= 1
    assert trend_ctx.metrics["allocator_signal_factor"] >= grid_ctx.metrics["allocator_signal_factor"]


def test_risk_profile_budget_allocation_respects_profiles() -> None:
    snapshots = [_snapshot(101.0 + i, 2500 + i) for i in range(3)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    strategy_c = DummyStrategy()
    feed_a = DummyFeed(snapshots)
    feed_b = DummyFeed(snapshots)
    feed_c = DummyFeed(snapshots)
    sink = DummySink()

    policy = RiskProfileBudgetAllocation(
        {"balanced": 0.7, "aggressive": 0.3},
        inner_policy_factory=lambda: SignalStrengthAllocation(),
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=policy,
        allocation_rebalance_seconds=0.0,
    )

    scheduler.register_schedule(
        name="trend_primary",
        strategy_name="trend_engine",
        strategy=strategy_a,
        feed=feed_a,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=5,
    )
    scheduler.register_schedule(
        name="trend_secondary",
        strategy_name="trend_engine",
        strategy=strategy_b,
        feed=feed_b,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=5,
    )
    scheduler.register_schedule(
        name="grid_aggressive",
        strategy_name="grid_engine",
        strategy=strategy_c,
        feed=feed_c,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="aggressive",
        max_signals=5,
    )

    trend_primary, trend_secondary, grid_schedule = scheduler._schedules
    trend_primary.metrics.update({"signals": 6.0, "avg_confidence": 0.85})
    trend_secondary.metrics.update({"signals": 1.0, "avg_confidence": 0.2})
    grid_schedule.metrics.update({"signals": 2.0, "avg_confidence": 0.6})

    asyncio.run(
        scheduler._maybe_rebalance_allocation(datetime(2024, 1, 1, tzinfo=timezone.utc))
    )

    balanced_share = trend_primary.allocator_signal_factor + trend_secondary.allocator_signal_factor
    aggressive_share = grid_schedule.allocator_signal_factor

    assert pytest.approx(balanced_share + aggressive_share) == 1.0
    assert balanced_share == pytest.approx(0.7, rel=1e-3)
    assert aggressive_share == pytest.approx(0.3, rel=1e-3)
    assert trend_primary.allocator_signal_factor > trend_secondary.allocator_signal_factor
    assert trend_primary.metrics["allocator_profile_weight"] == pytest.approx(0.7, rel=1e-3)
    assert grid_schedule.metrics["allocator_profile_weight"] == pytest.approx(0.3, rel=1e-3)


def test_scheduler_run_forever_stops_gracefully() -> None:
    snapshots = [_snapshot(150.0 + i, 4000 + i) for i in range(3)]
    strategy = DummyStrategy()
    feed = DummyFeed(snapshots)
    sink = DummySink()

    class _Clock:
        def __init__(self) -> None:
            self._current = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            current = self._current
            self._current = self._current + timedelta(seconds=1)
            return current

    clock = _Clock()
    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=clock,
        allocation_rebalance_seconds=0.0,
    )
    scheduler.register_schedule(
        name="demo_schedule",
        strategy_name="demo_engine",
        strategy=strategy,
        feed=feed,
        sink=sink,
        cadence_seconds=1,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    async def _run() -> None:
        runner = asyncio.create_task(scheduler.run_forever())
        await asyncio.sleep(0.2)
        scheduler.stop()
        await asyncio.wait_for(runner, timeout=2.0)

    asyncio.run(_run())

    assert sink.calls


def test_fixed_weight_allocation_prefers_profile_weights() -> None:
    snapshots_a = [_snapshot(101.0 + i, 5000 + i) for i in range(2)]
    snapshots_b = [_snapshot(201.0 + i, 6000 + i) for i in range(2)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots_a)
    feed_b = DummyFeed(snapshots_b)
    sink = DummySink()

    allocator = FixedWeightAllocation(
        {
            "trend_engine:balanced": 0.25,
            "grid_schedule": 0.75,
        },
        label="manual",
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=allocator,
        allocation_rebalance_seconds=0.0,
    )

    scheduler.register_schedule(
        name="trend_schedule",
        strategy_name="trend_engine",
        strategy=strategy_a,
        feed=feed_a,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )
    scheduler.register_schedule(
        name="grid_schedule",
        strategy_name="grid_engine",
        strategy=strategy_b,
        feed=feed_b,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    asyncio.run(scheduler.run_once())

    trend_ctx, grid_ctx = scheduler._schedules
    assert allocator.name == "manual"
    assert trend_ctx.allocator_signal_factor < grid_ctx.allocator_signal_factor
    assert trend_ctx.allocator_signal_factor == pytest.approx(0.25, rel=1e-3)
    assert grid_ctx.allocator_signal_factor == pytest.approx(0.75, rel=1e-3)


def test_replace_capital_policy_rebalances_immediately() -> None:
    snapshots = [_snapshot(101.0 + i, 7000 + i) for i in range(3)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots)
    feed_b = DummyFeed(snapshots)
    sink = DummySink()

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        allocation_rebalance_seconds=3600.0,
    )

    scheduler.register_schedule(
        name="trend_schedule",
        strategy_name="trend_engine",
        strategy=strategy_a,
        feed=feed_a,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )
    scheduler.register_schedule(
        name="grid_schedule",
        strategy_name="grid_engine",
        strategy=strategy_b,
        feed=feed_b,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    async def _apply() -> None:
        await scheduler.replace_capital_policy(
            FixedWeightAllocation({
                "trend_schedule": 0.1,
                "grid_schedule": 0.9,
            }),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    asyncio.run(_apply())

    snapshot = scheduler.allocation_snapshot()
    assert snapshot["trend_schedule"] == pytest.approx(0.1, rel=1e-3)
    assert snapshot["grid_schedule"] == pytest.approx(0.9, rel=1e-3)
