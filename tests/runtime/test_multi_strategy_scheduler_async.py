"""Asynchroniczne testy integracyjne MultiStrategyScheduler."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from bot_core.runtime.multi_strategy_scheduler import (
    DrawdownAdaptiveAllocation,
    FixedWeightAllocation,
    MetricWeightedAllocation,
    MetricWeightRule,
    MultiStrategyScheduler,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
    SmoothedCapitalAllocationPolicy,
    TagQuotaAllocation,
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


class OscillatingAllocator:
    name = "oscillating"

    def __init__(self) -> None:
        self._toggle = False

    def allocate(self, schedules):
        self._toggle = not self._toggle
        if self._toggle:
            return {
                schedule.name: 4.0 if schedule.name == "trend_schedule" else 1.0
                for schedule in schedules
            }
        return {
            schedule.name: 1.0 if schedule.name == "trend_schedule" else 4.0
            for schedule in schedules
        }


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


def test_tag_quota_policy_updates_scheduler_state() -> None:
    snapshots = [_snapshot(101.0 + i, 2000 + i) for i in range(3)]
    trend_a = DummyStrategy()
    trend_b = DummyStrategy()
    mean = DummyStrategy()
    orphan = DummyStrategy()

    trend_a.metadata = {"tags": ("trend", "core"), "primary_tag": "trend"}
    trend_b.metadata = {"tags": ("trend",), "primary_tag": "trend"}
    mean.metadata = {"tags": ("mean",), "primary_tag": "mean"}
    orphan.metadata = {"tags": tuple()}

    sink = DummySink()

    policy = TagQuotaAllocation(
        {"trend": 2.0, "mean": 1.0},
        default_weight=1.0,
        label="tag_mix",
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
        strategy_name="trend_a",
        strategy=trend_a,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="trend_secondary",
        strategy_name="trend_b",
        strategy=trend_b,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="mean_schedule",
        strategy_name="mean_engine",
        strategy=mean,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="orphan_schedule",
        strategy_name="grid_engine",
        strategy=orphan,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )

    asyncio.run(scheduler.run_once())

    state = scheduler.capital_allocation_state()
    assert state["tags"]["trend"] == pytest.approx(0.5, rel=1e-3)
    assert state["tags"]["mean"] == pytest.approx(0.25, rel=1e-3)
    assert state["tags"]["unassigned"] == pytest.approx(0.25, rel=1e-3)

    diag = scheduler.capital_policy_diagnostics()
    assert diag["tag_weights"]["trend"] == pytest.approx(0.5, rel=1e-3)
    assert diag["tag_members"]["trend"] == pytest.approx(2.0)
    assert not diag["flags"].get("fallback_used", False)

    trend_primary = next(ctx for ctx in scheduler._schedules if ctx.name == "trend_primary")
    assert trend_primary.tags == ("trend", "core")
    assert trend_primary.metrics["allocator_tag_weight"] == pytest.approx(0.5, rel=1e-3)
    assert trend_primary.metrics["allocator_tag_members"] == pytest.approx(2.0)


def test_suspend_tag_blocks_all_members() -> None:
    snapshots = [_snapshot(101.0 + i, 2000 + i) for i in range(4)]
    trend_a = DummyStrategy()
    trend_b = DummyStrategy()
    mean = DummyStrategy()

    trend_a.metadata = {"tags": ("trend",), "primary_tag": "trend"}
    trend_b.metadata = {"tags": ("trend", "beta"), "primary_tag": "trend"}
    mean.metadata = {"tags": ("mean",), "primary_tag": "mean"}

    sink = DummySink()

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    scheduler.register_schedule(
        name="trend_primary",
        strategy_name="trend_a",
        strategy=trend_a,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="trend_secondary",
        strategy_name="trend_b",
        strategy=trend_b,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(
        name="mean_schedule",
        strategy_name="mean_engine",
        strategy=mean,
        feed=DummyFeed(list(snapshots)),
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )

    scheduler.suspend_tag("trend", reason="risk guard")

    asyncio.run(scheduler.run_once())

    dispatched_names = [name for name, _ in sink.calls]
    assert "mean_schedule" in dispatched_names
    assert "trend_primary" not in dispatched_names
    assert "trend_secondary" not in dispatched_names

    snapshot = scheduler.suspension_snapshot()
    assert "trend" in snapshot["tags"]
    assert snapshot["tags"]["trend"]["origin"] == "tag"

    trend_primary_ctx, trend_secondary_ctx, mean_ctx = scheduler._schedules
    assert trend_primary_ctx.metrics.get("suspended") == 1.0
    assert trend_secondary_ctx.metrics.get("suspended") == 1.0
    assert mean_ctx.metrics.get("suspended") != 1.0

    sink.calls.clear()
    scheduler.resume_tag("trend")

    asyncio.run(scheduler.run_once())

    dispatched_names = [name for name, _ in sink.calls]
    assert "trend_primary" in dispatched_names
    assert "trend_secondary" in dispatched_names


def test_smoothed_policy_exposes_snapshots() -> None:
    snapshots = [_snapshot(101.0 + i, 2000 + i) for i in range(3)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots)
    feed_b = DummyFeed(snapshots)
    sink = DummySink()

    policy = SmoothedCapitalAllocationPolicy(
        OscillatingAllocator(),
        smoothing_factor=0.5,
        min_delta=0.0,
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=policy,
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
        risk_profile="aggressive",
        max_signals=4,
    )

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    asyncio.run(scheduler._maybe_rebalance_allocation(now))

    first_state = scheduler.capital_allocation_state()
    assert first_state["raw"]["trend_schedule"] == pytest.approx(0.8)
    assert first_state["effective"]["trend_schedule"] == pytest.approx(0.8)
    assert first_state["profiles"]["balanced"] == pytest.approx(0.8)

    later = now + timedelta(seconds=1)
    asyncio.run(scheduler._maybe_rebalance_allocation(later))

    state = scheduler.capital_allocation_state()
    assert state["raw"]["trend_schedule"] == pytest.approx(0.2)
    assert state["effective"]["trend_schedule"] == pytest.approx(0.5)
    assert state["smoothed"]["grid_schedule"] == pytest.approx(0.5)
    assert state["profiles"]["aggressive"] == pytest.approx(0.5)

    trend_ctx, grid_ctx = scheduler._schedules
    assert trend_ctx.metrics["allocator_raw_weight"] == pytest.approx(0.2)
    assert grid_ctx.metrics["allocator_raw_weight"] == pytest.approx(0.8)
    assert trend_ctx.metrics["allocator_smoothed_weight"] == pytest.approx(0.5)
    assert grid_ctx.metrics["allocator_smoothed_weight"] == pytest.approx(0.5)

    diagnostics = scheduler.capital_policy_diagnostics()
    assert diagnostics["policy_name"] == "smoothed"
    assert diagnostics["flags"] == {}
    assert diagnostics["details"] == {}


def test_metric_weighted_policy_emits_diagnostics() -> None:
    snapshots = [_snapshot(101.0 + i, 2000 + i) for i in range(3)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots)
    feed_b = DummyFeed(snapshots)
    sink = DummySink()

    policy = MetricWeightedAllocation(
        (
            MetricWeightRule("avg_confidence", weight=1.2, clamp_min=0.0, clamp_max=1.0),
            MetricWeightRule("last_latency_ms", weight=-0.015, clamp_min=0.0, default=120.0),
        ),
        default_score=0.05,
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=policy,
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
        risk_profile="aggressive",
        max_signals=4,
    )

    trend_ctx, grid_ctx = scheduler._schedules
    trend_ctx.metrics["avg_confidence"] = 0.95
    trend_ctx.metrics["last_latency_ms"] = 35.0
    grid_ctx.metrics["avg_confidence"] = 0.62
    grid_ctx.metrics["last_latency_ms"] = 210.0

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    asyncio.run(scheduler._maybe_rebalance_allocation(now))

    state = scheduler.capital_allocation_state()["effective"]
    assert state["trend_schedule"] > state["grid_schedule"]

    diagnostics = scheduler.capital_policy_diagnostics()
    assert diagnostics["policy_name"] == "metric_weighted"
    details = diagnostics["details"]
    assert details["trend_schedule"]["metric:avg_confidence"] == pytest.approx(0.95)
    assert details["grid_schedule"]["metric:last_latency_ms"] == pytest.approx(210.0)
    assert details["grid_schedule"]["contribution:last_latency_ms"] < 0.0


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


def test_drawdown_policy_diagnostics_are_exposed() -> None:
    snapshots = [_snapshot(101.0 + i, 2000 + i) for i in range(3)]
    strategy_a = DummyStrategy()
    strategy_b = DummyStrategy()
    feed_a = DummyFeed(snapshots)
    feed_b = DummyFeed(snapshots)
    sink = DummySink()

    policy = DrawdownAdaptiveAllocation(
        warning_drawdown_pct=5.0,
        panic_drawdown_pct=15.0,
        pressure_weight=0.6,
        min_weight=0.05,
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="paper",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        capital_policy=policy,
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
        max_signals=4,
    )
    scheduler.register_schedule(
        name="hedge_schedule",
        strategy_name="hedge_engine",
        strategy=strategy_b,
        feed=feed_b,
        sink=sink,
        cadence_seconds=5,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )

    trend_ctx, hedge_ctx = scheduler._schedules
    trend_ctx.metrics.update({"max_drawdown_pct": 15.0, "drawdown_pressure": 0.7})
    hedge_ctx.metrics.update({"max_drawdown_pct": 2.0, "drawdown_pressure": 0.1})

    asyncio.run(scheduler._maybe_rebalance_allocation(datetime(2024, 1, 1, tzinfo=timezone.utc)))

    state = scheduler.capital_allocation_state()
    assert state["raw"] == {}
    assert state["effective"]["trend_schedule"] < state["effective"]["hedge_schedule"]

    diagnostics = scheduler.capital_policy_diagnostics()
    assert diagnostics["policy_name"] == "drawdown_adaptive"
    details = diagnostics["details"]
    assert details["trend_schedule"]["drawdown_pct"] == pytest.approx(15.0)
    assert details["trend_schedule"]["penalty"] > details["hedge_schedule"]["penalty"]


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
