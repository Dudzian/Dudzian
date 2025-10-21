"""Testy pomocniczych funkcji polityk kapitału w pipeline."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping, Sequence

import pytest

from bot_core.runtime.capital_policies import (
    BlendedCapitalAllocation,
    DrawdownAdaptiveAllocation,
    EqualWeightAllocation,
    FixedWeightAllocation,
    MetricWeightedAllocation,
    MetricWeightRule,
    RiskParityAllocation,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
    SmoothedCapitalAllocationPolicy,
    TagQuotaAllocation,
    VolatilityTargetAllocation,
)
from bot_core.runtime.pipeline import (
    _collect_fixed_weight_entries,
    _collect_profile_weight_entries,
    _collect_tag_weight_entries,
    _resolve_capital_policy,
)


def test_collect_fixed_weight_entries_handles_nested_dicts() -> None:
    weights = _collect_fixed_weight_entries(
        {
            "strategies": {
                "trend_engine": {"balanced": 0.6},
                "grid_engine": 0.4,
            },
            "schedules": {"mean_schedule": 0.2},
        }
    )
    assert weights["trend_engine:balanced"] == 0.6
    assert weights["grid_engine"] == 0.4
    assert weights["mean_schedule"] == 0.2


def test_collect_profile_weight_entries_supports_sequence_and_mapping() -> None:
    mapping = {"balanced": 0.7, "aggressive": 0.3}
    assert _collect_profile_weight_entries(mapping) == {"balanced": 0.7, "aggressive": 0.3}

    sequence = [
        {"profile": "balanced", "weight": 0.55},
        {"name": "aggressive", "weight": 0.45},
    ]
    collected = _collect_profile_weight_entries(sequence)
    assert collected["balanced"] == pytest.approx(0.55)
    assert collected["aggressive"] == pytest.approx(0.45)


def test_collect_tag_weight_entries_handles_sequence_and_mapping() -> None:
    mapping = {"trend": 0.6, "grid": 0.4}
    assert _collect_tag_weight_entries(mapping) == {"trend": 0.6, "grid": 0.4}

    sequence = [
        {"tag": "trend", "weight": 0.5},
        {"name": "mean", "weight": 0.3},
        {"label": "arb", "weight": 0.2},
    ]
    collected = _collect_tag_weight_entries(sequence)
    assert collected["trend"] == pytest.approx(0.5)
    assert collected["mean"] == pytest.approx(0.3)
    assert collected["arb"] == pytest.approx(0.2)


def test_resolve_capital_policy_fixed_weight_returns_allocator() -> None:
    spec = {
        "name": "fixed_weight",
        "label": "manual_core",
        "rebalance_seconds": 42,
        "weights": {
            "trend_engine": {"balanced": 0.2},
            "grid_schedule": 0.8,
        },
    }
    policy, interval = _resolve_capital_policy(spec)
    assert isinstance(policy, FixedWeightAllocation)
    assert policy.name == "manual_core"
    assert interval == 42

    schedules = [
        SimpleNamespace(name="trend_schedule", strategy_name="trend_engine", risk_profile="balanced"),
        SimpleNamespace(name="grid_schedule", strategy_name="grid_engine", risk_profile="balanced"),
    ]

    allocation = policy.allocate(schedules)
    assert allocation["trend_schedule"] == pytest.approx(0.2)
    assert allocation["grid_schedule"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "policy_name,expected_type",
    [
        ("equal_weight", EqualWeightAllocation),
        ("volatility_target", VolatilityTargetAllocation),
        ("signal_strength", SignalStrengthAllocation),
        ("risk_parity", RiskParityAllocation),
    ],
)
def test_resolve_capital_policy_known_names(policy_name: str, expected_type: type) -> None:
    policy, interval = _resolve_capital_policy(policy_name)
    assert isinstance(policy, expected_type)
    assert interval is None


def test_resolve_capital_policy_risk_profile_budget() -> None:
    spec = {
        "name": "risk_profile",
        "profiles": {"balanced": 0.7, "aggressive": 0.3},
        "profile_floor": 0.1,
        "within_profile": "signal_strength",
    }
    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, RiskProfileBudgetAllocation)
    assert policy.name == "risk_profile"
    assert interval is None

    schedules = [
        SimpleNamespace(
            name="trend_high",
            strategy_name="trend_engine",
            risk_profile="balanced",
            metrics={"signals": 6.0, "avg_confidence": 0.9},
        ),
        SimpleNamespace(
            name="trend_low",
            strategy_name="trend_engine",
            risk_profile="balanced",
            metrics={"signals": 1.0, "avg_confidence": 0.3},
        ),
        SimpleNamespace(
            name="grid",
            strategy_name="grid_engine",
            risk_profile="aggressive",
            metrics={"signals": 2.0, "avg_confidence": 0.6},
        ),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    balanced_share = allocation["trend_high"] + allocation["trend_low"]
    assert balanced_share == pytest.approx(0.7, rel=1e-3)
    assert allocation["trend_high"] > allocation["trend_low"]
    assert allocation["grid"] == pytest.approx(0.3, rel=1e-3)

    snapshot = policy.profile_allocation_snapshot()
    assert snapshot["balanced"] == pytest.approx(0.7, rel=1e-3)
    assert snapshot["aggressive"] == pytest.approx(0.3, rel=1e-3)
    assert not policy.floor_adjustment_applied


def test_resolve_capital_policy_metric_weighted() -> None:
    spec = {
        "name": "metric_weighted",
        "label": "telemetry_mix",
        "default_score": 0.1,
        "metrics": {
            "avg_confidence": {"weight": 1.4, "min": 0.0, "max": 1.0},
            "last_latency_ms": {"weight": -0.02, "min": 0.0, "default": 150.0},
        },
        "fallback": "signal_strength",
    }

    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, MetricWeightedAllocation)
    assert policy.name == "telemetry_mix"
    assert interval is None
    assert len(policy.metrics) == 2

    schedules = [
        SimpleNamespace(
            name="trend",
            strategy_name="trend_engine",
            metrics={"avg_confidence": 0.9, "last_latency_ms": 80.0},
        ),
        SimpleNamespace(
            name="grid",
            strategy_name="grid_engine",
            metrics={"avg_confidence": 0.55, "last_latency_ms": 220.0},
        ),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend"] > allocation["grid"]

    diagnostics = policy.allocation_diagnostics()
    trend_diag = diagnostics["trend"]
    grid_diag = diagnostics["grid"]
    assert trend_diag["metric:avg_confidence"] == pytest.approx(0.9)
    assert grid_diag["metric:last_latency_ms"] == pytest.approx(220.0)
    assert trend_diag["shift"] >= 0.0
    assert grid_diag["contribution:last_latency_ms"] < 0.0


def test_resolve_capital_policy_tag_quota() -> None:
    spec = {
        "name": "tag_quota",
        "label": "tag_mix",
        "tags": {"trend": 2.0, "mean": 1.0},
        "default_weight": 1.0,
        "within_tag": "equal_weight",
    }

    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, TagQuotaAllocation)
    assert policy.name == "tag_mix"
    assert interval is None

    schedules = [
        SimpleNamespace(
            name="trend_a",
            strategy_name="trend_engine",
            tags=("trend", "core"),
            primary_tag="trend",
            metrics={},
        ),
        SimpleNamespace(
            name="trend_b",
            strategy_name="trend_engine",
            tags=("trend",),
            primary_tag="trend",
            metrics={},
        ),
        SimpleNamespace(
            name="mean_a",
            strategy_name="mean_engine",
            tags=("mean",),
            primary_tag="mean",
            metrics={},
        ),
        SimpleNamespace(
            name="orphan",
            strategy_name="grid_engine",
            tags=tuple(),
            primary_tag=None,
            metrics={},
        ),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend_a"] == pytest.approx(0.25, rel=1e-3)
    assert allocation["trend_b"] == pytest.approx(0.25, rel=1e-3)
    assert allocation["mean_a"] == pytest.approx(0.25, rel=1e-3)
    assert allocation["orphan"] == pytest.approx(0.25, rel=1e-3)

    tags_snapshot = policy.tag_allocation_snapshot()
    assert tags_snapshot["trend"] == pytest.approx(0.5, rel=1e-3)
    assert tags_snapshot["mean"] == pytest.approx(0.25, rel=1e-3)
    assert tags_snapshot["unassigned"] == pytest.approx(0.25, rel=1e-3)

    members_snapshot = policy.tag_member_snapshot()
    assert members_snapshot["trend"] == pytest.approx(2.0)
    assert members_snapshot["mean"] == pytest.approx(1.0)
    assert members_snapshot["unassigned"] == pytest.approx(1.0)
    assert not policy.used_fallback


def test_tag_quota_policy_uses_fallback_when_no_tags() -> None:
    spec = {"name": "tag_quota", "fallback": "equal_weight"}
    policy, _ = _resolve_capital_policy(spec)

    schedules = [
        SimpleNamespace(name="trend", strategy_name="trend", tags=tuple(), primary_tag=None, metrics={}),
        SimpleNamespace(name="grid", strategy_name="grid", tags=tuple(), primary_tag=None, metrics={}),
    ]

    allocation = policy.allocate(schedules)
    assert pytest.approx(allocation["trend"]) == 0.5
    assert pytest.approx(allocation["grid"]) == 0.5
    assert policy.used_fallback


def test_risk_profile_budget_respects_floor_constraints() -> None:
    policy = RiskProfileBudgetAllocation(
        {"balanced": 0.7, "aggressive": 0.2, "defensive": 0.1},
        profile_floor=0.25,
        inner_policy_factory=lambda: EqualWeightAllocation(),
    )
    schedules = [
        SimpleNamespace(name="trend_a", strategy_name="trend", risk_profile="balanced"),
        SimpleNamespace(name="trend_b", strategy_name="trend", risk_profile="balanced"),
        SimpleNamespace(name="grid", strategy_name="grid", risk_profile="aggressive"),
        SimpleNamespace(name="vol", strategy_name="vol", risk_profile="defensive"),
    ]

    allocation = policy.allocate(schedules)

    per_profile: dict[str, float] = {"balanced": 0.0, "aggressive": 0.0, "defensive": 0.0}
    for schedule in schedules:
        per_profile[schedule.risk_profile] += allocation.get(schedule.name, 0.0)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert per_profile["balanced"] >= 0.25 - 1e-6
    assert per_profile["aggressive"] >= 0.25 - 1e-6
    assert per_profile["defensive"] >= 0.25 - 1e-6

    aggressive_weight = allocation["grid"]
    defensive_weight = allocation["vol"]
    assert aggressive_weight == pytest.approx(per_profile["aggressive"], rel=1e-3)
    assert defensive_weight == pytest.approx(per_profile["defensive"], rel=1e-3)

    capped_policy = RiskProfileBudgetAllocation(
        {"balanced": 0.8, "aggressive": 0.2},
        profile_floor=0.6,
        inner_policy_factory=lambda: EqualWeightAllocation(),
    )
    capped_schedules = [
        SimpleNamespace(name="trend", strategy_name="trend", risk_profile="balanced"),
        SimpleNamespace(name="grid", strategy_name="grid", risk_profile="aggressive"),
    ]

    capped_allocation = capped_policy.allocate(capped_schedules)

    assert capped_allocation["trend"] == pytest.approx(0.5, rel=1e-3)
    assert capped_allocation["grid"] == pytest.approx(0.5, rel=1e-3)

    snapshot = policy.profile_allocation_snapshot()
    assert snapshot["balanced"] >= 0.25 - 1e-6
    assert snapshot["aggressive"] >= 0.25 - 1e-6
    assert snapshot["defensive"] >= 0.25 - 1e-6
    assert policy.floor_adjustment_applied

    capped_snapshot = capped_policy.profile_allocation_snapshot()
    assert capped_snapshot["balanced"] == pytest.approx(0.5, rel=1e-3)
    assert capped_snapshot["aggressive"] == pytest.approx(0.5, rel=1e-3)
    assert capped_policy.floor_adjustment_applied


def test_metric_weighted_allocation_falls_back_to_signal_policy() -> None:
    fallback = SignalStrengthAllocation()
    policy = MetricWeightedAllocation((), fallback_policy=fallback)

    schedules = [
        SimpleNamespace(
            name="trend",
            strategy_name="trend_engine",
            metrics={"signals": 5.0, "avg_confidence": 0.8},
        ),
        SimpleNamespace(
            name="grid",
            strategy_name="grid_engine",
            metrics={"signals": 1.0, "avg_confidence": 0.4},
        ),
    ]

    allocation = policy.allocate(schedules)
    fallback_allocation = fallback.allocate(schedules)

    assert allocation["trend"] == pytest.approx(fallback_allocation["trend"])
    assert allocation["grid"] == pytest.approx(fallback_allocation["grid"])

    diagnostics = policy.allocation_diagnostics()
    assert diagnostics["trend"]["fallback"] == pytest.approx(1.0)
    assert diagnostics["grid"]["fallback"] == pytest.approx(1.0)


def test_signal_strength_allocation_prefers_signal_rich_schedules() -> None:
    policy = SignalStrengthAllocation()
    schedules = [
        SimpleNamespace(
            name="high",
            metrics={"signals": 8.0, "avg_confidence": 0.85},
        ),
        SimpleNamespace(
            name="low",
            metrics={"signals": 1.0, "avg_confidence": 0.4},
        ),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["high"] > allocation["low"]


def test_resolve_capital_policy_drawdown_adaptive() -> None:
    spec = {
        "name": "drawdown_guard",
        "warning_pct": 6.0,
        "panic_pct": 12.0,
        "pressure_weight": 0.5,
        "min_weight": 0.1,
    }

    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, DrawdownAdaptiveAllocation)
    assert policy.name == "drawdown_guard"
    assert interval is None

    schedules = [
        SimpleNamespace(
            name="trend",
            metrics={"max_drawdown_pct": 4.0, "drawdown_pressure": 0.2},
        ),
        SimpleNamespace(
            name="grid",
            metrics={"max_drawdown_pct": 14.0, "drawdown_pressure": 0.6},
        ),
    ]

    allocation = policy.allocate(schedules)
    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend"] > allocation["grid"]

    diagnostics = policy.allocation_diagnostics()
    assert diagnostics["trend"]["drawdown_pct"] == pytest.approx(4.0)
    assert diagnostics["grid"]["penalty"] > diagnostics["trend"]["penalty"]


def test_drawdown_adaptive_clamps_ratios_when_metrics_missing() -> None:
    policy = DrawdownAdaptiveAllocation(warning_drawdown_pct=5.0, panic_drawdown_pct=10.0)
    schedules = [
        SimpleNamespace(name="trend", metrics={}),
        SimpleNamespace(name="mean", metrics={"drawdown": 0.08, "drawdown_trend": 1.4}),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend"] >= 0.0
    assert allocation["mean"] <= allocation["trend"]

    diagnostics = policy.allocation_diagnostics()
    assert diagnostics["trend"]["drawdown_pct"] == pytest.approx(0.0)
    assert diagnostics["mean"]["drawdown_pct"] == pytest.approx(8.0)


def test_smoothed_allocation_policy_applies_smoothing_and_threshold() -> None:
    class _DeterministicPolicy:
        def __init__(self, frames: list[Mapping[str, float]]) -> None:
            self._frames = list(frames)
            self.name = "deterministic"

        def allocate(self, schedules: Sequence[SimpleNamespace]) -> Mapping[str, float]:
            frame = self._frames.pop(0)
            return dict(frame)

    inner = _DeterministicPolicy(
        [
            {"trend": 0.8, "mean": 0.2},
            {"trend": 0.2, "mean": 0.8},
            {"trend": 0.48, "mean": 0.52},
        ]
    )
    policy = SmoothedCapitalAllocationPolicy(
        inner,
        smoothing_factor=0.5,
        min_delta=0.1,
    )

    schedules = [
        SimpleNamespace(name="trend", metrics={}),
        SimpleNamespace(name="mean", metrics={}),
    ]

    first = policy.allocate(schedules)
    assert first["trend"] == pytest.approx(0.8)
    assert first["mean"] == pytest.approx(0.2)

    second = policy.allocate(schedules)
    assert pytest.approx(sum(second.values())) == 1.0
    assert second["trend"] == pytest.approx(0.5)
    assert second["mean"] == pytest.approx(0.5)

    third = policy.allocate(schedules)
    assert third["trend"] == pytest.approx(0.5)
    assert third["mean"] == pytest.approx(0.5)

    raw_snapshot = policy.raw_allocation_snapshot()
    assert raw_snapshot["trend"] == pytest.approx(0.48)
    assert raw_snapshot["mean"] == pytest.approx(0.52)

    smoothed_snapshot = policy.smoothed_allocation_snapshot()
    assert smoothed_snapshot["trend"] == pytest.approx(0.5)
    assert smoothed_snapshot["mean"] == pytest.approx(0.5)


def test_resolve_capital_policy_smoothed_wrapper() -> None:
    spec = {
        "name": "smoothed",
        "label": "smoothed_signal",
        "alpha": 0.4,
        "min_delta": 0.05,
        "base": {"name": "signal_strength"},
    }

    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, SmoothedCapitalAllocationPolicy)
    assert policy.name == "smoothed_signal"
    assert interval is None
    assert isinstance(policy.inner_policy, SignalStrengthAllocation)
    assert policy.smoothing_factor == pytest.approx(0.4)
    assert policy.min_delta == pytest.approx(0.05)


def test_resolve_capital_policy_blended_combines_components() -> None:
    spec = {
        "name": "blended",
        "label": "combo",
        "components": [
            {"policy": "signal_strength", "weight": 2.0, "label": "signals"},
            {
                "policy": {
                    "name": "drawdown_guard",
                    "warning_pct": 5.0,
                    "panic_pct": 15.0,
                    "pressure_weight": 0.6,
                    "min_weight": 0.05,
                },
                "weight": 1.0,
                "label": "drawdown",
            },
        ],
        "normalize_components": True,
        "fallback": {"name": "equal_weight"},
    }

    policy, interval = _resolve_capital_policy(spec)

    assert isinstance(policy, BlendedCapitalAllocation)
    assert policy.name == "combo"
    assert interval is None

    schedules = [
        SimpleNamespace(
            name="trend_fast",
            strategy_name="trend_engine",
            risk_profile="balanced",
            metrics={
                "signals": 6.0,
                "avg_confidence": 0.85,
                "max_drawdown_pct": 4.0,
                "drawdown_pressure": 0.1,
            },
        ),
        SimpleNamespace(
            name="trend_slow",
            strategy_name="trend_engine",
            risk_profile="balanced",
            metrics={
                "signals": 1.0,
                "avg_confidence": 0.35,
                "max_drawdown_pct": 18.0,
                "drawdown_pressure": 0.8,
            },
        ),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend_fast"] > allocation["trend_slow"]

    diagnostics = policy.allocation_diagnostics()
    assert diagnostics["signals"]["mix_weight"] == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert diagnostics["signals"]["trend_fast"] > diagnostics["signals"]["trend_slow"]
    assert diagnostics["drawdown"]["trend_fast"] > diagnostics["drawdown"]["trend_slow"]


def test_blended_allocation_uses_fallback_when_components_missing() -> None:
    spec = {
        "name": "blended",
        "components": [],
        "fallback": "equal_weight",
    }

    policy, _ = _resolve_capital_policy(spec)

    schedules = [
        SimpleNamespace(name="trend", strategy_name="trend", risk_profile="core", metrics={}),
        SimpleNamespace(name="grid", strategy_name="grid", risk_profile="core", metrics={}),
    ]

    allocation = policy.allocate(schedules)

    assert pytest.approx(sum(allocation.values())) == 1.0
    assert allocation["trend"] == pytest.approx(0.5)
    assert allocation["grid"] == pytest.approx(0.5)
    assert policy.allocation_diagnostics() == {}
