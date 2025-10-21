"""Testy pomocniczych funkcji polityk kapitału w pipeline."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping, Sequence

import pytest

from bot_core.runtime.multi_strategy_scheduler import (
    DrawdownAdaptiveAllocation,
    EqualWeightAllocation,
    FixedWeightAllocation,
    RiskParityAllocation,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
    SmoothedCapitalAllocationPolicy,
    VolatilityTargetAllocation,
)
from bot_core.runtime.pipeline import (
    _collect_fixed_weight_entries,
    _collect_profile_weight_entries,
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
