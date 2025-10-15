from datetime import datetime, timezone

import pytest

from bot_core.config.models import (
    PortfolioGovernorConfig,
    PortfolioGovernorScoringWeights,
    PortfolioGovernorStrategyConfig,
)
from bot_core.portfolio import PortfolioGovernor



def _build_governor(*, enabled: bool = True, **overrides: object) -> PortfolioGovernor:
    params = {
        "enabled": enabled,
        "rebalance_interval_minutes": 0.0,
        "smoothing": 1.0,
        "min_score_threshold": 0.0,
        "default_cost_bps": 0.5,
        "scoring": PortfolioGovernorScoringWeights(alpha=1.0, cost=0.5, slo=0.25, risk=0.0),
        "strategies": {
            "trend": PortfolioGovernorStrategyConfig(
                baseline_weight=0.5,
                min_weight=0.2,
                max_weight=0.8,
                baseline_max_signals=4,
                max_signal_factor=2.0,
            ),
            "mean_reversion": PortfolioGovernorStrategyConfig(
                baseline_weight=0.5,
                min_weight=0.1,
                max_weight=0.6,
                baseline_max_signals=3,
                max_signal_factor=1.5,
            ),
        },
    }
    params.update(overrides)
    config = PortfolioGovernorConfig(**params)
    return PortfolioGovernor(config, clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))


def test_portfolio_governor_rebalances_based_on_scores() -> None:
    governor = _build_governor(default_cost_bps=0.0)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    governor.observe_strategy_metrics(
        "trend",
        {"alpha_score": 2.0, "slo_violation_rate": 0.05, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    governor.observe_strategy_metrics(
        "mean_reversion",
        {"alpha_score": 0.4, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )

    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)
    assert decision is not None
    assert decision.weights["trend"] > decision.weights["mean_reversion"]
    allocation = governor.resolve_allocation("trend")
    assert allocation.max_signal_hint == 4
    assert allocation.signal_factor == pytest.approx(1.56, rel=1e-2)


def test_portfolio_governor_requires_complete_metrics_when_configured() -> None:
    governor = _build_governor(require_complete_metrics=True)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    governor.observe_strategy_metrics(
        "trend",
        {"alpha_score": 1.5, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    assert governor.maybe_rebalance(timestamp=timestamp, force=True) is None

    governor.observe_strategy_metrics(
        "mean_reversion",
        {"alpha_score": 1.0, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)
    assert decision is not None
    assert all(weight >= 0.0 for weight in decision.weights.values())


def test_portfolio_governor_uses_cost_report_updates() -> None:
    governor = _build_governor(
        scoring=PortfolioGovernorScoringWeights(alpha=1.0, cost=1.0, slo=0.0, risk=0.0),
        default_cost_bps=10.0,
    )
    report = {
        "strategies": {
            "trend": {"total": {"cost_bps": 1.0}},
            "mean_reversion": {"total": {"cost_bps": 8.0}},
        },
        "total": {"cost_bps": 12.0},
    }
    governor.update_costs_from_report(report)

    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    payload = {"alpha_score": 2.0, "slo_violation_rate": 0.0, "risk_penalty": 0.0}
    governor.observe_strategy_metrics("trend", payload, timestamp=timestamp)
    governor.observe_strategy_metrics("mean_reversion", payload, timestamp=timestamp)

    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)
    assert decision is not None
    assert decision.cost_components["trend"] == pytest.approx(1.0)
    assert decision.cost_components["mean_reversion"] == pytest.approx(8.0)
    assert decision.weights["trend"] > decision.weights["mean_reversion"]
