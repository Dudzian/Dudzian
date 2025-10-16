from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability import SLOStatus
from bot_core.portfolio import (
    PortfolioAssetConfig,
    PortfolioDriftTolerance,
    PortfolioDecisionLog,
    PortfolioGovernor,
    PortfolioGovernorConfig,
    PortfolioRiskBudgetConfig,
    PortfolioSloOverrideConfig,
)
from bot_core.risk import StressOverrideRecommendation


def _snapshot(
    *,
    volatility: float | None = None,
    liquidity: float | None = None,
    drawdown: float | None = None,
) -> MarketIntelSnapshot:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return MarketIntelSnapshot(
        symbol="BTC_USDT",
        interval="1h",
        start=now,
        end=now,
        bar_count=24,
        price_change_pct=5.0,
        volatility_pct=volatility,
        max_drawdown_pct=drawdown,
        average_volume=1_000.0,
        liquidity_usd=liquidity,
        momentum_score=2.0,
        metadata={},
    )


def _governor_config() -> PortfolioGovernorConfig:
    return PortfolioGovernorConfig(
        name="core",
        portfolio_id="core",
        drift_tolerance=PortfolioDriftTolerance(absolute=0.02, relative=0.1),
        min_rebalance_value=100.0,
        min_rebalance_weight=0.01,
        assets=(
            PortfolioAssetConfig(
                symbol="BTC_USDT",
                target_weight=0.5,
                min_weight=0.1,
                max_weight=0.6,
                max_volatility_pct=20.0,
                min_liquidity_usd=500.0,
                risk_budget="balanced",
            ),
        ),
        risk_budgets={
            "balanced": PortfolioRiskBudgetConfig(
                name="balanced",
                max_var_pct=25.0,
                max_drawdown_pct=35.0,
                max_leverage=1.0,
                severity="warning",
            )
        },
    )


def test_portfolio_governor_detects_drift() -> None:
    config = _governor_config()
    governor = PortfolioGovernor(config, clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))

    decision = governor.evaluate(
        portfolio_value=100_000.0,
        allocations={"BTC_USDT": 0.3},
        market_data={"BTC_USDT": _snapshot(volatility=10.0, liquidity=10_000.0, drawdown=5.0)},
    )

    assert decision.rebalance_required is True
    assert len(decision.adjustments) == 1
    adjustment = decision.adjustments[0]
    assert adjustment.symbol == "BTC_USDT"
    assert adjustment.proposed_weight == pytest.approx(0.5)
    assert governor.last_rebalance_at == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_portfolio_governor_enforces_risk_budget() -> None:
    config = _governor_config()
    governor = PortfolioGovernor(config)

    decision = governor.evaluate(
        portfolio_value=50_000.0,
        allocations={"BTC_USDT": 0.5},
        market_data={"BTC_USDT": _snapshot(volatility=32.0, liquidity=10_000.0, drawdown=40.0)},
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.1)
    assert "volatility" in adjustment.reason
    assert decision.advisories
    advisory = decision.advisories[0]
    assert advisory.code == "risk_budget.balanced"
    assert "volatility" in advisory.message
    assert "drawdown" in advisory.message


def test_portfolio_governor_applies_slo_overrides() -> None:
    config = PortfolioGovernorConfig(
        name="core",
        portfolio_id="core",
        drift_tolerance=PortfolioDriftTolerance(absolute=0.01, relative=0.05),
        min_rebalance_value=0.0,
        min_rebalance_weight=0.0,
        assets=(
            PortfolioAssetConfig(
                symbol="ETH_USDT",
                target_weight=0.4,
                min_weight=0.1,
                max_weight=0.6,
                tags=("core",),
            ),
        ),
        risk_budgets={},
        slo_overrides=(
            PortfolioSloOverrideConfig(
                slo_name="latency",
                apply_on=("warning", "breach"),
                weight_multiplier=0.5,
                severity="critical",
                force_rebalance=True,
            ),
        ),
    )

    governor = PortfolioGovernor(config)
    status = SLOStatus(
        name="latency",
        indicator="router_latency_ms",
        value=320.0,
        target=250.0,
        comparison="<=",
        status="breach",
        severity="critical",
        warning_threshold=200.0,
        error_budget_pct=0.28,
        window_start=None,
        window_end=None,
        sample_size=7200,
    )

    decision = governor.evaluate(
        portfolio_value=100_000.0,
        allocations={"ETH_USDT": 0.4},
        market_data={"ETH_USDT": _snapshot(volatility=15.0, liquidity=20_000.0, drawdown=5.0)},
        slo_statuses={"latency": status},
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.2)
    assert adjustment.severity == "critical"
    assert adjustment.metadata["slo::latency"] == pytest.approx(0.28)
    assert adjustment.metadata["slo::latency::force_rebalance"] == pytest.approx(1.0)


def test_portfolio_governor_applies_stress_override_for_symbol() -> None:
    config = _governor_config()
    config = PortfolioGovernorConfig(
        name=config.name,
        portfolio_id=config.portfolio_id,
        drift_tolerance=config.drift_tolerance,
        min_rebalance_value=0.0,
        min_rebalance_weight=1.0,
        assets=config.assets,
        risk_budgets=config.risk_budgets,
    )
    governor = PortfolioGovernor(config)
    overrides = [
        StressOverrideRecommendation(
            severity="critical",
            reason="latency_spike",
            symbol="BTC_USDT",
            weight_multiplier=0.3,
            min_weight=0.1,
            force_rebalance=True,
        )
    ]

    decision = governor.evaluate(
        portfolio_value=150_000.0,
        allocations={"BTC_USDT": 0.5},
        market_data={"BTC_USDT": _snapshot(volatility=12.0, liquidity=50_000.0, drawdown=5.0)},
        stress_overrides=overrides,
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.15)
    assert adjustment.severity == "critical"
    assert "stress::latency_spike" in adjustment.reason
    assert adjustment.metadata["stress::count"] == pytest.approx(1.0)
    assert adjustment.metadata["stress::1::weight_multiplier"] == pytest.approx(0.3)
    assert adjustment.metadata["stress::1::force_rebalance"] == pytest.approx(1.0)


def test_portfolio_governor_applies_stress_override_for_risk_budget() -> None:
    config = _governor_config()
    governor = PortfolioGovernor(config)
    overrides = [
        StressOverrideRecommendation(
            severity="warning",
            reason="drawdown_pressure",
            risk_budget="balanced",
            weight_multiplier=0.5,
        )
    ]

    decision = governor.evaluate(
        portfolio_value=90_000.0,
        allocations={"BTC_USDT": 0.45},
        market_data={"BTC_USDT": _snapshot(volatility=10.0, liquidity=60_000.0, drawdown=6.0)},
        stress_overrides=overrides,
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.25)
    assert adjustment.severity == "warning"
    assert "stress::drawdown_pressure" in adjustment.reason


def test_portfolio_governor_writes_decision_log(tmp_path: Path) -> None:
    config = _governor_config()
    log_path = (tmp_path / "audit").joinpath("portfolio_decisions.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    key = b"S" * 48
    log = PortfolioDecisionLog(jsonl_path=log_path, signing_key=key, signing_key_id="stage6")
    clock_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    governor = PortfolioGovernor(config, clock=lambda: clock_time, decision_log=log)

    overrides = [
        StressOverrideRecommendation(
            severity="critical",
            reason="stress_log_test",
            symbol="BTC_USDT",
            weight_multiplier=0.4,
            force_rebalance=True,
        )
    ]

    decision = governor.evaluate(
        portfolio_value=120_000.0,
        allocations={"BTC_USDT": 0.2},
        market_data={"BTC_USDT": _snapshot(volatility=12.0, liquidity=15_000.0, drawdown=10.0)},
        stress_overrides=overrides,
        log_context={"environment": "paper", "run_id": "unit-test"},
    )

    assert decision.rebalance_required is True
    assert log_path.exists()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["portfolio_id"] == config.portfolio_id
    assert entry["metadata"]["stress_overrides"][0]["reason"] == "stress_log_test"
    assert entry["metadata"]["environment"] == "paper"
    assert entry["metadata"]["adjustment_count"] == 1
    assert "signature" in entry and entry["signature"]["key_id"] == "stage6"
