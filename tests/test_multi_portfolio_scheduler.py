from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot_core.portfolio.governor import (
    PortfolioAdjustment,
    PortfolioAdvisory,
    PortfolioDecision,
)
from bot_core.portfolio.scheduler import (
    CopyTradingFollowerConfig,
    MultiPortfolioScheduler,
    PortfolioBinding,
)
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.catalog import (
    PresetLicenseState,
    PresetLicenseStatus,
    StrategyCatalog,
    StrategyPresetDescriptor,
    StrategyPresetProfile,
)


@pytest.fixture()
def strategy_catalog() -> StrategyCatalog:
    provider = HwIdProvider(fingerprint_reader=lambda: "test-hwid")
    catalog = StrategyCatalog(hwid_provider=provider)
    active_status = PresetLicenseStatus(
        preset_id="alpha-grid",
        module_id="module-alpha",
        status=PresetLicenseState.ACTIVE,
        fingerprint=None,
        fingerprint_candidates=(),
        fingerprint_verified=True,
        activated_at=datetime.now(timezone.utc),
        expires_at=None,
        edition="pro",
        capability="grid",
        signature_verified=True,
        issues=(),
        metadata={},
    )
    fallback_status = PresetLicenseStatus(
        preset_id="beta-ai",
        module_id="module-beta",
        status=PresetLicenseState.ACTIVE,
        fingerprint=None,
        fingerprint_candidates=(),
        fingerprint_verified=True,
        activated_at=datetime.now(timezone.utc),
        expires_at=None,
        edition="elite",
        capability="ai",
        signature_verified=True,
        issues=(),
        metadata={},
    )
    active_descriptor = StrategyPresetDescriptor(
        preset_id="alpha-grid",
        name="Alpha Grid",
        profile=StrategyPresetProfile.GRID,
        strategies=(
            {
                "name": "grid",
                "engine": "GridTradingStrategy",
                "parameters": {},
            },
        ),
        required_parameters={"grid": ()},
        license_status=active_status,
        signature_verified=True,
        metadata={},
    )
    fallback_descriptor = StrategyPresetDescriptor(
        preset_id="beta-ai",
        name="Beta AI",
        profile=StrategyPresetProfile.AI,
        strategies=(
            {
                "name": "ai",
                "engine": "MachineLearningStrategy",
                "parameters": {},
            },
        ),
        required_parameters={"ai": ()},
        license_status=fallback_status,
        signature_verified=True,
        metadata={},
    )
    # Wstrzykujemy przygotowane presety do katalogu
    catalog._presets[active_descriptor.preset_id] = active_descriptor  # type: ignore[attr-defined]
    catalog._presets[fallback_descriptor.preset_id] = fallback_descriptor  # type: ignore[attr-defined]
    catalog.install_license_override("alpha-grid", {"fingerprint": "test-hwid"})
    catalog.install_license_override("beta-ai", {"fingerprint": "test-hwid"})
    return catalog


def make_decision(rebalance: bool = True) -> PortfolioDecision:
    timestamp = datetime(2024, 5, 17, 12, tzinfo=timezone.utc)
    adjustment = PortfolioAdjustment(
        symbol="BTCUSDT",
        current_weight=0.4,
        proposed_weight=0.55,
        reason="trend-follow",
        severity="info",
        metadata={"volatility": 0.12},
    )
    advisory = PortfolioAdvisory(
        code="drawdown",
        severity="info",
        message="Drawdown within limits",
        symbols=("BTCUSDT",),
        metrics={"drawdown": 0.03},
    )
    return PortfolioDecision(
        timestamp=timestamp,
        portfolio_id="master-001",
        portfolio_value=250_000.0,
        adjustments=(adjustment,),
        advisories=(advisory,),
        rebalance_required=rebalance,
    )


def test_copy_trading_scaling(strategy_catalog: StrategyCatalog) -> None:
    scheduler = MultiPortfolioScheduler(strategy_catalog, clock=lambda: datetime(2024, 5, 17, 12, tzinfo=timezone.utc))
    scheduler.register_portfolio(
        PortfolioBinding(
            portfolio_id="master-001",
            primary_preset="alpha-grid",
            fallback_presets=("beta-ai",),
            followers=(CopyTradingFollowerConfig(portfolio_id="follower-01", scaling=0.5),),
        )
    )

    decision = make_decision(rebalance=True)
    result = scheduler.process_decision(decision)

    assert result.rebalance, "Master powinien otrzymać instrukcję rebalancingu"
    rebalance = result.rebalance[0]
    assert rebalance.portfolio_id == "master-001"
    assert pytest.approx(rebalance.adjustments[0].proposed_weight, abs=1e-6) == 0.55

    assert result.copy_trades, "Follower powinien otrzymać instrukcje kopiowania"
    follower = result.copy_trades[0]
    assert follower.follower_id == "follower-01"
    assert pytest.approx(follower.adjustments[0].proposed_weight, abs=1e-6) == 0.475
    assert follower.adjustments[0].metadata.get("copy_scale") == pytest.approx(0.5)
    assert follower.adjustments[0].metadata.get("copy_source_weight") == pytest.approx(0.55)
    assert result.active_preset == "alpha-grid"


def test_self_healing_switches_to_fallback(strategy_catalog: StrategyCatalog) -> None:
    clock = [datetime(2024, 5, 17, 12, tzinfo=timezone.utc)]

    def tick() -> datetime:
        current = clock[0]
        clock[0] = current + timedelta(minutes=5)
        return current

    scheduler = MultiPortfolioScheduler(strategy_catalog, clock=tick)
    scheduler.register_portfolio(
        PortfolioBinding(
            portfolio_id="master-001",
            primary_preset="alpha-grid",
            fallback_presets=("beta-ai",),
            followers=(),
        )
    )

    healthy = make_decision(rebalance=False)
    scheduler.process_decision(healthy)
    degraded = make_decision(rebalance=False)

    result = scheduler.process_decision(
        degraded,
        metadata={"strategy_health": "failed", "failure_reason": "drawdown-limit"},
    )

    assert result.active_preset == "beta-ai"
    messages = [event.message for event in result.events]
    assert "strategy-health-degraded" in messages
    assert "strategy-self-healing" in messages
