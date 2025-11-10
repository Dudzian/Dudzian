from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability.slo import SLOStatus
from bot_core.portfolio import (
    AssetPortfolioGovernorConfig,
    PortfolioAssetConfig,
    PortfolioDecisionLog,
    PortfolioGovernor,
)
from bot_core.risk import StressOverrideRecommendation
from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator


def _governor() -> PortfolioGovernor:
    config = AssetPortfolioGovernorConfig(
        name="core",
        portfolio_id="core",
        assets=(
            PortfolioAssetConfig(
                symbol="BTC_USDT",
                target_weight=0.6,
                min_weight=0.1,
                max_weight=0.7,
            ),
        ),
        market_intel_interval="1h",
        market_intel_lookback_bars=48,
    )
    return PortfolioGovernor(config, decision_log=PortfolioDecisionLog())


def test_portfolio_runtime_coordinator_evaluates_and_logs() -> None:
    governor = _governor()
    allocations_called = 0

    def _allocations() -> tuple[float, Mapping[str, float]]:
        nonlocal allocations_called
        allocations_called += 1
        return 100_000.0, {"BTC_USDT": 0.2}

    snapshot = MarketIntelSnapshot(
        symbol="BTC_USDT",
        interval="1h",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        bar_count=24,
        price_change_pct=4.0,
        volatility_pct=12.0,
        max_drawdown_pct=6.0,
        average_volume=15_000.0,
        liquidity_usd=500_000.0,
        momentum_score=1.2,
    )

    def _market_data() -> Mapping[str, MarketIntelSnapshot]:
        return {"BTC_USDT": snapshot}

    slo_status = SLOStatus(
        name="latency",
        indicator="router_latency_ms",
        status="ok",
        severity="info",
        value=180.0,
        target=250.0,
        comparison="<=",
        warning_threshold=220.0,
        sample_size=500,
    )

    def _slo_provider() -> Mapping[str, SLOStatus]:
        return {"latency": slo_status}

    override = StressOverrideRecommendation(
        severity="critical",
        reason="latency_spike",
        symbol="BTC_USDT",
        weight_multiplier=0.4,
        force_rebalance=True,
    )

    def _override_provider() -> Sequence[StressOverrideRecommendation]:
        return (override,)

    coordinator = PortfolioRuntimeCoordinator(
        governor,
        allocation_provider=_allocations,
        market_data_provider=_market_data,
        slo_status_provider=_slo_provider,
        stress_override_provider=_override_provider,
        metadata_provider=lambda: {"source": "test"},
        clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    decision = coordinator.evaluate(force=True)
    assert decision is not None
    assert decision.rebalance_required is True
    assert allocations_called == 1
    log_entries = governor._decision_log.tail(limit=1)  # type: ignore[attr-defined]
    assert log_entries and log_entries[-1]["metadata"]["source"] == "test"


def test_portfolio_runtime_coordinator_respects_cooldown() -> None:
    governor = _governor()
    current_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _clock() -> datetime:
        return current_time

    calls = {"alloc": 0}

    def _allocations() -> tuple[float, Mapping[str, float]]:
        calls["alloc"] += 1
        return 10_000.0, {"BTC_USDT": 0.4}

    coordinator = PortfolioRuntimeCoordinator(
        governor,
        allocation_provider=_allocations,
        market_data_provider=lambda: {},
        clock=_clock,
    )

    assert coordinator.evaluate(force=True) is not None
    assert calls["alloc"] == 1

    current_time = current_time + timedelta(seconds=10)
    assert coordinator.evaluate() is not None
    assert calls["alloc"] == 1

    current_time = current_time + timedelta(seconds=governor._config.rebalance_cooldown_seconds + 10)  # type: ignore[attr-defined]
    assert coordinator.evaluate() is not None
    assert calls["alloc"] == 2
