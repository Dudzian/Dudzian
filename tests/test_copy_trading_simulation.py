from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.execution.base import ExecutionContext
from bot_core.exchanges.base import OrderRequest
from bot_core.portfolio.governor import PortfolioAdjustment, PortfolioAdvisory, PortfolioDecision
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


def build_catalog() -> StrategyCatalog:
    provider = HwIdProvider(fingerprint_reader=lambda: "test-hwid")
    catalog = StrategyCatalog(hwid_provider=provider)
    status = PresetLicenseStatus(
        preset_id="grid-pro",
        module_id="grid-pro",
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
        preset_id="ml-ai",
        module_id="ml-ai",
        status=PresetLicenseState.ACTIVE,
        fingerprint=None,
        fingerprint_candidates=(),
        fingerprint_verified=True,
        activated_at=datetime.now(timezone.utc),
        expires_at=None,
        edition="ai",
        capability="ai",
        signature_verified=True,
        issues=(),
        metadata={},
    )
    descriptor = StrategyPresetDescriptor(
        preset_id="grid-pro",
        name="Grid Pro",
        profile=StrategyPresetProfile.GRID,
        strategies=({"name": "grid", "engine": "GridStrategy", "parameters": {}},),
        required_parameters={"grid": ()},
        license_status=status,
        signature_verified=True,
        metadata={},
    )
    fallback = StrategyPresetDescriptor(
        preset_id="ml-ai",
        name="ML AI",
        profile=StrategyPresetProfile.AI,
        strategies=({"name": "ai", "engine": "AIStrategy", "parameters": {}},),
        required_parameters={"ai": ()},
        license_status=fallback_status,
        signature_verified=True,
        metadata={},
    )
    catalog._presets[descriptor.preset_id] = descriptor  # type: ignore[attr-defined]
    catalog._presets[fallback.preset_id] = fallback  # type: ignore[attr-defined]
    catalog.install_license_override("grid-pro", {"fingerprint": "test-hwid"})
    catalog.install_license_override("ml-ai", {"fingerprint": "test-hwid"})
    return catalog


class PaperHarness:
    def __init__(self, price: float) -> None:
        metadata = {"environment": "paper"}
        self.master_value = 200_000.0
        self.followers_value: dict[str, float] = {}
        self.price = price
        market = MarketMetadata(base_asset="BTC", quote_asset="USDT")
        self.master_service = PaperTradingExecutionService({"BTCUSDT": market}, initial_balances={"USDT": 1_000_000.0})
        self.services: dict[str, PaperTradingExecutionService] = {}
        self.metadata = metadata

    def execute_master(self, instruction, *, symbol: str = "BTCUSDT") -> None:
        self._execute(self.master_service, instruction, self.master_value, symbol)

    def execute_follower(self, follower_id: str, instruction, scale: float, *, symbol: str = "BTCUSDT") -> None:
        if follower_id not in self.services:
            self.services[follower_id] = PaperTradingExecutionService(
                {symbol: MarketMetadata(base_asset="BTC", quote_asset="USDT")},
                initial_balances={"USDT": 1_000_000.0},
            )
            self.followers_value[follower_id] = self.master_value * scale
        value = self.followers_value[follower_id]
        service = self.services[follower_id]
        self._execute(service, instruction, value, symbol)

    def _execute(self, service: PaperTradingExecutionService, instruction, portfolio_value: float, symbol: str) -> None:
        for adjustment in instruction.adjustments:
            delta = adjustment.proposed_weight - adjustment.current_weight
            if abs(delta) < 1e-9:
                continue
            notional = delta * portfolio_value
            quantity = abs(notional) / self.price
            side = "buy" if notional > 0 else "sell"
            order = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                price=self.price,
            )
            context = ExecutionContext(
                portfolio_id="paper",
                risk_profile="balanced",
                environment="paper",
                metadata={"source": "test"},
                price_resolver=lambda _symbol: self.price,
            )
            service.execute(order, context)


def make_decision() -> PortfolioDecision:
    adjustment = PortfolioAdjustment(
        symbol="BTCUSDT",
        current_weight=0.35,
        proposed_weight=0.6,
        reason="momentum",
        severity="info",
        metadata={},
    )
    advisory = PortfolioAdvisory(
        code="risk-ok",
        severity="info",
        message="Risk within target",
    )
    return PortfolioDecision(
        timestamp=datetime(2024, 5, 17, 11, tzinfo=timezone.utc),
        portfolio_id="master-grid",
        portfolio_value=200_000.0,
        adjustments=(adjustment,),
        advisories=(advisory,),
        rebalance_required=True,
    )


def test_copy_trading_paper_flow() -> None:
    catalog = build_catalog()
    audit_events: list[dict[str, object]] = []

    def audit(payload: dict[str, object]) -> None:
        audit_events.append(payload)

    scheduler = MultiPortfolioScheduler(catalog, audit_logger=audit)
    scheduler.register_portfolio(
        PortfolioBinding(
            portfolio_id="master-grid",
            primary_preset="grid-pro",
            fallback_presets=("ml-ai",),
            followers=(
                CopyTradingFollowerConfig(portfolio_id="follower-a", scaling=0.5),
                CopyTradingFollowerConfig(portfolio_id="follower-b", scaling=0.75),
            ),
        )
    )

    decision = make_decision()
    result = scheduler.process_decision(decision)

    harness = PaperHarness(price=20_000.0)
    for rebalance in result.rebalance:
        harness.execute_master(rebalance)
    for copy in result.copy_trades:
        harness.execute_follower(copy.follower_id, copy, copy.scale)

    # powinniśmy mieć trzy wpisy audytowe: rejestracja portfela + eventy self-healing/brak
    assert audit_events, "Audyt powinien odnotować rejestrację"
    assert len(result.copy_trades) == 2
    assert all(instruction.adjustments for instruction in result.copy_trades)

    degraded = make_decision()
    fallback_result = scheduler.process_decision(
        degraded,
        metadata={"strategy_health": "failed", "failure_reason": "volatility-spike"},
    )
    assert fallback_result.active_preset == "ml-ai"
    assert any(event["message"] == "strategy-self-healing" for event in audit_events if "message" in event)
