from __future__ import annotations

import threading
from datetime import timedelta
from typing import Mapping

import pytest

from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.execution import ExecutionService
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.risk import RiskCheckResult, RiskEngine, RiskProfile
from bot_core.runtime import TradingController
from bot_core.runtime.journal import TradingDecisionEvent
from bot_core.strategies import StrategySignal
from bot_core.strategies.catalog import (
    StrategyCatalog,
    StrategyDefinition,
    StrategyEngineSpec,
)
from bot_core.ui.api import (
    RuntimeStateSync,
    build_runtime_snapshot,
    describe_strategy_catalog,
)

from tests._alert_channel_helpers import CollectingChannel


class _StubJournal:
    def __init__(self) -> None:
        self._events: list[Mapping[str, str]] = []

    def record(self, event: TradingDecisionEvent) -> None:  # type: ignore[override]
        self._events.append(event.as_dict())

    def export(self) -> list[Mapping[str, str]]:  # type: ignore[override]
        return list(self._events)


class _StubRiskEngine(RiskEngine):
    def register_profile(self, profile: RiskProfile) -> None:  # pragma: no cover - nieużywane
        return None

    def apply_pre_trade_checks(
        self,
        request: OrderRequest,
        *,
        account: AccountSnapshot,
        profile_name: str,
    ) -> RiskCheckResult:
        return RiskCheckResult(allowed=True)

    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp=None,
    ) -> None:  # pragma: no cover - nieużywane
        return None

    def should_liquidate(self, *, profile_name: str) -> bool:
        return False

    def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
        return {
            "profile": profile_name,
            "portfolioValue": 125_000.0,
            "maxDailyLoss": 0.05,
        }


class _Breaker:
    def __init__(self, state: str) -> None:
        self.state = state
        self.failure_count = 0


class _StubExecutionService(ExecutionService):
    def __init__(self) -> None:
        self._adapters = {"BINANCE": object(), "KRAKEN": object()}
        self._breakers = {"BINANCE": _Breaker("closed"), "KRAKEN": _Breaker("half_open")}
        self.requests: list[OrderRequest] = []

    def list_adapters(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(self._adapters.keys())

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
        return OrderResult(
            order_id="test-order",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"context": context.metadata},
        )

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:  # type: ignore[override]
        return None


def _account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": 120_000.0, "BTC": 1.5},
        total_equity=150_000.0,
        available_margin=120_000.0,
        maintenance_margin=15_000.0,
    )


def _build_controller() -> tuple[TradingController, _StubJournal]:
    risk_engine = _StubRiskEngine()
    execution = _StubExecutionService()
    audit = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit)
    channel = CollectingChannel()
    router.register(channel)
    journal = _StubJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-01",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(minutes=5),
        decision_journal=journal,
        strategy_name="theta_income_balanced",
        exchange_name="BINANCE",
        ai_signal_modes=("ai",),
        rules_signal_modes=("rules",),
        signal_mode_priorities={"ai": 2, "rules": 1},
    )
    return controller, journal


def _prepare_catalog() -> tuple[StrategyCatalog, Mapping[str, StrategyDefinition]]:
    catalog = StrategyCatalog()

    def _factory(*, name: str, parameters, metadata=None):
        return object()

    catalog.register(
        StrategyEngineSpec(
            key="theta_engine",
            factory=_factory,
            license_tier="enterprise",
            risk_classes=("derivatives", "income"),
            required_data=("options_chain", "ohlcv"),
            default_tags=("options", "income"),
        )
    )

    definitions = {
        "theta_income_balanced": StrategyDefinition(
            name="theta_income_balanced",
            engine="theta_engine",
            license_tier="enterprise",
            risk_classes=("derivatives",),
            required_data=("options_chain",),
            parameters={"min_iv": 0.3},
            tags=("options", "income"),
            metadata={"description": "Dochód theta"},
        )
    }
    return catalog, definitions


def test_describe_strategy_catalog_returns_entries() -> None:
    catalog, definitions = _prepare_catalog()

    entries = describe_strategy_catalog(catalog, definitions)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "theta_income_balanced"
    assert entry.engine == "theta_engine"
    assert "options_chain" in entry.required_data


def test_build_runtime_snapshot_collects_state() -> None:
    controller, _ = _build_controller()
    catalog, definitions = _prepare_catalog()

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.9,
        metadata={"quantity": "1", "order_type": "market"},
    )
    controller.process_signals([signal])

    snapshot = build_runtime_snapshot(controller, catalog=catalog, strategies=definitions)

    assert snapshot.portfolio.portfolio_id == "paper-01"
    assert snapshot.runtime_state.name == "theta_income_balanced"
    assert snapshot.exchanges and snapshot.exchanges[0].name == "BINANCE"
    # Breaker half-open powinien zostać zmapowany na status degraded
    assert any(status.status == "degraded" for status in snapshot.exchanges)
    assert snapshot.alerts, "Oczekiwano zapisanych alertów w audycie"


@pytest.mark.timeout(5)
def test_runtime_state_sync_notifies_listeners() -> None:
    controller, _ = _build_controller()
    catalog, definitions = _prepare_catalog()

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="SELL",
        confidence=0.8,
        metadata={"quantity": "0.5", "order_type": "market"},
    )
    controller.process_signals([signal])

    sync = RuntimeStateSync(
        controller,
        poll_interval=0.1,
        catalog=catalog,
        strategies=definitions,
    )

    ready = threading.Event()
    snapshots: list = []

    def _listener(snapshot):
        snapshots.append(snapshot)
        ready.set()

    sync.add_listener(_listener)
    sync.start()
    try:
        assert ready.wait(2.0), "Listener nie otrzymał snapshotu w oczekiwanym czasie"
    finally:
        sync.stop()

    assert snapshots, "Brak zebranych snapshotów runtime"
