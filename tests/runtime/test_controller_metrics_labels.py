from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.execution import ExecutionService
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.observability import MetricsRegistry
from bot_core.risk import RiskCheckResult, RiskEngine, RiskProfile
from bot_core.runtime import TradingController
from bot_core.strategies import StrategySignal


class _RiskEngineStub(RiskEngine):
    def register_profile(self, profile: RiskProfile) -> None:  # pragma: no cover - test helper
        return None

    def apply_pre_trade_checks(
        self,
        request: OrderRequest,
        *,
        account: AccountSnapshot,
        profile_name: str,
    ) -> RiskCheckResult:
        return RiskCheckResult(allowed=True)

    def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
        return {"profile": profile_name}

    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp: datetime | None = None,
    ) -> None:  # pragma: no cover - test helper
        return None

    def should_liquidate(self, *, profile_name: str) -> bool:
        return False


class _ExecutionStub(ExecutionService):
    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        return OrderResult(
            order_id=f"order-{request.symbol}",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"context": context.metadata},
        )

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:
        return None


def _account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        balances={},
        total_equity=100_000.0,
        available_margin=90_000.0,
        maintenance_margin=10_000.0,
    )


def _router() -> DefaultAlertRouter:
    return DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())


def test_controller_order_metrics_labels_do_not_leak_between_symbols() -> None:
    registry = MetricsRegistry()
    controller = TradingController(
        risk_engine=_RiskEngineStub(),
        execution_service=_ExecutionStub(),
        alert_router=_router(),
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(seconds=0),
        metrics_registry=registry,
    )

    btc_signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.75,
        metadata={"quantity": "1", "price": "100", "order_type": "market"},
    )
    eth_signal = StrategySignal(
        symbol="ETH/USDT",
        side="SELL",
        confidence=0.75,
        metadata={"quantity": "1", "price": "200", "order_type": "market"},
    )

    controller.process_signals([btc_signal, eth_signal])

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    base = {"environment": "paper", "portfolio": "paper-1", "risk_profile": "balanced"}
    btc = {**base, "symbol": "BTC/USDT"}
    eth = {**base, "symbol": "ETH/USDT"}

    assert orders_counter.value(labels={**btc, "result": "submitted", "side": "BUY"}) == 1.0
    assert orders_counter.value(labels={**btc, "result": "executed", "side": "BUY"}) == 1.0
    assert orders_counter.value(labels={**eth, "result": "submitted", "side": "SELL"}) == 1.0
    assert orders_counter.value(labels={**eth, "result": "executed", "side": "SELL"}) == 1.0
