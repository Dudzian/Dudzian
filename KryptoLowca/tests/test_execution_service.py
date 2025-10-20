from __future__ import annotations

from datetime import datetime

import pytest

from bot_core.execution import ExecutionContext
from bot_core.exchanges.base import OrderResult

from KryptoLowca.core.services.execution_service import ExecutionService
from KryptoLowca.strategies.base import StrategyContext, StrategyMetadata, StrategySignal


class _DummyRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def execute(self, request: object, context: object) -> OrderResult:
        self.calls.append((request, context))
        return OrderResult(
            order_id="42",
            status="filled",
            filled_quantity=1.0,
            avg_price=101.5,
            raw_response={"source": "router"},
        )

    def cancel(self, *_: object) -> None:  # pragma: no cover - nie używane w teście
        return None

    def flush(self) -> None:  # pragma: no cover - nie używane w teście
        return None


class _DummyAccountManager:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def dispatch_order(self, request: object) -> OrderResult:
        self.requests.append(request)
        return OrderResult(
            order_id="99",
            status="queued",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"source": "manager"},
        )


def _strategy_context(symbol: str = "BTC/USDT") -> StrategyContext:
    metadata = StrategyMetadata(name="Dummy", description="Test strategy")
    return StrategyContext(
        symbol=symbol,
        timeframe="1h",
        portfolio_value=10_000.0,
        position=0.0,
        timestamp=datetime.utcnow(),
        metadata=metadata,
        extra={"mode": "paper"},
    )


def _strategy_signal(action: str = "BUY", *, size: float = 0.5) -> StrategySignal:
    return StrategySignal(
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        size=size,
        payload={"price": 101.5},
    )


@pytest.mark.asyncio
async def test_execution_service_dispatches_via_router() -> None:
    router = _DummyRouter()

    def builder(ctx: StrategyContext) -> ExecutionContext:
        return ExecutionContext(
            portfolio_id="demo",
            risk_profile="balanced",
            environment="paper",
            metadata={"symbol": ctx.symbol},
        )

    service = ExecutionService(None, router=router, context_builder=builder)

    ctx = _strategy_context()
    signal = _strategy_signal()

    result = await service.execute(signal, ctx)

    assert isinstance(result, OrderResult)
    assert router.calls, "Router powinien otrzymać zlecenie"
    recorded_request, recorded_context = router.calls[-1]
    assert recorded_request.symbol == "BTC/USDT"
    assert recorded_request.quantity == pytest.approx(0.5)
    assert recorded_context.portfolio_id == "demo"


@pytest.mark.asyncio
async def test_execution_service_uses_account_manager_when_bound() -> None:
    manager = _DummyAccountManager()
    service = ExecutionService(None, account_manager=manager)

    ctx = _strategy_context()
    signal = _strategy_signal(size=0.75)

    result = await service.execute(signal, ctx)

    assert isinstance(result, OrderResult)
    assert manager.requests, "MultiExchangeAccountManager powinien otrzymać zlecenie"
    recorded_request = manager.requests[-1]
    assert recorded_request.quantity == pytest.approx(0.75)

