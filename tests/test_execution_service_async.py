import asyncio

import pytest

from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import OrderRequest, OrderResult


class _DummyExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.execute_calls: list[tuple[OrderRequest, ExecutionContext]] = []
        self.cancel_calls: list[tuple[str, ExecutionContext]] = []
        self.flush_calls: int = 0
        self.close_calls: int = 0

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:  # type: ignore[override]
        self.execute_calls.append((request, context))
        return OrderResult(
            order_id="dummy-1",
            status="FILLED",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"source": "dummy"},
        )

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # type: ignore[override]
        self.cancel_calls.append((order_id, context))

    def flush(self) -> None:  # type: ignore[override]
        self.flush_calls += 1

    def close(self) -> None:  # type: ignore[override]
        self.close_calls += 1


class _FailingExecutionService(ExecutionService):
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:  # type: ignore[override]
        raise ValueError("execute-boom")

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # type: ignore[override]
        raise ValueError("cancel-boom")

    def flush(self) -> None:  # type: ignore[override]
        raise ValueError("flush-boom")

    def close(self) -> None:  # type: ignore[override]
        raise ValueError("close-boom")


@pytest.fixture()
def sample_context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="p-1",
        risk_profile="balanced",
        environment="live",
        metadata={},
    )


@pytest.fixture()
def sample_request() -> OrderRequest:
    return OrderRequest(symbol="BTC/USDT", side="BUY", quantity=1.0, order_type="MARKET", price=25_000.0)


def test_execute_async_calls_sync_implementation(sample_request: OrderRequest, sample_context: ExecutionContext) -> None:
    service = _DummyExecutionService()

    result = asyncio.run(service.execute_async(sample_request, sample_context))

    assert result.order_id == "dummy-1"
    assert service.execute_calls == [(sample_request, sample_context)]


def test_execute_async_propagates_errors(sample_request: OrderRequest, sample_context: ExecutionContext) -> None:
    service = _FailingExecutionService()

    with pytest.raises(ValueError):
        asyncio.run(service.execute_async(sample_request, sample_context))


def test_cancel_async_calls_sync_implementation(sample_context: ExecutionContext) -> None:
    service = _DummyExecutionService()

    asyncio.run(service.cancel_async("ORD-1", sample_context))

    assert service.cancel_calls == [("ORD-1", sample_context)]


def test_cancel_async_propagates_errors(sample_context: ExecutionContext) -> None:
    service = _FailingExecutionService()

    with pytest.raises(ValueError):
        asyncio.run(service.cancel_async("ORD-2", sample_context))


def test_flush_async_calls_sync_implementation() -> None:
    service = _DummyExecutionService()

    asyncio.run(service.flush_async())

    assert service.flush_calls == 1


def test_flush_async_propagates_errors() -> None:
    service = _FailingExecutionService()

    with pytest.raises(ValueError):
        asyncio.run(service.flush_async())


def test_close_async_calls_sync_implementation() -> None:
    service = _DummyExecutionService()

    asyncio.run(service.close_async())

    assert service.close_calls == 1


def test_close_async_propagates_errors() -> None:
    service = _FailingExecutionService()

    with pytest.raises(ValueError):
        asyncio.run(service.close_async())


def test_execution_service_context_manager_calls_close(sample_request: OrderRequest, sample_context: ExecutionContext) -> None:
    service = _DummyExecutionService()

    with service as ctx:
        assert ctx is service
        service.execute(sample_request, sample_context)

    assert service.close_calls == 1


def test_execution_service_context_manager_propagates_close_errors() -> None:
    service = _FailingExecutionService()

    with pytest.raises(ValueError):
        with service:
            pass


def test_execution_service_async_context_manager_calls_close(sample_request: OrderRequest, sample_context: ExecutionContext) -> None:
    service = _DummyExecutionService()

    async def _scenario() -> None:
        async with service as ctx:
            assert ctx is service
            await ctx.execute_async(sample_request, sample_context)

    asyncio.run(_scenario())

    assert service.close_calls == 1


def test_execution_service_async_context_manager_propagates_close_errors() -> None:
    service = _FailingExecutionService()

    async def _scenario() -> None:
        async with service:
            pass

    with pytest.raises(ValueError):
        asyncio.run(_scenario())
