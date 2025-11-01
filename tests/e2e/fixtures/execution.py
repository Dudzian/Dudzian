"""Atrapy usług egzekucji wykorzystywane w testach E2E."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bot_core.execution import ExecutionContext, ExecutionService
from bot_core.exchanges.base import OrderRequest, OrderResult


@dataclass(slots=True)
class RecordedOrder:
    """Pojedynczy zapis obsłużonego zlecenia z pełnym kontekstem."""

    request: OrderRequest
    context: ExecutionContext


class FakeExecutionService(ExecutionService):
    """Minimalna implementacja ``ExecutionService`` używana w testach."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        failure: Exception | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.failure = failure or RuntimeError("execution failure")
        self.executed: List[RecordedOrder] = []
        self.cancelled: List[tuple[str, ExecutionContext]] = []
        self.flush_called = False

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        self.executed.append(RecordedOrder(request=request, context=context))
        if self.should_fail:
            raise self.failure
        return OrderResult(
            order_id="fake-order",
            status="submitted",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"source": "fake"},
        )

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # pragma: no cover - nieużywane w testach
        self.cancelled.append((order_id, context))

    def flush(self) -> None:  # pragma: no cover - nieużywane w testach
        self.flush_called = True
