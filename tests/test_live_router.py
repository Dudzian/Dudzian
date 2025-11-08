"""Testy routera egzekucji live z fallbackami."""

from __future__ import annotations

from typing import Mapping

import pytest

from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.execution.base import ExecutionContext
from bot_core.exchanges.base import (
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.observability.metrics import MetricsRegistry


class _DummyAdapter(ExchangeAdapter):
    def __init__(self, name: str, *, should_fail: bool = False) -> None:
        super().__init__(ExchangeCredentials(key_id="dummy"))
        self.name = name
        self.should_fail = should_fail
        self.executed: list[OrderRequest] = []
        self.cancelled: list[str] = []

    # --- ExchangeAdapter API -------------------------------------------------
    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - nieużywane
        return None

    def fetch_account_snapshot(self):  # pragma: no cover - nieużywane
        raise NotImplementedError

    def fetch_symbols(self):  # pragma: no cover - nieużywane
        return ()

    def fetch_ohlcv(self, *_, **__):  # pragma: no cover - nieużywane
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        if self.should_fail:
            raise ExchangeNetworkError(message=f"Adapter {self.name} offline")
        self.executed.append(request)
        return OrderResult(
            order_id=f"{self.name}-123",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"exchange": self.name},
        )

    def cancel_order(self, order_id: str, *, symbol=None) -> None:
        self.cancelled.append(order_id)

    def stream_public_data(self, *, channels):  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - nieużywane
        raise NotImplementedError


def _context() -> ExecutionContext:
    return ExecutionContext(portfolio_id="P1", risk_profile="conservative", environment="live", metadata={})


def _request(symbol: str = "BTCUSDT", *, quantity: float = 1.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, side="buy", quantity=quantity, order_type="market")


def test_router_uses_fallback_on_network_error() -> None:
    primary = _DummyAdapter("primary", should_fail=True)
    secondary = _DummyAdapter("secondary", should_fail=False)
    metrics = MetricsRegistry()
    router = LiveExecutionRouter(
        adapters={"primary": primary, "secondary": secondary},
        default_route=("primary", "secondary"),
        metrics_registry=metrics,
    )

    try:
        result = router.execute(_request(), _context())

        assert result.order_id.startswith("secondary")
        assert primary.executed == []
        assert len(secondary.executed) == 1
        fallback_counter = metrics.get("live_router_fallbacks_total")
        assert (
            fallback_counter.value(
                labels={"exchange": "secondary", "symbol": "BTCUSDT", "portfolio": "P1"}
            )
            == 1.0
        )
    finally:
        router.close()


def test_router_records_failure_metric_when_all_exchanges_fail() -> None:
    adapters: Mapping[str, _DummyAdapter] = {
        "primary": _DummyAdapter("primary", should_fail=True),
        "secondary": _DummyAdapter("secondary", should_fail=True),
    }
    metrics = MetricsRegistry()
    router = LiveExecutionRouter(
        adapters=adapters,
        default_route=("primary", "secondary"),
        metrics_registry=metrics,
    )

    try:
        with pytest.raises(ExchangeNetworkError):
            router.execute(_request(), _context())

        failures = metrics.get("live_router_failures_total")
        assert failures.value(labels={"symbol": "BTCUSDT", "portfolio": "P1"}) == 1.0
    finally:
        router.close()


def test_router_cancel_delegates_to_origin_adapter() -> None:
    primary = _DummyAdapter("primary", should_fail=False)
    router = LiveExecutionRouter(adapters={"primary": primary}, default_route=("primary",))

    try:
        result = router.execute(_request(), _context())
        router.cancel(result.order_id, _context())

        assert primary.cancelled == [result.order_id]
    finally:
        router.close()


def test_router_uses_overrides_when_available() -> None:
    adapters: Mapping[str, _DummyAdapter] = {
        "a": _DummyAdapter("a", should_fail=True),
        "b": _DummyAdapter("b"),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        default_route=("a",),
        route_overrides={"ETHUSDT": ("b",)},
    )

    try:
        result = router.execute(_request(symbol="ETHUSDT"), _context())
        assert result.order_id.startswith("b")
    finally:
        router.close()
