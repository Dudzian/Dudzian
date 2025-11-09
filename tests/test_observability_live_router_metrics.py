"""Testy potwierdzające rejestrację metryk przez LiveExecutionRouter."""
from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition
from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
)


class _StubStream(Protocol):
    def close(self) -> None:
        ...


class _StubAdapter(ExchangeAdapter):
    name = "stub"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="stub"))

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        raise NotImplementedError

    def stream_public_data(self, *, channels: Sequence[str]) -> _StubStream:
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]) -> _StubStream:
        raise NotImplementedError


_EXPECTED_METRICS: Mapping[str, type] = {
    "live_execution_latency_seconds": HistogramMetric,
    "live_orders_attempts_total": CounterMetric,
    "live_orders_failed_total": CounterMetric,
    "live_orders_fallback_total": CounterMetric,
    "live_orders_success_total": CounterMetric,
    "live_orders_total": CounterMetric,
    "live_router_fallbacks_total": CounterMetric,
    "live_router_failures_total": CounterMetric,
    "live_orders_fill_ratio": HistogramMetric,
    "live_orders_errors_total": CounterMetric,
    "live_breaker_open": GaugeMetric,
    "live_execution_queue_depth": GaugeMetric,
}


def test_live_router_registers_metrics_on_startup() -> None:
    registry = MetricsRegistry()
    adapter = _StubAdapter()
    router = LiveExecutionRouter(
        adapters={adapter.name: adapter},
        routes=[RouteDefinition(name="primary", exchanges=(adapter.name,))],
        metrics_registry=registry,
    )

    try:
        for metric_name, metric_type in _EXPECTED_METRICS.items():
            metric = registry.get(metric_name)
            assert isinstance(metric, metric_type)
    finally:
        router.close()
