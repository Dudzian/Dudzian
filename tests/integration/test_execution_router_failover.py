"""Integration tests for live/paper execution routing with CCXT mocks."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Mapping, Sequence

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.live_router import LiveExecutionRouter, QoSConfig
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.observability.metrics import MetricsRegistry


class DummyStream:
    def close(self) -> None:  # pragma: no cover - compatibility helper
        pass


class FakeAdapter(ExchangeAdapter):
    """Minimal ExchangeAdapter implementation used in tests."""

    def __init__(self, name: str, *, side_effect=None, result: OrderResult | None = None) -> None:
        super().__init__(ExchangeCredentials(key_id=f"{name}-key"))
        self.name = name
        self._side_effect = side_effect
        self._result = result
        self.calls: list[OrderRequest] = []
        self._active_calls = 0
        self.max_active_calls = 0
        self._lock = threading.Lock()

    # --- ExchangeAdapter abstract API -------------------------------------
    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # pragma: no cover - not used
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:  # pragma: no cover - not used
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self) -> Iterable[str]:  # pragma: no cover - not used
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # pragma: no cover - not used
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        with self._lock:
            self.calls.append(request)
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            if callable(self._side_effect):
                return self._side_effect(request)
            if isinstance(self._side_effect, BaseException):
                raise self._side_effect
            if self._result is not None:
                time.sleep(0.05)
                return self._result
            raise RuntimeError("No result configured for FakeAdapter")
        finally:
            with self._lock:
                self._active_calls -= 1

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover - not used
        return None

    def stream_public_data(self, *, channels: Sequence[str]) -> DummyStream:  # pragma: no cover - not used
        return DummyStream()

    def stream_private_data(self, *, channels: Sequence[str]) -> DummyStream:  # pragma: no cover - not used
        return DummyStream()


@pytest.fixture()
def execution_context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="portfolio-1",
        risk_profile="balanced",
        environment="demo",
        metadata={},
    )


@pytest.mark.integration
def test_live_router_failover_to_backup_exchange(execution_context: ExecutionContext) -> None:
    metrics = MetricsRegistry()
    failing_adapter = FakeAdapter("primary", side_effect=ExchangeNetworkError("down"))
    successful_result = OrderResult(
        order_id="backup-1",
        status="filled",
        filled_quantity=1.0,
        avg_price=101.5,
        raw_response={"exchange": "backup"},
    )
    backup_adapter = FakeAdapter("backup", result=successful_result)

    router = LiveExecutionRouter(
        adapters={"primary": failing_adapter, "backup": backup_adapter},
        default_route=("primary", "backup"),
        metrics_registry=metrics,
        qos=QoSConfig(worker_concurrency=2, per_exchange_concurrency={"primary": 1, "backup": 1}),
    )

    request = OrderRequest(symbol="BTC/USDT", side="buy", quantity=1.0, order_type="market")

    try:
        result = router.execute(request, execution_context)
    finally:
        router.close()

    assert result.order_id == "backup-1"
    assert failing_adapter.calls, "Primary adapter should receive at least one attempt"
    assert backup_adapter.calls, "Backup adapter should handle the fallback order"
    assert router.binding_for_order(result.order_id) == "backup"


@pytest.mark.integration
def test_live_router_enforces_rate_limit_per_exchange(execution_context: ExecutionContext) -> None:
    metrics = MetricsRegistry()
    response = OrderResult(
        order_id="primary-1",
        status="filled",
        filled_quantity=0.5,
        avg_price=100.0,
        raw_response={"exchange": "primary"},
    )
    rate_limited_adapter = FakeAdapter("primary", result=response)

    router = LiveExecutionRouter(
        adapters={"primary": rate_limited_adapter},
        default_route=("primary",),
        metrics_registry=metrics,
        qos=QoSConfig(worker_concurrency=2, per_exchange_concurrency={"primary": 1}),
    )

    request_a = OrderRequest(symbol="ETH/USDT", side="buy", quantity=1.0, order_type="market")
    request_b = OrderRequest(symbol="ETH/USDT", side="buy", quantity=2.0, order_type="market")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(router.execute, request_a, execution_context)
            future_b = executor.submit(router.execute, request_b, execution_context)
            future_a.result(timeout=2)
            future_b.result(timeout=2)
    finally:
        router.close()

    assert rate_limited_adapter.max_active_calls == 1, "Per-exchange semaphore should serialize requests"


@pytest.mark.integration
def test_paper_service_respects_market_constraints(execution_context: ExecutionContext) -> None:
    markets: Mapping[str, MarketMetadata] = {
        "BTC/USDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            tick_size=0.5,
            step_size=0.001,
            min_notional=5.0,
            min_quantity=0.001,
        )
    }
    paper = PaperTradingExecutionService(markets, initial_balances={"USDT": 1000.0})

    request = OrderRequest(symbol="BTC/USDT", side="buy", quantity=0.05, order_type="market", price=101.0)

    result = paper.execute(request, execution_context)

    assert result.status == "filled"
    assert result.filled_quantity == pytest.approx(request.quantity)

    ledger_entries = list(paper.ledger())
    assert ledger_entries, "Paper router should record ledger entries for executed trades"

