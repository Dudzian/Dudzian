from __future__ import annotations

from typing import Protocol, Sequence

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


class CanaryExchangeAdapter(ExchangeAdapter):
    """DI canary: every live I/O method records and fails if touched."""

    name = "canary"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="canary-test-key"))
        self.calls: list[str] = []

    def _fail(self, method_name: str) -> None:
        self.calls.append(method_name)
        raise AssertionError(f"disabled LiveExecutionRouter touched live I/O: {method_name}")

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:
        del ip_allowlist
        self._fail("configure_network")

    def fetch_account_snapshot(self) -> AccountSnapshot:
        self._fail("fetch_account_snapshot")

    def fetch_symbols(self) -> Sequence[str]:
        self._fail("fetch_symbols")

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        del symbol, interval, start, end, limit
        self._fail("fetch_ohlcv")

    def place_order(self, request: OrderRequest) -> OrderResult:
        del request
        self._fail("place_order")

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        del order_id, symbol
        self._fail("cancel_order")

    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        del channels
        self._fail("stream_public_data")

    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        del channels
        self._fail("stream_private_data")


def _request() -> OrderRequest:
    return OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        client_order_id="preview-disabled-canary",
    )


def _context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="preview",
        risk_profile="safety",
        environment="live",
        metadata={"symbol": "BTCUSDT"},
    )


def test_test_mode_disabled_router_fails_closed_before_touching_injected_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUDZIAN_TEST_MODE", "1")
    monkeypatch.delenv("DUDZIAN_ALLOW_LIVE_ROUTER", raising=False)
    canary = CanaryExchangeAdapter()

    router = LiveExecutionRouter(adapters={"canary": canary}, default_route="canary")

    stats = router.get_runtime_stats()
    assert stats.closed is True
    assert stats.queue_depth == 0
    assert canary.calls == []

    with pytest.raises(RuntimeError, match="LiveExecutionRouter został zamknięty"):
        router.execute(_request(), _context())

    router.cancel("preview-order-id", _context())
    router.flush()
    router.close()

    assert canary.calls == []
    assert not router._loop_thread.is_alive()  # noqa: SLF001 - safety proof for no runtime loop


def test_allow_live_router_env_opt_in_contract_keeps_router_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUDZIAN_TEST_MODE", "1")
    monkeypatch.setenv("DUDZIAN_ALLOW_LIVE_ROUTER", "1")
    canary = CanaryExchangeAdapter()
    router = LiveExecutionRouter(adapters={"canary": canary}, default_route="canary")

    try:
        stats = router.get_runtime_stats()
        assert stats.closed is False
        assert router._loop_thread.is_alive()  # noqa: SLF001 - env contract proof
        assert canary.calls == []
    finally:
        router.close()

    assert canary.calls == []
