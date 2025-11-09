"""Testy integracyjne dla failoveru ExchangeManagera z wykorzystaniem atrap CCXT."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from bot_core.exchanges.errors import ExchangeNetworkError, ExchangeThrottlingError
from bot_core.exchanges.manager import (
    ExchangeManager,
    Mode,
    RateLimitRule,
    register_native_adapter,
    unregister_native_adapter,
)


class _DummyCCXTModule:
    """Lekka atrapa modułu ``ccxt`` wykorzystana w testach."""

    class NetworkError(Exception):
        """Symuluje ``ccxt.NetworkError`` do celów monitoringu."""

    def __init__(self, *, ticker_price: float = 101.0) -> None:
        self.ticker_price = ticker_price
        self.clients: dict[str, list[object]] = {}

    def __getattr__(self, exchange_id: str) -> type["_Client"]:
        if exchange_id.startswith("__"):
            raise AttributeError(exchange_id)

        module = self

        class _Client:  # noqa: D401 - prosta atrapa klienta CCXT
            def __init__(self, options: Mapping[str, object]) -> None:
                self.options = dict(options)
                self.exchange_id = exchange_id
                self.order_counter = 0
                self.orders: list[tuple] = []
                self.sandbox_enabled = False

            def setSandboxMode(self, enabled: bool) -> None:  # pragma: no cover - konfiguracja testowa
                self.sandbox_enabled = bool(enabled)

            def set_sandbox_mode(self, enabled: bool) -> None:  # pragma: no cover - alias
                self.setSandboxMode(enabled)

            def load_markets(self) -> Mapping[str, object]:
                return {
                    "BTC/USDT": {
                        "limits": {
                            "amount": {"min": 0.001, "step": 0.001},
                            "price": {"step": 0.1},
                            "cost": {"min": 1.0},
                        },
                        "precision": {"amount": 3, "price": 1},
                    }
                }

            def fetch_ticker(self, symbol: str) -> Mapping[str, float]:
                return {"last": float(module.ticker_price)}

            def create_order(
                self,
                symbol: str,
                order_type: str,
                side: str,
                quantity: float,
                price: float | None,
                params: Mapping[str, object] | None,
            ) -> Mapping[str, object]:
                self.order_counter += 1
                order_id_int = self.order_counter
                order_id = f"{self.exchange_id}-order-{order_id_int}"
                self.orders.append((symbol, order_type, side, quantity, price, dict(params or {})))
                return {
                    "id": order_id_int,
                    "orderId": order_id,
                    "clientOrderId": (params or {}).get("newClientOrderId"),
                    "filled": float(quantity),
                    "remaining": 0.0,
                    "amount": float(quantity),
                    "average": float(price) if price is not None else float(module.ticker_price),
                    "status": "closed",
                }

            def fetch_balance(self) -> Mapping[str, object]:  # pragma: no cover - zgodność API
                return {"total": {}, "free": {}, "used": {}}

        return _Client


class _FailingNativeAdapter:
    """Atrapa natywnego adaptera zgłaszająca wyjątek z dostarczonej fabryki."""

    def __init__(
        self,
        error_factory,
        *,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        self._error_factory = error_factory
        self.calls = 0
        self.settings = dict(settings or {})

    def place_order(self, request) -> None:  # pragma: no cover - proste zachowanie
        self.calls += 1
        raise self._error_factory()


def _register_failing_native(
    *,
    exchange_id: str,
    error_factory,
    captured: list[_FailingNativeAdapter],
) -> None:
    def factory(credentials, **kwargs):  # pragma: no cover - konstrukcja atrapy
        adapter = _FailingNativeAdapter(error_factory, settings=kwargs.get("settings"))
        captured.append(adapter)
        return adapter

    register_native_adapter(exchange_id=exchange_id, mode=Mode.MARGIN, factory=factory)


@pytest.fixture()
def ccxt_stub(monkeypatch) -> _DummyCCXTModule:
    stub = _DummyCCXTModule()
    monkeypatch.setattr("bot_core.exchanges.manager.ccxt", stub)
    return stub


@pytest.fixture()
def manager(exchange_id: str, ccxt_stub: _DummyCCXTModule) -> ExchangeManager:
    mgr = ExchangeManager(exchange_id)
    mgr.set_mode(margin=True)
    mgr.set_credentials("api-key", "secret")
    mgr.configure_failover(enabled=True, failure_threshold=1, cooldown_seconds=5.0)
    mgr.configure_rate_limits([RateLimitRule(rate=10, per=1.0)])
    return mgr


@pytest.fixture()
def exchange_id() -> str:
    return "testex"


@pytest.fixture(autouse=True)
def cleanup_native_adapter(exchange_id: str):
    try:
        yield
    finally:
        unregister_native_adapter(exchange_id=exchange_id, mode=Mode.MARGIN, allow_dynamic=True)


@pytest.mark.integration
def test_exchange_manager_failover_to_ccxt_backend(manager: ExchangeManager, exchange_id: str) -> None:
    captured: list[_FailingNativeAdapter] = []
    _register_failing_native(
        exchange_id=exchange_id,
        error_factory=lambda: ExchangeNetworkError("native backend unavailable"),
        captured=captured,
    )

    result = manager.create_order("BTC/USDT", "BUY", "MARKET", 0.01)

    assert result.extra["order_id"] == 1
    assert result.extra["raw_response"]["orderId"].startswith(f"{exchange_id}-order-"), "Fallback CCXT order ID expected"
    assert captured and captured[0].calls == 1, "Native adapter should be attempted exactly once"
    assert "rate_limit_rules" in captured[0].settings, "Shared rate limit rules should be passed to native adapter"


@pytest.mark.integration
def test_exchange_manager_stays_on_ccxt_after_rate_limit(
    manager: ExchangeManager, exchange_id: str
) -> None:
    captured: list[_FailingNativeAdapter] = []
    _register_failing_native(
        exchange_id=exchange_id,
        error_factory=lambda: ExchangeThrottlingError(429, "rate limit"),
        captured=captured,
    )

    first = manager.create_order("BTC/USDT", "BUY", "MARKET", 0.02)
    second = manager.create_order("BTC/USDT", "BUY", "MARKET", 0.03)

    assert first.extra["order_id"] == 1
    assert second.extra["order_id"] == 2, "Subsequent orders should reuse CCXT fallback"
    assert captured and captured[0].calls == 1, "Rate limit failure should switch permanently to CCXT during cooldown"
