"""Testy modułu :mod:`bot_core.exchanges.manager`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import bot_core.exchanges.manager as manager_module

from bot_core.exchanges.manager import ExchangeManager
from bot_core.exchanges.base import AccountSnapshot, OrderResult
from bot_core.exchanges.core import Mode
from bot_core.exchanges.health import (
    CircuitBreaker,
    HealthCheck,
    HealthMonitor,
    RetryPolicy,
    Watchdog,
)


class _DummyExchange:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.rateLimit = 200
        self.urls = {"api": "prod", "test": "test"}
        self._order_calls: List[Any] = []
        self._canceled: List[Any] = []
        self.markets = {
            "BTC/USDT": {
                "precision": {"amount": 3, "price": 2},
                "limits": {
                    "amount": {"min": 0.001},
                    "price": {"min": 1.0},
                    "cost": {"min": 10.0},
                },
            }
        }

    def set_sandbox_mode(self, value: bool) -> None:  # pragma: no cover - used when futures
        self.sandbox = value

    def load_markets(self) -> Dict[str, Any]:
        return self.markets

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return {"last": 100.0}

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        levels = [[100.0 + i * 0.1, 0.05] for i in range(limit)]
        return {"asks": levels, "bids": levels}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> List[List[float]]:
        return [[1, 99.0, 101.0, 98.0, 100.0, 1.0]] * limit

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._order_calls.append((symbol, order_type, side, amount, price, params))
        return {"id": "1", "status": "open", "symbol": symbol, "type": order_type, "side": side}

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "id": "2",
                "symbol": symbol or "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "amount": 0.1,
                "price": 100.0,
            }
        ]

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return [
            {
                "symbol": "BTC/USDT",
                "contracts": 0.2,
                "entryPrice": 90.0,
                "unrealizedPnl": 2.0,
            }
        ]

    def fetch_balance(self) -> Dict[str, Any]:
        return {"total": {"BTC": 0.25, "USDT": 200.0}}

    def cancel_order(self, order_id: Any, symbol: str) -> Dict[str, Any]:
        self._canceled.append((order_id, symbol))
        return {"id": order_id}


class _StubMarginAdapter:
    instances: List["_StubMarginAdapter"] = []

    def __init__(self, credentials, *, environment, settings=None, **kwargs) -> None:
        self.credentials = credentials
        self.environment = environment
        self.settings = settings or {}
        self.orders: List[Any] = []
        self.canceled: List[Any] = []
        self.watchdog = kwargs.get("watchdog")
        _StubMarginAdapter.instances.append(self)

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 500.0, "BTC": 0.5},
            total_equity=1500.0,
            available_margin=1000.0,
            maintenance_margin=120.0,
        )

    def place_order(self, request) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            order_id="M-1",
            status="NEW",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"clientOrderId": request.client_order_id or "generated"},
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        self.canceled.append((order_id, symbol))

    def fetch_open_orders(self):
        return [
            SimpleNamespace(
                order_id="M-2",
                symbol="BTC/USDT",
                status="NEW",
                side="BUY",
                order_type="LIMIT",
                price=101.0,
                orig_quantity=0.2,
                client_order_id="margin-open",
            )
        ]


class _StubFuturesAdapter(_StubMarginAdapter):
    instances: List["_StubFuturesAdapter"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _StubFuturesAdapter.instances.append(self)

    def fetch_positions(self):
        return [
            SimpleNamespace(
                symbol="BTCUSDT",
                side="LONG",
                quantity=0.3,
                entry_price=90.0,
                unrealized_pnl=12.5,
            )
        ]

@pytest.fixture(autouse=True)
def stub_ccxt(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(binance=_DummyExchange)
    monkeypatch.setattr("bot_core.exchanges.manager.ccxt", module)


@pytest.fixture(autouse=True)
def stub_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubMarginAdapter.instances.clear()
    _StubFuturesAdapter.instances.clear()
    monkeypatch.setitem(manager_module._NATIVE_MARGIN_ADAPTERS, "binance", _StubMarginAdapter)
    monkeypatch.setitem(manager_module._NATIVE_FUTURES_ADAPTERS, "binance", _StubFuturesAdapter)


def test_load_markets_and_quantizers() -> None:
    manager = ExchangeManager()
    rules = manager.load_markets()
    assert "BTC/USDT" in rules
    assert manager.quantize_amount("BTC/USDT", 0.12345) == pytest.approx(0.123)
    assert manager.quantize_price("BTC/USDT", 100.567) == pytest.approx(100.56)


def test_paper_market_order_updates_balance() -> None:
    manager = ExchangeManager()
    manager.load_markets()
    order = manager.create_order("BTC/USDT", "BUY", "MARKET", 0.1)
    assert order.quantity > 0
    balance = manager.fetch_balance()
    assert "total" in balance


def test_spot_limit_order_and_positions() -> None:
    manager = ExchangeManager()
    manager.set_mode(spot=True)
    manager.set_credentials("key", "secret")
    manager.load_markets()
    order = manager.create_order("BTC/USDT", "SELL", "LIMIT", 0.2, price=101.23)
    assert order.type.value == "LIMIT"
    assert order.side.value == "SELL"
    assert manager.cancel_order("abc", "BTC/USDT") is True
    open_orders = manager.fetch_open_orders()
    assert open_orders and open_orders[0].symbol == "BTC/USDT"
    positions = manager.fetch_positions()
    assert positions and positions[0].symbol.startswith("BTC")


def test_futures_positions_from_backend() -> None:
    manager = ExchangeManager()
    manager.set_mode(futures=True, testnet=True)
    manager.set_credentials("key", "secret")
    manager.load_markets()
    positions = manager.fetch_positions()
    assert positions and positions[0].mode.value == "futures"


def test_simulate_vwap_price() -> None:
    manager = ExchangeManager()
    manager.load_markets()
    price, bps = manager.simulate_vwap_price("BTC/USDT", "buy", 0.2)
    assert price is not None
    assert bps >= 0


def test_margin_mode_uses_native_adapter() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(margin=True)
    manager.configure_native_adapter(settings={"margin_type": "isolated"})
    manager.set_credentials("key", "secret")
    manager.load_markets()
    order = manager.create_order("BTC/USDT", "BUY", "LIMIT", 0.1, price=101.0, client_order_id="margin-open")
    assert order.mode is Mode.MARGIN
    balance = manager.fetch_balance()
    assert balance["total_equity"] == pytest.approx(1500.0)
    open_orders = manager.fetch_open_orders()
    assert open_orders and open_orders[0].mode is Mode.MARGIN
    assert manager.cancel_order("M-1", "BTC/USDT") is True


def test_native_adapter_receives_watchdog_instance() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(margin=True)
    manager.set_credentials("key", "secret")
    manager.load_markets()

    balance = manager.fetch_balance()
    assert balance["total_equity"] == pytest.approx(1500.0)
    assert len(_StubMarginAdapter.instances) == 1
    adapter = _StubMarginAdapter.instances[-1]
    assert isinstance(adapter.watchdog, Watchdog)

    order = manager.create_order("BTC/USDT", "BUY", "LIMIT", 0.1, price=101.0)
    assert order.mode is Mode.MARGIN
    assert adapter.watchdog is not None
    assert _StubMarginAdapter.instances[-1] is adapter


def test_configure_watchdog_builds_custom_policy() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(margin=True)
    manager.set_credentials("key", "secret")
    manager.configure_watchdog(
        retry_policy={"max_attempts": 5, "base_delay": 0.1, "max_delay": 0.2, "jitter": (0.0, 0.0)},
        circuit_breaker={"failure_threshold": 2, "recovery_timeout": 1.0, "half_open_success_threshold": 1},
    )
    manager.fetch_balance()

    adapter = _StubMarginAdapter.instances[-1]
    assert adapter.watchdog is not None
    assert adapter.watchdog.retry_policy.max_attempts == 5
    assert adapter.watchdog.circuit_breaker.failure_threshold == 2


def test_set_watchdog_replaces_existing_instance() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(margin=True)
    manager.set_credentials("key", "secret")
    manager.fetch_balance()
    first_adapter = _StubMarginAdapter.instances[-1]
    default_watchdog = first_adapter.watchdog
    assert isinstance(default_watchdog, Watchdog)

    custom_watchdog = Watchdog(
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0)),
        circuit_breaker=CircuitBreaker(failure_threshold=1, recovery_timeout=0.5, half_open_success_threshold=1),
        sleep=lambda _: None,
    )
    manager.set_watchdog(custom_watchdog)

    manager.fetch_balance()
    assert len(_StubMarginAdapter.instances) == 2
    second_adapter = _StubMarginAdapter.instances[-1]
    assert second_adapter.watchdog is custom_watchdog


def test_configure_watchdog_validates_types() -> None:
    manager = ExchangeManager(exchange_id="binance")
    with pytest.raises(TypeError):
        manager.configure_watchdog(retry_policy="invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manager.configure_watchdog(circuit_breaker="invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manager.configure_watchdog(retry_exceptions="error")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manager.configure_watchdog(retry_exceptions=["not-exception"])  # type: ignore[list-item]


def test_create_health_monitor_uses_shared_watchdog() -> None:
    manager = ExchangeManager(exchange_id="binance")
    custom_watchdog = Watchdog(
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0)),
        circuit_breaker=CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_success_threshold=1),
        sleep=lambda _: None,
    )
    manager.set_watchdog(custom_watchdog)

    check = HealthCheck(name="noop", check=lambda: None)
    monitor = manager.create_health_monitor([check])

    assert isinstance(monitor, HealthMonitor)
    assert monitor._watchdog is custom_watchdog  # type: ignore[attr-defined]
    results = monitor.run()
    assert results and results[0].name == "noop"


def test_create_health_monitor_validates_input() -> None:
    manager = ExchangeManager(exchange_id="binance")
    with pytest.raises(TypeError):
        manager.create_health_monitor(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manager.create_health_monitor(["invalid"])  # type: ignore[list-item]


def test_futures_mode_uses_native_adapter() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(futures=True, testnet=True)
    manager.set_credentials("key", "secret")
    manager.load_markets()
    order = manager.create_order("BTC/USDT", "SELL", "MARKET", 0.3)
    assert order.mode is Mode.FUTURES
    balance = manager.fetch_balance()
    assert balance["available_margin"] == pytest.approx(1000.0)
    positions = manager.fetch_positions()
    assert positions and positions[0].mode is Mode.FUTURES

