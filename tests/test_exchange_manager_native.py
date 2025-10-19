"""Testy modułu :mod:`bot_core.exchanges.manager`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from bot_core.exchanges.manager import ExchangeManager


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


@pytest.fixture(autouse=True)
def stub_ccxt(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(binance=_DummyExchange)
    monkeypatch.setattr("bot_core.exchanges.manager.ccxt", module)


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
    assert "USDT" in balance


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

