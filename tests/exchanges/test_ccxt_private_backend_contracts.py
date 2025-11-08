from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from bot_core.exchanges import manager
from bot_core.exchanges.core import Mode, OrderStatus, OrderType, OrderSide


@dataclass
class _FakeOrder:
    symbol: str
    side: str
    type: str
    amount: float
    price: float | None
    params: dict[str, Any]


class _FakeCCXTClient:
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.orders: List[_FakeOrder] = []
        self.cancelled: List[tuple[Any, Optional[str], dict[str, Any] | None]] = []
        self.open_orders: List[Dict[str, Any]] = []
        self.positions: List[Dict[str, Any]] = []
        self.balance: Dict[str, Any] = {}
        self.sandbox_mode: List[bool] = []

    # CCXT compatibility -------------------------------------------------
    def setSandboxMode(self, enabled: bool) -> None:  # noqa: N802 - zgodnie z CCXT
        self.sandbox_mode.append(bool(enabled))

    def load_markets(self) -> Dict[str, Any]:
        return {
            "BTC/USDT": {
                "limits": {
                    "amount": {"min": 0.001, "step": 0.001},
                    "price": {"min": 10.0, "step": 0.1},
                    "cost": {"min": 10.0},
                },
                "precision": {"amount": 3, "price": 1},
            }
        }

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:  # noqa: ARG002
        return {"last": 100.0}

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float],
        params: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = _FakeOrder(symbol, side, order_type, amount, price, params or {})
        self.orders.append(payload)
        response: Dict[str, Any] = {
            "id": "123456",
            "status": "open",
            "amount": amount,
            "remaining": max(0.0, amount - (amount * 0.25)),
            "filled": amount * 0.25,
            "price": price,
            "average": price,
            "clientOrderId": (params or {}).get("newClientOrderId"),
        }
        return response

    def cancel_order(
        self,
        order_id: Any,
        symbol: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.cancelled.append((order_id, symbol, params))

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        return list(self.open_orders)

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        return list(self.positions)

    def fetch_balance(self) -> Dict[str, Any]:
        return dict(self.balance)


class _Module:
    def __init__(self) -> None:
        self.created: List[_FakeCCXTClient] = []

    def binance(self, options: Dict[str, Any]) -> _FakeCCXTClient:  # type: ignore[override]
        client = _FakeCCXTClient(options)
        self.created.append(client)
        return client


def _build_backend(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> manager._CCXTPrivateBackend:  # type: ignore[attr-defined]
    module = _Module()
    monkeypatch.setattr(manager, "ccxt", module)
    backend = manager._CCXTPrivateBackend(exchange_id="binance", **kwargs)
    assert module.created  # sanity
    backend.load_markets()
    return backend


def test_create_order_maps_response(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret")
    result = backend.create_order(
        "BTC/USDT",
        OrderSide.BUY,
        OrderType.LIMIT,
        quantity=0.01,
        price=20100.12,
        client_order_id="cli-1",
    )

    assert result.status is OrderStatus.PARTIALLY_FILLED
    assert result.client_order_id == "cli-1"
    assert result.extra["filled_quantity"] == pytest.approx(0.0025)
    assert result.extra["remaining_quantity"] == pytest.approx(0.0075)
    assert result.extra["order_id"] == "123456"
    assert isinstance(result.extra["raw_response"], dict)
    assert backend.client.orders[0].params["newClientOrderId"] == "cli-1"
    assert backend.mode is Mode.SPOT


def test_fetch_open_orders_maps_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret")
    backend.client.open_orders = [
        {
            "id": "789",
            "status": "open",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "limit",
            "amount": 0.1,
            "remaining": 0.04,
            "price": 21000.0,
            "filled": 0.02,
            "average": 20500.0,
            "info": {"clientOrderId": "abc-1"},
        }
    ]

    orders = backend.fetch_open_orders("BTC/USDT")

    assert len(orders) == 1
    assert orders[0].client_order_id == "abc-1"
    assert orders[0].side is OrderSide.SELL
    assert orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert orders[0].extra["filled_quantity"] == pytest.approx(0.02)
    assert orders[0].extra["remaining_quantity"] == pytest.approx(0.04)
    assert orders[0].extra["avg_price"] == pytest.approx(20500.0)


def test_fetch_positions_returns_futures_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret", market_type="future")
    backend.client.positions = [
        {
            "symbol": "BTCUSDT",
            "contracts": 2,
            "entryPrice": 20000.0,
            "unrealizedPnl": 12.5,
        }
    ]

    positions = backend.fetch_positions()

    assert len(positions) == 1
    assert positions[0].mode is Mode.FUTURES
    assert positions[0].quantity == pytest.approx(2.0)


def test_fetch_positions_handles_negative_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret", market_type="future")
    backend.client.positions = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "-0.75",
            "entryPrice": "20500.5",
            "unrealizedPnl": "-5.5",
        }
    ]

    positions = backend.fetch_positions()

    assert len(positions) == 1
    assert positions[0].side == "SHORT"
    assert positions[0].quantity == pytest.approx(0.75)
    assert positions[0].avg_price == pytest.approx(20500.5)
    assert positions[0].unrealized_pnl == pytest.approx(-5.5)


def test_fetch_positions_uses_side_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret", market_type="future")
    backend.client.positions = [
        {
            "symbol": "ETHUSDT",
            "contracts": "1.5",
            "side": "short",
            "entryPrice": "1500",
            "info": {"unrealizedPnl": "1.2"},
        }
    ]

    positions = backend.fetch_positions()

    assert len(positions) == 1
    assert positions[0].side == "SHORT"
    assert positions[0].quantity == pytest.approx(1.5)
    assert positions[0].unrealized_pnl == pytest.approx(1.2)


def test_fetch_positions_respects_symbol_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret", market_type="future")
    backend.client.positions = [
        {"symbol": "BTC/USDT", "contracts": "2"},
        {"symbol": "ETHUSDT", "contracts": "1"},
    ]

    positions = backend.fetch_positions("BTCUSDT")

    assert len(positions) == 1
    assert positions[0].symbol in {"BTC/USDT", "BTCUSDT"}


def test_cancel_order_tracks_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret")

    assert backend.cancel_order("1", "BTC/USDT") is True
    assert backend.client.cancelled == [("1", "BTC/USDT", {})]


def test_cancel_order_without_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_backend(monkeypatch, api_key="key", secret="secret")

    assert backend.cancel_order("2") is True
    assert backend.client.cancelled[-1] == ("2", None, {})
