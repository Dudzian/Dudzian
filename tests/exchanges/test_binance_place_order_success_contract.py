from __future__ import annotations

import types

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter


def _spot_adapter() -> BinanceSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="spot-key",
        secret="spot-secret",
        permissions=("read", "trade"),
        environment=Environment.PAPER,
    )
    return BinanceSpotAdapter(credentials, environment=Environment.PAPER)


def _futures_adapter() -> BinanceFuturesAdapter:
    credentials = ExchangeCredentials(
        key_id="futures-key",
        secret="futures-secret",
        permissions=("read", "trade"),
        environment=Environment.PAPER,
    )
    return BinanceFuturesAdapter(credentials, environment=Environment.PAPER)


def _spot_request() -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=30000.0,
        time_in_force="GTC",
    )


def _futures_request() -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=30000.0,
        time_in_force="GTC",
    )


def test_spot_place_order_accepts_non_empty_order_id() -> None:
    adapter = _spot_adapter()

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        assert path == "/api/v3/order"
        assert method == "POST"
        return {"orderId": 123, "status": "NEW", "executedQty": "0", "price": "0"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    result = adapter.place_order(_spot_request())
    assert result.order_id == "123"


@pytest.mark.parametrize("payload", [{"orderId": ""}, {"orderId": None}, {"status": "NEW"}])
def test_spot_place_order_rejects_missing_or_empty_order_id(payload: dict[str, object]) -> None:
    adapter = _spot_adapter()

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        assert path == "/api/v3/order"
        assert method == "POST"
        return payload

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    with pytest.raises(RuntimeError, match="orderId"):
        adapter.place_order(_spot_request())


def test_futures_place_order_accepts_non_empty_order_id() -> None:
    adapter = _futures_adapter()

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        assert path == "/fapi/v1/order"
        assert method == "POST"
        return {"orderId": 456, "status": "NEW", "executedQty": "0", "avgPrice": "0"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    result = adapter.place_order(_futures_request())
    assert result.order_id == "456"


@pytest.mark.parametrize("payload", [{"orderId": ""}, {"orderId": None}, {"status": "NEW"}])
def test_futures_place_order_rejects_missing_or_empty_order_id(payload: dict[str, object]) -> None:
    adapter = _futures_adapter()

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        assert path == "/fapi/v1/order"
        assert method == "POST"
        return payload

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    with pytest.raises(RuntimeError, match="orderId"):
        adapter.place_order(_futures_request())
