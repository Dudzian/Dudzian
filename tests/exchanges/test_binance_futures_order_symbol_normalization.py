from __future__ import annotations

import types

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter


def _build_adapter() -> BinanceFuturesAdapter:
    credentials = ExchangeCredentials(
        key_id="binance-key",
        secret="binance-secret",
        permissions=("read", "trade"),
        environment=Environment.PAPER,
    )
    return BinanceFuturesAdapter(credentials, environment=Environment.PAPER)


def test_place_order_normalizes_contract_symbol() -> None:
    adapter = _build_adapter()
    captured: dict[str, object] = {}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        captured["path"] = path
        captured["method"] = method
        captured["params"] = dict(params or {})
        return {"orderId": 123, "status": "NEW", "executedQty": "0", "price": "0"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    adapter.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            order_type="limit",
            price=30000.0,
            time_in_force="GTC",
        )
    )

    assert captured["path"] == "/fapi/v1/order"
    assert captured["method"] == "POST"
    assert captured["params"]["symbol"] == "BTCUSDT"


def test_cancel_order_normalizes_contract_symbol() -> None:
    adapter = _build_adapter()
    captured: dict[str, object] = {}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        captured["path"] = path
        captured["method"] = method
        captured["params"] = dict(params or {})
        return {"status": "CANCELED"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    adapter.cancel_order("order-1", symbol="BTC/USDT")

    assert captured["path"] == "/fapi/v1/order"
    assert captured["method"] == "DELETE"
    assert captured["params"]["symbol"] == "BTCUSDT"


def test_order_paths_use_same_symbol_normalization_as_market_data() -> None:
    adapter = _build_adapter()
    captured: dict[str, object] = {}

    def fake_public_request(self, path: str, params=None, *, method: str = "GET"):
        captured["market_data_symbol"] = dict(params or {}).get("symbol")
        return {"bids": [], "asks": [], "lastUpdateId": 1, "E": 1_700_000_000_000}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        if method == "POST":
            captured["place_symbol"] = dict(params or {}).get("symbol")
            return {"orderId": 1, "status": "NEW", "executedQty": "0", "price": "0"}
        captured["cancel_symbol"] = dict(params or {}).get("symbol")
        return {"status": "CANCELED"}

    adapter._public_request = types.MethodType(fake_public_request, adapter)
    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    adapter.fetch_order_book("BTC/USDT", depth=5)
    adapter.place_order(
        OrderRequest(symbol="BTC/USDT", side="buy", quantity=0.01, order_type="market")
    )
    adapter.cancel_order("order-2", symbol="BTC/USDT")

    assert captured["market_data_symbol"] == "BTCUSDT"
    assert captured["place_symbol"] == "BTCUSDT"
    assert captured["cancel_symbol"] == "BTCUSDT"
