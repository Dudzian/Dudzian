from __future__ import annotations

import types

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.spot import BinanceSpotAdapter


def _build_adapter() -> BinanceSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="binance-key",
        secret="binance-secret",
        permissions=("read", "trade"),
        environment=Environment.PAPER,
    )
    return BinanceSpotAdapter(credentials, environment=Environment.PAPER)


def test_cancel_order_success_with_valid_symbol() -> None:
    adapter = _build_adapter()
    captured: dict[str, object] = {}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        captured["path"] = path
        captured["method"] = method
        captured["params"] = dict(params or {})
        return {"status": "CANCELED"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    adapter.cancel_order("order-1", symbol="BTC/USDT")

    assert captured["path"] == "/api/v3/order"
    assert captured["method"] == "DELETE"
    assert captured["params"]["orderId"] == "order-1"
    assert captured["params"]["symbol"] == "BTCUSDT"


def test_cancel_order_without_symbol_fails_before_request() -> None:
    adapter = _build_adapter()
    called = {"signed_request": 0}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        called["signed_request"] += 1
        return {"status": "CANCELED"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    with pytest.raises(ValueError):
        adapter.cancel_order("order-2")

    assert called["signed_request"] == 0


def test_cancel_order_with_whitespace_symbol_fails_before_request() -> None:
    adapter = _build_adapter()
    called = {"signed_request": 0}

    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        called["signed_request"] += 1
        return {"status": "CANCELED"}

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)

    with pytest.raises(ValueError):
        adapter.cancel_order("order-3", symbol="   ")

    assert called["signed_request"] == 0
