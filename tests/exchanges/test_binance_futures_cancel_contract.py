from __future__ import annotations

import types

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter


def _build_adapter() -> BinanceFuturesAdapter:
    credentials = ExchangeCredentials(
        key_id="binance-key",
        secret="binance-secret",
        permissions=("read", "trade"),
        environment=Environment.PAPER,
    )
    return BinanceFuturesAdapter(credentials, environment=Environment.PAPER)


def _install_cancel_response(adapter: BinanceFuturesAdapter, response: object) -> None:
    def fake_signed_request(self, path: str, *, method: str = "GET", params=None):
        assert path == "/fapi/v1/order"
        assert method == "DELETE"
        payload = dict(params or {})
        assert payload["orderId"] == "order-1"
        assert payload["symbol"] == "BTCUSDT"
        return response

    adapter._signed_request = types.MethodType(fake_signed_request, adapter)


def test_cancel_order_accepts_status_canceled() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"status": "CANCELED"})

    adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_accepts_status_pending_cancel() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"status": "PENDING_CANCEL"})

    adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_rejects_status_new() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"status": "NEW"})

    with pytest.raises(RuntimeError, match="Nieoczekiwana odpowiedź anulowania"):
        adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_rejects_other_statuses() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"status": "PARTIALLY_FILLED"})

    with pytest.raises(RuntimeError, match="Nieoczekiwana odpowiedź anulowania"):
        adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_rejects_non_mapping_response() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, ["unexpected"])

    with pytest.raises(RuntimeError, match="Niepoprawna odpowiedź anulowania"):
        adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_rejects_mapping_without_status() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"orderId": "order-1"})

    with pytest.raises(RuntimeError, match="Nieoczekiwana odpowiedź anulowania"):
        adapter.cancel_order("order-1", symbol="BTC/USDT")


def test_cancel_order_rejects_mapping_with_none_status() -> None:
    adapter = _build_adapter()
    _install_cancel_response(adapter, {"status": None})

    with pytest.raises(RuntimeError, match="Nieoczekiwana odpowiedź anulowania"):
        adapter.cancel_order("order-1", symbol="BTC/USDT")
