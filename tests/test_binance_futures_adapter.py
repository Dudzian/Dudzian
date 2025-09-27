"""Testy jednostkowe adaptera Binance Futures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_account_snapshot_parses_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "assets": [
                {"asset": "USDT", "walletBalance": "120.5"},
                {"asset": "BUSD", "walletBalance": "50"},
            ],
            "totalMarginBalance": "180.5",
            "totalAvailableBalance": "150.5",
            "totalMaintMargin": "12.0",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="futures-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceFuturesAdapter(credentials)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["USDT"] == pytest.approx(120.5)
    assert snapshot.balances["BUSD"] == pytest.approx(50.0)
    assert snapshot.total_equity == pytest.approx(180.5)
    assert snapshot.available_margin == pytest.approx(150.5)
    assert snapshot.maintenance_margin == pytest.approx(12.0)
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "futures-key"
    assert "signature=" in captured_request.full_url
    assert captured_request.full_url.startswith("https://fapi.binance.com/fapi/v2/account")


def test_place_order_builds_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "orderId": 999,
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_100.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceFuturesAdapter(credentials)

    order_request = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.05,
        order_type="limit",
        price=25_000,
        time_in_force="GTC",
        client_order_id="abc-123",
    )

    result = adapter.place_order(order_request)

    assert result.order_id == "999"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0)
    assert result.avg_price is None
    assert captured_request is not None
    assert captured_request.get_method() == "POST"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    body = captured_request.data.decode("utf-8") if captured_request.data else ""
    expected_prefix = (
        "symbol=BTCUSDT&side=SELL&type=LIMIT&quantity=0.05&price=25000&timeInForce=GTC&newClientOrderId=abc-123"
    )
    assert body.startswith(expected_prefix)
    assert "signature=" in body
    assert captured_request.full_url.startswith("https://fapi.binance.com/fapi/v1/order")


def test_cancel_order_requires_symbol() -> None:
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceFuturesAdapter(credentials)

    with pytest.raises(ValueError):
        adapter.cancel_order("1")


def test_cancel_order_uses_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {"status": "CANCELED"}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_200.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceFuturesAdapter(credentials)

    adapter.cancel_order("42", symbol="BTCUSDT")

    assert captured_request is not None
    assert captured_request.get_method() == "DELETE"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    url = captured_request.full_url
    assert url.startswith("https://fapi.binance.com/fapi/v1/order")
    assert "symbol=BTCUSDT" in url
    assert "orderId=42" in url
    assert "signature=" in url


def test_fetch_symbols_filters_non_trading(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read",),
        environment=Environment.LIVE,
    )
    adapter = BinanceFuturesAdapter(credentials)

    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "CURRENT_QUARTER"},
            {"symbol": "BAD", "status": "HALT"},
            {"symbol": None, "status": "TRADING"},
        ]
    }

    monkeypatch.setattr(adapter, "_public_request", lambda path, params=None, method="GET": payload)

    symbols = list(adapter.fetch_symbols())

    assert symbols == ["BTCUSDT", "ETHUSDT"]

