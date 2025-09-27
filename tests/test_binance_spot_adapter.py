"""Testy jednostkowe adaptera Binance Spot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.spot import BinanceSpotAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_account_snapshot_parses_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "balances": [
                {"asset": "BTC", "free": "0.5", "locked": "0.1"},
                {"asset": "USDT", "free": "100", "locked": "0"},
            ],
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["BTC"] == pytest.approx(0.6)
    assert snapshot.balances["USDT"] == pytest.approx(100.0)
    assert snapshot.total_equity == pytest.approx(100.6)
    assert snapshot.available_margin == pytest.approx(100.5)
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "test-key"
    assert "signature=" in captured_request.full_url


def test_place_order_builds_signed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "orderId": 12345,
            "status": "NEW",
            "executedQty": "0",
            "price": "0",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    order_request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=20_000,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(order_request)

    assert result.order_id == "12345"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0)
    assert result.avg_price is None
    assert captured_request is not None
    assert captured_request.get_method() == "POST"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    body = captured_request.data.decode("utf-8") if captured_request.data else ""
    expected_prefix = "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=0.01&price=20000&timeInForce=GTC&newClientOrderId=cli-1&timestamp=1700000000000"
    assert body.startswith(expected_prefix)
    assert "signature=" in body


def test_cancel_order_uses_delete_method(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {"status": "CANCELED"}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    adapter.cancel_order("123", symbol="BTCUSDT")

    assert captured_request is not None
    assert captured_request.get_method() == "DELETE"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    assert "symbol=BTCUSDT" in captured_request.full_url
    assert "orderId=123" in captured_request.full_url
    assert "signature=" in captured_request.full_url
