"""Testy jednostkowe adaptera Kraken Spot."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl
from urllib.request import Request

import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _build_credentials() -> ExchangeCredentials:
    secret = base64.b64encode(b"super-secret").decode("utf-8")
    return ExchangeCredentials(
        key_id="kraken-key",
        secret=secret,
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )


def test_fetch_account_snapshot_uses_private_signed_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured_requests.append(request)
        if request.full_url.endswith("TradeBalance"):
            payload = {"error": [], "result": {"eb": "1200.0", "mf": "800.0", "m": "200.0"}}
        else:
            payload = {"error": [], "result": {"ZUSD": "1000.0", "XXBT": "0.5"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    adapter = KrakenSpotAdapter(_build_credentials(), environment=Environment.LIVE)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["ZUSD"] == pytest.approx(1000.0)
    assert snapshot.total_equity == pytest.approx(1200.0)
    assert snapshot.available_margin == pytest.approx(800.0)
    assert snapshot.maintenance_margin == pytest.approx(200.0)

    assert len(captured_requests) == 2
    first_headers = {name.lower(): value for name, value in captured_requests[0].header_items()}
    assert first_headers["api-key"] == "kraken-key"
    # Sprawdzamy deterministycznie podpis API-Sign.
    signature = first_headers["api-sign"]
    body_params = dict(parse_qsl(captured_requests[0].data.decode("utf-8")))
    expected_signature = _expected_signature(
        path="/0/private/Balance",
        params=body_params,
        secret=_build_credentials().secret or "",
    )
    assert signature == expected_signature
    trade_params = dict(parse_qsl(captured_requests[1].data.decode("utf-8")))
    assert trade_params.get("asset") == "ZUSD"


def test_fetch_account_snapshot_respects_custom_valuation_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured_requests.append(request)
        if request.full_url.endswith("TradeBalance"):
            payload = {"error": [], "result": {"eb": "100.0", "mf": "80.0", "m": "20.0"}}
        else:
            payload = {"error": [], "result": {"ZEUR": "50.0"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        settings={"valuation_asset": "eur"},
    )

    adapter.fetch_account_snapshot()
    assert len(captured_requests) == 2
    trade_params = dict(parse_qsl(captured_requests[1].data.decode("utf-8")))
    assert trade_params.get("asset") == "ZEUR"


def test_place_order_builds_payload_with_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {"error": [], "result": {"txid": ["OID123"], "descr": {"order": "buy 0.1 XBTUSD"}}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    adapter = KrakenSpotAdapter(_build_credentials(), environment=Environment.LIVE)

    order = OrderRequest(
        symbol="XBTUSD",
        side="buy",
        quantity=0.1,
        order_type="limit",
        price=25_000.0,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(order)

    assert result.order_id == "OID123"
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["api-key"] == "kraken-key"
    signature = headers["api-sign"]
    body_params = dict(parse_qsl(captured_request.data.decode("utf-8"))) if captured_request else {}
    expected_signature = _expected_signature(
        path="/0/private/AddOrder",
        params=body_params,
        secret=_build_credentials().secret or "",
    )
    assert signature == expected_signature
    body = captured_request.data.decode("utf-8") if captured_request and captured_request.data else ""
    assert "pair=XBTUSD" in body
    assert "ordertype=limit" in body
    assert "userref=cli-1" in body


def test_cancel_order_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = {"error": [], "result": {"count": 1}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    adapter = KrakenSpotAdapter(_build_credentials(), environment=Environment.LIVE)

    adapter.cancel_order("OID123")


def _expected_signature(*, path: str, params: dict[str, Any], secret: str) -> str:
    params = dict(params)
    nonce = params.pop("nonce", None)
    if nonce is None:
        raise AssertionError("Brak pola nonce w parametrów podpisu")
    sorted_items = sorted(params.items())
    encoded_params = "&".join(f"{k}={v}" for k, v in sorted_items)
    message = (nonce + encoded_params).encode("utf-8")
    sha_digest = hashlib.sha256(message).digest()
    decoded_secret = base64.b64decode(secret)
    mac = hmac.new(decoded_secret, (path.encode("utf-8") + sha_digest), hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("utf-8")
