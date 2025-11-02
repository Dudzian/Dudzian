import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest



from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter, _RequestContext
from bot_core.exchanges.health import CircuitBreaker, RetryPolicy, Watchdog


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _credentials() -> ExchangeCredentials:
    secret = base64.b64encode(b"kraken-futures-secret").decode("utf-8")
    return ExchangeCredentials(
        key_id="kraken-futures-key",
        secret=secret,
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )


def test_fetch_account_snapshot_signed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 20):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "result": "success",
            "accounts": {
                "futures": {
                    "balances": {
                        "USD": {"balance": "1000.0"},
                        "BTC": {"available": "0.05"},
                    },
                    "accountValue": "1200.0",
                    "availableMargin": "800.0",
                    "initialMargin": "200.0",
                }
            },
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.futures.time.time", lambda: 1_700_000_000.0)

    adapter = KrakenFuturesAdapter(_credentials(), environment=Environment.LIVE)
    adapter.configure_network()
    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["USD"] == pytest.approx(1000.0)
    assert snapshot.balances["BTC"] == pytest.approx(0.05)
    assert snapshot.total_equity == pytest.approx(1200.0)
    assert snapshot.available_margin == pytest.approx(800.0)
    assert snapshot.maintenance_margin == pytest.approx(200.0)

    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["apikey"] == "kraken-futures-key"
    expected_signature = _expected_signature(
        path="/accounts",
        body=b"",
        secret=_credentials().secret or "",
        nonce=headers["nonce"],
    )
    assert headers["authent"] == expected_signature


def test_place_order_builds_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 20):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        if request.get_method() == "POST":
            payload = {
                "result": "success",
                "sendStatus": {
                    "order_id": "OF12345",
                    "status": "accepted",
                    "orderEvents": [
                        {"type": "fill", "fill_size": "0.01", "price": "30000.0"}
                    ],
                },
            }
        else:
            payload = {"result": "success", "cancelStatus": {"status": "cancelled"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.futures.time.time", lambda: 1_700_000_000.0)

    adapter = KrakenFuturesAdapter(_credentials(), environment=Environment.LIVE)
    order = OrderRequest(
        symbol="pi_xbtusd",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=30_000.0,
        time_in_force="GTC",
        client_order_id="cli-kraken",
    )
    adapter.configure_network()

    result = adapter.place_order(order)
    assert result.order_id == "OF12345"
    assert result.status == "ACCEPTED"
    assert result.filled_quantity == pytest.approx(0.01)
    assert result.avg_price == pytest.approx(30_000.0)

    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["apikey"] == "kraken-futures-key"
    body_bytes = captured_request.data or b""
    assert b"\"cliOrdId\":\"cli-kraken\"" in body_bytes
    assert b"\"limitPrice\":\"30000.00\"" in body_bytes
    expected_signature = _expected_signature(
        path="/orders",
        body=body_bytes,
        secret=_credentials().secret or "",
        nonce=headers["nonce"],
    )
    assert headers["authent"] == expected_signature


def test_cancel_order_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 20):  # type: ignore[override]
        payload = {"result": "success", "cancelStatus": {"status": "cancelled"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.futures.urlopen", fake_urlopen)
    adapter = KrakenFuturesAdapter(_credentials(), environment=Environment.LIVE)

    adapter.configure_network()
    adapter.cancel_order("OID-1")


class _RecordingWatchdog(Watchdog):
    def __init__(self) -> None:
        super().__init__(retry_policy=RetryPolicy(max_attempts=1), circuit_breaker=CircuitBreaker())
        self.operations: list[str] = []

    def execute(self, operation: str, func):  # type: ignore[override]
        self.operations.append(operation)
        return func()


def test_watchdog_is_used_for_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    watchdog = _RecordingWatchdog()
    adapter = KrakenFuturesAdapter(_credentials(), environment=Environment.LIVE, watchdog=watchdog)
    adapter.configure_network()
    def fake_private(context):
        if context.path == "/accounts":
            return {"accounts": {"futures": {}}}
        if context.path == "/orders" and context.method == "POST":
            return {"sendStatus": {"status": "accepted", "orderEvents": []}}
        if context.path.startswith("/orders/") and context.method == "DELETE":
            return {"cancelStatus": {"status": "cancelled"}}
        return {}

    def fake_public(path, params=None):
        if path == "/instruments":
            return {"result": "success", "instruments": []}
        if path == "/ohlc":
            return {"result": "success", "series": []}
        return {}

    monkeypatch.setattr(adapter, "_private_request", fake_private)
    monkeypatch.setattr(adapter, "_public_request", fake_public)

    adapter.fetch_account_snapshot()
    adapter.fetch_symbols()
    adapter.fetch_ohlcv("pi_xbtusd", "1m")
    adapter.place_order(OrderRequest(symbol="pi_xbtusd", side="buy", quantity=0.1, order_type="market"))
    adapter.cancel_order("OID-2")

    assert "kraken_futures_private_request" in watchdog.operations
    assert "kraken_futures_fetch_symbols" in watchdog.operations
    assert "kraken_futures_fetch_ohlcv" in watchdog.operations


def test_public_request_rejects_absolute_path() -> None:
    adapter = KrakenFuturesAdapter(_credentials(), environment=Environment.LIVE)
    adapter.configure_network()

    with pytest.raises(ValueError):
        adapter._public_request("https://attacker.invalid/instruments", params={})


def test_request_context_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError):
        _RequestContext(path="/orders", method="TRACE", params={})


def test_request_context_rejects_absolute_path() -> None:
    with pytest.raises(ValueError):
        _RequestContext(path="https://demo.invalid/orders", method="GET", params={})


def _expected_signature(*, path: str, body: bytes, secret: str, nonce: str) -> str:
    decoded_secret = base64.b64decode(secret)
    message = nonce.encode("utf-8") + path.encode("utf-8") + b"" + body
    sha_digest = hashlib.sha256(message).digest()
    mac = hmac.new(decoded_secret, sha_digest, hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("utf-8")
