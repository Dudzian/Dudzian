import json
import sys
from pathlib import Path
from hashlib import sha512
import hmac
from urllib.request import Request

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_account_snapshot_builds_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured
        captured = request
        payload = {"status": "Ok", "balances": []}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert captured is not None
    headers = {name.lower(): value for name, value in captured.header_items()}
    assert headers["api-key"] == "key"
    timestamp = headers["request-timestamp"]
    body = captured.data.decode("utf-8") if captured.data else ""
    expected_signature = hmac.new(
        b"secret",
        f"{timestamp}POST/trading/balance{body}".encode(),
        sha512,
    ).hexdigest()
    assert headers["api-hash"] == expected_signature


def test_fetch_ohlcv_maps_items(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(ExchangeCredentials(key_id="public", environment=Environment.LIVE))

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/candle/history/BTC-PLN/86400"
        assert params["from"] == 0
        assert params["to"] == 100
        return {
            "status": "Ok",
            "items": [
                {
                    "time": 1,
                    "open": "10",
                    "high": "12",
                    "low": "9",
                    "close": "11",
                    "volume": "5",
                }
            ],
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    rows = adapter.fetch_ohlcv("BTC-PLN", "1d", start=0, end=100_000, limit=None)

    assert rows == [[1000.0, 10.0, 12.0, 9.0, 11.0, 5.0]]


def test_place_order_uses_private_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] | None = None

    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        nonlocal captured_payload
        assert method == "POST"
        assert path == "/trading/offer"
        captured_payload = dict(data or {})
        return {"order": {"id": "123", "status": "new", "filledAmount": "0"}}

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)

    credentials = ExchangeCredentials(
        key_id="trade",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    request = OrderRequest(
        symbol="BTC-PLN",
        side="buy",
        quantity=1.5,
        order_type="limit",
        price=100.0,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(request)

    assert captured_payload == {
        "market": "BTC-PLN",
        "side": "buy",
        "type": "limit",
        "amount": "1.5",
        "price": "100.0",
        "timeInForce": "GTC",
        "clientOrderId": "cli-1",
    }
    assert result.order_id == "123"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.0)


def test_cancel_order_accepts_cancelled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        assert method == "DELETE"
        assert path == "/trading/order/XYZ"
        return {"order": {"id": "XYZ", "status": "cancelled"}}

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)

    credentials = ExchangeCredentials(
        key_id="trade",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    adapter.cancel_order("XYZ")
