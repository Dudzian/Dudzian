from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import parse_qsl, urlparse

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter, _RequestContext as FuturesRequestContext
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter, _RequestContext as SpotRequestContext


@pytest.fixture()
def kraken_credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="test-key",
        secret=base64.b64encode(b"super-secret").decode("utf-8"),
        environment=Environment.PAPER,
        permissions=("trade", "read"),
    )


def test_spot_private_request_signature(monkeypatch: pytest.MonkeyPatch, kraken_credentials: ExchangeCredentials) -> None:
    adapter = KrakenSpotAdapter(credentials=kraken_credentials, environment=Environment.PAPER, settings={})
    adapter.configure_network(ip_allowlist=())

    captured: dict[str, object] = {}

    def _fake_perform(request_factory, *, endpoint: str, signed: bool):  # type: ignore[no-untyped-def]
        request = request_factory()
        captured["endpoint"] = endpoint
        captured["signed"] = signed
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data
        return {"error": []}

    monkeypatch.setattr(adapter, "_perform_request", _fake_perform)

    context = SpotRequestContext(path="/0/private/Balance", params={"asset": "XBT"})
    adapter._private_request(context)

    headers = {str(key).lower(): value for key, value in captured["headers"].items()}
    assert captured["signed"] is True
    assert headers["api-key"] == "test-key"

    body_bytes = captured["body"]
    assert isinstance(body_bytes, (bytes, bytearray))
    payload = dict(parse_qsl(body_bytes.decode("utf-8")))
    nonce = payload["nonce"]
    encoded_params = "asset=XBT"
    message = (nonce + encoded_params).encode("utf-8")
    sha_digest = hashlib.sha256(message).digest()
    decoded_secret = base64.b64decode(kraken_credentials.secret)
    expected_signature = base64.b64encode(
        hmac.new(decoded_secret, b"/0/private/Balance" + sha_digest, hashlib.sha512).digest()
    ).decode("utf-8")
    assert headers["api-sign"] == expected_signature


def test_futures_private_request_signature(
    monkeypatch: pytest.MonkeyPatch, kraken_credentials: ExchangeCredentials
) -> None:
    adapter = KrakenFuturesAdapter(credentials=kraken_credentials, environment=Environment.PAPER, settings={})
    adapter.configure_network(ip_allowlist=())

    captured: dict[str, object] = {}

    class _DummyResponse:
        def __init__(self) -> None:
            self._payload = b"{\"result\": \"success\", \"error\": []}"

        def __enter__(self) -> "_DummyResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - brak wyjątków w teście
            return None

        def read(self) -> bytes:
            return self._payload

    def _fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr("bot_core.exchanges.kraken.futures.urlopen", _fake_urlopen)

    context = FuturesRequestContext(
        path="/orders", method="POST", params={"symbol": "pi_xbtusd"}, body={"size": 1, "limitPrice": 31000.0}
    )
    adapter._private_request(context)

    headers = {str(key).lower(): value for key, value in captured["headers"].items()}  # type: ignore[assignment]
    assert headers["apikey"] == "test-key"
    nonce = headers["nonce"]

    parsed = urlparse(captured["url"])  # type: ignore[arg-type]
    path = parsed.path.replace("/derivatives/api/v3", "", 1)
    query = parsed.query
    query_fragment = f"?{query}" if query else ""
    body_bytes = captured["data"]
    assert isinstance(body_bytes, (bytes, bytearray))

    message = nonce.encode("utf-8") + path.encode("utf-8") + query_fragment.encode("utf-8") + body_bytes
    sha_digest = hashlib.sha256(message).digest()
    decoded_secret = base64.b64decode(kraken_credentials.secret)
    expected_signature = base64.b64encode(hmac.new(decoded_secret, sha_digest, hashlib.sha256).digest()).decode("utf-8")
    assert headers["authent"] == expected_signature
