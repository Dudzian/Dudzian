from __future__ import annotations

import json
from collections import Counter
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import urlsplit

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.kraken import KrakenSpotAdapter

from tests.integration.exchanges.helpers import make_order_request


def test_kraken_spot_rate_limit_and_retry(monkeypatch, rate_limiter_registry):
    sleep_calls: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    responses = {
        "/0/private/Balance": {"result": {"ZUSD": "900"}},
        "/0/private/TradeBalance": {
            "result": {"eb": "1000", "mf": "900", "m": "50"}
        },
        "/0/public/AssetPairs": {
            "result": {"XBTUSDT": {"altname": "BTC/USDT"}}
        },
        "/0/public/OHLC": {
            "result": {"XBTUSDT": [[1_700_000_000, 100, 110, 90, 105, 5]]}
        },
        "/0/private/AddOrder": {"result": {"txid": ["order-123"]}},
    }
    order_attempts = {"count": 0}

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status = 200
            self.headers = {}

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=15):
        path = urlsplit(request.full_url).path
        if path == "/0/private/AddOrder":
            order_attempts["count"] += 1
            if order_attempts["count"] == 1:
                payload = json.dumps({"error": ["EOrder:Rate limit"]}).encode("utf-8")
                raise HTTPError(request.full_url, 429, "Too Many Requests", {}, BytesIO(payload))
        payload = responses[path]
        return FakeResponse(payload)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="c2VjcmV0",
        environment=Environment.TESTNET,
        permissions=("trade", "read"),
    )

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", fake_sleep)

    adapter = KrakenSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        settings={},
    )

    adapter.fetch_symbols()
    adapter.fetch_account_snapshot()
    adapter.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    adapter.place_order(make_order_request())

    limiter_key = f"{adapter.name}:{adapter._environment.value}"
    limiter = rate_limiter_registry.created[limiter_key]

    counts = Counter(limiter.calls)
    assert counts[1.0] >= 2
    assert any(weight >= 2.0 for weight in limiter.calls)
    assert order_attempts["count"] == 2
    assert sleep_calls, "Retry logic powinien uśpić w przypadku błędu"
