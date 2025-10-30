from __future__ import annotations

import json
from collections import Counter
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import urlsplit

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance import BinanceSpotAdapter

from tests.integration.exchanges.helpers import make_order_request


def test_binance_spot_rate_limit_and_retry(monkeypatch, rate_limiter_registry):
    sleep_calls: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    responses = {
        "/api/v3/account": {
            "balances": [
                {"asset": "USDT", "free": "900", "locked": "100"},
            ]
        },
        "/api/v3/ticker/price": [
            {"symbol": "BTCUSDT", "price": "100"},
        ],
        "/api/v3/exchangeInfo": {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
            ]
        },
        "/api/v3/klines": [[1_700_000_000_000, 100, 110, 90, 105, 5]],
        "/api/v3/order": {
            "orderId": 123,
            "status": "FILLED",
            "executedQty": "0.1",
            "price": "105",
        },
    }
    order_attempts = {"count": 0}

    class FakeResponse:
        def __init__(self, payload, headers=None):
            self.payload = payload
            self.status = 200
            self.headers = headers or {"X-MBX-USED-WEIGHT-1M": "10"}

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=15):
        path = urlsplit(request.full_url).path
        if path == "/api/v3/order":
            order_attempts["count"] += 1
            if order_attempts["count"] == 1:
                payload = json.dumps({"code": -1003, "msg": "limit"}).encode("utf-8")
                raise HTTPError(request.full_url, 429, "Too Many Requests", {}, BytesIO(payload))
        payload = responses[path]
        return FakeResponse(payload)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        environment=Environment.TESTNET,
        permissions=("trade", "read"),
    )

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", fake_sleep)

    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        settings={"rate_limit_rules": ()},
    )

    adapter.fetch_symbols()
    adapter.fetch_account_snapshot()
    adapter.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    adapter.place_order(make_order_request())

    limiter_key = f"{adapter.name}:{adapter._environment.value}"
    limiter = rate_limiter_registry.created[limiter_key]

    counts = Counter(limiter.calls)
    assert counts[1.0] >= 3
    assert any(weight >= 5.0 for weight in limiter.calls)
    assert order_attempts["count"] == 2
    assert sleep_calls, "Powinno wystąpić opóźnienie między próbami"
