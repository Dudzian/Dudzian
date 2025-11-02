from __future__ import annotations

from collections import Counter

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.coinbase import CoinbaseSpotAdapter
from bot_core.exchanges.rate_limiter import RateLimitRule

from tests.integration.exchanges.helpers import CCXTFakeClient, make_order_request


class FakeNetworkError(Exception):
    pass


def test_coinbase_spot_rate_limit_and_retry(monkeypatch, rate_limiter_registry):
    sleeps: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        environment=Environment.TESTNET,
        permissions=("trade", "read"),
    )
    client = CCXTFakeClient(exception=FakeNetworkError)
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        settings={
            "network_error_types": (FakeNetworkError,),
            "sleep_callable": fake_sleep,
            "rate_limit_rules": (RateLimitRule(rate=100, per=60.0),),
        },
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    symbols = adapter.fetch_symbols()
    snapshot = adapter.fetch_account_snapshot()
    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    order = adapter.place_order(make_order_request())

    limiter_key = f"{adapter.name}:{adapter._exchange_id}:{adapter._environment.value}"  # type: ignore[attr-defined]
    limiter = rate_limiter_registry.created[limiter_key]

    assert symbols == ("BTC/USDT",)
    assert snapshot.total_equity == 1000.0
    assert candles[0][4] == 105.0
    assert order.status == "closed"
    assert Counter(limiter.calls)[1.0] >= 4
    assert client.create_attempts == 2
    assert sleeps, "Retry logic should invoke backoff sleep"
