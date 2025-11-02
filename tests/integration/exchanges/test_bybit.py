from __future__ import annotations

from collections import Counter

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.bybit import BybitSpotAdapter
from bot_core.exchanges.rate_limiter import RateLimitRule

from tests.integration.exchanges.helpers import CCXTFakeClient, make_order_request


class FakeNetworkError(Exception):
    pass


def test_bybit_spot_rate_limit(monkeypatch, rate_limiter_registry):
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        environment=Environment.TESTNET,
        permissions=("trade", "read"),
    )
    client = CCXTFakeClient(exception=FakeNetworkError)
    adapter = BybitSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        settings={
            "network_error_types": (FakeNetworkError,),
            "sleep_callable": lambda delay: None,
            "rate_limit_rules": (RateLimitRule(rate=80, per=60.0),),
        },
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    adapter.fetch_symbols()
    adapter.fetch_account_snapshot()
    adapter.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    adapter.place_order(make_order_request())

    limiter_key = f"{adapter.name}:{adapter._exchange_id}:{adapter._environment.value}"  # type: ignore[attr-defined]
    limiter = rate_limiter_registry.created[limiter_key]

    assert Counter(limiter.calls)[1.0] >= 4
    assert client.create_attempts == 2
