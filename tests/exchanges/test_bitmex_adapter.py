"""Testy adapterów BitMEX."""

from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.bitmex import BitmexFuturesAdapter, BitmexSpotAdapter
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.streaming import LocalLongPollStream


class _StubClient:
    def __init__(self) -> None:
        self.urls: dict[str, str] = {}

    def load_markets(self) -> dict[str, object]:  # pragma: no cover - używane tylko w testach
        return {}


def _credentials(environment: Environment = Environment.TESTNET) -> ExchangeCredentials:
    return ExchangeCredentials(key_id="key", secret="secret", environment=environment)


def test_bitmex_spot_stream_paths() -> None:
    adapter = BitmexSpotAdapter(_credentials(), environment=Environment.TESTNET, client=_StubClient())
    stream = adapter.stream_public_data(channels=["orderbook"])
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url == "http://127.0.0.1:8765"  # noqa: SLF001
    assert stream._path == "/stream/bitmex_spot/public"  # noqa: SLF001


def test_bitmex_private_requires_secret() -> None:
    creds = ExchangeCredentials(key_id="key", secret=None, environment=Environment.TESTNET)
    adapter = BitmexFuturesAdapter(creds, environment=Environment.TESTNET, client=_StubClient())
    with pytest.raises(PermissionError):
        adapter.stream_private_data(channels=["positions"])


def test_bitmex_futures_custom_stream_settings() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_StubClient(),
        settings={"stream": {"private_path": "stream/custom", "poll_interval": 0.75}},
    )
    stream = adapter.stream_private_data(channels=["fills"])
    assert isinstance(stream, LocalLongPollStream)
    assert stream._path == "/stream/custom"  # noqa: SLF001
    assert stream._poll_interval == 0.75  # noqa: SLF001


def test_bitmex_futures_stream_defaults_per_environment() -> None:
    testnet_adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_StubClient(),
    )
    testnet_stream = testnet_adapter.stream_public_data(channels=["orderbook"])
    assert isinstance(testnet_stream, LocalLongPollStream)
    assert testnet_stream._base_url == "https://stream.sandbox.dudzian.ai/exchanges"  # noqa: SLF001
    assert testnet_stream._path == "/bitmex/futures/public"  # noqa: SLF001

    live_adapter = BitmexFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_StubClient(),
    )
    live_stream = live_adapter.stream_public_data(channels=["orderbook"])
    assert isinstance(live_stream, LocalLongPollStream)
    assert live_stream._base_url == "https://stream.hyperion.dudzian.ai/exchanges"  # noqa: SLF001
    assert live_stream._path == "/bitmex/futures/public"  # noqa: SLF001


class _ErrorClient:
    def __init__(self, payload: str, *, status: int = 400) -> None:
        self.payload = payload
        self.status = status

    def fetch_balance(self):  # pragma: no cover - invoked via adapter
        raise ExchangeAPIError("bitmex error", status_code=self.status, payload=self.payload)


def test_bitmex_futures_maps_auth_errors() -> None:
    payload = '{"error": {"name": "InvalidApiKey", "message": "key disabled"}}'
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_ErrorClient(payload, status=403),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()


def test_bitmex_futures_maps_rate_limit_errors() -> None:
    payload = '{"error": {"name": "RateLimit", "message": "Too many requests"}}'
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient(payload, status=429),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError):
        adapter.fetch_account_snapshot()

