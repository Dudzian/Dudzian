"""Testy adapterów Deribit opartych o CCXT."""

from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.deribit import DeribitFuturesAdapter, DeribitSpotAdapter
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.streaming import LocalLongPollStream


class _StubClient:
    def __init__(self) -> None:
        self.urls: dict[str, str] = {}

    def load_markets(self) -> dict[str, object]:  # pragma: no cover - test double
        return {}


def _credentials(environment: Environment = Environment.TESTNET) -> ExchangeCredentials:
    return ExchangeCredentials(key_id="key", secret="secret", environment=environment)


def test_deribit_spot_public_stream_defaults() -> None:
    adapter = DeribitSpotAdapter(_credentials(), environment=Environment.TESTNET, client=_StubClient())
    stream = adapter.stream_public_data(channels=["ticker", "trades"])
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url == "http://127.0.0.1:8765"  # noqa: SLF001 - dostęp testowy
    assert stream._path == "/stream/deribit_spot/public"  # noqa: SLF001


def test_deribit_spot_private_requires_secret() -> None:
    creds = ExchangeCredentials(key_id="only-key", secret=None, environment=Environment.LIVE)
    adapter = DeribitSpotAdapter(creds, environment=Environment.LIVE, client=_StubClient())
    with pytest.raises(PermissionError):
        adapter.stream_private_data(channels=["orders"])


def test_deribit_futures_honours_custom_stream_base() -> None:
    adapter = DeribitFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_StubClient(),
        settings={"stream": {"base_url": "http://192.0.2.10:9000"}},
    )
    stream = adapter.stream_private_data(channels=["fills"])
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url == "http://192.0.2.10:9000"  # noqa: SLF001
    assert stream._path == "/deribit/futures/private"  # noqa: SLF001


def test_deribit_futures_stream_defaults_per_environment() -> None:
    testnet_adapter = DeribitFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_StubClient(),
    )
    testnet_stream = testnet_adapter.stream_public_data(channels=["ticker"])
    assert isinstance(testnet_stream, LocalLongPollStream)
    assert testnet_stream._base_url == "https://stream.sandbox.dudzian.ai/exchanges"  # noqa: SLF001
    assert testnet_stream._path == "/deribit/futures/public"  # noqa: SLF001

    live_adapter = DeribitFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_StubClient(),
    )
    live_stream = live_adapter.stream_public_data(channels=["ticker"])
    assert isinstance(live_stream, LocalLongPollStream)
    assert live_stream._base_url == "https://stream.hyperion.dudzian.ai/exchanges"  # noqa: SLF001
    assert live_stream._path == "/deribit/futures/public"  # noqa: SLF001


class _ErrorClient:
    def __init__(self, payload: str, *, status: int = 400) -> None:
        self.payload = payload
        self.status = status

    def fetch_balance(self):  # pragma: no cover - invoked via adapter
        raise ExchangeAPIError("deribit error", status_code=self.status, payload=self.payload)


def test_deribit_futures_maps_auth_errors() -> None:
    payload = '{"error": {"code": 13002, "message": "authorization invalid"}}'
    adapter = DeribitFuturesAdapter(
        _credentials(Environment.LIVE),
        environment=Environment.LIVE,
        client=_ErrorClient(payload),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()


def test_deribit_futures_maps_rate_limit_errors() -> None:
    payload = '{"error": {"code": 10028, "message": "Too many requests"}}'
    adapter = DeribitFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient(payload, status=429),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError):
        adapter.fetch_account_snapshot()

