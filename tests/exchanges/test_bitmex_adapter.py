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
    adapter = BitmexSpotAdapter(
        _credentials(), environment=Environment.TESTNET, client=_StubClient()
    )
    stream = adapter.stream_public_data(channels=["orderbook"])
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url == "https://stream.sandbox.dudzian.ai/exchanges"  # noqa: SLF001
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
    def __init__(self, payload: object, *, status: int | None = 400) -> None:
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


def test_bitmex_futures_maps_plain_text_rate_limit_without_status() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("rate limit exceeded", status=0),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 429


def test_bitmex_futures_maps_non_json_string_auth_without_status() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("permission denied by upstream", status=0),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAuthError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 401


def test_bitmex_futures_keeps_api_error_for_missing_payload_without_status() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient(None, status=0),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 500


def test_bitmex_futures_maps_bytes_payload_rate_limit_without_status() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient(b"rate limit exceeded", status=0),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 429


def test_bitmex_futures_maps_json_array_string_payload() -> None:
    payload = '[{"message":"Too many requests"}]'
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient(payload, status=0),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 429


def test_bitmex_futures_neutral_plain_text_503_stays_api_error() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("upstream node returned nonsense", status=503),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert not isinstance(exc_info.value, (ExchangeAuthError, ExchangeThrottlingError))
    assert exc_info.value.status_code == 503


def test_bitmex_futures_keyword_like_plain_text_without_match_stays_api_error() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("ratelimiter dashboard offline", status=503),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert not isinstance(exc_info.value, (ExchangeAuthError, ExchangeThrottlingError))


def test_bitmex_futures_none_status_falls_back_without_type_error() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("rate limit exceeded", status=None),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 429


def test_bitmex_futures_auth_like_neutral_plain_text_stays_api_error() -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("author service metadata mismatch", status=503),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert not isinstance(exc_info.value, (ExchangeAuthError, ExchangeThrottlingError))
    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("status", [200, 503])
def test_bitmex_futures_plain_text_without_keywords_stays_api_error(status: int) -> None:
    adapter = BitmexFuturesAdapter(
        _credentials(Environment.TESTNET),
        environment=Environment.TESTNET,
        client=_ErrorClient("plain upstream glitch", status=status),
    )
    adapter.configure_network(ip_allowlist=())
    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert not isinstance(exc_info.value, (ExchangeAuthError, ExchangeThrottlingError))
    assert exc_info.value.status_code == status
