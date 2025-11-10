"""Testy adapterów BitMEX."""

from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.bitmex import BitmexFuturesAdapter, BitmexSpotAdapter
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

