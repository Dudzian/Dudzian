import gzip
import json
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from types import SimpleNamespace

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.errors import ExchangeAPIError
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter


@dataclass
class _FakeResponse:
    payload: bytes
    headers_map: dict[str, str] | None = None

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    @property
    def headers(self) -> dict[str, str]:  # noqa: D401
        return dict(self.headers_map or {})

    def read(self) -> bytes:  # noqa: D401
        return self.payload


def _build_stream_settings() -> dict[str, Any]:
    return {
        "base_url": "http://127.0.0.1:9876",
        "public_path": "/binance/public",
        "poll_interval": 0.0,
        "timeout": 0.1,
        "max_retries": 2,
        "backoff_base": 0.0,
        "backoff_cap": 0.0,
        "jitter": (0.0, 0.0),
    }


def test_binance_spot_stream_long_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 100.0, "symbol": "BTCUSDT"}],
                    "cursor": "cursor-1",
                }
            ],
            "retry_after": 0.0,
        },
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [],
                    "cursor": "cursor-1",
                    "heartbeat": True,
                }
            ],
            "retry_after": 0.0,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        if not responses:
            raise AssertionError("Żądano większej liczby pollingów niż oczekiwano")
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(key_id="test", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": _build_stream_settings()},
    )

    stream = adapter.stream_public_data(channels=["ticker"])
    first = next(stream)
    assert isinstance(first, StreamBatch)
    assert first.events and first.events[0]["price"] == 100.0

    second = next(stream)
    assert second.heartbeat is True
    assert list(second.events) == []

    stream.close()

    assert len(captured_urls) == 2
    query_first = parse_qs(urlparse(captured_urls[0]).query)
    assert query_first.get("channels") == ["ticker"]
    query_second = parse_qs(urlparse(captured_urls[1]).query)
    assert query_second.get("cursor") == ["cursor-1"]


def test_zonda_private_stream_requires_permissions() -> None:
    credentials = ExchangeCredentials(key_id="demo", environment=Environment.PAPER, permissions=())
    adapter = ZondaSpotAdapter(credentials, environment=Environment.PAPER)

    with pytest.raises(PermissionError):
        adapter.stream_private_data(channels=["orders"])


def test_stream_retries_after_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def flaky_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("temporary failure")
        payload = json.dumps(
            {
                "batches": [
                    {
                        "channel": "ticker",
                        "events": [{"price": 42.0}],
                        "cursor": "after-retry",
                    }
                ],
                "retry_after": 0.0,
            }
        ).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", flaky_urlopen)

    credentials = ExchangeCredentials(key_id="retry", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                **_build_stream_settings(),
                "max_retries": 3,
                "poll_interval": 0.0,
            }
        },
    )

    stream = adapter.stream_public_data(channels=["ticker"])
    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 42.0
    assert attempts == 2
    stream.close()


def test_kraken_stream_applies_scope_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 123.0, "pair": "XBT/USD"}],
                    "cursor": "kraken-1",
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(
        key_id="kraken",
        secret="dummy",
        permissions=("read",),
        environment=Environment.PAPER,
    )
    adapter = KrakenSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                **_build_stream_settings(),
                "public_path": "/kraken/public",
                "public_params": {"pair": "XBT/USD"},
            }
        },
    )

    stream = adapter.stream_public_data(channels=["ticker", "depth"])
    batch = next(stream)
    assert batch.events and batch.events[0]["pair"] == "XBT/USD"

    query = parse_qs(urlparse(captured_urls[0]).query)
    assert query.get("channels") == ["ticker,depth"]
    assert query.get("pair") == ["XBT/USD"]

    stream.close()


def test_stream_custom_channel_and_cursor_names(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    responses = [
        {
            "position": "cursor-1",
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 11.0}],
                    "position": "cursor-1",
                }
            ],
            "retry_after": 0.0,
        },
        {
            "position": "cursor-2",
            "batches": [
                {
                    "channel": "ticker",
                    "events": [],
                    "position": "cursor-2",
                    "heartbeat": True,
                }
            ],
            "retry_after": 0.0,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(key_id="custom", permissions=("read",), environment=Environment.PAPER)
    stream_settings = {
        **_build_stream_settings(),
        "channel_param": "topics",
        "cursor_param": "position",
        "initial_cursor": "initial-0",
        "channel_separator": "|",
    }
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": stream_settings},
    )

    stream = adapter.stream_public_data(channels=["ticker", "depth"])

    first = next(stream)
    assert first.events and first.events[0]["price"] == 11.0
    assert first.cursor == "cursor-1"

    second = next(stream)
    assert second.heartbeat is True
    assert second.cursor == "cursor-2"

    topics_first = parse_qs(urlparse(captured_urls[0]).query)
    assert topics_first.get("topics") == ["ticker|depth"]
    assert topics_first.get("position") == ["initial-0"]

    topics_second = parse_qs(urlparse(captured_urls[1]).query)
    assert topics_second.get("position") == ["cursor-1"]

    stream.close()


def test_stream_channel_serializer_supports_mappings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 33.0}],
                    "cursor": "cursor-map",
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(key_id="mapping", permissions=("read",), environment=Environment.PAPER)
    stream_settings = {
        **_build_stream_settings(),
        "channel_serializer": lambda values: {"topic": list(values)},
    }
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": stream_settings},
    )

    stream = adapter.stream_public_data(channels=["ticker", "depth"])
    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 33.0

    query = parse_qs(urlparse(captured_urls[0]).query)
    assert query.get("topic") == ["ticker", "depth"]
    assert "channels" not in query

    stream.close()


def test_binance_futures_private_stream_includes_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    responses = [
        {
            "batches": [
                {
                    "channel": "userData",
                    "events": [{"event": "ACCOUNT_UPDATE"}],
                    "cursor": "futures-1",
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(
        key_id="futures",
        secret="secret",
        permissions=("trade",),
        environment=Environment.PAPER,
    )
    adapter = BinanceFuturesAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                **_build_stream_settings(),
                "private_path": "/futures/private",
                "private_token": "listen-key-1",
            }
        },
    )

    stream = adapter.stream_private_data(channels=["userData"])
    batch = next(stream)
    assert batch.events and batch.events[0]["event"] == "ACCOUNT_UPDATE"

    query = parse_qs(urlparse(captured_urls[0]).query)
    assert query.get("token") == ["listen-key-1"]

    stream.close()


def test_local_long_poll_stream_context_manager_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 77.0}],
                    "cursor": "ctx-1",
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        if not responses:
            raise AssertionError("Oczekiwano tylko jednego zapytania long-pollowego")
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9001",
        path="/ctx/test",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    with stream as ctx:
        batch = next(ctx)
        assert batch.events and batch.events[0]["price"] == 77.0
        assert ctx.closed is False

    assert stream.closed is True
    with pytest.raises(StopIteration):
        next(stream)


def test_local_long_poll_stream_sets_accept_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers: list[str] = []

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_headers.append(request.get_header("Accept-encoding"))
        payload = json.dumps(
            {
                "batches": [
                    {"channel": "ticker", "events": [{"price": 42.0}]},
                ],
            }
        ).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9002",
        path="/headers",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 42.0
    from bot_core.exchanges import streaming as streaming_module

    expected_header = streaming_module._build_default_headers()["Accept-Encoding"]
    assert captured_headers == [expected_header]

    stream.close()


def test_local_long_poll_stream_decompresses_gzip(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 11.0}]},
            ]
        }
    ).encode("utf-8")
    compressed = gzip.compress(payload)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(compressed, {"Content-Encoding": "gzip"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9003",
        path="/gzip",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 11.0

    stream.close()


def test_local_long_poll_stream_decompresses_deflate(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 77.5}]},
            ]
        }
    ).encode("utf-8")
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(payload) + compressor.flush()

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(compressed, {"Content-Encoding": "deflate"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9004",
        path="/deflate",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 77.5

    stream.close()


def test_local_long_poll_stream_sends_accept_encoding_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_header: list[str | None] = []

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_header.append(request.get_header("Accept-encoding"))
        payload = json.dumps({"batches": [], "retry_after": 0.0}).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9006",
        path="/accept",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    next(stream)
    stream.close()

    from bot_core.exchanges import streaming as streaming_module

    expected_header = streaming_module._build_default_headers()["Accept-Encoding"]
    assert captured_header == [expected_header]


@pytest.mark.parametrize("brotli_variant", ["brotli", "brotlicffi", None])
def test_local_long_poll_stream_handles_brotli_encoding(
    monkeypatch: pytest.MonkeyPatch, brotli_variant: str | None
) -> None:
    from bot_core.exchanges import streaming as streaming_module

    payload = json.dumps(
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 88.0}],
                }
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded_payload = b"brotli-encoded"
    captured_header: list[str | None] = []

    if brotli_variant:
        decompress_calls: list[bytes] = []

        def fake_decompress(data: bytes) -> bytes:  # noqa: D401
            decompress_calls.append(data)
            assert data == encoded_payload
            return payload

        fake_brotli = SimpleNamespace(decompress=fake_decompress)
        error_attribute = "error" if brotli_variant == "brotli" else "BrotliError"
        setattr(fake_brotli, error_attribute, RuntimeError)
        monkeypatch.setattr(streaming_module, "brotli", fake_brotli, raising=False)
    else:
        monkeypatch.setattr(streaming_module, "brotli", None, raising=False)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_header.append(request.get_header("Accept-encoding"))
        body = encoded_payload if brotli_variant else payload
        return _FakeResponse(body, {"Content-Encoding": "br"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9005",
        path="/brotli",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 88.0

    stream.close()

    expected_header = streaming_module._build_default_headers()["Accept-Encoding"]
    assert captured_header == [expected_header]

    if brotli_variant:
        assert decompress_calls == [encoded_payload]


@pytest.mark.parametrize("zstd_variant", ["module", "decompressor", None])
def test_local_long_poll_stream_handles_zstd_encoding(
    monkeypatch: pytest.MonkeyPatch, zstd_variant: str | None
) -> None:
    from bot_core.exchanges import streaming as streaming_module

    payload = json.dumps(
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 93.0}],
                }
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded_payload = b"zstd" + payload
    captured_header: list[str | None] = []

    if zstd_variant:
        decompress_calls: list[bytes] = []

        if zstd_variant == "module":
            def fake_decompress(data: bytes) -> bytes:  # noqa: D401
                decompress_calls.append(data)
                assert data == encoded_payload
                return payload

            fake_zstd = SimpleNamespace(decompress=fake_decompress, ZstdError=RuntimeError)
        else:
            class FakeZstdDecompressor:
                def decompress(self, data: bytes) -> bytes:  # noqa: D401
                    decompress_calls.append(data)
                    assert data == encoded_payload
                    return payload

            fake_zstd = SimpleNamespace(ZstdDecompressor=FakeZstdDecompressor, ZstdError=RuntimeError)

        monkeypatch.setattr(streaming_module, "zstandard", fake_zstd, raising=False)
    else:
        monkeypatch.setattr(streaming_module, "zstandard", None, raising=False)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_header.append(request.get_header("Accept-encoding"))
        body = encoded_payload if zstd_variant else payload
        return _FakeResponse(body, {"Content-Encoding": "zstd"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9006",
        path="/zstd",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 93.0

    stream.close()

    expected_header = streaming_module._build_default_headers()["Accept-Encoding"]
    assert captured_header == [expected_header]

    if zstd_variant:
        assert decompress_calls == [encoded_payload]


def test_local_long_poll_stream_ignores_encoding_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 12.5}],
                }
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    gzip_payload = gzip.compress(payload)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(gzip_payload, {"Content-Encoding": "gzip; q=1.0"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9999",
        path="/params",
        channels=["ticker"],
        adapter="binance",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 12.5

    stream.close()


def test_local_long_poll_stream_reads_lowercase_content_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 99.0}],
                }
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded = gzip.compress(payload)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(encoded, {"content-encoding": ["GZIP"]})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9007",
        path="/gzip",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 99.0

    stream.close()


def test_local_long_poll_stream_multiple_encodings(monkeypatch: pytest.MonkeyPatch) -> None:
    brotli = pytest.importorskip("brotli")

    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 101.0}]},
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    brotli_compressed = brotli.compress(payload)
    gzip_then_brotli = gzip.compress(brotli_compressed)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(
            gzip_then_brotli,
            {"Content-Encoding": "br; q=0.8, gzip; level=1"},
        )

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9007",
        path="/multi",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 101.0

    stream.close()


def test_local_long_poll_stream_sequence_content_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    brotli = pytest.importorskip("brotli")

    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 77.0}]},
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded = gzip.compress(brotli.compress(payload))

    header_value = ("x-brotli; q=1.0", "X-GZIP; level=1")

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(encoded, {"Content-Encoding": header_value})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9011",
        path="/sequence",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 77.0

    stream.close()


def test_local_long_poll_stream_bytes_content_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 55.0}]},
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded = gzip.compress(payload)

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(encoded, {"Content-Encoding": b"gzip"})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9013",
        path="/bytes",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 55.0

    stream.close()


def test_local_long_poll_stream_iterable_content_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "batches": [
                {"channel": "ticker", "events": [{"price": 61.0}]},
            ],
            "retry_after": 0.0,
        }
    ).encode("utf-8")
    encoded = gzip.compress(payload)

    header_iterable = (value for value in ("identity", "gzip"))

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        return _FakeResponse(encoded, {"Content-Encoding": header_iterable})

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9014",
        path="/iterable",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 61.0

    stream.close()


def test_local_long_poll_stream_retry_after_header_skips_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"price": 105.0}],
                    "cursor": "after429",
                }
            ],
            "retry_after": 0.0,
        }
    ]
    attempts = {"count": 0}

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise HTTPError(
                request.full_url,  # type: ignore[attr-defined]
                429,
                "Too Many Requests",
                {"Retry-After": "1.5"},
                None,
            )
        attempts["count"] += 1
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    sleep_calls: list[float] = []

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9100",
        path="/retry/test",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=2,
        backoff_base=1.0,
        backoff_cap=1.0,
        jitter=(0.0, 0.0),
        sleep=sleep_calls.append,
        clock=lambda: 0.0,
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 105.0
    assert attempts["count"] == 2
    assert sleep_calls == [pytest.approx(1.5)]

    stream.close()


def test_local_long_poll_stream_retry_after_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    future = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    current = future - timedelta(seconds=42)

    monkeypatch.setattr("bot_core.exchanges.streaming.time.time", lambda: current.timestamp())

    value = LocalLongPollStream._retry_after({"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    assert value == pytest.approx(42.0)


def test_local_long_poll_stream_parses_string_retry_and_numeric_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": "PING",
                    "cursor": 5,
                }
            ],
            "retry_after": "0.2",
        },
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [],
                    "cursor": 6,
                    "heartbeat": True,
                }
            ],
            "poll_after": "0.3",
        },
    ]
    sleep_calls: list[float] = []

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9200",
        path="/retry/string",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        sleep=lambda value: sleep_calls.append(value),
        clock=lambda: 0.0,
    )

    first = next(stream)
    assert first.cursor == "5"
    assert first.events and first.events[0]["value"] == "PING"

    second = next(stream)
    assert second.cursor == "6"
    assert second.heartbeat is True
    assert not second.events

    assert sleep_calls == [pytest.approx(0.2), pytest.approx(0.3)]

    stream.close()


def test_local_long_poll_stream_raises_on_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        payload = json.dumps(
            {
                "error": {"code": "INVALID", "message": "Invalid channel"},
            }
        ).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9300",
        path="/error/test",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    with pytest.raises(ExchangeAPIError) as exc_info:
        next(stream)

    assert "Invalid channel" in str(exc_info.value)


def test_local_long_poll_stream_cursor_from_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "meta": {"cursor": "meta-1"},
            "events": [{"seq": 1}],
        },
        {
            "meta": {"next_cursor": "meta-2"},
            "events": [],
            "heartbeat": True,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9400",
        path="/meta/test",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    first = next(stream)
    assert first.cursor == "meta-1"
    assert first.events and first.events[0]["seq"] == 1

    second = next(stream)
    assert second.cursor == "meta-2"
    assert second.heartbeat is True

    stream.close()


def test_binance_private_stream_posts_channels_in_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    responses = [
        {
            "batches": [
                {
                    "channel": "orders",
                    "events": [],
                    "cursor": "seed-1",
                    "heartbeat": True,
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(
        key_id="binance", secret="secret", permissions=("read", "trade"), environment=Environment.PAPER
    )
    stream_settings = {
        **_build_stream_settings(),
        "method": "POST",
        "params_in_body": True,
        "channels_in_body": True,
        "cursor_in_body": True,
        "body_params": {"listenKey": "abc123"},
        "channel_serializer": lambda values: list(values),
        "private_params": {"account_type": "spot"},
        "private_initial_cursor": "seed-0",
        "private_body_params": {"token": "override-token"},
    }
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": stream_settings},
    )

    stream = adapter.stream_private_data(channels=["orders"])
    batch = next(stream)
    assert batch.heartbeat is True
    assert not batch.events

    assert captured["method"] == "POST"
    assert urlparse(captured["url"]).query == ""
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers.get("content-type") == "application/json"

    body = json.loads(captured["body"].decode("utf-8"))
    assert body["channels"] == ["orders"]
    assert body["cursor"] == "seed-0"
    assert body["listenKey"] == "abc123"
    assert body["token"] == "override-token"
    assert body["account_type"] == "spot"
    assert body["scope"] == "private"
    assert body["exchange"] == "binance_spot"
    assert body["environment"] == "paper"

    stream.close()


def test_local_long_poll_stream_form_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    responses = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"value": 1}],
                    "cursor": "form-1",
                }
            ],
            "retry_after": 0.0,
        }
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data
        payload = json.dumps(responses.pop(0)).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9500",
        path="/form/test",
        channels=["ticker"],
        adapter="binance_spot",
        scope="public",
        environment="paper",
        params={"version": "1"},
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        channel_serializer=lambda values: list(values),
        initial_cursor="form-0",
        http_method="POST",
        params_in_body=True,
        channels_in_body=True,
        cursor_in_body=True,
        body_params={"static": "value"},
        body_encoder="form",
    )

    batch = next(stream)
    assert batch.events and batch.events[0]["value"] == 1

    assert captured["method"] == "POST"
    assert urlparse(captured["url"]).query == ""
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers.get("content-type") == "application/x-www-form-urlencoded"

    parsed_body = parse_qs(captured["body"].decode("utf-8"))
    assert parsed_body["channels"] == ["ticker"]
    assert parsed_body["cursor"] == ["form-0"]
    assert parsed_body["version"] == ["1"]
    assert parsed_body["static"] == ["value"]
    assert parsed_body["exchange"] == ["binance_spot"]
    assert parsed_body["environment"] == ["paper"]
    assert parsed_body["scope"] == ["public"]

    stream.close()

