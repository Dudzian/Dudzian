import asyncio
import gzip
import json
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from types import SimpleNamespace

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeNetworkError
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter
from bot_core.exchanges.network_guard import NetworkAccessViolation
from bot_core.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
)


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


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._iterator = iter(values)
        self._lock = threading.Lock()
        self._last = 0.0

    def __call__(self) -> float:
        with self._lock:
            try:
                self._last = next(self._iterator)
            except StopIteration:
                pass
            return self._last


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
    fallback_response = {
        "batches": [
            {
                "channel": "ticker",
                "events": [],
                "cursor": "cursor-1",
                "heartbeat": True,
            }
        ],
        "retry_after": 1.0,
    }

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        if responses:
            payload_map = responses.pop(0)
        else:
            payload_map = fallback_response
        payload = json.dumps(payload_map).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(key_id="test", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": _build_stream_settings()},
    )

    adapter.configure_network()

    stream = adapter.stream_public_data(channels=["ticker"])
    first = next(stream)
    assert isinstance(first, StreamBatch)
    assert first.events and first.events[0]["price"] == 100.0

    second = next(stream)
    assert second.heartbeat is True
    assert list(second.events) == []

    stream.close()

    assert len(captured_urls) >= 2
    query_first = parse_qs(urlparse(captured_urls[0]).query)
    assert query_first.get("channels") == ["ticker"]
    query_second = parse_qs(urlparse(captured_urls[1]).query)
    assert query_second.get("cursor") == ["cursor-1"]


def test_zonda_private_stream_requires_permissions() -> None:
    credentials = ExchangeCredentials(key_id="demo", environment=Environment.PAPER, permissions=())
    adapter = ZondaSpotAdapter(credentials, environment=Environment.PAPER)
    adapter.configure_network()

    with pytest.raises(PermissionError):
        adapter.stream_private_data(channels=["orders"])


def test_binance_stream_requires_network_configuration() -> None:
    credentials = ExchangeCredentials(key_id="guard", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": _build_stream_settings()},
    )

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter.stream_public_data(channels=["ticker"])

    violation = excinfo.value.reason
    assert isinstance(violation, NetworkAccessViolation)
    assert violation.reason == "network_not_configured"


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

    adapter.configure_network()

    stream = adapter.stream_public_data(channels=["ticker"])
    batch = next(stream)
    assert batch.events and batch.events[0]["price"] == 42.0
    assert attempts == 2
    stream.close()


def test_local_long_poll_stream_backpressure_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    payload = {
        "batches": [
            {"channel": "ticker", "events": [{"seq": 1}]},
            {"channel": "ticker", "events": [{"seq": 2}]},
            {"channel": "ticker", "events": [{"seq": 3}]},
            {"channel": "ticker", "events": [{"seq": 4}]},
        ]
    }
    responses = [json.dumps(payload).encode("utf-8")]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        if not responses:
            raise AssertionError("Nieoczekiwany dodatkowy polling")
        return _FakeResponse(responses.pop(0))

    fake_clock = _SequenceClock([0.0, 0.0, 0.2, 0.2, 1.2, 1.4])

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/stream",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        clock=fake_clock,
        sleep=lambda _: None,
        buffer_size=2,
        metrics_registry=registry,
    )

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    first = next(stream)
    second = next(stream)
    stream.close()

    assert first.events and first.events[0]["seq"] == 3
    assert second.events and second.events[0]["seq"] == 4

    expected_labels = {"adapter": "test", "scope": "public", "environment": "test"}

    backpressure_metric = registry.get("bot_exchange_stream_backpressure_total")
    assert isinstance(backpressure_metric, CounterMetric)
    assert backpressure_metric.value(labels=expected_labels) == 2

    queue_metric = registry.get("bot_exchange_stream_pending_batches")
    assert isinstance(queue_metric, GaugeMetric)
    assert queue_metric.value(labels=expected_labels) == pytest.approx(0.0)

    latency_metric = registry.get("bot_exchange_stream_long_poll_latency_seconds")
    assert isinstance(latency_metric, HistogramMetric)
    latency_state = latency_metric.snapshot(labels=expected_labels)
    assert latency_state.count == 1
    assert latency_state.sum == pytest.approx(0.2, rel=1e-6)

    lag_metric = registry.get("bot_exchange_stream_delivery_lag_seconds")
    assert isinstance(lag_metric, HistogramMetric)
    lag_state = lag_metric.snapshot(labels=expected_labels)
    assert lag_state.count == 2
    assert lag_state.sum == pytest.approx(2.2, rel=1e-6)

    lag_gauge = registry.get("bot_exchange_stream_last_delivery_lag_seconds")
    assert isinstance(lag_gauge, GaugeMetric)
    assert lag_gauge.value(labels=expected_labels) >= 0.0


def test_local_long_poll_stream_reconnect_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    payload = {"batches": [{"channel": "ticker", "events": [{"seq": 1}], "cursor": "abc"}]}

    attempts = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("temporary-down")
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    fake_clock = _SequenceClock([0.0, 0.0, 0.4, 0.8, 1.2, 1.4, 1.6])

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/stream",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=2,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        clock=fake_clock,
        sleep=lambda _: None,
        metrics_registry=registry,
    )

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    batch = next(stream)
    assert batch.events and batch.events[0]["seq"] == 1
    stream.close()

    base_labels = {"adapter": "test", "scope": "public", "environment": "test"}

    reconnect_metric = registry.get("bot_exchange_stream_reconnects_total")
    assert isinstance(reconnect_metric, CounterMetric)
    attempt_value = reconnect_metric.value(
        labels={**base_labels, "status": "attempt", "reason": "network"}
    )
    success_value = reconnect_metric.value(
        labels={**base_labels, "status": "success", "reason": "network"}
    )
    assert attempt_value == pytest.approx(1.0)
    assert success_value == pytest.approx(1.0)

    latency_metric = registry.get("bot_exchange_stream_reconnect_duration_seconds")
    assert isinstance(latency_metric, HistogramMetric)
    latency_state = latency_metric.snapshot(
        labels={**base_labels, "status": "success", "reason": "network"}
    )
    assert latency_state.count == 1
    assert latency_state.sum > 0.0


def test_local_long_poll_stream_http_error_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    payload = {"batches": [{"channel": "ticker", "events": [{"seq": 7}], "cursor": "xyz"}]}

    attempts = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            url = getattr(request, "full_url", "http://127.0.0.1/stream")
            raise HTTPError(url, 502, "Bad Gateway", {"Retry-After": "0.1"}, None)
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    current = -0.1

    def fake_clock() -> float:
        nonlocal current
        current += 0.1
        return current

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/stream",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=3,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        clock=fake_clock,
        sleep=lambda _: None,
        metrics_registry=registry,
    )

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    batch = next(stream)
    assert batch.events and batch.events[0]["seq"] == 7
    stream.close()

    base_labels = {"adapter": "test", "scope": "public", "environment": "test"}
    error_labels = {**base_labels, "status_code": "502", "retryable": "true", "reason": "http_5xx"}

    http_errors_metric = registry.get("bot_exchange_stream_http_errors_total")
    assert isinstance(http_errors_metric, CounterMetric)
    assert http_errors_metric.value(labels=error_labels) == pytest.approx(1.0)

    http_latency_metric = registry.get("bot_exchange_stream_http_error_duration_seconds")
    assert isinstance(http_latency_metric, HistogramMetric)
    http_latency_state = http_latency_metric.snapshot(labels=error_labels)
    assert http_latency_state.count == 1
    assert http_latency_state.sum >= 0.0

    reconnect_gauge = registry.get("bot_exchange_stream_reconnect_in_progress")
    assert isinstance(reconnect_gauge, GaugeMetric)
    assert reconnect_gauge.value(labels=base_labels) == pytest.approx(0.0)


def test_local_long_poll_stream_prefetches_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {"batches": [{"channel": "ticker", "events": [{"seq": 1}]}], "retry_after": 0.0},
        {"batches": [{"channel": "ticker", "events": [{"seq": 2}]}], "retry_after": 0.0},
    ]
    call_lock = threading.Lock()
    call_index = 0
    second_request_started = threading.Event()
    release_second_response = threading.Event()

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_index
        with call_lock:
            index = call_index
            call_index += 1
        payload_map = responses[index if index < len(responses) else len(responses) - 1]
        if index == 1:
            second_request_started.set()
            if not release_second_response.wait(timeout=1.0):
                raise AssertionError("Drugie zapytanie long-pollowe nie zostało odblokowane na czas")
        payload = json.dumps(payload_map).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/background",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    first = next(stream)
    assert [event.get("seq") for event in first.events] == [1]

    assert second_request_started.wait(timeout=1.0) is True
    assert call_index >= 2

    release_second_response.set()
    second = next(stream)
    assert [event.get("seq") for event in second.events] == [2]

    stream.close()


def test_local_long_poll_stream_prefills_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {"batches": [{"channel": "ticker", "events": [{"seq": 1}]}], "retry_after": 0.0},
        {"batches": [{"channel": "ticker", "events": [{"seq": 2}]}], "retry_after": 0.0},
        {"batches": [{"channel": "ticker", "events": [{"seq": 3}]}], "retry_after": 0.0},
        {"batches": [], "retry_after": 1.0},
    ]
    call_lock = threading.Lock()
    call_index = 0
    call_events = [threading.Event() for _ in payloads]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_index
        with call_lock:
            index = call_index
            call_index += 1
        if index >= len(payloads):
            index = len(payloads) - 1
        call_events[index].set()
        payload = json.dumps(payloads[index]).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/prefill",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        buffer_size=2,
        clock=_SequenceClock([0.0, 0.1, 0.2, 0.3, 0.4]),
        sleep=lambda _: None,
    ).start()

    assert call_events[0].wait(timeout=1.0) is True
    assert call_events[1].is_set() is False

    first = next(stream)
    assert [event.get("seq") for event in first.events] == [1]

    assert call_events[1].wait(timeout=1.0) is True
    assert call_events[2].wait(timeout=1.0) is True
    second = next(stream)
    assert [event.get("seq") for event in second.events] == [2]

    third = next(stream)
    assert [event.get("seq") for event in third.events] == [3]

    stream.close()


def test_local_long_poll_stream_async_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"batches": [{"channel": "ticker", "events": [{"seq": 10}]}], "retry_after": 0.0},
        {"batches": [{"channel": "ticker", "events": [{"seq": 11}]}], "retry_after": 0.0},
        {"batches": [], "retry_after": 0.0},
    ]
    call_lock = threading.Lock()
    call_index = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_index
        with call_lock:
            if call_index < len(responses):
                payload_map = responses[call_index]
            else:
                payload_map = responses[-1]
            call_index += 1
        payload = json.dumps(payload_map).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1",
        path="/async",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="test",
        poll_interval=0.0,
        timeout=0.1,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    async def _consume() -> list[StreamBatch]:
        batches: list[StreamBatch] = []
        async for batch in stream:
            batches.append(batch)
            if len(batches) == 2:
                break
        return batches

    batches = asyncio.run(_consume())
    stream.close()

    assert len(batches) == 2
    assert [event.get("seq") for event in batches[0].events] == [10]
    assert [event.get("seq") for event in batches[1].events] == [11]


def test_local_long_poll_stream_async_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"batches": [{"channel": "ticker", "events": [{"seq": 21}]}], "retry_after": 0.0},
        {"batches": [{"channel": "ticker", "events": [{"seq": 22}]}], "retry_after": 0.0},
        {"batches": [], "retry_after": 1.0},
    ]
    call_lock = threading.Lock()
    call_index = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_index
        with call_lock:
            if call_index < len(responses):
                payload_map = responses[call_index]
            else:
                payload_map = responses[-1]
            call_index += 1
        payload = json.dumps(payload_map).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    async def _consume() -> list[list[int | None]]:
        async with LocalLongPollStream(
            base_url="http://127.0.0.1",
            path="/async-context",
            channels=["ticker"],
            adapter="test",
            scope="public",
            environment="test",
            poll_interval=0.0,
            timeout=0.1,
            max_retries=1,
            backoff_base=0.0,
            backoff_cap=0.0,
            jitter=(0.0, 0.0),
        ) as stream:
            batches: list[list[int | None]] = []
            async for batch in stream:
                batches.append([event.get("seq") for event in batch.events])
                if len(batches) >= 2:
                    break
            return batches

    batches = asyncio.run(_consume())

    assert batches == [[21], [22]]


def test_binance_stream_respects_scope_buffer_size() -> None:
    credentials = ExchangeCredentials(key_id="buffer", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                **_build_stream_settings(),
                "buffer_size": 19,
                "public_buffer_size": 3,
            }
        },
    )

    adapter.configure_network()

    stream = adapter.stream_public_data(channels=["ticker"])
    try:
        assert stream._buffer_size == 3
    finally:
        stream.close()


def test_binance_stream_uses_adapter_metrics_registry() -> None:
    registry = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="metrics", permissions=("read",), environment=Environment.PAPER)
    adapter = BinanceSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": _build_stream_settings()},
        metrics_registry=registry,
    )

    adapter.configure_network()

    stream = adapter.stream_public_data(channels=["ticker"])
    try:
        metric = registry.get("bot_exchange_stream_pending_batches")
        assert isinstance(metric, GaugeMetric)
        labels = {"adapter": adapter.name, "scope": "public", "environment": Environment.PAPER.value}
        assert metric.value(labels=labels) == pytest.approx(0.0)
    finally:
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

    adapter.configure_network()

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
    fallback_response = {
        "position": "cursor-2",
        "batches": [
            {
                "channel": "ticker",
                "events": [],
                "position": "cursor-2",
                "heartbeat": True,
            }
        ],
        "retry_after": 1.0,
    }

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        if responses:
            payload_map = responses.pop(0)
        else:
            payload_map = fallback_response
        payload = json.dumps(payload_map).encode("utf-8")
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

    adapter.configure_network()

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

    adapter.configure_network()

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

    adapter.configure_network()

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

    adapter.configure_network()

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


def test_local_long_poll_stream_wait_prefill_minimizes_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "batches": [
                {"channel": "ticker", "events": [{"sequence": 1}], "cursor": "c-1"},
            ],
            "retry_after": 0.0,
        },
        {
            "batches": [
                {"channel": "ticker", "events": [{"sequence": 2}], "cursor": "c-2"},
            ],
            "retry_after": 0.0,
        },
    ]
    poll_delays = [0.02, 0.0]
    call_count = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_count
        delay = poll_delays[call_count] if call_count < len(poll_delays) else 0.0
        if delay:
            time.sleep(delay)
        payload_map = responses[call_count] if call_count < len(responses) else responses[-1]
        call_count += 1
        return _FakeResponse(json.dumps(payload_map).encode("utf-8"))

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9100",
        path="/perf",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.2,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
        buffer_size=4,
    )

    assert stream.wait_prefill(timeout=1.0) is True

    start_first = time.perf_counter()
    first = next(stream)
    first_elapsed = time.perf_counter() - start_first
    assert first_elapsed < 0.02
    assert first.events and first.events[0]["sequence"] == 1

    assert stream.wait_prefill(timeout=1.0) is True

    start_second = time.perf_counter()
    second = next(stream)
    second_elapsed = time.perf_counter() - start_second
    assert second_elapsed < 0.02
    assert second.events and second.events[0]["sequence"] == 2

    stream.close()
    assert call_count >= 2


def test_local_long_poll_stream_wait_prefill_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        time.sleep(0.05)
        payload = json.dumps(
            {
                "batches": [
                    {"channel": "ticker", "events": [{"sequence": 1}], "cursor": "c-1"},
                ],
                "retry_after": 0.0,
            }
        ).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9101",
        path="/timeout",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.2,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    result = stream.wait_prefill(timeout=0.01)
    assert result is False

    stream.close()


def test_local_long_poll_stream_wait_prefill_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        raise HTTPError(request.full_url, 500, "err", hdrs=None, fp=None)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9102",
        path="/errors",
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

    with pytest.raises(ExchangeNetworkError):
        stream.wait_prefill(timeout=0.5)

    stream.close()


def test_local_long_poll_stream_wait_prefill_async(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "batches": [
                {"channel": "ticker", "events": [{"sequence": 41}], "cursor": "c-41"},
            ],
            "retry_after": 0.0,
        },
        {
            "batches": [
                {"channel": "ticker", "events": [{"sequence": 42}], "cursor": "c-42"},
            ],
            "retry_after": 0.0,
        },
    ]
    call_index = 0

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        nonlocal call_index
        if call_index < len(responses):
            payload_map = responses[call_index]
        else:
            payload_map = responses[-1]
        call_index += 1
        payload = json.dumps(payload_map).encode("utf-8")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9103",
        path="/prefill-async",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=0.2,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    async def _consume() -> list[int]:
        result = await stream.wait_prefill_async(timeout=1.0)
        assert result is True

        batches: list[StreamBatch] = []
        async for batch in stream:
            batches.append(batch)
            if len(batches) >= 2:
                break

        await stream.aclose()
        return [event.get("sequence") for batch in batches for event in batch.events]

    sequences = asyncio.run(_consume())
    assert sequences == [41, 42]


def test_local_long_poll_stream_wait_prefill_async_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        raise HTTPError(request.full_url, 500, "err", hdrs=None, fp=None)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9104",
        path="/prefill-async-error",
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

    async def _call() -> None:
        with pytest.raises(ExchangeNetworkError):
            await stream.wait_prefill_async(timeout=0.5)

    asyncio.run(_call())
    stream.close()

