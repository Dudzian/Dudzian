from __future__ import annotations

import json
import time

import pytest

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import httpx

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter, NowaGieldaStreamClient, symbols
from bot_core.exchanges.streaming import StreamBatch

_BASE_URL = "https://paper.nowa-gielda.example"

respx = pytest.importorskip("respx")


@pytest.fixture
def api_mock() -> "respx.Router":
    with respx.mock(base_url=_BASE_URL) as router:
        yield router


def _build_adapter() -> NowaGieldaSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        environment=Environment.PAPER,
    )
    return NowaGieldaSpotAdapter(credentials)


def _build_stream_adapter(
    stream_settings: Mapping[str, Any],
    *,
    permissions: Sequence[str] = ("read", "trade"),
) -> NowaGieldaSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="stream-key",
        secret="stream-secret",
        environment=Environment.PAPER,
        permissions=permissions,
    )
    return NowaGieldaSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        settings={"stream": dict(stream_settings)},
    )


def test_symbol_mapping_roundtrip() -> None:
    assert symbols.to_exchange_symbol("BTC_USDT") == "BTC-USDT"
    assert symbols.to_internal_symbol("BTC-USDT") == "BTC_USDT"
    supported = tuple(symbols.supported_internal_symbols())
    assert "ETH_USDT" in supported


def test_sign_request_is_deterministic() -> None:
    adapter = _build_adapter()
    payload = {
        "symbol": "BTC-USDT",
        "type": "limit",
        "quantity": 1,
        "price": 25_000,
    }
    signature = adapter.sign_request(
        1_700_000_000_000,
        "POST",
        "/private/orders",
        body=payload,
    )
    assert signature == "7263373ca065b01dc73517607c92ca6505e420be70df6482221e68c539f9ab9d"


def test_sign_request_includes_query_params() -> None:
    adapter = _build_adapter()
    timestamp = 1_700_000_000_000
    base_signature = adapter.sign_request(
        timestamp,
        "DELETE",
        "/private/orders",
        params={"orderId": "sim-1"},
    )
    other_signature = adapter.sign_request(
        timestamp,
        "DELETE",
        "/private/orders",
        params={"orderId": "sim-2"},
    )

    assert base_signature != other_signature


def test_rate_limit_rules() -> None:
    adapter = _build_adapter()

    trading_rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert trading_rule is not None
    assert trading_rule.weight == 5
    assert trading_rule.max_requests == 5

    account_rule = adapter.rate_limit_rule("GET", "/private/account")
    assert account_rule is not None
    assert account_rule.weight == 2

    ticker_rule = adapter.rate_limit_rule("GET", "/public/ticker")
    assert ticker_rule is not None
    assert ticker_rule.weight == 1

    ohlcv_rule = adapter.rate_limit_rule("GET", "/public/ohlcv")
    assert ohlcv_rule is not None
    assert ohlcv_rule.weight == 2

    trades_rule = adapter.rate_limit_rule("GET", "/private/trades")
    assert trades_rule is not None
    assert trades_rule.weight == 3

    open_orders_rule = adapter.rate_limit_rule("GET", "/private/orders")
    assert open_orders_rule is not None
    assert open_orders_rule.weight == 3

    closed_orders_rule = adapter.rate_limit_rule("GET", "/private/orders/history")
    assert closed_orders_rule is not None
    assert closed_orders_rule.weight == 3

    assert adapter.request_weight("GET", "/non-existent") == 1


class _FakeStreamResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    @property
    def headers(self) -> Mapping[str, str]:  # noqa: D401
        return {}

    def read(self) -> bytes:  # noqa: D401
        return self._payload


def test_fetch_ticker_validates_symbol_translation(api_mock: "respx.Router") -> None:
    adapter = _build_adapter()
    route = api_mock.get("/public/ticker").mock(
        return_value=httpx.Response(
            200,
            json={
                "symbol": "BTC-USDT",
                "bestBid": "50000.5",
                "bestAsk": "50010.5",
                "lastPrice": "50005.1",
                "timestamp": 1_700_000_000_000,
            },
        )
    )

    ticker = adapter.fetch_ticker("BTC_USDT")

    assert ticker == {
        "symbol": "BTC_USDT",
        "best_bid": 50000.5,
        "best_ask": 50010.5,
        "last_price": 50005.1,
        "timestamp": 1_700_000_000_000.0,
    }


def test_fetch_ticker_rejects_symbol_mismatch(api_mock: "respx.Router") -> None:
    adapter = _build_adapter()
    api_mock.get("/public/ticker").mock(
        return_value=httpx.Response(
            200,
            json={
                "symbol": "ETH-USDT",
                "bestBid": "1800",
                "bestAsk": "1800.5",
                "lastPrice": "1800.25",
            },
        )
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_ticker("BTC_USDT")


def test_fetch_orderbook_translates_symbol(api_mock: "respx.Router") -> None:
    adapter = _build_adapter()
    api_mock.get("/public/orderbook").mock(
        return_value=httpx.Response(
            200,
            json={
                "symbol": "BTC-USDT",
                "bids": [["50000", "1"]],
                "asks": [["50010", "2"]],
            },
        )
    )

    orderbook = adapter.fetch_orderbook("BTC_USDT")

    assert orderbook["bids"][0][0] == "50000"
    assert orderbook["asks"][0][0] == "50010"


def test_place_order_sends_signed_payload(api_mock: "respx.Router") -> None:
    adapter = _build_adapter()
    route = api_mock.post("/private/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "orderId": "sim-1",
                "status": "NEW",
                "filledQuantity": "0",
                "avgPrice": None,
            },
        )
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="limit",
        price=25_000.0,
    )

    result = adapter.place_order(request)

    assert result.order_id == "sim-1"
    assert result.status == "NEW"
    assert result.filled_quantity == 0.0
    assert result.avg_price is None

    call = route.calls.last
    body = json.loads(call.request.content)
    assert body["symbol"] == "BTC-USDT"
    assert call.request.headers["X-API-KEY"] == "test-key"


def test_place_order_maps_auth_errors(api_mock: "respx.Router") -> None:
    adapter = _build_adapter()
    api_mock.post("/private/orders").mock(
        return_value=httpx.Response(
            401,
            json={"code": "INVALID_SIGNATURE", "message": "signature error"},
        )
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="limit",
    )

    with pytest.raises(ExchangeAuthError):
        adapter.place_order(request)


def test_place_market_order_strips_null_fields(
    api_mock: "respx.Router", monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_000_123_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)
    route = api_mock.post("/private/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "orderId": "sim-2",
                "status": "FILLED",
                "filledQuantity": "1",
                "avgPrice": "25010.0",
            },
        )
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="market",
    )

    adapter.place_order(request)

    call = route.calls.last
    body = json.loads(call.request.content)
    assert "price" not in body
    assert body == {
        "symbol": "BTC-USDT",
        "side": "buy",
        "type": "market",
        "quantity": 1.0,
    }
    expected_signature = adapter.sign_request(
        fixed_ts,
        "POST",
        "/private/orders",
        body=body,
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


def test_rate_limiter_blocks_excessive_requests() -> None:
    adapter = _build_adapter()
    rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert rule is not None

    # Wypełnij licznik limitem, a następnie spróbuj złożyć dodatkowe zlecenie.
    client = adapter._http_client  # type: ignore[attr-defined]
    client.rate_limiter.consume("POST", "/private/orders")

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=0.1,
        order_type="market",
    )

    with pytest.raises(ExchangeThrottlingError):
        adapter.place_order(request)


def test_cancel_order_translates_symbol_and_headers(
    api_mock: "respx.Router", monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_000_000_000)
    route = api_mock.delete("/private/orders").mock(
        return_value=httpx.Response(200, json={"status": "CANCELLED"})
    )

    adapter.cancel_order("sim-1", symbol="BTC_USDT")

    call = route.calls.last
    assert call.request.headers["X-API-KEY"] == "test-key"
    assert (
        call.request.headers["X-API-SIGN"]
        == "a629ae5ede954c14a17990e03653c5b2a8fb30c19ef2ffece6ff26d722be22b2"
    )


def test_rate_limiter_window_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    client = adapter._http_client  # type: ignore[attr-defined]
    rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert rule is not None

    client.rate_limiter.consume("POST", "/private/orders")

    fake_now = time.monotonic() + rule.window_seconds + 0.1

    monkeypatch.setattr(time, "monotonic", lambda: fake_now)

    # Po upłynięciu okna limit powinien zostać zresetowany.
    client.rate_limiter.consume("POST", "/private/orders")


def test_stream_reconnect_attempt_counter_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    stream_settings = {
        "base_url": "http://127.0.0.1:9876",
        "public_path": "/nowa/public",
        "poll_interval": 0.0,
        "timeout": 0.1,
        "reconnect_attempts": 1,
        "backoff_base": 0.0,
        "backoff_cap": 0.0,
        "jitter": (0.0, 0.0),
    }
    adapter = _build_stream_adapter(stream_settings)

    responses_queue: list[Any] = [
        URLError("temporary outage"),
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"symbol": "BTC-USDT", "last_price": 101.0}],
                    "cursor": "cursor-1",
                }
            ],
            "retry_after": 0.0,
        },
        URLError("another outage"),
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"symbol": "BTC-USDT", "last_price": 102.0}],
                    "cursor": "cursor-2",
                }
            ],
            "retry_after": 0.0,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        if not responses_queue:
            raise AssertionError("Zabrakło zaplanowanych odpowiedzi streamu")
        payload = responses_queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeStreamResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = adapter.stream_public_data(channels=["ticker"])

    first = next(stream)
    assert first.cursor == "cursor-1"
    assert first.events[0]["last_price"] == 101.0

    replayed = next(stream)
    assert replayed.cursor == "cursor-1"

    recovered = next(stream)
    assert recovered.cursor == "cursor-2"
    assert recovered.events[0]["last_price"] == 102.0

    stream.close()

    assert not responses_queue


def test_stream_client_closes_after_exhausting_reconnects() -> None:
    class _FailingStream:
        def __init__(self, error: Exception) -> None:
            self._error = error
            self.closed = False

        def __iter__(self) -> "_FailingStream":
            return self

        def __next__(self) -> StreamBatch:
            raise self._error

        def close(self) -> None:
            self.closed = True

    errors = [
        ExchangeNetworkError("pierwsze zerwanie"),
        ExchangeNetworkError("drugie zerwanie"),
    ]
    created_streams: list[_FailingStream] = []

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _FailingStream:
        assert mapped_channels == ("ticker",)
        if not errors:
            raise AssertionError("Oczekiwano kolejnych błędów streamu")
        stream = _FailingStream(errors.pop(0))
        created_streams.append(stream)
        return stream

    client = NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
        channel_mapping={},
        max_reconnects=1,
    )

    with pytest.raises(ExchangeNetworkError):
        next(client)

    assert client.closed
    assert all(stream.closed for stream in created_streams)


def test_stream_client_context_manager_closes() -> None:
    class _StaticStream:
        def __init__(self) -> None:
            self.closed = False
            self._yielded = False

        def __iter__(self) -> "_StaticStream":
            return self

        def __next__(self) -> StreamBatch:
            if self._yielded:
                raise StopIteration
            self._yielded = True
            return StreamBatch(
                channel="ticker",
                events=({"symbol": "BTC-USDT", "price": 100.0},),
                received_at=1.0,
                cursor="cur-1",
            )

        def close(self) -> None:
            self.closed = True

    static_stream = _StaticStream()

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _StaticStream:
        assert cursor is None
        return static_stream

    with NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
    ) as client:
        batch = next(client)
        assert batch.cursor == "cur-1"
        assert not client.closed

    assert client.closed
    assert static_stream.closed


def test_stream_client_manual_history_replay() -> None:
    class _SequenceStream:
        def __init__(self, batches: Sequence[StreamBatch]) -> None:
            self._batches = list(batches)
            self.closed = False

        def __iter__(self) -> "_SequenceStream":
            return self

        def __next__(self) -> StreamBatch:
            if not self._batches:
                raise StopIteration
            return self._batches.pop(0)

        def close(self) -> None:
            self.closed = True

    batches = (
        StreamBatch(
            channel="ticker",
            events=({"symbol": "BTC-USDT", "price": 100.0},),
            received_at=1.0,
            cursor="cursor-1",
        ),
        StreamBatch(
            channel="ticker",
            events=(),
            received_at=2.0,
            cursor="cursor-1",
            heartbeat=True,
        ),
        StreamBatch(
            channel="ticker",
            events=({"symbol": "BTC-USDT", "price": 105.0},),
            received_at=3.0,
            cursor="cursor-2",
        ),
    )

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _SequenceStream:
        assert mapped_channels == ("ticker",)
        assert cursor is None
        return _SequenceStream(batches)

    client = NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
        channel_mapping={},
        buffer_size=4,
    )

    first = next(client)
    heartbeat = next(client)
    second = next(client)

    assert first.cursor == "cursor-1"
    assert heartbeat.heartbeat
    assert second.cursor == "cursor-2"
    assert client.last_cursor == "cursor-2"
    assert client.buffer_size == 4
    assert client.history_size == 3
    assert client.pending_size == 0

    assert client.replay_history(include_heartbeats=False)
    assert not client.replay_history()
    assert client.pending_size == 2

    replay_first = next(client)
    replay_second = next(client)
    assert replay_first.cursor == "cursor-1"
    assert not replay_first.heartbeat
    assert replay_second.cursor == "cursor-2"
    assert replay_second.events[0]["price"] == 105.0

    assert client.pending_size == 0
    assert client.replay_history(include_heartbeats=False)

    # kolejne pobrania pochodzą z odtworzonej historii
    assert next(client).cursor == "cursor-1"
    assert next(client).cursor == "cursor-2"

    client.close()


def test_stream_client_replay_history_force_overrides_guard() -> None:
    class _FlakyStream:
        def __init__(self, batches: Sequence[StreamBatch], error: Exception) -> None:
            self._batches = list(batches)
            self._error = error
            self.closed = False

        def __iter__(self) -> "_FlakyStream":
            return self

        def __next__(self) -> StreamBatch:
            if self._batches:
                return self._batches.pop(0)
            raise self._error

        def close(self) -> None:
            self.closed = True

    base_batch = StreamBatch(
        channel="ticker",
        events=({"symbol": "BTC-USDT", "price": 100.0},),
        received_at=1.0,
        cursor="cursor-1",
    )

    errors = [ExchangeNetworkError("zerwanie"), ExchangeNetworkError("koniec")]

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _FlakyStream:
        assert mapped_channels == ("ticker",)
        if not cursor:
            return _FlakyStream([base_batch], errors[0])
        return _FlakyStream([], errors[1])

    client = NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
        channel_mapping={},
        backoff_base=0.0,
        backoff_cap=0.0,
        max_reconnects=2,
    )

    first = next(client)
    assert first.cursor == "cursor-1"

    replayed = next(client)
    assert replayed.cursor == "cursor-1"

    assert not client.replay_history()
    assert client.replay_history(force=True)

    manual = next(client)
    assert manual.cursor == "cursor-1"

    client.close()


def test_stream_client_force_reconnect_handles_history_and_cursor() -> None:
    class _SequenceStream:
        def __init__(self, batches: Sequence[StreamBatch]) -> None:
            self._batches = list(batches)
            self.closed = False

        def __iter__(self) -> "_SequenceStream":
            return self

        def __next__(self) -> StreamBatch:
            if not self._batches:
                raise StopIteration
            return self._batches.pop(0)

        def close(self) -> None:
            self.closed = True

    payloads: dict[str | None, tuple[StreamBatch, ...]] = {
        None: (
            StreamBatch(
                channel="ticker",
                events=({"symbol": "BTC-USDT", "price": 100.0},),
                received_at=1.0,
                cursor="cur-1",
            ),
            StreamBatch(
                channel="ticker",
                events=({"symbol": "BTC-USDT", "price": 101.0},),
                received_at=2.0,
                cursor="cur-2",
            ),
        ),
        "cursor-override": (
            StreamBatch(
                channel="ticker",
                events=({"symbol": "BTC-USDT", "price": 102.0},),
                received_at=3.0,
                cursor="cur-3",
            ),
        ),
        "cur-3": (
            StreamBatch(
                channel="ticker",
                events=({"symbol": "BTC-USDT", "price": 103.0},),
                received_at=4.0,
                cursor="cur-4",
            ),
        ),
    }

    requests: list[tuple[tuple[str, ...], str | None]] = []

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _SequenceStream:
        requests.append((tuple(mapped_channels), cursor))
        batches = payloads.get(cursor)
        if batches is None:
            raise AssertionError(f"Nieoczekiwany kursor: {cursor!r}")
        return _SequenceStream(batches)

    client = NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
        channel_mapping={"ticker": "remote-ticker"},
        buffer_size=4,
        max_reconnects=5,
    )

    first = next(client)
    second = next(client)

    assert client.scope == "public"
    assert client.remote_channels == ("remote-ticker",)
    assert client.max_reconnects == 5
    assert client.reconnect_attempt == 0
    assert requests == [(("remote-ticker",), None)]

    client.force_reconnect(cursor="cursor-override")
    assert client.pending_size == 2
    assert client.reconnect_attempt == 0
    assert requests[-1] == (("remote-ticker",), "cursor-override")
    assert client.last_cursor == "cursor-override"

    replay_first = next(client)
    replay_second = next(client)
    assert replay_first.cursor == first.cursor
    assert replay_second.cursor == second.cursor
    assert client.pending_size == 0

    resumed = next(client)
    assert resumed.cursor == "cur-3"
    assert client.last_cursor == "cur-3"

    client.force_reconnect(replay_history=False)
    assert client.pending_size == 0
    assert requests[-1] == (("remote-ticker",), "cur-3")

    resumed_second = next(client)
    assert resumed_second.cursor == "cur-4"
    assert client.last_cursor == "cur-4"

    client.close()
    with pytest.raises(RuntimeError):
        client.force_reconnect()


def test_stream_client_diagnostics_counters_and_reset() -> None:
    class _ScriptedStream:
        def __init__(self, steps: list[StreamBatch | Exception]) -> None:
            self._steps = list(steps)
            self.closed = False

        def __iter__(self) -> "_ScriptedStream":
            return self

        def __next__(self) -> StreamBatch:
            if not self._steps:
                raise StopIteration
            step = self._steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        def close(self) -> None:
            self.closed = True

    batches = {
        "cur-1": StreamBatch(
            channel="ticker",
            events=({"symbol": "BTC-USDT", "price": 100.0},),
            received_at=1.0,
            cursor="cur-1",
        ),
        "heartbeat": StreamBatch(
            channel="ticker",
            events=(),
            received_at=1.5,
            cursor="cur-1",
            heartbeat=True,
        ),
        "cur-2": StreamBatch(
            channel="ticker",
            events=({"symbol": "BTC-USDT", "price": 101.5},),
            received_at=2.0,
            cursor="cur-2",
        ),
        "cur-3": StreamBatch(
            channel="ticker",
            events=({"symbol": "BTC-USDT", "price": 102.0},),
            received_at=3.0,
            cursor="cur-3",
        ),
    }

    scripted_sequences: list[list[StreamBatch | Exception]] = [
        [batches["cur-1"], batches["heartbeat"], ExchangeNetworkError("disconnect")],
        [batches["cur-2"]],
        [batches["cur-3"]],
    ]

    cursors_seen: list[str | None] = []

    def factory(mapped_channels: Sequence[str], cursor: str | None) -> _ScriptedStream:
        assert mapped_channels == ("ticker",)
        cursors_seen.append(cursor)
        if not scripted_sequences:
            raise AssertionError("Brak przygotowanych sekwencji dla streamu")
        return _ScriptedStream(scripted_sequences.pop(0))

    client = NowaGieldaStreamClient(
        scope="public",
        channels=["ticker"],
        fallback_factory=factory,  # type: ignore[arg-type]
        channel_mapping={},
        backoff_base=0.0,
        backoff_cap=0.0,
        max_reconnects=3,
    )

    first = next(client)
    assert first.cursor == batches["cur-1"].cursor
    assert first.events == batches["cur-1"].events
    assert client.total_batches == 1
    assert client.total_events == 1
    assert client.heartbeats_received == 0
    assert client.reconnects_total == 0

    heartbeat = next(client)
    assert heartbeat.heartbeat
    assert client.total_batches == 2
    assert client.total_events == 1
    assert client.heartbeats_received == 1

    replay_first = next(client)
    assert replay_first.cursor == "cur-1"
    assert client.total_batches == 3
    assert client.total_events == 2
    assert client.heartbeats_received == 1
    assert client.reconnects_total == 1

    replay_second = next(client)
    assert replay_second.heartbeat
    assert client.total_batches == 4
    assert client.total_events == 2
    assert client.heartbeats_received == 2

    resumed = next(client)
    assert resumed.cursor == "cur-2"
    assert client.total_batches == 5
    assert client.total_events == 3
    assert client.heartbeats_received == 2
    assert client.reconnect_attempt == 0

    client.force_reconnect(replay_history=False)
    assert client.reconnects_total == 2
    assert cursors_seen[-1] == "cur-2"
    assert client.pending_size == 0

    client.reset_counters()
    assert client.total_batches == 0
    assert client.total_events == 0
    assert client.heartbeats_received == 0
    assert client.reconnects_total == 0

    final = next(client)
    assert final.cursor == "cur-3"
    assert client.total_batches == 1
    assert client.total_events == 1
    assert client.heartbeats_received == 0
    assert client.reconnect_attempt == 0

    client.close()
    assert cursors_seen == [None, "cur-1", "cur-2"]

def test_stream_public_data_filters_and_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    stream_settings = {
        "base_url": "http://127.0.0.1:9876",
        "public_path": "/nowa/public",
        "poll_interval": 0.0,
        "timeout": 0.1,
        "max_retries": 2,
        "backoff_base": 0.0,
        "backoff_cap": 0.0,
        "jitter": (0.0, 0.0),
        "public_symbols": ["BTC_USDT"],
    }
    adapter = _build_stream_adapter(stream_settings)

    captured_urls: list[str] = []
    responses_queue: list[Any] = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [
                        {"symbol": "BTC-USDT", "last_price": 100.0},
                        {"symbol": "ETH-USDT", "last_price": 2000.0},
                    ],
                    "cursor": "cursor-1",
                }
            ],
            "retry_after": 0.0,
        },
        URLError("temporary disconnect"),
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [
                        {"symbol": "BTC-USDT", "last_price": 101.0},
                    ],
                    "cursor": "cursor-2",
                }
            ],
            "retry_after": 0.0,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        if not responses_queue:
            raise AssertionError("Żądano większej liczby pollingów niż oczekiwano")
        payload = responses_queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeStreamResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = adapter.stream_public_data(channels=["ticker"])
    first_batch = next(stream)
    assert isinstance(first_batch, StreamBatch)
    assert len(first_batch.events) == 1
    assert first_batch.events[0]["last_price"] == 100.0

    buffered_batch = next(stream)
    assert buffered_batch.events[0]["last_price"] == 100.0

    recovered_batch = next(stream)
    assert recovered_batch.events[0]["last_price"] == 101.0

    stream.close()

    assert len(captured_urls) >= 2
    reconnect_query = parse_qs(urlparse(captured_urls[-1]).query)
    assert reconnect_query.get("cursor") == ["cursor-1"]


def test_stream_private_data_requires_permissions() -> None:
    adapter = _build_stream_adapter({"base_url": "http://127.0.0.1:9876"}, permissions=("read",))

    with pytest.raises(PermissionError):
        adapter.stream_private_data(channels=["orders"])


def test_stream_private_data_emits_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    stream_settings = {
        "base_url": "http://127.0.0.1:9876",
        "private_path": "/nowa/private",
        "poll_interval": 0.0,
        "timeout": 0.1,
        "max_retries": 1,
        "backoff_base": 0.0,
        "backoff_cap": 0.0,
        "jitter": (0.0, 0.0),
        "private_symbols": ("BTC_USDT",),
    }
    adapter = _build_stream_adapter(stream_settings)

    captured_urls: list[str] = []
    payload = {
        "batches": [
            {
                "channel": "orders",
                "events": [
                    {"symbol": "BTC-USDT", "status": "NEW"},
                    {"symbol": "ETH-USDT", "status": "FILLED"},
                ],
                "cursor": "orders-1",
            },
            {
                "channel": "fills",
                "events": [
                    {"symbol": "BTC-USDT", "price": 100.0},
                ],
                "cursor": "fills-1",
            },
        ],
        "retry_after": 0.0,
    }

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        captured_urls.append(request.full_url)  # type: ignore[attr-defined]
        return _FakeStreamResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = adapter.stream_private_data(channels=["orders", "fills"])

    first = next(stream)
    assert first.channel == "orders"
    assert len(first.events) == 1
    assert first.events[0]["status"] == "NEW"

    second = next(stream)
    assert second.channel == "fills"
    assert second.events[0]["price"] == 100.0

    stream.close()

    assert len(captured_urls) == 1
    query = parse_qs(urlparse(captured_urls[0]).query)
    assert query.get("channels") == ["orders,fills"]


def test_stream_client_exposes_channels_and_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    stream_settings = {
        "base_url": "http://127.0.0.1:9876",
        "public_path": "/nowa/public",
        "poll_interval": 0.0,
        "timeout": 0.1,
        "jitter": (0.0, 0.0),
    }
    adapter = _build_stream_adapter(stream_settings)

    responses_queue: list[Any] = [
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"symbol": "BTC-USDT", "price": 100.0}],
                    "cursor": "cursor-1",
                }
            ],
            "retry_after": 0.0,
        },
        {
            "batches": [
                {
                    "channel": "ticker",
                    "events": [{"symbol": "BTC-USDT", "price": 101.0}],
                    "cursor": "cursor-2",
                }
            ],
            "retry_after": 0.0,
        },
    ]

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        if not responses_queue:
            raise AssertionError("Oczekiwano maksymalnie dwóch pollingów streamu")
        payload = responses_queue.pop(0)
        return _FakeStreamResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = adapter.stream_public_data(channels=["ticker"])

    assert stream.channels == ("ticker",)
    assert stream.last_cursor is None

    first_batch = next(stream)
    assert first_batch.cursor == "cursor-1"
    assert stream.last_cursor == "cursor-1"

    second_batch = next(stream)
    assert second_batch.cursor == "cursor-2"
    assert stream.last_cursor == "cursor-2"

    stream.close()

    assert stream.last_cursor == "cursor-2"
