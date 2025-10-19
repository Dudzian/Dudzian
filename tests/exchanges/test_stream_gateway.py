from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderResult,
)
from bot_core.exchanges.kraken.spot import (
    KrakenOpenOrder,
    KrakenOrderBook,
    KrakenOrderBookEntry,
    KrakenSpotAdapter,
    KrakenTicker,
)
from bot_core.exchanges.stream_gateway import StreamGateway, start_stream_gateway


@pytest.fixture
def kraken_credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="demo",
        secret="secret",
        environment=Environment.PAPER,
        permissions=("read", "trade"),
    )


def _start_gateway(adapter: KrakenSpotAdapter) -> tuple[StreamGateway, object, object, int]:
    gateway = StreamGateway(retry_after=0.0)
    gateway.register_adapter(getattr(adapter, "name", "kraken_spot"), environment=Environment.PAPER.value, adapter=adapter)
    server, thread = start_stream_gateway("127.0.0.1", 0, gateway=gateway)
    port = server.server_address[1]  # type: ignore[misc]
    return gateway, server, thread, port


def _stop_gateway(gateway: StreamGateway, server: object, thread: object) -> None:
    server.shutdown()  # type: ignore[attr-defined]
    thread.join(timeout=2.0)  # type: ignore[attr-defined]
    gateway.close()


def test_stream_gateway_public_stream_returns_batches(kraken_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = KrakenSpotAdapter(
        kraken_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
            }
        },
    )

    ticker_values: list[KrakenTicker] = [
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=100.5,
            best_bid=99.5,
            last_price=100.0,
            volume_24h=10.0,
            vwap_24h=100.1,
            high_24h=105.0,
            low_24h=95.0,
            open_price=98.0,
            timestamp=time.time(),
        ),
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=101.5,
            best_bid=100.5,
            last_price=101.0,
            volume_24h=11.0,
            vwap_24h=101.1,
            high_24h=106.0,
            low_24h=96.0,
            open_price=99.0,
            timestamp=time.time(),
        ),
    ]
    ticker_call = {"count": 0}

    def fake_fetch_ticker(symbol: str) -> KrakenTicker:
        index = min(ticker_call["count"], len(ticker_values) - 1)
        ticker_call["count"] += 1
        return ticker_values[index]

    order_book = KrakenOrderBook(
        symbol="XBT/USD",
        bids=(
            KrakenOrderBookEntry(price=101.0, volume=1.0, timestamp=time.time()),
            KrakenOrderBookEntry(price=100.5, volume=2.0, timestamp=time.time()),
        ),
        asks=(
            KrakenOrderBookEntry(price=101.5, volume=1.5, timestamp=time.time()),
            KrakenOrderBookEntry(price=102.0, volume=1.2, timestamp=time.time()),
        ),
        depth=2,
        timestamp=time.time(),
    )

    def fake_fetch_order_book(symbol: str, *, depth: int = 50) -> KrakenOrderBook:
        return order_book

    monkeypatch.setattr(adapter, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(adapter, "fetch_order_book", fake_fetch_order_book)

    gateway, server, thread, port = _start_gateway(adapter)
    stream_settings = adapter._settings.setdefault("stream", {})
    stream_settings["base_url"] = f"http://127.0.0.1:{port}"
    stream_settings.setdefault("public_params", {"symbol": "XBT/USD"})

    try:
        stream = adapter.stream_public_data(channels=["ticker", "depth"])
        first_batch = next(stream)
        second_batch = next(stream)
        batches = [first_batch, second_batch]
        ticker_batch = next(batch for batch in batches if batch.channel == "ticker")
        depth_batch = next(batch for batch in batches if batch.channel == "depth")

        assert ticker_batch.events[0]["symbol"] == "XBT/USD"
        assert ticker_batch.events[0]["last_price"] == pytest.approx(100.0)
        assert depth_batch.events[0]["symbol"] == "XBT/USD"
        assert depth_batch.events[0]["bids"][0]["price"] == pytest.approx(101.0)

        # kolejne zapytanie powinno zwrócić zaktualizowany ticker
        next_batches = [next(stream), next(stream)]
        next_ticker = next(batch for batch in next_batches if batch.channel == "ticker")
        assert next_ticker.cursor != ticker_batch.cursor
        assert next_ticker.events[0]["last_price"] == pytest.approx(101.0)
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_heartbeat_and_state_isolation(
    kraken_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = KrakenSpotAdapter(
        kraken_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
            }
        },
    )

    ticker_value = KrakenTicker(
        symbol="XBT/USD",
        best_ask=100.5,
        best_bid=99.5,
        last_price=100.0,
        volume_24h=10.0,
        vwap_24h=100.1,
        high_24h=105.0,
        low_24h=95.0,
        open_price=98.0,
        timestamp=time.time(),
    )

    def fake_fetch_ticker(symbol: str) -> KrakenTicker:
        return ticker_value

    monkeypatch.setattr(adapter, "fetch_ticker", fake_fetch_ticker)

    gateway, server, thread, port = _start_gateway(adapter)
    base_params = {
        "channels": "ticker",
        "symbol": "XBT/USD",
        "environment": adapter._environment.value,  # noqa: SLF001 - używane tylko w testach
    }
    base_url = f"http://127.0.0.1:{port}/stream/{adapter.name}/public"

    def _get(params: Mapping[str, str]) -> dict[str, Any]:
        query = urlencode(dict(params))
        with urlopen(f"{base_url}?{query}") as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        first_payload = _get(base_params)
        assert first_payload["batches"], "pierwsza odpowiedź powinna zawierać paczki"
        first_batch = first_payload["batches"][0]
        assert first_batch["events"], "powinny zostać zwrócone zdarzenia"
        assert first_batch["events"][0]["symbol"] == "XBT/USD"
        cursor = str(first_payload["cursor"])
        assert cursor

        # Mutacja lokalnej kopii nie powinna wpływać na przyszłe odpowiedzi
        first_batch["events"][0]["symbol"] = "HACK"

        heartbeat_payload = _get({**base_params, "cursor": cursor})
        heartbeat_batch = heartbeat_payload["batches"][0]
        assert heartbeat_batch["heartbeat"] is True
        assert heartbeat_batch["events"] == []
        assert heartbeat_payload["cursor"] == cursor

        third_payload = _get(base_params)
        third_batch = third_payload["batches"][0]
        assert third_batch["events"], "powinny zostać zwrócone zdarzenia po usunięciu kursora"
        assert third_batch["events"][0]["symbol"] == "XBT/USD"
        assert third_payload["cursor"] == cursor
    finally:
        _stop_gateway(gateway, server, thread)


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self._value = start

    def __call__(self) -> float:
        return self._value

    def advance(self, delta: float) -> None:
        self._value += delta


class _DummyAdapter(ExchangeAdapter):
    name = "dummy"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="demo"))
        self.calls = 0

    def configure_network(self, *, ip_allowlist=None):  # type: ignore[override]
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self):  # type: ignore[override]
        return ()

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # type: ignore[override]
        return ()

    def place_order(self, request):  # type: ignore[override]
        return OrderResult(order_id="0", status="open", filled_quantity=0.0, avg_price=None, raw_response={})

    def cancel_order(self, order_id, *, symbol=None):  # type: ignore[override]
        return None

    def stream_public_data(self, *, channels):  # type: ignore[override]
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # type: ignore[override]
        raise NotImplementedError

    def fetch_ticker(self, symbol: str):
        self.calls += 1
        return {"symbol": symbol, "price": 42.0}


def test_stream_gateway_evicts_stale_channel_state() -> None:
    clock = _Clock()
    gateway = StreamGateway(retry_after=0.0, state_ttl=1.0, clock=clock)
    adapter = _DummyAdapter()
    gateway.register_adapter(adapter.name, environment=None, adapter=adapter)

    params = {"symbol": ("BTC/USDT",)}

    first = gateway.handle_request(
        adapter_name=adapter.name,
        environment=None,
        scope="public",
        channels=["ticker"],
        params=params,
        cursor=None,
    )
    cursor = first["cursor"]
    first_batch = first["batches"][0]
    assert first_batch["events"], "pierwsza odpowiedź powinna zawierać zdarzenia"
    assert first_batch["heartbeat"] is False

    clock.advance(0.4)
    second = gateway.handle_request(
        adapter_name=adapter.name,
        environment=None,
        scope="public",
        channels=["ticker"],
        params=params,
        cursor=cursor,
    )
    second_batch = second["batches"][0]
    assert not second_batch["events"], "przed wygaśnięciem TTL powinien być heartbeat"
    assert second_batch["heartbeat"] is True

    clock.advance(2.0)
    third = gateway.handle_request(
        adapter_name=adapter.name,
        environment=None,
        scope="public",
        channels=["ticker"],
        params=params,
        cursor=cursor,
    )
    third_batch = third["batches"][0]
    assert third_batch["events"], "po wygaśnięciu TTL powinny pojawić się zdarzenia"
    assert third_batch["cursor"] != cursor
    assert third_batch["heartbeat"] is False
    assert adapter.calls == 3

    gateway.close()


def test_stream_gateway_accepts_post_body(
    kraken_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = KrakenSpotAdapter(
        kraken_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
            }
        },
    )

    ticker_values: list[KrakenTicker] = [
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=100.5,
            best_bid=99.5,
            last_price=100.0,
            volume_24h=10.0,
            vwap_24h=100.1,
            high_24h=105.0,
            low_24h=95.0,
            open_price=98.0,
            timestamp=time.time(),
        ),
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=101.5,
            best_bid=100.5,
            last_price=101.0,
            volume_24h=11.0,
            vwap_24h=101.1,
            high_24h=106.0,
            low_24h=96.0,
            open_price=99.0,
            timestamp=time.time(),
        ),
    ]
    ticker_call = {"count": 0}

    def fake_fetch_ticker(symbol: str) -> KrakenTicker:
        index = min(ticker_call["count"], len(ticker_values) - 1)
        ticker_call["count"] += 1
        return ticker_values[index]

    monkeypatch.setattr(adapter, "fetch_ticker", fake_fetch_ticker)

    gateway, server, thread, port = _start_gateway(adapter)
    stream_settings = adapter._settings.setdefault("stream", {})
    stream_settings.update(
        {
            "base_url": f"http://127.0.0.1:{port}",
            "method": "POST",
            "params_in_body": True,
            "channels_in_body": True,
            "cursor_in_body": True,
        }
    )
    stream_settings.setdefault("public_params", {"symbol": "XBT/USD"})

    try:
        stream = adapter.stream_public_data(channels=["ticker"])
        first_batch = next(stream)
        assert first_batch.events[0]["last_price"] == pytest.approx(100.0)

        second_batch = next(stream)
        assert second_batch.cursor != first_batch.cursor
        assert second_batch.events[0]["last_price"] == pytest.approx(101.0)
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_response_cursor_tracks_latest_batch(
    kraken_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = KrakenSpotAdapter(
        kraken_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
            }
        },
    )

    ticker_values: list[KrakenTicker] = [
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=100.5,
            best_bid=99.5,
            last_price=100.0,
            volume_24h=10.0,
            vwap_24h=100.1,
            high_24h=105.0,
            low_24h=95.0,
            open_price=98.0,
            timestamp=time.time(),
        ),
        KrakenTicker(
            symbol="XBT/USD",
            best_ask=101.5,
            best_bid=100.5,
            last_price=101.0,
            volume_24h=11.0,
            vwap_24h=101.1,
            high_24h=106.0,
            low_24h=96.0,
            open_price=99.0,
            timestamp=time.time(),
        ),
    ]
    ticker_call = {"count": 0}

    def fake_fetch_ticker(symbol: str) -> KrakenTicker:
        index = min(ticker_call["count"], len(ticker_values) - 1)
        ticker_call["count"] += 1
        return ticker_values[index]

    order_book = KrakenOrderBook(
        symbol="XBT/USD",
        bids=(
            KrakenOrderBookEntry(price=101.0, volume=1.0, timestamp=time.time()),
            KrakenOrderBookEntry(price=100.5, volume=2.0, timestamp=time.time()),
        ),
        asks=(
            KrakenOrderBookEntry(price=101.5, volume=1.5, timestamp=time.time()),
            KrakenOrderBookEntry(price=102.0, volume=1.2, timestamp=time.time()),
        ),
        depth=2,
        timestamp=time.time(),
    )

    def fake_fetch_order_book(symbol: str, *, depth: int = 50) -> KrakenOrderBook:
        return order_book

    monkeypatch.setattr(adapter, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(adapter, "fetch_order_book", fake_fetch_order_book)

    gateway, server, thread, port = _start_gateway(adapter)
    base_url = f"http://127.0.0.1:{port}/stream/{adapter.name}/public"
    common_params = {
        "channels": "ticker,depth",
        "symbol": "XBT/USD",
        "environment": adapter._environment.value,  # noqa: SLF001 - testowy dostęp
    }

    def _request(params: dict[str, str]) -> dict[str, object]:
        query = urlencode(params)
        with urlopen(f"{base_url}?{query}") as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        first_response = _request(common_params.copy())
        cursor = str(first_response["cursor"])
        assert cursor

        second_params = dict(common_params)
        second_params["cursor"] = cursor
        second_response = _request(second_params)

        ticker_batch = next(
            batch for batch in second_response["batches"] if batch["channel"] == "ticker"
        )
        assert ticker_batch["cursor"] != cursor
        assert second_response["cursor"] == ticker_batch["cursor"]
    finally:
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_private_stream_returns_orders(kraken_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = KrakenSpotAdapter(
        kraken_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
            }
        },
    )

    orders_sequence: list[list[KrakenOpenOrder]] = [
        [
            KrakenOpenOrder(
                order_id="OID-1",
                symbol="XBT/USD",
                side="buy",
                order_type="limit",
                price=100.0,
                volume=1.0,
                volume_executed=0.0,
                timestamp=time.time(),
                flags=("post",),
            )
        ],
        [
            KrakenOpenOrder(
                order_id="OID-1",
                symbol="XBT/USD",
                side="buy",
                order_type="limit",
                price=100.0,
                volume=1.0,
                volume_executed=0.5,
                timestamp=time.time(),
                flags=("post",),
            ),
            KrakenOpenOrder(
                order_id="OID-2",
                symbol="XBT/USD",
                side="sell",
                order_type="limit",
                price=102.0,
                volume=0.7,
                volume_executed=0.0,
                timestamp=time.time(),
                flags=("post",),
            ),
        ],
    ]
    order_calls = {"count": 0}

    def fake_fetch_open_orders() -> Iterable[KrakenOpenOrder]:
        index = min(order_calls["count"], len(orders_sequence) - 1)
        order_calls["count"] += 1
        return orders_sequence[index]

    monkeypatch.setattr(adapter, "fetch_open_orders", fake_fetch_open_orders)

    gateway, server, thread, port = _start_gateway(adapter)
    adapter._settings.setdefault("stream", {})["base_url"] = f"http://127.0.0.1:{port}"

    try:
        stream = adapter.stream_private_data(channels=["orders"])
        first_batch = next(stream)
        assert first_batch.events[0]["order_id"] == "OID-1"

        second_batch = next(stream)
        assert second_batch.cursor != first_batch.cursor
        assert [event["order_id"] for event in second_batch.events] == ["OID-1", "OID-2"]
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)
