from __future__ import annotations

import time
from typing import Iterable

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
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
