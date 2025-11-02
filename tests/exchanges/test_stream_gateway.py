from __future__ import annotations

import json
import time
import types
from typing import Iterable
from urllib.request import urlopen
import pytest

from bot_core.exchanges.base import Environment, ExchangeAdapter, ExchangeCredentials
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
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


@pytest.fixture
def binance_spot_credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="spot",
        secret="secret",
        environment=Environment.PAPER,
        permissions=("read", "trade"),
    )


@pytest.fixture
def binance_futures_credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="futures",
        secret="secret",
        environment=Environment.PAPER,
        permissions=("read", "trade"),
    )


def _start_gateway(adapter: ExchangeAdapter) -> tuple[StreamGateway, object, object, int]:
    gateway = StreamGateway(retry_after=0.0)
    adapter_name = getattr(adapter, "name", adapter.__class__.__name__.lower())
    environment_value: str | None = None
    credentials = getattr(adapter, "credentials", None)
    if isinstance(credentials, ExchangeCredentials):
        environment_value = credentials.environment.value
    configure = getattr(adapter, "configure_network", None)
    if callable(configure):
        configure(ip_allowlist=None)
    gateway.register_adapter(adapter_name, environment=environment_value, adapter=adapter)
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


def test_stream_gateway_status_and_reset_endpoint(
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

    ticker = KrakenTicker(
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

    monkeypatch.setattr(adapter, "fetch_ticker", lambda symbol: ticker)

    gateway, server, thread, port = _start_gateway(adapter)
    stream_settings = adapter._settings.setdefault("stream", {})
    stream_settings["base_url"] = f"http://127.0.0.1:{port}"
    stream_settings.setdefault("public_params", {"symbol": "XBT/USD"})

    try:
        stream = adapter.stream_public_data(channels=["ticker"])
        next(stream)

        snapshot = gateway.status_snapshot()
        assert snapshot["channels"], "gateway powinien raportować aktywne kanały"

        reset_url = (
            f"http://127.0.0.1:{port}/stream/{adapter.name}/public"
            "?channels=ticker&symbol=XBT/USD&reset=1&environment=paper"
        )
        with urlopen(reset_url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload.get("reset") is True
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_binance_spot_public_channels(
    binance_spot_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = BinanceSpotAdapter(
        binance_spot_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
                "public_params": {"symbol": "BTC/USDT", "depth": 10},
            }
        },
    )

    ticker_payloads = [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "30000.10",
            "askPrice": "30000.50",
            "lastPrice": "30000.20",
            "priceChangePercent": "1.5",
            "openPrice": "29500.00",
            "highPrice": "30500.00",
            "lowPrice": "29000.00",
            "volume": "125.5",
            "quoteVolume": "3765000",
            "closeTime": 1_700_000_000_000,
        },
        {
            "symbol": "BTCUSDT",
            "bidPrice": "30010.10",
            "askPrice": "30010.50",
            "lastPrice": "30010.20",
            "priceChangePercent": "1.8",
            "openPrice": "29500.00",
            "highPrice": "30500.00",
            "lowPrice": "29000.00",
            "volume": "150.0",
            "quoteVolume": "4503000",
            "closeTime": 1_700_000_100_000,
        },
    ]
    depth_payload = {
        "lastUpdateId": 123,
        "bids": [["30000.10", "0.5"], ["30000.00", "1.0"]],
        "asks": [["30000.50", "0.4"], ["30000.75", "0.8"]],
    }
    call_state = {"ticker": 0, "depth": 0}

    def fake_public_request(self, path: str, params: dict[str, object] | None = None, *, method: str = "GET"):
        del method
        assert params is not None
        if path == "/api/v3/ticker/24hr":
            assert params.get("symbol") == "BTCUSDT"
            index = min(call_state["ticker"], len(ticker_payloads) - 1)
            call_state["ticker"] += 1
            return ticker_payloads[index]
        if path == "/api/v3/depth":
            assert params.get("symbol") == "BTCUSDT"
            assert params.get("limit") == 10
            call_state["depth"] += 1
            return depth_payload
        raise AssertionError(f"Nieoczekiwany endpoint {path}")

    monkeypatch.setattr(
        adapter,
        "_public_request",
        types.MethodType(fake_public_request, adapter),
    )

    gateway, server, thread, port = _start_gateway(adapter)
    adapter._settings.setdefault("stream", {})["base_url"] = f"http://127.0.0.1:{port}"

    try:
        stream = adapter.stream_public_data(channels=["ticker", "depth"])
        first_batches = {batch.channel: batch for batch in (next(stream), next(stream))}
        ticker_batch = first_batches["ticker"]
        depth_batch = first_batches["depth"]

        assert ticker_batch.events[0]["symbol"] == "BTC/USDT"
        assert ticker_batch.events[0]["last_price"] == pytest.approx(30000.20)
        assert depth_batch.events[0]["symbol"] == "BTC/USDT"
        assert depth_batch.events[0]["bids"][0]["price"] == pytest.approx(30000.10)

        next_batches = {batch.channel: batch for batch in (next(stream), next(stream))}
        next_ticker = next_batches["ticker"]
        assert next_ticker.cursor != ticker_batch.cursor
        assert next_ticker.events[0]["last_price"] == pytest.approx(30010.20)
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_binance_spot_orders(
    binance_spot_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = BinanceSpotAdapter(
        binance_spot_credentials,
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

    open_orders_payloads = [
        [
            {
                "orderId": 1,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "side": "BUY",
                "type": "LIMIT",
                "price": "30000.10",
                "origQty": "0.010",
                "executedQty": "0",
                "timeInForce": "GTC",
                "clientOrderId": "order-1",
                "isWorking": True,
                "updateTime": 1_700_000_200_000,
            }
        ],
        [
            {
                "orderId": 1,
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "side": "BUY",
                "type": "LIMIT",
                "price": "30000.10",
                "origQty": "0.010",
                "executedQty": "0.005",
                "timeInForce": "GTC",
                "clientOrderId": "order-1",
                "isWorking": True,
                "updateTime": 1_700_000_300_000,
            },
            {
                "orderId": 2,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "side": "SELL",
                "type": "LIMIT",
                "price": "30050.10",
                "origQty": "0.015",
                "executedQty": "0",
                "timeInForce": "GTC",
                "clientOrderId": "order-2",
                "isWorking": True,
                "updateTime": 1_700_000_400_000,
            },
        ],
    ]
    call_state = {"count": 0}

    def fake_signed_request(self, path: str, *, method: str = "GET", params: dict[str, object] | None = None):
        del method, params
        assert path == "/api/v3/openOrders"
        index = min(call_state["count"], len(open_orders_payloads) - 1)
        call_state["count"] += 1
        return open_orders_payloads[index]

    monkeypatch.setattr(
        adapter,
        "_signed_request",
        types.MethodType(fake_signed_request, adapter),
    )

    gateway, server, thread, port = _start_gateway(adapter)
    adapter._settings.setdefault("stream", {})["base_url"] = f"http://127.0.0.1:{port}"

    try:
        stream = adapter.stream_private_data(channels=["orders"])
        first_batch = next(stream)
        assert first_batch.events[0]["order_id"] == "1"
        assert first_batch.events[0]["symbol"] == "BTC/USDT"

        second_batch = next(stream)
        assert second_batch.cursor != first_batch.cursor
        symbols = [event["symbol"] for event in second_batch.events]
        assert symbols == ["BTC/USDT", "BTC/USDT"]
        statuses = [event["status"] for event in second_batch.events]
        assert statuses == ["PARTIALLY_FILLED", "NEW"]
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_binance_futures_public_channels(
    binance_futures_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = BinanceFuturesAdapter(
        binance_futures_credentials,
        environment=Environment.PAPER,
        settings={
            "stream": {
                "poll_interval": 0.0,
                "timeout": 0.5,
                "max_retries": 1,
                "backoff_base": 0.0,
                "backoff_cap": 0.0,
                "jitter": (0.0, 0.0),
                "public_params": {"symbol": "BTCUSDT", "depth": 5},
            }
        },
    )

    ticker_payloads = [
        {
            "symbol": "BTCUSDT",
            "bidPrice": "29900.10",
            "askPrice": "29900.50",
            "lastPrice": "29900.30",
            "priceChangePercent": "-0.5",
            "openPrice": "30100.00",
            "highPrice": "30500.00",
            "lowPrice": "29500.00",
            "volume": "200",
            "quoteVolume": "5980000",
            "openInterest": "1500",
            "closeTime": 1_700_001_000_000,
        },
        {
            "symbol": "BTCUSDT",
            "bidPrice": "29910.10",
            "askPrice": "29910.50",
            "lastPrice": "29910.30",
            "priceChangePercent": "-0.3",
            "openPrice": "30100.00",
            "highPrice": "30500.00",
            "lowPrice": "29500.00",
            "volume": "210",
            "quoteVolume": "6281100",
            "openInterest": "1510",
            "closeTime": 1_700_001_100_000,
        },
    ]
    depth_payload = {
        "lastUpdateId": 456,
        "bids": [["29900.10", "5"], ["29900.00", "3"]],
        "asks": [["29900.50", "4"], ["29900.75", "2"]],
    }
    call_state = {"ticker": 0}

    def fake_public_request(self, path: str, params: dict[str, object] | None = None, *, method: str = "GET"):
        del method
        assert params is not None
        if path == "/fapi/v1/ticker/24hr":
            assert params.get("symbol") == "BTCUSDT"
            index = min(call_state["ticker"], len(ticker_payloads) - 1)
            call_state["ticker"] += 1
            return ticker_payloads[index]
        if path == "/fapi/v1/depth":
            assert params.get("symbol") == "BTCUSDT"
            assert params.get("limit") == 5
            return depth_payload
        raise AssertionError(path)

    monkeypatch.setattr(
        adapter,
        "_public_request",
        types.MethodType(fake_public_request, adapter),
    )

    gateway, server, thread, port = _start_gateway(adapter)
    adapter._settings.setdefault("stream", {})["base_url"] = f"http://127.0.0.1:{port}"

    try:
        stream = adapter.stream_public_data(channels=["ticker", "depth"])
        first_batches = {batch.channel: batch for batch in (next(stream), next(stream))}
        ticker_batch = first_batches["ticker"]
        depth_batch = first_batches["depth"]

        assert ticker_batch.events[0]["symbol"] == "BTCUSDT"
        assert ticker_batch.events[0]["open_interest"] == pytest.approx(1500.0)
        assert depth_batch.events[0]["bids"][0]["quantity"] == pytest.approx(5.0)

        next_batches = {batch.channel: batch for batch in (next(stream), next(stream))}
        next_ticker = next_batches["ticker"]
        assert next_ticker.cursor != ticker_batch.cursor
        assert next_ticker.events[0]["last_price"] == pytest.approx(29910.30)
    finally:
        stream.close()
        _stop_gateway(gateway, server, thread)


def test_stream_gateway_binance_futures_orders(
    binance_futures_credentials: ExchangeCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = BinanceFuturesAdapter(
        binance_futures_credentials,
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

    open_orders_payloads = [
        [
            {
                "orderId": 10,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "side": "BUY",
                "type": "LIMIT",
                "price": "29900.10",
                "origQty": "0.2",
                "executedQty": "0",
                "timeInForce": "GTC",
                "clientOrderId": "f-order-1",
                "reduceOnly": False,
                "closePosition": False,
                "workingType": "CONTRACT_PRICE",
                "priceProtect": True,
                "positionSide": "LONG",
                "updateTime": 1_700_002_000_000,
            }
        ],
        [
            {
                "orderId": 10,
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "side": "BUY",
                "type": "LIMIT",
                "price": "29900.10",
                "origQty": "0.2",
                "executedQty": "0.1",
                "timeInForce": "GTC",
                "clientOrderId": "f-order-1",
                "reduceOnly": False,
                "closePosition": False,
                "workingType": "CONTRACT_PRICE",
                "priceProtect": True,
                "positionSide": "LONG",
                "updateTime": 1_700_002_100_000,
            },
            {
                "orderId": 11,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "side": "SELL",
                "type": "STOP_MARKET",
                "price": "0",
                "stopPrice": "29800.00",
                "origQty": "0.2",
                "executedQty": "0",
                "timeInForce": "GTC",
                "clientOrderId": "f-order-2",
                "reduceOnly": True,
                "closePosition": True,
                "workingType": "MARK_PRICE",
                "priceProtect": False,
                "positionSide": "LONG",
                "updateTime": 1_700_002_200_000,
            },
        ],
    ]
    call_state = {"count": 0}

    def fake_signed_request(self, path: str, *, method: str = "GET", params: dict[str, object] | None = None):
        del method, params
        assert path == "/fapi/v1/openOrders"
        index = min(call_state["count"], len(open_orders_payloads) - 1)
        call_state["count"] += 1
        return open_orders_payloads[index]

    monkeypatch.setattr(
        adapter,
        "_signed_request",
        types.MethodType(fake_signed_request, adapter),
    )

    gateway, server, thread, port = _start_gateway(adapter)
    adapter._settings.setdefault("stream", {})["base_url"] = f"http://127.0.0.1:{port}"

    try:
        stream = adapter.stream_private_data(channels=["orders"])
        first_batch = next(stream)
        assert first_batch.events[0]["order_id"] == "10"
        assert first_batch.events[0]["price_protect"] is True

        second_batch = next(stream)
        assert second_batch.cursor != first_batch.cursor
        symbols = [event["symbol"] for event in second_batch.events]
        assert symbols == ["BTCUSDT", "BTCUSDT"]
        assert second_batch.events[1]["stop_price"] == pytest.approx(29800.00)
        assert second_batch.events[1]["reduce_only"] is True
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
