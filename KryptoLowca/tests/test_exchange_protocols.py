from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution import live_router as live_router_module
from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition
from bot_core.exchanges.base import OrderResult

from KryptoLowca.exchanges import (
    BinanceTestnetAdapter,
    ExchangeCredentials,
    MarketDataPoller,
    MarketSubscription,
    KrakenDemoAdapter,
    OrderRequest,
    OrderStatus,
    ZondaAdapter,
)
import KryptoLowca.exchanges.binance as binance_module
import KryptoLowca.exchanges.kraken as kraken_module
import KryptoLowca.exchanges.zonda as zonda_module


class StubResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class RecordingHTTPClient:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def request(self, method, url, *, params=None, data=None, headers=None, timeout=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": data,
                "headers": headers,
            }
        )
        if not self.responses:
            raise RuntimeError("No more responses configured")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_binance_adapter_signed_requests():
    response = StubResponse(
        {"orderId": 1, "status": "NEW", "executedQty": "0", "origQty": "1", "price": "0"}
    )
    http = RecordingHTTPClient([response])
    adapter = BinanceTestnetAdapter(http_client=http)
    await adapter.connect()
    await adapter.authenticate(ExchangeCredentials(api_key="k", api_secret="s"))

    order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=1.0, order_type="MARKET")
    status = await adapter.submit_order(order)

    call = http.requests[0]
    assert call["headers"]["X-MBX-APIKEY"] == "k"
    assert "signature" in call["params"]
    assert status.order_id == 1
    assert status.status == "NEW"


@pytest.mark.asyncio
async def test_binance_rate_limit_retry(monkeypatch):
    success = StubResponse(
        {
            "orderId": 1,
            "status": "FILLED",
            "executedQty": "1",
            "origQty": "1",
            "price": "20000",
        }
    )
    http = RecordingHTTPClient([StubResponse({"error": "429"}, status_code=429), success])

    async def fast_sleep(delay: float) -> None:  # pragma: no cover - test helper
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    adapter = BinanceTestnetAdapter(http_client=http)
    await adapter.connect()
    await adapter.authenticate(ExchangeCredentials(api_key="k", api_secret="s"))

    status = await adapter.fetch_order_status("1", symbol="BTCUSDT")
    assert status.status == "FILLED"
    assert len(http.requests) == 2


@pytest.mark.asyncio
async def test_kraken_headers_signed():
    http = RecordingHTTPClient(
        [
            StubResponse({"result": {"txid": ["ABC"], "descr": {"price": "10"}}}),
            StubResponse({"result": {"ABC": {"status": "closed", "vol": "1", "vol_exec": "1"}}}),
        ]
    )
    adapter = KrakenDemoAdapter(http_client=http)
    await adapter.connect()
    secret = base64.b64encode(b"topsecret").decode()
    await adapter.authenticate(ExchangeCredentials(api_key="key", api_secret=secret))

    order = OrderRequest(symbol="XBTUSD", side="buy", quantity=1.0, order_type="limit", price=10)
    status = await adapter.submit_order(order)
    assert status.order_id == "ABC"
    call = http.requests[0]
    assert call["headers"]["API-Key"] == "key"
    assert "API-Sign" in call["headers"]

    detail = await adapter.fetch_order_status("ABC")
    assert detail.status == "CLOSED"
    assert len(http.requests) == 2


@pytest.mark.asyncio
async def test_zonda_signed_headers(monkeypatch):
    response = StubResponse(
        {
            "status": "Ok",
            "offerId": "1",
            "filledAmount": "0",
            "remainingAmount": "1",
            "avgPrice": "100",
        }
    )
    http = RecordingHTTPClient([response])
    adapter = ZondaAdapter(http_client=http)
    await adapter.connect()
    await adapter.authenticate(ExchangeCredentials(api_key="pub", api_secret="priv"))

    monkeypatch.setattr(zonda_module.time, "time", lambda: 1_700_000_000.0)

    order = OrderRequest(symbol="BTC-PLN", side="buy", quantity=1.0, price=100.0, order_type="limit")
    status = await adapter.submit_order(order)

    call = http.requests[0]
    assert call["headers"]["API-Key"] == "pub"
    timestamp = call["headers"]["Request-Timestamp"]
    body = call["data"] or ""
    expected_signature = hmac.new(
        b"priv",
        f"{timestamp}POST/trading/offer{body}".encode(),
        hashlib.sha512,
    ).hexdigest()
    assert call["headers"]["API-Hash"] == expected_signature
    assert status.order_id == "1"
    assert status.status == "OK"
    assert status.remaining_quantity == 1.0


@pytest.mark.asyncio
async def test_zonda_fetch_and_cancel(monkeypatch):
    responses = [
        StubResponse(
            {
                "status": "Ok",
                "order": {
                    "id": "XYZ",
                    "status": "active",
                    "filledAmount": "0.1",
                    "remainingAmount": "0.9",
                    "avgPrice": "99.5",
                },
            }
        ),
        StubResponse(
            {
                "status": "Ok",
                "order": {
                    "id": "XYZ",
                    "status": "cancelled",
                    "filledAmount": "0.1",
                    "remainingAmount": "0.9",
                    "avgPrice": "99.5",
                },
            }
        ),
    ]
    http = RecordingHTTPClient(responses)
    adapter = ZondaAdapter(http_client=http)
    await adapter.connect()
    await adapter.authenticate(ExchangeCredentials(api_key="pub", api_secret="priv"))

    monkeypatch.setattr(zonda_module.time, "time", lambda: 1_700_000_000.0)

    detail = await adapter.fetch_order_status("XYZ")
    call_status = http.requests[0]
    ts_status = call_status["headers"]["Request-Timestamp"]
    expected_status_sig = hmac.new(
        b"priv",
        f"{ts_status}GET/trading/order/XYZ".encode(),
        hashlib.sha512,
    ).hexdigest()
    assert call_status["headers"]["API-Hash"] == expected_status_sig
    assert detail.status == "ACTIVE"
    assert detail.filled_quantity == 0.1

    cancelled = await adapter.cancel_order("XYZ")
    call_cancel = http.requests[1]
    ts_cancel = call_cancel["headers"]["Request-Timestamp"]
    expected_cancel_sig = hmac.new(
        b"priv",
        f"{ts_cancel}DELETE/trading/order/XYZ".encode(),
        hashlib.sha512,
    ).hexdigest()
    assert call_cancel["headers"]["API-Hash"] == expected_cancel_sig
    assert cancelled.status == "CANCELLED"
    assert len(http.requests) == 2


@pytest.mark.asyncio
async def test_zonda_status_mapping_variants(monkeypatch):
    responses = [
        StubResponse(
            {
                "status": "Ok",
                "order": {
                    "id": "AAA",
                    "status": "PartiallyFilled",
                    "filledAmount": "0.5",
                    "remainingAmount": "0.5",
                },
            }
        ),
        StubResponse(
            {
                "status": "Ok",
                "order": {
                    "id": "AAA",
                    "status": "closed",
                    "filledAmount": "1",
                    "remainingAmount": "0",
                },
            }
        ),
        StubResponse(
            {
                "status": "Ok",
                "order": {
                    "id": "AAA",
                    "status": "waiting",
                    "filledAmount": "0",
                    "remainingAmount": "1",
                },
            }
        ),
    ]
    http = RecordingHTTPClient(responses)
    adapter = ZondaAdapter(http_client=http)
    await adapter.connect()
    await adapter.authenticate(ExchangeCredentials(api_key="pub", api_secret="priv"))

    monkeypatch.setattr(zonda_module.time, "time", lambda: 1_700_000_000.0)

    partial = await adapter.fetch_order_status("AAA")
    filled = await adapter.fetch_order_status("AAA")
    pending = await adapter.fetch_order_status("AAA")

    assert partial.status == "PARTIALLY_FILLED"
    assert partial.filled_quantity == 0.5
    assert filled.status == "FILLED"
    assert pending.status == "PENDING"
    assert len(http.requests) == 3


@pytest.mark.asyncio
async def test_zonda_websocket_subscription(monkeypatch):
    events: list[dict] = []
    done = asyncio.Event()

    class FakeWebSocket:
        def __init__(self, messages: list[str]) -> None:
            self._messages = list(messages)
            self._closed = asyncio.Event()
            self.sent: list[dict] = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            await self._closed.wait()
            raise StopAsyncIteration

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        async def close(self) -> None:
            self._closed.set()

    fake_ws = FakeWebSocket([json.dumps({"type": "ticker", "payload": {"symbol": "BTC-PLN"}})])

    class FakeWebSocketsModule:
        def __init__(self, websocket: FakeWebSocket) -> None:
            self._websocket = websocket
            self.connected: list[tuple[str, float]] = []

        async def connect(self, endpoint: str, ping_interval: int = 20):
            self.connected.append((endpoint, ping_interval))
            return self._websocket

    fake_module = FakeWebSocketsModule(fake_ws)
    monkeypatch.setattr(zonda_module, "websockets", fake_module)

    adapter = ZondaAdapter(http_client=RecordingHTTPClient([]), enable_streaming=True)

    async def callback(payload: dict) -> None:
        events.append(payload)
        done.set()

    subscription = await adapter.stream_market_data(
        [MarketSubscription(channel="trading/ticker", symbols=["BTC-PLN"])],
        callback,
    )

    async with subscription:
        await asyncio.wait_for(done.wait(), timeout=0.2)

    assert fake_module.connected[0][0] == "wss://api.zondacrypto.exchange/websocket"
    actions = [message["action"] for message in fake_ws.sent]
    assert "subscribe-public" in actions
    assert "unsubscribe" in actions
    assert events and events[0]["payload"]["symbol"] == "BTC-PLN"


@pytest.mark.asyncio
async def test_zonda_streaming_disabled_by_default():
    adapter = ZondaAdapter(http_client=RecordingHTTPClient([]))

    with pytest.raises(RuntimeError, match="Streaming danych rynkowych przez WebSocket Zonda jest wyłączony"):
        await adapter.stream_market_data(
            [MarketSubscription(channel="trading/ticker", symbols=["BTC-PLN"])],
            lambda payload: asyncio.sleep(0),
        )


@pytest.mark.asyncio
async def test_market_data_poller_polls_symbols():
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def fetch_market_data(self, symbol: str) -> dict:
            self.calls.append(symbol)
            return {"symbol": symbol, "price": 100.0}

    adapter = StubAdapter()
    events: list[tuple[str, dict]] = []
    done = asyncio.Event()

    async def callback(symbol: str, payload: dict) -> None:
        events.append((symbol, payload))
        if len(events) >= 3:
            done.set()

    poller = MarketDataPoller(adapter, symbols=["BTC-PLN", "ETH-PLN"], interval=0.05, callback=callback)

    async with poller:
        await asyncio.wait_for(done.wait(), timeout=0.5)

    assert any(symbol == "BTC-PLN" for symbol, _ in events)
    assert any(symbol == "ETH-PLN" for symbol, _ in events)
    assert len(adapter.calls) >= len(events)


@pytest.mark.asyncio
async def test_market_data_poller_reports_errors(caplog):
    class FlakyAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_market_data(self, symbol: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return {"symbol": symbol, "price": 99.0}

    adapter = FlakyAdapter()
    events: list[dict] = []
    done = asyncio.Event()
    errors: list[tuple[str, Exception]] = []

    async def callback(symbol: str, payload: dict) -> None:
        events.append(payload)
        done.set()

    async def error_callback(symbol: str, exc: Exception) -> None:
        errors.append((symbol, exc))

    poller = MarketDataPoller(
        adapter,
        symbols=["BTC-PLN"],
        interval=0.05,
        callback=callback,
        error_callback=error_callback,
    )

    caplog.set_level("WARNING")
    async with poller:
        await asyncio.wait_for(done.wait(), timeout=0.5)

    assert events and events[0]["price"] == 99.0
    assert any("Nie udało się pobrać" in message for message in caplog.messages)
    assert errors and isinstance(errors[0][1], RuntimeError)


def test_market_data_poller_requires_symbols_and_positive_interval():
    class StubAdapter:
        async def fetch_market_data(self, symbol: str) -> dict:
            return {"symbol": symbol}

    adapter = StubAdapter()

    with pytest.raises(ValueError, match="co najmniej jednego symbolu"):
        MarketDataPoller(adapter, symbols=[], callback=lambda *_: None)

    with pytest.raises(ValueError, match="Odstęp odpytywania musi być dodatni"):
        MarketDataPoller(adapter, symbols=["BTC-PLN"], interval=0.0, callback=lambda *_: None)


def test_zonda_build_ws_messages_infers_defaults():
    subscription = MarketSubscription(channel="trading/ticker", symbols=["BTC-PLN"])

    subscribe, unsubscribe = zonda_module._build_ws_messages([subscription])

    assert subscribe == [
        {
            "action": "subscribe-public",
            "module": "trading",
            "path": "ticker",
            "params": {"symbol": "BTC-PLN"},
        }
    ]
    assert unsubscribe == [
        {
            "action": "unsubscribe",
            "module": "trading",
            "path": "ticker",
            "params": {"symbol": "BTC-PLN"},
        }
    ]


def test_zonda_build_ws_messages_respects_custom_actions_and_params():
    subscription = MarketSubscription(
        channel="public/trades",
        symbols=["BTC-PLN", "ETH-PLN"],
        params={
            "module": "spot",
            "path": "marketTrades",
            "action": "subscribe-private",
            "unsubscribe_action": "unsubscribe-private",
            "params": {"depth": 50, "foo": "bar"},
        },
    )

    subscribe, unsubscribe = zonda_module._build_ws_messages([subscription])

    expected_params = [
        {
            "action": "subscribe-private",
            "module": "spot",
            "path": "marketTrades",
            "params": {"depth": 50, "foo": "bar", "symbol": symbol},
        }
        for symbol in ("BTC-PLN", "ETH-PLN")
    ]
    expected_unsub = [
        {
            "action": "unsubscribe-private",
            "module": "spot",
            "path": "marketTrades",
            "params": {"depth": 50, "foo": "bar", "symbol": symbol},
        }
        for symbol in ("BTC-PLN", "ETH-PLN")
    ]

    assert subscribe == expected_params
    assert unsubscribe == expected_unsub


@pytest.mark.asyncio
async def test_binance_websocket_subscription(monkeypatch):
    events: list[dict] = []
    done = asyncio.Event()

    class FakeWebSocket:
        def __init__(self, messages: list[str]) -> None:
            self._messages = list(messages)
            self._closed = asyncio.Event()
            self.sent: list[dict] = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            await self._closed.wait()
            raise StopAsyncIteration

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        async def close(self) -> None:
            self._closed.set()

    fake_ws = FakeWebSocket(
        [json.dumps({"stream": "btcusdt@ticker", "data": {"c": "123.45"}})]
    )

    class FakeWebSocketsModule:
        def __init__(self, websocket: FakeWebSocket) -> None:
            self._websocket = websocket
            self.connected: list[tuple[str, int]] = []

        async def connect(self, endpoint: str, ping_interval: int = 20):
            self.connected.append((endpoint, ping_interval))
            return self._websocket

    fake_module = FakeWebSocketsModule(fake_ws)
    monkeypatch.setattr(binance_module, "websockets", fake_module)
    monkeypatch.setattr(binance_module, "_WSState", None, raising=False)

    adapter = BinanceTestnetAdapter(http_client=RecordingHTTPClient([]))

    async def callback(payload: dict) -> None:
        events.append(payload)
        done.set()

    subscription = await adapter.stream_market_data(
        [MarketSubscription(channel="ticker", symbols=["BTCUSDT"])],
        callback,
    )

    async with subscription:
        await asyncio.wait_for(done.wait(), timeout=0.2)

    assert fake_module.connected[0][0] == binance_module._BINANCE_WS_ENDPOINT
    assert fake_ws.sent[0]["method"] == "SUBSCRIBE"
    assert fake_ws.sent[0]["params"] == ["btcusdt@ticker"]
    assert fake_ws.sent[1]["method"] == "UNSUBSCRIBE"
    assert events and events[0]["data"]["c"] == "123.45"


@pytest.mark.asyncio
async def test_kraken_websocket_subscription(monkeypatch):
    events: list[dict | list] = []
    done = asyncio.Event()

    class FakeWebSocket:
        def __init__(self, messages: list[str]) -> None:
            self._messages = list(messages)
            self._closed = asyncio.Event()
            self.sent: list[dict] = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            await self._closed.wait()
            raise StopAsyncIteration

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        async def close(self) -> None:
            self._closed.set()

    fake_ws = FakeWebSocket(
        [json.dumps([42, {"c": ["123.45", "123.46"]}, "ticker", "XBT/USDT"])]
    )

    class FakeWebSocketsModule:
        def __init__(self, websocket: FakeWebSocket) -> None:
            self._websocket = websocket
            self.connected: list[tuple[str, int]] = []

        async def connect(self, endpoint: str, ping_interval: int = 20):
            self.connected.append((endpoint, ping_interval))
            return self._websocket

    fake_module = FakeWebSocketsModule(fake_ws)
    monkeypatch.setattr(kraken_module, "websockets", fake_module)
    monkeypatch.setattr(kraken_module, "_WSState", None, raising=False)

    adapter = KrakenDemoAdapter(http_client=RecordingHTTPClient([]))

    async def callback(payload):
        events.append(payload)
        done.set()

    subscription = await adapter.stream_market_data(
        [
            MarketSubscription(
                channel="ticker",
                symbols=["XBT/USDT"],
                params={"subscription": {"name": "ticker", "interval": 1}},
            )
        ],
        callback,
    )

    async with subscription:
        await asyncio.wait_for(done.wait(), timeout=0.2)

    assert fake_module.connected[0][0] == kraken_module._KRAKEN_WS_ENDPOINT
    assert fake_ws.sent[0]["event"] == "subscribe"
    assert fake_ws.sent[0]["pair"] == ["XBT/USDT"]
    assert fake_ws.sent[0]["subscription"]["interval"] == 1
    assert fake_ws.sent[1]["event"] == "unsubscribe"
    assert events and isinstance(events[0], list) and events[0][1]["c"][0] == "123.45"


class RouterStubAdapter:
    def __init__(self, name: str, outcomes: list[OrderResult | Exception]) -> None:
        self.name = name
        self._outcomes = list(outcomes)
        self.cancelled: list[str] = []

    def place_order(self, request: OrderRequest) -> OrderResult:
        if not self._outcomes:
            raise RuntimeError("Brak zdefiniowanego wyniku dla adaptera")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        self.cancelled.append(order_id)


def _make_execution_context(metadata: dict[str, str] | None = None) -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="test-portfolio",
        risk_profile="balanced",
        environment="paper",
        metadata=metadata or {},
    )


def test_live_execution_router_fallback_and_binding(tmp_path):
    primary = RouterStubAdapter(
        "binance",
        [live_router_module.ExchangeNetworkError("temporary outage")],
    )
    success_result = OrderResult(
        order_id="kraken-1",
        status="FILLED",
        filled_quantity=1.0,
        avg_price=101.0,
        raw_response={"exchange": "kraken"},
    )
    secondary = RouterStubAdapter("kraken", [success_result])

    router = LiveExecutionRouter(
        adapters={"binance": primary, "kraken": secondary},
        default_route=("binance", "kraken"),
        decision_log_path=tmp_path / "router.log",
        decision_log_hmac_key=os.urandom(64),
        decision_log_key_id="tests",
    )

    request = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=1.0, order_type="MARKET")
    context = _make_execution_context()

    result = router.execute(request, context)

    assert result.order_id == "kraken-1"
    assert router.binding_for_order(result.order_id) == "kraken"

    router.cancel(result.order_id, context)
    assert result.order_id in secondary.cancelled

    router.flush()


def test_live_execution_router_route_overrides():
    kraken_result = OrderResult(
        order_id="kraken-override",
        status="FILLED",
        filled_quantity=2.0,
        avg_price=200.0,
        raw_response={"exchange": "kraken"},
    )
    binance_default = OrderResult(
        order_id="binance-default",
        status="FILLED",
        filled_quantity=0.5,
        avg_price=50.0,
        raw_response={"exchange": "binance"},
    )
    binance_followup = OrderResult(
        order_id="binance-followup",
        status="FILLED",
        filled_quantity=0.75,
        avg_price=55.0,
        raw_response={"exchange": "binance"},
    )

    router = LiveExecutionRouter(
        adapters={
            "kraken": RouterStubAdapter("kraken", [kraken_result]),
            "binance": RouterStubAdapter("binance", [binance_default, binance_followup]),
        },
        default_route=("binance",),
        route_overrides={"BTCUSDT": ("kraken",), "ETHUSDT": ("binance",)},
    )

    btc_request = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=2.0, order_type="MARKET")
    context = _make_execution_context()
    btc_result = router.execute(btc_request, context)
    assert btc_result.order_id == "kraken-override"
    assert router.binding_for_order("kraken-override") == "kraken"

    eth_request = OrderRequest(symbol="ETHUSDT", side="BUY", quantity=1.0, order_type="MARKET")
    eth_result = router.execute(eth_request, context)
    assert eth_result.order_id == "binance-default"
    assert router.binding_for_order("binance-default") == "binance"

    second_eth = router.execute(
        OrderRequest(symbol="ETHUSDT", side="BUY", quantity=0.5, order_type="MARKET"),
        context,
    )
    assert second_eth.order_id == "binance-followup"


def test_live_execution_router_named_route_selection():
    vip_result = OrderResult(
        order_id="vip-route",
        status="FILLED",
        filled_quantity=1.5,
        avg_price=150.0,
        raw_response={"exchange": "kraken"},
    )
    standard_result = OrderResult(
        order_id="standard-route",
        status="FILLED",
        filled_quantity=1.0,
        avg_price=100.0,
        raw_response={"exchange": "binance"},
    )

    router = LiveExecutionRouter(
        adapters={
            "kraken": RouterStubAdapter("kraken", [vip_result]),
            "binance": RouterStubAdapter("binance", [standard_result]),
        },
        routes=[
            RouteDefinition(
                name="vip",
                exchanges=("kraken",),
                risk_profiles=("vip",),
                metadata={"owner": "vip-desk"},
            ),
            RouteDefinition(
                name="standard",
                exchanges=("binance",),
                metadata={"owner": "default"},
            ),
        ],
        default_route="standard",
    )

    request = OrderRequest(symbol="ADAUSDT", side="BUY", quantity=1.5, order_type="MARKET")
    context = _make_execution_context({"execution_route": "vip"})
    result = router.execute(request, context)
    assert result.order_id == "vip-route"
    assert router.binding_for_order("vip-route") == "kraken"


@pytest.mark.asyncio
async def test_live_mode_requires_ack(monkeypatch):
    http = RecordingHTTPClient([])
    with pytest.raises(ValueError):
        BinanceTestnetAdapter(demo_mode=False, http_client=http)

    adapter = BinanceTestnetAdapter(demo_mode=False, http_client=http, compliance_ack=True)
    await adapter.connect()
