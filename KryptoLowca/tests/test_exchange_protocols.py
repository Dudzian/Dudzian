from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet
import websockets

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.exchanges import (
    BinanceTestnetAdapter,
    ExchangeCredentials,
    MarketSubscription,
    KrakenDemoAdapter,
    OrderRequest,
    OrderStatus,
)
from KryptoLowca.managers.multi_account_manager import MultiExchangeAccountManager

import KryptoLowca.exchanges.binance as binance_module
import KryptoLowca.exchanges.kraken as kraken_module


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
async def test_binance_websocket_subscription(monkeypatch):
    events: list[dict] = []
    received_messages: list[dict] = []
    received_event = asyncio.Event()

    async def handler(websocket):
        subscribe = json.loads(await websocket.recv())
        received_messages.append(subscribe)
        await websocket.send(json.dumps({"stream": "btcusdt@trade", "data": {"p": "100"}}))
        try:
            unsub = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            received_messages.append(json.loads(unsub))
        except asyncio.TimeoutError:
            pass

    server = await websockets.serve(handler, "localhost", 0)
    port = server.sockets[0].getsockname()[1]
    endpoint = f"ws://localhost:{port}"
    monkeypatch.setattr(binance_module, "_BINANCE_WS_ENDPOINT", endpoint)
    monkeypatch.setattr(binance_module, "_BINANCE_WS_INITIAL_BACKOFF", 0.05)
    monkeypatch.setattr(binance_module, "_BINANCE_WS_MAX_BACKOFF", 0.1)

    adapter = BinanceTestnetAdapter(http_client=RecordingHTTPClient([]))
    await adapter.connect()

    async def callback(payload):
        events.append(payload)
        received_event.set()

    subscription = MarketSubscription(channel="trade", symbols=["BTCUSDT"])
    context = await adapter.stream_market_data([subscription], callback)
    try:
        async with context:
            await asyncio.wait_for(received_event.wait(), timeout=1.0)
    finally:
        server.close()
        await server.wait_closed()
        await adapter.close()

    assert events and events[0]["stream"] == "btcusdt@trade"
    assert received_messages[0]["method"] == "SUBSCRIBE"
    assert "btcusdt@trade" in received_messages[0]["params"]
    if len(received_messages) > 1:
        assert received_messages[1]["method"] == "UNSUBSCRIBE"


@pytest.mark.asyncio
async def test_binance_websocket_reconnect(monkeypatch):
    connection_count = 0
    events: list[dict] = []
    ready = asyncio.Event()

    async def handler(websocket):
        nonlocal connection_count
        connection_count += 1
        await websocket.recv()
        await websocket.send(
            json.dumps(
                {
                    "stream": "btcusdt@trade",
                    "data": {"p": "100", "connection": connection_count},
                }
            )
        )
        if connection_count == 1:
            await websocket.close()
        else:
            await asyncio.sleep(0.1)

    server = await websockets.serve(handler, "localhost", 0)
    port = server.sockets[0].getsockname()[1]
    endpoint = f"ws://localhost:{port}"
    monkeypatch.setattr(binance_module, "_BINANCE_WS_ENDPOINT", endpoint)
    monkeypatch.setattr(binance_module, "_BINANCE_WS_INITIAL_BACKOFF", 0.05)
    monkeypatch.setattr(binance_module, "_BINANCE_WS_MAX_BACKOFF", 0.1)

    adapter = BinanceTestnetAdapter(http_client=RecordingHTTPClient([]))
    await adapter.connect()

    async def callback(payload):
        events.append(payload)
        if len(events) >= 2:
            ready.set()

    subscription = MarketSubscription(channel="trade", symbols=["BTCUSDT"])
    context = await adapter.stream_market_data([subscription], callback)
    try:
        async with context:
            await asyncio.wait_for(ready.wait(), timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()
        await adapter.close()

    assert connection_count >= 2
    assert len(events) >= 2
    assert {event["data"]["connection"] for event in events} >= {1, 2}


@pytest.mark.asyncio
async def test_kraken_websocket_subscription(monkeypatch):
    events: list[dict] = []
    received: list[dict] = []
    received_event = asyncio.Event()

    async def handler(websocket):
        subscribe = json.loads(await websocket.recv())
        received.append(subscribe)
        await websocket.send(json.dumps({"channel": "ticker", "data": {"price": "100"}}))
        try:
            unsub = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            received.append(json.loads(unsub))
        except asyncio.TimeoutError:
            pass

    server = await websockets.serve(handler, "localhost", 0)
    port = server.sockets[0].getsockname()[1]
    endpoint = f"ws://localhost:{port}"
    monkeypatch.setattr(kraken_module, "_KRAKEN_WS_ENDPOINT", endpoint)
    monkeypatch.setattr(kraken_module, "_KRAKEN_WS_INITIAL_BACKOFF", 0.05)
    monkeypatch.setattr(kraken_module, "_KRAKEN_WS_MAX_BACKOFF", 0.1)

    adapter = KrakenDemoAdapter(http_client=RecordingHTTPClient([]))
    await adapter.connect()

    async def callback(payload):
        events.append(payload)
        received_event.set()

    subscription = MarketSubscription(
        channel="ticker",
        symbols=["XBT/USD"],
        params={"subscription": {"interval": 2}},
    )
    context = await adapter.stream_market_data([subscription], callback)
    try:
        async with context:
            await asyncio.wait_for(received_event.wait(), timeout=1.0)
    finally:
        server.close()
        await server.wait_closed()
        await adapter.close()

    assert events and events[0]["channel"] == "ticker"
    assert received[0]["event"] == "subscribe"
    assert received[0]["pair"] == ["XBT/USD"]
    assert received[0]["subscription"]["name"] == "ticker"
    assert received[0]["subscription"]["interval"] == 2
    if len(received) > 1:
        assert received[1]["event"] == "unsubscribe"


@pytest.mark.asyncio
async def test_api_key_manager_rotation(tmp_path):
    encryption_key = Fernet.generate_key()
    cfg = await ConfigManager.create(config_path=str(tmp_path / "config.json"), encryption_key=encryption_key)
    manager = cfg.api_key_manager

    creds = ExchangeCredentials(api_key="demo", api_secret="secret", metadata={"environment": "demo"})
    record = manager.save_credentials("binance", "acct", creds)
    stored = json.loads((tmp_path / "api_keys_store.json").read_text())
    assert stored["records"][0]["data"]["api_key"] != "demo"

    rotated = manager.rotate_credentials(
        "binance",
        "acct",
        ExchangeCredentials(api_key="demo2", api_secret="secret2", metadata={"environment": "demo"}),
    )
    assert rotated.version == 2
    latest = manager.load_credentials("binance", "acct")
    assert latest.api_key == "demo2"

    with pytest.raises(ValueError):
        manager.save_credentials(
            "binance",
            "live",
            ExchangeCredentials(api_key="live", api_secret="live", metadata={"environment": "live"}),
        )

    manager.save_credentials(
        "binance",
        "live",
        ExchangeCredentials(api_key="live", api_secret="live", metadata={"environment": "live"}),
        compliance_ack=True,
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    removed = manager.purge_expired()
    assert removed >= 1


class StubAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.demo_mode = True
        self._orders: dict[str, OrderStatus] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:  # pragma: no cover - not used in tests
        return None

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        self.credentials = credentials

    async def fetch_market_data(self, symbol: str) -> dict[str, str]:  # pragma: no cover - not used
        return {"symbol": symbol}

    async def stream_market_data(self, subscriptions, callback):  # pragma: no cover - not used
        class _Dummy:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                return None

        return _Dummy()

    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        order_id = f"{self.name}-{len(self._orders) + 1}"
        status = OrderStatus(
            order_id=order_id,
            status="NEW",
            filled_quantity=0.0,
            remaining_quantity=order.quantity,
            average_price=None,
            raw={"exchange": self.name},
        )
        self._orders[order_id] = status
        return status

    async def fetch_order_status(self, order_id: str, *, symbol: str | None = None) -> OrderStatus:
        return self._orders[order_id]

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> OrderStatus:
        status = OrderStatus(
            order_id=order_id,
            status="CANCELED",
            filled_quantity=0.0,
            remaining_quantity=0.0,
            average_price=None,
            raw={"exchange": self.name},
        )
        self._orders[order_id] = status
        return status

    async def monitor_order(
        self,
        order_id: str,
        *,
        poll_interval: float = 0.0,
        symbol: str | None = None,
        timeout: float = 0.0,
    ) -> OrderStatus:
        status = self._orders[order_id]
        filled = OrderStatus(
            order_id=order_id,
            status="FILLED",
            filled_quantity=status.remaining_quantity,
            remaining_quantity=0.0,
            average_price=status.average_price,
            raw=status.raw,
        )
        self._orders[order_id] = filled
        return filled


@pytest.mark.asyncio
async def test_multi_account_round_robin():
    manager = MultiExchangeAccountManager()
    adapter_a = StubAdapter("binance")
    adapter_b = StubAdapter("kraken")
    manager.register_account(exchange="binance", account="a", adapter=adapter_a)
    manager.register_account(exchange="kraken", account="b", adapter=adapter_b)

    creds = {
        ("binance", "a"): ExchangeCredentials(api_key="a", api_secret="a"),
        ("kraken", "b"): ExchangeCredentials(api_key="b", api_secret="b"),
    }
    await manager.connect_all(creds)

    order = OrderRequest(symbol="BTCUSDT", side="buy", quantity=1.0)
    first = await manager.dispatch_order(order)
    second = await manager.dispatch_order(order)

    assert first.order_id.startswith("binance")
    assert second.order_id.startswith("kraken")

    status = await manager.fetch_order_status(first.order_id)
    assert status.status == "NEW"

    cancel_status = await manager.cancel_order(second.order_id)
    assert cancel_status.status == "CANCELED"

    summary = await manager.monitor_open_orders()
    assert summary[first.order_id].status == "FILLED"


@pytest.mark.asyncio
async def test_live_mode_requires_ack(monkeypatch):
    http = RecordingHTTPClient([])
    with pytest.raises(ValueError):
        BinanceTestnetAdapter(demo_mode=False, http_client=http)

    adapter = BinanceTestnetAdapter(demo_mode=False, http_client=http, compliance_ack=True)
    await adapter.connect()

