from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pytest
from cryptography.fernet import Fernet

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.exchanges import (
    BinanceTestnetAdapter,
    ExchangeCredentials,
    MarketSubscription,
    KrakenDemoAdapter,
    OrderRequest,
    OrderStatus,
    ZondaAdapter,
)
import KryptoLowca.exchanges.binance as binance_module
import KryptoLowca.exchanges.kraken as kraken_module
import KryptoLowca.exchanges.zonda as zonda_module
from KryptoLowca.managers.multi_account_manager import MultiExchangeAccountManager

from bot_core.execution import ExecutionContext, LiveExecutionRouter
from bot_core.execution.live_router import RouteDefinition
from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter as CoreExchangeAdapter,
    ExchangeCredentials as CoreExchangeCredentials,
    OrderRequest as CoreOrderRequest,
    OrderResult as CoreOrderResult,
)


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

    adapter = ZondaAdapter(http_client=RecordingHTTPClient([]))

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


class StubLiveAdapter(CoreExchangeAdapter):
    """Minimalny adapter zgodny z bot_core do testów routera live."""

    class _Stream:
        async def __aenter__(self):  # pragma: no cover - prosty kontekst
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - prosty kontekst
            return False

    def __init__(self, name: str) -> None:
        super().__init__(CoreExchangeCredentials(key_id=f"{name}-key"))
        self.name = name
        self._orders: dict[str, CoreOrderResult] = {}
        self._counter = 0

    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - brak logiki
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:  # pragma: no cover - brak logiki
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self) -> Sequence[str]:  # pragma: no cover - brak logiki
        return ("BTCUSDT",)

    def fetch_ohlcv(self, symbol: str, interval: str, start=None, end=None, limit=None):  # pragma: no cover
        return []

    def place_order(self, request: CoreOrderRequest) -> CoreOrderResult:
        self._counter += 1
        order_id = f"{self.name}-{self._counter}"
        result = CoreOrderResult(
            order_id=order_id,
            status="NEW",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"exchange": self.name},
        )
        self._orders[order_id] = result
        return result

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        self._orders.pop(order_id, None)

    def stream_public_data(self, *, channels):  # pragma: no cover - brak logiki
        return self._Stream()

    def stream_private_data(self, *, channels):  # pragma: no cover - brak logiki
        return self._Stream()


@pytest.mark.asyncio
async def test_multi_account_round_robin():
    adapters = {
        "binance": StubLiveAdapter("binance"),
        "kraken": StubLiveAdapter("kraken"),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=(
            RouteDefinition(name="binance_route", exchanges=("binance",)),
            RouteDefinition(name="kraken_route", exchanges=("kraken",)),
        ),
        default_route="binance_route",
    )
    context = ExecutionContext(
        portfolio_id="default",
        risk_profile="balanced",
        environment="live",
        metadata={},
    )
    manager = MultiExchangeAccountManager(router, base_context=context)
    manager.register_account(exchange="binance", account="a", execution_route="binance_route")
    manager.register_account(exchange="kraken", account="b", execution_route="kraken_route")

    order = CoreOrderRequest(symbol="BTCUSDT", side="BUY", quantity=1.0, order_type="MARKET")
    first = await manager.dispatch_order(order)
    second = await manager.dispatch_order(order)

    assert first.order_id.startswith("binance")
    assert second.order_id.startswith("kraken")
    assert set(manager.supported_exchanges) == {"binance", "kraken"}

    await manager.cancel_order(first.order_id)


def test_multi_account_supported_exchanges():
    adapters = {"binance": StubLiveAdapter("binance")}
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=(RouteDefinition(name="binance", exchanges=("binance",)),),
        default_route="binance",
    )
    context = ExecutionContext(
        portfolio_id="demo",
        risk_profile="paper",
        environment="testnet",
        metadata={},
    )
    manager = MultiExchangeAccountManager(router, base_context=context)
    manager.register_account(exchange="binance", account="primary")

    assert manager.supported_exchanges == ("binance",)


@pytest.mark.asyncio
async def test_live_mode_requires_ack(monkeypatch):
    http = RecordingHTTPClient([])
    with pytest.raises(ValueError):
        BinanceTestnetAdapter(demo_mode=False, http_client=http)

    adapter = BinanceTestnetAdapter(demo_mode=False, http_client=http, compliance_ack=True)
    await adapter.connect()
