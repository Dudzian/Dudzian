"""Integracje REST + long-poll dla adapterów Deribit/BitMEX."""

from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.bitmex import BitmexFuturesAdapter, BitmexSpotAdapter
from bot_core.exchanges.deribit import DeribitFuturesAdapter, DeribitSpotAdapter
from bot_core.exchanges.base import OrderRequest
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.streaming import LocalLongPollStream


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def execute(self, operation: str, func):  # pragma: no cover - uproszczony watchdog
        self.operations.append(operation)
        return func()


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def load_markets(self):  # pragma: no cover - wymagane przez bazę CCXT
        return {}

    def fetch_order_book(self, symbol: str, limit=None, params=None):
        self.calls.append(("fetch_order_book", (symbol, limit), {"params": params}))
        return {"symbol": symbol, "bids": [[1.0, 2.0]]}

    def fetch_ticker(self, symbol: str, params=None):
        self.calls.append(("fetch_ticker", (symbol,), {"params": params}))
        return {"symbol": symbol, "last": 42.0}

    def fetch_my_trades(self, symbol: str, since=None, limit=None, params=None):
        self.calls.append(("fetch_my_trades", (symbol, since, limit), {"params": params}))
        return [{"symbol": symbol, "amount": 0.1}]


class _TransientError(RuntimeError):
    pass


class _MutationClient(_RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.create_attempts = 0
        self.cancel_attempts = 0
        self.ticker_attempts = 0

    def create_order(
        self, symbol: str, order_type: str, side: str, amount: float, price=None, params=None
    ):
        self.create_attempts += 1
        raise _TransientError("temporary failure during create")

    def cancel_order(self, order_id: str, symbol=None, params=None):
        self.cancel_attempts += 1
        raise _TransientError("temporary failure during cancel")

    def fetch_ticker(self, symbol: str, params=None):
        self.ticker_attempts += 1
        if self.ticker_attempts < 3:
            raise _TransientError("temporary failure during read")
        return {"symbol": symbol, "last": 100.0}


def _creds(secret: str | None = "s") -> ExchangeCredentials:
    return ExchangeCredentials(key_id="k", secret=secret, environment=Environment.LIVE)


@pytest.mark.parametrize(
    "adapter_cls, env, base",
    [
        (DeribitSpotAdapter, Environment.TESTNET, "https://stream.sandbox.dudzian.ai/exchanges"),
        (DeribitSpotAdapter, Environment.LIVE, "https://stream.hyperion.dudzian.ai/exchanges"),
        (BitmexSpotAdapter, Environment.TESTNET, "https://stream.sandbox.dudzian.ai/exchanges"),
        (BitmexSpotAdapter, Environment.LIVE, "https://stream.hyperion.dudzian.ai/exchanges"),
    ],
)
def test_spot_stream_helpers_pick_environment_base(adapter_cls, env, base):
    adapter = adapter_cls(_creds(), environment=env, client=_RecordingClient())

    book_stream = adapter.stream_order_book("BTC/USDT", depth=25)
    ticker_stream = adapter.stream_ticker("BTC/USDT")
    fills_stream = adapter.stream_fills("BTC/USDT")

    assert isinstance(book_stream, LocalLongPollStream)
    assert book_stream._base_url == base  # noqa: SLF001
    assert ticker_stream._base_url == base  # noqa: SLF001
    assert fills_stream._base_url == base  # noqa: SLF001
    assert "order_book:BTC/USDT:25" in book_stream._channels  # noqa: SLF001
    assert "ticker:BTC/USDT" in ticker_stream._channels  # noqa: SLF001
    assert "fills:BTC/USDT" in fills_stream._channels  # noqa: SLF001


@pytest.mark.parametrize("adapter_cls", [DeribitFuturesAdapter, BitmexFuturesAdapter])
def test_futures_rest_helpers_and_stream_channels(adapter_cls):
    watchdog = _RecordingWatchdog()
    client = _RecordingClient()
    adapter = adapter_cls(
        _creds(),
        environment=Environment.LIVE,
        client=client,
        watchdog=watchdog,
    )
    adapter.configure_network(ip_allowlist=())

    book = adapter.fetch_order_book("ETH/USDT", limit=10)
    ticker = adapter.fetch_ticker("ETH/USDT")
    fills = adapter.fetch_my_trades("ETH/USDT", limit=5, since=123)

    assert book["symbol"] == "ETH/USDT"
    assert ticker["last"] == 42.0
    assert fills and fills[0]["symbol"] == "ETH/USDT"
    assert any(op.endswith("fetch_order_book") for op in watchdog.operations)

    stream = adapter.stream_fills("ETH/USDT")
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url.endswith("/exchanges")  # noqa: SLF001
    assert "fills:ETH/USDT" in stream._channels  # noqa: SLF001


@pytest.mark.parametrize(
    "adapter_cls",
    [DeribitFuturesAdapter, BitmexFuturesAdapter],
)
def test_custom_stream_override_respected(adapter_cls):
    adapter = adapter_cls(
        _creds(),
        environment=Environment.LIVE,
        client=_RecordingClient(),
        settings={"stream": {"base_url": "http://192.0.2.10:9000", "poll_interval": 1.0}},
    )

    stream = adapter.stream_order_book("BTC/USDT", depth=10)
    assert isinstance(stream, LocalLongPollStream)
    assert stream._base_url == "http://192.0.2.10:9000"  # noqa: SLF001
    assert stream._poll_interval == pytest.approx(1.0)  # noqa: SLF001
    assert "order_book:BTC/USDT:10" in stream._channels  # noqa: SLF001


@pytest.mark.parametrize("adapter_cls", [DeribitFuturesAdapter, BitmexFuturesAdapter])
def test_mutations_disable_retry_explosion(adapter_cls):
    watchdog = _RecordingWatchdog()
    client = _MutationClient()
    adapter = adapter_cls(
        _creds(),
        environment=Environment.LIVE,
        client=client,
        watchdog=watchdog,
        settings={
            "network_error_types": (_TransientError,),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    request = OrderRequest(
        symbol="BTC/USDT", side="buy", quantity=1.0, order_type="limit", price=10.0
    )

    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(request)
    with pytest.raises(ExchangeNetworkError):
        adapter.cancel_order("order-1", symbol="BTC/USDT")

    assert client.create_attempts == 1
    assert client.cancel_attempts == 1
    assert not any(op.endswith("create_order") for op in watchdog.operations)
    assert not any(op.endswith("cancel_order") for op in watchdog.operations)


@pytest.mark.parametrize("adapter_cls", [DeribitFuturesAdapter, BitmexFuturesAdapter])
def test_reads_still_retry_when_enabled(adapter_cls):
    watchdog = _RecordingWatchdog()
    client = _MutationClient()
    adapter = adapter_cls(
        _creds(),
        environment=Environment.LIVE,
        client=client,
        watchdog=watchdog,
        settings={
            "network_error_types": (_TransientError,),
            "retry_policy": {
                "max_attempts": 3,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    ticker = adapter.fetch_ticker("BTC/USDT")

    assert ticker["last"] == pytest.approx(100.0)
    assert client.ticker_attempts == 3
    assert any(op.endswith("fetch_ticker") for op in watchdog.operations)
