from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter
from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.okx.spot import OKXSpotAdapter


@dataclass
class _FakeOrder:
    id: str = "order-1"
    status: str = "open"
    filled: float = 0.0
    remaining: float = 0.5
    price: float | None = 10.5


class _FakeClient:
    def __init__(self) -> None:
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        self._orders: dict[str, _FakeOrder] = {}
        self._cancelled: list[tuple[str, str | None]] = []
        self._calls: dict[str, int] = {}

    def _hit(self, name: str) -> None:
        self._calls[name] = self._calls.get(name, 0) + 1

    def load_markets(self):
        self._hit("load_markets")
        return {symbol: {} for symbol in self.symbols}

    def fetch_balance(self):
        self._hit("fetch_balance")
        return {
            "free": {"BTC": 0.25, "USDT": 5000.0},
            "total": {"BTC": 0.3, "USDT": 5500.0},
            "used": {"BTC": 0.05, "USDT": 500.0},
        }

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        self._hit("fetch_ohlcv")
        base_timestamp = 1_600_000_000_000
        return [
            [base_timestamp, 10.0, 12.0, 9.0, 11.0, 42.0],
            [base_timestamp + 60_000, 11.0, 13.0, 10.5, 12.5, 43.0],
        ]

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self._hit("create_order")
        order = _FakeOrder(id=f"{symbol}-001", status="open", filled=amount / 2, remaining=amount / 2, price=price)
        self._orders[order.id] = order
        return {
            "id": order.id,
            "status": order.status,
            "filled": order.filled,
            "remaining": order.remaining,
            "price": order.price,
        }

    def cancel_order(self, order_id, symbol=None, params=None):
        self._hit("cancel_order")
        self._cancelled.append((order_id, symbol))
        return {"status": "cancelled"}


class _OfflineClient(_FakeClient):
    def __init__(self, error: type[Exception]) -> None:
        super().__init__()
        self._error = error

    def fetch_ohlcv(self, *args, **kwargs):  # noqa: D401 - dziedziczenie
        raise self._error("network down")


class _CustomNetworkError(Exception):
    pass


def _build_request(symbol: str = "BTC/USDT") -> OrderRequest:
    return OrderRequest(symbol=symbol, side="buy", quantity=1.0, order_type="limit", price=10.5)


def test_coinbase_adapter_basic_flow():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _FakeClient()
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    symbols = adapter.fetch_symbols()
    assert set(symbols) == {"BTC/USDT", "ETH/USDT"}

    snapshot = adapter.fetch_account_snapshot()
    assert math.isclose(snapshot.total_equity, 5500.3, rel_tol=1e-6)
    assert math.isclose(snapshot.available_margin, 5000.25, rel_tol=1e-6)
    assert math.isclose(snapshot.maintenance_margin, 500.05, rel_tol=1e-6)

    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=2_000_000_000_000, limit=2)
    assert len(candles) == 2

    order = adapter.place_order(_build_request())
    assert order.order_id.endswith("-001")
    assert order.status == "open"

    adapter.cancel_order(order.order_id, symbol="BTC/USDT")
    assert client._cancelled == [(order.order_id, "BTC/USDT")]


def test_adapter_translates_network_errors():
    credentials = ExchangeCredentials(key_id="k")
    client = _OfflineClient(_CustomNetworkError)
    adapter = BitfinexSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={"network_error_types": (_CustomNetworkError,)},
    )

    with pytest.raises(ExchangeNetworkError):
        adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=1, limit=1)


def test_okx_adapter_respects_offline_cancellation():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _FakeClient()
    adapter = OKXSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
        settings={"cancel_order_params": {"simulate": True}},
    )

    order = adapter.place_order(_build_request("ETH/USDT"))
    adapter.cancel_order(order.order_id, symbol="ETH/USDT")
    assert client._cancelled[-1] == (order.order_id, "ETH/USDT")

    with pytest.raises(NotImplementedError):
        adapter.stream_public_data(channels=["ticker"])

