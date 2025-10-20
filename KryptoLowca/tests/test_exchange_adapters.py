"""Testy adaptera CCXT oparte na pakiecie `bot_core`."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter, merge_adapter_settings
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)


class DummyClient:
    def __init__(self) -> None:
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        self.ohlcv_calls: list[tuple] = []
        self.create_order_calls: list[tuple] = []
        self.cancel_calls: list[tuple] = []
        self.balance = {
            "free": {"USDT": "900", "BTC": 0.4},
            "total": {"USDT": 1_100, "BTC": 0.4},
            "used": {"USDT": 200},
        }

    def load_markets(self) -> dict[str, dict[str, Any]]:
        return {symbol: {"precision": {"price": 2}} for symbol in self.symbols}

    def fetch_balance(self) -> dict[str, Any]:
        return self.balance

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        *,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> Sequence[Sequence[float]]:
        self.ohlcv_calls.append((symbol, interval, since, limit, params))
        return [[since or 0, 100.0, 101.0, 99.0, 100.5, 12.0]]

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.create_order_calls.append((symbol, order_type, side, amount, price, params))
        return {
            "id": "order-1",
            "status": "closed",
            "filled": amount,
            "average": price or 100.0,
        }

    def cancel_order(
        self,
        order_id: str,
        symbol: str | None,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.cancel_calls.append((order_id, symbol, params))


class DummyNetworkError(Exception):
    ...


class DummyAuthError(Exception):
    ...


class DummyRateLimitError(Exception):
    ...


class DummyBaseError(Exception):
    ...


@pytest.fixture()
def adapter() -> CCXTSpotAdapter:
    client = DummyClient()
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        environment=Environment.TESTNET,
    )
    return CCXTSpotAdapter(
        credentials,
        exchange_id="binance",
        client=client,
        settings={
            "fetch_ohlcv_params": {"weight": 2},
            "create_order_params": {"timeInForce": "IOC"},
            "cancel_order_params": {"recvWindow": 5000},
        },
    )


def test_merge_adapter_settings_merges_nested_payloads():
    defaults = {"a": 1, "nested": {"x": 1, "y": 2}}
    overrides = {"nested": {"y": 5, "z": 6}, "extra": True}

    merged = merge_adapter_settings(defaults, overrides)

    assert merged["a"] == 1
    assert merged["nested"] == {"x": 1, "y": 5, "z": 6}
    assert merged["extra"] is True


def test_fetch_symbols_returns_sorted_exchange_universe(adapter: CCXTSpotAdapter):
    symbols = adapter.fetch_symbols()
    assert symbols == ("BTC/USDT", "ETH/USDT")


def test_fetch_account_snapshot_normalizes_numbers(adapter: CCXTSpotAdapter):
    snapshot = adapter.fetch_account_snapshot()

    assert snapshot.balances["USDT"] == pytest.approx(900.0)
    assert snapshot.total_equity == pytest.approx(1_100.4)
    assert snapshot.available_margin == pytest.approx(900.4)
    assert snapshot.maintenance_margin == pytest.approx(200.0)


def test_fetch_ohlcv_forwards_params(adapter: CCXTSpotAdapter):
    client: DummyClient = adapter._client  # type: ignore[attr-defined]
    candles = adapter.fetch_ohlcv("BTC/USDT", "1h", start=1_700_000_000, end=1_700_001_000, limit=10)

    assert candles
    assert client.ohlcv_calls == [
        ("BTC/USDT", "1h", 1_700_000_000, 10, {"weight": 2, "until": 1_700_001_000})
    ]


def test_place_order_uses_request_payload(adapter: CCXTSpotAdapter):
    client: DummyClient = adapter._client  # type: ignore[attr-defined]
    request = OrderRequest(symbol="BTC/USDT", side="buy", quantity=0.25, order_type="limit", price=101.0)

    result = adapter.place_order(request)

    assert result.order_id == "order-1"
    assert result.avg_price == pytest.approx(101.0)
    assert client.create_order_calls == [
        ("BTC/USDT", "limit", "buy", 0.25, 101.0, {"timeInForce": "IOC"})
    ]


def test_cancel_order_applies_adapter_settings(adapter: CCXTSpotAdapter):
    client: DummyClient = adapter._client  # type: ignore[attr-defined]

    adapter.cancel_order("order-1", symbol="BTC/USDT")

    assert client.cancel_calls == [("order-1", "BTC/USDT", {"recvWindow": 5000})]


def test_wrap_call_translates_ccxt_errors():
    client = DummyClient()
    credentials = ExchangeCredentials(key_id="key", secret="secret")
    failing_adapter = CCXTSpotAdapter(
        credentials,
        exchange_id="binance",
        client=client,
        settings={
            "network_error_types": (DummyNetworkError,),
            "auth_error_types": (DummyAuthError,),
            "rate_limit_error_types": (DummyRateLimitError,),
            "base_error_types": (DummyBaseError,),
        },
    )

    def raise_network():
        raise DummyNetworkError("offline")

    client.fetch_balance = raise_network  # type: ignore[assignment]
    with pytest.raises(ExchangeNetworkError):
        failing_adapter.fetch_account_snapshot()

    def raise_auth():
        raise DummyAuthError("invalid")

    client.fetch_balance = raise_auth  # type: ignore[assignment]
    with pytest.raises(ExchangeAuthError):
        failing_adapter.fetch_account_snapshot()

    def raise_rate():
        raise DummyRateLimitError("429")

    client.fetch_balance = raise_rate  # type: ignore[assignment]
    with pytest.raises(ExchangeThrottlingError):
        failing_adapter.fetch_account_snapshot()

    def raise_base():
        raise DummyBaseError("oops")

    client.fetch_balance = raise_base  # type: ignore[assignment]
    with pytest.raises(ExchangeAPIError):
        failing_adapter.fetch_account_snapshot()


