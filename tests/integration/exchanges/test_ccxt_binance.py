from __future__ import annotations

from typing import Any, Callable

import pytest

import ccxt

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter
from tests.integration.exchanges.helpers import make_order_request


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_ccxt_spot_adapter_operates_on_binance_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    if getattr(ccxt.binance, "__module__", "") == "ccxt":
        pytest.skip("Środowisko testowe korzysta z fallbackowego stubu ccxt.")

    created_clients: list[Any] = []
    constructor: Callable[[dict[str, Any]], Any] = ccxt.binance

    def _build_stub_client(options: dict[str, Any]) -> Any:
        client = constructor(options)
        sandbox_calls: list[bool] = []
        original_set_sandbox = getattr(client, "set_sandbox_mode", None)

        if callable(original_set_sandbox):
            def _set_sandbox(enabled: bool) -> None:
                sandbox_calls.append(bool(enabled))
                original_set_sandbox(enabled)

            client.set_sandbox_mode = _set_sandbox  # type: ignore[attr-defined]
        else:
            client.set_sandbox_mode = sandbox_calls.append  # type: ignore[attr-defined,assignment]

        client._sandbox_calls = sandbox_calls  # type: ignore[attr-defined]
        client.load_markets = lambda: {"BTC/USDT": {"precision": {"price": 2, "amount": 6}}}
        client.fetch_balance = lambda: {
            "free": {"USDT": 900.0},
            "total": {"USDT": 1000.0},
            "used": {"USDT": 100.0},
        }

        def _fake_fetch_ohlcv(symbol: str, timeframe: str, since=None, limit=None, params=None):
            return [[1_700_000_000, 100.0, 110.0, 90.0, 105.0, 5.0]]

        client.fetch_ohlcv = _fake_fetch_ohlcv

        def _fake_create_order(symbol, order_type, side, amount, price, params=None):
            return {
                "id": "order-1",
                "status": "closed",
                "filled": amount,
                "price": price or 105.0,
            }

        client.create_order = _fake_create_order
        created_clients.append(client)
        return client

    monkeypatch.setattr(ccxt, "binance", _build_stub_client)

    credentials = ExchangeCredentials(
        key_id="demo",
        secret="secret",
        environment=Environment.TESTNET,
        permissions=("trade", "read"),
    )
    adapter = CCXTSpotAdapter(credentials, exchange_id="binance")
    adapter.configure_network(ip_allowlist=())

    assert created_clients, "Adapter powinien utworzyć klienta CCXT"
    client = created_clients[0]

    symbols = adapter.fetch_symbols()
    snapshot = adapter.fetch_account_snapshot()
    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", limit=1)
    request = make_order_request()
    order = adapter.place_order(request)

    assert client._sandbox_calls == [True]  # type: ignore[attr-defined]
    assert symbols == ("BTC/USDT",)
    assert snapshot.balances["USDT"] == pytest.approx(900.0)
    assert snapshot.total_equity == pytest.approx(1000.0)
    assert candles == [[1_700_000_000, 100.0, 110.0, 90.0, 105.0, 5.0]]
    assert order.order_id == "order-1"
    assert order.status == "closed"
    assert order.filled_quantity == pytest.approx(request.quantity)
    assert order.avg_price == pytest.approx(105.0)
