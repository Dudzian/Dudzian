import pytest

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.health import RetryPolicy, Watchdog
from bot_core.exchanges.kraken.margin import KrakenMarginAdapter


def _credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.TESTNET,
    )


def _watchdog() -> Watchdog:
    return Watchdog(retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0)), sleep=lambda _: None)


def test_fetch_account_snapshot_aggregates_trade_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_private(self, context):
        if context.path == "/0/private/Balance":
            return {"error": [], "result": {"ZUSD": "1000.0", "XXBT": "0.25"}}
        if context.path == "/0/private/TradeBalance":
            return {"error": [], "result": {"eb": "1500.0", "tb": "1400.0", "m": "120.0"}}
        raise AssertionError(f"unexpected path {context.path}")

    monkeypatch.setattr(KrakenMarginAdapter, "_private_request", fake_private)

    adapter = KrakenMarginAdapter(_credentials(), environment=Environment.TESTNET, watchdog=_watchdog())
    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["ZUSD"] == pytest.approx(1000.0)
    assert snapshot.balances["XXBT"] == pytest.approx(0.25)
    assert snapshot.total_equity == pytest.approx(1500.0)
    assert snapshot.available_margin == pytest.approx(1400.0)
    assert snapshot.maintenance_margin == pytest.approx(120.0)


def test_place_order_retries_on_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    responses = iter(
        [
            ExchangeThrottlingError("limit", status_code=429, payload=None),
            {"error": [], "result": {"txid": ["OID-1"]}},
        ]
    )

    def fake_private(self, context):
        calls.append(context.path)
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(KrakenMarginAdapter, "_private_request", fake_private)

    adapter = KrakenMarginAdapter(_credentials(), environment=Environment.TESTNET, watchdog=_watchdog())
    order = adapter.place_order(
        OrderRequest(symbol="BTC/USD", side="buy", quantity=0.1, order_type="market")
    )

    assert order.order_id == "OID-1"
    assert calls.count("/0/private/AddOrder") == 2


def test_place_order_maps_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_private(self, context):
        return {"error": ["EAPI:Invalid key"]}

    monkeypatch.setattr(KrakenMarginAdapter, "_private_request", fake_private)

    adapter = KrakenMarginAdapter(_credentials(), environment=Environment.TESTNET, watchdog=_watchdog())

    with pytest.raises(ExchangeAuthError):
        adapter.place_order(
            OrderRequest(symbol="BTC/USD", side="sell", quantity=0.1, order_type="limit", price=20000.0)
        )
