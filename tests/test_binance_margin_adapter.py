import pytest

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.margin import BinanceMarginAdapter
from bot_core.exchanges.errors import ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.health import RetryPolicy, Watchdog


def _credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.TESTNET,
    )


def _watchdog() -> Watchdog:
    return Watchdog(retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0)), sleep=lambda _: None)


def test_fetch_account_snapshot_valuates_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed(self, path: str, method: str = "GET", params=None):
        assert path == "/sapi/v1/margin/account"
        return {
            "userAssets": [
                {"asset": "USDT", "free": "500", "locked": "100", "borrowed": "0"},
                {"asset": "BTC", "free": "0.1", "locked": "0.0", "borrowed": "0.01", "netAsset": "0.09"},
            ],
            "marginLevel": "3.5",
        }

    def fake_public(self, path: str, *, params=None, method="GET"):
        assert path == "/api/v3/ticker/price"
        return [
            {"symbol": "BTCUSDT", "price": "30000"},
            {"symbol": "ETHUSDT", "price": "2000"},
        ]

    monkeypatch.setattr(BinanceMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(BinanceMarginAdapter, "_public_request", fake_public)

    adapter = BinanceMarginAdapter(_credentials(), watchdog=_watchdog())
    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["USDT"] == pytest.approx(600.0)
    assert snapshot.balances["BTC"] == pytest.approx(0.09 + 0.01)
    assert snapshot.total_equity == pytest.approx(600.0 + 0.1 * 30000)
    assert snapshot.available_margin == pytest.approx(500.0 + 0.1 * 30000)
    assert snapshot.maintenance_margin == pytest.approx(3.5)


def test_place_order_retries_on_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    responses = iter(
        [
            ExchangeThrottlingError("rate limit", status_code=429, payload=None),
            {"orderId": 123, "status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0"},
        ]
    )

    def fake_signed(self, path: str, method: str = "GET", params=None):
        calls.append(path)
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(BinanceMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(BinanceMarginAdapter, "_public_request", lambda *args, **kwargs: [])

    adapter = BinanceMarginAdapter(_credentials(), watchdog=_watchdog())
    order = adapter.place_order(
        OrderRequest(symbol="BTC-USDT", side="buy", quantity=0.1, order_type="limit", price=20000.0)
    )

    assert order.order_id == "123"
    assert calls.count("/sapi/v1/margin/order") == 2


def test_place_order_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed(self, path: str, method: str = "GET", params=None):
        return {"code": -2015, "msg": "Invalid API-key"}

    monkeypatch.setattr(BinanceMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(BinanceMarginAdapter, "_public_request", lambda *args, **kwargs: [])

    adapter = BinanceMarginAdapter(_credentials(), watchdog=_watchdog())

    with pytest.raises(ExchangeAuthError):
        adapter.place_order(
            OrderRequest(symbol="BTC-USDT", side="buy", quantity=0.1, order_type="market")
        )
