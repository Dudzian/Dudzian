import pytest

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.health import RetryPolicy, Watchdog
from bot_core.exchanges.zonda.margin import ZondaMarginAdapter


def _credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )


def _watchdog() -> Watchdog:
    return Watchdog(retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=(0.0, 0.0)), sleep=lambda _: None)


def test_fetch_account_snapshot_handles_high_volatility(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed(self, method: str, path: str, *, params=None, data=None):
        assert path == "/trading/margin/balance"
        return {
            "status": "Ok",
            "balances": [
                {"currency": "BTC", "available": "0.5", "locked": "0.1", "borrowed": "0.05"},
                {"currency": "PLN", "available": "1000", "locked": "0", "borrowed": "0"},
            ],
            "requiredMargin": "1500",
        }

    def fake_public(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/ticker"
        return {
            "status": "Ok",
            "items": {
                "BTC-PLN": {"rate": "160000"},
                "ETH-PLN": {"rate": "9000"},
            },
        }

    monkeypatch.setattr(ZondaMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(ZondaMarginAdapter, "_public_request", fake_public)

    adapter = ZondaMarginAdapter(_credentials(), environment=Environment.LIVE, watchdog=_watchdog())
    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.total_equity == pytest.approx((0.5 + 0.1 + 0.05) * 160000 + 1000)
    assert snapshot.available_margin == pytest.approx(0.5 * 160000 + 1000)
    assert snapshot.maintenance_margin == pytest.approx(1500.0)


def test_place_order_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            ExchangeThrottlingError("limit", status_code=429, payload=None),
            {"status": "Ok", "order": {"id": "123", "status": "new", "filledAmount": "0", "avgPrice": "0"}},
        ]
    )

    def fake_signed(self, method: str, path: str, *, params=None, data=None):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ZondaMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(ZondaMarginAdapter, "_public_request", lambda *args, **kwargs: {"status": "Ok", "items": {}})

    adapter = ZondaMarginAdapter(_credentials(), environment=Environment.LIVE, watchdog=_watchdog())
    result = adapter.place_order(
        OrderRequest(symbol="BTC-PLN", side="sell", quantity=0.1, order_type="limit", price=150000.0)
    )

    assert result.order_id == "123"


def test_cancel_order_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed(self, method: str, path: str, *, params=None, data=None):
        return {"status": "Fail", "errors": [{"code": 4002, "message": "Invalid signature"}]}

    monkeypatch.setattr(ZondaMarginAdapter, "_signed_request", fake_signed)
    monkeypatch.setattr(ZondaMarginAdapter, "_public_request", lambda *args, **kwargs: {"status": "Ok", "items": {}})

    adapter = ZondaMarginAdapter(_credentials(), environment=Environment.LIVE, watchdog=_watchdog())

    with pytest.raises(ExchangeAuthError):
        adapter.cancel_order("OID-123")
