from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import ExchangeAPIError
from bot_core.exchanges.zonda.margin import ZondaMarginAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter


class _ImmediateWatchdog:
    def execute(self, _operation: str, func):
        return func()


def _build_request() -> OrderRequest:
    return OrderRequest(
        symbol="BTC-PLN",
        side="buy",
        quantity=0.5,
        order_type="limit",
        price=123_456.0,
        client_order_id="cli-zonda-1",
    )


def test_zonda_spot_place_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )
    captured: dict[str, object] = {}

    def _fake_signed_request(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["data"] = kwargs.get("data")
        return {
            "order": {
                "id": "spot-1",
                "status": "new",
                "filledAmount": "0.25",
                "avgPrice": "123456.0",
            }
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "spot-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.25)
    assert result.avg_price == pytest.approx(123_456.0)
    assert captured["method"] == "POST"
    assert captured["path"] == "/trading/offer"
    assert isinstance(captured["data"], dict)
    assert captured["data"]["clientOrderId"] == "cli-zonda-1"


def test_zonda_spot_place_order_failure_propagates_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fail_signed_request(_method: str, _path: str, **_kwargs):
        raise ExchangeAPIError("spot create failed", status_code=503, payload={"error": "upstream"})

    monkeypatch.setattr(adapter, "_signed_request", _fail_signed_request)

    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.place_order(_build_request())

    assert exc_info.value.status_code == 503


def test_zonda_spot_cancel_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"order": {"id": "spot-1", "status": "cancelled"}}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    adapter.cancel_order("spot-1", symbol="BTC-PLN")


def test_zonda_spot_cancel_order_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"order": {"id": "spot-1", "status": "open"}}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(RuntimeError):
        adapter.cancel_order("spot-1", symbol="BTC-PLN")


def test_zonda_spot_place_order_requires_trade_permission() -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=()),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    with pytest.raises(PermissionError):
        adapter.place_order(_build_request())


def test_zonda_margin_place_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )
    captured: dict[str, object] = {}

    def _fake_signed_request(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["data"] = kwargs.get("data")
        return {
            "status": "Ok",
            "order": {
                "id": "margin-1",
                "status": "new",
                "filledAmount": "0.10",
                "avgPrice": "123456.0",
            },
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "margin-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.1)
    assert result.avg_price == pytest.approx(123_456.0)
    assert captured["method"] == "POST"
    assert captured["path"] == "/trading/margin/offer"
    assert isinstance(captured["data"], dict)
    assert captured["data"]["clientOrderId"] == "cli-zonda-1"


def test_zonda_margin_place_order_failure_fail_status_raises_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Fail", "errors": [{"message": "margin create rejected"}]}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(ExchangeAPIError):
        adapter.place_order(_build_request())


def test_zonda_margin_cancel_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Ok"}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    adapter.cancel_order("margin-1", symbol="BTC-PLN")


def test_zonda_margin_cancel_order_failure_fail_status_raises_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Fail", "errors": [{"message": "margin cancel rejected"}]}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(ExchangeAPIError):
        adapter.cancel_order("margin-1", symbol="BTC-PLN")


def test_zonda_margin_place_order_requires_trade_permission() -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=()),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    with pytest.raises(PermissionError):
        adapter.place_order(_build_request())


def test_zonda_margin_cancel_order_requires_trade_permission() -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=()),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    with pytest.raises(PermissionError):
        adapter.cancel_order("margin-1", symbol="BTC-PLN")
