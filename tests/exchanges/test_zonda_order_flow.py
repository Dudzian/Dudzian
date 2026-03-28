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


def test_zonda_spot_place_order_accepts_offer_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {
            "offer": {
                "offerId": "spot-offer-1",
                "status": "new",
                "amountFilled": "0.125",
                "averagePrice": "123500.0",
            }
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "spot-offer-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.125)
    assert result.avg_price == pytest.approx(123_500.0)


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


def test_zonda_margin_place_order_accepts_offer_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {
            "status": "Ok",
            "offer": {
                "offerId": "margin-offer-1",
                "status": "new",
                "amountFilled": "0.125",
                "averagePrice": "123500.0",
            },
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "margin-offer-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.125)
    assert result.avg_price == pytest.approx(123_500.0)


def test_zonda_margin_place_order_missing_order_shape_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Ok", "data": {"foo": "bar"}}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(RuntimeError):
        adapter.place_order(_build_request())


@pytest.mark.parametrize("adapter_cls", [ZondaSpotAdapter, ZondaMarginAdapter])
def test_zonda_place_order_accepts_top_level_order_fields_for_parity(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls,
) -> None:
    adapter = adapter_cls(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {
            "status": "Ok",
            "orderId": "top-level-1",
            "statusText": "ignored",
            "status": "new",
            "executed": "0.2",
            "price": "123456.0",
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "top-level-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.2)
    assert result.avg_price == pytest.approx(123_456.0)


@pytest.mark.parametrize("adapter_cls", [ZondaSpotAdapter, ZondaMarginAdapter])
@pytest.mark.parametrize(
    ("price_key", "price_value", "filled_key", "filled_value"),
    [
        ("avgPrice", "123456.0", "filledAmount", "0.3"),
        ("averagePrice", "123500.0", "amountFilled", "0.4"),
        ("price", "123400.0", "executed", "0.5"),
        ("price", "123300.0", "filled", "0.6"),
    ],
)
def test_zonda_place_order_field_alias_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls,
    price_key: str,
    price_value: str,
    filled_key: str,
    filled_value: str,
) -> None:
    adapter = adapter_cls(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {
            "status": "Ok",
            "order": {
                "id": "alias-1",
                "status": "new",
                filled_key: filled_value,
                price_key: price_value,
            },
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "alias-1"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(float(filled_value))
    assert result.avg_price == pytest.approx(float(price_value))


@pytest.mark.parametrize("adapter_cls", [ZondaSpotAdapter, ZondaMarginAdapter])
@pytest.mark.parametrize("order_id_payload", ["", None])
def test_zonda_place_order_requires_non_empty_order_id(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls,
    order_id_payload,
) -> None:
    adapter = adapter_cls(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Ok", "order": {"id": order_id_payload, "status": "new"}}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(RuntimeError):
        adapter.place_order(_build_request())


@pytest.mark.parametrize("adapter_cls", [ZondaSpotAdapter, ZondaMarginAdapter])
@pytest.mark.parametrize(
    ("price_fields", "filled_fields", "expected_price", "expected_filled"),
    [
        ({"avgPrice": "111.0", "averagePrice": "222.0"}, {"filledAmount": "1.1"}, 111.0, 1.1),
        ({"avgPrice": "333.0", "price": "444.0"}, {"filledAmount": "1.2"}, 333.0, 1.2),
        ({"avgPrice": "555.0"}, {"filledAmount": "1.3", "amountFilled": "9.9"}, 555.0, 1.3),
        ({"avgPrice": "666.0"}, {"filledAmount": "1.4", "executed": "8.8"}, 666.0, 1.4),
        ({"avgPrice": "777.0"}, {"filledAmount": "1.5", "filled": "7.7"}, 777.0, 7.7),
    ],
)
def test_zonda_place_order_alias_precedence_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls,
    price_fields: dict[str, str],
    filled_fields: dict[str, str],
    expected_price: float,
    expected_filled: float,
) -> None:
    adapter = adapter_cls(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {
            "status": "Ok",
            "order": {
                "id": "precedence-1",
                "status": "new",
                **price_fields,
                **filled_fields,
            },
        }

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    result = adapter.place_order(_build_request())

    assert result.order_id == "precedence-1"
    assert result.status == "NEW"
    assert result.avg_price == pytest.approx(expected_price)
    assert result.filled_quantity == pytest.approx(expected_filled)


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
        return {"status": "Ok", "order": {"id": "margin-1", "status": "cancelled"}}

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


def test_zonda_margin_cancel_order_ok_without_final_status_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ZondaMarginAdapter(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Ok"}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(RuntimeError):
        adapter.cancel_order("margin-1", symbol="BTC-PLN")


@pytest.mark.parametrize("adapter_cls", [ZondaSpotAdapter, ZondaMarginAdapter])
def test_zonda_cancel_order_ok_but_open_status_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls,
) -> None:
    adapter = adapter_cls(
        ExchangeCredentials(key_id="k", secret="s", permissions=("trade",)),
        environment=Environment.PAPER,
        watchdog=_ImmediateWatchdog(),
    )

    def _fake_signed_request(_method: str, _path: str, **_kwargs):
        return {"status": "Ok", "order": {"id": "any-1", "status": "open"}}

    monkeypatch.setattr(adapter, "_signed_request", _fake_signed_request)

    with pytest.raises(RuntimeError):
        adapter.cancel_order("any-1", symbol="BTC-PLN")


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
