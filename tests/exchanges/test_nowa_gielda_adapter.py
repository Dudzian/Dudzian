from __future__ import annotations

import json
import time

import pytest
import responses

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter, symbols

_BASE_URL = "https://paper.nowa-gielda.example"


def _build_adapter() -> NowaGieldaSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        environment=Environment.PAPER,
    )
    return NowaGieldaSpotAdapter(credentials)


def test_symbol_mapping_roundtrip() -> None:
    assert symbols.to_exchange_symbol("BTC_USDT") == "BTC-USDT"
    assert symbols.to_internal_symbol("BTC-USDT") == "BTC_USDT"
    supported = tuple(symbols.supported_internal_symbols())
    assert "ETH_USDT" in supported


def test_sign_request_is_deterministic() -> None:
    adapter = _build_adapter()
    payload = {
        "symbol": "BTC-USDT",
        "type": "limit",
        "quantity": 1,
        "price": 25_000,
    }
    signature = adapter.sign_request(
        1_700_000_000_000,
        "POST",
        "/private/orders",
        body=payload,
    )
    assert signature == "7263373ca065b01dc73517607c92ca6505e420be70df6482221e68c539f9ab9d"


def test_sign_request_includes_query_params() -> None:
    adapter = _build_adapter()
    timestamp = 1_700_000_000_000
    base_signature = adapter.sign_request(
        timestamp,
        "DELETE",
        "/private/orders",
        params={"orderId": "sim-1"},
    )
    other_signature = adapter.sign_request(
        timestamp,
        "DELETE",
        "/private/orders",
        params={"orderId": "sim-2"},
    )

    assert base_signature != other_signature


def test_rate_limit_rules() -> None:
    adapter = _build_adapter()

    trading_rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert trading_rule is not None
    assert trading_rule.weight == 5
    assert trading_rule.max_requests == 5

    ticker_rule = adapter.rate_limit_rule("GET", "/public/ticker")
    assert ticker_rule is not None
    assert ticker_rule.weight == 1

    assert adapter.request_weight("GET", "/non-existent") == 1


@responses.activate
def test_fetch_ticker_validates_symbol_translation() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/public/ticker",
        json={
            "symbol": "BTC-USDT",
            "bestBid": "50000.5",
            "bestAsk": "50010.5",
            "lastPrice": "50005.1",
            "timestamp": 1_700_000_000_000,
        },
        match=[responses.matchers.query_param_matcher({"symbol": "BTC-USDT"})],
    )

    ticker = adapter.fetch_ticker("BTC_USDT")

    assert ticker == {
        "symbol": "BTC_USDT",
        "best_bid": 50000.5,
        "best_ask": 50010.5,
        "last_price": 50005.1,
        "timestamp": 1_700_000_000_000.0,
    }


@responses.activate
def test_fetch_ticker_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/public/ticker",
        json={
            "symbol": "ETH-USDT",
            "bestBid": "1800",
            "bestAsk": "1800.5",
            "lastPrice": "1800.25",
        },
        match=[responses.matchers.query_param_matcher({"symbol": "BTC-USDT"})],
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_ticker("BTC_USDT")


@responses.activate
def test_fetch_orderbook_translates_symbol() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/public/orderbook",
        json={
            "symbol": "BTC-USDT",
            "bids": [["50000", "1"]],
            "asks": [["50010", "2"]],
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "depth": "50",
                }
            )
        ],
    )

    orderbook = adapter.fetch_orderbook("BTC_USDT")

    assert orderbook["bids"][0][0] == "50000"
    assert orderbook["asks"][0][0] == "50010"


@responses.activate
def test_place_order_sends_signed_payload() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.POST,
        f"{_BASE_URL}/private/orders",
        json={
            "orderId": "sim-1",
            "status": "NEW",
            "filledQuantity": "0",
            "avgPrice": None,
        },
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="limit",
        price=25_000.0,
    )

    result = adapter.place_order(request)

    assert result.order_id == "sim-1"
    assert result.status == "NEW"
    assert result.filled_quantity == 0.0
    assert result.avg_price is None

    call = responses.calls[0]
    body = json.loads(call.request.body)
    assert body["symbol"] == "BTC-USDT"
    assert call.request.headers["X-API-KEY"] == "test-key"


@responses.activate
def test_place_order_maps_auth_errors() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.POST,
        f"{_BASE_URL}/private/orders",
        json={"code": "INVALID_SIGNATURE", "message": "signature error"},
        status=401,
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="limit",
    )

    with pytest.raises(ExchangeAuthError):
        adapter.place_order(request)


@responses.activate
def test_place_market_order_strips_null_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_000_123_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)
    responses.add(
        responses.POST,
        f"{_BASE_URL}/private/orders",
        json={
            "orderId": "sim-2",
            "status": "FILLED",
            "filledQuantity": "1",
            "avgPrice": "25010.0",
        },
    )

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=1.0,
        order_type="market",
    )

    adapter.place_order(request)

    call = responses.calls[0]
    body = json.loads(call.request.body)
    assert "price" not in body
    assert body == {
        "symbol": "BTC-USDT",
        "side": "buy",
        "type": "market",
        "quantity": 1.0,
    }
    expected_signature = adapter.sign_request(
        fixed_ts,
        "POST",
        "/private/orders",
        body=body,
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


def test_rate_limiter_blocks_excessive_requests() -> None:
    adapter = _build_adapter()
    rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert rule is not None

    # Wypełnij licznik limitem, a następnie spróbuj złożyć dodatkowe zlecenie.
    client = adapter._http_client  # type: ignore[attr-defined]
    client.rate_limiter.consume("POST", "/private/orders")

    request = OrderRequest(
        symbol="BTC_USDT",
        side="buy",
        quantity=0.1,
        order_type="market",
    )

    with pytest.raises(ExchangeThrottlingError):
        adapter.place_order(request)


@responses.activate
def test_cancel_order_translates_symbol_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_000_000_000)
    responses.add(
        responses.DELETE,
        f"{_BASE_URL}/private/orders",
        json={"status": "CANCELLED"},
        match=[
            responses.matchers.query_param_matcher(
                {
                    "orderId": "sim-1",
                    "symbol": "BTC-USDT",
                }
            )
        ],
    )

    adapter.cancel_order("sim-1", symbol="BTC_USDT")

    call = responses.calls[0]
    assert call.request.headers["X-API-KEY"] == "test-key"
    assert (
        call.request.headers["X-API-SIGN"]
        == "a629ae5ede954c14a17990e03653c5b2a8fb30c19ef2ffece6ff26d722be22b2"
    )


def test_rate_limiter_window_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    client = adapter._http_client  # type: ignore[attr-defined]
    rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert rule is not None

    client.rate_limiter.consume("POST", "/private/orders")

    fake_now = time.monotonic() + rule.window_seconds + 0.1

    monkeypatch.setattr(time, "monotonic", lambda: fake_now)

    # Po upłynięciu okna limit powinien zostać zresetowany.
    client.rate_limiter.consume("POST", "/private/orders")
