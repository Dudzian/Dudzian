from __future__ import annotations

import json
import time

import pytest
import responses

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
)
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

    account_rule = adapter.rate_limit_rule("GET", "/private/account")
    assert account_rule is not None
    assert account_rule.weight == 2

    ticker_rule = adapter.rate_limit_rule("GET", "/public/ticker")
    assert ticker_rule is not None
    assert ticker_rule.weight == 1

    ohlcv_rule = adapter.rate_limit_rule("GET", "/public/ohlcv")
    assert ohlcv_rule is not None
    assert ohlcv_rule.weight == 2

    trades_rule = adapter.rate_limit_rule("GET", "/private/trades")
    assert trades_rule is not None
    assert trades_rule.weight == 3

    open_orders_rule = adapter.rate_limit_rule("GET", "/private/orders")
    assert open_orders_rule is not None
    assert open_orders_rule.weight == 3

    closed_orders_rule = adapter.rate_limit_rule("GET", "/private/orders/history")
    assert closed_orders_rule is not None
    assert closed_orders_rule.weight == 3

    deposits_rule = adapter.rate_limit_rule("GET", "/private/deposits")
    assert deposits_rule is not None
    assert deposits_rule.weight == 2

    withdrawals_rule = adapter.rate_limit_rule("GET", "/private/withdrawals")
    assert withdrawals_rule is not None
    assert withdrawals_rule.weight == 3

    fees_rule = adapter.rate_limit_rule("GET", "/private/fees")
    assert fees_rule is not None
    assert fees_rule.weight == 2

    rebates_rule = adapter.rate_limit_rule("GET", "/private/rebates")
    assert rebates_rule is not None
    assert rebates_rule.weight == 2

    interest_rule = adapter.rate_limit_rule("GET", "/private/interest")
    assert interest_rule is not None
    assert interest_rule.weight == 2

    transfers_rule = adapter.rate_limit_rule("GET", "/private/transfers")
    assert transfers_rule is not None
    assert transfers_rule.weight == 2

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


@responses.activate
def test_fetch_account_snapshot_parses_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_000_555_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)
    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/account",
        json={
            "balances": [
                {"asset": "USDT", "total": "1000.5"},
                {"asset": "BTC", "total": "0.01"},
            ],
            "totalEquity": "1000.51",
            "availableMargin": "900.0",
            "maintenanceMargin": "50.0",
        },
    )

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances == {"USDT": 1000.5, "BTC": 0.01}
    assert snapshot.total_equity == 1000.51
    assert snapshot.available_margin == 900.0
    assert snapshot.maintenance_margin == 50.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(fixed_ts, "GET", "/private/account")
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_account_snapshot_rejects_invalid_balance() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/account",
        json={"balances": [{"asset": "USDT", "total": "not-a-number"}]},
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_account_snapshot()


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


@responses.activate
def test_fetch_open_orders_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_000_777_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders",
        json={
            "orders": [
                {
                    "orderId": "sim-1",
                    "symbol": "BTC-USDT",
                    "status": "OPEN",
                    "side": "buy",
                    "type": "limit",
                    "price": "25000.0",
                    "avgPrice": None,
                    "quantity": "1.5",
                    "filledQuantity": "0.5",
                    "timestamp": 1_700_000_000_000,
                }
            ]
        },
    )

    orders = adapter.fetch_open_orders()

    assert len(orders) == 1
    order = orders[0]
    assert order["order_id"] == "sim-1"
    assert order["symbol"] == "BTC_USDT"
    assert order["quantity"] == 1.5
    assert order["filled_quantity"] == 0.5
    assert order["price"] == 25000.0
    assert order["avg_price"] is None
    assert order["status"] == "OPEN"

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/orders",
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_open_orders_supports_symbol_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_001_111_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders",
        json={
            "orders": [
                {
                    "orderId": "sim-2",
                    "symbol": "BTC-USDT",
                    "status": "NEW",
                    "side": "sell",
                    "type": "limit",
                    "price": "26000.0",
                    "quantity": "0.25",
                    "filledQuantity": "0.0",
                    "timestamp": 1_700_001_000_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "limit": "50",
                }
            )
        ],
    )

    orders = adapter.fetch_open_orders(symbol="BTC_USDT", limit=50)

    assert len(orders) == 1
    assert orders[0]["side"] == "sell"
    assert orders[0]["type"] == "limit"

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/orders",
        params={"symbol": "BTC-USDT", "limit": 50},
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_open_orders_rejects_invalid_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_002_222_000)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders",
        json={"orders": ["not-a-mapping"]},
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_open_orders()


@responses.activate
def test_fetch_open_orders_rejects_symbol_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_003_333_000)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders",
        json={
            "orders": [
                {
                    "orderId": "sim-3",
                    "symbol": "ETH-USDT",
                    "status": "OPEN",
                    "side": "buy",
                    "type": "limit",
                    "price": "1800.0",
                    "quantity": "0.5",
                    "filledQuantity": "0.0",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_open_orders(symbol="BTC_USDT")


@responses.activate
def test_fetch_closed_orders_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_004_444_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders/history",
        json={
            "orders": [
                {
                    "orderId": "hist-1",
                    "symbol": "BTC-USDT",
                    "status": "FILLED",
                    "side": "sell",
                    "type": "limit",
                    "price": "27500.0",
                    "avgPrice": "27510.0",
                    "quantity": "0.4",
                    "executedQuantity": "0.4",
                    "timestamp": 1_700_003_000_000,
                    "closedAt": 1_700_003_500_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "limit": "20",
                    "start": "1699999000000",
                    "end": "1700003600000",
                }
            )
        ],
    )

    orders = adapter.fetch_closed_orders(
        symbol="BTC_USDT",
        limit=20,
        start=1_699_999_000_000,
        end=1_700_003_600_000,
    )

    assert len(orders) == 1
    order = orders[0]
    assert order["order_id"] == "hist-1"
    assert order["symbol"] == "BTC_USDT"
    assert order["status"] == "FILLED"
    assert order["side"] == "sell"
    assert order["type"] == "limit"
    assert order["quantity"] == 0.4
    assert order["filled_quantity"] == 0.4
    assert order["price"] == 27500.0
    assert order["avg_price"] == 27510.0
    assert order["timestamp"] == 1_700_003_000_000.0
    assert order["closed_timestamp"] == 1_700_003_500_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/orders/history",
        params={
            "symbol": "BTC-USDT",
            "limit": 20,
            "start": 1_699_999_000_000,
            "end": 1_700_003_600_000,
        },
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_closed_orders_rejects_symbol_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_005_555_000)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders/history",
        json={
            "orders": [
                {
                    "orderId": "hist-2",
                    "symbol": "ETH-USDT",
                    "status": "FILLED",
                    "side": "buy",
                    "type": "market",
                    "price": "1805.0",
                    "quantity": "1.0",
                    "filledQuantity": "1.0",
                    "timestamp": 1_700_004_000_000,
                    "closedAt": 1_700_004_100_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_closed_orders(symbol="BTC_USDT")


@responses.activate
def test_fetch_closed_orders_rejects_invalid_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    monkeypatch.setattr(adapter, "_timestamp", lambda: 1_700_006_666_000)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/orders/history",
        json={"orders": ["not-a-mapping"]},
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_closed_orders()


@responses.activate
def test_fetch_deposits_history_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_100_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/deposits",
        json={
            "deposits": [
                {
                    "depositId": "dep-1",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "0.5",
                    "fee": "0.0005",
                    "network": "BTC",
                    "txId": "0xabc",
                    "timestamp": 1_700_099_900_000,
                    "completedAt": 1_700_100_100_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "status": "completed",
                    "start": "1699999999000",
                    "end": "1700100000000",
                    "limit": "50",
                }
            )
        ],
    )

    deposits = adapter.fetch_deposits_history(
        symbol="BTC_USDT",
        status="completed",
        start=1_699_999_999_000,
        end=1_700_100_000_000,
        limit=50,
    )

    assert len(deposits) == 1
    deposit = deposits[0]
    assert deposit["transfer_id"] == "dep-1"
    assert deposit["symbol"] == "BTC_USDT"
    assert deposit["status"] == "COMPLETED"
    assert deposit["amount"] == 0.5
    assert deposit["fee"] == 0.0005
    assert deposit["network"] == "BTC"
    assert deposit["tx_id"] == "0xabc"
    assert deposit["timestamp"] == 1_700_099_900_000.0
    assert deposit["completed_timestamp"] == 1_700_100_100_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/deposits",
        params={
            "symbol": "BTC-USDT",
            "status": "completed",
            "start": 1_699_999_999_000,
            "end": 1_700_100_000_000,
            "limit": 50,
        },
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_deposits_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/deposits",
        json={
            "deposits": [
                {
                    "depositId": "dep-2",
                    "symbol": "ETH-USDT",
                    "status": "PENDING",
                    "amount": "1.0",
                    "timestamp": 1_700_100_200_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_deposits_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_deposits_history_rejects_invalid_amount() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/deposits",
        json={
            "deposits": [
                {
                    "depositId": "dep-3",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "not-a-number",
                    "timestamp": 1_700_101_000_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_deposits_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_withdrawals_history_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_200_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/withdrawals",
        json={
            "withdrawals": [
                {
                    "withdrawalId": "wd-1",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "0.3",
                    "fee": "0.0004",
                    "network": "BTC",
                    "address": "1BitcoinAddr",
                    "tag": None,
                    "txId": "0xdef",
                    "timestamp": 1_700_199_900_000,
                    "completedAt": 1_700_200_100_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "status": "completed",
                    "start": "1699998888000",
                    "end": "1700200000000",
                    "limit": "25",
                }
            )
        ],
    )

    withdrawals = adapter.fetch_withdrawals_history(
        symbol="BTC_USDT",
        status="completed",
        start=1_699_998_888_000,
        end=1_700_200_000_000,
        limit=25,
    )

    assert len(withdrawals) == 1
    withdrawal = withdrawals[0]
    assert withdrawal["transfer_id"] == "wd-1"
    assert withdrawal["symbol"] == "BTC_USDT"
    assert withdrawal["status"] == "COMPLETED"
    assert withdrawal["amount"] == 0.3
    assert withdrawal["fee"] == 0.0004
    assert withdrawal["network"] == "BTC"
    assert withdrawal["address"] == "1BitcoinAddr"
    assert withdrawal["tag"] == ""
    assert withdrawal["tx_id"] == "0xdef"
    assert withdrawal["timestamp"] == 1_700_199_900_000.0
    assert withdrawal["completed_timestamp"] == 1_700_200_100_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/withdrawals",
        params={
            "symbol": "BTC-USDT",
            "status": "completed",
            "start": 1_699_998_888_000,
            "end": 1_700_200_000_000,
            "limit": 25,
        },
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_withdrawals_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/withdrawals",
        json={
            "withdrawals": [
                {
                    "withdrawalId": "wd-2",
                    "symbol": "ETH-USDT",
                    "status": "PENDING",
                    "amount": "1.0",
                    "timestamp": 1_700_200_200_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_withdrawals_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_withdrawals_history_rejects_invalid_amount() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/withdrawals",
        json={
            "withdrawals": [
                {
                    "withdrawalId": "wd-3",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "oops",
                    "timestamp": 1_700_201_000_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_withdrawals_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_transfers_history_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_300_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/transfers",
        json={
            "transfers": [
                {
                    "transferId": "tr-1",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "0.2",
                    "fromAccount": "spot",
                    "toAccount": "margin",
                    "timestamp": 1_700_299_900_000,
                    "completedAt": 1_700_300_100_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "direction": "outbound",
                    "from": "spot",
                    "to": "margin",
                    "start": "1699997777000",
                    "end": "1700300000000",
                    "limit": "20",
                }
            )
        ],
    )

    transfers = adapter.fetch_transfers_history(
        symbol="BTC_USDT",
        direction="outbound",
        from_account="spot",
        to_account="margin",
        start=1_699_997_777_000,
        end=1_700_300_000_000,
        limit=20,
    )

    assert len(transfers) == 1
    transfer = transfers[0]
    assert transfer["transfer_id"] == "tr-1"
    assert transfer["symbol"] == "BTC_USDT"
    assert transfer["amount"] == 0.2
    assert transfer["status"] == "COMPLETED"
    assert transfer["from_account"] == "spot"
    assert transfer["to_account"] == "margin"
    assert transfer["timestamp"] == 1_700_299_900_000.0
    assert transfer["completed_timestamp"] == 1_700_300_100_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/transfers",
        params={
            "symbol": "BTC-USDT",
            "direction": "outbound",
            "from": "spot",
            "to": "margin",
            "start": 1_699_997_777_000,
            "end": 1_700_300_000_000,
            "limit": 20,
        },
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_transfers_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/transfers",
        json={
            "transfers": [
                {
                    "transferId": "tr-2",
                    "symbol": "ETH-USDT",
                    "status": "COMPLETED",
                    "amount": "0.1",
                    "timestamp": 1_700_300_200_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_transfers_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_transfers_history_rejects_invalid_amount() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/transfers",
        json={
            "transfers": [
                {
                    "transferId": "tr-3",
                    "symbol": "BTC-USDT",
                    "status": "COMPLETED",
                    "amount": "not-a-number",
                    "timestamp": 1_700_300_500_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_transfers_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_fee_rates_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_200_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/fees",
        json={
            "fees": [
                {
                    "symbol": "BTC-USDT",
                    "maker": "0.0012",
                    "taker": "0.0015",
                    "thirtyDayVolume": "12345.678",
                }
            ]
        },
        match=[responses.matchers.query_param_matcher({"symbol": "BTC-USDT"})],
    )

    fees = adapter.fetch_fee_rates(symbol="BTC_USDT")

    assert len(fees) == 1
    fee_entry = fees[0]
    assert fee_entry["symbol"] == "BTC_USDT"
    assert fee_entry["maker_fee"] == 0.0012
    assert fee_entry["taker_fee"] == 0.0015
    assert fee_entry["thirty_day_volume"] == 12345.678

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/fees",
        params={"symbol": "BTC-USDT"},
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_fee_rates_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/fees",
        json={
            "fees": [
                {
                    "symbol": "ETH-USDT",
                    "maker": "0.001",
                    "taker": "0.0015",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_fee_rates(symbol="BTC_USDT")


@responses.activate
def test_fetch_fee_rates_rejects_invalid_values() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/fees",
        json={
            "fees": [
                {
                    "symbol": "BTC-USDT",
                    "maker": "not-a-number",
                    "taker": "0.0015",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_fee_rates(symbol="BTC_USDT")


@responses.activate
def test_fetch_rebates_history_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_250_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/rebates",
        json={
            "rebates": [
                {
                    "rebateId": "rb-1",
                    "symbol": "BTC-USDT",
                    "amount": "0.0005",
                    "rate": "0.001",
                    "type": "maker",
                    "orderId": "o-1",
                    "timestamp": 1_700_249_900_000,
                    "settledAt": 1_700_250_100_000,
                }
            ]
        },
        match=[responses.matchers.query_param_matcher({"symbol": "BTC-USDT"})],
    )

    rebates = adapter.fetch_rebates_history(symbol="BTC_USDT")

    assert len(rebates) == 1
    rebate = rebates[0]
    assert rebate["rebate_id"] == "rb-1"
    assert rebate["symbol"] == "BTC_USDT"
    assert rebate["amount"] == 0.0005
    assert rebate["rate"] == 0.001
    assert rebate["type"] == "maker"
    assert rebate["order_id"] == "o-1"
    assert rebate["timestamp"] == 1_700_249_900_000.0
    assert rebate["settled_timestamp"] == 1_700_250_100_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/rebates",
        params={"symbol": "BTC-USDT"},
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_rebates_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/rebates",
        json={
            "rebates": [
                {
                    "rebateId": "rb-2",
                    "symbol": "ETH-USDT",
                    "amount": "0.0003",
                    "type": "maker",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_rebates_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_rebates_history_rejects_invalid_amount() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/rebates",
        json={
            "rebates": [
                {
                    "rebateId": "rb-3",
                    "symbol": "BTC-USDT",
                    "amount": "not-a-number",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_rebates_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_interest_history_translates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_300_000_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/interest",
        json={
            "interest": [
                {
                    "interestId": "in-1",
                    "symbol": "BTC-USDT",
                    "amount": "0.0002",
                    "rate": "0.0005",
                    "type": "margin",
                    "orderId": "o-1",
                    "timestamp": 1_700_299_900_000,
                    "accrualTimestamp": 1_700_299_800_000,
                }
            ]
        },
        match=[responses.matchers.query_param_matcher({"symbol": "BTC-USDT"})],
    )

    interest = adapter.fetch_interest_history(symbol="BTC_USDT")

    assert len(interest) == 1
    entry = interest[0]
    assert entry["interest_id"] == "in-1"
    assert entry["symbol"] == "BTC_USDT"
    assert entry["amount"] == 0.0002
    assert entry["rate"] == 0.0005
    assert entry["type"] == "margin"
    assert entry["order_id"] == "o-1"
    assert entry["timestamp"] == 1_700_299_900_000.0
    assert entry["accrual_timestamp"] == 1_700_299_800_000.0

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/interest",
        params={"symbol": "BTC-USDT"},
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_interest_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/interest",
        json={
            "interest": [
                {
                    "interestId": "in-2",
                    "symbol": "ETH-USDT",
                    "amount": "0.0003",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_interest_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_interest_history_rejects_invalid_amount() -> None:
    adapter = _build_adapter()

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/interest",
        json={
            "interest": [
                {
                    "interestId": "in-3",
                    "symbol": "BTC-USDT",
                    "amount": "not-a-number",
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_interest_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_ohlcv_translates_payload_and_validates_types() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/public/ohlcv",
        json={
            "symbol": "BTC-USDT",
            "candles": [
                [1_700_000_000_000, "50000", "50100", "49950", "50050", "120"],
                [1_700_000_060_000, "50050", "50200", "50025", "50150", "95.5"],
            ],
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "interval": "1m",
                    "start": "1699999800000",
                    "end": "1700000060000",
                    "limit": "2",
                }
            )
        ],
    )

    candles = adapter.fetch_ohlcv(
        "BTC_USDT",
        "1m",
        start=1_699_999_800_000,
        end=1_700_000_060_000,
        limit=2,
    )

    assert candles == [
        [1_700_000_000_000.0, 50000.0, 50100.0, 49950.0, 50050.0, 120.0],
        [1_700_000_060_000.0, 50050.0, 50200.0, 50025.0, 50150.0, 95.5],
    ]


@responses.activate
def test_fetch_ohlcv_rejects_invalid_candles() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/public/ohlcv",
        json={
            "symbol": "BTC-USDT",
            "candles": [["invalid"]],
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_ohlcv("BTC_USDT", "1m")


@responses.activate
def test_fetch_trades_history_translates_and_casts(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter()
    fixed_ts = 1_700_000_777_000
    monkeypatch.setattr(adapter, "_timestamp", lambda: fixed_ts)

    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/trades",
        json={
            "trades": [
                {
                    "tradeId": "t-1",
                    "orderId": "o-1",
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "price": "50000.0",
                    "quantity": "0.1",
                    "fee": "0.05",
                    "timestamp": 1_700_000_100_000,
                }
            ]
        },
        match=[
            responses.matchers.query_param_matcher(
                {
                    "symbol": "BTC-USDT",
                    "start": "1699999800000",
                    "limit": "50",
                }
            )
        ],
    )

    trades = adapter.fetch_trades_history(symbol="BTC_USDT", start=1_699_999_800_000, limit=50)

    assert trades == [
        {
            "trade_id": "t-1",
            "order_id": "o-1",
            "symbol": "BTC_USDT",
            "side": "buy",
            "price": 50000.0,
            "quantity": 0.1,
            "fee": 0.05,
            "timestamp": 1_700_000_100_000.0,
            "raw": {
                "tradeId": "t-1",
                "orderId": "o-1",
                "symbol": "BTC-USDT",
                "side": "buy",
                "price": "50000.0",
                "quantity": "0.1",
                "fee": "0.05",
                "timestamp": 1_700_000_100_000,
            },
        }
    ]

    call = responses.calls[0]
    expected_signature = adapter.sign_request(
        fixed_ts,
        "GET",
        "/private/trades",
        params={"symbol": "BTC-USDT", "start": 1_699_999_800_000, "limit": 50},
    )
    assert call.request.headers["X-API-SIGN"] == expected_signature


@responses.activate
def test_fetch_trades_history_rejects_invalid_entry() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/trades",
        json={
            "trades": [
                {
                    "tradeId": "t-1",
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "price": "not-a-number",
                    "quantity": "0.1",
                    "timestamp": 1_700_000_100_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_trades_history(symbol="BTC_USDT")


@responses.activate
def test_fetch_trades_history_rejects_symbol_mismatch() -> None:
    adapter = _build_adapter()
    responses.add(
        responses.GET,
        f"{_BASE_URL}/private/trades",
        json={
            "trades": [
                {
                    "tradeId": "t-1",
                    "orderId": "o-1",
                    "symbol": "ETH-USDT",
                    "side": "buy",
                    "price": "50000.0",
                    "quantity": "0.1",
                    "timestamp": 1_700_000_100_000,
                }
            ]
        },
    )

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_trades_history(symbol="BTC_USDT")
