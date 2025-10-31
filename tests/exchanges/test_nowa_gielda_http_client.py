from __future__ import annotations

from typing import Mapping

import json
import httpx
import pytest

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.nowa_gielda.spot import NowaGieldaHTTPClient


BASE_URL = "https://api.nowa-gielda.test"

respx = pytest.importorskip("respx")


@pytest.fixture
def api_mock() -> "respx.Router":
    with respx.mock(base_url=BASE_URL) as router:
        yield router


@pytest.fixture
def client() -> NowaGieldaHTTPClient:
    instance = NowaGieldaHTTPClient(BASE_URL)
    try:
        yield instance
    finally:
        instance.close()


def _query(request: httpx.Request) -> Mapping[str, str]:
    return dict(request.url.params)


def test_fetch_account_forwards_headers(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.get("/private/account").mock(
        return_value=httpx.Response(200, json={"balances": []})
    )

    result = client.fetch_account(headers={"X-Auth": "abc"})

    assert result == {"balances": []}
    request = route.calls.last.request
    assert request.headers["X-Auth"] == "abc"
    assert request.method == "GET"


def test_fetch_account_does_not_mutate_headers(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/private/account").mock(return_value=httpx.Response(200, json={"balances": []}))

    headers = {"X-Auth": "abc"}
    result = client.fetch_account(headers=headers)

    assert result == {"balances": []}
    assert headers == {"X-Auth": "abc"}


def test_fetch_ohlcv_builds_query_from_arguments(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.get("/public/ohlcv").mock(
        return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "candles": []})
    )

    result = client.fetch_ohlcv(
        "BTCUSDT",
        "1m",
        start=1,
        end=2,
        limit=3,
        params={"foo": "bar", "start": None},
    )

    assert result == {"symbol": "BTCUSDT", "candles": []}
    query = _query(route.calls.last.request)
    assert query == {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "start": "1",
        "end": "2",
        "limit": "3",
        "foo": "bar",
    }


def test_fetch_trades_forwards_params_and_headers(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    route = api_mock.get("/private/trades").mock(
        return_value=httpx.Response(200, json={"trades": []})
    )

    params = {"symbol": "BTCUSDT", "limit": 10, "start": None}
    result = client.fetch_trades(headers={"X-Auth": "abc"}, params=params)

    assert result == {"trades": []}
    query = _query(route.calls.last.request)
    assert query == {"symbol": "BTCUSDT", "limit": "10"}
    assert route.calls.last.request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "BTCUSDT", "limit": 10, "start": None}


def test_fetch_open_orders_forwards_params(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.get("/private/orders").mock(
        return_value=httpx.Response(200, json={"orders": []})
    )

    params = {"symbol": "ETHUSDT", "limit": 5, "start": None}
    result = client.fetch_open_orders(headers={"X-Auth": "abc"}, params=params)

    assert result == {"orders": []}
    query = _query(route.calls.last.request)
    assert query == {"symbol": "ETHUSDT", "limit": "5"}
    assert route.calls.last.request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "ETHUSDT", "limit": 5, "start": None}


def test_fetch_order_history_forwards_params(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    route = api_mock.get("/private/orders/history").mock(
        return_value=httpx.Response(200, json={"orders": []})
    )

    params = {"symbol": "BTCUSDT", "start": 1, "end": 2, "limit": None}
    result = client.fetch_order_history(headers={"X-Auth": "abc"}, params=params)

    assert result == {"orders": []}
    query = _query(route.calls.last.request)
    assert query == {"symbol": "BTCUSDT", "start": "1", "end": "2"}
    assert route.calls.last.request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "BTCUSDT", "start": 1, "end": 2, "limit": None}


def test_fetch_account_strips_none_params(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.get("/private/account").mock(
        return_value=httpx.Response(200, json={"balances": []})
    )

    params = {"recvWindow": 5000, "foo": None}
    result = client.fetch_account(headers={"X-Auth": "abc"}, params=params)

    assert result == {"balances": []}
    assert params == {"recvWindow": 5000, "foo": None}
    assert _query(route.calls.last.request) == {"recvWindow": "5000"}


def test_fetch_trades_without_params_does_not_append_query(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    route = api_mock.get("/private/trades").mock(
        return_value=httpx.Response(200, json={"trades": []})
    )

    result = client.fetch_trades(headers={"X-Auth": "abc"})

    assert result == {"trades": []}
    request = route.calls.last.request
    assert not request.url.params
    assert request.headers["X-Auth"] == "abc"


def test_fetch_account_without_params_does_not_append_query(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    route = api_mock.get("/private/account").mock(
        return_value=httpx.Response(200, json={"balances": []})
    )

    result = client.fetch_account(headers={"X-Auth": "abc"})

    assert result == {"balances": []}
    request = route.calls.last.request
    assert not request.url.params
    assert request.headers["X-Auth"] == "abc"


def test_fetch_ohlcv_does_not_mutate_params(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/public/ohlcv").mock(
        return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "candles": []})
    )

    params = {"foo": "bar", "symbol": None}
    result = client.fetch_ohlcv("BTCUSDT", "1m", params=params)

    assert result == {"symbol": "BTCUSDT", "candles": []}
    assert params == {"foo": "bar", "symbol": None}


def test_fetch_trades_does_not_mutate_params(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/private/trades").mock(return_value=httpx.Response(200, json={"trades": []}))

    params = {"symbol": "BTCUSDT", "limit": 10, "start": None}
    client.fetch_trades(headers={"X-Auth": "abc"}, params=params)

    assert params == {"symbol": "BTCUSDT", "limit": 10, "start": None}


def test_create_order_sends_json_payload(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.post("/private/orders").mock(
        return_value=httpx.Response(200, json={"orderId": "1"})
    )

    payload = {"symbol": "BTCUSDT", "side": "buy", "quantity": 1}
    result = client.create_order(payload, headers={"X-Auth": "abc"})

    assert result == {"orderId": "1"}
    request = route.calls.last.request
    assert request.headers["X-Auth"] == "abc"
    assert json.loads(request.content) == payload


def test_cancel_order_builds_query_with_symbol(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.delete("/private/orders").mock(
        return_value=httpx.Response(200, json={"success": True})
    )

    result = client.cancel_order("1", headers={"X-Auth": "abc"}, symbol="BTCUSDT")

    assert result == {"success": True}
    query = _query(route.calls.last.request)
    assert query == {"orderId": "1", "symbol": "BTCUSDT"}


def test_cancel_order_without_symbol(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    route = api_mock.delete("/private/orders").mock(
        return_value=httpx.Response(200, json={"success": True})
    )

    client.cancel_order("1", headers={"X-Auth": "abc"})

    query = _query(route.calls.last.request)
    assert query == {"orderId": "1"}


def test_fetch_account_raises_on_auth_error(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/private/account").mock(
        return_value=httpx.Response(401, json={"message": "auth", "code": "INVALID_SIGNATURE"})
    )

    with pytest.raises(ExchangeAuthError):
        client.fetch_account(headers={"X-Auth": "abc"})


def test_fetch_account_maps_rate_limit(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/private/account").mock(
        return_value=httpx.Response(429, json={"message": "Too many", "code": "RATE_LIMIT_EXCEEDED"})
    )

    with pytest.raises(ExchangeThrottlingError):
        client.fetch_account(headers={"X-Auth": "abc"})


def test_fetch_account_handles_text_error_payload(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    api_mock.get("/private/account").mock(return_value=httpx.Response(500, text="Internal error"))

    with pytest.raises(ExchangeAPIError):
        client.fetch_account(headers={"X-Auth": "abc"})


def test_fetch_account_wraps_network_errors(
    api_mock: "respx.Router", client: NowaGieldaHTTPClient
) -> None:
    api_mock.get("/private/account").mock(side_effect=httpx.TransportError("boom"))

    with pytest.raises(ExchangeNetworkError):
        client.fetch_account(headers={"X-Auth": "abc"})


def test_fetch_ticker_raises_for_invalid_json(api_mock: "respx.Router", client: NowaGieldaHTTPClient) -> None:
    api_mock.get("/public/ticker").mock(return_value=httpx.Response(200, text="not-json"))

    with pytest.raises(ExchangeAPIError):
        client.fetch_ticker("BTCUSDT")


def test_rate_limiter_blocks_after_threshold(client: NowaGieldaHTTPClient) -> None:
    limiter = client.rate_limiter
    rule = limiter._rules["GET /private/account"]

    remaining = rule.max_requests
    while remaining >= rule.weight:
        limiter.consume("GET", "/private/account")
        remaining -= rule.weight

    with pytest.raises(ExchangeThrottlingError):
        limiter.consume("GET", "/private/account")
