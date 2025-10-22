from __future__ import annotations

import json
from typing import Mapping
from urllib.parse import parse_qs, urlparse

import pytest
from requests.exceptions import RequestException

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.exchanges.nowa_gielda.spot import NowaGieldaHTTPClient
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)


BASE_URL = "https://api.nowa-gielda.test"

responses = pytest.importorskip("responses")


@pytest.fixture
def client() -> NowaGieldaHTTPClient:
    return NowaGieldaHTTPClient(BASE_URL)


def _recorded_query(index: int = 0) -> Mapping[str, list[str]]:
    request = responses.calls[index].request
    parsed = urlparse(request.url)
    return parse_qs(parsed.query)


@responses.activate
def test_fetch_account_forwards_headers(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"balances": []},
        status=200,
    )

    result = client.fetch_account(headers={"X-Auth": "abc"})

    assert result == {"balances": []}
    request = responses.calls[0].request
    assert request.headers["X-Auth"] == "abc"
    assert request.method == "GET"


@responses.activate
def test_fetch_account_does_not_mutate_headers(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"balances": []},
        status=200,
    )

    headers = {"X-Auth": "abc"}
    result = client.fetch_account(headers=headers)

    assert result == {"balances": []}
    assert headers == {"X-Auth": "abc"}


@responses.activate
def test_fetch_ohlcv_builds_query_from_arguments(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/public/ohlcv",
        json={"symbol": "BTCUSDT", "candles": []},
        status=200,
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
    query = _recorded_query()
    assert query == {
        "symbol": ["BTCUSDT"],
        "interval": ["1m"],
        "start": ["1"],
        "end": ["2"],
        "limit": ["3"],
        "foo": ["bar"],
    }


@responses.activate
def test_fetch_trades_forwards_params_and_headers(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/trades",
        json={"trades": []},
        status=200,
    )

    params = {"symbol": "BTCUSDT", "limit": 10, "start": None}
    result = client.fetch_trades(headers={"X-Auth": "abc"}, params=params)

    assert result == {"trades": []}
    query = _recorded_query()
    assert query == {"symbol": ["BTCUSDT"], "limit": ["10"]}
    request = responses.calls[0].request
    assert request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "BTCUSDT", "limit": 10, "start": None}


@responses.activate
def test_fetch_open_orders_forwards_params(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/orders",
        json={"orders": []},
        status=200,
    )

    params = {"symbol": "ETHUSDT", "limit": 5, "start": None}
    result = client.fetch_open_orders(
        headers={"X-Auth": "abc"}, params=params
    )

    assert result == {"orders": []}
    query = _recorded_query()
    assert query == {"symbol": ["ETHUSDT"], "limit": ["5"]}
    request = responses.calls[0].request
    assert request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "ETHUSDT", "limit": 5, "start": None}


@responses.activate
def test_fetch_order_history_forwards_params(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/orders/history",
        json={"orders": []},
        status=200,
    )

    params = {"symbol": "BTCUSDT", "start": 1, "end": 2, "limit": None}
    result = client.fetch_order_history(
        headers={"X-Auth": "abc"},
        params=params,
    )

    assert result == {"orders": []}
    query = _recorded_query()
    assert query == {
        "symbol": ["BTCUSDT"],
        "start": ["1"],
        "end": ["2"],
    }
    request = responses.calls[0].request
    assert request.headers["X-Auth"] == "abc"
    assert params == {"symbol": "BTCUSDT", "start": 1, "end": 2, "limit": None}


@responses.activate
def test_fetch_account_strips_none_params(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"balances": []},
        status=200,
        match=[responses.matchers.query_param_matcher({"recvWindow": "5000"})],
    )

    params = {"recvWindow": 5000, "foo": None}
    result = client.fetch_account(headers={"X-Auth": "abc"}, params=params)

    assert result == {"balances": []}
    assert params == {"recvWindow": 5000, "foo": None}


@responses.activate
def test_fetch_trades_without_params_does_not_append_query(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/trades",
        json={"trades": []},
        status=200,
    )

    result = client.fetch_trades(headers={"X-Auth": "abc"})

    assert result == {"trades": []}
    request = responses.calls[0].request
    assert "?" not in request.url
    assert request.headers["X-Auth"] == "abc"


@responses.activate
def test_fetch_account_without_params_does_not_append_query(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"balances": []},
        status=200,
    )

    result = client.fetch_account(headers={"X-Auth": "abc"})

    assert result == {"balances": []}
    request = responses.calls[0].request
    assert "?" not in request.url
    assert request.headers["X-Auth"] == "abc"


@responses.activate
def test_fetch_open_orders_without_params_does_not_append_query(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/orders",
        json={"orders": []},
        status=200,
    )

    result = client.fetch_open_orders(headers={"X-Auth": "abc"})

    assert result == {"orders": []}
    request = responses.calls[0].request
    assert "?" not in request.url
    assert request.headers["X-Auth"] == "abc"


@responses.activate
def test_fetch_order_history_without_params_does_not_append_query(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/orders/history",
        json={"orders": []},
        status=200,
    )

    result = client.fetch_order_history(headers={"X-Auth": "abc"})

    assert result == {"orders": []}
    request = responses.calls[0].request
    assert "?" not in request.url
    assert request.headers["X-Auth"] == "abc"


@responses.activate
def test_fetch_ohlcv_does_not_mutate_params(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/public/ohlcv",
        json={"symbol": "BTCUSDT", "candles": []},
        status=200,
    )

    params = {"foo": "bar", "limit": None}
    result = client.fetch_ohlcv(
        "BTCUSDT",
        "1m",
        start=None,
        end=None,
        limit=10,
        params=params,
        headers={"X-Test": "1"},
    )

    assert result == {"symbol": "BTCUSDT", "candles": []}
    request = responses.calls[0].request
    assert request.headers["X-Test"] == "1"
    assert params == {"foo": "bar", "limit": None}


@responses.activate
def test_fetch_ohlcv_allows_overriding_with_extra_params(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/public/ohlcv",
        json={"symbol": "OVERRIDDEN", "candles": []},
        status=200,
    )

    result = client.fetch_ohlcv(
        "BTCUSDT",
        "1m",
        params={"symbol": "OVERRIDDEN", "interval": "5m"},
    )

    assert result == {"symbol": "OVERRIDDEN", "candles": []}
    query = _recorded_query()
    assert query == {"symbol": ["OVERRIDDEN"], "interval": ["5m"]}


@responses.activate
def test_create_order_sends_json_payload(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.POST,
        f"{BASE_URL}/private/orders",
        json={"orderId": "123"},
        status=200,
    )

    payload = {"symbol": "BTCUSDT", "side": "BUY"}
    headers = {"X-Auth": "abc"}
    result = client.create_order(payload, headers=headers)

    assert result == {"orderId": "123"}
    request = responses.calls[0].request
    assert request.headers["X-Auth"] == "abc"
    assert json.loads(request.body) == payload
    assert request.method == "POST"
    assert payload == {"symbol": "BTCUSDT", "side": "BUY"}
    assert headers == {"X-Auth": "abc"}


@responses.activate
def test_cancel_order_builds_query_with_symbol(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.DELETE,
        f"{BASE_URL}/private/orders",
        json={"status": "CANCELLED"},
        status=200,
    )

    result = client.cancel_order("123", headers={"X-Auth": "abc"}, symbol="BTCUSDT")

    assert result == {"status": "CANCELLED"}
    request = responses.calls[0].request
    assert request.method == "DELETE"
    assert request.headers["X-Auth"] == "abc"
    query = _recorded_query()
    assert query == {"orderId": ["123"], "symbol": ["BTCUSDT"]}


@responses.activate
def test_cancel_order_without_symbol_omits_query_param(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.DELETE,
        f"{BASE_URL}/private/orders",
        json={"status": "CANCELLED"},
        status=200,
    )

    result = client.cancel_order("123", headers={"X-Auth": "abc"})

    assert result == {"status": "CANCELLED"}
    request = responses.calls[0].request
    assert request.method == "DELETE"
    assert request.headers["X-Auth"] == "abc"
    assert _recorded_query() == {"orderId": ["123"]}


@responses.activate
def test_fetch_account_raises_auth_error_on_unauthorized(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"message": "Auth required"},
        status=401,
    )

    with pytest.raises(ExchangeAuthError) as exc:
        client.fetch_account(headers={"X-Auth": "bad"})

    assert exc.value.status_code == 401
    assert exc.value.payload == {"message": "Auth required"}


@responses.activate
def test_fetch_account_raises_error_based_on_payload_code(
    client: NowaGieldaHTTPClient,
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"code": "INVALID_SYMBOL", "message": "Bad symbol"},
        status=400,
    )

    with pytest.raises(ExchangeAPIError) as exc:
        client.fetch_account(headers={"X-Auth": "abc"})

    assert exc.value.status_code == 400
    assert exc.value.payload == {"code": "INVALID_SYMBOL", "message": "Bad symbol"}


@responses.activate
@pytest.mark.parametrize(
    "status, payload, expected",
    [
        (403, {"message": "Forbidden"}, ExchangeAuthError),
        (429, {"message": "Too many requests"}, ExchangeThrottlingError),
    ],
)
def test_fetch_account_maps_http_status_to_errors(
    client: NowaGieldaHTTPClient,
    status: int,
    payload: Mapping[str, str],
    expected: type[ExchangeAPIError],
) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json=payload,
        status=status,
    )

    with pytest.raises(expected) as exc:
        client.fetch_account(headers={"X-Auth": "abc"})

    assert exc.value.status_code == status
    assert exc.value.payload == payload


@responses.activate
def test_fetch_account_uses_error_code_mapping(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        json={"code": "AUTHENTICATION_REQUIRED", "message": "Login"},
        status=400,
    )

    with pytest.raises(ExchangeAuthError) as exc:
        client.fetch_account(headers={"X-Auth": "abc"})

    assert exc.value.status_code == 400
    assert exc.value.payload == {"code": "AUTHENTICATION_REQUIRED", "message": "Login"}


@responses.activate
def test_fetch_account_handles_text_error_payload(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/private/account",
        body="Service unavailable",
        status=503,
        content_type="text/plain",
    )

    with pytest.raises(ExchangeAPIError) as exc:
        client.fetch_account(headers={"X-Auth": "abc"})

    assert exc.value.status_code == 503
    assert exc.value.payload == "Service unavailable"


def test_fetch_account_wraps_network_errors(client: NowaGieldaHTTPClient) -> None:
    class FailingSession:
        def request(self, *_args: object, **_kwargs: object) -> None:
            raise RequestException("boom")

    failing_client = NowaGieldaHTTPClient(BASE_URL, session=FailingSession())

    with pytest.raises(ExchangeNetworkError) as exc:
        failing_client.fetch_account(headers={"X-Auth": "abc"})

    assert "boom" in str(exc.value.reason)


@responses.activate
def test_fetch_ticker_raises_for_invalid_json(client: NowaGieldaHTTPClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/public/ticker",
        body="{invalid",
        status=200,
        content_type="application/json",
    )

    with pytest.raises(ExchangeAPIError) as exc:
        client.fetch_ticker("BTCUSDT")

    assert "Niepoprawny format" in str(exc.value)


@responses.activate
def test_rate_limiter_blocks_after_threshold(client: NowaGieldaHTTPClient) -> None:
    for _ in range(5):
        responses.add(
            responses.GET,
            f"{BASE_URL}/private/account",
            json={"balances": []},
            status=200,
        )

    headers = {"X-Auth": "abc"}
    for _ in range(5):
        assert client.fetch_account(headers=headers) == {"balances": []}

    with pytest.raises(ExchangeThrottlingError):
        client.fetch_account(headers=headers)
