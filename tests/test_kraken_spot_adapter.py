"""Testy jednostkowe adaptera Kraken Spot."""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl
from urllib.request import Request

import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeAuthError, ExchangeThrottlingError
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.observability.metrics import MetricsRegistry


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _build_credentials() -> ExchangeCredentials:
    secret = base64.b64encode(b"super-secret").decode("utf-8")
    return ExchangeCredentials(
        key_id="kraken-key",
        secret=secret,
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )


def test_fetch_account_snapshot_uses_private_signed_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured_requests.append(request)
        if request.full_url.endswith("TradeBalance"):
            payload = {"error": [], "result": {"eb": "1200.0", "mf": "800.0", "m": "200.0"}}
        else:
            payload = {"error": [], "result": {"ZUSD": "1000.0", "XXBT": "0.5"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["ZUSD"] == pytest.approx(1000.0)
    assert snapshot.total_equity == pytest.approx(1200.0)
    assert snapshot.available_margin == pytest.approx(800.0)
    assert snapshot.maintenance_margin == pytest.approx(200.0)

    assert len(captured_requests) == 2
    first_headers = {name.lower(): value for name, value in captured_requests[0].header_items()}
    assert first_headers["api-key"] == "kraken-key"
    signature = first_headers["api-sign"]
    body_params = dict(parse_qsl(captured_requests[0].data.decode("utf-8")))
    expected_signature = _expected_signature(
        path="/0/private/Balance",
        params=body_params,
        secret=_build_credentials().secret or "",
    )
    assert signature == expected_signature
    trade_params = dict(parse_qsl(captured_requests[1].data.decode("utf-8")))
    assert trade_params.get("asset") == "ZUSD"

    signed_counter = registry.counter(
        "kraken_spot_signed_requests_total",
        "Liczba podpisanych zapytań HTTP wysłanych do API Kraken Spot.",
    )
    assert signed_counter.value(labels={"exchange": "kraken_spot", "environment": "live"}) == pytest.approx(2.0)


def test_fetch_account_snapshot_respects_custom_valuation_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured_requests.append(request)
        if request.full_url.endswith("TradeBalance"):
            payload = {"error": [], "result": {"eb": "100.0", "mf": "80.0", "m": "20.0"}}
        else:
            payload = {"error": [], "result": {"ZEUR": "50.0"}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        settings={"valuation_asset": "eur"},
        metrics_registry=registry,
    )

    adapter.fetch_account_snapshot()
    assert len(captured_requests) == 2
    trade_params = dict(parse_qsl(captured_requests[1].data.decode("utf-8")))
    assert trade_params.get("asset") == "ZEUR"


def test_place_order_builds_payload_with_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {"error": [], "result": {"txid": ["OID123"], "descr": {"order": "buy 0.1 XBTUSD"}}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    order = OrderRequest(
        symbol="XBTUSD",
        side="buy",
        quantity=0.1,
        order_type="limit",
        price=25_000.0,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(order)

    assert result.order_id == "OID123"
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["api-key"] == "kraken-key"
    signature = headers["api-sign"]
    body_params = dict(parse_qsl(captured_request.data.decode("utf-8"))) if captured_request else {}
    expected_signature = _expected_signature(
        path="/0/private/AddOrder",
        params=body_params,
        secret=_build_credentials().secret or "",
    )
    assert signature == expected_signature
    body = captured_request.data.decode("utf-8") if captured_request and captured_request.data else ""
    assert "pair=XBTUSD" in body
    assert "ordertype=limit" in body
    assert "userref=cli-1" in body


def test_cancel_order_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = {"error": [], "result": {"count": 1}}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    adapter.cancel_order("OID123")


def test_private_request_maps_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = {"error": ["EAPI:Invalid key"]}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()

    counter = registry.counter(
        "kraken_spot_api_errors_total",
        "Błędy API Kraken Spot (powód=auth/throttled/api_error/http_error/network/json).",
    )
    assert counter.value(
        labels={
            "exchange": "kraken_spot",
            "environment": "live",
            "endpoint": "/0/private/Balance",
            "signed": "true",
            "reason": "auth",
        }
    ) == pytest.approx(1.0)


def test_public_request_retries_on_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0
    sleeps: list[float] = []

    def fake_sleep(value: float) -> None:
        sleeps.append(value)

    def fake_uniform(_start: float, _end: float) -> float:
        return 0.0

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs={}, fp=io.BytesIO(b""))
        payload = {
            "error": [],
            "result": {
                "XXBTZUSD": {"altname": "XBTUSD"},
            },
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.sleep", fake_sleep)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.random.uniform", fake_uniform)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    symbols = adapter.fetch_symbols()

    assert symbols == ["XBTUSD"]
    assert sleeps, "Oczekiwano zastosowania backoffu przy błędzie 429"
    retry_counter = registry.counter(
        "kraken_spot_retries_total",
        "Liczba ponowień zapytań do API Kraken Spot (powód=throttled/server_error/network).",
    )
    assert retry_counter.value(
        labels={
            "exchange": "kraken_spot",
            "environment": "live",
            "endpoint": "/0/public/AssetPairs",
            "signed": "false",
            "reason": "throttled",
        }
    ) == pytest.approx(1.0)


def test_public_request_raises_throttling_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs={}, fp=io.BytesIO(b""))

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.random.uniform", lambda _a, _b: 0.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    with pytest.raises(ExchangeThrottlingError):
        adapter.fetch_symbols()

    counter = registry.counter(
        "kraken_spot_api_errors_total",
        "Błędy API Kraken Spot (powód=auth/throttled/api_error/http_error/network/json).",
    )
    assert counter.value(
        labels={
            "exchange": "kraken_spot",
            "environment": "live",
            "endpoint": "/0/public/AssetPairs",
            "signed": "false",
            "reason": "throttled",
        }
    ) == pytest.approx(3.0)


def test_private_request_propagates_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = {"error": ["EGeneral:Invalid arguments"]}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    with pytest.raises(ExchangeAPIError):
        adapter.cancel_order("OID123")


def test_fetch_open_orders_returns_sorted_orders_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        payload = {
            "error": [],
            "result": {
                "open": {
                    "OID1": {
                        "descr": {"pair": "XBTUSD", "type": "buy", "ordertype": "limit", "price": "25000"},
                        "vol": "0.2",
                        "vol_exec": "0.05",
                        "opentm": 1_700_000_100.0,
                        "oflags": ["post"],
                    },
                    "OID2": {
                        "descr": {"pair": "ETHUSD", "type": "sell", "ordertype": "market"},
                        "vol": "1.0",
                        "vol_exec": "0.3",
                        "opentm": 1_700_000_050.0,
                        "oflags": "fcib",
                    },
                }
            },
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    orders = adapter.fetch_open_orders()

    assert [order.order_id for order in orders] == ["OID2", "OID1"]
    assert orders[0].flags == ("fcib",)
    assert orders[1].price == pytest.approx(25_000.0)

    gauge = registry.gauge(
        "kraken_spot_open_orders",
        "Liczba otwartych zleceń raportowanych przez Kraken Spot.",
    )
    assert gauge.value(
        labels={
            "exchange": "kraken_spot",
            "environment": "live",
            "endpoint": "/0/private/OpenOrders",
            "signed": "true",
        }
    ) == pytest.approx(2.0)
    assert call_count == 1


def test_fetch_trades_history_paginates_and_updates_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "error": [],
            "result": {
                "trades": {
                    "T1": {
                        "pair": "XBTUSD",
                        "type": "buy",
                        "ordertype": "limit",
                        "price": "25000",
                        "vol": "0.1",
                        "cost": "2500",
                        "fee": "2.5",
                        "time": 1_700_000_001.0,
                        "ordertxid": "OID1",
                    },
                    "T2": {
                        "pair": "ETHUSD",
                        "type": "sell",
                        "ordertype": "market",
                        "price": "2000",
                        "vol": "1.0",
                        "cost": "2000",
                        "fee": "1.0",
                        "time": 1_700_000_005.0,
                        "ordertxid": "OID2",
                    },
                },
                "count": 3,
            },
        },
        {
            "error": [],
            "result": {
                "trades": {
                    "T3": {
                        "pair": "XBTUSD",
                        "type": "buy",
                        "ordertype": "limit",
                        "price": "25500",
                        "vol": "0.05",
                        "cost": "1275",
                        "fee": "1.3",
                        "time": 1_700_000_010.0,
                        "ordertxid": "OID3",
                    }
                },
                "count": 3,
            },
        },
    ]

    call_count = 0
    observed_ofs: list[str | None] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal call_count
        if call_count >= len(responses):
            raise AssertionError("Nieoczekiwana liczba wywołań URLopen")
        body = dict(parse_qsl(request.data.decode("utf-8")))
        observed_ofs.append(body.get("ofs"))
        payload = responses[call_count]
        call_count += 1
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    trades = adapter.fetch_trades_history(symbol="XBTUSD")

    assert [trade.trade_id for trade in trades] == ["T1", "T3"]
    assert trades[0].price == pytest.approx(25_000.0)
    assert trades[1].timestamp > trades[0].timestamp

    counter = registry.counter(
        "kraken_spot_trades_fetched_total",
        "Łączna liczba transakcji pobranych z historii Kraken Spot.",
    )
    assert counter.value(
        labels={
            "exchange": "kraken_spot",
            "environment": "live",
            "endpoint": "/0/private/TradesHistory",
            "signed": "true",
        }
    ) == pytest.approx(2.0)

    assert observed_ofs == ["0", "2"]


def test_fetch_ticker_parses_payload_and_updates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert "/0/public/Ticker" in request.full_url
        payload = {
            "error": [],
            "result": {
                "XXBTZUSD": {
                    "a": ["31000.10000", "1", "1.000"],
                    "b": ["30999.90000", "1", "1.000"],
                    "c": ["31000.05000", "0.100"],
                    "v": ["120.0", "240.0"],
                    "p": ["30900.0", "30850.0"],
                    "h": ["31500.0", "32000.0"],
                    "l": ["30000.0", "29500.0"],
                    "o": ["30500.0", "30200.0"],
                }
            },
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_500.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    ticker = adapter.fetch_ticker("XBTUSD")

    assert ticker.symbol == "XBTUSD"
    assert ticker.best_ask == pytest.approx(31_000.1)
    assert ticker.best_bid == pytest.approx(30_999.9)
    assert ticker.last_price == pytest.approx(31_000.05)
    assert ticker.volume_24h == pytest.approx(240.0)
    assert ticker.high_24h == pytest.approx(32_000.0)
    assert ticker.low_24h == pytest.approx(29_500.0)
    assert ticker.timestamp == pytest.approx(1_700_000_500.0)

    price_gauge = registry.gauge(
        "kraken_spot_ticker_last_price",
        "Ostatnia cena transakcyjna raportowana przez Kraken Spot.",
    )
    spread_gauge = registry.gauge(
        "kraken_spot_ticker_spread",
        "Spread między najlepszą ofertą kupna i sprzedaży na Kraken Spot.",
    )
    labels = {
        "exchange": "kraken_spot",
        "environment": "live",
        "endpoint": "/0/public/Ticker",
        "signed": "false",
        "symbol": "XBTUSD",
    }
    assert price_gauge.value(labels=labels) == pytest.approx(31_000.05)
    assert spread_gauge.value(labels=labels) == pytest.approx(0.20000)


def test_fetch_order_book_normalizes_levels_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert "/0/public/Depth" in request.full_url
        query = dict(parse_qsl(request.full_url.split("?", 1)[1])) if "?" in request.full_url else {}
        assert query.get("count") == "10"
        payload = {
            "error": [],
            "result": {
                "XXBTZUSD": {
                    "bids": [
                        ["30999.9", "1.0", 1_700_000_400.0],
                        ["30950.0", "0.5", 1_700_000_350.0],
                    ],
                    "asks": [
                        ["31000.1", "0.4", 1_700_000_410.0],
                        ["31020.0", "2.0", 1_700_000_420.0],
                    ],
                }
            },
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.kraken.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.kraken.spot.time.time", lambda: 1_700_000_600.0)

    registry = MetricsRegistry()
    adapter = KrakenSpotAdapter(
        _build_credentials(),
        environment=Environment.LIVE,
        metrics_registry=registry,
    )

    order_book = adapter.fetch_order_book("XBTUSD", depth=10)

    assert order_book.symbol == "XBTUSD"
    assert [entry.price for entry in order_book.bids] == [30999.9, 30950.0]
    assert [entry.price for entry in order_book.asks] == [31000.1, 31020.0]
    assert order_book.timestamp == pytest.approx(1_700_000_600.0)
    assert order_book.depth == 4

    levels_gauge = registry.gauge(
        "kraken_spot_orderbook_levels",
        "Łączna liczba poziomów orderbooka (bids+asks) zwracanych przez Kraken Spot.",
    )
    labels = {
        "exchange": "kraken_spot",
        "environment": "live",
        "endpoint": "/0/public/Depth",
        "signed": "false",
        "symbol": "XBTUSD",
    }
    assert levels_gauge.value(labels=labels) == pytest.approx(4.0)

def _expected_signature(*, path: str, params: dict[str, Any], secret: str) -> str:
    params = dict(params)
    nonce = params.pop("nonce", None)
    if nonce is None:
        raise AssertionError("Brak pola nonce w parametrów podpisu")
    sorted_items = sorted(params.items())
    encoded_params = "&".join(f"{k}={v}" for k, v in sorted_items)
    message = (nonce + encoded_params).encode("utf-8")
    sha_digest = hashlib.sha256(message).digest()
    decoded_secret = base64.b64decode(secret)
    mac = hmac.new(decoded_secret, (path.encode("utf-8") + sha_digest), hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("utf-8")
