"""Testy jednostkowe adaptera Binance Spot."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest



from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.errors import (
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.binance.symbols import (
    filter_supported_exchange_symbols,
    normalize_symbol,
    to_exchange_symbol,
)
from bot_core.observability.metrics import MetricsRegistry


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, operation: str, func):
        self.calls.append(operation)
        return func()


class _FakeResponse:
    def __init__(self, payload: Any, headers: Mapping[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        if isinstance(self._payload, (bytes, bytearray)):
            return bytes(self._payload)
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_account_snapshot_parses_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None
    ticker_requested = False

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request, ticker_requested
        if "/api/v3/account" in request.full_url:
            captured_request = request
            payload: dict[str, Any] = {
                "balances": [
                    {"asset": "BTC", "free": "0.5", "locked": "0.1"},
                    {"asset": "USDT", "free": "100", "locked": "0"},
                ],
            }
        elif "/api/v3/ticker/price" in request.full_url:
            ticker_requested = True
            payload = [
                {"symbol": "BTCUSDT", "price": "10000"},
                {"symbol": "USDCUSDT", "price": "1"},
            ]
        else:  # pragma: no cover - zabezpieczenie dodatkowych endpointów
            raise AssertionError(f"Unexpected endpoint requested: {request.full_url}")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["BTC"] == pytest.approx(0.6)
    assert snapshot.balances["USDT"] == pytest.approx(100.0)
    assert snapshot.total_equity == pytest.approx(6100.0)
    assert snapshot.available_margin == pytest.approx(5100.0)
    assert ticker_requested is True
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "test-key"
    assert "signature=" in captured_request.full_url


def test_fetch_account_snapshot_uses_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    watchdog = _RecordingWatchdog()

    credentials = ExchangeCredentials(
        key_id="watchdog", secret="secret", permissions=("read", "trade"), environment=Environment.LIVE
    )
    adapter = BinanceSpotAdapter(credentials, watchdog=watchdog)

    monkeypatch.setattr(
        adapter,
        "_signed_request",
        lambda *args, **kwargs: {
            "balances": [
                {"asset": "USDT", "free": "10", "locked": "0"},
            ]
        },
    )
    monkeypatch.setattr(
        adapter,
        "_public_request",
        lambda *args, **kwargs: [{"symbol": "USDTUSDT", "price": "1"}],
    )

    snapshot = adapter.fetch_account_snapshot()

    assert snapshot.total_equity == pytest.approx(10.0)
    assert "binance_spot_fetch_account" in watchdog.calls


def test_fetch_account_snapshot_maps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        if "/api/v3/account" in request.full_url:
            return _FakeResponse({"code": -2015, "msg": "Invalid API-key"})
        raise AssertionError(f"Unexpected endpoint {request.full_url}")

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()


def test_fetch_account_snapshot_values_mixed_portfolio(monkeypatch: pytest.MonkeyPatch) -> None:
    signed_request: Request | None = None
    ticker_requested = False

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal signed_request, ticker_requested
        if "/api/v3/account" in request.full_url:
            signed_request = request
            payload: dict[str, Any] = {
                "balances": [
                    {"asset": "BTC", "free": "0.2", "locked": "0"},
                    {"asset": "USDT", "free": "50", "locked": "0"},
                ]
            }
        elif "/api/v3/ticker/price" in request.full_url:
            ticker_requested = True
            payload = [
                {"symbol": "BTCUSDC", "price": "27000"},
                {"symbol": "USDCUSDT", "price": "1"},
            ]
        else:  # pragma: no cover - zabezpieczenie dodatkowych endpointów
            raise AssertionError(f"Unexpected endpoint requested: {request.full_url}")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["BTC"] == pytest.approx(0.2)
    assert snapshot.balances["USDT"] == pytest.approx(50.0)
    assert snapshot.total_equity == pytest.approx(5450.0)
    assert snapshot.available_margin == pytest.approx(5450.0)
    assert ticker_requested is True
    assert signed_request is not None


def test_fetch_ticker_maps_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        if "/api/v3/ticker/24hr" in request.full_url:
            return _FakeResponse({"code": -1003, "msg": "Too many requests"})
        raise AssertionError(f"Unexpected endpoint {request.full_url}")

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    with pytest.raises(ExchangeThrottlingError):
        adapter.fetch_ticker("BTC-USDT")


def test_fetch_account_snapshot_respects_custom_valuation_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        if "/api/v3/account" in request.full_url:
            payload: dict[str, Any] = {
                "balances": [
                    {"asset": "BTC", "free": "0.1", "locked": "0"},
                    {"asset": "USDT", "free": "100", "locked": "0"},
                ],
            }
        elif "/api/v3/ticker/price" in request.full_url:
            payload = [
                {"symbol": "BTCEUR", "price": "26000"},
                {"symbol": "EURUSDT", "price": "1.1"},
            ]
        else:  # pragma: no cover - zabezpieczenie dodatkowych endpointów
            raise AssertionError(f"Unexpected endpoint requested: {request.full_url}")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(
        credentials,
        settings={
            "valuation_asset": "eur",
            "secondary_valuation_assets": ["USDT"],
        },
    )

    snapshot = adapter.fetch_account_snapshot()

    assert snapshot.total_equity == pytest.approx(2690.90909, rel=1e-6)
    assert snapshot.available_margin == pytest.approx(2690.90909, rel=1e-6)


def test_place_order_builds_signed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "orderId": 12345,
            "status": "NEW",
            "executedQty": "0",
            "price": "0",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    order_request = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=20_000,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(order_request)

    assert result.order_id == "12345"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0)
    assert result.avg_price is None
    assert captured_request is not None
    assert captured_request.get_method() == "POST"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    body = captured_request.data.decode("utf-8") if captured_request.data else ""
    expected_prefix = "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=0.01&price=20000&timeInForce=GTC&newClientOrderId=cli-1&timestamp=1700000000000"
    assert body.startswith(expected_prefix)
    assert "signature=" in body


def test_cancel_order_uses_delete_method(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {"status": "CANCELED"}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials)

    adapter.cancel_order("123", symbol="BTC/USDT")

    assert captured_request is not None
    assert captured_request.get_method() == "DELETE"
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "key"
    assert "symbol=BTCUSDT" in captured_request.full_url
    assert "orderId=123" in captured_request.full_url
    assert "signature=" in captured_request.full_url


def test_execute_request_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        if not attempts:
            attempts.append(1)
            raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)
        attempts.append(2)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda _: None)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.random.uniform", lambda *_: 0.0)

    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials, metrics_registry=MetricsRegistry())

    request = Request("https://api.binance.com/api/v3/time")
    payload = adapter._execute_request(request, endpoint="/api/v3/time")

    assert attempts == [1, 2]
    assert payload == {"ok": True}


def test_execute_request_raises_throttling_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = io.BytesIO(json.dumps({"code": 429, "msg": "Too Many"}).encode("utf-8"))
        raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs={}, fp=payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda _: None)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.random.uniform", lambda *_: 0.0)

    registry = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials, metrics_registry=registry)

    request = Request("https://api.binance.com/api/v3/time")

    with pytest.raises(ExchangeThrottlingError):
        adapter._execute_request(request, endpoint="/api/v3/time")

    counter = registry.get("binance_spot_retries_total")
    labels = {
        "exchange": "binance_spot",
        "environment": "live",
        "endpoint": "/api/v3/time",
        "method": "GET",
        "reason": "throttled",
        "status": "429",
    }
    assert counter.value(labels=labels) == pytest.approx(3.0 if hasattr(counter, "value") else 3.0)


def test_execute_request_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = io.BytesIO(json.dumps({"code": -2015, "msg": "Invalid API-key"}).encode("utf-8"))
        raise HTTPError(request.full_url, 403, "Forbidden", hdrs={}, fp=payload)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)

    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials, metrics_registry=MetricsRegistry())

    request = Request("https://api.binance.com/api/v3/account")

    with pytest.raises(ExchangeAuthError):
        adapter._execute_request(request, endpoint="/api/v3/account", signed=True)


def test_execute_request_raises_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise URLError("network down")

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda _: None)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.random.uniform", lambda *_: 0.0)

    registry = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials, metrics_registry=registry)

    request = Request("https://api.binance.com/api/v3/time")

    with pytest.raises(ExchangeNetworkError):
        adapter._execute_request(request, endpoint="/api/v3/time")

    counter = registry.get("binance_spot_retries_total")
    labels = {
        "exchange": "binance_spot",
        "environment": "live",
        "endpoint": "/api/v3/time",
        "method": "GET",
        "reason": "network",
    }
    assert counter.value(labels=labels) == pytest.approx(3.0)


def test_execute_request_records_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = {"X-MBX-USED-WEIGHT-1M": "1200", "X-MBX-USED-WEIGHT-1S": "80"}

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"ok": True}, headers=headers)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)

    registry = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials, metrics_registry=registry)

    request = Request("https://api.binance.com/api/v3/time")
    payload = adapter._execute_request(request, endpoint="/api/v3/time")

    assert payload == {"ok": True}

    histogram = registry.get("binance_spot_http_latency_seconds")
    hist_labels = {
        "exchange": "binance_spot",
        "environment": "live",
        "endpoint": "/api/v3/time",
        "method": "GET",
    }
    state = histogram.snapshot(labels=hist_labels)
    assert state.count == 1

    gauge = registry.get("binance_spot_used_weight")
    gauge_1m = {"exchange": "binance_spot", "environment": "live", "window": "1m"}
    gauge_1s = {"exchange": "binance_spot", "environment": "live", "window": "1s"}
    assert gauge.value(labels=gauge_1m) == pytest.approx(1200.0)
    assert gauge.value(labels=gauge_1s) == pytest.approx(80.0)


def test_signed_request_increments_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"orderId": 1, "status": "NEW", "executedQty": "0", "price": "0"})

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = BinanceSpotAdapter(credentials, metrics_registry=registry)

    request = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        order_type="limit",
        price=25_000.0,
    )

    adapter.place_order(request)

    counter = registry.get("binance_spot_signed_requests_total")
    labels = {"exchange": "binance_spot", "environment": "live", "method": "POST"}
    assert counter.value(labels=labels) == pytest.approx(1.0)


def test_fetch_symbols_filters_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials)

    fake_payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING"},
            {"symbol": "ETHUSDT", "status": "TRADING"},
            {"symbol": "ABCUSDT", "status": "TRADING"},
            {"symbol": "ETHUSDT", "status": "BREAK"},
        ]
    }

    monkeypatch.setattr(
        BinanceSpotAdapter,
        "_public_request",
        lambda self, path, params=None, method="GET": fake_payload,
    )

    symbols = adapter.fetch_symbols()
    assert symbols == ("BTC/USDT", "ETH/USDT")


def test_fetch_ohlcv_converts_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials)

    captured_params: dict[str, object] | None = None

    def fake_public_request(self, path, params=None, method="GET"):
        nonlocal captured_params
        captured_params = dict(params or {})
        return [[0, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(BinanceSpotAdapter, "_public_request", fake_public_request)

    candles = adapter.fetch_ohlcv("BTC/USDT", "1d")

    assert candles == [[0, 1, 1, 1, 1, 1]]
    assert captured_params is not None
    assert captured_params["symbol"] == "BTCUSDT"


def test_filter_supported_exchange_symbols_handles_duplicates() -> None:
    filtered = filter_supported_exchange_symbols(["BTCUSDT", "BTCUSDT", "ETHUSDT", "ABCUSDT"])
    assert filtered == ("BTC/USDT", "ETH/USDT")


def test_symbols_module_maps_formats() -> None:
    assert normalize_symbol("btc_usdt") == "BTC/USDT"
    assert normalize_symbol("BTCUSDT") == "BTC/USDT"
    assert normalize_symbol("BTC/USDT") == "BTC/USDT"
    assert to_exchange_symbol("BTC/USDT") == "BTCUSDT"
    assert to_exchange_symbol("eth-pln") == "ETHPLN"


def test_binance_spot_adapter_global_throttle_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="key", secret="secret", permissions=("read",), environment=Environment.LIVE
    )
    adapter = BinanceSpotAdapter(credentials)

    monotonic_now = [1_000.0]

    def fake_monotonic() -> float:
        return monotonic_now[0]

    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.monotonic", fake_monotonic)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda *_: None)

    def throttle_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "2"}, None)

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", throttle_urlopen)

    with pytest.raises(ExchangeThrottlingError):
        adapter._public_request("/api/v3/time")

    status = adapter.failover_status()
    assert status["throttle_active"] is True
    assert status["throttle_remaining"] >= pytest.approx(2.0, rel=0.2)

    success_calls = 0

    def success_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal success_calls
        success_calls += 1
        return _FakeResponse({"serverTime": 1})

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", success_urlopen)

    with pytest.raises(ExchangeThrottlingError):
        adapter._public_request("/api/v3/time")
    assert success_calls == 0

    monotonic_now[0] += 2.5
    result = adapter._public_request("/api/v3/time")
    assert result == {"serverTime": 1}
    assert adapter.failover_status()["throttle_active"] is False
    assert success_calls == 1


def test_binance_spot_adapter_reconnect_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="key", environment=Environment.LIVE)
    adapter = BinanceSpotAdapter(credentials)

    monotonic_now = [5_000.0]

    def fake_monotonic() -> float:
        return monotonic_now[0]

    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.monotonic", fake_monotonic)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda *_: None)

    network_calls = 0

    def failing_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal network_calls
        network_calls += 1
        raise URLError("network down")

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", failing_urlopen)

    with pytest.raises(ExchangeNetworkError):
        adapter._public_request("/api/v3/time")
    assert network_calls >= 3

    status = adapter.failover_status()
    assert status["reconnect_required"] is True

    with pytest.raises(ExchangeNetworkError):
        adapter._public_request("/api/v3/time")

    success_calls = 0

    def success_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal success_calls
        success_calls += 1
        return _FakeResponse({"serverTime": 2})

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", success_urlopen)
    monotonic_now[0] += status["reconnect_remaining"] + 0.5

    result = adapter._public_request("/api/v3/time")
    assert result == {"serverTime": 2}
    assert adapter.failover_status()["reconnect_required"] is False
    assert success_calls == 1
