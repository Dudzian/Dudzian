import json
from io import BytesIO
from pathlib import Path
from hashlib import sha512
import hmac
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest


from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.zonda.spot import (
    ZondaOrderBook,
    ZondaOrderBookLevel,
    ZondaSpotAdapter,
    ZondaTicker,
    ZondaTrade,
)
from bot_core.observability.metrics import MetricsRegistry


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, operation: str, func):
        self.calls.append(operation)
        return func()


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self._status = status
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def getcode(self) -> int:
        return self._status


def test_fetch_account_snapshot_builds_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured
        url = request.full_url
        if url.endswith("/trading/balance"):
            captured = request
            payload = {"status": "Ok", "balances": []}
            headers = {"X-RateLimit-Remaining": "98"}
        elif url.endswith("/trading/ticker"):
            payload = {"status": "Ok", "items": {}}
            headers = {"X-RateLimit-Remaining": "97"}
        else:  # pragma: no cover - zabezpieczenie przed nieoczekiwanym endpointem
            raise AssertionError(f"Unexpected endpoint {url}")
        return _FakeResponse(payload, headers=headers)

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.time", lambda: 1_700_000_000.0)
    metrics = MetricsRegistry()

    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert captured is not None
    headers = {name.lower(): value for name, value in captured.header_items()}
    assert headers["api-key"] == "key"
    timestamp = headers["request-timestamp"]
    body = captured.data.decode("utf-8") if captured.data else ""
    expected_signature = hmac.new(
        b"secret",
        f"{timestamp}POST/trading/balance{body}".encode(),
        sha512,
    ).hexdigest()
    assert headers["api-hash"] == expected_signature
    assert (
        adapter._metric_signed_requests.value(labels=adapter._metric_base_labels)  # type: ignore[attr-defined]
        == pytest.approx(1.0)
    )
    assert (
        adapter._metric_rate_limit_remaining.value(labels=adapter._metric_base_labels)  # type: ignore[attr-defined]
        == pytest.approx(97.0)
    )


def test_fetch_account_snapshot_uses_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    watchdog = _RecordingWatchdog()

    credentials = ExchangeCredentials(
        key_id="watchdog",
        secret="secret",
        permissions=("read",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials, watchdog=watchdog)

    monkeypatch.setattr(adapter, "_signed_request", lambda *args, **kwargs: {"balances": []})
    monkeypatch.setattr(adapter, "_fetch_price_map", lambda: {})

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert "zonda_spot_fetch_account" in watchdog.calls


def test_fetch_ohlcv_maps_items(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(ExchangeCredentials(key_id="public", environment=Environment.LIVE))

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/candle/history/BTC-PLN/86400"
        assert params["from"] == 0
        assert params["to"] == 100
        return {
            "status": "Ok",
            "items": [
                {
                    "time": 1,
                    "open": "10",
                    "high": "12",
                    "low": "9",
                    "close": "11",
                    "volume": "5",
                }
            ],
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    rows = adapter.fetch_ohlcv("BTC-PLN", "1d", start=0, end=100_000, limit=None)

    assert rows == [[1000.0, 10.0, 12.0, 9.0, 11.0, 5.0]]


def test_place_order_uses_private_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] | None = None

    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        nonlocal captured_payload
        assert method == "POST"
        assert path == "/trading/offer"
        captured_payload = dict(data or {})
        return {"order": {"id": "123", "status": "new", "filledAmount": "0"}}

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)

    credentials = ExchangeCredentials(
        key_id="trade",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    request = OrderRequest(
        symbol="BTC-PLN",
        side="buy",
        quantity=1.5,
        order_type="limit",
        price=100.0,
        time_in_force="GTC",
        client_order_id="cli-1",
    )

    result = adapter.place_order(request)

    assert captured_payload == {
        "market": "BTC-PLN",
        "side": "buy",
        "type": "limit",
        "amount": "1.5",
        "price": "100.0",
        "timeInForce": "GTC",
        "clientOrderId": "cli-1",
    }
    assert result.order_id == "123"
    assert result.status == "NEW"
    assert result.filled_quantity == pytest.approx(0.0)


def test_cancel_order_accepts_cancelled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        assert method == "DELETE"
        assert path == "/trading/order/XYZ"
        return {"order": {"id": "XYZ", "status": "cancelled"}}

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)

    credentials = ExchangeCredentials(
        key_id="trade",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    adapter.cancel_order("XYZ")


def test_fetch_ticker_updates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/ticker"
        return {
            "status": "Ok",
            "items": {
                "BTC-PLN": {
                    "time": 1_700_000_000,
                    "highestBid": "99",
                    "lowestAsk": "101",
                    "rate": "100",
                    "volume": "123.45",
                    "max": "110",
                    "min": "90",
                    "average": "98",
                }
            },
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    ticker = adapter.fetch_ticker("BTC-PLN")

    assert isinstance(ticker, ZondaTicker)
    assert ticker.best_bid == pytest.approx(99.0)
    assert ticker.best_ask == pytest.approx(101.0)
    assert ticker.last_price == pytest.approx(100.0)
    labels = adapter._labels(symbol="BTC-PLN")  # type: ignore[attr-defined]
    assert (
        adapter._metric_ticker_last_price.value(labels=labels)  # type: ignore[attr-defined]
        == pytest.approx(100.0)
    )
    assert (
        adapter._metric_ticker_spread.value(labels=labels)  # type: ignore[attr-defined]
        == pytest.approx(2.0)
    )


def test_fetch_ticker_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    adapter = ZondaSpotAdapter(credentials)

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/ticker"
        payload = {"status": "Fail", "errors": [{"code": 429, "message": "limit"}]}
        adapter._ensure_success(payload, status=429, endpoint=path, signed=False)
        return payload

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    with pytest.raises(ExchangeThrottlingError):
        adapter.fetch_ticker("BTC-PLN")


def test_fetch_order_book_parses_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/orderbook-limited/BTC-PLN/5"
        return {
            "status": "Ok",
            "time": 1_700_000_123,
            "buy": [["100", "1"], {"ra": "99.5", "ca": "2"}],
            "sell": [["101", "1.5"]],
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    orderbook = adapter.fetch_order_book("BTC-PLN", depth=5)

    assert isinstance(orderbook, ZondaOrderBook)
    assert orderbook.timestamp == pytest.approx(1_700_000_123.0)
    assert [level.price for level in orderbook.bids] == pytest.approx([100.0, 99.5])
    assert [level.quantity for level in orderbook.asks] == pytest.approx([1.5])
    labels = adapter._labels(symbol="BTC-PLN")  # type: ignore[attr-defined]
    assert (
        adapter._metric_orderbook_levels.value(labels=labels)  # type: ignore[attr-defined]
        == pytest.approx(3.0)
    )


def test_fetch_recent_trades_updates_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = MetricsRegistry()
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/transactions/BTC-PLN"
        assert params == {"limit": 3}
        return {
            "status": "Ok",
            "items": [
                {"id": "1", "time": 1_700_000_100, "rate": "100", "amount": "0.5", "type": "buy"},
                {"id": "2", "time": 1_700_000_200, "rate": "101", "amount": "0.3", "type": "sell"},
            ],
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    trades = adapter.fetch_recent_trades("BTC-PLN", limit=3)

    assert isinstance(trades, Sequence)
    assert len(trades) == 2
    assert isinstance(trades[0], ZondaTrade)
    assert trades[0].side == "buy"
    labels = adapter._labels(symbol="BTC-PLN")  # type: ignore[attr-defined]
    assert (
        adapter._metric_trades_fetched.value(labels=labels)  # type: ignore[attr-defined]
        == pytest.approx(2.0)
    )


def test_public_request_retries_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    metrics = MetricsRegistry()
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.sleep", lambda _: None)

    calls: list[str] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        url = request.full_url
        calls.append(url)
        if len(calls) == 1:
            body = b'{"status":"Error","errors":[{"code":101,"message":"limit"}]}'
            raise HTTPError(
                url,
                429,
                "Too Many Requests",
                {"X-RateLimit-Remaining": "0"},
                BytesIO(body),
            )
        return _FakeResponse(
            {"status": "Ok", "items": {"BTC-PLN": {"market": {"first": "BTC", "second": "PLN"}, "rate": "100"}}},
            headers={"X-RateLimit-Remaining": "99"},
        )

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)

    symbols = adapter.fetch_symbols()

    assert symbols == ["BTC-PLN"]
    assert len(calls) == 2
    retries_metric = adapter._metric_retries.value(  # type: ignore[attr-defined]
        labels={**adapter._metric_base_labels, "reason": "throttled"},
    )
    assert retries_metric == pytest.approx(1.0)
    assert (
        adapter._metric_rate_limit_remaining.value(labels=adapter._metric_base_labels)  # type: ignore[attr-defined]
        == pytest.approx(99.0)
    )


def test_fetch_account_snapshot_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)

    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        assert method == "POST"
        assert path == "/trading/balance"
        payload = {"status": "Fail", "errors": [{"code": 4002, "message": "Invalid signature"}]}
        adapter._ensure_success(payload, status=403, endpoint=path, signed=True)
        return payload

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)
    monkeypatch.setattr(ZondaSpotAdapter, "_fetch_price_map", lambda self: {})

    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()


def test_fetch_account_snapshot_converts_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_signed_request(self, method: str, path: str, *, params=None, data=None):
        assert method == "POST"
        assert path == "/trading/balance"
        return {
            "status": "Ok",
            "balances": [
                {"currency": "BTC", "available": "0.5", "locked": "0.1"},
                {"currency": "PLN", "available": "1000", "locked": "0"},
                {"currency": "USDT", "available": "200", "locked": "0"},
            ],
        }

    def fake_public_request(self, path: str, *, params=None, method="GET"):
        assert path == "/trading/ticker"
        return {
            "status": "Ok",
            "items": {
                "BTC-PLN": {"rate": "100000"},
                "USDT-PLN": {"rate": "4.2"},
                "EUR-PLN": {"rate": "4.5"},
            },
        }

    monkeypatch.setattr(ZondaSpotAdapter, "_signed_request", fake_signed_request)
    monkeypatch.setattr(ZondaSpotAdapter, "_public_request", fake_public_request)

    credentials = ExchangeCredentials(
        key_id="paper",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )

    adapter = ZondaSpotAdapter(
        credentials,
        settings={"valuation_asset": "EUR", "secondary_valuation_assets": ["PLN", "USDT"]},
    )

    snapshot = adapter.fetch_account_snapshot()

    assert snapshot.balances == {
        "BTC": pytest.approx(0.6),
        "PLN": pytest.approx(1000.0),
        "USDT": pytest.approx(200.0),
    }
    assert snapshot.total_equity == pytest.approx(13742.222222, rel=1e-6)
    assert snapshot.available_margin == pytest.approx(11519.999999, rel=1e-6)


def test_signed_request_maps_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="key",
        secret="secret",
        permissions=("read",),
        environment=Environment.LIVE,
    )
    metrics = MetricsRegistry()
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"status":"Fail","errors":[{"message":"auth"}]}'),
        )

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)

    with pytest.raises(ExchangeAuthError):
        adapter.fetch_account_snapshot()

    api_errors = adapter._metric_api_errors.value(  # type: ignore[attr-defined]
        labels={**adapter._metric_base_labels, "reason": "auth"},
    )
    assert api_errors == pytest.approx(1.0)


def test_public_request_raises_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    metrics = MetricsRegistry()
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.sleep", lambda _: None)

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise URLError("boom")

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)

    with pytest.raises(ExchangeNetworkError):
        adapter.fetch_symbols()

    retries = adapter._metric_retries.value(  # type: ignore[attr-defined]
        labels={**adapter._metric_base_labels, "reason": "network"},
    )
    expected_retries = adapter._watchdog.retry_policy.max_attempts * 3.0  # type: ignore[attr-defined]
    assert retries == pytest.approx(expected_retries)
    api_errors = adapter._metric_api_errors.value(  # type: ignore[attr-defined]
        labels={**adapter._metric_base_labels, "reason": "network"},
    )
    expected_api_errors = float(adapter._watchdog.retry_policy.max_attempts)  # type: ignore[attr-defined]
    assert api_errors == pytest.approx(expected_api_errors)


def test_public_request_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="public", environment=Environment.LIVE)
    metrics = MetricsRegistry()
    adapter = ZondaSpotAdapter(credentials, metrics_registry=metrics)

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(
            request.full_url,
            500,
            "Server Error",
            {},
            BytesIO(b'{"status":"Error","errors":[{"message":"oops"}]}'),
        )

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", fake_urlopen)

    with pytest.raises(ExchangeAPIError):
        adapter.fetch_symbols()

    api_errors = adapter._metric_api_errors.value(  # type: ignore[attr-defined]
        labels={**adapter._metric_base_labels, "reason": "server_error"},
    )
    assert api_errors == pytest.approx(1.0)


def test_zonda_spot_adapter_global_throttle_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE))

    monotonic_now = [2_000.0]

    def fake_monotonic() -> float:
        return monotonic_now[0]

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.monotonic", fake_monotonic)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.sleep", lambda *_: None)

    def throttle_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", throttle_urlopen)

    with pytest.raises(ExchangeThrottlingError):
        adapter._public_request("/trading/ticker")

    status = adapter.failover_status()
    assert status["throttle_active"] is True

    success_calls = 0

    def success_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal success_calls
        success_calls += 1
        return _FakeResponse({"status": "Ok", "items": {}})

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", success_urlopen)

    with pytest.raises(ExchangeThrottlingError):
        adapter._public_request("/trading/ticker")
    assert success_calls == 0

    monotonic_now[0] += status["throttle_remaining"] + 0.5
    payload = adapter._public_request("/trading/ticker")
    assert payload == {"status": "Ok", "items": {}}
    assert adapter.failover_status()["throttle_active"] is False
    assert success_calls == 1


def test_zonda_spot_adapter_reconnect_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ZondaSpotAdapter(ExchangeCredentials(key_id="k", secret="s", environment=Environment.LIVE))

    monotonic_now = [4_000.0]

    def fake_monotonic() -> float:
        return monotonic_now[0]

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.monotonic", fake_monotonic)
    monkeypatch.setattr("bot_core.exchanges.zonda.spot.time.sleep", lambda *_: None)

    calls = 0

    def failing_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal calls
        calls += 1
        raise URLError("down")

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", failing_urlopen)

    with pytest.raises(ExchangeNetworkError):
        adapter._public_request("/trading/ticker")
    assert calls >= 3

    status = adapter.failover_status()
    assert status["reconnect_required"] is True

    with pytest.raises(ExchangeNetworkError):
        adapter._public_request("/trading/ticker")

    success_calls = 0

    def success_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal success_calls
        success_calls += 1
        return _FakeResponse({"status": "Ok", "items": {}})

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", success_urlopen)
    monotonic_now[0] += status["reconnect_remaining"] + 0.5

    payload = adapter._public_request("/trading/ticker")
    assert payload == {"status": "Ok", "items": {}}
    assert adapter.failover_status()["reconnect_required"] is False
    assert success_calls == 1
