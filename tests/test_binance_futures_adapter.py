"""Testy jednostkowe adaptera Binance Futures."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest


from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.binance import futures as futures_module
from bot_core.exchanges.binance.futures import (
    BinanceFuturesAdapter,
    FundingRateEvent,
    FuturesPosition,
)
from bot_core.exchanges.errors import ExchangeAuthError, ExchangeNetworkError, ExchangeThrottlingError
from bot_core.observability.metrics import MetricsRegistry


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


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, operation: str, func):
        self.calls.append(operation)
        return func()


def _build_credentials() -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.TESTNET,
    )


def test_fetch_account_snapshot_parses_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured_requests.append(request)
        if "/fapi/v2/account" in request.full_url:
            payload: dict[str, Any] = {
                "assets": [
                    {"asset": "USDT", "walletBalance": "1200", "availableBalance": "1100"},
                    {"asset": "BUSD", "walletBalance": "50", "availableBalance": "50"},
                ],
                "totalMarginBalance": "1300",
                "totalAvailableBalance": "1150",
                "totalMaintMargin": "120",
            }
        else:
            raise AssertionError(f"Unexpected endpoint requested: {request.full_url}")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_000.0)

    registry = MetricsRegistry()
    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)

    snapshot = adapter.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.balances["USDT"] == pytest.approx(1200.0)
    assert snapshot.balances["BUSD"] == pytest.approx(50.0)
    assert snapshot.total_equity == pytest.approx(1300.0)
    assert snapshot.available_margin == pytest.approx(1150.0)
    assert snapshot.maintenance_margin == pytest.approx(120.0)
    assert any("signature=" in req.full_url for req in captured_requests)
    signed_counter = registry.counter(
        "binance_futures_signed_requests_total",
        "Łączna liczba podpisanych zapytań HTTP wysłanych do API Binance Futures.",
    )
    assert signed_counter.value(labels={"exchange": "binance_futures", "environment": "testnet"}) == pytest.approx(1.0)


def test_fetch_symbols_filters_non_trading(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING"},
            {"symbol": "ETHUSDT", "status": "TRADING"},
            {"symbol": "ADAUSDT", "status": "BREAK"},
        ]
    }

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert request.full_url.endswith("/fapi/v1/exchangeInfo")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)

    adapter = BinanceFuturesAdapter(_build_credentials())
    symbols = adapter.fetch_symbols()
    assert list(symbols) == ["BTCUSDT", "ETHUSDT"]


def test_fetch_ohlcv_casts_values(monkeypatch: pytest.MonkeyPatch) -> None:
    klines = [
        [1, "100", "110", "90", "105", "1000"],
        [2, "105", "115", "95", "110", "800"],
    ]

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert "/fapi/v1/klines" in request.full_url
        return _FakeResponse(klines)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)

    adapter = BinanceFuturesAdapter(_build_credentials())
    candles = adapter.fetch_ohlcv("BTCUSDT", "1h")
    assert candles == [
        [1.0, 100.0, 110.0, 90.0, 105.0, 1000.0],
        [2.0, 105.0, 115.0, 95.0, 110.0, 800.0],
    ]


def test_fetch_funding_rates_updates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    payload = [
        {
            "symbol": "BTCUSDT",
            "fundingRate": "0.00025",
            "fundingTime": 1_700_000_000_000,
            "markPrice": "21000.5",
            "nextFundingTime": 1_700_000_108_000,
            "interestRate": "0.0001",
        },
        {
            "symbol": "ETHUSDT",
            "fundingRate": "-0.00045",
            "fundingTime": "1700000005000",
        },
    ]

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert "/fapi/v1/fundingRate" in request.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)

    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)
    events = adapter.fetch_funding_rates(symbol="BTCUSDT", limit=2)

    assert isinstance(events, Sequence)
    assert len(events) == 2
    first = events[0]
    assert isinstance(first, FundingRateEvent)
    assert first.symbol == "BTCUSDT"
    assert first.funding_rate == pytest.approx(0.00025)
    assert first.mark_price == pytest.approx(21000.5)
    assert first.next_funding_time == 1_700_000_108_000

    second = events[1]
    assert second.symbol == "ETHUSDT"
    assert second.funding_time == 1_700_000_005_000
    assert second.mark_price is None

    gauge = registry.gauge(
        "binance_futures_funding_rate",
        "Ostatnio zarejestrowane stopy finansowania na Binance Futures.",
    )
    btc_labels = {"exchange": "binance_futures", "environment": "testnet", "symbol": "BTCUSDT"}
    eth_labels = {"exchange": "binance_futures", "environment": "testnet", "symbol": "ETHUSDT"}
    assert gauge.value(labels=btc_labels) == pytest.approx(0.00025)
    assert gauge.value(labels=eth_labels) == pytest.approx(-0.00045)


def test_place_order_builds_signed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: Request | None = None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal captured_request
        captured_request = request
        payload = {
            "orderId": 123,
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_001.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    request = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=1.0, order_type="LIMIT", price=10.0)
    result = adapter.place_order(request)

    assert result.order_id == "123"
    assert result.status == "NEW"
    assert result.avg_price is None
    assert captured_request is not None
    headers = {name.lower(): value for name, value in captured_request.header_items()}
    assert headers["x-mbx-apikey"] == "test-key"
    assert captured_request.method == "POST"


def test_cancel_order_requires_symbol() -> None:
    adapter = BinanceFuturesAdapter(_build_credentials())
    with pytest.raises(ValueError):
        adapter.cancel_order("1")


def test_cancel_order_accepts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        payload = {"status": "CANCELED"}
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_002.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    adapter.cancel_order("1", symbol="BTCUSDT")


def test_create_listen_key_validates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        captured.append(request)
        assert request.full_url.endswith("/fapi/v1/listenKey")
        return _FakeResponse({"listenKey": "abc"})

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_010.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    listen_key = adapter.create_listen_key()
    assert listen_key == "abc"
    assert captured and (captured[0].method or "POST").upper() == "POST"


def test_create_listen_key_maps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"code": -2015, "msg": "Invalid key"})

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)

    adapter = BinanceFuturesAdapter(_build_credentials())

    with pytest.raises(ExchangeAuthError):
        adapter.create_listen_key()


def test_keepalive_listen_key_requires_value() -> None:
    adapter = BinanceFuturesAdapter(_build_credentials())
    with pytest.raises(ValueError):
        adapter.keepalive_listen_key("")


def test_keepalive_listen_key_raises_on_unexpected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"code": -1000})

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_011.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    with pytest.raises(RuntimeError):
        adapter.keepalive_listen_key("abc")


def test_close_listen_key_accepts_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({})

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_012.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    adapter.close_listen_key("abc")


def test_close_listen_key_rejects_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"code": -1021})

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_013.0)

    adapter = BinanceFuturesAdapter(_build_credentials())
    with pytest.raises(RuntimeError):
        adapter.close_listen_key("abc")


def test_fetch_positions_parses_payload_and_updates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    payload = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "entryPrice": "20000",
            "markPrice": "20500",
            "unRealizedProfit": "250",
            "leverage": "10",
            "isolated": "true",
            "liquidationPrice": "15000",
        },
        {
            "symbol": "ETHUSDT",
            "positionAmt": "-1.2",
            "entryPrice": "1500",
            "markPrice": "1490",
            "unRealizedProfit": "-12",
            "marginType": "cross",
            "leverage": "5",
        },
        {
            "symbol": "XRPUSDT",
            "positionAmt": "0",
            "entryPrice": "0.5",
            "markPrice": "0.6",
        },
    ]

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert request.full_url.startswith("https://testnet.binancefuture.com/fapi/v2/positionRisk")
        return _FakeResponse(payload)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_014.0)

    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)
    positions = adapter.fetch_positions()

    assert len(positions) == 2
    first = positions[0]
    assert first.symbol == "BTCUSDT"
    assert first.side == "long"
    assert first.isolated is True
    assert first.notional == pytest.approx(0.5 * 20500)
    assert first.liquidation_price == pytest.approx(15000.0)

    second = positions[1]
    assert second.symbol == "ETHUSDT"
    assert second.side == "short"
    assert second.isolated is False
    assert second.notional == pytest.approx(abs(-1.2 * 1490))

    per_symbol_gauge = registry.gauge("binance_futures_position_notional", "")
    btc_labels = {
        "exchange": "binance_futures",
        "environment": "testnet",
        "symbol": "BTCUSDT",
        "side": "long",
    }
    eth_labels = {
        "exchange": "binance_futures",
        "environment": "testnet",
        "symbol": "ETHUSDT",
        "side": "short",
    }
    assert per_symbol_gauge.value(labels=btc_labels) == pytest.approx(first.notional)
    assert per_symbol_gauge.value(labels=eth_labels) == pytest.approx(second.notional)

    long_total = registry.gauge("binance_futures_long_notional_total", "")
    short_total = registry.gauge("binance_futures_short_notional_total", "")
    gross_total = registry.gauge("binance_futures_gross_notional", "")
    net_total = registry.gauge("binance_futures_net_notional", "")
    open_positions = registry.gauge("binance_futures_open_positions", "")
    base_labels = {"exchange": "binance_futures", "environment": "testnet"}

    expected_long = first.notional
    expected_short = second.notional
    assert long_total.value(labels=base_labels) == pytest.approx(expected_long)
    assert short_total.value(labels=base_labels) == pytest.approx(expected_short)
    assert gross_total.value(labels=base_labels) == pytest.approx(expected_long + expected_short)
    assert net_total.value(labels=base_labels) == pytest.approx(expected_long - expected_short)
    assert open_positions.value(labels=base_labels) == pytest.approx(2.0)


def test_fetch_positions_resets_closed_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _FakeResponse(
            [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "1",
                    "entryPrice": "21000",
                    "markPrice": "22000",
                }
            ]
        ),
        _FakeResponse(
            [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0",
                    "entryPrice": "21000",
                    "markPrice": "22000",
                }
            ]
        ),
    ]

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        assert responses, "Zbyt wiele wywołań positionRisk"
        return responses.pop(0)

    time_values = iter([1_700_000_020.0, 1_700_000_021.0])
    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: next(time_values))

    registry = MetricsRegistry()
    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)

    adapter.fetch_positions()
    per_symbol_gauge = registry.gauge("binance_futures_position_notional", "")
    base_labels = {
        "exchange": "binance_futures",
        "environment": "testnet",
        "symbol": "BTCUSDT",
        "side": "long",
    }
    assert per_symbol_gauge.value(labels=base_labels) == pytest.approx(22000.0)

    adapter.fetch_positions()
    assert per_symbol_gauge.value(labels=base_labels) == pytest.approx(0.0)


def test_futures_account_snapshot_uses_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="watchdog",
        secret="secret",
        permissions=("read", "trade"),
        environment=Environment.LIVE,
    )
    watchdog = _RecordingWatchdog()
    adapter = BinanceFuturesAdapter(credentials, watchdog=watchdog)

    monkeypatch.setattr(
        adapter,
        "_signed_request",
        lambda *args, **kwargs: {
            "assets": [{"asset": "USDT", "walletBalance": "5"}],
            "totalAvailableBalance": "5",
            "totalMaintMargin": "1",
            "totalMarginBalance": "7",
        },
    )

    snapshot = adapter.fetch_account_snapshot()

    assert snapshot.total_equity == pytest.approx(7.0)
    assert "binance_futures_fetch_account" in watchdog.calls



def test_build_hedging_report_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)

    positions = [
        FuturesPosition(
            symbol="BTCUSDT",
            side="long",
            quantity=0.5,
            entry_price=20000.0,
            mark_price=20500.0,
            notional=10250.0,
            unrealized_pnl=250.0,
            leverage=10.0,
            isolated=True,
            liquidation_price=15000.0,
        ),
        FuturesPosition(
            symbol="ETHUSDT",
            side="short",
            quantity=-1.0,
            entry_price=1500.0,
            mark_price=1490.0,
            notional=1490.0,
            unrealized_pnl=-10.0,
            leverage=5.0,
            isolated=False,
            liquidation_price=None,
        ),
    ]

    def fake_fetch(self):
        return positions

    monkeypatch.setattr(BinanceFuturesAdapter, "fetch_positions", fake_fetch)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.time", lambda: 1_700_000_030.0)

    report = adapter.build_hedging_report()
    assert report["exchange"] == "binance_futures"
    assert report["environment"] == "testnet"
    assert report["valuation_asset"] == "USDT"
    assert report["timestamp"] == 1_700_000_030_000

    summary = report["summary"]
    assert summary["open_positions"] == 2
    assert summary["gross_notional"] == pytest.approx(10250.0 + 1490.0)
    assert summary["long_notional"] == pytest.approx(10250.0)
    assert summary["short_notional"] == pytest.approx(1490.0)
    assert summary["net_notional"] == pytest.approx(10250.0 - 1490.0)

    rendered_positions = report["positions"]
    assert isinstance(rendered_positions, list)
    assert rendered_positions[0]["symbol"] == "BTCUSDT"
    assert rendered_positions[0]["isolated"] is True
    assert rendered_positions[1]["side"] == "short"


def test_execute_request_handles_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    def fake_sleep(seconds: float) -> None:  # pragma: no cover - brak logiki
        return None

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        payload = {"msg": "Too many requests"}
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
        )

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.BinanceFuturesAdapter._sleep", lambda self, s: fake_sleep(s))
    monkeypatch.setattr("bot_core.exchanges.binance.futures.random.uniform", lambda *_args, **_kwargs: 0.0)

    adapter = BinanceFuturesAdapter(_build_credentials())

    with pytest.raises(ExchangeThrottlingError) as exc:
        adapter.fetch_symbols()
    assert "too many" in str(exc.value).lower()
    assert call_count == futures_module._MAX_RETRIES + 1


def test_execute_request_converts_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"{\"msg\": \"unauthorized\"}"),
        )

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)

    adapter = BinanceFuturesAdapter(_build_credentials())
    with pytest.raises(ExchangeAuthError):
        adapter.fetch_symbols()


def test_execute_request_converts_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        raise URLError("connection reset")

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.random.uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.BinanceFuturesAdapter._sleep", lambda self, s: None)

    adapter = BinanceFuturesAdapter(_build_credentials())

    with pytest.raises(ExchangeNetworkError):
        adapter.fetch_symbols()


def test_metrics_recorded_for_public_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    headers = {"X-MBX-USED-WEIGHT-1M": "25"}

    def fake_urlopen(request: Request, timeout: int = 15):  # type: ignore[override]
        return _FakeResponse({"symbols": []}, headers=headers)

    monkeypatch.setattr("bot_core.exchanges.binance.futures.urlopen", fake_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.futures.time.perf_counter", lambda: 1.0)

    adapter = BinanceFuturesAdapter(_build_credentials(), metrics_registry=registry)
    adapter.fetch_symbols()

    gauge = registry.gauge(
        "binance_futures_used_weight",
        "Ostatnie wartości nagłówków X-MBX-USED-WEIGHT z API Binance Futures.",
    )
    assert gauge.value(labels={"exchange": "binance_futures", "environment": "testnet"}) == pytest.approx(25.0)
