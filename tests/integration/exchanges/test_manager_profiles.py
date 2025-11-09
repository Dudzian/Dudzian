from __future__ import annotations

from typing import Any, Sequence

import pytest

import bot_core.exchanges.manager as manager_module
from bot_core.auto_trader.app import AutoTrader
from bot_core.exchanges.manager import ExchangeManager, register_native_adapter, unregister_native_adapter
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.exchanges.core import Mode, OrderDTO, OrderSide, OrderStatus, OrderType
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.signal_quality import SignalQualityReporter as BaseSignalQualityReporter
from bot_core.exchanges.health import HealthCheck, HealthCheckResult, HealthStatus


class _StaticPublicFeed:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._rules = {}

    def load_markets(self) -> dict[str, Any]:
        from bot_core.exchanges.core import MarketRules

        rules = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
                min_amount=0.001,
                max_amount=None,
                min_price=1.0,
                max_price=None,
            )
        }
        self._rules = rules
        return rules

    def get_market_rules(self, symbol: str) -> Any:
        return self._rules.get(symbol)

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": 100.0, "bid": 99.5, "ask": 100.5}

    def fetch_order_book(self, symbol: str, limit: int = 50) -> dict[str, Sequence[tuple[float, float]]]:
        return {
            "asks": [(100.0 + i * 0.1, 0.05) for i in range(limit)],
            "bids": [(100.0 - i * 0.1, 0.05) for i in range(limit)],
        }

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        return [[1_700_000_000.0, 99.0, 101.0, 98.0, 100.0, 1.0] for _ in range(limit)]


class _FailingFuturesAdapter:
    def __init__(self, credentials, *, environment, settings=None, **_: Any) -> None:
        self.credentials = credentials
        self.environment = environment
        self.settings = settings or {}
        self._calls = 0

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 5000.0},
            total_equity=5000.0,
            available_margin=4800.0,
            maintenance_margin=100.0,
        )

    def place_order(self, request: OrderRequest) -> OrderResult:
        self._calls += 1
        raise ExchangeNetworkError("native adapter offline")

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        raise ExchangeNetworkError("native adapter offline")

    def fetch_open_orders(self) -> list[Any]:
        raise ExchangeNetworkError("native adapter offline")


class _CCXTFallback:
    def __init__(self, *, exchange_id: str, **_: Any) -> None:
        self.exchange_id = exchange_id
        self._rules = {}
        self.cancel_calls = 0
        self.fetch_open_orders_calls = 0

    def load_markets(self) -> dict[str, Any]:
        return _StaticPublicFeed().load_markets()

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": 100.0}

    def fetch_order_book(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        return _StaticPublicFeed().fetch_order_book(symbol, limit)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        return _StaticPublicFeed().fetch_ohlcv(symbol, timeframe, limit)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderDTO:
        return OrderDTO(
            id="42",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            mode=Mode.FUTURES,
            extra={
                "order_id": "42",
                "filled_quantity": quantity,
                "avg_price": price or 100.0,
            },
        )

    def cancel_order(self, order_id: Any, symbol: str | None = None) -> bool:
        self.cancel_calls += 1
        return True

    def fetch_open_orders(self, symbol: str | None = None) -> list[OrderDTO]:
        self.fetch_open_orders_calls += 1
        return []

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 5000.0},
            total_equity=5000.0,
            available_margin=4800.0,
            maintenance_margin=100.0,
        )


@pytest.mark.parametrize(
    "exchange_id,profile,expected_variant",
    [
        ("binance", "paper", "spot"),
        ("kraken", "paper", "margin"),
        ("okx", "paper", "futures"),
        ("bybit", "paper", "spot"),
    ],
)
def test_paper_profiles_cover_major_exchanges(exchange_id: str, profile: str, expected_variant: str, monkeypatch, tmp_path):
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", _StaticPublicFeed)

    class ReporterProxy(BaseSignalQualityReporter):
        def __init__(self, *, exchange_id: str, **kwargs: Any) -> None:
            super().__init__(exchange_id=exchange_id, report_dir=tmp_path, **kwargs)

    monkeypatch.setattr(manager_module, "SignalQualityReporter", ReporterProxy)

    manager = ExchangeManager(exchange_id=exchange_id)
    manager.apply_environment_profile(profile, exchange=exchange_id)
    markets = manager.load_markets()
    assert "BTC/USDT" in markets
    price, slip = manager.simulate_vwap_price("BTC/USDT", "buy", 0.5)
    assert price is not None
    assert slip >= 0.0
    assert manager.get_paper_variant() == expected_variant

    description = manager.describe_environment_profile()
    exchange_cfg = description.get("exchange_manager", {})
    assert "rate_limit_rules" in exchange_cfg


def test_failover_switches_to_ccxt_and_emits_signal_report(monkeypatch, tmp_path, request):
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", _StaticPublicFeed)
    monkeypatch.setattr(manager_module, "_CCXTPrivateBackend", _CCXTFallback)

    class ReporterProxy(BaseSignalQualityReporter):
        def __init__(self, *, exchange_id: str, **kwargs: Any) -> None:
            super().__init__(exchange_id=exchange_id, report_dir=tmp_path, **kwargs)

    monkeypatch.setattr(manager_module, "SignalQualityReporter", ReporterProxy)

    register_native_adapter(
        exchange_id="binance",
        mode=Mode.FUTURES,
        factory=_FailingFuturesAdapter,
        default_settings={},
        supports_testnet=True,
    )
    request.addfinalizer(lambda: unregister_native_adapter(exchange_id="binance", mode=Mode.FUTURES))
    manager = ExchangeManager(exchange_id="binance")
    manager.configure_failover(enabled=True, failure_threshold=1, cooldown_seconds=0.0)
    manager.apply_environment_profile("testnet", exchange="binance")
    manager.set_credentials("key", "secret")
    manager.set_mode(futures=True, testnet=True)

    dto = manager.create_order("BTC/USDT", "buy", "limit", 1.0, price=101.0)
    assert dto.extra.get("order_id") == "42"

    cancel_ok = manager.cancel_order("42", symbol="BTC/USDT")
    assert cancel_ok is True

    open_orders = manager.fetch_open_orders("BTC/USDT")
    assert open_orders == []

    summary = manager.describe_signal_quality()
    assert summary["total"] == 1
    assert summary["failures"] == 0
    assert summary["records"][0]["backend"] == "ccxt"
    assert summary["records"][0]["status"] in {"filled", "partial"}

    def _degraded_private_api() -> None:
        raise ExchangeNetworkError("ccxt degraded")

    monitor = manager.create_health_monitor(
        [HealthCheck(name="private_api", check=_degraded_private_api, critical=False)]
    )
    results = monitor.run()
    statuses = {result.name: result.status for result in results}
    assert statuses["private_api"] is HealthStatus.DEGRADED

    updated_summary = manager.describe_signal_quality()
    watchdog_summary = updated_summary.get("watchdog", {})
    last_status = watchdog_summary.get("last_status", {})
    assert last_status.get("private_api", {}).get("status") == "degraded"
    alerts = watchdog_summary.get("alerts", [])
    matching_alerts = [alert for alert in alerts if alert.get("check") == "private_api"]
    assert matching_alerts, f"Brak alertu dla private_api: {alerts}"
    assert matching_alerts[0].get("status") == "degraded"
    assert matching_alerts[0].get("backend") == "ccxt"

    degradation = updated_summary.get("degradation", {})
    assert degradation.get("rolling_score", 0.0) >= 0.6

    private_backend = manager._private
    assert private_backend is not None
    assert private_backend.cancel_calls == 1
    assert private_backend.fetch_open_orders_calls == 1


def test_watchdog_degradation_blocks_trading(monkeypatch, tmp_path, request):
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", _StaticPublicFeed)
    monkeypatch.setattr(manager_module, "_CCXTPrivateBackend", _CCXTFallback)

    class ReporterProxy(BaseSignalQualityReporter):
        def __init__(self, *, exchange_id: str, **kwargs: Any) -> None:
            super().__init__(exchange_id=exchange_id, report_dir=tmp_path, **kwargs)

    monkeypatch.setattr(manager_module, "SignalQualityReporter", ReporterProxy)

    register_native_adapter(
        exchange_id="binance",
        mode=Mode.FUTURES,
        factory=_FailingFuturesAdapter,
        default_settings={},
        supports_testnet=True,
    )
    request.addfinalizer(lambda: unregister_native_adapter(exchange_id="binance", mode=Mode.FUTURES))

    manager = ExchangeManager(exchange_id="binance")
    manager.configure_failover(enabled=True, failure_threshold=1, cooldown_seconds=0.0)
    manager.apply_environment_profile("testnet", exchange="binance")
    manager.set_credentials("key", "secret")
    manager.set_mode(futures=True, testnet=True)

    manager.create_order("BTC/USDT", "buy", "limit", 1.0, price=101.0)

    def _degraded_private_api() -> None:
        raise ExchangeNetworkError("ccxt degraded")

    monitor = manager.create_health_monitor(
        [HealthCheck(name="private_api", check=_degraded_private_api, critical=False)]
    )
    monitor.run()

    summary = manager.describe_signal_quality()
    degradation = summary.get("degradation", {})
    assert degradation.get("rolling_score", 0.0) >= 0.6

    class _Emitter:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def log(self, *_args: Any, **_kwargs: Any) -> None:
            return

        def emit(self, event: str, **payload: Any) -> None:
            self.events.append((event, payload))

    class _Var:
        def __init__(self, value: str) -> None:
            self._value = value

        def get(self) -> str:
            return self._value

    class _GUI:
        def __init__(self) -> None:
            self.timeframe_var = _Var("1h")
            self.ai_mgr = None

        def is_demo_mode_active(self) -> bool:
            return True

    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(emitter, gui, lambda: "BTCUSDT")
    guardrails_cfg = trader._thresholds["auto_trader"].setdefault("signal_guardrails", {})
    guardrails_cfg["signal_quality_degradation"] = {
        "rolling_score": 0.2,
        "kill_switch": 0.5,
        "release": 0.1,
        "max_leverage": 0.25,
    }
    trader.set_signal_quality_provider(manager.describe_signal_quality)
    trader.current_leverage = 1.2

    result_signal = trader._apply_signal_guardrails("buy", 0.2, None)

    assert result_signal == "hold"
    assert trader.current_leverage == 0.0
    assert trader._exchange_degradation_kill_switch is True
    assert trader._last_guardrail_triggers
    assert any(trigger.name == "exchange_degradation" for trigger in trader._last_guardrail_triggers)


def test_degradation_event_triggers_failover_switch(monkeypatch, tmp_path, request):
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", _StaticPublicFeed)
    monkeypatch.setattr(manager_module, "_CCXTPrivateBackend", _CCXTFallback)

    class ReporterProxy(BaseSignalQualityReporter):
        def __init__(self, *, exchange_id: str, **kwargs: Any) -> None:
            super().__init__(exchange_id=exchange_id, report_dir=tmp_path, **kwargs)

    monkeypatch.setattr(manager_module, "SignalQualityReporter", ReporterProxy)

    register_native_adapter(
        exchange_id="binance",
        mode=Mode.FUTURES,
        factory=_FailingFuturesAdapter,
        default_settings={},
        supports_testnet=True,
    )
    request.addfinalizer(lambda: unregister_native_adapter(exchange_id="binance", mode=Mode.FUTURES))

    manager = ExchangeManager(exchange_id="binance")
    manager.set_credentials("key", "secret")
    manager.set_mode(futures=True, testnet=True)

    engaged_events: list[dict[str, Any]] = []

    def _capture(event) -> None:
        engaged_events.append(event.payload)

    manager.event_bus.subscribe("exchange.failover.engaged", _capture)

    def _degraded_native() -> None:
        raise ExchangeNetworkError("native backend degraded")

    monitor = manager.create_health_monitor(
        [HealthCheck(name="native_private_api", check=_degraded_native, critical=False)]
    )

    assert manager._failover_enabled is False
    assert manager._active_backend == "native"

    monitor.run()

    assert manager._failover_enabled is True
    assert manager._active_backend == "ccxt"
    assert manager._failover_auto_enabled is True
    assert engaged_events, "Powinno pojawić się zdarzenie failover engaged"
    assert engaged_events[-1].get("previous_backend") == "native"


def test_exchange_weighting_tracks_signal_quality(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", _StaticPublicFeed)
    monkeypatch.setattr(manager_module, "_CCXTPrivateBackend", _CCXTFallback)

    class ReporterProxy(BaseSignalQualityReporter):
        def __init__(self, *, exchange_id: str, **kwargs: Any) -> None:
            super().__init__(exchange_id=exchange_id, report_dir=tmp_path, **kwargs)

    monkeypatch.setattr(manager_module, "SignalQualityReporter", ReporterProxy)

    manager = ExchangeManager(exchange_id="binance")
    reporter = manager._signal_reporter

    for _ in range(6):
        reporter.record_success(
            backend="native",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            requested_quantity=1.0,
            requested_price=100.0,
            filled_quantity=1.0,
            executed_price=100.0,
            latency=0.05,
        )

    baseline = manager.describe_weighting()
    assert baseline["weight"] > 0.0
    assert baseline["rolling_weight"] >= baseline["weight"] * 0.8

    reporter.record_watchdog_results(
        [
            HealthCheckResult(
                name="private_api",
                status=HealthStatus.DEGRADED,
                latency=0.5,
                details={"degradation_score": 0.9},
            )
        ],
        backend="native",
    )

    degraded = manager.describe_weighting()
    assert degraded["weight"] < baseline["weight"]
    assert degraded["degradation"]["rolling_score"] >= 0.9
