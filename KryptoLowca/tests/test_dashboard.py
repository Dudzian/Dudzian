from types import SimpleNamespace
from typing import Any, Dict, Sequence

import pytest

try:  # pragma: no cover - zależność opcjonalna
    from bot_core.market_intel import MarketIntelAggregator
except Exception:  # pragma: no cover - fallback gdy brak modułu
    MarketIntelAggregator = None  # type: ignore[assignment]

from bot_core.config.models import DailyTrendMomentumStrategyConfig
from KryptoLowca.dashboard import DashboardApp, DashboardController
from KryptoLowca.runtime.bootstrap import FrontendBootstrap


class DummyConfigManager:
    def load_strategy_config(self):
        return DailyTrendMomentumStrategyConfig(
            name="trend",
            fast_ma=9,
            slow_ma=21,
            breakout_lookback=20,
            momentum_window=14,
            atr_window=14,
            atr_multiplier=1.5,
            min_trend_strength=0.2,
            min_momentum=0.1,
        )


class DummyAIManager:
    def __init__(self):
        self._recent_signals = {"btcusdt": [0.1, -0.1, 0.2]}

    def active_schedules(self):
        return {
            "btcusdt": SimpleNamespace(interval_seconds=60.0, model_types=("rf",), seq_len=32),
        }


class DummyExchangeManager:
    def get_api_metrics(self):
        return {"total_calls": 10, "total_errors": 0}


class DummyRiskManager:
    def latest_guard_state(self):
        return {"state": "ok", "cooldown": False}


@pytest.fixture
def dummy_market_intel() -> Any:
    if MarketIntelAggregator is None:  # pragma: no cover - środowisko bez modułu bot_core
        return None

    class _MemoryStorage:
        def __init__(self) -> None:
            self._payload: Dict[str, Dict[str, Sequence[Sequence[float]]]] = {}

        def read(self, key: str) -> Dict[str, Sequence[Sequence[float]]]:
            return self._payload.get(
                key,
                {
                    "columns": ("open_time", "close", "volume"),
                    "rows": (
                        (0.0, 100.0, 5.0),
                        (60.0, 101.0, 4.5),
                        (120.0, 99.5, 4.0),
                    ),
                },
            )

        def write(self, key: str, payload: Dict[str, Sequence[Sequence[float]]]) -> None:
            self._payload[key] = payload

        def metadata(self) -> Dict[str, str]:
            return {}

        def latest_timestamp(self, key: str) -> float | None:  # noqa: ARG002
            return 120.0

    storage = _MemoryStorage()
    return MarketIntelAggregator(storage)  # type: ignore[call-arg]


def test_dashboard_app_headless_updates_state(tmp_path, dummy_market_intel):
    log_file = tmp_path / "bot.log"
    log_file.write_text("line1\nline2\n", encoding="utf-8")
    app = DashboardApp(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        log_path=log_file,
        market_intel=dummy_market_intel,
    )
    app.refresh_strategy()
    assert app.state.strategy["name"] == "trend"
    assert app.state.strategy["fast_ma"] == 9
    app.update_metrics({"test": 123})
    assert app.state.metrics["test"] == 123
    app.tail_logs()
    assert len(app.state.logs) == 2
    app.append_alert({"message": "test", "severity": "INFO"})
    assert app.state.alerts[-1]["message"] == "test"


def test_dashboard_controller_collects_metrics(tmp_path, dummy_market_intel):
    controller = DashboardController(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        log_path=tmp_path / "bot.log",
        refresh_interval=1.0,
        market_intel=dummy_market_intel,
    )
    metrics = controller._collect_metrics()
    assert "exchange_api" in metrics
    assert "ai_schedules" in metrics
    assert "risk_state" in metrics


def test_dashboard_controller_reuses_frontend_services(monkeypatch, dummy_market_intel):
    if dummy_market_intel is None:
        pytest.skip("MarketIntelAggregator not available in this environment")

    from KryptoLowca.dashboard import desktop as dashboard_module

    def _fail_bootstrap(*_args, **_kwargs):  # pragma: no cover - powinno zostać zastąpione
        raise AssertionError("bootstrap_frontend_services should not be called")

    monkeypatch.setattr(
        dashboard_module,
        "bootstrap_frontend_services",
        _fail_bootstrap,
    )

    services = FrontendBootstrap(
        exchange_manager=DummyExchangeManager(),
        market_intel=dummy_market_intel,
    )

    controller = DashboardController(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        frontend_services=services,
    )

    assert controller.frontend_services is services
    assert controller.market_intel is dummy_market_intel


def test_dashboard_app_reuses_frontend_services(monkeypatch, dummy_market_intel):
    if dummy_market_intel is None:
        pytest.skip("MarketIntelAggregator not available in this environment")

    from KryptoLowca.dashboard import desktop as dashboard_module

    def _fail_bootstrap(*_args, **_kwargs):  # pragma: no cover - powinno zostać zastąpione
        raise AssertionError("bootstrap_frontend_services should not be called")

    monkeypatch.setattr(
        dashboard_module,
        "bootstrap_frontend_services",
        _fail_bootstrap,
    )

    services = FrontendBootstrap(
        exchange_manager=DummyExchangeManager(),
        market_intel=dummy_market_intel,
    )
    app = DashboardApp(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        frontend_services=services,
    )

    assert app.frontend_services is services
    assert app.market_intel is dummy_market_intel
