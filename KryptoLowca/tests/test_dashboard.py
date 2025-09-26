from types import SimpleNamespace

from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.dashboard import DashboardApp, DashboardController


class DummyConfigManager:
    def load_strategy_config(self):
        return StrategyConfig.presets()["SAFE"].validate()


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


def test_dashboard_app_headless_updates_state(tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("line1\nline2\n", encoding="utf-8")
    app = DashboardApp(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        log_path=log_file,
    )
    app.refresh_strategy()
    assert app.state.strategy["mode"] == "demo"
    app.update_metrics({"test": 123})
    assert app.state.metrics["test"] == 123
    app.tail_logs()
    assert len(app.state.logs) == 2
    app.append_alert({"message": "test", "severity": "INFO"})
    assert app.state.alerts[-1]["message"] == "test"


def test_dashboard_controller_collects_metrics(tmp_path):
    controller = DashboardController(
        config_manager=DummyConfigManager(),
        ai_manager=DummyAIManager(),
        exchange_manager=DummyExchangeManager(),
        risk_manager=DummyRiskManager(),
        headless=True,
        log_path=tmp_path / "bot.log",
        refresh_interval=1.0,
    )
    metrics = controller._collect_metrics()
    assert "exchange_api" in metrics
    assert "ai_schedules" in metrics
    assert "risk_state" in metrics
