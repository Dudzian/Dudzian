from __future__ import annotations

from pathlib import Path

from bot_core.config.loader import load_runtime_app_config


def test_load_runtime_app_config_sample(tmp_path: Path) -> None:
    sample = Path(__file__).resolve().parents[1] / "config" / "runtime.yaml"
    runtime_copy = tmp_path / "runtime.yaml"
    runtime_copy.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    config = load_runtime_app_config(runtime_copy)

    assert config.core.path == "core.yaml"
    assert config.core.resolved_path == str(runtime_copy.parent / "core.yaml")
    assert config.ai.model_registry_path.endswith("models")
    assert config.trading.default_entrypoint in config.trading.entrypoints
    entry = config.trading.entrypoints[config.trading.default_entrypoint]
    assert entry.environment == "binance_paper"
    assert config.risk.decision_log is not None
    assert config.licensing.license is not None
    assert config.ui.theme == "dark"
    assert config.observability is not None
    assert config.observability.prometheus is not None
    assert config.observability.prometheus.host == "127.0.0.1"
    assert config.optimization is not None
    assert config.optimization.enabled is True
    assert config.optimization.tasks
    first_task = config.optimization.tasks[0]
    assert first_task.strategy
    assert config.marketplace is not None
    assert config.marketplace.enabled is True
    assert config.marketplace.presets_path.endswith("config/marketplace/presets")
    assert isinstance(config.marketplace.signing_keys, dict)
    assert "kraken_paper" in config.execution.paper_profiles
    assert config.execution.paper_profiles["kraken_paper"]["entrypoint"] == "kraken_desktop_paper"
    kraken_metrics = config.execution.paper_profiles["kraken_paper"]["metrics"]
    assert kraken_metrics["health"] == "bot_exchange_health_status"
    assert kraken_metrics["thresholds"]["health_min"] == 1.0
    assert "okx_paper" in config.execution.paper_profiles
    assert config.execution.paper_profiles["okx_paper"]["io_queue"] == "okx_spot"
    assert config.execution.paper_profiles["okx_paper"]["metrics"]["thresholds"]["network_errors_max"] == 1
    assert config.execution.trading_profiles["bybit_desktop"]["entrypoint"] == "bybit_desktop_paper"
