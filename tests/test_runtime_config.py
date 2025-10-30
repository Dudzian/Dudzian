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
