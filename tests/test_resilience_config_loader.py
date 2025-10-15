from __future__ import annotations

from bot_core.config import load_core_config


def test_resilience_config_section_loaded() -> None:
    config = load_core_config("config/core.yaml")
    assert config.resilience is not None
    assert config.resilience.enabled is True
    assert len(config.resilience.drills) >= 2
    first = config.resilience.drills[0]
    assert first.dataset_path.endswith("binance_failover.json")
    assert first.thresholds.max_latency_ms > 0
