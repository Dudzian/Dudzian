from __future__ import annotations

from pathlib import Path

from bot_core.config import load_core_config


def test_core_config_includes_stress_lab_section() -> None:
    config_path = Path("config/core.yaml")
    config = load_core_config(config_path)

    stress = config.stress_lab
    assert stress is not None, "stress_lab section should be present in core.yaml"
    assert stress.enabled is True
    assert stress.require_success is True
    assert "BTCUSDT" in stress.datasets
    btc_dataset = stress.datasets["BTCUSDT"]
    assert Path(btc_dataset.metrics_path).name == "btcusdt.json"
    assert btc_dataset.allow_synthetic is True

    scenario_names = {scenario.name for scenario in stress.scenarios}
    assert "cross_market_liquidity_crunch" in scenario_names
    assert "exchange_blackout_and_latency" in scenario_names

    thresholds = stress.thresholds
    assert thresholds.max_liquidity_loss_pct >= 0.6
    assert thresholds.max_dispersion_bps >= 60.0
