from __future__ import annotations

from pathlib import Path

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.strategies.marketplace import load_marketplace_presets, load_preset


def test_marketplace_catalog_contains_presets():
    presets = load_marketplace_presets()
    ids = {preset.preset_id for preset in presets}
    assert "trend_following_balanced" in ids
    assert "mean_reversion_safe" in ids


def test_apply_marketplace_preset_updates_strategy(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    manager = ConfigManager(config_path)
    preset = load_preset("trend_following_balanced")

    merged = manager.apply_marketplace_preset(preset.preset_id)
    assert merged["strategy"]["max_leverage"] == 2.0
    assert merged["strategy"]["default_tp"] == 0.06
    strategy = manager.load_strategy_config()
    assert strategy.max_leverage == 2.0
    assert strategy.preset == "BALANCED"
