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
    manager = ConfigManager()
    preset = load_preset("trend_following_balanced")

    merged = manager.apply_marketplace_preset(preset.preset_id)
    assert merged["strategy"]["max_leverage"] == 2.0
    assert merged["strategy"]["default_tp"] == 0.06
    ranking = manager.get_marketplace_ranking()
    assert any(entry["preset_id"] == preset.preset_id for entry in ranking)
    risk_labels = manager.get_marketplace_risk_labels()
    assert preset.preset_id in risk_labels
    summary = manager.get_marketplace_risk_summary()
    risk_label = preset.effective_risk_label() or preset.risk_level or "unknown"
    assert risk_label in summary
    assert summary[risk_label]["count"] >= 1
