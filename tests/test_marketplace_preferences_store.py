from pathlib import Path

from bot_core.marketplace.preferences import PresetPreferenceStore


def test_preference_store_persists_entries(tmp_path: Path) -> None:
    store_path = tmp_path / "preferences.json"
    store = PresetPreferenceStore(store_path)

    entry = store.set_entry(
        "alpha",
        "portfolio-main",
        preferences={"risk_target": "balanced", "budget": 1500},
        overrides={"alpha-strategy": {"risk_multiplier": 1.2}},
    )

    assert entry["preferences"]["budget"] == 1500
    assert entry["overrides"]["alpha-strategy"]["risk_multiplier"] == 1.2

    reloaded = PresetPreferenceStore(store_path)
    stored_entry = reloaded.entry("alpha", "portfolio-main")
    assert stored_entry is not None
    assert stored_entry["preferences"]["risk_target"] == "balanced"
    assert stored_entry["overrides"]["alpha-strategy"]["risk_multiplier"] == 1.2

    assert reloaded.clear_entry("alpha", "portfolio-main") is True
    assert reloaded.entry("alpha", "portfolio-main") is None

