from __future__ import annotations

import json
from datetime import datetime, timezone

from pathlib import Path

from KryptoLowca.strategies.marketplace import (
    StrategyPreset,
    load_marketplace_presets,
    load_preset,
)


def test_load_marketplace_presets_with_metadata(tmp_path: Path) -> None:
    marketplace_dir = tmp_path
    payload = {
        "id": "demo_strategy",
        "name": "Demo Strategy",
        "description": "Testowa strategia do sprawdzenia metadanych.",
        "risk_level": "balanced",
        "recommended_min_balance": 1234.5,
        "timeframe": "1h",
        "exchanges": ["binance"],
        "tags": ["demo"],
        "version": "0.1.0",
        "last_updated": "2024-05-05T10:15:00+00:00",
        "compatibility": {"app": ">=2.7.0"},
        "compliance": {"required_flags": ["compliance_confirmed"]},
        "config": {"strategy": {"preset": "SAFE", "mode": "demo"}},
    }
    (marketplace_dir / "demo_strategy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    presets = load_marketplace_presets(base_path=marketplace_dir)
    assert any(p.preset_id == "demo_strategy" for p in presets)

    preset = load_preset("demo_strategy", base_path=marketplace_dir)
    assert isinstance(preset, StrategyPreset)
    assert preset.version == "0.1.0"
    assert preset.compatibility["app"] == ">=2.7.0"
    assert preset.compliance["required_flags"] == ["compliance_confirmed"]
    assert preset.last_updated == datetime(2024, 5, 5, 10, 15, tzinfo=timezone.utc)
