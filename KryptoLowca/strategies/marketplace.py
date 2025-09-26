"""Loader presetów strategii wraz z metadanymi marketplace."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - moduł opcjonalny w starszych środowiskach
    from .presets import load_builtin_presets
except Exception:  # pragma: no cover
    def load_builtin_presets():  # type: ignore
        return []


DEFAULT_MARKETPLACE_DIR = Path(__file__).parent / "marketplace"


@dataclass(slots=True)
class StrategyPreset:
    """Struktura opisująca preset strategii dostępny w marketplace."""

    preset_id: str
    name: str
    description: str
    risk_level: str
    recommended_min_balance: float
    timeframe: str
    exchanges: List[str]
    tags: List[str]
    config: Dict[str, object]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_marketplace_presets(base_path: Optional[Path] = None) -> List[StrategyPreset]:
    presets: Dict[str, StrategyPreset] = {
        preset.preset_id: preset for preset in load_builtin_presets()
    }

    directory = base_path or DEFAULT_MARKETPLACE_DIR
    if directory.exists():
        for file in sorted(directory.glob("*.json")):
            data = _load_json(file)
            config = data.get("config", {})
            preset = StrategyPreset(
                preset_id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                risk_level=data.get("risk_level", "unknown"),
                recommended_min_balance=float(data.get("recommended_min_balance", 0.0)),
                timeframe=data.get("timeframe", ""),
                exchanges=list(data.get("exchanges", [])),
                tags=list(data.get("tags", [])),
                config=dict(config),
            )
            presets[preset.preset_id] = preset

    return list(presets.values())


def load_preset(preset_id: str, base_path: Optional[Path] = None) -> StrategyPreset:
    for preset in load_marketplace_presets(base_path=base_path):
        if preset.preset_id == preset_id:
            return preset
    raise FileNotFoundError(f"Preset '{preset_id}' nie istnieje w marketplace")


__all__ = ["StrategyPreset", "load_marketplace_presets", "load_preset"]
