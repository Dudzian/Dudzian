"""Wbudowane presety strategii udostępniane bezpośrednio w repozytorium."""
from __future__ import annotations

from typing import Dict, Iterable

from KryptoLowca.strategies.marketplace import StrategyPreset

from .daily_trend import DAILY_TREND_PRESET
from .trend_following import INTRADAY_TREND_PRESET


_BUILTIN_PRESETS: Dict[str, StrategyPreset] = {
    DAILY_TREND_PRESET.preset_id: DAILY_TREND_PRESET,
    INTRADAY_TREND_PRESET.preset_id: INTRADAY_TREND_PRESET,
}


def load_builtin_presets() -> Iterable[StrategyPreset]:
    """Zwraca iterowalny zbiór presetów (łatwe do rozszerzenia w przyszłości)."""

    return _BUILTIN_PRESETS.values()


def get_builtin_preset(preset_id: str) -> StrategyPreset:
    try:
        return _BUILTIN_PRESETS[preset_id]
    except KeyError as exc:  # pragma: no cover - defensywne
        raise KeyError(f"Brak wbudowanego presetu '{preset_id}'") from exc


__all__ = [
    "StrategyPreset",
    "load_builtin_presets",
    "get_builtin_preset",
    "DAILY_TREND_PRESET",
    "INTRADAY_TREND_PRESET",
]
