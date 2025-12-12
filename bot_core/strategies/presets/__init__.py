"""Pakiet presetów strategii i narzędzi instalacyjnych.

Zapewnia lekkie API do pracy z presetami (walidacja, kreator, instalator)
bez wymuszania importu pełnych implementacji silników strategii.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "StrategyPresetWizard",
    "StrategyPresetDescriptor",
    "StrategyPresetProfile",
    "PresetLicenseStatus",
    "PresetLicenseState",
    "StrategyPresetValidationError",
    "StrategyPresetSchema",
    "MarketplacePresetInstaller",
    "MarketplaceInstallResult",
]

_CATALOG_EXPORTS = {
    "StrategyPresetWizard",
    "StrategyPresetDescriptor",
    "StrategyPresetProfile",
    "PresetLicenseStatus",
    "PresetLicenseState",
    "StrategyPresetValidationError",
    "StrategyPresetSchema",
}
_INSTALLER_EXPORTS = {
    "MarketplacePresetInstaller",
    "MarketplaceInstallResult",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin proxy
    if name in _CATALOG_EXPORTS:
        catalog = importlib.import_module("bot_core.strategies.catalog")
        return getattr(catalog, name)
    if name in _INSTALLER_EXPORTS:
        installer = importlib.import_module("bot_core.strategies.presets.installer")
        return getattr(installer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
