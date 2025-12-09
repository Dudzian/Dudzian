"""Shim zachowujący kompatybilność po przeniesieniu instalatora presetów."""
from __future__ import annotations

from bot_core.strategies.presets.installer import (  # noqa: F401
    MarketplaceInstallResult,
    MarketplacePresetInstaller,
)

__all__ = ["MarketplacePresetInstaller", "MarketplaceInstallResult"]
