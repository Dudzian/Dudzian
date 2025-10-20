"""Warstwa kompatybilności dla modułów trading GUI."""

from __future__ import annotations

from .controller import TradingSessionController
from .license_context import COMMUNITY_NOTICE, LicenseUiContext, build_license_ui_context
from .state import AppState

__all__ = [
    "TradingSessionController",
    "AppState",
    "COMMUNITY_NOTICE",
    "LicenseUiContext",
    "build_license_ui_context",
]

