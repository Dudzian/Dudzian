"""Zgodność wsteczna: re-eksportuje kontekst licencji z nowego pakietu."""

from __future__ import annotations

import warnings

from bot_core.ui.trading.license_context import (
    COMMUNITY_NOTICE,
    LicenseUiContext,
    build_license_ui_context,
)

warnings.warn(
    "KryptoLowca.ui.trading.license_context jest przestarzałe – użyj bot_core.ui.trading.license_context",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["COMMUNITY_NOTICE", "LicenseUiContext", "build_license_ui_context"]
