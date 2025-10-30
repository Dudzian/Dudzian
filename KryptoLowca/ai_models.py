"""Shim zachowujący kompatybilność po przeniesieniu modułu do ``bot_core.ai``.

Moduł ``KryptoLowca.ai_models`` został oznaczony jako przestarzały.  Zaimplementowano
pełną wersję modeli w ``bot_core.ai.legacy_models`` i nowe komponenty powinny
importować właśnie ten moduł.  Ten plik pozostaje wyłącznie w celu zapewnienia
kompatybilności dla istniejących integracji oraz testów."""
from __future__ import annotations

import warnings

from bot_core.ai import legacy_models as _legacy_models
from bot_core.ai.legacy_models import *  # noqa: F401,F403 - re-eksport pełnego API

__all__ = getattr(_legacy_models, "__all__", tuple())

warnings.warn(
    "KryptoLowca.ai_models jest przestarzałe – użyj bot_core.ai.legacy_models",
    DeprecationWarning,
    stacklevel=2,
)
