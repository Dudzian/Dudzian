"""Pakiet OEM zawierający zapakowane modele AI Decision Engine.

Moduł udostępnia główną klasę ``AIModels`` zgodną z historycznym API,
jednocześnie pozwalając dystrybucyjnym buildom na dołączenie artefaktów
modeli w katalogu ``packaged``.  W środowisku developerskim katalog może
pozostawać pusty – wówczas ``OEM_MODEL_REPOSITORY`` przyjmuje wartość
``None`` i system AI przełącza się na fallbackowe modele.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from bot_core.ai.legacy_models import AIModels as _LegacyAIModels

__all__ = ["AIModels", "OEM_MODEL_REPOSITORY"]


def _detect_packaged_repository() -> Path | None:
    """Zwraca ścieżkę do zapakowanego repozytorium modeli (jeśli istnieje)."""

    module_dir = Path(__file__).resolve().parent
    # Katalog ``packaged`` jest tworzony podczas procesu bundlowania OEM.
    candidate = module_dir / "packaged"
    manifest = candidate / "manifest.json"
    if manifest.exists():
        return candidate
    return None


OEM_MODEL_REPOSITORY: Final[Path | None] = _detect_packaged_repository()


class AIModels(_LegacyAIModels):
    """Alias zachowujący kompatybilność ze starszym API importującym ``ai_models``."""


