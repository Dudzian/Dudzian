"""Domyślne profile konfiguracji modeli wykorzystywane przez moduł AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class ModelPreset:
    """Opisuje nazwany preset modelu wraz z etykietą wersji."""

    name: str
    version: str

    def as_tag(self) -> str:
        """Zwraca tag w formacie ``name:version`` używany w logach i telemetrii."""

        return f"{self.name}:{self.version}"


BASELINE_PRESET: Final[ModelPreset] = ModelPreset(name="baseline", version="v1")
__all__ = ["ModelPreset", "BASELINE_PRESET"]
