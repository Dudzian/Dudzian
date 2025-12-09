"""Minimalne kontrakty dla builderów pipeline'u oraz schedulerów."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Scheduler(Protocol):
    """Minimalny interfejs schedulerów uruchamialnych w runtime."""

    def start(self) -> None:
        """Uruchamia scheduler w aktywnym środowisku."""

    def stop(self) -> None:
        """Zatrzymuje scheduler i porządkuje zasoby."""


@runtime_checkable
class PipelineBuilder(Protocol):
    """Interfejs buildera odpowiedzialnego za tworzenie pipeline'u runtime."""

    def build(self) -> Scheduler:
        """Buduje scheduler gotowy do uruchomienia."""


__all__ = ["PipelineBuilder", "Scheduler"]
