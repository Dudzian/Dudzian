"""PySide6 bootstrap for the Stage6 desktop shell."""

from __future__ import annotations

from typing import Any

__all__ = ["AppOptions", "BotPysideApplication"]


def __getattr__(name: str) -> Any:  # pragma: no cover - deleguje import ciężkich modułów
    if name in {"AppOptions", "BotPysideApplication"}:
        from .app import AppOptions, BotPysideApplication

        return {"AppOptions": AppOptions, "BotPysideApplication": BotPysideApplication}[name]
    raise AttributeError(name)
