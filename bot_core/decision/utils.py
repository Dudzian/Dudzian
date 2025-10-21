"""Common decision-engine utilities."""
from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    """Attempt to coerce various value types to ``float``."""

    if value is None:
        return None
    if isinstance(value, bool):  # bool dziedziczy po int; traktujemy go osobno
        return float(value)
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensywna gałąź
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - brak konwersji
        return None


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["coerce_float"]
