"""Common decision-engine utilities."""
from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    """Attempt to coerce ``value`` to ``float`` while guarding failures."""

    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
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
    except (TypeError, ValueError):  # pragma: no cover - conversion failure
        return None


__all__ = ["coerce_float"]
