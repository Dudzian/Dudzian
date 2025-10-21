"""Narzędzia wspólne dla modułów decision engine."""
from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    """Próbuje zinterpretować przekazaną wartość jako liczbę zmiennoprzecinkową."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensywne
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


__all__ = ["coerce_float"]
