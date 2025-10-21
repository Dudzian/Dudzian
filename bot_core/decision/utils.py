"""Narzędzia wspólne dla modułów Decision Engine."""
from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    """Bezpiecznie konwertuje dowolną wartość na ``float``.

    Funkcja akceptuje liczby, łańcuchy oraz typy konwertowalne do ``float``.
    W przypadku niepowodzenia zwraca ``None`` zamiast zgłaszać wyjątek.
    """

    if value is None:
        return None
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


__all__ = ["coerce_float"]
