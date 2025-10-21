"""Narzędzia wspólne dla modułów Decision Engine."""
"""Narzędzia wspólne dla modułów decision engine."""
"""Common decision-engine utilities."""
from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
    """Bezpiecznie konwertuje dowolną wartość na ``float``.

    Funkcja akceptuje liczby, łańcuchy oraz typy konwertowalne do ``float``.
    W przypadku niepowodzenia zwraca ``None`` zamiast zgłaszać wyjątek.
    """Próbuje zinterpretować przekazaną wartość jako liczbę zmiennoprzecinkową."""
    """Attempt to coerce various value types to ``float``.

    Mirrors parsing rules previously duplicated across decision modules:
    - ``None`` stays ``None``.
    - Numeric values are cast defensively to ``float``.
    - Strings are stripped; empty strings result in ``None``; other strings
      are parsed as floats when possible.
    - All other inputs fall back to ``None``.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensywna gałąź
        except (TypeError, ValueError):  # pragma: no cover - defensywne
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
    except (TypeError, ValueError):  # pragma: no cover - brak konwersji
        return None
    return None


__all__ = ["coerce_float"]
