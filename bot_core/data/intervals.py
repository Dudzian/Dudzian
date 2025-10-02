"""Narzędzia do pracy z interwałami czasowymi strategii/manifestu."""
from __future__ import annotations

_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def normalize_interval_token(token: str | None) -> str | None:
    """Sprowadza zapis interwału do znormalizowanej postaci (np. ``1d``).

    Funkcja akceptuje zarówno notację ``1d``/``4h`` jak i ``D1``/``H4`` i
    sprowadza ją do zapisu z wartością liczbową przed jednostką.
    Zwraca ``None`` dla pustych wartości.
    """

    if token is None:
        return None
    text = token.strip().lower()
    if not text:
        return None
    if len(text) >= 2 and text[0].isalpha() and text[1:].isdigit():
        # format D1/H4 → 1d/4h
        return text[1:] + text[0]
    return text


def interval_to_milliseconds(interval: str) -> int:
    """Konwertuje zapis interwału (np. ``1d``, ``1h``, ``15m``) na milisekundy."""

    token = (interval or "").strip().lower()
    if len(token) < 2:
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'")

    unit = token[-1]
    value_token = token[:-1]
    try:
        value = int(value_token)
    except ValueError as exc:  # pragma: no cover - walidacja defensywna
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'") from exc

    unit_seconds = _UNIT_SECONDS.get(unit)
    if unit_seconds is None:
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'")

    return max(1, value) * unit_seconds * 1000


__all__ = [
    "normalize_interval_token",
    "interval_to_milliseconds",
]
