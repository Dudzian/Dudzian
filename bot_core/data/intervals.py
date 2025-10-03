"""Narzędzia do pracy z interwałami czasowymi strategii/manifestu."""
from __future__ import annotations

_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "M": 2_592_000,  # 30 dni w sekundach – przybliżenie dla interwałów miesięcznych
}


def _canonical_unit(token: str) -> str | None:
    """Zwraca kanoniczny symbol jednostki czasowej."""

    if not token:
        return None
    if token == "M":  # rozróżnij miesiące od minut
        return "M"
    lowered = token.lower()
    if lowered in {"s", "m", "h", "d", "w"}:
        return lowered
    return None


def normalize_interval_token(token: str | None) -> str | None:
    """Sprowadza zapis interwału do znormalizowanej postaci (np. ``1d``).

    Funkcja akceptuje zarówno notację ``1d``/``4h`` jak i ``D1``/``H4`` i
    sprowadza ją do zapisu z wartością liczbową przed jednostką. W przypadku
    miesięcznych interwałów ``1M`` zachowujemy wielką literę ``M`` aby
    rozróżnić je od minut.
    Zwraca ``None`` dla pustych wartości.
    """

    if token is None:
        return None
    text = token.strip()
    if not text:
        return None

    unit: str | None
    value: str
    if len(text) >= 2 and text[0].isalpha() and text[1:].isdigit():
        unit = text[0]
        value = text[1:]
    else:
        unit = text[-1]
        value = text[:-1]

    if not value.isdigit():
        return text.lower()

    canonical_unit = _canonical_unit(unit)
    if canonical_unit is None:
        return text.lower()

    return value + canonical_unit


def interval_to_milliseconds(interval: str) -> int:
    """Konwertuje zapis interwału (np. ``1d``, ``1h``, ``15m``) na milisekundy."""

    token = (interval or "").strip()
    if len(token) < 2:
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'")

    unit = token[-1]
    value_token = token[:-1]
    try:
        value = int(value_token)
    except ValueError as exc:  # pragma: no cover - walidacja defensywna
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'") from exc

    canonical_unit = _canonical_unit(unit)
    if canonical_unit is None:
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'")

    unit_seconds = _UNIT_SECONDS.get(canonical_unit)
    if unit_seconds is None:
        raise ValueError(f"Nieobsługiwany interwał: '{interval}'")

    return max(1, value) * unit_seconds * 1000


__all__ = [
    "normalize_interval_token",
    "interval_to_milliseconds",
]
