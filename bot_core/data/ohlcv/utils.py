"""Wspólne funkcje pomocnicze dla modułów OHLCV."""

from __future__ import annotations

import re

_INTERVAL_PATTERN = re.compile(r"(?P<value>\d+)(?P<unit>[smhdw])", re.IGNORECASE)


def interval_to_minutes(interval: str) -> int:
    """Konwertuje zapis interwału (np. ``1h``) na liczbę minut."""

    match = _INTERVAL_PATTERN.fullmatch(interval.strip())
    if not match:
        raise ValueError(f"Nieprawidłowy interwał OHLCV: {interval!r}")
    value = int(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "s":
        return max(1, value // 60)
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 24 * 60
    if unit == "w":
        return value * 7 * 24 * 60
    raise ValueError(f"Nieobsługiwany interwał OHLCV: {interval!r}")


__all__ = ["interval_to_minutes"]
