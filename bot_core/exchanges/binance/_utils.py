"""Wspólne funkcje pomocnicze dla adapterów Binance."""
from __future__ import annotations

import time
from typing import Mapping

__all__ = [
    "_stringify_params",
    "_to_float",
    "_normalize_depth",
    "_timestamp_ms_to_seconds",
]


def _stringify_params(params: Mapping[str, object]) -> list[tuple[str, str]]:
    """Konwertuje wartości parametrów na tekst wymagany przez API Binance."""

    normalized: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, bool):
            normalized.append((key, "true" if value else "false"))
        elif value is None:
            continue
        else:
            normalized.append((key, str(value)))
    return normalized


def _to_float(value: object, default: float = 0.0) -> float:
    """Bezpiecznie konwertuje wartość na ``float`` z wartością domyślną."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalize_depth(depth: int) -> int:
    """Zwraca najbliższą wartość ``limit`` akceptowaną przez API depth."""

    if depth <= 0:
        raise ValueError("Parametr depth musi być dodatni.")

    allowed = (5, 10, 20, 50, 100, 500, 1000)
    for candidate in allowed:
        if depth <= candidate:
            return candidate
    return allowed[-1]


def _timestamp_ms_to_seconds(timestamp: object, *, fallback: float | None = None) -> float:
    """Konwertuje znaczniki czasu Binance (ms) na sekundy."""

    value = _to_float(timestamp, default=0.0)
    if value <= 0:
        return fallback if fallback is not None else time.time()
    if value > 10_000_000_000:  # wartość w ms
        return value / 1000.0
    return value
