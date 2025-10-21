"""Wspólne funkcje pomocnicze do obliczeń zmienności wykorzystywanych w strategiach."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


def _tail(values: Sequence[float] | Iterable[float], length: int | None) -> list[float]:
    items = [float(value) for value in values]
    if length is None or length <= 0:
        return items
    return items[-length:]


def realized_volatility(returns: Sequence[float] | Iterable[float], *, lookback: int | None = None) -> float:
    """Oblicza zrealizowaną zmienność (odchylenie standardowe log-zwrotów)."""

    data = _tail(returns, lookback)
    if not data:
        return 0.0
    mean_ret = sum(data) / len(data)
    variance = sum((value - mean_ret) ** 2 for value in data) / max(len(data) - 1, 1)
    return sqrt(max(variance, 0.0))


__all__ = ["realized_volatility"]

