"""Wspólne stałe dla progów sygnałów autotradera."""
from __future__ import annotations

SUPPORTED_SIGNAL_THRESHOLD_METRICS: tuple[str, ...] = (
    "signal_after_adjustment",
    "signal_after_clamp",
)

__all__ = ["SUPPORTED_SIGNAL_THRESHOLD_METRICS"]
