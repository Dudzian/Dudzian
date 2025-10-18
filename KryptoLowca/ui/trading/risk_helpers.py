"""Helpers for presenting risk profile information in the UI."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

__all__ = [
    "format_profile_name",
    "format_key_limits",
]


_LIMIT_DEFINITIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("risk_per_trade", ("risk_per_trade", "risk_per_trade_pct", "per_trade"), "per trade"),
    ("portfolio_risk", ("portfolio_risk", "portfolio_risk_pct", "exposure", "max_exposure_pct"), "exposure cap"),
    ("max_daily_loss_pct", ("max_daily_loss_pct", "daily_loss_limit", "daily_loss_pct"), "daily loss cap"),
)


def _normalize_percentage(value: Any | None) -> float | None:
    """Return a percentage float regardless of whether the input is given as 0-1 or 0-100."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if -1.0 <= numeric <= 1.0:
        numeric *= 100.0
    return numeric


def format_profile_name(name: Any | None) -> str:
    """Return a user friendly representation of a risk profile name."""
    if not name:
        return "—"
    try:
        text = str(name)
    except Exception:
        return "—"
    return text.strip() or "—"


def _iter_limit_values(settings: Mapping[str, Any] | None) -> Iterable[tuple[str, float]]:
    if not isinstance(settings, Mapping):
        return []
    pairs: list[tuple[str, float]] = []
    for _canonical, aliases, label in _LIMIT_DEFINITIONS:
        value = None
        for alias in aliases:
            if alias in settings:
                value = settings[alias]
                break
        pct = _normalize_percentage(value)
        if pct is None:
            continue
        pairs.append((label, pct))
    return pairs


def format_key_limits(settings: Mapping[str, Any] | None) -> str:
    """Format risk limits (per-trade, exposure, etc.) into a compact bullet."""
    segments = [f"{pct:.1f}% {label}" for label, pct in _iter_limit_values(settings)]
    return " • ".join(segments) if segments else "—"
