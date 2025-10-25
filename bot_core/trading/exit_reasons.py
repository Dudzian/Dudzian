"""Canonical exit reason utilities shared across trading components."""
from __future__ import annotations

import re
from typing import Any, Optional, Set

import pandas as pd


class ExitReason:
    """Utility helpers for canonical exit reason values."""

    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MOMENTUM_FADE = "momentum_fade"
    TIME_EXIT = "time_exit"

    _DEFAULT = SIGNAL
    _SUPPORTED: Set[str] = {SIGNAL, STOP_LOSS, TAKE_PROFIT, MOMENTUM_FADE, TIME_EXIT}
    _ALIASES = {
        "stoploss": STOP_LOSS,
        "takeprofit": TAKE_PROFIT,
        "momentumfade": MOMENTUM_FADE,
        "timeexit": TIME_EXIT,
    }

    @classmethod
    def default(cls) -> str:
        """Return the default exit reason used when metadata is missing."""

        return cls._DEFAULT

    @classmethod
    def canonical(cls, raw: Any) -> Optional[str]:
        """Return the canonical representation for a raw exit reason value."""

        if raw is None:
            return None

        # Gracefully handle pandas missing sentinels without raising warnings.
        try:
            if pd.isna(raw):  # type: ignore[arg-type]
                return None
        except TypeError:
            # Some exotic objects (e.g., dicts) do not support ``pd.isna``.
            pass

        text = str(raw).strip()
        if not text or text.upper() == "NA":
            return None

        normalized = re.sub(r"[\s-]+", "_", text, flags=re.UNICODE).lower()
        normalized = cls._ALIASES.get(normalized, normalized)

        if normalized in cls._SUPPORTED:
            return normalized

        return None

    @classmethod
    def supported(cls) -> Set[str]:
        """Return the set of allowed canonical values."""

        return set(cls._SUPPORTED)

    @classmethod
    def normalize(
        cls,
        raw: Any,
        *,
        allow_unknown: bool = False,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Return canonical value or optionally normalized fallback for raw metadata.

        Args:
            raw: Arbitrary exit reason metadata (string, scalar, pandas NA, etc.).
            allow_unknown: When ``True`` unknown values are converted to a sanitized
                ``snake_case`` representation instead of being discarded.
            default: Optional explicit default to return when ``raw`` cannot be
                interpreted. When omitted the method returns ``None`` for
                unsupported inputs unless ``allow_unknown`` is enabled.

        Returns:
            Canonical exit reason string, sanitized fallback, or ``None``.
        """

        canonical = cls.canonical(raw)
        if canonical is not None:
            return canonical

        if default is not None:
            return default

        if not allow_unknown:
            return None

        if raw is None:
            return None

        try:
            if pd.isna(raw):  # type: ignore[arg-type]
                return None
        except TypeError:
            pass

        text = str(raw).strip()
        if not text:
            return None

        return re.sub(r"[\s-]+", "_", text, flags=re.UNICODE).lower()


__all__ = ["ExitReason"]
