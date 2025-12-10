"""Bazowe narzędzia do obsługi reconnectu i backoffu dla streamingu."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Tuple

from bot_core.exchanges.errors import ExchangeNetworkError

_DEFAULT_BACKOFF_BASE = 0.25
_DEFAULT_BACKOFF_CAP = 2.0
_DEFAULT_JITTER = (0.05, 0.30)


def _normalize_jitter(jitter: Iterable[float] | None) -> Tuple[float, float]:
    if jitter is None:
        return _DEFAULT_JITTER
    try:
        values = tuple(float(value) for value in jitter)
    except (TypeError, ValueError):
        return _DEFAULT_JITTER
    if len(values) != 2:
        return _DEFAULT_JITTER
    low, high = values
    if low < 0:
        low = 0.0
    if high < low:
        high = low
    return (low, high)


@dataclass(slots=True)
class StreamingBackoff:
    """Zarządza stanem reconnectu oraz obliczaniem opóźnień backoffu."""

    base: float = _DEFAULT_BACKOFF_BASE
    cap: float = _DEFAULT_BACKOFF_CAP
    jitter: Tuple[float, float] = _DEFAULT_JITTER

    _backoff_until: float = field(init=False, default=0.0)
    _reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.base = max(0.0, float(self.base))
        self.cap = max(self.base, float(self.cap))
        self.jitter = _normalize_jitter(self.jitter)

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def deadline(self) -> float:
        return self._backoff_until

    def reset(self) -> None:
        self._backoff_until = 0.0
        self._reason = None

    def status(self) -> tuple[bool, float, str | None]:
        now = time.monotonic()
        remaining = max(0.0, self._backoff_until - now)
        return (remaining > 0.0, remaining, self._reason)

    def calculate_delay(self, attempt: int, *, with_jitter: bool = True) -> float:
        base_delay = min(self.base * (2 ** max(0, attempt - 1)), self.cap)
        if with_jitter and self.jitter[1] > 0.0:
            return base_delay + random.uniform(*self.jitter)
        return base_delay

    def register_cooldown(self, duration: float, *, reason: str | None = None) -> None:
        try:
            cooldown = float(duration)
        except (TypeError, ValueError):
            return
        if cooldown <= 0:
            return
        deadline = time.monotonic() + cooldown
        if deadline > self._backoff_until:
            self._backoff_until = deadline
            self._reason = reason

    def enforce(self) -> None:
        if self._backoff_until <= 0.0:
            return
        now = time.monotonic()
        if now < self._backoff_until:
            remaining = self._backoff_until - now
            raise ExchangeNetworkError(
                message=(
                    "Adapter oczekuje na ponowne połączenie (pozostało %.2fs, powód=%s)."
                    % (remaining, self._reason or "network")
                ),
                reason=None,
            )
        self.reset()

