"""Polityka mapowania błędów i retry dla warstwy egzekucji."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

try:  # pragma: no cover - zależne od obecności modułu giełdowego
    from bot_core.exchanges.errors import (  # type: ignore
        ExchangeAPIError,
        ExchangeAuthError,
        ExchangeNetworkError,
        ExchangeThrottlingError,
    )
except Exception:  # pragma: no cover - defensywny fallback

    class ExchangeAPIError(Exception):
        status_code: int | None = None
        message: str | None = None

    class ExchangeAuthError(ExchangeAPIError):
        pass

    class ExchangeThrottlingError(ExchangeAPIError):
        pass

    class ExchangeNetworkError(Exception):
        pass


def _exp_backoff_with_jitter(attempt: int, *, base: float, cap: float) -> float:
    exp = min(cap, base * (2 ** max(0, attempt - 1)))
    return random.uniform(0.0, exp)


@dataclass(slots=True)
class ExecutionErrorPolicy:
    """Centralne reguły klasyfikacji błędów i retry/backoff."""

    backoff_base: float = 0.05
    backoff_cap: float = 0.5
    retryable_exceptions: Tuple[type[Exception], ...] | None = None
    classify_extra: Mapping[type[Exception], str] | None = None
    backoff_override: Callable[[int, Exception], float] | None = None

    def __post_init__(self) -> None:
        if self.retryable_exceptions is None:
            self.retryable_exceptions = (
                ExchangeNetworkError,
                ExchangeThrottlingError,
            )
        if self.classify_extra is None:
            self.classify_extra = {}

    def classify(self, error: Exception) -> str:
        for exc_type, category in (self.classify_extra or {}).items():
            if isinstance(error, exc_type):
                return category
        if isinstance(error, ExchangeAuthError):
            return "auth"
        if isinstance(error, ExchangeThrottlingError):
            return "throttling"
        if isinstance(error, ExchangeNetworkError):
            return "network"
        if isinstance(error, ExchangeAPIError):
            return "api"
        return "unknown"

    def should_retry(self, error: Exception) -> bool:
        return isinstance(error, self.retryable_exceptions or ())

    def backoff(self, attempt: int, error: Exception, *, allow_unknown: bool = False) -> float:
        if self.backoff_override is not None:
            try:
                return max(0.0, float(self.backoff_override(attempt, error)))
            except Exception:  # pragma: no cover - polityka backoff nie powinna psuć przepływu
                return 0.0

        category = self.classify(error)
        if category in {"network", "throttling"} or (allow_unknown and category == "unknown"):
            return _exp_backoff_with_jitter(
                attempt,
                base=self.backoff_base,
                cap=self.backoff_cap,
            )
        return 0.0


__all__ = [
    "ExecutionErrorPolicy",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeNetworkError",
    "ExchangeThrottlingError",
]
