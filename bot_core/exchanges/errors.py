"""Dedykowane wyjątki dla adapterów giełdowych."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ExchangeError(RuntimeError):
    """Bazowy wyjątek specyficzny dla adapterów giełdowych."""


@dataclass(slots=True)
class ExchangeAPIError(ExchangeError):
    """Błąd zwrócony przez API giełdy."""

    message: str
    status_code: int
    payload: Any | None = None

    def __str__(self) -> str:  # pragma: no cover - delegacja do RuntimeError
        return f"{self.message} (status={self.status_code})"


@dataclass(slots=True)
class ExchangeAuthError(ExchangeAPIError):
    """Błąd uwierzytelnienia zgłoszony przez API giełdy."""


@dataclass(slots=True)
class ExchangeThrottlingError(ExchangeAPIError):
    """Limit przepustowości API został przekroczony."""


@dataclass(slots=True)
class ExchangeNetworkError(ExchangeError):
    """Błąd sieci uniemożliwiający komunikację z API giełdy."""

    message: str
    reason: Exception | None = None

    def __str__(self) -> str:  # pragma: no cover - delegacja do RuntimeError
        return self.message


__all__ = [
    "ExchangeError",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeThrottlingError",
    "ExchangeNetworkError",
]
