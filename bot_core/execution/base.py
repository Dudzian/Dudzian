"""Interfejs modułu egzekucji z obsługą retry i sanity-check."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Mapping, Protocol

from bot_core.exchanges.base import OrderRequest, OrderResult


@dataclass(slots=True)
class ExecutionContext:
    """Parametry wykonania przekazywane z warstwy strategii/ryzyka."""

    portfolio_id: str
    risk_profile: str
    environment: str
    metadata: Mapping[str, str]


class ExecutionService(abc.ABC):
    """Abstrakcyjny interfejs modułu egzekucji."""

    @abc.abstractmethod
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        """Realizuje zlecenie z pełną obsługą retry/backoff."""

    @abc.abstractmethod
    def cancel(self, order_id: str, context: ExecutionContext) -> None:
        """Anuluje zlecenie, uwzględniając wymogi giełdy."""

    @abc.abstractmethod
    def flush(self) -> None:
        """Pozwala zakończyć proces (np. wysłać zaległe anulacje)."""


class RetryPolicy(Protocol):
    """Kontrakt polityki retry/backoff."""

    def on_error(self, attempt: int, error: Exception) -> float:
        ...


__all__ = ["ExecutionContext", "ExecutionService", "RetryPolicy"]
