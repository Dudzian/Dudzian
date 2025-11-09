"""Interfejs modułu egzekucji z obsługą retry i sanity-check."""
from __future__ import annotations

import abc
import asyncio
import types
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Protocol, TypeVar

from bot_core.exchanges.base import OrderRequest, OrderResult


PriceResolver = Callable[[str], Optional[float]]


@dataclass(slots=True)
class ExecutionContext:
    """Parametry wykonania przekazywane z warstwy strategii/ryzyka."""

    portfolio_id: str
    risk_profile: str
    environment: str
    metadata: Mapping[str, str]
    price_resolver: PriceResolver | None = None


ESelf = TypeVar("ESelf", bound="ExecutionService")


class ExecutionService(abc.ABC):
    """Abstrakcyjny interfejs modułu egzekucji."""

    @abc.abstractmethod
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        """Realizuje zlecenie z pełną obsługą retry/backoff."""

    def __enter__(self: ESelf) -> ESelf:
        """Zwraca instancję jako kontekst menedżera."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Optional[types.TracebackType],
    ) -> Literal[False]:
        """Zapewnia automatyczne domknięcie zasobów w kontekście synchronicznym."""

        self.close()
        return False

    async def __aenter__(self: ESelf) -> ESelf:
        """Zwraca instancję jako asynchroniczny kontekst menedżera."""

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Optional[types.TracebackType],
    ) -> Literal[False]:
        """Zapewnia asynchroniczne domknięcie zasobów po wyjściu z kontekstu."""

        await self.close_async()
        return False

    async def execute_async(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        """Asynchroniczna wersja ``execute`` z domyślnym fallbackiem do wątku."""

        return await asyncio.to_thread(self.execute, request, context)

    @abc.abstractmethod
    def cancel(self, order_id: str, context: ExecutionContext) -> None:
        """Anuluje zlecenie, uwzględniając wymogi giełdy."""

    async def cancel_async(self, order_id: str, context: ExecutionContext) -> None:
        """Asynchroniczna wersja ``cancel`` uruchamiana w wątku pomocniczym."""

        await asyncio.to_thread(self.cancel, order_id, context)

    @abc.abstractmethod
    def flush(self) -> None:
        """Pozwala zakończyć proces (np. wysłać zaległe anulacje)."""

    async def flush_async(self) -> None:
        """Asynchroniczny fallback ``flush`` wykorzystujący ``asyncio.to_thread``."""

        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        """Zamyka zasoby wykonawcze (opcjonalne)."""

    async def close_async(self) -> None:
        """Asynchroniczny fallback ``close`` – uruchamia blokujące zamknięcie w wątku."""

        if type(self).close is ExecutionService.close:
            return None
        await asyncio.to_thread(self.close)


class RetryPolicy(Protocol):
    """Kontrakt polityki retry/backoff."""

    def on_error(self, attempt: int, error: Exception) -> float:
        ...


__all__ = ["ExecutionContext", "ExecutionService", "RetryPolicy", "PriceResolver"]
