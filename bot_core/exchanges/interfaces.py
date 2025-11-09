"""Wspólne interfejsy warstwy wymiany używane przez testy integracyjne.

Moduł ten zapewnia lekki zestaw abstrakcji kompatybilny z poprzednim
pakietem ``KryptoLowca.exchanges``. Dzięki temu testy oraz narzędzia, które
oczekują istnienia obiektów takich jak ``MarketSubscription`` czy
``MarketStreamHandle``, mogą korzystać bezpośrednio z przestrzeni
``bot_core`` bez sięgania po warstwę legacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Protocol, Sequence

from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


MarketPayload = MutableMapping[str, Any]
CallbackT = Callable[[MarketPayload], Awaitable[None]]
OrderCallbackT = Callable[[OrderResult], Awaitable[None]]


class MarketStreamHandle(Protocol):
    """Minimalny kontrakt na obiekty zarządzające strumieniami long-pollowymi."""

    async def __aenter__(self) -> "MarketStreamHandle":  # pragma: no cover - interfejs
        ...

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:  # pragma: no cover - interfejs
        ...

    async def start(self) -> None:  # pragma: no cover - interfejs
        ...

    async def stop(self) -> None:  # pragma: no cover - interfejs
        ...


@dataclass(slots=True)
class MarketSubscription:
    """Definicja pojedynczej subskrypcji kanału long-pollowego."""

    channel: str
    symbols: Sequence[str]
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamSubscription:
    """Bazowa reprezentacja subskrypcji dla kanałów publicznych/prywatnych."""

    scope: str
    channels: Sequence[str]
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scope = self.scope.lower().strip()
        if not self.scope:
            raise ValueError("Scope subskrypcji nie może być pusty")


@dataclass(slots=True)
class PublicStreamSubscription(StreamSubscription):
    """Subskrypcja kanałów publicznych (np. kursy, orderbook)."""

    def __init__(
        self,
        channels: Sequence[str],
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(scope="public", channels=channels, params=params or {})


@dataclass(slots=True)
class PrivateStreamSubscription(StreamSubscription):
    """Subskrypcja kanałów prywatnych (np. fill'e, zmiany zleceń)."""

    def __init__(
        self,
        channels: Sequence[str],
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(scope="private", channels=channels, params=params or {})


__all__ = [
    "AccountSnapshot",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "OrderRequest",
    "OrderResult",
    "MarketPayload",
    "CallbackT",
    "OrderCallbackT",
    "MarketStreamHandle",
    "MarketSubscription",
    "StreamSubscription",
    "PublicStreamSubscription",
    "PrivateStreamSubscription",
]
