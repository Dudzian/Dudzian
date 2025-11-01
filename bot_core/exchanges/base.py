"""Abstrakcyjne interfejsy adapterów giełdowych dla nowej architektury bota."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


class Environment(str, Enum):
    """Wspiera rozdzielenie środowisk (live, paper, testnet)."""

    LIVE = "live"
    PAPER = "paper"
    TESTNET = "testnet"


@dataclass(slots=True)
class ExchangeCredentials:
    """Przekrojowa reprezentacja kluczy API w modelu najmniejszych uprawnień."""

    key_id: str
    secret: Optional[str] = None
    passphrase: Optional[str] = None
    environment: Environment = Environment.LIVE
    permissions: Sequence[str] = ()


@dataclass(slots=True)
class AccountSnapshot:
    """Podstawowy model danych konta używany przez silnik ryzyka."""

    balances: Mapping[str, float]
    total_equity: float
    available_margin: float
    maintenance_margin: float


@dataclass(slots=True)
class OrderRequest:
    """Znormalizowany model zlecenia przekazywany do modułu egzekucji.

    Pola opcjonalne są ignorowane przez adapter, jeśli nie są wspierane.
    """

    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    client_order_id: Optional[str] = None

    # Dodatkowe, opcjonalne rozszerzenia:
    stop_price: Optional[float] = None   # np. stop/stop-limit
    atr: Optional[float] = None          # referencyjne ATR do SL/TP, jeśli strategia je dostarcza

    # Dowolne metadane strategii (audyt/telemetria)
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class OrderResult:
    """Standardowa odpowiedź adaptera giełdowego."""

    order_id: str
    status: str
    filled_quantity: float
    avg_price: Optional[float]
    raw_response: Mapping[str, Any]


class ExchangeAdapter(abc.ABC):
    """Bazowy interfejs wszystkich adapterów giełdowych."""

    name: str

    def __init__(self, credentials: ExchangeCredentials) -> None:
        self._credentials = credentials

    @property
    def credentials(self) -> ExchangeCredentials:
        """Udostępnia referencję do aktualnych poświadczeń."""
        return self._credentials

    @abc.abstractmethod
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        """Ustawia specyficzne wymagania sieciowe (np. IP allowlisting)."""

    @abc.abstractmethod
    def fetch_account_snapshot(self) -> AccountSnapshot:
        """Pobiera bieżące informacje o kapitale, marginesie i saldach."""

    @abc.abstractmethod
    def fetch_symbols(self) -> Iterable[str]:
        """Zwraca listę obsługiwanych instrumentów w formacie wewnętrznym."""

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        """Pobiera świece OHLCV w UTC na potrzeby warstwy danych."""

    @abc.abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        """Składa zlecenie w imieniu strategii."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        """Anuluje zlecenie po identyfikatorze."""

    @abc.abstractmethod
    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        """Udostępnia strumień danych publicznych (gRPC lub REST long-poll)."""

    @abc.abstractmethod
    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        """Udostępnia strumień zdarzeń prywatnych (gRPC lub REST long-poll)."""


class ExchangeAdapterFactory(Protocol):
    """Prosty kontrakt dla fabryk adapterów (używany w konfiguracji)."""

    def __call__(self, credentials: ExchangeCredentials, **kwargs: Any) -> ExchangeAdapter:
        ...


__all__ = [
    "Environment",
    "ExchangeCredentials",
    "AccountSnapshot",
    "OrderRequest",
    "OrderResult",
    "ExchangeAdapter",
    "ExchangeAdapterFactory",
]
