"""Interfejs silnika zarządzania ryzykiem oraz profili."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

from bot_core.exchanges.base import AccountSnapshot, OrderRequest


@dataclass(slots=True)
class RiskCheckResult:
    """Wynik kontroli ryzyka."""

    allowed: bool
    reason: str | None = None
    adjustments: Mapping[str, float] | None = None
    metadata: Mapping[str, object] | None = None


class RiskProfile(abc.ABC):
    """Każdy profil musi dostarczać zestaw limitów i procedur kontroli."""

    name: str

    @abc.abstractmethod
    def max_positions(self) -> int:
        ...

    @abc.abstractmethod
    def max_leverage(self) -> float:
        ...

    @abc.abstractmethod
    def drawdown_limit(self) -> float:
        ...

    @abc.abstractmethod
    def daily_loss_limit(self) -> float:
        ...

    @abc.abstractmethod
    def max_position_exposure(self) -> float:
        ...

    @abc.abstractmethod
    def target_volatility(self) -> float:
        ...

    @abc.abstractmethod
    def stop_loss_atr_multiple(self) -> float:
        ...


class StaticRiskProfile(RiskProfile):
    """Profil ryzyka zdefiniowany stałymi parametrami.

    Podczas migracji z gałęzi legacy wiele profili kopiowało identyczne metody
    zwracające wartości stałych atrybutów.  Ta klasa zapewnia wspólne
    implementacje bazujące na nazwanych atrybutach klasy, dzięki czemu
    poszczególne profile mogą definiować jedynie wartości graniczne, bez
    powielania logiki.
    """

    _max_positions: int
    _max_leverage: float
    _drawdown_limit: float
    _daily_loss_limit: float
    _max_position_pct: float
    _target_volatility: float
    _stop_loss_atr_multiple: float

    def max_positions(self) -> int:
        return self._max_positions

    def max_leverage(self) -> float:
        return self._max_leverage

    def drawdown_limit(self) -> float:
        return self._drawdown_limit

    def daily_loss_limit(self) -> float:
        return self._daily_loss_limit

    def max_position_exposure(self) -> float:
        return self._max_position_pct

    def target_volatility(self) -> float:
        return self._target_volatility

    def stop_loss_atr_multiple(self) -> float:
        return self._stop_loss_atr_multiple


class RiskEngine(abc.ABC):
    """Silnik ryzyka odpowiada za enforce limitów i pre-trade checks."""

    @abc.abstractmethod
    def register_profile(self, profile: RiskProfile) -> None:
        ...

    @abc.abstractmethod
    def apply_pre_trade_checks(
        self,
        request: OrderRequest,
        *,
        account: AccountSnapshot,
        profile_name: str,
    ) -> RiskCheckResult:
        ...

    @abc.abstractmethod
    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp: datetime | None = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def should_liquidate(self, *, profile_name: str) -> bool:
        ...

    def snapshot_state(self, profile_name: str) -> Mapping[str, object] | None:
        """Zwraca bieżący stan profilu ryzyka do celów raportowych.

        Domyślna implementacja pozostawia metodę niezaimplementowaną –
        konkretne silniki powinny ją nadpisać, jeśli udostępniają możliwość
        inspekcji stanu w trybie tylko do odczytu.
        """

        raise NotImplementedError


class RiskRepository(Protocol):
    """Kontrakt dla repozytoriów stanu ryzyka (np. SQLite, Parquet)."""

    def load(self, profile: str) -> Mapping[str, object] | None:
        ...

    def store(self, profile: str, state: Mapping[str, object]) -> None:
        ...


__all__ = [
    "RiskCheckResult",
    "RiskProfile",
    "StaticRiskProfile",
    "RiskEngine",
    "RiskRepository",
]
