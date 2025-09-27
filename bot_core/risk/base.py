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


class RiskRepository(Protocol):
    """Kontrakt dla repozytoriów stanu ryzyka (np. SQLite, Parquet)."""

    def load(self, profile: str) -> Mapping[str, object] | None:
        ...

    def store(self, profile: str, state: Mapping[str, object]) -> None:
        ...


__all__ = ["RiskCheckResult", "RiskProfile", "RiskEngine", "RiskRepository"]
