from __future__ import annotations

from typing import Mapping, Protocol

from bot_core.decision.models import DecisionCandidate
from bot_core.tco.models import TCOReport


class DecisionCostResolverProtocol(Protocol):
    """Interfejs serwisu rozwiązującego koszty/TCO dla kandydata."""

    def update_costs_from_report(self, report: TCOReport | Mapping[str, object]) -> None:
        """Buduje indeks kosztów (bps) na podstawie raportu TCO."""

    def resolve_cost(self, candidate: DecisionCandidate) -> tuple[float | None, bool]:
        """Zwraca koszt strategii oraz informację o brakujących danych."""


__all__ = ["DecisionCostResolverProtocol"]
