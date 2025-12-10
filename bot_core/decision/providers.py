from __future__ import annotations

from typing import Mapping, Protocol

from bot_core.ai import ModelScore
from bot_core.config.models import DecisionOrchestratorThresholds
from bot_core.decision.models import DecisionCandidate, RiskSnapshot


class DecisionProvider(Protocol):
    """Źródło informacji potrzebnych do ewaluacji kandydata."""

    def score_with_model(
        self, candidate: DecisionCandidate
    ) -> tuple[str | None, ModelScore | None, object | None]:
        """Zwraca wynik modelu, nazwę oraz metadane selekcji modeli."""

    def thresholds_for_profile(self, profile: str) -> DecisionOrchestratorThresholds:
        """Udostępnia progi dla danego profilu ryzyka."""

    def threshold_snapshot(
        self, thresholds: DecisionOrchestratorThresholds
    ) -> Mapping[str, float | None]:
        """Tworzy snapshot progów użyty w odpowiedzi."""

    def ensure_snapshot(
        self, profile: str, snapshot: Mapping[str, object] | RiskSnapshot
    ) -> RiskSnapshot:
        """Normalizuje snapshot ryzyka."""

    def resolve_cost(self, candidate: DecisionCandidate) -> tuple[float | None, bool]:
        """Zwraca koszt strategii oraz informację o brakujących danych."""

    def resolve_regime(self, candidate: DecisionCandidate):
        """Określa reżim rynkowy dla kandydata."""


__all__ = ["DecisionProvider"]
