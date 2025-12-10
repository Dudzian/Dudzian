from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from bot_core.decision.models import DecisionCandidate, DecisionContext, DecisionEvaluation


class DecisionEvaluator(Protocol):
    """Kontrakt dla serwisu ewaluującego kandydatów."""

    def evaluate_candidate(
        self,
        candidate: DecisionCandidate,
        context: DecisionContext,
    ) -> DecisionEvaluation:
        """Ewaluacja pojedynczego kandydata."""

    def evaluate_candidates(
        self,
        candidates: Sequence[DecisionCandidate],
        contexts: Mapping[str, DecisionContext],
    ) -> Sequence[DecisionEvaluation]:
        """Ewaluacja wielu kandydatów."""


__all__ = ["DecisionEvaluator"]
