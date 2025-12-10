from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bot_core.ai import MarketRegime, ModelScore
from bot_core.decision.models import (
    DecisionCandidate,
    DecisionEvaluation,
    ModelSelectionMetadata,
)


@dataclass(slots=True)
class BanditRecommendation:
    modes: tuple[str, ...]
    position_size: float | None
    risk_score: float


class StrategyAdvisor(Protocol):
    """Kontrakt dla doradcy wyboru strategii używanego przez orchestrator."""

    def recommend(
        self,
        candidate: DecisionCandidate,
        *,
        regime: MarketRegime | str,
        model_score: ModelScore | None,
        selection: ModelSelectionMetadata | None,
        cost_bps: float | None,
        net_edge_bps: float,
    ) -> BanditRecommendation:
        """Zwraca rekomendowane tryby wykonania i wielkość pozycji."""

    def observe(self, candidate: DecisionCandidate, evaluation: DecisionEvaluation) -> None:
        """Aktualizuje stan doradcy po zakończonej ewaluacji."""


class _LinUCBArm:
    """Minimalna implementacja ramienia LinUCB do eksploracji strategii."""

    def __init__(self, dimension: int, alpha: float) -> None:
        self.alpha = float(alpha)
        self.dimension = int(max(1, dimension))
        self.A = np.eye(self.dimension, dtype=float)
        self.b = np.zeros(self.dimension, dtype=float)

    def ensure_dimension(self, dimension: int) -> None:
        if dimension <= self.dimension:
            return
        dimension = int(max(1, dimension))
        new_A = np.eye(dimension, dtype=float)
        new_A[: self.dimension, : self.dimension] = self.A
        new_b = np.zeros(dimension, dtype=float)
        new_b[: self.dimension] = self.b
        self.A = new_A
        self.b = new_b
        self.dimension = dimension

    def predict(self, context: np.ndarray) -> float:
        self.ensure_dimension(context.size)
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(self.A) @ self.b
        try:
            inv_a = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            inv_a = np.linalg.pinv(self.A)
        mean = float(context @ theta)
        exploration = float(np.sqrt(context @ inv_a @ context))
        return mean + self.alpha * exploration

    def update(self, context: np.ndarray, reward: float) -> None:
        self.ensure_dimension(context.size)
        context = context.reshape(-1, 1)
        self.A += context @ context.T
        self.b += reward * context.ravel()


class _ThompsonArm:
    """Beta-Bernoulli ramię używane do rekomendacji ryzyka."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = float(max(alpha, 1e-3))
        self.beta = float(max(beta, 1e-3))

    def posterior_mean(self) -> float:
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    def update(self, outcome: float) -> None:
        outcome = float(max(0.0, min(1.0, outcome)))
        self.alpha += outcome
        self.beta += 1.0 - outcome


__all__ = ["BanditRecommendation", "StrategyAdvisor", "_LinUCBArm", "_ThompsonArm"]
