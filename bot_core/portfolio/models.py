"""Modele danych wykorzystywane przez PortfolioGovernora."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(slots=True)
class StrategyMetricsSnapshot:
    """Metryki jakościowe wykorzystywane do oceny strategii."""

    timestamp: datetime
    alpha_score: float = 0.0
    slo_violation_rate: float = 0.0
    risk_penalty: float = 0.0
    cost_bps: float | None = None
    net_edge_bps: float | None = None
    sample_weight: float = 1.0
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyAllocationDecision:
    """Decyzja PortfolioGovernora dotycząca pojedynczej strategii."""

    strategy: str
    weight: float
    baseline_weight: float
    signal_factor: float
    max_signal_hint: int | None = None
    metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioRebalanceDecision:
    """Podsumowanie decyzji rebalansującej alokację portfela."""

    timestamp: datetime
    weights: Mapping[str, float]
    scores: Mapping[str, float]
    alpha_components: Mapping[str, float]
    slo_components: Mapping[str, float]
    cost_components: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        """Reprezentacja decyzji w formie słownika do logowania."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "weights": dict(self.weights),
            "scores": dict(self.scores),
            "alpha": dict(self.alpha_components),
            "slo": dict(self.slo_components),
            "costs_bps": dict(self.cost_components),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "StrategyMetricsSnapshot",
    "StrategyAllocationDecision",
    "PortfolioRebalanceDecision",
]
