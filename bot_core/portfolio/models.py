"""Modele danych wykorzystywane przez PortfolioGovernora."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


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


@dataclass(slots=True)
class RiskSyncPayload:
    """Dane synchronizacji z risk engine przekazywane do governora strategii."""

    timestamp: datetime
    risk_profile: str | None = None
    drawdown_pct: float = 0.0
    risk_penalty: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_strategy_snapshot(self) -> StrategyMetricsSnapshot:
        penalty = self.risk_penalty
        if penalty is None:
            penalty = self.drawdown_pct
        return StrategyMetricsSnapshot(
            timestamp=self.timestamp,
            alpha_score=0.0,
            slo_violation_rate=0.0,
            risk_penalty=max(0.0, float(penalty)),
            metrics=self.metrics,
        )


@dataclass(slots=True)
class PayoutRecord:
    """Ustandaryzowana reprezentacja wypłaty do synchronizacji i logowania."""

    account_id: str
    asset: str
    amount: float
    destination: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "StrategyMetricsSnapshot",
    "StrategyAllocationDecision",
    "PortfolioRebalanceDecision",
    "RiskSyncPayload",
    "PayoutRecord",
]
