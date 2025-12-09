"""Komponenty scoringu/rankingu dla strategii portfelowych."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(slots=True)
class ScoreResult:
    """Pojedynczy wkład komponentu scoringowego."""

    component: str
    raw: float
    weighted: float


class StrategyScoreProvider(Protocol):
    """Prosty interfejs komponentu scoringowego."""

    name: str

    def evaluate(self, strategy: str, state: object) -> ScoreResult:  # pragma: no cover - Protocol
        ...


class AlphaScore:
    """Dodatnia kontrybucja z alfy strategii."""

    name = "alpha"

    def __init__(self, weight: float) -> None:
        self._weight = max(0.0, float(weight))

    def evaluate(self, strategy: str, state: object) -> ScoreResult:
        alpha_value = float(getattr(state, "smoothed_alpha", 0.0))
        return ScoreResult(
            component=self.name,
            raw=alpha_value,
            weighted=alpha_value * self._weight,
        )


class SLOScore:
    """Ujemna kontrybucja za naruszenia SLO."""

    name = "slo"

    def __init__(self, weight: float) -> None:
        self._weight = max(0.0, float(weight))

    def evaluate(self, strategy: str, state: object) -> ScoreResult:
        slo_value = max(0.0, float(getattr(state, "smoothed_slo", 0.0)))
        return ScoreResult(
            component=self.name,
            raw=slo_value,
            weighted=-slo_value * self._weight,
        )


class RiskScore:
    """Ujemna kontrybucja za karę ryzyka."""

    name = "risk"

    def __init__(self, weight: float) -> None:
        self._weight = max(0.0, float(weight))

    def evaluate(self, strategy: str, state: object) -> ScoreResult:
        risk_penalty = max(0.0, float(getattr(state, "risk_penalty", 0.0)))
        return ScoreResult(
            component=self.name,
            raw=risk_penalty,
            weighted=-risk_penalty * self._weight,
        )


class CostScore:
    """Ujemna kontrybucja za koszty strategii."""

    name = "cost"

    def __init__(self, weight: float, resolver: Callable[[str, object], float]) -> None:
        self._weight = max(0.0, float(weight))
        self._resolver = resolver

    def evaluate(self, strategy: str, state: object) -> ScoreResult:
        cost_value = max(0.0, float(self._resolver(strategy, state)))
        return ScoreResult(
            component=self.name,
            raw=cost_value,
            weighted=-cost_value * self._weight,
        )


__all__ = [
    "AlphaScore",
    "SLOScore",
    "RiskScore",
    "CostScore",
    "ScoreResult",
    "StrategyScoreProvider",
]
