"""AI Governor odpowiedzialny za dobór trybu AutoTradera."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensywne
        return None


@dataclass(slots=True)
class AIGovernorDecision:
    """Reprezentuje ostatnią decyzję AI Governora."""

    mode: str
    reason: str
    confidence: float
    regime: str
    risk_score: float
    transaction_cost_bps: float | None
    risk_metrics: Mapping[str, float] = field(default_factory=dict)
    cycle_metrics: Mapping[str, float] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "regime": self.regime,
            "risk_score": float(self.risk_score),
            "transaction_cost_bps": self.transaction_cost_bps,
            "risk_metrics": dict(self.risk_metrics),
            "cycle_metrics": dict(self.cycle_metrics),
        }
        return payload


class AutoTraderAIGovernor:
    """Heurystyka sterująca przełączaniem trybów scalping/hedge/grid."""

    _MODE_BY_REGIME: Mapping[MarketRegime, str] = {
        MarketRegime.TREND: "scalping",
        MarketRegime.DAILY: "grid",
        MarketRegime.MEAN_REVERSION: "hedge",
    }

    def __init__(
        self,
        *,
        history_limit: int = 32,
        cost_threshold_grid: float = 35.0,
        cost_threshold_scalping: float = 12.0,
        risk_ceiling: float = 0.65,
    ) -> None:
        self._history: Deque[AIGovernorDecision] = deque(maxlen=max(1, history_limit))
        self._last_decision: AIGovernorDecision | None = None
        self._cost_threshold_grid = float(cost_threshold_grid)
        self._cost_threshold_scalping = float(cost_threshold_scalping)
        self._risk_ceiling = float(risk_ceiling)

    def update_context(
        self,
        *,
        assessment: MarketRegimeAssessment,
        risk_metrics: Mapping[str, float],
        cycle_metrics: Mapping[str, float],
        transaction_cost_bps: float | None,
    ) -> AIGovernorDecision:
        """Zbuduj rekomendację trybu bazując na reżimie i telemetrii."""

        base_mode = self._MODE_BY_REGIME.get(assessment.regime, "grid")
        risk_score = _safe_float(risk_metrics.get("risk_score")) or float(assessment.risk_score)
        guardrail_active = float(risk_metrics.get("guardrail_active", 0.0))
        cooldown_active = float(risk_metrics.get("cooldown_active", 0.0))
        cycle_latency = _safe_float(cycle_metrics.get("cycle_latency_p95_ms")) or 0.0

        mode = base_mode
        reason = f"Regime {assessment.regime.value}"

        if risk_score >= self._risk_ceiling or guardrail_active:
            mode = "hedge"
            reason = "Podwyższone ryzyko lub aktywne guardraile"
        elif transaction_cost_bps is not None:
            if transaction_cost_bps >= self._cost_threshold_grid:
                mode = "grid"
                reason = "Wysokie koszty transakcyjne"
            elif transaction_cost_bps <= self._cost_threshold_scalping:
                mode = "scalping"
                reason = "Niskie koszty transakcyjne"

        if cooldown_active:
            mode = "hedge"
            reason = "Aktywny cooldown decyzji"

        confidence = self._estimate_confidence(
            risk_score=risk_score,
            guardrail_active=guardrail_active,
            cycle_latency=cycle_latency,
            transaction_cost_bps=transaction_cost_bps,
        )

        decision = AIGovernorDecision(
            mode=mode,
            reason=reason,
            confidence=confidence,
            regime=assessment.regime.value,
            risk_score=risk_score,
            transaction_cost_bps=transaction_cost_bps,
            risk_metrics=risk_metrics,
            cycle_metrics=cycle_metrics,
        )
        self._last_decision = decision
        self._history.appendleft(decision)
        return decision

    def _estimate_confidence(
        self,
        *,
        risk_score: float,
        guardrail_active: float,
        cycle_latency: float,
        transaction_cost_bps: float | None,
    ) -> float:
        confidence = 0.35 + max(0.0, 0.5 - abs(risk_score - 0.5))
        if guardrail_active:
            confidence += 0.15
        if transaction_cost_bps is not None:
            if transaction_cost_bps >= self._cost_threshold_grid:
                confidence += 0.1
            elif transaction_cost_bps <= self._cost_threshold_scalping:
                confidence += 0.05
        if cycle_latency > 2500.0:
            confidence -= 0.15
        return max(0.1, min(confidence, 0.95))

    def mode_adjustment(self, profile_name: str) -> float:
        """Zwróć korektę wyniku selekcji profilu bazującą na rekomendacji AI."""

        decision = self._last_decision
        if decision is None:
            return 0.0
        if profile_name == decision.mode:
            return min(0.6, 0.5 * decision.confidence)
        return -min(0.3, 0.25 * decision.confidence)

    def snapshot(self) -> dict[str, Any]:
        """Udostępnij bieżący stan wraz z historią dla UI PySide6."""

        last = self._last_decision.to_mapping() if self._last_decision else {}
        history = [entry.to_mapping() for entry in self._history]
        return {"last_decision": last, "history": history}

