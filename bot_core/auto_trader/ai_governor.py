"""AI Governor odpowiedzialny za dobór trybu AutoTradera."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Iterable, Mapping, Sequence

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.decision.orchestrator import StrategyPerformanceSummary

try:  # pragma: no cover - DecisionOrchestrator może być opcjonalny w buildach light
    from bot_core.decision.orchestrator import DecisionOrchestrator
except Exception:  # pragma: no cover - fallback dla środowisk okrojonych
    DecisionOrchestrator = Any  # type: ignore[misc, assignment]


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


@dataclass
class AutoTraderAIGovernorRunner:
    """Wykonuje cykle AI Governora bazując na metrykach DecisionOrchestratora."""

    orchestrator: "DecisionOrchestrator"
    governor: AutoTraderAIGovernor = field(default_factory=AutoTraderAIGovernor)
    _cycles_total: int = field(default=0, init=False, repr=False)
    _switch_total: int = field(default=0, init=False, repr=False)
    _last_mode: str | None = field(default=None, init=False, repr=False)
    _last_cycle_metrics: Mapping[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_risk_metrics: Mapping[str, float] = field(default_factory=dict, init=False, repr=False)
    _mode_transitions: Deque[dict[str, object]] = field(
        default_factory=lambda: deque(maxlen=16), init=False, repr=False
    )

    _REGIME_BY_MODE: Mapping[str, MarketRegime] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._REGIME_BY_MODE = {
            mode: regime for regime, mode in AutoTraderAIGovernor._MODE_BY_REGIME.items()
        }

    # ------------------------------------------------------------------
    def run_cycle(self, *, regime: MarketRegime | None = None) -> AIGovernorDecision:
        """Uruchamia pojedynczy cykl rekomendacji bazując na stanie orchestratora."""

        summary = self._select_summary(regime)
        assessment = self._build_assessment(summary)
        risk_metrics = self._build_risk_metrics(summary, assessment)
        cycle_metrics = self._preview_cycle_metrics(summary)
        transaction_cost_bps = self._estimate_transaction_cost(summary)
        decision = self.governor.update_context(
            assessment=assessment,
            risk_metrics=risk_metrics,
            cycle_metrics=cycle_metrics,
            transaction_cost_bps=transaction_cost_bps,
        )
        self._register_cycle(decision.mode, summary, assessment)
        self._last_cycle_metrics = {
            **cycle_metrics,
            "strategy_switch_total": float(self._switch_total),
        }
        self._last_risk_metrics = dict(risk_metrics)
        return decision

    def run_until(
        self,
        *,
        mode: str | None = None,
        predicate: Callable[[AIGovernorDecision], bool] | None = None,
        limit: int = 32,
        regimes: Sequence[MarketRegime] | None = None,
    ) -> tuple[AIGovernorDecision, ...]:
        """Powtarza cykle, aż warunek zostanie spełniony lub osiągnięto limit."""

        decisions: list[AIGovernorDecision] = []
        normalized_mode = mode.lower() if mode else None
        target_regimes: Iterable[MarketRegime]
        if regimes:
            target_regimes = regimes
        elif normalized_mode and normalized_mode in self._REGIME_BY_MODE:
            target_regimes = (self._REGIME_BY_MODE[normalized_mode],)
        else:
            target_regimes = AutoTraderAIGovernor._MODE_BY_REGIME.keys()
        target_cycle = list(target_regimes)
        predicate_fn: Callable[[AIGovernorDecision], bool]
        if predicate is not None:
            predicate_fn = predicate
        elif normalized_mode:
            predicate_fn = lambda decision: decision.mode == normalized_mode
        else:
            predicate_fn = lambda decision: True

        idx = 0
        while idx < max(1, int(limit)):
            regime_hint = target_cycle[idx % len(target_cycle)] if target_cycle else None
            decision = self.run_cycle(regime=regime_hint)
            decisions.append(decision)
            if predicate_fn(decision):
                break
            idx += 1
        return tuple(decisions)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        payload = self.governor.snapshot()
        telemetry = dict(payload.get("telemetry", {}))
        if self._last_cycle_metrics:
            telemetry["cycleMetrics"] = dict(self._last_cycle_metrics)
        if self._last_risk_metrics:
            telemetry["riskMetrics"] = dict(self._last_risk_metrics)
        telemetry.setdefault("cycleLatency", self._cycle_latency_demo_metrics())
        telemetry.setdefault("modeTransitions", list(self._mode_transitions))
        telemetry.setdefault(
            "guardrails",
            {"active": False, "killSwitch": False, "recent": []},
        )
        if telemetry:
            payload["telemetry"] = telemetry
        return payload

    # ------------------------------------------------------------------
    def _select_summary(self, regime: MarketRegime | None) -> StrategyPerformanceSummary:
        snapshot = self.orchestrator.strategy_performance_snapshot()
        candidates = [
            summary
            for summary in snapshot.values()
            if regime is None or self._normalize_regime(summary.regime) == regime
        ]
        if not candidates:
            fallback_regime = regime or MarketRegime.TREND
            return StrategyPerformanceSummary(
                strategy="fallback",
                regime=fallback_regime,
                hit_rate=0.55,
                pnl=5.0,
                sharpe=0.4,
                updated_at=datetime.now(timezone.utc),
                observations=1,
            )
        return max(candidates, key=self._score_summary)

    @staticmethod
    def _normalize_regime(value: MarketRegime | str) -> MarketRegime:
        if isinstance(value, MarketRegime):
            return value
        try:
            return MarketRegime(value)
        except ValueError:
            return MarketRegime.TREND

    @staticmethod
    def _score_summary(summary: StrategyPerformanceSummary) -> float:
        sharpe = max(summary.sharpe, 0.0)
        return summary.hit_rate * (1.0 + sharpe) + summary.pnl

    def _build_assessment(self, summary: StrategyPerformanceSummary) -> MarketRegimeAssessment:
        regime = self._normalize_regime(summary.regime)
        confidence = max(0.3, min(0.95, 0.5 + summary.sharpe / 3.0))
        risk_score = max(0.05, min(0.95, 0.7 - summary.hit_rate * 0.4))
        metrics = {
            "hit_rate": float(summary.hit_rate),
            "sharpe": float(summary.sharpe),
            "pnl": float(summary.pnl),
            "observations": float(summary.observations),
        }
        return MarketRegimeAssessment(
            regime=regime,
            confidence=confidence,
            risk_score=risk_score,
            metrics=metrics,
        )

    def _build_risk_metrics(
        self, summary: StrategyPerformanceSummary, assessment: MarketRegimeAssessment
    ) -> Mapping[str, float]:
        metrics = {
            "risk_score": float(assessment.risk_score),
            "guardrail_active": 0.0,
            "kill_switch": 0.0,
            "cooldown_active": 0.0,
            "guardrail_reasons": 0.0,
            "observations": float(summary.observations),
        }
        return metrics

    def _preview_cycle_metrics(self, summary: StrategyPerformanceSummary) -> Mapping[str, float]:
        projected_cycles = self._cycles_total + 1
        return {
            "cycles_total": float(projected_cycles),
            "strategy_switch_total": float(self._switch_total),
            "hit_rate": float(summary.hit_rate),
            "sharpe": float(summary.sharpe),
        }

    def _estimate_transaction_cost(self, summary: StrategyPerformanceSummary) -> float:
        base = 20.0 - summary.pnl
        regime = self._normalize_regime(summary.regime)
        if summary.sharpe > 1.0:
            base -= 3.0
        if summary.hit_rate > 0.7:
            base -= 2.0
        if regime is MarketRegime.DAILY:
            base += 5.0
        elif regime is MarketRegime.MEAN_REVERSION:
            base += 2.0
        return max(5.0, min(40.0, base))

    def _register_cycle(
        self,
        mode: str,
        summary: StrategyPerformanceSummary,
        assessment: MarketRegimeAssessment,
    ) -> None:
        self._cycles_total += 1
        normalized_mode = mode.lower()
        if self._last_mode is not None and normalized_mode != self._last_mode:
            self._switch_total += 1
        self._last_mode = normalized_mode
        self._mode_transitions.appendleft(
            {
                "mode": normalized_mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": assessment.regime.value,
                "risk": float(assessment.risk_score),
            }
        )

    def _cycle_latency_demo_metrics(self) -> dict[str, float]:
        base = 650.0 + (self._cycles_total % 5) * 85.0
        return {
            "lastMs": base,
            "p50Ms": max(450.0, base * 0.9),
            "p95Ms": base * 1.15,
            "sampleCount": float(self._cycles_total),
        }


__all__ = [
    "AIGovernorDecision",
    "AutoTraderAIGovernor",
    "AutoTraderAIGovernorRunner",
]

