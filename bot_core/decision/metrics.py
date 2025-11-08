"""Prometheus-compatible metrics for the decision engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from bot_core.decision.models import DecisionEvaluation
from bot_core.observability.metrics import (
    CounterMetric,
    HistogramMetric,
    MetricsRegistry,
    get_global_metrics_registry,
)


def _sanitize_label(value: str | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _coerce_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


_ALLOWED_INTENT_KEYS: Sequence[str] = (
    "intent",
    "signal_intent",
    "strategy_intent",
    "execution_intent",
)


_REJECTION_REASON_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("brak snapshotu ryzyka", "missing_risk_snapshot"),
    ("brak danych kosztowych", "missing_cost_data"),
    ("prawdopodobieństwo", "probability_below_threshold"),
    ("koszt", "cost_above_limit"),
    ("net edge", "net_edge_below_threshold"),
    ("latencja", "latency_above_limit"),
    ("przekroczony dzienny limit straty", "daily_loss_limit"),
    ("przekroczony limit obsunięcia", "drawdown_limit"),
    ("ekspozycja", "exposure_limit"),
    ("liczba pozycji", "open_positions_limit"),
    ("wartość zlecenia", "trade_notional_limit"),
    ("force_liquidation", "force_liquidation"),
)


@dataclass(slots=True)
class DecisionMetricSet:
    """Collection of decision-engine metrics exposed to Prometheus."""

    registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    latency_buckets: Sequence[float] = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
    edge_buckets: Sequence[float] = (
        -100.0,
        -50.0,
        -25.0,
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
    )
    probability_buckets: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98)
    risk_score_buckets: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)

    _evaluations_total: CounterMetric = field(init=False, repr=False)
    _latency_ms: HistogramMetric = field(init=False, repr=False)
    _net_edge_bps: HistogramMetric = field(init=False, repr=False)
    _expected_probability: HistogramMetric = field(init=False, repr=False)
    _risk_score: HistogramMetric = field(init=False, repr=False)
    _risk_flag_total: CounterMetric = field(init=False, repr=False)
    _stress_failures_total: CounterMetric = field(init=False, repr=False)
    _rejection_reasons_total: CounterMetric = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._evaluations_total = self.registry.counter(
            "decision_candidate_evaluations_total",
            "Liczba kandydatów ocenionych przez DecisionOrchestrator (podział na wynik).",
        )
        self._latency_ms = self.registry.histogram(
            "decision_candidate_latency_ms",
            "Rozkład zmierzonej latencji kandydatów przekazanej przez strategie.",
            buckets=tuple(self.latency_buckets),
        )
        self._net_edge_bps = self.registry.histogram(
            "decision_candidate_net_edge_bps",
            "Rozkład oczekiwanego edge'u netto (bps) po uwzględnieniu kosztów.",
            buckets=tuple(self.edge_buckets),
        )
        self._expected_probability = self.registry.histogram(
            "decision_candidate_success_probability",
            "Rozkład prawdopodobieństwa sukcesu estymowanego przez modele decyzyjne.",
            buckets=tuple(self.probability_buckets),
        )
        self._risk_score = self.registry.histogram(
            "decision_candidate_recommended_risk_score",
            "Rozkład rekomendowanych scoringów ryzyka z doradcy strategii.",
            buckets=tuple(self.risk_score_buckets),
        )
        self._risk_flag_total = self.registry.counter(
            "decision_candidate_risk_flags_total",
            "Liczba flag ryzyka wygenerowanych podczas ewaluacji kandydatów.",
        )
        self._stress_failures_total = self.registry.counter(
            "decision_candidate_stress_failures_total",
            "Liczba porażek testów stresowych w ewaluacjach kandydatów.",
        )
        self._rejection_reasons_total = self.registry.counter(
            "decision_candidate_rejection_reasons_total",
            "Liczba odrzuceń kandydatów z podziałem na skategoryzowane powody.",
        )

    def observe_evaluation(self, evaluation: DecisionEvaluation) -> None:
        """Feed an evaluation into the metrics set."""

        candidate = evaluation.candidate
        intent_label = self.intent_from_metadata(candidate.metadata)
        labels = {
            "strategy": _sanitize_label(candidate.strategy),
            "profile": _sanitize_label(candidate.risk_profile),
            "intent": intent_label,
            "result": "accepted" if evaluation.accepted else "rejected",
        }
        self._evaluations_total.inc(labels=labels)

        latency_value = _coerce_float(candidate.latency_ms)
        if latency_value is not None:
            self._latency_ms.observe(
                latency_value,
                labels={
                    "strategy": labels["strategy"],
                    "profile": labels["profile"],
                    "intent": intent_label,
                },
            )

        net_edge_value = _coerce_float(evaluation.net_edge_bps)
        if net_edge_value is not None:
            self._net_edge_bps.observe(
                net_edge_value,
                labels={
                    "strategy": labels["strategy"],
                    "profile": labels["profile"],
                    "intent": intent_label,
                },
            )

        probability_value = _coerce_float(evaluation.model_success_probability)
        if probability_value is None:
            probability_value = _coerce_float(candidate.expected_probability)
        if probability_value is not None:
            self._expected_probability.observe(
                probability_value,
                labels={
                    "strategy": labels["strategy"],
                    "profile": labels["profile"],
                    "intent": intent_label,
                },
            )

        risk_score_value = _coerce_float(evaluation.recommended_risk_score)
        if risk_score_value is not None:
            self._risk_score.observe(
                risk_score_value,
                labels={
                    "strategy": labels["strategy"],
                    "profile": labels["profile"],
                    "intent": intent_label,
                },
            )

        for flag in evaluation.risk_flags:
            flag_label = _sanitize_label(str(flag))
            self._risk_flag_total.inc(
                labels={
                    "profile": labels["profile"],
                    "flag": flag_label,
                    "intent": intent_label,
                }
            )

        for failure in evaluation.stress_failures:
            failure_label = _sanitize_label(str(failure))
            self._stress_failures_total.inc(
                labels={
                    "profile": labels["profile"],
                    "failure": failure_label,
                    "intent": intent_label,
                }
            )

        if not evaluation.accepted:
            self._record_rejection_reasons(
                evaluation,
                strategy=labels["strategy"],
                profile=labels["profile"],
                intent=intent_label,
            )

    def _record_rejection_reasons(
        self,
        evaluation: DecisionEvaluation,
        *,
        strategy: str,
        profile: str,
        intent: str,
    ) -> None:
        reasons = evaluation.reasons
        if not reasons:
            self._rejection_reasons_total.inc(
                labels={
                    "strategy": strategy,
                    "profile": profile,
                    "intent": intent,
                    "reason": "unspecified",
                }
            )
            return

        for reason in reasons:
            label = self.reason_label(reason)
            self._rejection_reasons_total.inc(
                labels={
                    "strategy": strategy,
                    "profile": profile,
                    "intent": intent,
                    "reason": label,
                }
            )

    def reason_label(self, reason: object) -> str:
        if reason is None:
            return "unspecified"
        text = str(reason).strip().lower()
        if not text:
            return "unspecified"
        for keyword, label in _REJECTION_REASON_KEYWORDS:
            if keyword in text:
                return label
        return "other"

    def intent_from_metadata(self, metadata: Mapping[str, object] | None) -> str:
        if not metadata:
            return "unknown"
        for key in _ALLOWED_INTENT_KEYS:
            value = metadata.get(key)
            if value in (None, ""):
                continue
            return _sanitize_label(str(value))
        return "unknown"


__all__ = ["DecisionMetricSet"]
