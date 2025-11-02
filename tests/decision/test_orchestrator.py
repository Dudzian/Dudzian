import pytest

from bot_core.decision.models import DecisionCandidate, DecisionEvaluation
from bot_core.decision.orchestrator import (
    DecisionOrchestrator,
    _BanditRecommendation,
)

from tests.test_decision_orchestrator import _make_config, _snapshot


class RecordingAdvisor:
    def __init__(self, recommendation: _BanditRecommendation) -> None:
        self.recommendation = recommendation
        self.recommend_calls: list[tuple[DecisionCandidate, dict[str, object]]] = []
        self.observe_calls: list[tuple[DecisionCandidate, DecisionEvaluation]] = []

    def recommend(
        self,
        candidate: DecisionCandidate,
        *,
        regime,
        model_score,
        selection,
        cost_bps,
        net_edge_bps,
    ) -> _BanditRecommendation:
        self.recommend_calls.append(
            (
                candidate,
                {
                    "regime": regime,
                    "model_score": model_score,
                    "selection": selection,
                    "cost_bps": cost_bps,
                    "net_edge_bps": net_edge_bps,
                },
            )
        )
        return self.recommendation

    def observe(self, candidate: DecisionCandidate, evaluation: DecisionEvaluation) -> None:
        self.observe_calls.append((candidate, evaluation))


def test_custom_strategy_advisor_controls_recommendations() -> None:
    advisor = RecordingAdvisor(
        _BanditRecommendation(("shadow", "deterministic"), 2_500.0, 0.42)
    )
    orchestrator = DecisionOrchestrator(
        _make_config(),
        strategy_advisor=advisor,
    )
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=5_000.0,
        expected_return_bps=12.0,
        expected_probability=0.8,
        cost_bps_override=2.0,
        latency_ms=150.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.recommended_modes == ("shadow", "deterministic")
    assert evaluation.recommended_position_size == pytest.approx(2_500.0)
    assert evaluation.recommended_risk_score == pytest.approx(0.42)
    assert advisor.recommend_calls
    assert advisor.observe_calls
    observed_evaluation = advisor.observe_calls[0][1]
    assert observed_evaluation.recommended_modes == ("shadow", "deterministic")
    assert observed_evaluation.recommended_risk_score == pytest.approx(0.42)


def test_strategy_advisor_not_invoked_without_snapshot() -> None:
    advisor = RecordingAdvisor(
        _BanditRecommendation(("live",), 1_000.0, 0.9)
    )
    orchestrator = DecisionOrchestrator(
        _make_config(),
        strategy_advisor=advisor,
    )
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=4_000.0,
        expected_return_bps=10.0,
        expected_probability=0.75,
        latency_ms=180.0,
    )

    evaluations = orchestrator.evaluate_candidates([candidate], risk_snapshots={})

    assert advisor.recommend_calls == []
    assert evaluations[0].recommended_modes == ()
    assert evaluations[0].recommended_position_size is None
    assert evaluations[0].recommended_risk_score is None
