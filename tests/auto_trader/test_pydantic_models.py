from bot_core.auto_trader.ai_governor import AIGovernorDecision
from bot_core.auto_trader.lifecycle import LifecycleBootstrapSnapshot
import pytest


def test_ai_governor_decision_casts_and_normalizes_fields() -> None:
    decision = AIGovernorDecision(
        mode=" grid ",
        reason=" test decision ",
        confidence="0.55",
        regime=" TREND ",
        risk_score="0.8",
        transaction_cost_bps="12.5",
        risk_metrics={"risk_score": "0.6", "guardrail_active": 1},
        cycle_metrics=[("cycle_latency_p95_ms", "1200.0")],
    )

    assert decision.mode == "grid"
    assert decision.reason == "test decision"
    assert decision.confidence == 0.55
    assert decision.regime == "TREND"
    assert decision.risk_score == 0.8
    assert decision.transaction_cost_bps == 12.5
    assert decision.risk_metrics == {"risk_score": 0.6, "guardrail_active": 1.0}
    assert decision.cycle_metrics == {"cycle_latency_p95_ms": 1200.0}
    assert decision.to_mapping()["mode"] == "grid"
    assert decision.decision_source == "policy"


def test_lifecycle_bootstrap_snapshot_normalizes_text() -> None:
    snapshot = LifecycleBootstrapSnapshot(
        risk_profile=" balanced ",
        market_regime=" trend  ",
        decision_state=" ",
        decision_signal=None,
        extra_key="ignored",
    )

    assert snapshot.risk_profile == "balanced"
    assert snapshot.market_regime == "trend"
    assert snapshot.decision_state is None
    assert snapshot.decision_signal is None
    assert snapshot.to_metadata() == {"risk_profile": "balanced", "market_regime": "trend"}

    snapshot.decision_state = "resume"
    assert snapshot.decision_state == "resume"


def test_ai_governor_decision_supports_model_assisted_contract_fields() -> None:
    decision = AIGovernorDecision(
        mode="hedge",
        reason="model override",
        confidence=0.9,
        regime="trend",
        risk_score=0.2,
        transaction_cost_bps=9.0,
        decision_source="model",
        inference_model="decision_model",
        inference_model_version="2026.04.02",
    )

    payload = decision.to_mapping()
    assert payload["decision_source"] == "model"
    assert payload["inference_model"] == "decision_model"
    assert payload["inference_model_version"] == "2026.04.02"


def test_ai_governor_decision_rejects_model_path_without_metadata() -> None:
    with pytest.raises(ValueError, match="decision_source=model/hybrid"):
        AIGovernorDecision(
            mode="hedge",
            reason="model override",
            confidence=0.9,
            regime="trend",
            risk_score=0.2,
            transaction_cost_bps=9.0,
            decision_source="model",
            inference_model="decision_model",
            inference_model_version=None,
        )
