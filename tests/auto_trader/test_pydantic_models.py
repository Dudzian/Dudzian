from bot_core.auto_trader.ai_governor import AIGovernorDecision
from bot_core.auto_trader.lifecycle import LifecycleBootstrapSnapshot


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
