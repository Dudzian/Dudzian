import pytest
from pydantic import ValidationError

from bot_core.decision.schemas import DecisionEngineSummary


def test_decision_engine_summary_validates_minimal_payload() -> None:
    payload = DecisionEngineSummary.model_validate(
        {
            "type": "decision_engine_summary",
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "history_limit": 0,
            "history_window": 0,
            "full_total": 0,
            "full_accepted": 0,
            "full_rejected": 0,
            "full_acceptance_rate": 0.0,
            "rejection_reasons": {},
            "unique_rejection_reasons": 0,
            "unique_risk_flags": 0,
            "risk_flags_with_accepts": 0,
            "unique_stress_failures": 0,
            "stress_failures_with_accepts": 0,
            "unique_models": 0,
            "models_with_accepts": 0,
            "unique_actions": 0,
            "actions_with_accepts": 0,
            "unique_strategies": 0,
            "strategies_with_accepts": 0,
            "unique_symbols": 0,
            "symbols_with_accepts": 0,
            "current_acceptance_streak": 0,
            "current_rejection_streak": 0,
            "longest_acceptance_streak": 0,
            "longest_rejection_streak": 0,
        }
    )
    assert payload.type == "decision_engine_summary"


def test_decision_engine_summary_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        DecisionEngineSummary.model_validate({
            "type": "invalid",
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "history_limit": 0,
            "history_window": 0,
            "full_total": 0,
            "full_accepted": 0,
            "full_rejected": 0,
            "full_acceptance_rate": 0.0,
            "rejection_reasons": {},
            "unique_rejection_reasons": 0,
            "unique_risk_flags": 0,
            "risk_flags_with_accepts": 0,
            "unique_stress_failures": 0,
            "stress_failures_with_accepts": 0,
            "unique_models": 0,
            "models_with_accepts": 0,
            "unique_actions": 0,
            "actions_with_accepts": 0,
            "unique_strategies": 0,
            "strategies_with_accepts": 0,
            "unique_symbols": 0,
            "symbols_with_accepts": 0,
            "current_acceptance_streak": 0,
            "current_rejection_streak": 0,
            "longest_acceptance_streak": 0,
            "longest_rejection_streak": 0,
        })
