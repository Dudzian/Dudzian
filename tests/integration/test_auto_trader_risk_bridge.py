"""Integration tests for the auto-trader risk bridge helpers."""
from __future__ import annotations

import pytest

from bot_core.auto_trader.risk_bridge import GuardrailTrigger, RiskDecision


def test_risk_decision_to_dict_includes_optional_fields() -> None:
    decision = RiskDecision(
        should_trade=True,
        fraction=0.65,
        state="ready",
        reason="ok",
        details={"signal": "buy", "metadata": {"regime": "trend"}},
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        mode="live",
        cooldown_active=True,
        cooldown_remaining_s=12.5,
        cooldown_reason="post_trade_cooldown",
    )

    payload = decision.to_dict()

    assert payload["should_trade"] is True
    assert payload["fraction"] == pytest.approx(0.65)
    assert payload["state"] == "ready"
    assert payload["stop_loss_pct"] == pytest.approx(0.02)
    assert payload["take_profit_pct"] == pytest.approx(0.05)
    assert payload["cooldown_active"] is True
    assert payload["cooldown_remaining_s"] == pytest.approx(12.5)
    assert payload["cooldown_reason"] == "post_trade_cooldown"
    assert payload["details"] == {"signal": "buy", "metadata": {"regime": "trend"}}


def test_guardrail_trigger_serialisation_preserves_thresholds() -> None:
    trigger = GuardrailTrigger(
        name="volatility_guard",
        label="Volatility spike",
        comparator=">=",
        threshold=1.5,
        unit="sigma",
        value=1.75,
    )

    payload = trigger.to_dict()

    assert payload == {
        "name": "volatility_guard",
        "label": "Volatility spike",
        "comparator": ">=",
        "threshold": pytest.approx(1.5),
        "unit": "sigma",
        "value": pytest.approx(1.75),
    }
