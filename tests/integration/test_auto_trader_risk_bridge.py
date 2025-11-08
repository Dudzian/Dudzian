"""Integration tests for the auto-trader risk bridge helpers."""
from __future__ import annotations

from typing import Any, Mapping

import pytest

from bot_core.auto_trader.risk_bridge import (
    GuardrailTrigger,
    RiskDecision,
    normalize_guardrail_triggers,
)


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


def test_risk_decision_roundtrip_from_mapping_handles_strings() -> None:
    payload = {
        "should_trade": "1",
        "fraction": "0.45",
        "state": "active",
        "reason": "threshold_met",
        "details": {"edge_bps": "12.5"},
        "stop_loss_pct": "0.015",
        "take_profit_pct": "0.045",
        "mode": "paper",
        "cooldown_active": "",
        "cooldown_remaining_s": "30",
        "cooldown_reason": "risk_reset",
    }

    decision = RiskDecision.from_mapping(payload)
    assert decision.should_trade is True
    assert decision.fraction == pytest.approx(0.45)
    assert decision.state == "active"
    assert decision.reason == "threshold_met"
    assert decision.stop_loss_pct == pytest.approx(0.015)
    assert decision.take_profit_pct == pytest.approx(0.045)
    assert decision.mode == "paper"
    assert decision.cooldown_active is False
    assert decision.cooldown_remaining_s == pytest.approx(30.0)
    assert decision.cooldown_reason == "risk_reset"
    assert decision.details == {"edge_bps": "12.5"}

    assert decision.to_dict()["fraction"] == pytest.approx(0.45)


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


def test_guardrail_trigger_from_mapping_defaults_optional_fields() -> None:
    payload = {
        "name": "leverage_guard",
        "comparator": "<",
        "threshold": "3",
        "value": "2.5",
    }

    trigger = GuardrailTrigger.from_mapping(payload)
    assert trigger.name == "leverage_guard"
    assert trigger.label == ""
    assert trigger.comparator == "<"
    assert trigger.threshold == pytest.approx(3.0)
    assert trigger.value == pytest.approx(2.5)
    assert trigger.unit is None


def test_normalize_guardrail_triggers_preserves_missing_fields() -> None:
    class _LegacyTrigger:
        def to_dict(self) -> Mapping[str, Any]:
            return {
                "name": "legacy_guard",
                "label": "Legacy",
                "comparator": ">=",
                "threshold": "1.75",
                "unit": "ratio",
            }

    raw_payload = [
        {"name": "volatility_guard", "threshold": "1.5", "value": "2.0"},
        GuardrailTrigger(
            name="leverage_guard",
            label="Leverage",
            comparator="<",
            threshold=3.0,
            unit="ratio",
            value=2.5,
        ),
        _LegacyTrigger(),
        "fallback_guard",
    ]

    normalized = normalize_guardrail_triggers(raw_payload)

    assert [trigger.name for trigger, _ in normalized] == [
        "volatility_guard",
        "leverage_guard",
        "legacy_guard",
        "fallback_guard",
    ]

    first_trigger, first_payload = normalized[0]
    assert first_trigger.threshold == pytest.approx(1.5)
    assert first_trigger.value == pytest.approx(2.0)
    assert "threshold" in first_payload and "value" in first_payload

    last_trigger, last_payload = normalized[-1]
    assert last_trigger.name == "fallback_guard"
    assert last_payload == {"name": "fallback_guard"}
