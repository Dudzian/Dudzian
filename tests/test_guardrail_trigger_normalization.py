import types

import pytest

from bot_core.auto_trader.app import normalize_guardrail_triggers
from bot_core.auto_trader.risk_bridge import GuardrailTrigger


@pytest.mark.parametrize(
    "raw",
    [None, [], (), set(), "", 0],
)
def test_normalize_guardrail_triggers_empty_inputs(raw):
    assert normalize_guardrail_triggers(raw) == []


def test_normalize_guardrail_triggers_accepts_guardrail_trigger_instance():
    trigger = GuardrailTrigger(
        name="effective_risk",
        label="Effective risk",
        comparator=">=",
        threshold=0.5,
        unit="ratio",
        value=0.75,
    )

    normalized = normalize_guardrail_triggers(trigger)

    assert len(normalized) == 1
    obj, payload = normalized[0]
    assert obj is trigger
    assert payload == trigger.to_dict()


def test_normalize_guardrail_triggers_converts_mapping_payload():
    raw = {
        "name": "volatility",
        "label": 42,
        "comparator": "<=",
        "threshold": "1.25",
        "unit": 123,
    }

    normalized = normalize_guardrail_triggers([raw])

    assert len(normalized) == 1
    obj, payload = normalized[0]
    assert isinstance(obj, types.SimpleNamespace)
    assert obj.name == "volatility"
    assert obj.label == "42"
    assert obj.comparator == "<="
    assert obj.threshold == "1.25"
    assert obj.unit == "123"
    assert payload == {
        "name": "volatility",
        "label": "42",
        "comparator": "<=",
        "threshold": "1.25",
        "unit": "123",
    }


def test_normalize_guardrail_triggers_handles_mapping_of_entries():
    raw = {
        "stress": {"threshold": 1.0},
        "entropy": None,
        "missing": 0.5,
    }

    normalized = normalize_guardrail_triggers(raw)

    assert len(normalized) == 2
    names = [payload["name"] for _, payload in normalized]
    assert names == ["stress", "missing"]
    assert normalized[0][1]["threshold"] == 1.0
    assert normalized[1][1] == {"name": "missing", "value": 0.5}


def test_normalize_guardrail_triggers_accepts_pair_payloads():
    trigger = GuardrailTrigger(
        name="tail_risk",
        label="Tail risk",
        comparator=">",
        threshold=0.9,
        unit="ratio",
        value=1.1,
    )
    normalized = normalize_guardrail_triggers(
        [
            (trigger, {"name": "overridden", "threshold": 0.8}),
            ({"label": "dict"}, {"name": "dict", "threshold": 1.2}),
        ]
    )

    assert len(normalized) == 2

    obj0, payload0 = normalized[0]
    assert obj0 is trigger
    assert payload0 == {"name": "overridden", "threshold": 0.8}

    obj1, payload1 = normalized[1]
    assert isinstance(obj1, types.SimpleNamespace)
    assert obj1.label == "dict"
    assert payload1 == {"name": "dict", "threshold": 1.2}


def test_normalize_guardrail_triggers_wraps_primitives_with_name():
    normalized = normalize_guardrail_triggers(["simple", 1])

    assert len(normalized) == 2
    assert normalized[0][1] == {"name": "simple"}
    assert normalized[1][1] == {"name": "1"}

    for obj, payload in normalized:
        assert isinstance(obj, types.SimpleNamespace)
        assert obj.name == payload["name"]
