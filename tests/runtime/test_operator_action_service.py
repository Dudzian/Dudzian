from __future__ import annotations

import re

import pytest

from ui.backend.operator_action_service import OperatorActionService


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("requestFreeze", "freeze"),
        ("requestUnfreeze", "unfreeze"),
        ("requestUnblock", "unblock"),
        ("freeze", "freeze"),
        ("unfreeze", "unfreeze"),
        ("unblock", "unblock"),
    ],
)
def test_record_action_normalizes_operator_aliases(source: str, expected: str) -> None:
    payload = OperatorActionService().record_action(source, None)

    assert payload["action"] == expected
    assert payload["entry"] == {}


def test_record_action_unwraps_nested_record_payload() -> None:
    payload = OperatorActionService().record_action(
        "requestFreeze",
        {"record": {"id": "evt-1", "timestamp": "2025-01-01T00:00:00+00:00"}},
    )

    assert payload["entry"] == {"id": "evt-1", "timestamp": "2025-01-01T00:00:00+00:00"}
    assert payload["timestamp"] == "2025-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "entry,expected_timestamp",
    [
        (
            {"timestamp": "2025-01-01T00:00:00+00:00", "time": "older", "ts": "oldest"},
            "2025-01-01T00:00:00+00:00",
        ),
        ({"time": "2025-01-02T00:00:00+00:00", "ts": "older"}, "2025-01-02T00:00:00+00:00"),
        ({"ts": "2025-01-03T00:00:00+00:00"}, "2025-01-03T00:00:00+00:00"),
    ],
)
def test_record_action_timestamp_priority(entry: dict[str, str], expected_timestamp: str) -> None:
    payload = OperatorActionService().record_action("freeze", entry)

    assert payload["timestamp"] == expected_timestamp


class _VariantCarrier:
    def toVariant(self) -> object:
        return {"record": {"id": "variant", "ts": "from-variant"}}


class _PyObjectCarrier:
    def toVariant(self) -> object:
        raise RuntimeError("simulate missing variant")

    def toPyObject(self) -> object:
        return {"record": {"id": "py-object", "time": "from-pyobject"}}


@pytest.mark.parametrize(
    "entry,expected",
    [
        (_VariantCarrier(), {"id": "variant", "ts": "from-variant"}),
        (_PyObjectCarrier(), {"id": "py-object", "time": "from-pyobject"}),
    ],
)
def test_record_action_supports_variant_branches(entry: object, expected: dict[str, str]) -> None:
    payload = OperatorActionService().record_action("freeze", entry)

    assert payload["entry"] == expected


class _UnmappableEntry:
    pass


def test_record_action_returns_empty_entry_for_unmappable_payload() -> None:
    payload = OperatorActionService().record_action("freeze", _UnmappableEntry())

    assert payload["entry"] == {}
    assert isinstance(payload["timestamp"], str)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", payload["timestamp"])
