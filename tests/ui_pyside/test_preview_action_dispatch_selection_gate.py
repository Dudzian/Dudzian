"""Tests for BLOK E selection preview gate contract."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import MappingProxyType
from typing import Any

from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_action_dispatch_selection_gate import (
    SELECTION_PREVIEW_GATE_KIND,
    SELECTION_PREVIEW_GATE_SCHEMA_VERSION,
    SELECTION_PREVIEW_GATE_STATUS,
    build_paper_runtime_action_dispatch_selection_preview_gate,
)

SIMPLE_TYPES = (dict, list, str, bool, int, type(None))
BLOCKED_METHODS = {
    "previewSelectAction",
    "previewSelectSourceControl",
    "resetPreviewSelection",
}


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    assert not is_dataclass(value)
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def _assert_gate_locked_no_execution(gate: dict[str, Any]) -> None:
    assert gate["qml_method_calls_allowed_now"] is False
    assert gate["execution_allowed"] is False
    assert gate["execution_performed"] is False
    assert gate["order_submission_allowed"] is False
    assert gate["lifecycle_execution_allowed"] is False
    assert gate["live_mode_allowed"] is False
    assert gate["testnet_mode_allowed"] is False
    assert gate["account_fetch_allowed"] is False
    assert gate["secrets_export_allowed"] is False
    assert gate["cloud_export_allowed"] is False
    assert gate["paper_only"] is True
    assert gate["local_only"] is True


def test_selection_preview_gate_contract_is_locked_read_only_and_qml_safe() -> None:
    gate = build_paper_runtime_action_dispatch_selection_preview_gate()

    assert gate["gate_schema_version"] == SELECTION_PREVIEW_GATE_SCHEMA_VERSION
    assert gate["gate_kind"] == SELECTION_PREVIEW_GATE_KIND
    assert gate["gate_status"] == SELECTION_PREVIEW_GATE_STATUS
    assert gate["selection_preview_allowed_in_next_step"] is True
    assert gate["runtime_mode"] == "paper"
    assert set(gate["blocked_current_qml_methods"]) == BLOCKED_METHODS
    assert gate["allowed_next_qml_methods"] == [
        {
            "method": "previewSelectAction",
            "availability": "next_step_only_not_active_now",
            "condition": "enable only after source guards, QML no-handler audit, and smoke tests pass",
        }
    ]
    assert "locked/read-only" in gate["operator_message"]
    assert "next step may enable previewSelectAction" in gate["operator_message"]
    _assert_gate_locked_no_execution(gate)
    _assert_simple_types_only(gate)
    json.dumps(gate, sort_keys=True)


def test_selection_preview_gate_is_deterministic_and_copy_safe() -> None:
    first = build_paper_runtime_action_dispatch_selection_preview_gate()
    first["blocked_current_qml_methods"].clear()
    first["allowed_next_qml_methods"][0]["method"] = "mutated"

    second = build_paper_runtime_action_dispatch_selection_preview_gate()
    third = build_paper_runtime_action_dispatch_selection_preview_gate()

    assert set(second["blocked_current_qml_methods"]) == BLOCKED_METHODS
    assert second["allowed_next_qml_methods"][0]["method"] == "previewSelectAction"
    assert second == third


def test_bridge_snapshot_exposes_selection_preview_gate_without_enabling_execution() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    gate = snapshot["selection_preview_gate"]

    assert gate["gate_kind"] == SELECTION_PREVIEW_GATE_KIND
    assert gate["gate_status"] == SELECTION_PREVIEW_GATE_STATUS
    assert snapshot["execution_allowed"] is False
    assert snapshot["execution_performed"] is False
    _assert_gate_locked_no_execution(gate)
