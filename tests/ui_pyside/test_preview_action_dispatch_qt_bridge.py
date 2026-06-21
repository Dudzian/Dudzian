"""Tests for the BLOK D thin PySide QtCore action dispatch bridge adapter."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import MappingProxyType
from typing import Any

import pytest

try:
    from PySide6.QtCore import QObject
except ImportError as exc:  # pragma: no cover - dependency preflight installs PySide6 first.
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_bridge_provider import (
    PaperRuntimeActionDispatchBridgeProvider,
)
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import NO_SELECTION_STATUS
from ui.pyside_app.preview_action_dispatch_contract import ALLOWED_PAPER_RUNTIME_ACTIONS
from ui.pyside_app.preview_action_dispatch_qt_bridge import (
    QT_BRIDGE_KIND,
    QT_BRIDGE_SCHEMA_VERSION,
    PaperRuntimeActionDispatchQtBridge,
)
from ui.pyside_app.preview_action_dispatch_selection import UNKNOWN_SELECTION_STATUS

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))


def _assert_no_execution(snapshot: dict[str, Any]) -> None:
    assert snapshot["execution_allowed"] is False
    assert snapshot["execution_performed"] is False
    assert snapshot["provider_execution_allowed"] is False
    assert snapshot["provider_execution_performed"] is False
    assert snapshot["qt_bridge_execution_allowed"] is False
    assert snapshot["qt_bridge_execution_performed"] is False
    assert snapshot["selected_result"]["execution_allowed"] is False
    assert snapshot["selected_result"]["execution_performed"] is False


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


def test_adapter_can_be_constructed_with_qtcore_only() -> None:
    bridge = PaperRuntimeActionDispatchQtBridge()

    assert isinstance(bridge, QObject)
    assert isinstance(bridge._provider, PaperRuntimeActionDispatchBridgeProvider)


def test_default_snapshot_no_selection_no_execution_json_qml_safe() -> None:
    snapshot = PaperRuntimeActionDispatchQtBridge().snapshot

    assert snapshot["qt_bridge_schema_version"] == QT_BRIDGE_SCHEMA_VERSION
    assert snapshot["qt_bridge_kind"] == QT_BRIDGE_KIND
    assert snapshot["status"] == NO_SELECTION_STATUS
    assert snapshot["last_requested_action_or_control"] is None
    _assert_no_execution(snapshot)
    _assert_simple_types_only(snapshot)
    assert json.loads(json.dumps(snapshot, sort_keys=True))["status"] == NO_SELECTION_STATUS


def test_preview_select_action_allowed_returns_accepted_intent_not_executed() -> None:
    bridge = PaperRuntimeActionDispatchQtBridge()
    emitted: list[str] = []
    bridge.snapshotChanged.connect(lambda: emitted.append("changed"))

    snapshot = bridge.previewSelectAction("paper_runtime_start_requested")

    assert emitted == ["changed"]
    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert snapshot["selected_result"]["resolved_action"] == "paper_runtime_start_requested"
    assert snapshot["selected_result"]["resolved_source_control"] == "paper-runtime-start"
    assert bridge.snapshot == snapshot
    _assert_no_execution(snapshot)


def test_preview_select_source_control_maps_to_action() -> None:
    bridge = PaperRuntimeActionDispatchQtBridge()
    snapshot = bridge.previewSelectSourceControl("paper-runtime-pause")

    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert snapshot["selected_result"]["resolved_action"] == "paper_runtime_pause_requested"
    assert snapshot["selected_result"]["resolved_source_control"] == "paper-runtime-pause"
    _assert_no_execution(snapshot)


@pytest.mark.parametrize(
    "requested",
    [
        "unknown",
        "start_live_runtime",
        "start_testnet_runtime",
        "submit_order",
        "account_balance_fetch",
        "export_cloud_report",
        "read_secret_value",
    ],
)
def test_unknown_and_rejected_inputs_fail_closed(requested: str) -> None:
    snapshot = PaperRuntimeActionDispatchQtBridge().previewSelectAction(requested)

    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["last_result_status"] == UNKNOWN_SELECTION_STATUS
    assert snapshot["selected_result"]["boundary_checks"]["selection_fail_closed"] is True
    _assert_no_execution(snapshot)


@pytest.mark.parametrize("requested", [None, "", "   ", 123, object()])
def test_invalid_input_fails_closed_without_exception_leakage_where_python_allows(
    requested: Any,
) -> None:
    snapshot = PaperRuntimeActionDispatchQtBridge().previewSelectAction(requested)

    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["last_result_status"] in {NO_SELECTION_STATUS, UNKNOWN_SELECTION_STATUS}
    _assert_no_execution(snapshot)


def test_operator_confirmation_does_not_enable_execution() -> None:
    snapshot = PaperRuntimeActionDispatchQtBridge().previewSelectAction(
        "paper_runtime_stop_requested",
        True,
        "operator acknowledged",
    )

    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    _assert_no_execution(snapshot)


def test_reset_returns_no_selection_and_emits_signal() -> None:
    bridge = PaperRuntimeActionDispatchQtBridge()
    emitted: list[str] = []
    bridge.snapshotChanged.connect(lambda: emitted.append("changed"))

    bridge.previewSelectAction("paper_runtime_resume_requested")
    snapshot = bridge.resetPreviewSelection()

    assert emitted == ["changed", "changed"]
    assert snapshot["status"] == NO_SELECTION_STATUS
    assert snapshot["last_requested_action_or_control"] is None
    _assert_no_execution(snapshot)


def test_returned_payload_is_copy_safe() -> None:
    bridge = PaperRuntimeActionDispatchQtBridge()
    snapshot = bridge.previewSelectAction("paper-runtime-start")
    snapshot["actions"].clear()
    snapshot["selected_result"]["boundary_checks"]["execution_disabled"] = False
    snapshot["boundary_checks"]["execution_disabled"] = False

    reread = bridge.snapshot
    assert len(reread["actions"]) == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert reread["selected_result"]["boundary_checks"]["execution_disabled"] is True
    assert reread["boundary_checks"]["execution_disabled"] is True
