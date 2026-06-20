"""Tests for controlled BLOK D Qt bridge context registration preflight."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

try:
    from PySide6.QtCore import QObject
except ImportError as exc:  # pragma: no cover - dependency preflight installs PySide6 first.
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import NO_SELECTION_STATUS
from ui.pyside_app.preview_action_dispatch_qt_bridge import PaperRuntimeActionDispatchQtBridge
from ui.pyside_app.preview_action_dispatch_qt_bridge_registration import (
    PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY,
    QT_BRIDGE_REGISTRATION_KIND,
    QT_BRIDGE_REGISTRATION_SCHEMA_VERSION,
    register_paper_runtime_action_dispatch_qt_bridge,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_ROOT = REPO_ROOT / "ui" / "pyside_app"
SIMPLE_TYPES = (dict, list, str, bool, int, type(None))


class FakeContext:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def setContextProperty(self, name: str, value: object) -> None:  # noqa: N802 - Qt API name.
        self.calls.append((name, value))


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


def _assert_registration_no_execution(evidence: dict[str, Any]) -> None:
    assert evidence["execution_allowed"] is False
    assert evidence["execution_performed"] is False
    assert evidence["snapshot"]["execution_allowed"] is False
    assert evidence["snapshot"]["execution_performed"] is False
    assert evidence["snapshot"]["qt_bridge_execution_allowed"] is False
    assert evidence["snapshot"]["qt_bridge_execution_performed"] is False
    assert evidence["snapshot"]["selected_result"]["execution_allowed"] is False
    assert evidence["snapshot"]["selected_result"]["execution_performed"] is False


def test_registration_helper_registers_bridge_on_fake_context_exactly_once() -> None:
    context = FakeContext()

    evidence = register_paper_runtime_action_dispatch_qt_bridge(context)

    assert len(context.calls) == 1
    name, bridge = context.calls[0]
    assert name == PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY
    assert isinstance(bridge, PaperRuntimeActionDispatchQtBridge)
    assert isinstance(bridge, QObject)
    assert evidence["schema_version"] == QT_BRIDGE_REGISTRATION_SCHEMA_VERSION
    assert evidence["registration_kind"] == QT_BRIDGE_REGISTRATION_KIND
    assert evidence["registered"] is True
    assert evidence["registration_performed"] is True
    assert (
        evidence["context_property_name"]
        == PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY
    )
    assert evidence["blocked_reason"] == ""
    _assert_registration_no_execution(evidence)


def test_default_property_name_is_stable() -> None:
    assert PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY == (
        "paperRuntimeActionDispatchBridge"
    )


def test_explicit_bridge_instance_is_reused_and_snapshot_matches() -> None:
    context = FakeContext()
    bridge = PaperRuntimeActionDispatchQtBridge()
    bridge_snapshot = bridge.previewSelectAction("paper_runtime_pause_requested")

    evidence = register_paper_runtime_action_dispatch_qt_bridge(context, bridge=bridge)

    assert context.calls == [(PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY, bridge)]
    assert evidence["snapshot"] == bridge_snapshot
    _assert_registration_no_execution(evidence)


def test_custom_property_name_is_trimmed_and_registered() -> None:
    context = FakeContext()

    evidence = register_paper_runtime_action_dispatch_qt_bridge(
        context,
        property_name="  customPaperDispatchBridge  ",
    )

    assert context.calls[0][0] == "customPaperDispatchBridge"
    assert evidence["context_property_name"] == "customPaperDispatchBridge"
    assert evidence["registered"] is True


@pytest.mark.parametrize("property_name", ["", "   ", None, 42])
def test_invalid_custom_property_name_fail_closes(property_name: object) -> None:
    context = FakeContext()

    evidence = register_paper_runtime_action_dispatch_qt_bridge(
        context,
        property_name=property_name,
    )

    assert context.calls == []
    assert evidence["registered"] is False
    assert evidence["registration_performed"] is False
    assert evidence["blocked_reason"] == "invalid_context_property_name"
    _assert_registration_no_execution(evidence)


def test_invalid_context_missing_set_context_property_fail_closed() -> None:
    context = object()

    evidence = register_paper_runtime_action_dispatch_qt_bridge(context)

    assert evidence["registered"] is False
    assert evidence["registration_performed"] is False
    assert evidence["blocked_reason"] == "missing_set_context_property"
    assert evidence["boundary_checks"]["missing_set_context_property"] is True
    _assert_registration_no_execution(evidence)


def test_registration_snapshot_qml_safe_and_json_serializable() -> None:
    evidence = register_paper_runtime_action_dispatch_qt_bridge(FakeContext())

    _assert_simple_types_only(evidence)
    encoded = json.dumps(evidence, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["snapshot"]["status"] == NO_SELECTION_STATUS


def test_registered_bridge_default_snapshot_no_selection_no_execution() -> None:
    context = FakeContext()

    evidence = register_paper_runtime_action_dispatch_qt_bridge(context)

    registered_bridge = context.calls[0][1]
    assert isinstance(registered_bridge, PaperRuntimeActionDispatchQtBridge)
    assert evidence["snapshot"]["status"] == NO_SELECTION_STATUS
    assert registered_bridge.snapshot["status"] == NO_SELECTION_STATUS
    _assert_registration_no_execution(evidence)


def test_registered_bridge_preview_select_action_accepted_not_executed() -> None:
    context = FakeContext()
    register_paper_runtime_action_dispatch_qt_bridge(context)
    registered_bridge = context.calls[0][1]
    assert isinstance(registered_bridge, PaperRuntimeActionDispatchQtBridge)

    snapshot = registered_bridge.previewSelectAction("paper_runtime_start_requested")

    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert snapshot["selected_result"]["execution_allowed"] is False
    assert snapshot["selected_result"]["execution_performed"] is False


def test_returned_evidence_and_snapshot_are_copy_safe() -> None:
    context = FakeContext()
    bridge = PaperRuntimeActionDispatchQtBridge()

    first = register_paper_runtime_action_dispatch_qt_bridge(context, bridge=bridge)
    first["snapshot"]["actions"].clear()
    first["snapshot"]["boundary_checks"]["execution_disabled"] = False
    first["boundary_checks"]["execution_disabled"] = False

    second = register_paper_runtime_action_dispatch_qt_bridge(FakeContext(), bridge=bridge)
    assert second["snapshot"]["actions"]
    assert second["snapshot"]["boundary_checks"]["execution_disabled"] is True
    assert second["boundary_checks"]["execution_disabled"] is True


def test_no_real_qml_engine_created_and_no_qml_files_changed() -> None:
    before = {path: path.read_text(encoding="utf-8") for path in QML_ROOT.rglob("*.qml")}

    evidence = register_paper_runtime_action_dispatch_qt_bridge(FakeContext())

    after = {path: path.read_text(encoding="utf-8") for path in QML_ROOT.rglob("*.qml")}
    assert after == before
    assert evidence["qml_engine_touched"] is False
    assert evidence["qml_files_changed"] is False
