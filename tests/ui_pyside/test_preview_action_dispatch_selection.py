"""Tests for the BLOK D non-executing action selection result contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from types import MappingProxyType
from typing import Any

import pytest

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_catalog import (
    SOURCE_PANEL,
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_selection import (
    SELECTION_RESULT_KIND,
    SELECTION_RESULT_SCHEMA_VERSION,
    UNKNOWN_SELECTION_STATUS,
    PaperRuntimeActionDispatchSelectionResult,
    build_paper_runtime_action_dispatch_selection_result,
)

REQUIRED_FIELDS = {
    "schema_version",
    "result_kind",
    "requested_action_or_control",
    "resolved_action",
    "resolved_source_control",
    "source_panel",
    "source_control",
    "catalog_action_found",
    "catalog_item",
    "audit_envelope",
    "dispatch_evidence",
    "safe_to_bind_from_ui",
    "execution_allowed",
    "execution_performed",
    "paper_only",
    "local_only",
    "runtime_mode",
    "blocked_reason",
    "refusal_reason",
    "result_status",
    "result_message",
    "boundary_checks",
}


def _assert_no_execution(result: PaperRuntimeActionDispatchSelectionResult) -> None:
    assert result.execution_allowed is False
    assert result.execution_performed is False
    assert result.audit_envelope.execution_allowed is False
    assert result.audit_envelope.execution_performed is False
    assert result.dispatch_evidence.execution_allowed is False
    assert result.dispatch_evidence.execution_performed is False
    assert result.dispatch_evidence.order_generation_allowed is False
    assert result.dispatch_evidence.order_submission_allowed is False


def _assert_fail_closed(result: PaperRuntimeActionDispatchSelectionResult) -> None:
    assert result.catalog_action_found is False
    assert result.safe_to_bind_from_ui is False
    assert result.execution_allowed is False
    assert result.execution_performed is False
    assert result.catalog_item is None
    assert result.blocked_reason
    assert result.refusal_reason
    assert result.result_status == UNKNOWN_SELECTION_STATUS
    assert result.boundary_checks["selection_fail_closed"] is True


def test_allowed_action_selection_result_is_accepted_not_executed() -> None:
    result = build_paper_runtime_action_dispatch_selection_result("paper_runtime_start_requested")

    assert result.schema_version == SELECTION_RESULT_SCHEMA_VERSION
    assert result.result_kind == SELECTION_RESULT_KIND
    assert result.requested_action_or_control == "paper_runtime_start_requested"
    assert result.resolved_action == "paper_runtime_start_requested"
    assert result.resolved_source_control == "paper-runtime-start"
    assert result.catalog_action_found is True
    assert result.catalog_item is not None
    assert result.catalog_item.action == "paper_runtime_start_requested"
    assert result.audit_envelope.audit_status == ACCEPTED_INTENT_NOT_EXECUTED
    assert result.result_status == ACCEPTED_INTENT_NOT_EXECUTED
    assert result.safe_to_bind_from_ui is True
    assert result.paper_only is True
    assert result.local_only is True
    assert result.runtime_mode == "paper"
    assert set(asdict(result)) == REQUIRED_FIELDS
    _assert_no_execution(result)


def test_source_control_selection_result_maps_to_action() -> None:
    result = build_paper_runtime_action_dispatch_selection_result("paper-runtime-start")

    assert result.catalog_action_found is True
    assert result.resolved_action == "paper_runtime_start_requested"
    assert result.resolved_source_control == "paper-runtime-start"
    assert result.source_control == "paper-runtime-start"
    assert result.catalog_item is not None
    assert result.catalog_item.source_control == "paper-runtime-start"
    assert result.result_status == ACCEPTED_INTENT_NOT_EXECUTED
    assert result.safe_to_bind_from_ui is True
    _assert_no_execution(result)


@pytest.mark.parametrize(
    ("requested", "reason"),
    [
        ("unexpected_action", "blocked_unknown_action"),
        ("unknown-source-control", "blocked_unknown_action"),
        ("start_live_runtime", "blocked_live_mode"),
        ("start_testnet_runtime", "blocked_testnet_mode"),
        ("submit_order", "blocked_order_generation_submission"),
        ("account_balance_fetch", "blocked_account_balance_fetch"),
        ("export_cloud_report", "blocked_export_cloud_secrets"),
        ("read_secret_value", "blocked_export_cloud_secrets"),
    ],
)
def test_rejected_selection_matrix_fails_closed(requested: str, reason: str) -> None:
    result = build_paper_runtime_action_dispatch_selection_result(requested)

    _assert_fail_closed(result)
    assert result.refusal_reason == reason
    assert result.audit_envelope.refusal_reason == reason
    _assert_no_execution(result)


@pytest.mark.parametrize("requested", [None, "", "   ", 123, object()])
def test_invalid_action_or_control_fails_closed_without_exception_leakage(requested: Any) -> None:
    result = build_paper_runtime_action_dispatch_selection_result(requested)

    _assert_fail_closed(result)
    assert result.refusal_reason in {"invalid_action_empty", "invalid_action_non_string"}
    _assert_no_execution(result)


def test_operator_confirmation_is_audited_but_does_not_enable_execution() -> None:
    result = build_paper_runtime_action_dispatch_selection_result(
        "paper_runtime_stop_requested",
        operator_confirmation=True,
        operator_note="operator acknowledged",
    )

    assert result.audit_envelope.operator_confirmation is True
    assert result.audit_envelope.operator_note == "operator acknowledged"
    assert result.safe_to_bind_from_ui is True
    assert result.execution_allowed is False
    assert result.execution_performed is False
    _assert_no_execution(result)


def test_source_panel_and_source_control_are_deterministic() -> None:
    result = build_paper_runtime_action_dispatch_selection_result(
        "paper-runtime-pause",
        source_panel="RuntimeSessionControlPanel",
    )

    assert result.source_panel == "runtimesessioncontrolpanel"
    assert result.source_control == "paper-runtime-pause"
    assert result.resolved_source_control == "paper-runtime-pause"
    assert result.audit_envelope.source_panel == "runtimesessioncontrolpanel"
    assert result.audit_envelope.source_control == "paper-runtime-pause"


def test_selection_output_is_deterministic() -> None:
    first = build_paper_runtime_action_dispatch_selection_result(
        "paper_runtime_resume_requested", operator_confirmation=True, reason="same"
    )
    second = build_paper_runtime_action_dispatch_selection_result(
        "paper_runtime_resume_requested", operator_confirmation=True, reason="same"
    )

    assert first == second


def test_result_nested_structures_are_immutable_and_copy_safe() -> None:
    result = build_paper_runtime_action_dispatch_selection_result("paper-runtime-snapshot-refresh")

    with pytest.raises(FrozenInstanceError):
        result.resolved_action = "changed"  # type: ignore[misc]
    assert not isinstance(result.boundary_checks, dict)
    assert not isinstance(result.boundary_checks, MappingProxyType)
    with pytest.raises(TypeError):
        result.boundary_checks["execution_disabled"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        result.audit_envelope.boundary_checks["execution_disabled"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        result.dispatch_evidence.boundary_checks["execution_disabled"] = False  # type: ignore[index]

    reread = build_paper_runtime_action_dispatch_selection_result("paper-runtime-snapshot-refresh")
    assert reread.boundary_checks["execution_disabled"] is True
    assert reread.boundary_checks["catalog_action_found"] is True


def test_result_reuses_catalog_item_and_audit_envelope_semantics() -> None:
    catalog = build_paper_runtime_action_dispatch_catalog()
    result = build_paper_runtime_action_dispatch_selection_result(
        "paper_runtime_snapshot_refresh_requested", catalog=catalog
    )
    expected_item = catalog.actions[-1]

    assert result.catalog_item is expected_item
    assert result.audit_envelope.audit_status == expected_item.audit_envelope.audit_status
    assert result.dispatch_evidence.normalized_action == expected_item.normalized_action
    assert result.boundary_checks["catalog_action_found"] is True
    assert result.boundary_checks["selection_safe_to_bind_from_ui"] is True


def test_catalog_action_found_behaves_correctly() -> None:
    found = build_paper_runtime_action_dispatch_selection_result("paper-runtime-start")
    missing = build_paper_runtime_action_dispatch_selection_result("paper-runtime-missing")

    assert found.catalog_action_found is True
    assert missing.catalog_action_found is False
    assert found.catalog_item is not None
    assert missing.catalog_item is None


def test_default_source_panel_uses_static_catalog_panel_without_qml_objects() -> None:
    result = build_paper_runtime_action_dispatch_selection_result("paper_runtime_start_requested")

    assert result.source_panel == SOURCE_PANEL
    assert result.audit_envelope.source_panel == SOURCE_PANEL
