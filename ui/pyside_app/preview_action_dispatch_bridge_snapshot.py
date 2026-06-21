"""QML-safe BLOK D paper action dispatch bridge snapshot contract.

This module converts the immutable action catalog and non-executing selection
result into plain Python containers for a future QML/UI binding.  It is still a
source-only preview helper: it does not import PySide/QML, wire handlers, start
runtime loops, dispatch commands, execute lifecycle commands, generate or submit
orders, read accounts or secrets, fetch live/testnet data, export files, access
cloud paths, or perform I/O.
"""

from __future__ import annotations

from typing import Final, Any

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_catalog import (
    PaperRuntimeActionDispatchCatalog,
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_contract import RUNTIME_MODE
from ui.pyside_app.preview_action_dispatch_selection import (
    UNKNOWN_SELECTION_STATUS,
    PaperRuntimeActionDispatchSelectionResult,
    build_paper_runtime_action_dispatch_selection_result,
)
from ui.pyside_app.preview_action_dispatch_selection_gate import (
    build_paper_runtime_action_dispatch_selection_preview_gate,
)
from ui.pyside_app.preview_decision_engine_dry_run_audit_envelope import (
    build_preview_decision_engine_dry_run_audit_envelope,
)
from ui.pyside_app.preview_paper_order_audit_envelope import (
    build_preview_paper_order_audit_envelope,
)

BRIDGE_SNAPSHOT_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_bridge_snapshot.v1"
BRIDGE_SNAPSHOT_KIND: Final[str] = "block_d_qml_safe_action_dispatch_bridge_snapshot"
NO_SELECTION_STATUS: Final[str] = "no_selection_preview_only"
_NO_SELECTION_MESSAGE: Final[str] = (
    "No paper runtime action selected; bridge snapshot is preview-only, execution is disabled, "
    "and no action was executed."
)
_ACCEPTED_OPERATOR_MESSAGE: Final[str] = (
    "Paper runtime action selected for future UI binding only; execution remains disabled "
    "and no action was executed."
)
_REJECTED_OPERATOR_MESSAGE: Final[str] = (
    "Paper runtime action selection rejected fail-closed; execution remains disabled "
    "and no action was executed."
)


def build_paper_runtime_action_dispatch_bridge_snapshot(
    requested_action_or_control: object = None,
    *,
    source_panel: object = "",
    source_control: object = "",
    operator_confirmation: bool = False,
    operator_note: object = "",
    reason: object = "",
    catalog: PaperRuntimeActionDispatchCatalog | None = None,
) -> dict[str, Any]:
    """Build a deterministic, QML-safe action bridge snapshot without execution."""

    action_catalog = catalog or build_paper_runtime_action_dispatch_catalog()
    has_selection = _has_requested_selection(requested_action_or_control, source_control)
    selection_request = (
        source_control
        if not _has_requested_selection(requested_action_or_control, "")
        else requested_action_or_control
    )
    selected_result = build_paper_runtime_action_dispatch_selection_result(
        selection_request,
        source_panel=source_panel,
        source_control=source_control,
        operator_confirmation=operator_confirmation,
        operator_note=operator_note,
        reason=reason,
        catalog=action_catalog,
    )
    selected_payload = _selection_to_payload(
        selected_result,
        requested_action_or_control=requested_action_or_control,
        source_control=source_control,
        has_selection=has_selection,
    )
    actions = [_catalog_item_to_payload(item) for item in action_catalog.actions]
    boundary_checks = _bridge_boundary_checks(action_catalog, selected_payload, has_selection)
    status = selected_payload["result_status"]
    decision_engine_dry_run_audit_envelope = build_preview_decision_engine_dry_run_audit_envelope()
    paper_order_audit_envelope = build_preview_paper_order_audit_envelope()
    paper_order_audit_summary = dict(paper_order_audit_envelope["audit_summary"])
    paper_order_audit_boundary_checks = dict(paper_order_audit_envelope["boundary_checks"])
    paper_order_audit_no_execution_summary = {
        "all_events_no_intent_generated": bool(
            paper_order_audit_summary["all_events_no_intent_generated"]
        ),
        "all_events_no_order_generated": bool(
            paper_order_audit_summary["all_events_no_order_generated"]
        ),
        "all_events_no_submission": bool(paper_order_audit_summary["all_events_no_submission"]),
        "all_events_no_fills": bool(paper_order_audit_summary["all_events_no_fills"]),
        "all_events_no_runtime_execution": bool(
            paper_order_audit_summary["all_events_no_runtime_execution"]
        ),
        "all_events_no_live_or_testnet": bool(
            paper_order_audit_summary["all_events_no_live_or_testnet"]
        ),
        "all_events_no_account_or_secrets": bool(
            paper_order_audit_summary["all_events_no_account_or_secrets"]
        ),
        "all_events_no_export": bool(paper_order_audit_summary["all_events_no_export"]),
        "audit_export_allowed": bool(paper_order_audit_boundary_checks["audit_export_allowed"]),
        "audit_export_performed": bool(paper_order_audit_boundary_checks["audit_export_performed"]),
    }

    return {
        "schema_version": BRIDGE_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_kind": BRIDGE_SNAPSHOT_KIND,
        "runtime_mode": RUNTIME_MODE,
        "paper_only": True,
        "local_only": True,
        "execution_allowed": False,
        "execution_performed": False,
        "safe_to_bind_from_ui": bool(action_catalog.safe_to_bind_from_ui),
        "action_count": len(actions),
        "actions": actions,
        "selected_result": selected_payload,
        "selection_preview_gate": build_paper_runtime_action_dispatch_selection_preview_gate(),
        "decision_engine_dry_run_audit_envelope": decision_engine_dry_run_audit_envelope,
        "decision_engine_dry_run_ui_surface_status": (
            "read_only_surface_ready_no_engine_execution"
        ),
        "ready_for_block_f_5": True,
        "next_step_after_ui_surface": "FUNCTIONAL-PREVIEW-8.5",
        "paper_order_audit_envelope": paper_order_audit_envelope,
        "paper_order_audit_status": paper_order_audit_envelope["audit_envelope_status"],
        "paper_order_audit_ready_for_next_step": bool(
            paper_order_audit_envelope["ready_for_block_g_4"]
        ),
        "paper_order_audit_next_step": paper_order_audit_envelope["next_step"],
        "paper_order_audit_ready_for_ui_surface": True,
        "paper_order_audit_ready_for_block_g_4": bool(
            paper_order_audit_envelope["ready_for_block_g_4"]
        ),
        "paper_order_audit_event_count": int(paper_order_audit_summary["event_count"]),
        "paper_order_audit_unknown_input_key_events": int(
            paper_order_audit_summary["unknown_input_key_events"]
        ),
        "paper_order_audit_no_execution_summary": paper_order_audit_no_execution_summary,
        "boundary_checks": boundary_checks,
        "operator_message": _operator_message(status),
        "status": status,
    }


def _has_requested_selection(requested_action_or_control: object, source_control: object) -> bool:
    if isinstance(requested_action_or_control, str):
        return bool(requested_action_or_control.strip())
    if requested_action_or_control is not None:
        return True
    if isinstance(source_control, str):
        return bool(source_control.strip())
    return source_control is not None


def _catalog_item_to_payload(item: object) -> dict[str, Any]:
    return {
        "action": str(getattr(item, "action", "")),
        "source_control": str(getattr(item, "source_control", "")),
        "label": str(getattr(item, "label", "")),
        "description": str(getattr(item, "description", "")),
        "requires_operator_confirmation": bool(
            getattr(item, "requires_operator_confirmation", False)
        ),
        "safe_to_bind_from_ui": bool(getattr(item, "safe_to_bind_from_ui", False)),
        "execution_allowed": False,
        "execution_performed": False,
        "paper_only": bool(getattr(item, "paper_only", True)),
        "local_only": bool(getattr(item, "local_only", True)),
        "runtime_mode": str(getattr(item, "runtime_mode", RUNTIME_MODE)),
        "audit_status": str(getattr(item, "audit_status", ACCEPTED_INTENT_NOT_EXECUTED)),
        "blocked_reason": str(getattr(item, "blocked_reason", "")),
        "refusal_reason": str(getattr(item, "refusal_reason", "")),
    }


def _selection_to_payload(
    result: PaperRuntimeActionDispatchSelectionResult,
    *,
    requested_action_or_control: object,
    source_control: object,
    has_selection: bool,
) -> dict[str, Any]:
    status = result.result_status if has_selection else NO_SELECTION_STATUS
    boundary_checks = dict(result.boundary_checks)
    boundary_checks.update(
        {
            "bridge_snapshot_preview_only": True,
            "bridge_snapshot_no_selection": not has_selection,
            "execution_disabled": True,
            "execution_not_performed": True,
        }
    )
    if not has_selection:
        boundary_checks.update(
            {
                "catalog_action_found": False,
                "selection_fail_closed": True,
                "selection_safe_to_bind_from_ui": False,
            }
        )

    return {
        "requested_action_or_control": _simple_value(requested_action_or_control),
        "resolved_action": result.resolved_action if has_selection else "",
        "resolved_source_control": result.resolved_source_control if has_selection else "",
        "source_panel": result.source_panel,
        "source_control": _simple_text(source_control) or result.source_control,
        "catalog_action_found": bool(result.catalog_action_found and has_selection),
        "safe_to_bind_from_ui": bool(result.safe_to_bind_from_ui and has_selection),
        "execution_allowed": False,
        "execution_performed": False,
        "paper_only": True,
        "local_only": True,
        "runtime_mode": RUNTIME_MODE,
        "blocked_reason": result.blocked_reason if has_selection else "no_action_selected",
        "refusal_reason": result.refusal_reason if has_selection else "no_action_selected",
        "result_status": status,
        "result_message": result.result_message if has_selection else _NO_SELECTION_MESSAGE,
        "boundary_checks": _plain_bool_dict(boundary_checks),
    }


def _bridge_boundary_checks(
    catalog: PaperRuntimeActionDispatchCatalog,
    selected_payload: dict[str, Any],
    has_selection: bool,
) -> dict[str, bool]:
    values = dict(catalog.boundary_checks)
    values.update(
        {
            "bridge_snapshot_preview_only": True,
            "bridge_snapshot_qml_safe_payload": True,
            "bridge_snapshot_no_selection": not has_selection,
            "selected_catalog_action_found": bool(selected_payload["catalog_action_found"]),
            "selected_fail_closed": not bool(selected_payload["catalog_action_found"]),
            "paper_only": True,
            "local_only": True,
            "execution_disabled": True,
            "execution_not_performed": True,
        }
    )
    return _plain_bool_dict(values)


def _operator_message(status: str) -> str:
    if status == NO_SELECTION_STATUS:
        return _NO_SELECTION_MESSAGE
    if status == ACCEPTED_INTENT_NOT_EXECUTED:
        return _ACCEPTED_OPERATOR_MESSAGE
    if status == UNKNOWN_SELECTION_STATUS:
        return _REJECTED_OPERATOR_MESSAGE
    return _REJECTED_OPERATOR_MESSAGE


def _plain_bool_dict(values: dict[str, object]) -> dict[str, bool]:
    return {str(key): bool(value) for key, value in values.items()}


def _simple_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _simple_value(value: object) -> str | int | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


__all__ = [
    "BRIDGE_SNAPSHOT_KIND",
    "BRIDGE_SNAPSHOT_SCHEMA_VERSION",
    "NO_SELECTION_STATUS",
    "build_paper_runtime_action_dispatch_bridge_snapshot",
]
