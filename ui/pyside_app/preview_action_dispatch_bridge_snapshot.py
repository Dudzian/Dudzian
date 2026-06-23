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
from ui.pyside_app.preview_read_only_market_data_audit_envelope import (
    build_preview_read_only_market_data_audit_envelope,
)
from ui.pyside_app.preview_read_only_market_data_controlled_refresh_preview import (
    build_preview_read_only_market_data_controlled_refresh_preview,
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
READ_ONLY_MARKET_DATA_UI_SURFACE_STATUS: Final[str] = (
    "read_only_market_data_ui_read_only_surface_ready_no_actions"
)
READ_ONLY_MARKET_DATA_UI_SURFACE_DECISION: Final[str] = (
    "BUILD_UI_READ_ONLY_SURFACE_ONLY_NO_QML_ACTIONS_NO_NETWORK_IO"
)
READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS: Final[str] = (
    "read_only_market_data_bridge_snapshot_ready_no_qml_actions_no_refresh"
)
READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION: Final[str] = (
    "BUILD_BRIDGE_SNAPSHOT_DATA_ONLY_NO_REFRESH_NO_BRIDGE_API_CHANGES"
)
READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.8"
READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP_TITLE: Final[str] = (
    "READ-ONLY MARKET DATA CLOSURE AUDIT"
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
    read_only_market_data_audit_envelope = build_preview_read_only_market_data_audit_envelope()
    market_data_fixture_summary = dict(
        read_only_market_data_audit_envelope["fixture_row_audit_summary"]
    )
    market_data_quality_summary = dict(
        read_only_market_data_audit_envelope["data_quality_audit_preview"]
    )
    market_data_no_fetch_evidence = dict(
        read_only_market_data_audit_envelope["no_fetch_no_export_no_execution_evidence"]
    )
    market_data_boundary_checks = dict(read_only_market_data_audit_envelope["boundary_checks"])
    read_only_market_data_no_network_summary = {
        "network_io_allowed_now": bool(market_data_boundary_checks["network_io_allowed_now"]),
        "network_io_performed": bool(market_data_no_fetch_evidence["network_io_performed"]),
        "exchange_connection_opened": bool(
            market_data_no_fetch_evidence["exchange_connection_opened"]
        ),
        "no_network_io": not bool(market_data_no_fetch_evidence["network_io_performed"]),
    }
    read_only_market_data_no_fetch_summary = {
        "market_data_fetch_allowed_now": bool(
            market_data_boundary_checks["market_data_fetch_allowed_now"]
        ),
        "market_data_fetch_performed": bool(
            market_data_no_fetch_evidence["market_data_fetch_performed"]
        ),
        "no_market_fetch": not bool(market_data_no_fetch_evidence["market_data_fetch_performed"]),
    }
    read_only_market_data_no_export_summary = {
        "audit_export_allowed_now": bool(market_data_boundary_checks["audit_export_allowed_now"]),
        "audit_export_performed": bool(market_data_no_fetch_evidence["audit_export_performed"]),
        "export_performed": bool(market_data_no_fetch_evidence["export_performed"]),
        "no_audit_export": not bool(market_data_no_fetch_evidence["audit_export_performed"]),
    }
    read_only_market_data_ui_read_only_summary = {
        "ui_surface_status": READ_ONLY_MARKET_DATA_UI_SURFACE_STATUS,
        "ui_surface_decision": READ_ONLY_MARKET_DATA_UI_SURFACE_DECISION,
        "read_only_surface_only": True,
        "qml_method_calls_added": False,
        "qml_actions_added": False,
        "buttons_added": False,
        "network_io_performed": False,
        "market_data_fetch_performed": False,
        "audit_export_performed": False,
        "next_step_after_ui_surface": "FUNCTIONAL-PREVIEW-10.5",
    }
    read_only_market_data_controlled_refresh_preview = (
        build_preview_read_only_market_data_controlled_refresh_preview()
    )
    refresh_scope = dict(
        read_only_market_data_controlled_refresh_preview["controlled_refresh_preview_scope"]
    )
    refresh_preview = dict(
        read_only_market_data_controlled_refresh_preview["controlled_refresh_preview"]
    )
    refresh_boundary_summary = dict(
        read_only_market_data_controlled_refresh_preview["refresh_preview_boundary_summary"]
    )
    refresh_no_execution_evidence = dict(
        read_only_market_data_controlled_refresh_preview[
            "no_refresh_no_fetch_no_execution_evidence"
        ]
    )
    read_only_market_data_bridge_no_refresh_summary = {
        "refresh_execution_allowed_now": bool(refresh_preview["refresh_execution_allowed_now"]),
        "refresh_performed_now": bool(refresh_preview["refresh_performed_now"]),
        "controlled_refresh_performed": bool(
            refresh_no_execution_evidence["controlled_refresh_performed"]
        ),
        "refresh_performed": bool(refresh_no_execution_evidence["refresh_performed"]),
        "no_real_refresh": not bool(refresh_no_execution_evidence["refresh_performed"]),
    }
    read_only_market_data_bridge_no_fetch_summary = {
        "market_data_fetch_allowed_now": bool(refresh_scope["market_data_fetch_allowed_now"]),
        "market_data_fetch_performed": bool(
            refresh_no_execution_evidence["market_data_fetch_performed"]
        ),
        "no_market_fetch": not bool(refresh_no_execution_evidence["market_data_fetch_performed"]),
    }
    read_only_market_data_bridge_no_network_summary = {
        "network_io_allowed_now": bool(refresh_preview["network_io_allowed_now"]),
        "network_io_performed": bool(refresh_no_execution_evidence["network_io_performed"]),
        "exchange_connection_opened": bool(
            refresh_no_execution_evidence["exchange_connection_opened"]
        ),
        "no_network_io": not bool(refresh_no_execution_evidence["network_io_performed"]),
    }
    read_only_market_data_no_bridge_api_change_summary = {
        "bridge_api_changes_allowed": False,
        "bridge_api_changes_performed": bool(
            refresh_no_execution_evidence["bridge_api_changes_performed"]
        ),
        "new_qml_method_calls_allowed": False,
        "qml_changes_performed": bool(refresh_no_execution_evidence["qml_changes_performed"]),
        "no_bridge_api_changes": not bool(
            refresh_no_execution_evidence["bridge_api_changes_performed"]
        ),
        "no_new_qml_method_calls": True,
    }
    read_only_market_data_bridge_snapshot_summary = {
        "snapshot_status": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS,
        "snapshot_decision": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION,
        "snapshot_data_only": True,
        "controlled_refresh_preview_read": True,
        "qml_safe": True,
        "refresh_performed": bool(refresh_no_execution_evidence["refresh_performed"]),
        "market_data_fetch_performed": bool(
            refresh_no_execution_evidence["market_data_fetch_performed"]
        ),
        "network_io_performed": bool(refresh_no_execution_evidence["network_io_performed"]),
        "bridge_api_changes_performed": bool(
            refresh_no_execution_evidence["bridge_api_changes_performed"]
        ),
        "qml_changes_performed": bool(refresh_no_execution_evidence["qml_changes_performed"]),
        "next_step": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP,
    }
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
        "read_only_market_data_audit_envelope": read_only_market_data_audit_envelope,
        "read_only_market_data_audit_status": read_only_market_data_audit_envelope[
            "market_data_audit_envelope_status"
        ],
        "read_only_market_data_audit_next_step": read_only_market_data_audit_envelope["next_step"],
        "read_only_market_data_audit_ready_for_ui_surface": True,
        "read_only_market_data_audit_ready_for_block_h_4": bool(
            read_only_market_data_audit_envelope["ready_for_block_h_4"]
        ),
        "read_only_market_data_audit_event_count": int(market_data_fixture_summary["row_count"]),
        "read_only_market_data_audited_symbols": list(
            market_data_fixture_summary["audited_symbols"]
        ),
        "read_only_market_data_normal_preview_symbols": list(
            market_data_fixture_summary["normal_preview_symbols"]
        ),
        "read_only_market_data_low_liquidity_preview_symbols": list(
            market_data_fixture_summary["low_liquidity_preview_symbols"]
        ),
        "read_only_market_data_stale_preview_symbols": list(
            market_data_fixture_summary["stale_preview_symbols"]
        ),
        "read_only_market_data_quality_summary": {
            "normal_preview_count": int(market_data_quality_summary["normal_preview_count"]),
            "low_liquidity_preview_count": int(
                market_data_quality_summary["low_liquidity_preview_count"]
            ),
            "stale_preview_count": int(market_data_quality_summary["stale_preview_count"]),
        },
        "read_only_market_data_no_network_summary": read_only_market_data_no_network_summary,
        "read_only_market_data_no_fetch_summary": read_only_market_data_no_fetch_summary,
        "read_only_market_data_no_export_summary": read_only_market_data_no_export_summary,
        "read_only_market_data_ui_read_only_summary": read_only_market_data_ui_read_only_summary,
        "read_only_market_data_bridge_snapshot_status": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS,
        "read_only_market_data_bridge_snapshot_decision": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION,
        "read_only_market_data_bridge_snapshot_next_step": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP,
        "read_only_market_data_bridge_snapshot_next_step_title": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP_TITLE,
        "read_only_market_data_bridge_snapshot_ready_for_block_h_8": True,
        "read_only_market_data_controlled_refresh_preview": read_only_market_data_controlled_refresh_preview,
        "read_only_market_data_controlled_refresh_status": read_only_market_data_controlled_refresh_preview[
            "market_data_controlled_refresh_preview_status"
        ],
        "read_only_market_data_controlled_refresh_next_step": (
            read_only_market_data_controlled_refresh_preview["next_step"]
        ),
        "read_only_market_data_controlled_refresh_ready_for_block_h_7": bool(
            read_only_market_data_controlled_refresh_preview["ready_for_block_h_7"]
        ),
        "read_only_market_data_allowed_refresh_preview_count": int(
            refresh_preview["allowed_refresh_preview_count"]
        ),
        "read_only_market_data_default_refresh_selection_id": refresh_preview[
            "default_selection_id"
        ],
        "read_only_market_data_allowed_refresh_symbols": list(
            refresh_boundary_summary["allowed_symbols"]
        ),
        "read_only_market_data_normal_refresh_preview_symbols": list(
            refresh_boundary_summary["normal_preview_symbols"]
        ),
        "read_only_market_data_low_liquidity_refresh_preview_symbols": list(
            refresh_boundary_summary["low_liquidity_preview_symbols"]
        ),
        "read_only_market_data_stale_refresh_preview_symbols": list(
            refresh_boundary_summary["stale_preview_symbols"]
        ),
        "read_only_market_data_refresh_preview_boundary_summary": refresh_boundary_summary,
        "read_only_market_data_no_refresh_summary": read_only_market_data_bridge_no_refresh_summary,
        "read_only_market_data_no_fetch_summary": read_only_market_data_bridge_no_fetch_summary,
        "read_only_market_data_no_network_summary": read_only_market_data_bridge_no_network_summary,
        "read_only_market_data_no_bridge_api_change_summary": read_only_market_data_no_bridge_api_change_summary,
        "read_only_market_data_bridge_snapshot_summary": read_only_market_data_bridge_snapshot_summary,
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
    "READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION",
    "READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP",
    "READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP_TITLE",
    "READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS",
    "build_paper_runtime_action_dispatch_bridge_snapshot",
]
