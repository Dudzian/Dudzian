"""Pure-data BLOK E closure audit for the controlled UI preview action path.

The audit is intentionally source/runtime inert: it imports no PySide modules, does
not touch QML engines, filesystems, networks, exchanges, orders, secrets, or any
runtime loop.  It freezes the BLOK E closure decision as deterministic
JSON-serializable data that can be safely exposed to QML if needed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

PREVIEW_BLOCK_E_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_e_closure_audit.v1"
PREVIEW_BLOCK_E_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_e_closure_audit"
BLOCK_ID: Final[str] = "E"
BLOCK_STATUS: Final[str] = "controlled_ui_preview_action_path_complete_no_execution"
CLOSURE_DECISION: Final[str] = "CLOSE_BLOCK_E_AS_CONTROLLED_PREVIEW_ACTION_PATH_READY"
READY_FOR_BLOCK_F: Final[bool] = True
NEXT_BLOCK: Final[str] = "F"
NEXT_BLOCK_TITLE: Final[str] = "DECISION ENGINE DRY-RUN INTEGRATION"
ALLOWED_QML_CALL_LITERAL: Final[str] = (
    "paperRuntimeActionDispatchBridge.previewSelectAction("
    '"paper_runtime_snapshot_refresh_requested")'
)

COMPLETED_CAPABILITIES: Final[tuple[str, ...]] = (
    "block_d_bridge_registered_in_real_qml_context",
    "operator_dashboard_reads_action_dispatch_snapshot",
    "disabled_action_catalog_visible",
    "selection_preview_gate_reconciled_after_first_call",
    "exactly_one_preview_select_action_qml_call",
    "snapshot_refresh_preview_only_call_runtime_proven",
    "no_shadowing_of_paper_runtime_action_dispatch_bridge",
    "no_smoke_ad_hoc_bridge_creation",
    "operator_workflow_pair_propagation_green",
    "selected_terminal_pair_direct_writer_guarded",
    "windows_qml_green_on_current_head",
)

REQUIRED_EVIDENCE: Final[tuple[str, ...]] = (
    "central_context_registration_present",
    "read_only_qml_snapshot_consumption_present",
    "disabled_intent_catalog_and_gate_surface_present",
    "exactly_one_snapshot_refresh_preview_select_action_call_present",
    "runtime_smoke_preview_call_accepted_not_executed",
    "no_dashboard_bridge_shadowing",
    "no_smoke_ad_hoc_bridge_injection",
    "operator_workflow_pair_propagation_green",
    "selected_terminal_pair_direct_writer_guard_green",
    "windows_qml_current_head_green",
)

BOUNDARY_CHECKS: Final[dict[str, bool]] = {
    "paper_only": True,
    "local_only": True,
    "execution_allowed": False,
    "execution_performed": False,
    "runtime_loop_started": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "order_generation_allowed": False,
    "order_submission_allowed": False,
    "fills_allowed": False,
    "trading_controller_touched": False,
    "decision_envelope_touched": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "preview_select_source_control_allowed": False,
    "reset_preview_selection_allowed": False,
    "start_stop_pause_resume_qml_calls_allowed": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}

BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "live trading",
    "testnet/sandbox trading",
    "order generation",
    "order submission",
    "fills",
    "runtime loop execution",
    "lifecycle command execution",
    "dynamic action dispatch",
    "previewSelectSourceControl",
    "resetPreviewSelection",
    "start/stop/pause/resume QML calls",
    "TradingController integration",
    "DecisionEnvelope integration",
    "account/balance fetch",
    "secrets read/export",
    "cloud/external export",
    "EXE packaging",
)

OPEN_ITEMS_FOR_FUTURE_BLOCKS: Final[tuple[str, ...]] = (
    "block_f_decision_engine_dry_run_integration",
    "block_g_paper_only_decision_to_order_path",
    "block_h_read_only_real_market_adapter",
    "block_i_testnet_sandbox_adapter",
    "block_j_risk_governor_limits_kill_switch",
    "block_k_observability_audit_rollback_soak",
    "block_l_live_canary_live_transition_gates",
    "future_exe_packaging_block",
)


def build_preview_block_e_closure_audit() -> dict[str, Any]:
    """Return deterministic plain-data BLOK E closure evidence."""

    audit: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_E_CLOSURE_AUDIT_SCHEMA_VERSION,
        "audit_kind": PREVIEW_BLOCK_E_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "block_status": BLOCK_STATUS,
        "closure_decision": CLOSURE_DECISION,
        "ready_for_block_f": READY_FOR_BLOCK_F,
        "next_block": NEXT_BLOCK,
        "next_block_title": NEXT_BLOCK_TITLE,
        "summary": (
            "BLOK E closes the controlled UI preview action path: one snapshot "
            "refresh intent can be selected from QML, accepted as preview-only, "
            "and never executed. Runtime, lifecycle, orders, live/testnet, "
            "account, secrets, export, and EXE packaging remain blocked."
        ),
        "completed_capabilities": list(COMPLETED_CAPABILITIES),
        "required_evidence": list(REQUIRED_EVIDENCE),
        "boundary_checks": dict(BOUNDARY_CHECKS),
        "action_dispatch_contract": {
            "allowed_qml_method_call_count": 1,
            "allowed_qml_method": "previewSelectAction",
            "allowed_qml_action": "paper_runtime_snapshot_refresh_requested",
            "allowed_qml_call_literal": ALLOWED_QML_CALL_LITERAL,
            "allowed_qml_call_status": "accepted_intent_not_executed",
            "runtime_proof_required": True,
            "runtime_proof_present": True,
            "execution_allowed": False,
            "execution_performed": False,
        },
        "qml_context_bridge_contract": {
            "registered_context_property": "paperRuntimeActionDispatchBridge",
            "central_registration_required": True,
            "operator_dashboard_shadowing_allowed": False,
            "operator_dashboard_shadowing_present": False,
            "smoke_ad_hoc_bridge_creation_allowed": False,
            "smoke_ad_hoc_bridge_creation_present": False,
            "second_qqmlapplicationengine_allowed": False,
            "app_py_ad_hoc_registration_allowed": False,
        },
        "runtime_smoke_contract": {
            "runtime_smoke_required": True,
            "runtime_smoke_present": True,
            "preview_call_accepted": True,
            "preview_call_executed": False,
            "runtime_loop_started": False,
        },
        "operator_workflow_contract": {
            "select_scanner_pair_propagates_to_terminal_pair": True,
            "terminal_open_preserves_selected_pair": True,
            "terminal_active_pair_uses_shared_state_before_default": True,
            "link_usdt_not_terminal_default": True,
            "selected_terminal_pair_direct_writers_guarded": True,
        },
        "selected_terminal_pair_writer_contract": {
            "direct_writers_guarded": True,
            "allowed_writer_helper": "setTerminalPairFromSource",
            "last_writer_instrumentation_required": True,
            "operator_workflow_regression_green": True,
        },
        "exe_direction_contract": {
            "final_artifact_direction": "windows_exe",
            "bat_files_are_dev_preview_only": True,
            "bat_productization_allowed": False,
            "pyinstaller_packaging_in_scope": False,
        },
        "blocked_capabilities": list(BLOCKED_CAPABILITIES),
        "open_items_for_future_blocks": list(OPEN_ITEMS_FOR_FUTURE_BLOCKS),
        "status": "closed_ready_for_block_f_no_execution",
    }
    return deepcopy(audit)


__all__ = [
    "BLOCK_ID",
    "BLOCK_STATUS",
    "CLOSURE_DECISION",
    "NEXT_BLOCK",
    "NEXT_BLOCK_TITLE",
    "PREVIEW_BLOCK_E_CLOSURE_AUDIT_KIND",
    "PREVIEW_BLOCK_E_CLOSURE_AUDIT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_F",
    "build_preview_block_e_closure_audit",
]
