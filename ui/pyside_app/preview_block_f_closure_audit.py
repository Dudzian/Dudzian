"""Pure-data BLOK F closure audit for decision engine dry-run integration.

This helper is intentionally inert. It closes BLOK F as a local/paper,
read-only dry-run integration path and records that no decision engine,
runtime loop, controller, envelope, order path, live/testnet adapter, secrets,
network, filesystem, QML runtime, or packaging path is enabled here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_decision_engine_dry_run_audit_envelope import (
    AUDIT_ENVELOPE_DECISION,
    AUDIT_ENVELOPE_STATUS,
    PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_SCHEMA_VERSION,
)
from ui.pyside_app.preview_decision_engine_dry_run_contract import (
    CONTRACT_DECISION,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
)
from ui.pyside_app.preview_decision_engine_dry_run_read_model import (
    PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
    READ_MODEL_DECISION,
    READ_MODEL_STATUS,
)
from ui.pyside_app.preview_decision_engine_dry_run_static_fixture import (
    PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION,
    STATIC_FIXTURE_DECISION,
    STATIC_FIXTURE_STATUS,
)

PREVIEW_BLOCK_F_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_f_closure_audit.v1"
PREVIEW_BLOCK_F_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_f_closure_audit"
BLOCK_ID: Final[str] = "F"
BLOCK_STATUS: Final[str] = "decision_engine_dry_run_read_only_complete_no_execution"
CLOSURE_DECISION: Final[str] = "CLOSE_BLOCK_F_AS_DRY_RUN_READ_ONLY_INTEGRATION_READY"
READY_FOR_BLOCK_G: Final[bool] = True
NEXT_BLOCK: Final[str] = "G"
NEXT_BLOCK_TITLE: Final[str] = "PAPER-ONLY DECISION-TO-ORDER PATH"
ALLOWED_QML_METHOD_CALL_LITERAL: Final[str] = (
    'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
)

COMPLETED_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_8_0_decision_engine_dry_run_contract",
    "functional_preview_8_1_decision_engine_dry_run_read_model_snapshot",
    "functional_preview_8_2_decision_engine_dry_run_static_fixture",
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
)

REQUIRED_EVIDENCE: Final[tuple[str, ...]] = (
    "contract_ready_no_execution",
    "read_model_snapshot_ready_no_engine_execution",
    "static_fixture_ready_no_engine_execution",
    "audit_envelope_ready_no_engine_execution",
    "ui_read_only_surface_ready_no_engine_execution",
    "exactly_one_qml_preview_select_action_call",
    "no_new_qml_method_calls",
    "no_execution_buttons_added",
    "no_decision_engine_execution",
    "no_trading_controller_touch",
    "no_decision_envelope_touch",
    "no_order_generation",
    "no_order_submission",
    "no_fills",
    "no_live_or_testnet",
    "no_account_fetch",
    "no_secrets_export",
    "no_cloud_or_external_export",
    "exe_direction_preserved",
)

BOUNDARY_CHECKS: Final[dict[str, bool]] = {
    "local_only": True,
    "paper_only": True,
    "dry_run_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "model_inference_execution_allowed": False,
    "trading_controller_allowed": False,
    "decision_envelope_allowed": False,
    "strategy_execution_allowed": False,
    "ai_scoring_execution_allowed": False,
    "runtime_loop_allowed": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "order_generation_allowed": False,
    "order_submission_allowed": False,
    "fills_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "new_qml_method_calls_allowed": False,
    "exe_packaging_in_scope": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}

BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "decision engine execution",
    "real decision recommendation",
    "model inference",
    "risk engine evaluation",
    "TradingController integration",
    "DecisionEnvelope integration",
    "strategy execution",
    "AI/scoring execution",
    "dynamic action dispatch",
    "runtime loop execution",
    "lifecycle command execution",
    "order generation",
    "order submission",
    "fills",
    "live trading",
    "testnet/sandbox trading",
    "account/balance fetch",
    "secrets read/export",
    "cloud/external export",
    "new QML action calls",
    "EXE packaging",
)

SOURCE_BOUNDARIES: Final[tuple[str, ...]] = (
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no strategy/AI/scoring/recommendation import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no account module import",
    "no secrets module import",
    "no filesystem I/O",
    "no network I/O",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
)

OPEN_ITEMS_FOR_FUTURE_BLOCKS: Final[tuple[str, ...]] = (
    "block_g_paper_only_decision_to_order_path",
    "block_h_read_only_real_market_adapter",
    "block_i_testnet_sandbox_adapter",
    "block_j_risk_governor_limits_kill_switch",
    "block_k_observability_audit_rollback_soak",
    "block_l_live_canary_live_transition_gates",
    "future_exe_packaging_block",
)


def _contract_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
        "contract_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": CONTRACT_DECISION,
    }


def _read_model_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
        "read_model_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND,
        "read_model_status": READ_MODEL_STATUS,
        "read_model_decision": READ_MODEL_DECISION,
    }


def _static_fixture_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION,
        "fixture_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND,
        "static_fixture_status": STATIC_FIXTURE_STATUS,
        "static_fixture_decision": STATIC_FIXTURE_DECISION,
    }


def _audit_envelope_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_SCHEMA_VERSION,
        "envelope_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_KIND,
        "audit_envelope_status": AUDIT_ENVELOPE_STATUS,
        "audit_envelope_decision": AUDIT_ENVELOPE_DECISION,
    }


def build_preview_block_f_closure_audit() -> dict[str, Any]:
    """Return deterministic plain-data BLOK F closure evidence."""

    audit: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_F_CLOSURE_AUDIT_SCHEMA_VERSION,
        "audit_kind": PREVIEW_BLOCK_F_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "block_status": BLOCK_STATUS,
        "closure_decision": CLOSURE_DECISION,
        "ready_for_block_g": READY_FOR_BLOCK_G,
        "next_block": NEXT_BLOCK,
        "next_block_title": NEXT_BLOCK_TITLE,
        "summary": (
            "BLOK F is closed as a dry-run/read-only/no-execution integration path. "
            "Steps 8.0 through 8.4 established the contract, read model, static fixture, "
            "audit envelope, and QML read-only surface while keeping decision execution, "
            "runtime dispatch expansion, orders, live/testnet, account, secrets, exports, "
            "and packaging blocked."
        ),
        "completed_steps": list(COMPLETED_STEPS),
        "required_evidence": list(REQUIRED_EVIDENCE),
        "contract_reference": _contract_reference(),
        "read_model_reference": _read_model_reference(),
        "static_fixture_reference": _static_fixture_reference(),
        "audit_envelope_reference": _audit_envelope_reference(),
        "ui_surface_reference": {
            "ui_surface_status": "read_only_surface_ready_no_engine_execution",
            "ready_for_block_f_5": True,
            "next_step_after_ui_surface": "FUNCTIONAL-PREVIEW-8.5",
            "surface_kind": "decision_engine_dry_run_ui_read_only_surface",
        },
        "boundary_checks": dict(BOUNDARY_CHECKS),
        "blocked_capabilities": list(BLOCKED_CAPABILITIES),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "qml_surface_contract": {
            "surface_present": True,
            "surface_object_name": "operatorDashboardDecisionEngineDryRunAuditCard",
            "surface_status_object_name": "operatorDashboardDecisionEngineDryRunAuditStatus",
            "surface_summary_object_name": "operatorDashboardDecisionEngineDryRunAuditSummary",
            "surface_events_object_name": "operatorDashboardDecisionEngineDryRunAuditEvents",
            "surface_read_only": True,
            "new_qml_method_calls_added": False,
            "execution_buttons_added": False,
            "allowed_qml_method_call_count": 1,
            "allowed_qml_method_call_literal": ALLOWED_QML_METHOD_CALL_LITERAL,
            "preview_select_source_control_present": False,
            "reset_preview_selection_present": False,
            "start_stop_pause_resume_calls_present": False,
            "dynamic_action_dispatch_present": False,
        },
        "next_block_contract": {
            "next_block": NEXT_BLOCK,
            "next_block_title": NEXT_BLOCK_TITLE,
            "block_g_start_allowed": True,
            "block_g_must_remain_paper_only": True,
            "block_g_live_trading_allowed": False,
            "block_g_testnet_allowed_initially": False,
            "block_g_requires_new_gates_before_orders": True,
            "block_g_requires_risk_governor_before_any_order_path": True,
            "block_g_must_not_enable_live_credentials": True,
            "block_g_must_not_use_real_account_balance": True,
        },
        "open_items_for_future_blocks": list(OPEN_ITEMS_FOR_FUTURE_BLOCKS),
        "status": "closed_ready_for_block_g_paper_only_no_execution",
    }
    return deepcopy(audit)


__all__ = [
    "BLOCK_ID",
    "BLOCK_STATUS",
    "CLOSURE_DECISION",
    "NEXT_BLOCK",
    "NEXT_BLOCK_TITLE",
    "PREVIEW_BLOCK_F_CLOSURE_AUDIT_KIND",
    "PREVIEW_BLOCK_F_CLOSURE_AUDIT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G",
    "build_preview_block_f_closure_audit",
]
