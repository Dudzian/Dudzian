"""Pure-data BLOK G paper decision-to-order path contract.

This helper is intentionally inert. It starts BLOK G as a contract-only,
local/paper-only decision-to-order preview path after BLOK F closure, while
keeping order intent generation, order submission, fills, runtime execution,
controllers, envelopes, live/testnet adapters, accounts, secrets, exports, QML,
app launchers, dependency declarations, workflows, and packaging out of scope.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_block_f_closure_audit import build_preview_block_f_closure_audit

PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_paper_decision_to_order_contract.v1"
)
PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_g_paper_decision_to_order_contract"
)
BLOCK_ID: Final[str] = "G"
BLOCK_STATUS: Final[str] = "paper_decision_to_order_contract_ready_no_order_execution"
CONTRACT_DECISION: Final[str] = "START_BLOCK_G_WITH_CONTRACT_ONLY_NO_ORDER_EXECUTION"
READY_FOR_BLOCK_G_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.1"
NEXT_STEP_TITLE: Final[str] = "PAPER ORDER INTENT READ MODEL"

PAPER_PATH_SCOPE: Final[dict[str, bool]] = {
    "paper_only": True,
    "local_only": True,
    "dry_run_source_allowed": True,
    "paper_order_intent_allowed_future_step": True,
    "paper_order_preview_allowed_future_step": True,
    "paper_order_submission_allowed_now": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_runtime_execution_allowed_now": False,
    "live_trading_allowed": False,
    "testnet_trading_allowed_initially": False,
    "real_account_balance_allowed": False,
    "live_credentials_allowed": False,
}

ALLOWED_FUTURE_INPUTS: Final[tuple[str, ...]] = (
    "dry_run_decision_preview",
    "decision_reason_summary",
    "risk_check_preview",
    "audit_event_preview",
    "operator_selected_pair",
    "operator_selected_candidate",
    "paper_order_intent_context_id",
    "paper_order_intent_size_preview",
    "paper_order_intent_side_preview",
    "paper_order_intent_type_preview",
)

ALLOWED_FUTURE_OUTPUTS: Final[tuple[str, ...]] = (
    "paper_order_intent_preview",
    "paper_order_preview",
    "paper_order_validation_preview",
    "paper_order_refusal_preview",
    "paper_order_audit_event_preview",
    "paper_order_gate_status",
    "paper_order_risk_gate_preview",
    "paper_order_no_execution_status",
)

REQUIRED_GATES_BEFORE_ORDER_PATH: Final[tuple[str, ...]] = (
    "block_g_1_paper_order_intent_read_model",
    "block_g_2_paper_order_static_fixture",
    "block_g_3_paper_order_audit_envelope",
    "block_g_4_ui_read_only_paper_order_surface",
    "block_g_5_controlled_paper_order_intent_selection_gate",
    "risk_governor_required_before_any_order_execution",
    "manual_operator_confirmation_required_before_any_order_submission",
    "kill_switch_required_before_any_order_submission",
    "paper_only_adapter_required_before_any_order_submission",
    "live_credentials_refusal_required",
)

BOUNDARY_CHECKS: Final[dict[str, bool]] = {
    "local_only": True,
    "paper_only": True,
    "contract_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_allowed_now": False,
    "paper_order_generation_allowed_now": False,
    "paper_order_submission_allowed_now": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_runtime_execution_allowed_now": False,
    "risk_governor_execution_allowed_now": False,
    "manual_operator_confirmation_required_before_order": True,
    "kill_switch_required_before_order": True,
    "trading_controller_allowed": False,
    "decision_envelope_allowed": False,
    "strategy_execution_allowed": False,
    "ai_scoring_execution_allowed": False,
    "model_inference_execution_allowed": False,
    "runtime_loop_allowed": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed_initially": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "real_account_balance_allowed": False,
    "live_credentials_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "new_qml_method_calls_allowed": False,
    "qml_changes_allowed": False,
    "exe_packaging_in_scope": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}

BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "order generation now",
    "order submission now",
    "paper fills now",
    "live trading",
    "testnet/sandbox trading initially",
    "real account balance",
    "live credentials",
    "TradingController integration",
    "DecisionEnvelope integration",
    "strategy execution",
    "AI/scoring execution",
    "model inference",
    "runtime loop execution",
    "command dispatch execution",
    "lifecycle command execution",
    "dynamic action dispatch",
    "new QML action calls",
    "QML runtime behavior changes",
    "secrets read/export",
    "cloud/external export",
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
    "no QML changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
)

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_1_paper_order_intent_read_model",
    "functional_preview_9_2_paper_order_static_fixture",
    "functional_preview_9_3_paper_order_audit_envelope",
    "functional_preview_9_4_ui_read_only_paper_order_surface",
    "functional_preview_9_5_controlled_paper_order_intent_selection_gate",
    "functional_preview_9_6_controlled_paper_order_intent_no_submission",
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)


def _block_f_closure_reference() -> dict[str, Any]:
    closure = build_preview_block_f_closure_audit()
    return {
        "schema_version": closure["schema_version"],
        "audit_kind": closure["audit_kind"],
        "block_status": closure["block_status"],
        "closure_decision": closure["closure_decision"],
        "ready_for_block_g": closure["ready_for_block_g"],
        "next_block": closure["next_block"],
        "next_block_title": closure["next_block_title"],
    }


def build_preview_paper_decision_to_order_contract() -> dict[str, Any]:
    """Return deterministic plain-data BLOK G paper path contract."""

    contract: dict[str, Any] = {
        "schema_version": PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION,
        "contract_kind": PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND,
        "block": BLOCK_ID,
        "block_status": BLOCK_STATUS,
        "contract_decision": CONTRACT_DECISION,
        "ready_for_block_g_1": READY_FOR_BLOCK_G_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_f_closure_reference": _block_f_closure_reference(),
        "paper_path_scope": dict(PAPER_PATH_SCOPE),
        "allowed_future_inputs": list(ALLOWED_FUTURE_INPUTS),
        "allowed_future_outputs": list(ALLOWED_FUTURE_OUTPUTS),
        "required_gates_before_order_path": list(REQUIRED_GATES_BEFORE_ORDER_PATH),
        "boundary_checks": dict(BOUNDARY_CHECKS),
        "blocked_capabilities": list(BLOCKED_CAPABILITIES),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "future_steps": list(FUTURE_STEPS),
        "status": "ready_for_functional_preview_9_1_no_order_execution",
    }
    return deepcopy(contract)


__all__ = [
    "BLOCK_ID",
    "BLOCK_STATUS",
    "CONTRACT_DECISION",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND",
    "PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_1",
    "build_preview_paper_decision_to_order_contract",
]
