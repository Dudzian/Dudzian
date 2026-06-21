"""Pure-data BLOK F decision engine dry-run contract preflight.

This module is intentionally inert.  It defines the deterministic data contract
for a future local/paper dry-run preview and does not execute or import the
real decision engine, runtime loops, controllers, adapters, orders, secrets, or
UI/QML bindings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_decision_engine_dry_run_contract.v1"
)
PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_f_decision_engine_dry_run_contract"
)
BLOCK_ID: Final[str] = "F"
BLOCK_STATUS: Final[str] = "decision_engine_dry_run_contract_ready_no_execution"
DRY_RUN_MODE: Final[str] = "local_paper_dry_run"
CONTRACT_DECISION: Final[str] = "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION"
READY_FOR_BLOCK_F_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-8.1"
NEXT_STEP_TITLE: Final[str] = "DECISION ENGINE DRY-RUN READ MODEL SNAPSHOT"

ALLOWED_DRY_RUN_INPUTS: Final[tuple[str, ...]] = (
    "local_preview_state_snapshot",
    "paper_runtime_snapshot",
    "scanner_candidate_snapshot",
    "risk_preview_snapshot",
    "portfolio_preview_snapshot",
    "operator_selected_pair",
    "operator_selected_candidate",
    "dry_run_context_id",
)

ALLOWED_DRY_RUN_OUTPUTS: Final[tuple[str, ...]] = (
    "dry_run_decision_preview",
    "decision_reason_summary",
    "input_snapshot_echo",
    "risk_check_preview",
    "confidence_preview",
    "no_order_decision_preview",
    "audit_event_preview",
    "blocked_reason_preview",
)

REQUIRED_BOUNDARIES: Final[dict[str, bool]] = {
    "local_only": True,
    "paper_only": True,
    "dry_run_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
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
    "live trading",
    "testnet/sandbox trading",
    "decision engine execution",
    "TradingController integration",
    "DecisionEnvelope integration",
    "strategy execution",
    "AI/scoring execution",
    "runtime loop execution",
    "lifecycle command execution",
    "dynamic action dispatch",
    "order generation",
    "order submission",
    "fills",
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
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no filesystem I/O",
    "no network I/O",
    "no secrets access",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
)

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_8_1_decision_engine_dry_run_read_model_snapshot",
    "functional_preview_8_2_decision_engine_dry_run_static_fixture",
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
    "functional_preview_8_5_block_f_closure_audit",
)


def build_preview_decision_engine_dry_run_contract() -> dict[str, Any]:
    """Return deterministic plain-data BLOK F dry-run contract preflight."""

    contract: dict[str, Any] = {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
        "contract_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
        "block": BLOCK_ID,
        "block_status": BLOCK_STATUS,
        "dry_run_mode": DRY_RUN_MODE,
        "contract_decision": CONTRACT_DECISION,
        "ready_for_block_f_1": READY_FOR_BLOCK_F_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "summary": (
            "BLOK F starts with a local/paper dry-run data contract only. "
            "The contract lists future preview inputs and outputs while keeping "
            "decision engine execution, controllers, order paths, live/testnet, "
            "secrets, exports, new QML method calls, and packaging out of scope."
        ),
        "allowed_dry_run_inputs": list(ALLOWED_DRY_RUN_INPUTS),
        "allowed_dry_run_outputs": list(ALLOWED_DRY_RUN_OUTPUTS),
        "required_boundaries": dict(REQUIRED_BOUNDARIES),
        "blocked_capabilities": list(BLOCKED_CAPABILITIES),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "future_steps": list(FUTURE_STEPS),
        "status": "contract_ready_no_execution",
    }
    return deepcopy(contract)


__all__ = [
    "BLOCK_ID",
    "BLOCK_STATUS",
    "CONTRACT_DECISION",
    "DRY_RUN_MODE",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_F_1",
    "build_preview_decision_engine_dry_run_contract",
]
