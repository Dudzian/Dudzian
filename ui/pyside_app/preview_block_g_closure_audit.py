"""Pure-data FUNCTIONAL-PREVIEW-9.9 BLOK G closure audit.

This helper statically closes BLOK G by referencing the already accepted 9.0
through 9.8 pure-data artifacts. It is intentionally inert: no runtime, no
orders, no fills, no lifecycle mutation, no market/account data, no export, and
no UI/QML integration are introduced here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_controlled_paper_order_intent import (
    CONTROLLED_INTENT_DECISION,
    CONTROLLED_INTENT_STATUS,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_7,
)
from ui.pyside_app.preview_paper_decision_to_order_contract import (
    BLOCK_STATUS as PAPER_DECISION_TO_ORDER_CONTRACT_STATUS,
    CONTRACT_DECISION,
    PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND,
    PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_1,
)
from ui.pyside_app.preview_paper_fill_simulator_contract import (
    FILL_SIMULATOR_CONTRACT_DECISION,
    FILL_SIMULATOR_CONTRACT_STATUS,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_8,
)
from ui.pyside_app.preview_paper_order_audit_envelope import (
    AUDIT_ENVELOPE_DECISION,
    AUDIT_ENVELOPE_STATUS,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_4,
)
from ui.pyside_app.preview_paper_order_intent_read_model import (
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND,
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION,
    READ_MODEL_DECISION,
    READ_MODEL_STATUS,
    READY_FOR_BLOCK_G_2,
)
from ui.pyside_app.preview_paper_order_intent_selection_gate import (
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_6,
    SELECTION_GATE_DECISION,
    SELECTION_GATE_STATUS,
)
from ui.pyside_app.preview_paper_order_lifecycle_audit import (
    LIFECYCLE_AUDIT_DECISION,
    LIFECYCLE_AUDIT_STATUS,
    PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND,
    PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_9,
)
from ui.pyside_app.preview_paper_order_static_fixture import (
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_3,
    STATIC_FIXTURE_DECISION,
    STATIC_FIXTURE_STATUS,
)

PREVIEW_BLOCK_G_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_g_closure_audit.v1"
PREVIEW_BLOCK_G_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_g_closure_audit"
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.9"
BLOCK_G_CLOSURE_STATUS: Final[str] = "block_g_closure_audit_complete_ready_for_block_h"
BLOCK_G_CLOSURE_DECISION: Final[str] = (
    "CLOSE_BLOCK_G_PAPER_ONLY_DECISION_TO_ORDER_PATH_NO_RUNTIME_EXECUTION"
)
READY_FOR_BLOCK_H: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.0"
NEXT_STEP_TITLE: Final[str] = "BLOK H — READ-ONLY MARKET DATA ADAPTER CONTRACT"

CASE_IDS: Final[tuple[str, ...]] = (
    "baseline_btc_no_intent_no_order",
    "eth_size_preview_no_intent_no_order",
    "sol_risk_blocked_preview_no_intent_no_order",
    "unknown_input_keys_reported_no_execution",
)

BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "paper order generation",
    "paper order submission",
    "paper fill simulation execution",
    "paper fill event generation",
    "paper order lifecycle mutation",
    "paper order lifecycle transition execution",
    "paper runtime execution",
    "risk governor execution now",
    "market data adapter implementation",
    "market data fetch",
    "account fetch",
    "audit export",
    "live/testnet/account/secrets/export/cloud",
    "TradingController / DecisionEnvelope",
    "QML changes / new QML calls",
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
    "no market data adapter import",
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


def _step_reference(
    *,
    step: str,
    title: str,
    schema_version: str | None = None,
    kind: str | None = None,
    surface_name: str | None = None,
    surface_kind: str | None = None,
    status: str,
    decision: str,
    ready_flag: bool,
) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "step": step,
        "title": title,
        "status": status,
        "decision": decision,
        "ready_flag": ready_flag,
        "execution_allowed": False,
        "runtime_allowed": False,
        "live_or_testnet_allowed": False,
    }
    if schema_version is not None:
        reference["schema_version"] = schema_version
    if kind is not None:
        reference["kind"] = kind
    if surface_name is not None:
        reference["surface_name"] = surface_name
    if surface_kind is not None:
        reference["surface_kind"] = surface_kind
    return reference


def _block_g_step_references() -> list[dict[str, Any]]:
    return [
        _step_reference(
            step="9.0",
            title="paper decision-to-order contract",
            schema_version=PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND,
            status=PAPER_DECISION_TO_ORDER_CONTRACT_STATUS,
            decision=CONTRACT_DECISION,
            ready_flag=READY_FOR_BLOCK_G_1,
        ),
        _step_reference(
            step="9.1",
            title="paper order intent read model",
            schema_version=PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND,
            status=READ_MODEL_STATUS,
            decision=READ_MODEL_DECISION,
            ready_flag=READY_FOR_BLOCK_G_2,
        ),
        _step_reference(
            step="9.2",
            title="paper order static fixture",
            schema_version=PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
            status=STATIC_FIXTURE_STATUS,
            decision=STATIC_FIXTURE_DECISION,
            ready_flag=READY_FOR_BLOCK_G_3,
        ),
        _step_reference(
            step="9.3",
            title="paper order audit envelope",
            schema_version=PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
            status=AUDIT_ENVELOPE_STATUS,
            decision=AUDIT_ENVELOPE_DECISION,
            ready_flag=READY_FOR_BLOCK_G_4,
        ),
        _step_reference(
            step="9.4",
            title="paper order UI read-only surface",
            surface_name="paper_order_ui_read_only_surface_confirmed_by_tests",
            surface_kind="functional_preview_block_g_paper_order_ui_read_only_surface",
            status="paper_order_ui_read_only_surface_ready_read_only_no_execution",
            decision="CONFIRM_PAPER_ORDER_UI_READ_ONLY_SURFACE_ONLY_NO_QML_EXPANSION",
            ready_flag=True,
        ),
        _step_reference(
            step="9.5",
            title="paper order intent selection gate",
            schema_version=PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
            status=SELECTION_GATE_STATUS,
            decision=SELECTION_GATE_DECISION,
            ready_flag=READY_FOR_BLOCK_G_6,
        ),
        _step_reference(
            step="9.6",
            title="controlled paper order intent preview",
            schema_version=PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
            kind=PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
            status=CONTROLLED_INTENT_STATUS,
            decision=CONTROLLED_INTENT_DECISION,
            ready_flag=READY_FOR_BLOCK_G_7,
        ),
        _step_reference(
            step="9.7",
            title="paper fill simulator contract",
            schema_version=PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
            status=FILL_SIMULATOR_CONTRACT_STATUS,
            decision=FILL_SIMULATOR_CONTRACT_DECISION,
            ready_flag=READY_FOR_BLOCK_G_8,
        ),
        _step_reference(
            step="9.8",
            title="paper order lifecycle audit",
            schema_version=PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION,
            kind=PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND,
            status=LIFECYCLE_AUDIT_STATUS,
            decision=LIFECYCLE_AUDIT_DECISION,
            ready_flag=READY_FOR_BLOCK_G_9,
        ),
    ]


def _case_coverage() -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "selection_gate_available": True,
            "controlled_intent_preview_available": True,
            "fill_simulator_contract_available": True,
            "lifecycle_audit_available": True,
            "order_generation_allowed": False,
            "submission_allowed": False,
            "fill_simulation_allowed": False,
            "fill_event_generation_allowed": False,
            "lifecycle_mutation_allowed": False,
            "runtime_execution_allowed": False,
            "live_or_testnet_allowed": False,
            "account_or_secrets_allowed": False,
            "export_allowed": False,
        }
        for case_id in CASE_IDS
    ]


def build_preview_block_g_closure_audit() -> dict[str, Any]:
    """Return deterministic plain-data BLOK G closure audit."""

    audit: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_G_CLOSURE_AUDIT_SCHEMA_VERSION,
        "closure_audit_kind": PREVIEW_BLOCK_G_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_g_closure_status": BLOCK_G_CLOSURE_STATUS,
        "block_g_closure_decision": BLOCK_G_CLOSURE_DECISION,
        "ready_for_block_h": READY_FOR_BLOCK_H,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_g_step_references": _block_g_step_references(),
        "block_g_completion_matrix": {
            "contract_9_0_complete": True,
            "read_model_9_1_complete": True,
            "static_fixture_9_2_complete": True,
            "audit_envelope_9_3_complete": True,
            "ui_read_only_surface_9_4_complete": True,
            "selection_gate_9_5_complete": True,
            "controlled_intent_preview_9_6_complete": True,
            "fill_simulator_contract_9_7_complete": True,
            "lifecycle_audit_9_8_complete": True,
            "block_g_complete": True,
            "ready_for_block_h": True,
            "paper_only_scope_preserved": True,
            "runtime_execution_added": False,
            "live_or_testnet_added": False,
            "order_submission_added": False,
            "fill_simulation_added": False,
            "lifecycle_mutation_added": False,
            "market_data_fetch_added": False,
            "audit_export_added": False,
            "qml_expansion_added": False,
        },
        "block_g_static_artifact_summary": {
            "paper_decision_to_order_contract_available": True,
            "paper_order_intent_read_model_available": True,
            "paper_order_static_fixture_available": True,
            "paper_order_audit_envelope_available": True,
            "paper_order_ui_read_only_surface_available": True,
            "paper_order_intent_selection_gate_available": True,
            "controlled_paper_order_intent_preview_available": True,
            "paper_fill_simulator_contract_available": True,
            "paper_order_lifecycle_audit_available": True,
            "all_artifacts_json_serializable": True,
            "all_artifacts_plain_data": True,
            "all_artifacts_local_only": True,
            "all_artifacts_paper_only": True,
            "all_artifacts_non_executable": True,
        },
        "block_g_case_coverage": _case_coverage(),
        "block_g_no_execution_evidence": {
            "closure_audit_evaluated": True,
            "decision_engine_execution_performed": False,
            "paper_order_intent_executable_generated": False,
            "paper_order_generated": False,
            "paper_order_submitted": False,
            "paper_fill_simulated": False,
            "paper_fill_event_generated": False,
            "paper_order_lifecycle_mutated": False,
            "paper_order_lifecycle_transition_executed": False,
            "paper_runtime_execution_performed": False,
            "risk_governor_execution_performed": False,
            "market_data_fetch_performed": False,
            "account_fetch_performed": False,
            "audit_export_performed": False,
            "trading_controller_touched": False,
            "decision_envelope_touched": False,
            "live_execution_performed": False,
            "testnet_execution_performed": False,
            "secrets_read_performed": False,
            "export_performed": False,
            "qml_runtime_expansion_performed": False,
        },
        "block_g_boundary_checks": {
            "local_only": True,
            "paper_only": True,
            "block_g_closure_audit_only": True,
            "block_g_complete": True,
            "ready_for_block_h": True,
            "read_only_market_data_next_block_only": True,
            "market_data_adapter_implemented_now": False,
            "paper_decision_to_order_path_static_complete": True,
            "paper_order_intent_preview_allowed_now": True,
            "paper_fill_simulation_contract_allowed_now": True,
            "order_lifecycle_audit_allowed_now": True,
            "runtime_loop_allowed": False,
            "command_dispatch_execution_allowed": False,
            "lifecycle_execution_allowed": False,
            "decision_engine_execution_allowed": False,
            "paper_order_generation_allowed_now": False,
            "paper_order_submission_allowed_now": False,
            "paper_fill_simulation_allowed_now": False,
            "fill_event_generation_allowed_now": False,
            "order_lifecycle_mutation_allowed_now": False,
            "lifecycle_transition_allowed_now": False,
            "paper_runtime_execution_allowed_now": False,
            "risk_governor_execution_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "account_fetch_allowed_now": False,
            "audit_export_allowed": False,
            "trading_controller_allowed": False,
            "decision_envelope_allowed": False,
            "strategy_execution_allowed": False,
            "ai_scoring_execution_allowed": False,
            "model_inference_execution_allowed": False,
            "live_mode_allowed": False,
            "testnet_mode_allowed_initially": False,
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
        },
        "blocked_capabilities": list(BLOCKED_CAPABILITIES),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "handoff_to_block_h": {
            "handoff_ready": True,
            "handoff_target_block": "H",
            "handoff_target_step": "FUNCTIONAL-PREVIEW-10.0",
            "handoff_target_title": "READ-ONLY MARKET DATA ADAPTER CONTRACT",
            "handoff_scope": "read_only_market_data_contract_only",
            "handoff_runtime_execution_allowed": False,
            "handoff_live_trading_allowed": False,
            "handoff_account_fetch_allowed": False,
            "handoff_order_submission_allowed": False,
            "handoff_requires_new_block_contract": True,
            "handoff_notes": [
                "BLOK H is separate from this closure audit.",
                "BLOK H may define a read-only market data contract only.",
                "Live trading, account access, orders, fills, runtime, and exports remain blocked.",
            ],
        },
        "closure_summary": {
            "block_g_closed": True,
            "block_g_name": "PAPER-ONLY DECISION-TO-ORDER PATH",
            "completed_steps": ["9.0", "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"],
            "closure_step": "9.9",
            "ready_for_block_h": True,
            "paper_only_path_complete": True,
            "runtime_execution_present": False,
            "live_or_testnet_present": False,
            "orders_or_fills_present": False,
            "qml_expansion_present": False,
            "status": "block_g_closed_ready_for_block_h",
        },
        "status": "ready_for_functional_preview_10_0_block_h_read_only_market_data_contract",
    }
    return deepcopy(audit)


__all__ = [
    "BLOCK_G_CLOSURE_DECISION",
    "BLOCK_G_CLOSURE_STATUS",
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_BLOCK_G_CLOSURE_AUDIT_KIND",
    "PREVIEW_BLOCK_G_CLOSURE_AUDIT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_H",
    "STEP_ID",
    "build_preview_block_g_closure_audit",
]
