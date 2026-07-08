"""FUNCTIONAL-PREVIEW-14.1 Block L runtime activation read model.

Pure-data read model derived from the accepted Block L 14.0 static contract.
It does not activate runtime, execute gates, mutate gate state, start canaries,
create orders, access private endpoints, perform network I/O, or perform
filesystem I/O.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_runtime_activation_contract import (
    build_preview_block_l_runtime_activation_contract,
)

PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_runtime_activation_read_model.v1"
)
PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_l_runtime_activation_read_model"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.1"
BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_STATUS: Final[str] = (
    "block_l_runtime_activation_read_model_ready_no_runtime_activation_no_gate_execution_no_io"
)
BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_DECISION: Final[str] = (
    "READ_MODEL_READY_FOR_14_2_NO_RUNTIME_ACTIVATION_NO_GATE_EXECUTION_NO_LIVE_CANARY_NO_ORDERS_NO_IO"
)
READY_FOR_BLOCK_L_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.2"
NEXT_STEP_TITLE: Final[str] = "RUNTIME ACTIVATION GATE MATRIX"
STATUS: Final[str] = "ready_for_functional_preview_14_2_runtime_activation_gate_matrix"

_CONTRACT_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_runtime_activation_contract_kind",
    "block",
    "step",
    "block_l_runtime_activation_contract_status",
    "block_l_runtime_activation_contract_decision",
    "ready_for_block_l_1",
    "next_step",
    "next_step_title",
]

_GATE_FALSE_FLAGS: Final[list[str]] = [
    "runtime_activation_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
]

_MODE_FALSE_FLAGS: Final[list[str]] = [
    "allowed_in_14_1",
    "runtime_activation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "live_trading_allowed_now",
]


def build_preview_block_l_runtime_activation_read_model() -> dict[str, Any]:
    """Build the Block L 14.1 source-only runtime activation read model."""
    contract = build_preview_block_l_runtime_activation_contract()
    gate_rows = _build_gate_rows(contract)
    mode_rows = _build_mode_rows(contract)
    blocked_capabilities = list(contract["blocked_runtime_activation_contract_capabilities"])

    return {
        "schema_version": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_SCHEMA_VERSION,
        "block_l_runtime_activation_read_model_kind": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_runtime_activation_read_model_status": BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_STATUS,
        "block_l_runtime_activation_read_model_decision": BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_DECISION,
        "ready_for_block_l_2": READY_FOR_BLOCK_L_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _build_contract_reference(contract),
        "runtime_activation_readiness": _build_readiness(),
        "runtime_activation_gate_rows": gate_rows,
        "runtime_activation_mode_rows": mode_rows,
        "blocked_capability_read_model": _build_blocked_capability_read_model(blocked_capabilities),
        "non_activation_evidence_read_model": _build_non_activation_evidence_read_model(contract),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(contract),
        "future_steps": [
            "functional_preview_14_2_runtime_activation_gate_matrix",
            "functional_preview_14_3_paper_runtime_activation_gate",
            "functional_preview_14_4_testnet_runtime_activation_gate",
            "functional_preview_14_5_live_canary_gate_contract",
            "functional_preview_14_6_block_l_closure_audit",
        ],
        "status": STATUS,
    }


def _build_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    reference = {key: contract[key] for key in _CONTRACT_REFERENCE_KEYS}
    reference["source_contract_step"] = "FUNCTIONAL-PREVIEW-14.0"
    reference["source_contract_read_by_14_1_read_model"] = True
    return reference


def _build_readiness() -> dict[str, bool]:
    return {
        "contract_available": True,
        "read_model_built": True,
        "ready_for_14_2_gate_matrix": True,
        "safe_to_activate_runtime_now": False,
        "safe_to_execute_gates_now": False,
        "safe_to_mutate_gate_state_now": False,
        "safe_for_paper_runtime_now": False,
        "safe_for_testnet_runtime_now": False,
        "safe_for_live_canary_now": False,
        "safe_for_orders_now": False,
        "safe_for_private_endpoints_now": False,
        "safe_for_network_io_now": False,
        "safe_for_filesystem_io_now": False,
    }


def _build_gate_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in contract["runtime_activation_candidate_gates"]:
        row = {
            "read_only": True,
            "runtime_activation_gate_id": gate["runtime_activation_gate_id"],
            "source_gate_id": gate["source_gate_id"],
            "display_name": gate["display_name"],
            "gate_domain": gate["gate_domain"],
            "gate_type": gate["gate_type"],
            "planned_source_step": gate["planned_source_step"],
            "required_for_future_activation": gate["required_for_future_activation"],
            "eligible_for_future_gate_matrix": gate["eligible_for_future_gate_matrix"],
            "safe_for_offline_tests": gate["safe_for_offline_tests"],
        }
        for flag in _GATE_FALSE_FLAGS:
            row[flag] = False
        row["gate_executed_by_read_model"] = False
        rows.append(row)
    return rows


def _build_mode_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in contract["runtime_activation_candidate_modes"]:
        row = {
            "read_only": True,
            "runtime_activation_mode_id": mode["runtime_activation_mode_id"],
            "source_mode_id": mode["source_mode_id"],
            "display_name": mode["display_name"],
            "mode_classification": mode["mode_classification"],
            "activation_stage": mode["activation_stage"],
            "requires_future_gate": mode["requires_future_gate"],
            "safe_for_offline_tests": mode["safe_for_offline_tests"],
        }
        for flag in _MODE_FALSE_FLAGS:
            row[flag] = False
        row["mode_activated_by_read_model"] = False
        rows.append(row)
    return rows


def _build_blocked_capability_read_model(capabilities: list[str]) -> dict[str, Any]:
    return {
        "blocked_capability_count": len(capabilities),
        "blocked_capabilities": capabilities,
        "runtime_activation_blocked": True,
        "gate_execution_blocked": True,
        "order_flow_blocked": True,
        "private_endpoint_access_blocked": True,
        "network_io_blocked": True,
        "filesystem_io_blocked": True,
        "live_canary_blocked": True,
    }


def _build_non_activation_evidence_read_model(contract: dict[str, Any]) -> dict[str, bool]:
    evidence = dict(contract["non_activation_evidence"])
    evidence["source_contract_read"] = True
    evidence["read_model_built"] = True
    evidence["runtime_activation_started"] = False
    evidence["runtime_gate_executed"] = False
    evidence["gate_state_mutated"] = False
    evidence["order_generated"] = False
    evidence["order_submitted"] = False
    evidence["private_endpoint_accessed"] = False
    evidence["network_io_performed"] = False
    evidence["filesystem_io_performed"] = False
    evidence["live_canary_started"] = False
    return evidence


def _build_read_model_boundaries() -> dict[str, bool]:
    return {
        "read_model_is_plain_data_only": True,
        "read_model_is_source_only": True,
        "read_model_can_feed_14_2_gate_matrix": True,
        "read_model_cannot_activate_runtime": True,
        "read_model_cannot_execute_gates": True,
        "read_model_cannot_mutate_gate_state": True,
        "read_model_cannot_start_paper_runtime": True,
        "read_model_cannot_start_testnet_runtime": True,
        "read_model_cannot_start_live_canary": True,
        "read_model_cannot_prepare_trade_orders": True,
        "read_model_cannot_access_private_endpoints": True,
        "read_model_cannot_perform_network_io": True,
        "read_model_cannot_perform_filesystem_io": True,
        "read_model_cannot_read_config_env_or_secrets": True,
        "read_model_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_contract": "FUNCTIONAL-PREVIEW-14.0",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_contract_boundaries": list(contract["source_boundaries"]),
    }
