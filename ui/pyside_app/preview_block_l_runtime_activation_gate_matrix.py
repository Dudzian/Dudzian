"""FUNCTIONAL-PREVIEW-14.2 Block L runtime activation gate matrix.

Static matrix derived from the Block L 14.1 read model. This module returns
plain data only and keeps every activation path blocked for this step.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_runtime_activation_read_model import (
    build_preview_block_l_runtime_activation_read_model,
)

PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_runtime_activation_gate_matrix.v1"
)
PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_KIND: Final[str] = (
    "functional_preview_block_l_runtime_activation_gate_matrix"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.2"
BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_STATUS: Final[str] = (
    "block_l_runtime_activation_gate_matrix_ready_no_gate_execution_no_runtime_activation_no_live_canary_no_orders_no_io"
)
BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_DECISION: Final[str] = (
    "GATE_MATRIX_READY_FOR_14_3_NO_GATE_EXECUTION_NO_RUNTIME_ACTIVATION_NO_LIVE_CANARY_NO_ORDERS_NO_IO"
)
READY_FOR_BLOCK_L_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.3"
NEXT_STEP_TITLE: Final[str] = "PAPER RUNTIME ACTIVATION GATE"
STATUS: Final[str] = "ready_for_functional_preview_14_3_paper_runtime_activation_gate"

_READ_MODEL_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_runtime_activation_read_model_kind",
    "block",
    "step",
    "block_l_runtime_activation_read_model_status",
    "block_l_runtime_activation_read_model_decision",
    "ready_for_block_l_2",
    "next_step",
    "next_step_title",
]


def build_preview_block_l_runtime_activation_gate_matrix() -> dict[str, Any]:
    """Build the Block L 14.2 static runtime activation gate matrix."""
    read_model = build_preview_block_l_runtime_activation_read_model()
    gate_rows = _build_gate_matrix_rows(read_model)
    mode_rows = _build_mode_matrix_rows(read_model)

    return {
        "schema_version": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_SCHEMA_VERSION,
        "block_l_runtime_activation_gate_matrix_kind": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_runtime_activation_gate_matrix_status": BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_STATUS,
        "block_l_runtime_activation_gate_matrix_decision": BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_DECISION,
        "ready_for_block_l_3": READY_FOR_BLOCK_L_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "read_model_reference": _build_read_model_reference(read_model),
        "runtime_activation_gate_matrix_readiness": _build_readiness(),
        "runtime_activation_gate_matrix_rows": gate_rows,
        "runtime_activation_mode_matrix_rows": mode_rows,
        "runtime_activation_gate_to_mode_matrix": _build_gate_to_mode_matrix(gate_rows, mode_rows),
        "blocked_capability_gate_matrix": _build_blocked_capability_gate_matrix(read_model),
        "fail_closed_decision_matrix": _build_fail_closed_decision_matrix(),
        "non_execution_evidence": _build_non_execution_evidence(read_model),
        "gate_matrix_boundaries": _build_gate_matrix_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": [
            "functional_preview_14_3_paper_runtime_activation_gate",
            "functional_preview_14_4_testnet_runtime_activation_gate",
            "functional_preview_14_5_live_canary_gate_contract",
            "functional_preview_14_6_block_l_closure_audit",
        ],
        "status": STATUS,
    }


def _build_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _READ_MODEL_REFERENCE_KEYS}
    reference["source_read_model_step"] = "FUNCTIONAL-PREVIEW-14.1"
    reference["source_read_model_read_by_14_2_gate_matrix"] = True
    reference["static_read_model_only"] = True
    reference["gate_execution_performed_by_14_2"] = False
    return reference


def _build_readiness() -> dict[str, bool]:
    return {
        "read_model_available": True,
        "gate_matrix_built": True,
        "ready_for_14_3_paper_runtime_activation_gate": True,
        "safe_to_execute_gates_now": False,
        "safe_to_mutate_gate_state_now": False,
        "safe_to_activate_runtime_now": False,
        "safe_for_paper_runtime_now": False,
        "safe_for_testnet_runtime_now": False,
        "safe_for_live_canary_now": False,
        "safe_for_orders_now": False,
        "safe_for_private_endpoints_now": False,
        "safe_for_network_io_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_credentials_now": False,
        "safe_for_live_trading_now": False,
    }


def _build_gate_matrix_rows(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in read_model["runtime_activation_gate_rows"]:
        rows.append(
            {
                "read_only": True,
                "matrix_row_type": "runtime_activation_gate_static_matrix_row",
                "matrix_included": True,
                "matrix_evaluated_by_14_2": True,
                "runtime_activation_gate_id": gate["runtime_activation_gate_id"],
                "source_gate_id": gate["source_gate_id"],
                "display_name": gate["display_name"],
                "gate_domain": gate["gate_domain"],
                "gate_type": gate["gate_type"],
                "planned_source_step": gate["planned_source_step"],
                "required_for_future_activation": gate["required_for_future_activation"],
                "eligible_for_future_gate_matrix": gate["eligible_for_future_gate_matrix"],
                "safe_for_offline_tests": gate["safe_for_offline_tests"],
                "gate_execution_result": "not_executed_static_matrix_only",
                "gate_execution_allowed_now": False,
                "gate_state_mutation_allowed_now": False,
                "runtime_activation_allowed_now": False,
                "order_flow_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "network_io_allowed_now": False,
                "filesystem_io_allowed_now": False,
            }
        )
    return rows


def _build_mode_matrix_rows(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in read_model["runtime_activation_mode_rows"]:
        rows.append(
            {
                "read_only": True,
                "matrix_row_type": "runtime_activation_mode_static_matrix_row",
                "matrix_included": True,
                "matrix_evaluated_by_14_2": True,
                "runtime_activation_mode_id": mode["runtime_activation_mode_id"],
                "source_mode_id": mode["source_mode_id"],
                "display_name": mode["display_name"],
                "mode_classification": mode["mode_classification"],
                "activation_stage": mode["activation_stage"],
                "requires_future_gate": mode["requires_future_gate"],
                "safe_for_offline_tests": mode["safe_for_offline_tests"],
                "activation_result": "not_activated_static_matrix_only",
                "activation_allowed_now": False,
                "runtime_activation_allowed_now": False,
                "order_flow_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "network_io_allowed_now": False,
                "credential_read_allowed_now": False,
                "live_trading_allowed_now": False,
            }
        )
    return rows


def _build_gate_to_mode_matrix(
    gate_rows: list[dict[str, Any]], mode_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gate_ids = [gate["source_gate_id"] for gate in gate_rows]
    mode_ids = [mode["source_mode_id"] for mode in mode_rows]
    return {
        "gate_ids": gate_ids,
        "mode_ids": mode_ids,
        "all_modes_require_future_gate": True,
        "all_gates_not_executed_now": True,
        "all_modes_not_activated_now": True,
        "paper_mode_requires_future_gate": True,
        "testnet_mode_requires_future_gate": True,
        "live_canary_mode_requires_future_gate": True,
        "live_scaled_mode_requires_future_gate": True,
        "gate_domains_by_id": {gate["source_gate_id"]: gate["gate_domain"] for gate in gate_rows},
        "mode_classifications_by_id": {
            mode["source_mode_id"]: mode["mode_classification"] for mode in mode_rows
        },
        "mode_activation_stages_by_id": {
            mode["source_mode_id"]: mode["activation_stage"] for mode in mode_rows
        },
        "future_required_gate_ids_by_mode_id": {mode_id: gate_ids for mode_id in mode_ids},
    }


def _build_blocked_capability_gate_matrix(read_model: dict[str, Any]) -> dict[str, Any]:
    blocked = read_model["blocked_capability_read_model"]
    capabilities = blocked["blocked_capabilities"]
    return {
        "blocked_capability_count": blocked["blocked_capability_count"],
        "blocked_capabilities": capabilities,
        "runtime_activation_blocked": True,
        "gate_execution_blocked": True,
        "order_flow_blocked": True,
        "private_endpoint_access_blocked": True,
        "network_io_blocked": True,
        "filesystem_io_blocked": True,
        "live_canary_blocked": True,
        "credential_read_blocked": True,
        "live_trading_blocked": True,
    }


def _build_fail_closed_decision_matrix() -> dict[str, str]:
    return {
        "missing_gate_policy": "fail_closed",
        "missing_read_model_policy": "fail_closed",
        "missing_contract_policy": "fail_closed",
        "runtime_activation_without_future_gate": "blocked",
        "gate_execution_in_14_2": "blocked",
        "runtime_activation_in_14_2": "blocked",
        "paper_runtime_in_14_2": "blocked",
        "testnet_runtime_in_14_2": "blocked",
        "live_canary_in_14_2": "blocked",
        "live_trading_in_14_2": "blocked",
        "order_flow_in_14_2": "blocked",
        "private_endpoint_in_14_2": "blocked",
        "network_io_in_14_2": "blocked",
        "filesystem_io_in_14_2": "blocked",
        "credential_read_in_14_2": "blocked",
    }


def _build_non_execution_evidence(read_model: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        key: value for key, value in read_model["non_activation_evidence_read_model"].items()
    }
    evidence["source_read_model_read"] = True
    evidence["gate_matrix_built"] = True
    evidence["runtime_activation_started"] = False
    evidence["runtime_gate_executed"] = False
    evidence["gate_state_mutated"] = False
    evidence["mode_activated"] = False
    evidence["order_generated"] = False
    evidence["order_submitted"] = False
    evidence["private_endpoint_accessed"] = False
    evidence["network_io_performed"] = False
    evidence["filesystem_io_performed"] = False
    evidence["live_canary_started"] = False
    return evidence


def _build_gate_matrix_boundaries() -> dict[str, bool]:
    return {
        "gate_matrix_is_plain_data_only": True,
        "gate_matrix_is_source_only": True,
        "gate_matrix_can_feed_14_3_paper_runtime_activation_gate": True,
        "gate_matrix_cannot_execute_gates": True,
        "gate_matrix_cannot_mutate_gate_state": True,
        "gate_matrix_cannot_activate_runtime": True,
        "gate_matrix_cannot_start_paper_runtime": True,
        "gate_matrix_cannot_start_testnet_runtime": True,
        "gate_matrix_cannot_start_live_canary": True,
        "gate_matrix_cannot_prepare_trade_orders": True,
        "gate_matrix_cannot_access_private_endpoints": True,
        "gate_matrix_cannot_perform_network_io": True,
        "gate_matrix_cannot_perform_filesystem_io": True,
        "gate_matrix_cannot_read_config_env_or_secrets": True,
        "gate_matrix_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_read_model": "FUNCTIONAL-PREVIEW-14.1",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_read_model_boundaries": read_model["source_boundaries"],
    }
