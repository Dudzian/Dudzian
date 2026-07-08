"""FUNCTIONAL-PREVIEW-14.3 Block L paper runtime activation gate.

Static source-only paper runtime activation gate derived from the 14.2 gate
matrix. This module returns plain data only and keeps every runtime, order,
private endpoint, network, filesystem, credential, testnet, and live path
blocked for this step.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_runtime_activation_gate_matrix import (
    build_preview_block_l_runtime_activation_gate_matrix,
)

PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_paper_runtime_activation_gate.v1"
)
PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_KIND: Final[str] = (
    "functional_preview_block_l_paper_runtime_activation_gate"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.3"
BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_STATUS: Final[str] = (
    "block_l_paper_runtime_activation_gate_ready_static_data_no_paper_runtime_start_no_runtime_loop_no_orders_no_private_endpoints_no_io"
)
BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_DECISION: Final[str] = (
    "PAPER_RUNTIME_ACTIVATION_GATE_READY_FOR_14_4_STATIC_ONLY_NO_PAPER_RUNTIME_START_NO_RUNTIME_LOOP_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_IO"
)
READY_FOR_BLOCK_L_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.4"
NEXT_STEP_TITLE: Final[str] = "TESTNET RUNTIME ACTIVATION GATE"
STATUS: Final[str] = "ready_for_functional_preview_14_4_testnet_runtime_activation_gate"

_GATE_MATRIX_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_runtime_activation_gate_matrix_kind",
    "block",
    "step",
    "block_l_runtime_activation_gate_matrix_status",
    "block_l_runtime_activation_gate_matrix_decision",
    "ready_for_block_l_3",
    "next_step",
    "next_step_title",
]


def build_preview_block_l_paper_runtime_activation_gate() -> dict[str, Any]:
    """Build the Block L 14.3 static paper runtime activation gate."""
    gate_matrix = build_preview_block_l_runtime_activation_gate_matrix()
    candidate_checks = _build_paper_runtime_candidate_gate_checks(gate_matrix)
    mode_decision = _build_paper_runtime_mode_decision(gate_matrix)

    return {
        "schema_version": PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION,
        "block_l_paper_runtime_activation_gate_kind": PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_paper_runtime_activation_gate_status": BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_STATUS,
        "block_l_paper_runtime_activation_gate_decision": BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_DECISION,
        "ready_for_block_l_4": READY_FOR_BLOCK_L_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "gate_matrix_reference": _build_gate_matrix_reference(gate_matrix),
        "paper_runtime_gate_readiness": _build_paper_runtime_gate_readiness(),
        "paper_runtime_candidate_gate_checks": candidate_checks,
        "paper_runtime_mode_decision": mode_decision,
        "paper_runtime_activation_blocked_summary": _build_blocked_summary(gate_matrix),
        "required_future_prerequisites": _build_required_future_prerequisites(),
        "fail_closed_paper_gate_decision": _build_fail_closed_paper_gate_decision(),
        "non_start_evidence": _build_non_start_evidence(gate_matrix),
        "paper_gate_boundaries": _build_paper_gate_boundaries(),
        "source_boundaries": _build_source_boundaries(gate_matrix),
        "future_steps": [
            "functional_preview_14_4_testnet_runtime_activation_gate",
            "functional_preview_14_5_live_canary_gate_contract",
            "functional_preview_14_6_block_l_closure_audit",
        ],
        "status": STATUS,
    }


def _build_gate_matrix_reference(gate_matrix: dict[str, Any]) -> dict[str, Any]:
    reference = {key: gate_matrix[key] for key in _GATE_MATRIX_REFERENCE_KEYS}
    reference["source_gate_matrix_step"] = "FUNCTIONAL-PREVIEW-14.2"
    reference["source_gate_matrix_read_by_14_3_paper_gate"] = True
    reference["static_gate_matrix_only"] = True
    reference["paper_runtime_started_by_14_3"] = False
    reference["runtime_loop_started_by_14_3"] = False
    return reference


def _build_paper_runtime_gate_readiness() -> dict[str, bool]:
    return {
        "gate_matrix_available": True,
        "paper_gate_built": True,
        "ready_for_14_4_testnet_runtime_activation_gate": True,
        "safe_to_start_paper_runtime_now": False,
        "safe_to_start_runtime_loop_now": False,
        "safe_to_execute_runtime_gate_now": False,
        "safe_to_mutate_gate_state_now": False,
        "safe_for_order_generation_now": False,
        "safe_for_order_submission_now": False,
        "safe_for_private_endpoints_now": False,
        "safe_for_network_io_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
        "safe_for_testnet_runtime_now": False,
        "safe_for_live_canary_now": False,
        "safe_for_live_trading_now": False,
    }


def _build_paper_runtime_candidate_gate_checks(gate_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in gate_matrix["runtime_activation_gate_matrix_rows"]:
        checks.append(
            {
                "runtime_activation_gate_id": row["runtime_activation_gate_id"],
                "source_gate_id": row["source_gate_id"],
                "display_name": row["display_name"],
                "gate_domain": row["gate_domain"],
                "gate_type": row["gate_type"],
                "planned_source_step": row["planned_source_step"],
                "required_for_future_activation": row["required_for_future_activation"],
                "safe_for_offline_tests": row["safe_for_offline_tests"],
                "paper_gate_check_type": "static_read_only_candidate_gate_check",
                "paper_gate_static_check_included": True,
                "paper_gate_static_check_result": "not_executed_static_paper_gate_only",
                "paper_gate_executed_by_14_3": False,
                "paper_runtime_start_allowed_now": False,
                "runtime_loop_allowed_now": False,
                "runtime_gate_execution_allowed_now": False,
                "gate_state_mutation_allowed_now": False,
                "order_flow_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "network_io_allowed_now": False,
                "filesystem_io_allowed_now": False,
            }
        )
    return checks


def _build_paper_runtime_mode_decision(gate_matrix: dict[str, Any]) -> dict[str, Any]:
    paper_mode = _find_paper_mode(gate_matrix["runtime_activation_mode_matrix_rows"])
    if paper_mode is None:
        return {
            "paper_mode_found": False,
            "paper_mode_static_decision": "paper_runtime_candidate_missing_fail_closed_not_started",
            "paper_runtime_started": False,
            "paper_runtime_start_allowed_now": False,
            "runtime_loop_started": False,
            "runtime_loop_allowed_now": False,
            "order_flow_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "credential_read_allowed_now": False,
            "filesystem_io_allowed_now": False,
            "live_trading_allowed_now": False,
        }
    return {
        "runtime_activation_mode_id": paper_mode["runtime_activation_mode_id"],
        "source_mode_id": paper_mode["source_mode_id"],
        "display_name": paper_mode["display_name"],
        "mode_classification": paper_mode["mode_classification"],
        "activation_stage": paper_mode["activation_stage"],
        "requires_future_gate": paper_mode["requires_future_gate"],
        "safe_for_offline_tests": paper_mode["safe_for_offline_tests"],
        "paper_mode_found": True,
        "paper_mode_static_decision": "paper_runtime_candidate_identified_but_not_started",
        "paper_runtime_started": False,
        "paper_runtime_start_allowed_now": False,
        "runtime_loop_started": False,
        "runtime_loop_allowed_now": False,
        "order_flow_allowed_now": False,
        "private_endpoint_access_allowed_now": False,
        "network_io_allowed_now": False,
        "credential_read_allowed_now": False,
        "filesystem_io_allowed_now": False,
        "live_trading_allowed_now": False,
    }


def _find_paper_mode(mode_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for mode in mode_rows:
        if mode["source_mode_id"] == "paper_runtime_candidate":
            return mode
        if mode["mode_classification"] == "paper_runtime":
            return mode
    return None


def _build_blocked_summary(gate_matrix: dict[str, Any]) -> dict[str, Any]:
    blocked = gate_matrix["blocked_capability_gate_matrix"]
    return {
        "blocked_capability_count": blocked["blocked_capability_count"],
        "blocked_capabilities": blocked["blocked_capabilities"],
        "paper_runtime_start_blocked": True,
        "runtime_loop_blocked": True,
        "runtime_gate_execution_blocked": True,
        "gate_state_mutation_blocked": True,
        "order_generation_blocked": True,
        "order_submission_blocked": True,
        "private_endpoint_access_blocked": True,
        "network_io_blocked": True,
        "filesystem_io_blocked": True,
        "credential_read_blocked": True,
        "testnet_runtime_blocked": True,
        "live_canary_blocked": True,
        "live_trading_blocked": True,
    }


def _build_required_future_prerequisites() -> list[dict[str, Any]]:
    prerequisite_rows = [
        ("explicit_operator_paper_activation_gate", "Explicit operator paper activation gate"),
        ("offline_fixture_paper_adapter_contract", "Offline fixture and paper adapter contract"),
        ("no_real_order_guarantee", "No-real-order guarantee"),
        ("no_private_endpoint_guarantee", "No-private-endpoint guarantee"),
        ("kill_switch_ready", "Kill switch ready"),
        ("observability_ready", "Observability ready"),
        ("rollback_ready", "Rollback ready"),
        ("soak_failure_policy_ready", "Soak and failure policy ready"),
    ]
    return [
        {
            "prerequisite_id": prerequisite_id,
            "display_name": display_name,
            "required_before_paper_runtime_start": True,
            "satisfied_in_14_3": False,
            "requires_future_step": True,
            "notes": "Static prerequisite placeholder only; no real-world check performed in 14.3.",
        }
        for prerequisite_id, display_name in prerequisite_rows
    ]


def _build_fail_closed_paper_gate_decision() -> dict[str, str]:
    return {
        "missing_gate_matrix_policy": "fail_closed",
        "missing_paper_mode_policy": "fail_closed",
        "missing_future_prerequisite_policy": "fail_closed",
        "paper_runtime_start_in_14_3": "blocked",
        "runtime_loop_start_in_14_3": "blocked",
        "runtime_gate_execution_in_14_3": "blocked",
        "gate_state_mutation_in_14_3": "blocked",
        "order_generation_in_14_3": "blocked",
        "order_submission_in_14_3": "blocked",
        "private_endpoint_in_14_3": "blocked",
        "network_io_in_14_3": "blocked",
        "filesystem_io_in_14_3": "blocked",
        "credential_read_in_14_3": "blocked",
        "testnet_runtime_in_14_3": "blocked",
        "live_canary_in_14_3": "blocked",
        "live_trading_in_14_3": "blocked",
    }


def _build_non_start_evidence(gate_matrix: dict[str, Any]) -> dict[str, bool]:
    source = gate_matrix["non_execution_evidence"]
    return {
        "source_gate_matrix_read": True,
        "paper_gate_built": True,
        "source_gate_matrix_runtime_activation_started": source["runtime_activation_started"],
        "source_gate_matrix_runtime_gate_executed": source["runtime_gate_executed"],
        "source_gate_matrix_gate_state_mutated": source["gate_state_mutated"],
        "source_gate_matrix_mode_activated": source["mode_activated"],
        "source_gate_matrix_order_generated": source["order_generated"],
        "source_gate_matrix_order_submitted": source["order_submitted"],
        "source_gate_matrix_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_gate_matrix_network_io_performed": source["network_io_performed"],
        "source_gate_matrix_filesystem_io_performed": source["filesystem_io_performed"],
        "source_gate_matrix_live_canary_started": source["live_canary_started"],
        "paper_runtime_started": False,
        "runtime_loop_started": False,
        "runtime_gate_executed": False,
        "gate_state_mutated": False,
        "mode_activated": False,
        "order_generated": False,
        "order_submitted": False,
        "private_endpoint_accessed": False,
        "network_io_performed": False,
        "filesystem_io_performed": False,
        "testnet_runtime_started": False,
        "live_canary_started": False,
        "live_trading_started": False,
    }


def _build_paper_gate_boundaries() -> dict[str, bool]:
    return {
        "paper_gate_is_plain_data_only": True,
        "paper_gate_is_source_only": True,
        "paper_gate_can_feed_14_4_testnet_runtime_activation_gate": True,
        "paper_gate_cannot_start_paper_runtime": True,
        "paper_gate_cannot_start_runtime_loop": True,
        "paper_gate_cannot_execute_runtime_gates": True,
        "paper_gate_cannot_mutate_gate_state": True,
        "paper_gate_cannot_generate_orders": True,
        "paper_gate_cannot_submit_orders": True,
        "paper_gate_cannot_access_private_endpoints": True,
        "paper_gate_cannot_perform_network_io": True,
        "paper_gate_cannot_perform_filesystem_io": True,
        "paper_gate_cannot_read_config_env_or_secrets": True,
        "paper_gate_cannot_start_testnet_runtime": True,
        "paper_gate_cannot_start_live_canary": True,
        "paper_gate_cannot_live_trade": True,
        "paper_gate_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(gate_matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_gate_matrix": "FUNCTIONAL-PREVIEW-14.2",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_gate_matrix_boundaries": gate_matrix["source_boundaries"],
    }
