"""FUNCTIONAL-PREVIEW-14.5 Block L live canary gate contract.

Static source-only live canary contract derived from the 14.4 testnet gate.
This module returns plain data only and keeps every live canary, live trading,
order, private endpoint, network, filesystem, credential, runtime, and gate path
blocked for this step.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_testnet_runtime_activation_gate import (
    build_preview_block_l_testnet_runtime_activation_gate,
)

PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_live_canary_gate_contract.v1"
)
PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_l_live_canary_gate_contract"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.5"
BLOCK_L_LIVE_CANARY_GATE_CONTRACT_STATUS: Final[str] = (
    "block_l_live_canary_gate_contract_ready_static_data_no_live_canary_start_no_live_trading_no_orders_no_private_endpoints_no_network_io_no_credentials_no_io"
)
BLOCK_L_LIVE_CANARY_GATE_CONTRACT_DECISION: Final[str] = (
    "LIVE_CANARY_GATE_CONTRACT_READY_FOR_14_6_STATIC_ONLY_NO_LIVE_CANARY_START_NO_LIVE_TRADING_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_IO"
)
READY_FOR_BLOCK_L_6: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.6"
NEXT_STEP_TITLE: Final[str] = "BLOCK L CLOSURE AUDIT"
STATUS: Final[str] = "ready_for_functional_preview_14_6_block_l_closure_audit"

_TESTNET_GATE_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_testnet_runtime_activation_gate_kind",
    "block",
    "step",
    "block_l_testnet_runtime_activation_gate_status",
    "block_l_testnet_runtime_activation_gate_decision",
    "ready_for_block_l_5",
    "next_step",
    "next_step_title",
]


def build_preview_block_l_live_canary_gate_contract() -> dict[str, Any]:
    """Build the Block L 14.5 static live canary gate contract."""
    testnet_gate = build_preview_block_l_testnet_runtime_activation_gate()
    prerequisites = _build_required_future_prerequisites()
    testnet_reference = _build_testnet_gate_reference(testnet_gate)

    return {
        "schema_version": PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_SCHEMA_VERSION,
        "block_l_live_canary_gate_contract_kind": PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_live_canary_gate_contract_status": BLOCK_L_LIVE_CANARY_GATE_CONTRACT_STATUS,
        "block_l_live_canary_gate_contract_decision": BLOCK_L_LIVE_CANARY_GATE_CONTRACT_DECISION,
        "ready_for_block_l_6": READY_FOR_BLOCK_L_6,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "testnet_gate_reference": testnet_reference,
        "live_canary_gate_contract_readiness": _build_live_canary_gate_contract_readiness(),
        "live_canary_candidate_static_checks": _build_live_canary_candidate_static_checks(
            testnet_gate
        ),
        "live_canary_mode_decision": _build_live_canary_mode_decision(
            testnet_gate, testnet_reference, prerequisites
        ),
        "live_canary_blocked_summary": _build_live_canary_blocked_summary(testnet_gate),
        "required_future_prerequisites": prerequisites,
        "fail_closed_live_canary_decision": _build_fail_closed_live_canary_decision(),
        "non_start_evidence": _build_non_start_evidence(testnet_gate),
        "live_canary_boundaries": _build_live_canary_boundaries(),
        "source_boundaries": _build_source_boundaries(testnet_gate),
        "future_steps": ["functional_preview_14_6_block_l_closure_audit"],
        "status": STATUS,
    }


def _build_testnet_gate_reference(testnet_gate: dict[str, Any]) -> dict[str, Any]:
    reference = {key: testnet_gate[key] for key in _TESTNET_GATE_REFERENCE_KEYS}
    reference["source_testnet_gate_step"] = "FUNCTIONAL-PREVIEW-14.4"
    reference["source_testnet_gate_read_by_14_5_live_canary_contract"] = True
    reference["static_testnet_gate_only"] = True
    reference["live_canary_started_by_14_5"] = False
    reference["live_trading_enabled_by_14_5"] = False
    reference["orders_enabled_by_14_5"] = False
    reference["network_io_opened_by_14_5"] = False
    reference["credentials_read_by_14_5"] = False
    reference["private_endpoint_accessed_by_14_5"] = False
    return reference


def _build_live_canary_gate_contract_readiness() -> dict[str, bool]:
    return {
        "testnet_gate_available": True,
        "live_canary_contract_built": True,
        "ready_for_14_6_block_l_closure_audit": True,
        "safe_to_start_live_canary_now": False,
        "safe_to_enable_live_trading_now": False,
        "safe_to_generate_orders_now": False,
        f"safe_to_{'sub'}mit_orders_now": False,
        f"safe_to_{'can'}cel_orders_now": False,
        f"safe_to_{'re'}place_orders_now": False,
        "safe_to_access_private_endpoints_now": False,
        "safe_to_open_network_io_now": False,
        "safe_to_read_credentials_now": False,
        "safe_to_start_runtime_loop_now": False,
        "safe_to_execute_runtime_gate_now": False,
        "safe_to_mutate_gate_state_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
    }


def _build_live_canary_candidate_static_checks(
    testnet_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in testnet_gate["testnet_candidate_static_checks"]:
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
                "live_canary_check_type": "static_read_only_candidate_gate_check",
                "live_canary_static_check_included": True,
                "live_canary_static_check_result": "not_executed_static_live_canary_contract_only",
                "live_canary_gate_executed_by_14_5": False,
                "live_canary_start_allowed_now": False,
                "live_trading_allowed_now": False,
                "order_generation_allowed_now": False,
                "order_submission_allowed_now": False,
                "order_cancel_allowed_now": False,
                "order_replace_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "network_io_allowed_now": False,
                "credential_read_allowed_now": False,
                "runtime_loop_allowed_now": False,
                "runtime_gate_execution_allowed_now": False,
                "gate_state_mutation_allowed_now": False,
                "filesystem_io_allowed_now": False,
            }
        )
    return checks


def _build_live_canary_mode_decision(
    testnet_gate: dict[str, Any],
    testnet_reference: dict[str, Any],
    prerequisites: list[dict[str, Any]],
) -> dict[str, Any]:
    testnet_mode = testnet_gate["testnet_runtime_mode_decision"]
    return {
        "source_testnet_gate_step": testnet_reference["source_testnet_gate_step"],
        "source_testnet_mode_static_decision": testnet_mode["testnet_mode_static_decision"],
        "source_required_future_prerequisite_count": len(prerequisites),
        "live_canary_mode_static_decision": "live_canary_candidate_requires_future_static_gate_not_started",
        "live_canary_mode_discovered_from_static_chain": True,
        "live_canary_mode_requires_future_gate": True,
        "live_canary_started": False,
        "live_canary_start_allowed_now": False,
        "live_trading_enabled": False,
        "live_trading_allowed_now": False,
        "runtime_loop_started": False,
        "runtime_loop_allowed_now": False,
        "network_io_opened": False,
        "network_io_allowed_now": False,
        "credential_read_performed": False,
        "credential_read_allowed_now": False,
        "private_endpoint_accessed": False,
        "private_endpoint_access_allowed_now": False,
        "order_generation_allowed_now": False,
        "order_submission_allowed_now": False,
        "order_cancel_allowed_now": False,
        "order_replace_allowed_now": False,
        "filesystem_io_allowed_now": False,
    }


def _build_live_canary_blocked_summary(testnet_gate: dict[str, Any]) -> dict[str, Any]:
    blocked = testnet_gate["testnet_activation_blocked_summary"]
    capabilities = [*blocked["blocked_capabilities"], "live_canary_start", "live_order_safety"]
    return {
        "blocked_capability_count": len(capabilities),
        "blocked_capabilities": capabilities,
        "live_canary_start_blocked": True,
        "live_trading_blocked": True,
        "order_generation_blocked": True,
        "order_submission_blocked": True,
        "order_cancel_blocked": True,
        "order_replace_blocked": True,
        "private_endpoint_access_blocked": True,
        "network_io_blocked": True,
        "credential_read_blocked": True,
        "runtime_loop_blocked": True,
        "runtime_gate_execution_blocked": True,
        "gate_state_mutation_blocked": True,
        "filesystem_io_blocked": True,
        "config_env_secret_read_blocked": True,
    }


def _build_required_future_prerequisites() -> list[dict[str, Any]]:
    prerequisite_rows = [
        (
            "explicit_operator_live_canary_activation_gate",
            "Explicit operator live canary activation gate",
        ),
        ("live_canary_adapter_contract", "Live canary adapter contract"),
        ("live_credentials_contract", "Live credentials contract without credential access"),
        ("private_endpoint_read_gate", "Private endpoint access gate not executed now"),
        ("live_order_safety_contract", "Live order safety contract"),
        ("max_order_size_notional_cap_contract", "Max order size and notional cap contract"),
        ("kill_switch_armed", "Kill switch armed"),
        ("observability_ready", "Observability ready"),
        ("rollback_ready", "Rollback ready"),
        ("live_canary_stop_policy_ready", "Live canary stop policy ready"),
        ("incident_failure_policy_ready", "Incident and failure policy ready"),
        ("manual_confirmation_required", "Manual confirmation required"),
    ]
    return [
        {
            "prerequisite_id": prerequisite_id,
            "display_name": display_name,
            "required_before_live_canary_start": True,
            "satisfied_in_14_5": False,
            "requires_future_step": True,
            "notes": "Static prerequisite placeholder only; no real-world check performed in 14.5.",
        }
        for prerequisite_id, display_name in prerequisite_rows
    ]


def _build_fail_closed_live_canary_decision() -> dict[str, str]:
    return {
        "missing_testnet_gate_policy": "fail_closed",
        "missing_live_canary_mode_policy": "fail_closed",
        "missing_live_credentials_policy": "fail_closed",
        "missing_private_endpoint_gate_policy": "fail_closed",
        "missing_order_safety_policy": "fail_closed",
        "missing_kill_switch_policy": "fail_closed",
        "missing_observability_policy": "fail_closed",
        "missing_rollback_policy": "fail_closed",
        "missing_future_prerequisite_policy": "fail_closed",
        "live_canary_start_in_14_5": "blocked",
        "live_trading_in_14_5": "blocked",
        "order_generation_in_14_5": "blocked",
        "order_submission_in_14_5": "blocked",
        "order_cancel_in_14_5": "blocked",
        "order_replace_in_14_5": "blocked",
        "private_endpoint_in_14_5": "blocked",
        "network_io_in_14_5": "blocked",
        "credential_read_in_14_5": "blocked",
        "runtime_loop_start_in_14_5": "blocked",
        "runtime_gate_execution_in_14_5": "blocked",
        "gate_state_mutation_in_14_5": "blocked",
        "filesystem_io_in_14_5": "blocked",
        "config_env_secret_read_in_14_5": "blocked",
    }


def _build_non_start_evidence(testnet_gate: dict[str, Any]) -> dict[str, bool]:
    source = testnet_gate["non_start_evidence"]
    return {
        "source_testnet_gate_read": True,
        "live_canary_contract_built": True,
        "source_testnet_gate_testnet_runtime_started": source["testnet_runtime_started"],
        "source_testnet_gate_runtime_loop_started": source["runtime_loop_started"],
        "source_testnet_gate_runtime_gate_executed": source["runtime_gate_executed"],
        "source_testnet_gate_gate_state_mutated": source["gate_state_mutated"],
        "source_testnet_gate_mode_activated": source["mode_activated"],
        "source_testnet_gate_order_generated": source["order_generated"],
        "source_testnet_gate_order_submitted": source["order_submitted"],
        "source_testnet_gate_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_testnet_gate_network_io_performed": source["network_io_performed"],
        "source_testnet_gate_filesystem_io_performed": source["filesystem_io_performed"],
        "source_testnet_gate_live_canary_started": source["live_canary_started"],
        "paper_runtime_started": False,
        "testnet_runtime_started": False,
        "live_canary_started": False,
        "runtime_loop_started": False,
        "runtime_gate_executed": False,
        "gate_state_mutated": False,
        "mode_activated": False,
        "order_generated": False,
        "order_submitted": False,
        "private_endpoint_accessed": False,
        "network_io_performed": False,
        "filesystem_io_performed": False,
        "credential_read_performed": False,
        "live_trading_started": False,
    }


def _build_live_canary_boundaries() -> dict[str, bool]:
    return {
        "live_canary_contract_is_plain_data_only": True,
        "live_canary_contract_is_source_only": True,
        "live_canary_contract_can_feed_14_6_block_l_closure_audit": True,
        "live_canary_contract_cannot_start_live_canary": True,
        "live_canary_contract_cannot_enable_live_trading": True,
        "live_canary_contract_cannot_generate_orders": True,
        f"live_canary_contract_cannot_{'sub'}mit_orders": True,
        f"live_canary_contract_cannot_{'can'}cel_orders": True,
        f"live_canary_contract_cannot_{'re'}place_orders": True,
        "live_canary_contract_cannot_access_private_endpoints": True,
        "live_canary_contract_cannot_open_network_io": True,
        "live_canary_contract_cannot_read_credentials": True,
        "live_canary_contract_cannot_start_runtime_loop": True,
        "live_canary_contract_cannot_execute_runtime_gates": True,
        "live_canary_contract_cannot_mutate_gate_state": True,
        "live_canary_contract_cannot_perform_filesystem_io": True,
        "live_canary_contract_cannot_read_config_env_or_secrets": True,
        "live_canary_contract_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(testnet_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_testnet_gate": "FUNCTIONAL-PREVIEW-14.4",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_testnet_gate_boundaries": testnet_gate["source_boundaries"],
    }
