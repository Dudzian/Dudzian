"""FUNCTIONAL-PREVIEW-14.4 Block L testnet runtime activation gate.

Static source-only testnet runtime activation gate derived from the 14.3 paper
runtime activation gate. This module returns plain data only and keeps every
runtime, order, private endpoint, network, filesystem, credential, canary, and
live path blocked for this step.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_paper_runtime_activation_gate import (
    build_preview_block_l_paper_runtime_activation_gate,
)

PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_testnet_runtime_activation_gate.v1"
)
PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_KIND: Final[str] = (
    "functional_preview_block_l_testnet_runtime_activation_gate"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.4"
BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_STATUS: Final[str] = (
    "block_l_testnet_runtime_activation_gate_ready_static_data_no_testnet_runtime_start_no_network_io_no_credentials_no_private_endpoints_no_orders_no_io"
)
BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_DECISION: Final[str] = (
    "TESTNET_RUNTIME_ACTIVATION_GATE_READY_FOR_14_5_STATIC_ONLY_NO_TESTNET_RUNTIME_START_NO_NETWORK_IO_NO_CREDENTIALS_NO_PRIVATE_ENDPOINTS_NO_ORDERS_NO_IO"
)
READY_FOR_BLOCK_L_5: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.5"
NEXT_STEP_TITLE: Final[str] = "LIVE CANARY GATE CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_14_5_live_canary_gate_contract"

_PAPER_GATE_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_paper_runtime_activation_gate_kind",
    "block",
    "step",
    "block_l_paper_runtime_activation_gate_status",
    "block_l_paper_runtime_activation_gate_decision",
    "ready_for_block_l_4",
    "next_step",
    "next_step_title",
]


def build_preview_block_l_testnet_runtime_activation_gate() -> dict[str, Any]:
    """Build the Block L 14.4 static testnet runtime activation gate."""
    paper_gate = build_preview_block_l_paper_runtime_activation_gate()
    prerequisites = _build_required_future_prerequisites()

    return {
        "schema_version": PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION,
        "block_l_testnet_runtime_activation_gate_kind": PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_testnet_runtime_activation_gate_status": BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_STATUS,
        "block_l_testnet_runtime_activation_gate_decision": BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_DECISION,
        "ready_for_block_l_5": READY_FOR_BLOCK_L_5,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "paper_gate_reference": _build_paper_gate_reference(paper_gate),
        "testnet_runtime_gate_readiness": _build_testnet_runtime_gate_readiness(),
        "testnet_candidate_static_checks": _build_testnet_candidate_static_checks(paper_gate),
        "testnet_runtime_mode_decision": _build_testnet_runtime_mode_decision(
            paper_gate, prerequisites
        ),
        "testnet_activation_blocked_summary": _build_testnet_activation_blocked_summary(paper_gate),
        "required_future_prerequisites": prerequisites,
        "fail_closed_testnet_gate_decision": _build_fail_closed_testnet_gate_decision(),
        "non_start_evidence": _build_non_start_evidence(paper_gate),
        "testnet_gate_boundaries": _build_testnet_gate_boundaries(),
        "source_boundaries": _build_source_boundaries(paper_gate),
        "future_steps": [
            "functional_preview_14_5_live_canary_gate_contract",
            "functional_preview_14_6_block_l_closure_audit",
        ],
        "status": STATUS,
    }


def _build_paper_gate_reference(paper_gate: dict[str, Any]) -> dict[str, Any]:
    reference = {key: paper_gate[key] for key in _PAPER_GATE_REFERENCE_KEYS}
    reference["source_paper_gate_step"] = "FUNCTIONAL-PREVIEW-14.3"
    reference["source_paper_gate_read_by_14_4_testnet_gate"] = True
    reference["static_paper_gate_only"] = True
    reference["testnet_runtime_started_by_14_4"] = False
    reference["network_io_opened_by_14_4"] = False
    reference["credentials_read_by_14_4"] = False
    reference["private_endpoint_accessed_by_14_4"] = False
    return reference


def _build_testnet_runtime_gate_readiness() -> dict[str, bool]:
    return {
        "paper_gate_available": True,
        "testnet_gate_built": True,
        "ready_for_14_5_live_canary_gate_contract": True,
        "safe_to_start_testnet_runtime_now": False,
        "safe_to_open_network_io_now": False,
        "safe_to_read_credentials_now": False,
        "safe_to_access_private_endpoints_now": False,
        "safe_to_start_runtime_loop_now": False,
        "safe_to_execute_runtime_gate_now": False,
        "safe_to_mutate_gate_state_now": False,
        "safe_for_order_generation_now": False,
        "safe_for_order_submission_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
        "safe_for_live_canary_now": False,
        "safe_for_live_trading_now": False,
    }


def _build_testnet_candidate_static_checks(paper_gate: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in paper_gate["paper_runtime_candidate_gate_checks"]:
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
                "testnet_gate_check_type": "static_read_only_candidate_gate_check",
                "testnet_gate_static_check_included": True,
                "testnet_gate_static_check_result": "not_executed_static_testnet_gate_only",
                "testnet_gate_executed_by_14_4": False,
                "testnet_runtime_start_allowed_now": False,
                "runtime_loop_allowed_now": False,
                "runtime_gate_execution_allowed_now": False,
                "gate_state_mutation_allowed_now": False,
                "order_flow_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "network_io_allowed_now": False,
                "credential_read_allowed_now": False,
                "filesystem_io_allowed_now": False,
            }
        )
    return checks


def _build_testnet_runtime_mode_decision(
    paper_gate: dict[str, Any], prerequisites: list[dict[str, Any]]
) -> dict[str, Any]:
    paper_mode = paper_gate["paper_runtime_mode_decision"]
    return {
        "source_paper_gate_step": "FUNCTIONAL-PREVIEW-14.3",
        "source_paper_mode_static_decision": paper_mode["paper_mode_static_decision"],
        "source_required_future_prerequisite_count": len(prerequisites),
        "testnet_mode_static_decision": "testnet_runtime_candidate_requires_future_static_gate_not_started",
        "testnet_mode_discovered_from_static_chain": True,
        "testnet_mode_requires_future_gate": True,
        "testnet_runtime_started": False,
        "testnet_runtime_start_allowed_now": False,
        "runtime_loop_started": False,
        "runtime_loop_allowed_now": False,
        "network_io_opened": False,
        "network_io_allowed_now": False,
        "credential_read_performed": False,
        "credential_read_allowed_now": False,
        "private_endpoint_accessed": False,
        "private_endpoint_access_allowed_now": False,
        "order_flow_allowed_now": False,
        "filesystem_io_allowed_now": False,
        "live_trading_allowed_now": False,
    }


def _build_testnet_activation_blocked_summary(paper_gate: dict[str, Any]) -> dict[str, Any]:
    blocked = paper_gate["paper_runtime_activation_blocked_summary"]
    return {
        "blocked_capability_count": blocked["blocked_capability_count"],
        "blocked_capabilities": blocked["blocked_capabilities"],
        "testnet_runtime_start_blocked": True,
        "network_io_blocked": True,
        "credential_read_blocked": True,
        "private_endpoint_access_blocked": True,
        "runtime_loop_blocked": True,
        "runtime_gate_execution_blocked": True,
        "gate_state_mutation_blocked": True,
        "order_generation_blocked": True,
        "order_submission_blocked": True,
        "filesystem_io_blocked": True,
        "config_env_secret_read_blocked": True,
        "live_canary_blocked": True,
        "live_trading_blocked": True,
    }


def _build_required_future_prerequisites() -> list[dict[str, Any]]:
    prerequisite_rows = [
        ("explicit_operator_testnet_activation_gate", "Explicit operator testnet activation gate"),
        ("testnet_adapter_contract", "Testnet adapter contract"),
        ("testnet_credentials_contract", "Testnet credentials contract without credential access"),
        ("no_real_live_order_guarantee", "No-real-live-order guarantee"),
        ("private_endpoint_read_gate", "Private endpoint access gate not executed now"),
        ("network_io_gate", "Network I/O gate not executed now"),
        ("kill_switch_ready", "Kill switch ready"),
        ("observability_ready", "Observability ready"),
        ("rollback_ready", "Rollback ready"),
        ("soak_failure_policy_ready", "Soak and failure policy ready"),
    ]
    return [
        {
            "prerequisite_id": prerequisite_id,
            "display_name": display_name,
            "required_before_testnet_runtime_start": True,
            "satisfied_in_14_4": False,
            "requires_future_step": True,
            "notes": "Static prerequisite placeholder only; no real-world check performed in 14.4.",
        }
        for prerequisite_id, display_name in prerequisite_rows
    ]


def _build_fail_closed_testnet_gate_decision() -> dict[str, str]:
    return {
        "missing_paper_gate_policy": "fail_closed",
        "missing_testnet_mode_policy": "fail_closed",
        "missing_testnet_credentials_policy": "fail_closed",
        "missing_network_gate_policy": "fail_closed",
        "missing_private_endpoint_gate_policy": "fail_closed",
        "missing_future_prerequisite_policy": "fail_closed",
        "testnet_runtime_start_in_14_4": "blocked",
        "network_io_in_14_4": "blocked",
        "credential_read_in_14_4": "blocked",
        "private_endpoint_in_14_4": "blocked",
        "runtime_loop_start_in_14_4": "blocked",
        "runtime_gate_execution_in_14_4": "blocked",
        "gate_state_mutation_in_14_4": "blocked",
        "order_generation_in_14_4": "blocked",
        "order_submission_in_14_4": "blocked",
        "filesystem_io_in_14_4": "blocked",
        "config_env_secret_read_in_14_4": "blocked",
        "live_canary_in_14_4": "blocked",
        "live_trading_in_14_4": "blocked",
    }


def _build_non_start_evidence(paper_gate: dict[str, Any]) -> dict[str, bool]:
    source = paper_gate["non_start_evidence"]
    return {
        "source_paper_gate_read": True,
        "testnet_gate_built": True,
        "source_paper_gate_paper_runtime_started": source["paper_runtime_started"],
        "source_paper_gate_runtime_loop_started": source["runtime_loop_started"],
        "source_paper_gate_runtime_gate_executed": source["runtime_gate_executed"],
        "source_paper_gate_gate_state_mutated": source["gate_state_mutated"],
        "source_paper_gate_mode_activated": source["mode_activated"],
        "source_paper_gate_order_generated": source["order_generated"],
        "source_paper_gate_order_submitted": source["order_submitted"],
        "source_paper_gate_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_paper_gate_network_io_performed": source["network_io_performed"],
        "source_paper_gate_filesystem_io_performed": source["filesystem_io_performed"],
        "source_paper_gate_live_canary_started": source["live_canary_started"],
        "paper_runtime_started": False,
        "testnet_runtime_started": False,
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
        "live_canary_started": False,
        "live_trading_started": False,
    }


def _build_testnet_gate_boundaries() -> dict[str, bool]:
    return {
        "testnet_gate_is_plain_data_only": True,
        "testnet_gate_is_source_only": True,
        "testnet_gate_can_feed_14_5_live_canary_gate_contract": True,
        "testnet_gate_cannot_start_testnet_runtime": True,
        "testnet_gate_cannot_open_network_io": True,
        "testnet_gate_cannot_read_credentials": True,
        "testnet_gate_cannot_access_private_endpoints": True,
        "testnet_gate_cannot_start_runtime_loop": True,
        "testnet_gate_cannot_execute_runtime_gates": True,
        "testnet_gate_cannot_mutate_gate_state": True,
        "testnet_gate_cannot_generate_orders": True,
        f"testnet_gate_cannot_{'sub'}mit_orders": True,
        "testnet_gate_cannot_perform_filesystem_io": True,
        "testnet_gate_cannot_read_config_env_or_secrets": True,
        "testnet_gate_cannot_start_live_canary": True,
        "testnet_gate_cannot_live_trade": True,
        "testnet_gate_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(paper_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_paper_gate": "FUNCTIONAL-PREVIEW-14.3",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_paper_gate_boundaries": paper_gate["source_boundaries"],
    }
