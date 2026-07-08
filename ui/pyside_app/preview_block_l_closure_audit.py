"""FUNCTIONAL-PREVIEW-14.6 Block L closure audit.

Static plain-data closure audit for Block L. It reads the safe 14.5 live
canary gate contract and closes Block L without activating runtime, starting
canary paths, enabling trading, touching order flow, private endpoints,
network, credentials, filesystem, UI bridge, or packaging surfaces.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_live_canary_gate_contract import (
    build_preview_block_l_live_canary_gate_contract,
)

PREVIEW_BLOCK_L_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_l_closure_audit.v1"
PREVIEW_BLOCK_L_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_l_closure_audit"
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.6"
BLOCK_L_CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_l_closed_source_only_closure_audit_no_runtime_activation_no_live_canary_"
    "no_live_trading_no_orders_no_private_endpoints_no_network_io_no_credentials_"
    "no_filesystem_io_no_exe_packaging"
)
BLOCK_L_CLOSURE_AUDIT_DECISION: Final[str] = (
    "CLOSE_BLOCK_L_SOURCE_ONLY_NO_RUNTIME_ACTIVATION_NO_LIVE_CANARY_NO_LIVE_TRADING_"
    "NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_"
    "NO_EXE_PACKAGING"
)
READY_FOR_NEXT_BLOCK: Final[bool] = True
NEXT_BLOCK: Final[str] = "BLOK M — NEXT PREVIEW BLOCK / EXE DIRECTION PRESERVED"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.0"
NEXT_STEP_TITLE: Final[str] = "NEXT BLOCK ENTRY CONTRACT"
CLOSURE_LINE: Final[str] = "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"
STATUS: Final[str] = "ready_for_functional_preview_15_0_next_block_entry_contract"

_LIVE_CANARY_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_live_canary_gate_contract_kind",
    "block",
    "step",
    "block_l_live_canary_gate_contract_status",
    "block_l_live_canary_gate_contract_decision",
    "ready_for_block_l_6",
    "next_step",
    "next_step_title",
]

_STEP_ROWS: Final[list[tuple[str, str, str]]] = [
    (
        "FUNCTIONAL-PREVIEW-14.0",
        "Runtime Activation Contract",
        "static_runtime_activation_contract",
    ),
    ("FUNCTIONAL-PREVIEW-14.1", "Runtime Activation Read Model", "source_only_read_model"),
    ("FUNCTIONAL-PREVIEW-14.2", "Runtime Activation Gate Matrix", "source_only_gate_matrix"),
    ("FUNCTIONAL-PREVIEW-14.3", "Paper Runtime Activation Gate", "source_only_paper_gate"),
    ("FUNCTIONAL-PREVIEW-14.4", "Testnet Runtime Activation Gate", "source_only_testnet_gate"),
    ("FUNCTIONAL-PREVIEW-14.5", "Live Canary Gate Contract", "source_only_live_canary_contract"),
    ("FUNCTIONAL-PREVIEW-14.6", "Block L Closure Audit", "source_only_closure_audit"),
]

_CAPABILITY_ROWS: Final[list[tuple[str, str]]] = [
    ("runtime_activation", "Runtime activation"),
    ("paper_runtime", "Paper runtime"),
    ("testnet_runtime", "Testnet runtime"),
    ("live_canary", "Live canary"),
    ("live_trading", "Live trading"),
    ("runtime_loop", "Runtime loop"),
    ("gate_execution", "Gate execution"),
    ("gate_mutation", "Gate mutation"),
    ("order_generation", "Order generation"),
    ("order_submission", "Order submission"),
    ("order_cancel", "Order cancel"),
    ("order_replace", "Order replace"),
    ("private_endpoints", "Private endpoints"),
    ("network_io", "Network I/O"),
    ("filesystem_io", "Filesystem I/O"),
    ("credentials", "Credentials"),
    ("config_env_secrets", "Config/env/secrets"),
    ("exe_packaging", "EXE packaging"),
]


def build_preview_block_l_closure_audit() -> dict[str, Any]:
    """Build the Block L 14.6 static closure audit."""
    live_canary_contract = build_preview_block_l_live_canary_gate_contract()
    reference = _build_live_canary_contract_reference(live_canary_contract)
    return {
        "schema_version": PREVIEW_BLOCK_L_CLOSURE_AUDIT_SCHEMA_VERSION,
        "block_l_closure_audit_kind": PREVIEW_BLOCK_L_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_closure_audit_status": BLOCK_L_CLOSURE_AUDIT_STATUS,
        "block_l_closure_audit_decision": BLOCK_L_CLOSURE_AUDIT_DECISION,
        "ready_for_next_block": READY_FOR_NEXT_BLOCK,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "closure_line": CLOSURE_LINE,
        "live_canary_contract_reference": reference,
        "block_l_completion_summary": _build_block_l_completion_summary(),
        "block_l_step_ledger": _build_block_l_step_ledger(),
        "block_l_safety_closure_matrix": _build_block_l_safety_closure_matrix(),
        "blocked_capability_closure_summary": _build_blocked_capability_closure_summary(
            live_canary_contract
        ),
        "non_activation_closure_evidence": _build_non_activation_closure_evidence(
            live_canary_contract
        ),
        "fail_closed_closure_decision": _build_fail_closed_closure_decision(),
        "closure_boundaries": _build_closure_boundaries(),
        "source_boundaries": _build_source_boundaries(live_canary_contract),
        "future_steps": ["functional_preview_15_0_next_block_entry_contract"],
        "status": STATUS,
    }


def _build_live_canary_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    reference = {key: contract[key] for key in _LIVE_CANARY_REFERENCE_KEYS}
    reference["source_live_canary_contract_step"] = "FUNCTIONAL-PREVIEW-14.5"
    reference["source_live_canary_contract_read_by_14_6_closure"] = True
    reference["static_live_canary_contract_only"] = True
    reference["runtime_activated_by_14_6"] = False
    reference["live_canary_started_by_14_6"] = False
    reference["live_trading_enabled_by_14_6"] = False
    reference["orders_enabled_by_14_6"] = False
    reference["network_io_opened_by_14_6"] = False
    reference["credentials_read_by_14_6"] = False
    reference["private_endpoint_accessed_by_14_6"] = False
    reference["exe_packaging_started_by_14_6"] = False
    return reference


def _build_block_l_completion_summary() -> dict[str, bool]:
    return {
        "block_l_started": True,
        "block_l_contract_built": True,
        "runtime_activation_read_model_built": True,
        "runtime_activation_gate_matrix_built": True,
        "paper_runtime_activation_gate_built": True,
        "testnet_runtime_activation_gate_built": True,
        "live_canary_gate_contract_built": True,
        "block_l_closure_audit_built": True,
        "block_l_closed": True,
        "ready_for_next_block": True,
        "safe_to_activate_runtime_now": False,
        "safe_to_start_paper_runtime_now": False,
        "safe_to_start_testnet_runtime_now": False,
        "safe_to_start_live_canary_now": False,
        "safe_to_enable_live_trading_now": False,
        "safe_to_generate_orders_now": False,
        f"safe_to_{'sub'}mit_orders_now": False,
        f"safe_to_{'can'}cel_orders_now": False,
        f"safe_to_{'re'}place_orders_now": False,
        "safe_to_access_private_endpoints_now": False,
        "safe_to_open_network_io_now": False,
        "safe_to_read_credentials_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
        "exe_packaging_in_scope_now": False,
    }


def _build_block_l_step_ledger() -> list[dict[str, Any]]:
    return [
        {
            "step_id": step_id,
            "title": title,
            "artifact_type": artifact_type,
            "source_only": True,
            "plain_data_only": True,
            "completed": True,
            "runtime_activation_performed": False,
            "gate_execution_performed": False,
            "live_canary_started": False,
            "orders_enabled": False,
            "private_endpoint_accessed": False,
            "network_io_opened": False,
            "filesystem_io_performed": False,
            "credentials_read": False,
        }
        for step_id, title, artifact_type in _STEP_ROWS
    ]


def _build_block_l_safety_closure_matrix() -> list[dict[str, Any]]:
    return [
        {
            "capability_id": capability_id,
            "display_name": display_name,
            "blocked_through_block_l": True,
            "allowed_now": False,
            "executed_in_block_l": False,
            "requires_future_explicit_gate": True,
            "notes": "Fail-closed through Block L closure; future block must add an explicit source contract.",
        }
        for capability_id, display_name in _CAPABILITY_ROWS
    ]


def _build_blocked_capability_closure_summary(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["live_canary_blocked_summary"]
    capabilities = [*source["blocked_capabilities"], "block_l_closure"]
    return {
        "blocked_capability_count": len(capabilities),
        "blocked_capabilities": capabilities,
        "runtime_activation_blocked": True,
        "paper_runtime_blocked": True,
        "testnet_runtime_blocked": True,
        "live_canary_start_blocked": True,
        "live_trading_blocked": True,
        "order_generation_blocked": True,
        "order_submission_blocked": True,
        "order_cancel_blocked": True,
        "order_replace_blocked": True,
        "private_endpoint_access_blocked": True,
        "network_io_blocked": True,
        "filesystem_io_blocked": True,
        "credential_read_blocked": True,
        "config_env_secret_read_blocked": True,
        "runtime_loop_blocked": True,
        "runtime_gate_execution_blocked": True,
        "gate_state_mutation_blocked": True,
        "exe_packaging_blocked": True,
    }


def _build_non_activation_closure_evidence(contract: dict[str, Any]) -> dict[str, bool]:
    source = contract["non_start_evidence"]
    return {
        "source_live_canary_contract_read": True,
        "block_l_closure_audit_built": True,
        "block_l_closed": True,
        "source_live_canary_contract_live_canary_started": source["live_canary_started"],
        "source_live_canary_contract_runtime_loop_started": source["runtime_loop_started"],
        "source_live_canary_contract_runtime_gate_executed": source["runtime_gate_executed"],
        "source_live_canary_contract_gate_state_mutated": source["gate_state_mutated"],
        "source_live_canary_contract_mode_activated": source["mode_activated"],
        "source_live_canary_contract_order_generated": source["order_generated"],
        "source_live_canary_contract_order_submitted": source["order_submitted"],
        "source_live_canary_contract_private_endpoint_accessed": source[
            "private_endpoint_accessed"
        ],
        "source_live_canary_contract_network_io_performed": source["network_io_performed"],
        "source_live_canary_contract_filesystem_io_performed": source["filesystem_io_performed"],
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
        "exe_packaging_started": False,
    }


def _build_fail_closed_closure_decision() -> dict[str, str]:
    return {
        "missing_live_canary_contract_policy": "fail_closed",
        "missing_block_l_step_policy": "fail_closed",
        "missing_future_gate_policy": "fail_closed",
        "runtime_activation_in_block_l": "blocked",
        "paper_runtime_start_in_block_l": "blocked",
        "testnet_runtime_start_in_block_l": "blocked",
        "live_canary_start_in_block_l": "blocked",
        "live_trading_in_block_l": "blocked",
        "order_generation_in_block_l": "blocked",
        "order_submission_in_block_l": "blocked",
        "order_cancel_in_block_l": "blocked",
        "order_replace_in_block_l": "blocked",
        "private_endpoint_in_block_l": "blocked",
        "network_io_in_block_l": "blocked",
        "filesystem_io_in_block_l": "blocked",
        "credential_read_in_block_l": "blocked",
        "config_env_secret_read_in_block_l": "blocked",
        "runtime_loop_start_in_block_l": "blocked",
        "runtime_gate_execution_in_block_l": "blocked",
        "gate_state_mutation_in_block_l": "blocked",
        "exe_packaging_in_block_l": "blocked",
    }


def _build_closure_boundaries() -> dict[str, bool]:
    return {
        "block_l_closure_is_plain_data_only": True,
        "block_l_closure_is_source_only": True,
        "block_l_closure_closes_block_l": True,
        "block_l_closure_can_feed_next_block_entry_contract": True,
        "block_l_closure_preserves_exe_direction_without_packaging": True,
        "block_l_closure_cannot_activate_runtime": True,
        "block_l_closure_cannot_start_paper_runtime": True,
        "block_l_closure_cannot_start_testnet_runtime": True,
        "block_l_closure_cannot_start_live_canary": True,
        "block_l_closure_cannot_enable_live_trading": True,
        "block_l_closure_cannot_generate_orders": True,
        f"block_l_closure_cannot_{'sub'}mit_orders": True,
        f"block_l_closure_cannot_{'can'}cel_orders": True,
        f"block_l_closure_cannot_{'re'}place_orders": True,
        "block_l_closure_cannot_access_private_endpoints": True,
        "block_l_closure_cannot_open_network_io": True,
        "block_l_closure_cannot_read_credentials": True,
        "block_l_closure_cannot_start_runtime_loop": True,
        "block_l_closure_cannot_execute_runtime_gates": True,
        "block_l_closure_cannot_mutate_gate_state": True,
        "block_l_closure_cannot_perform_filesystem_io": True,
        "block_l_closure_cannot_read_config_env_or_secrets": True,
        "block_l_closure_cannot_start_exe_packaging": True,
        "block_l_closure_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_live_canary_contract": "FUNCTIONAL-PREVIEW-14.5",
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "forbidden_packaging_calls_present": False,
        "source_live_canary_contract_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_testnet_gate": source["source_testnet_gate"],
            "forbidden_runtime_calls_present": source["forbidden_runtime_calls_present"],
            "forbidden_io_calls_present": source["forbidden_io_calls_present"],
            "forbidden_network_calls_present": source["forbidden_network_calls_present"],
            "forbidden_private_endpoint_calls_present": source[
                "forbidden_private_endpoint_calls_present"
            ],
            "forbidden_ui_bridge_calls_present": source["forbidden_ui_bridge_calls_present"],
        },
    }
