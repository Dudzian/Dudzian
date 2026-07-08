"""FUNCTIONAL-PREVIEW-15.0 Block M entry contract.

Static plain-data entry contract for Block M. It consumes the safe Block L
closure audit subset and opens Block M while preserving the future desktop EXE
direction without packaging, runtime activation, trading, endpoint access,
network, credentials, filesystem, UI bridge, or build-system changes.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_l_closure_audit import build_preview_block_l_closure_audit

PREVIEW_BLOCK_M_ENTRY_CONTRACT_SCHEMA_VERSION: Final[str] = "preview_block_m_entry_contract.v1"
PREVIEW_BLOCK_M_ENTRY_CONTRACT_KIND: Final[str] = "functional_preview_block_m_entry_contract"
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.0"
BLOCK_M_ENTRY_CONTRACT_STATUS: Final[str] = (
    "block_m_entry_contract_ready_exe_direction_preserved_no_packaging_no_pyinstaller_"
    "no_build_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_"
    "no_filesystem_io"
)
BLOCK_M_ENTRY_CONTRACT_DECISION: Final[str] = (
    "OPEN_BLOCK_M_SOURCE_ONLY_EXE_DIRECTION_PRESERVED_NO_PACKAGING_NO_PYINSTALLER_"
    "NO_BUILD_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_"
    "NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.1"
NEXT_STEP_TITLE: Final[str] = "BLOCK M READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_15_1_block_m_read_model"
SOURCE_BLOCK_L_CLOSURE_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.6"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_entry_contract_kind",
    "block",
    "step",
    "block_m_entry_contract_status",
    "block_m_entry_contract_decision",
    "ready_for_block_m_1",
    "next_step",
    "next_step_title",
    "block_l_closure_reference",
    "block_m_entry_summary",
    "block_m_scope_contract",
    "exe_direction_preservation_contract",
    "forbidden_scope_matrix",
    "fail_closed_entry_decision",
    "non_activation_evidence",
    "entry_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

_BLOCK_L_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_l_closure_audit_kind",
    "block",
    "step",
    "block_l_closure_audit_status",
    "block_l_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
]

_FORBIDDEN_CAPABILITY_ROWS: Final[list[tuple[str, str]]] = [
    ("exe_packaging", "EXE packaging"),
    ("pyinstaller", "PyInstaller"),
    ("build_artifact_creation", "Build artifact creation"),
    ("release_workflow_changes", "Release workflow changes"),
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
    ("qml_bridge_changes", "QML/bridge changes"),
]


def build_preview_block_m_entry_contract() -> dict[str, Any]:
    """Build the Block M 15.0 static source-only entry contract."""
    block_l_closure_audit = build_preview_block_l_closure_audit()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_ENTRY_CONTRACT_SCHEMA_VERSION,
        "block_m_entry_contract_kind": PREVIEW_BLOCK_M_ENTRY_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_entry_contract_status": BLOCK_M_ENTRY_CONTRACT_STATUS,
        "block_m_entry_contract_decision": BLOCK_M_ENTRY_CONTRACT_DECISION,
        "ready_for_block_m_1": READY_FOR_BLOCK_M_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_l_closure_reference": _build_block_l_closure_reference(block_l_closure_audit),
        "block_m_entry_summary": _build_block_m_entry_summary(),
        "block_m_scope_contract": _build_block_m_scope_contract(),
        "exe_direction_preservation_contract": _build_exe_direction_preservation_contract(),
        "forbidden_scope_matrix": _build_forbidden_scope_matrix(),
        "fail_closed_entry_decision": _build_fail_closed_entry_decision(),
        "non_activation_evidence": _build_non_activation_evidence(block_l_closure_audit),
        "entry_boundaries": _build_entry_boundaries(),
        "source_boundaries": _build_source_boundaries(block_l_closure_audit),
        "future_steps": ["functional_preview_15_1_block_m_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_l_closure_reference(audit: dict[str, Any]) -> dict[str, Any]:
    reference = {key: audit[key] for key in _BLOCK_L_REFERENCE_KEYS}
    reference["source_block_l_closure_step"] = SOURCE_BLOCK_L_CLOSURE_STEP
    reference["source_block_l_closure_read_by_15_0_entry_contract"] = True
    reference["block_l_closed_before_block_m_entry"] = True
    reference["static_closure_audit_only"] = True
    reference["runtime_activated_by_15_0"] = False
    reference["live_canary_started_by_15_0"] = False
    reference["live_trading_enabled_by_15_0"] = False
    reference["orders_enabled_by_15_0"] = False
    reference["network_io_opened_by_15_0"] = False
    reference["credentials_read_by_15_0"] = False
    reference["private_endpoint_accessed_by_15_0"] = False
    reference["filesystem_io_performed_by_15_0"] = False
    reference["exe_packaging_started_by_15_0"] = False
    return reference


def _build_block_m_entry_summary() -> dict[str, bool]:
    return {
        "block_l_closure_available": True,
        "block_l_closed": True,
        "block_m_entry_contract_built": True,
        "block_m_opened": True,
        "ready_for_block_m_1": True,
        "exe_direction_preserved": True,
        "exe_packaging_in_scope_now": False,
        "pyinstaller_in_scope_now": False,
        "build_artifact_creation_in_scope_now": False,
        "release_packaging_in_scope_now": False,
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
    }


def _build_block_m_scope_contract() -> dict[str, bool]:
    return {
        "block_m_scope_is_entry_contract_only": True,
        "block_m_scope_is_source_only": True,
        "block_m_scope_is_plain_data_only": True,
        "block_m_scope_preserves_exe_direction": True,
        "block_m_scope_does_not_package_exe": True,
        "block_m_scope_does_not_change_build_system": True,
        "block_m_scope_does_not_change_runtime": True,
        "block_m_scope_does_not_change_ui_bridge": True,
        "block_m_scope_does_not_touch_credentials": True,
        "block_m_scope_does_not_touch_network": True,
        "block_m_scope_does_not_touch_private_endpoints": True,
        "block_m_scope_does_not_touch_orders": True,
        "block_m_scope_does_not_touch_filesystem": True,
    }


def _build_exe_direction_preservation_contract() -> dict[str, Any]:
    return {
        "final_product_direction": "desktop_exe",
        "exe_direction_preserved": True,
        "exe_packaging_started_now": False,
        "pyinstaller_started_now": False,
        "build_command_added_now": False,
        "workflow_changed_for_packaging_now": False,
        "installer_changed_now": False,
        "release_artifact_created_now": False,
        "packaging_deferred_to_future_explicit_block": True,
        "future_packaging_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }


def _build_forbidden_scope_matrix() -> list[dict[str, Any]]:
    return [
        {
            "capability_id": capability_id,
            "display_name": display_name,
            "forbidden_in_15_0": True,
            "allowed_now": False,
            "executed_now": False,
            "requires_future_explicit_gate": True,
            "notes": "Blocked in 15.0 entry contract; future work requires an explicit gate.",
        }
        for capability_id, display_name in _FORBIDDEN_CAPABILITY_ROWS
    ]


def _build_fail_closed_entry_decision() -> dict[str, str]:
    return {
        "missing_block_l_closure_policy": "fail_closed",
        "missing_block_m_scope_policy": "fail_closed",
        "missing_future_gate_policy": "fail_closed",
        "exe_packaging_in_15_0": "blocked",
        "pyinstaller_in_15_0": "blocked",
        "build_artifact_creation_in_15_0": "blocked",
        "release_workflow_change_in_15_0": "blocked",
        "runtime_activation_in_15_0": "blocked",
        "paper_runtime_start_in_15_0": "blocked",
        "testnet_runtime_start_in_15_0": "blocked",
        "live_canary_start_in_15_0": "blocked",
        "live_trading_in_15_0": "blocked",
        "order_generation_in_15_0": "blocked",
        "order_submission_in_15_0": "blocked",
        "order_cancel_in_15_0": "blocked",
        "order_replace_in_15_0": "blocked",
        "private_endpoint_in_15_0": "blocked",
        "network_io_in_15_0": "blocked",
        "filesystem_io_in_15_0": "blocked",
        "credential_read_in_15_0": "blocked",
        "config_env_secret_read_in_15_0": "blocked",
        "qml_bridge_change_in_15_0": "blocked",
    }


def _build_non_activation_evidence(audit: dict[str, Any]) -> dict[str, bool]:
    source = audit["non_activation_closure_evidence"]
    return {
        "source_block_l_closure_read": True,
        "block_m_entry_contract_built": True,
        "block_m_opened": True,
        "source_block_l_closure_live_canary_started": source["live_canary_started"],
        "source_block_l_closure_runtime_loop_started": source["runtime_loop_started"],
        "source_block_l_closure_runtime_gate_executed": source["runtime_gate_executed"],
        "source_block_l_closure_gate_state_mutated": source["gate_state_mutated"],
        "source_block_l_closure_mode_activated": source["mode_activated"],
        "source_block_l_closure_order_generated": source["order_generated"],
        "source_block_l_closure_order_submitted": source["order_submitted"],
        "source_block_l_closure_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_block_l_closure_network_io_performed": source["network_io_performed"],
        "source_block_l_closure_filesystem_io_performed": source["filesystem_io_performed"],
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
        "pyinstaller_started": False,
        "build_artifact_created": False,
        "release_workflow_changed": False,
        "qml_bridge_changed": False,
    }


def _build_entry_boundaries() -> dict[str, bool]:
    return {
        "block_m_entry_is_plain_data_only": True,
        "block_m_entry_is_source_only": True,
        "block_m_entry_opens_block_m": True,
        "block_m_entry_preserves_exe_direction_without_packaging": True,
        "block_m_entry_can_feed_15_1_read_model": True,
        "block_m_entry_cannot_package_exe": True,
        "block_m_entry_cannot_start_pyinstaller": True,
        "block_m_entry_cannot_create_build_artifacts": True,
        "block_m_entry_cannot_change_release_workflows": True,
        "block_m_entry_cannot_activate_runtime": True,
        "block_m_entry_cannot_start_paper_runtime": True,
        "block_m_entry_cannot_start_testnet_runtime": True,
        "block_m_entry_cannot_start_live_canary": True,
        "block_m_entry_cannot_enable_live_trading": True,
        "block_m_entry_cannot_generate_orders": True,
        f"block_m_entry_cannot_{'sub'}mit_orders": True,
        f"block_m_entry_cannot_{'can'}cel_orders": True,
        f"block_m_entry_cannot_{'re'}place_orders": True,
        "block_m_entry_cannot_access_private_endpoints": True,
        "block_m_entry_cannot_open_network_io": True,
        "block_m_entry_cannot_read_credentials": True,
        "block_m_entry_cannot_start_runtime_loop": True,
        "block_m_entry_cannot_execute_runtime_gates": True,
        "block_m_entry_cannot_mutate_gate_state": True,
        "block_m_entry_cannot_perform_filesystem_io": True,
        "block_m_entry_cannot_read_config_env_or_secrets": True,
        "block_m_entry_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(audit: dict[str, Any]) -> dict[str, Any]:
    source = audit["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_l_closure": SOURCE_BLOCK_L_CLOSURE_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_l_closure_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_live_canary_contract": source["source_live_canary_contract"],
            "forbidden_runtime_calls_present": source["forbidden_runtime_calls_present"],
            "forbidden_io_calls_present": source["forbidden_io_calls_present"],
            "forbidden_network_calls_present": source["forbidden_network_calls_present"],
            "forbidden_private_endpoint_calls_present": source[
                "forbidden_private_endpoint_calls_present"
            ],
            "forbidden_ui_bridge_calls_present": source["forbidden_ui_bridge_calls_present"],
            "forbidden_packaging_calls_present": source["forbidden_packaging_calls_present"],
        },
    }
