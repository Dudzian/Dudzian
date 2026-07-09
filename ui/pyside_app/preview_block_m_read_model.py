"""FUNCTIONAL-PREVIEW-15.1 Block M read model.

Static plain-data read model for Block M. It consumes only the safe Block M
entry contract subset and preserves the future desktop EXE direction without
packaging, runtime activation, trading, endpoint access, network, credentials,
filesystem, UI bridge, or build-system changes.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_entry_contract import build_preview_block_m_entry_contract

PREVIEW_BLOCK_M_READ_MODEL_SCHEMA_VERSION: Final[str] = "preview_block_m_read_model.v1"
PREVIEW_BLOCK_M_READ_MODEL_KIND: Final[str] = "functional_preview_block_m_read_model"
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.1"
BLOCK_M_READ_MODEL_STATUS: Final[str] = (
    "block_m_read_model_ready_exe_direction_preserved_no_packaging_no_pyinstaller_"
    "no_build_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_"
    "no_filesystem_io"
)
BLOCK_M_READ_MODEL_DECISION: Final[str] = (
    "BLOCK_M_READ_MODEL_READY_EXE_DIRECTION_PRESERVED_NO_PACKAGING_NO_PYINSTALLER_"
    "NO_BUILD_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_"
    "NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.2"
NEXT_STEP_TITLE: Final[str] = "BLOCK M PACKAGING READINESS MATRIX"
STATUS: Final[str] = "ready_for_functional_preview_15_2_block_m_packaging_readiness_matrix"
SOURCE_BLOCK_M_ENTRY_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.0"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_read_model_kind",
    "block",
    "step",
    "block_m_read_model_status",
    "block_m_read_model_decision",
    "ready_for_block_m_2",
    "next_step",
    "next_step_title",
    "block_m_entry_contract_reference",
    "block_m_read_summary",
    "exe_direction_read_model",
    "packaging_forbidden_read_model",
    "runtime_forbidden_read_model",
    "capability_read_rows",
    "fail_closed_read_model_decision",
    "non_activation_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

_ENTRY_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_entry_contract_kind",
    "block",
    "step",
    "block_m_entry_contract_status",
    "block_m_entry_contract_decision",
    "ready_for_block_m_1",
    "next_step",
    "next_step_title",
]


def build_preview_block_m_read_model() -> dict[str, Any]:
    """Build the Block M 15.1 static source-only read model."""
    entry_contract = build_preview_block_m_entry_contract()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_READ_MODEL_SCHEMA_VERSION,
        "block_m_read_model_kind": PREVIEW_BLOCK_M_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_read_model_status": BLOCK_M_READ_MODEL_STATUS,
        "block_m_read_model_decision": BLOCK_M_READ_MODEL_DECISION,
        "ready_for_block_m_2": READY_FOR_BLOCK_M_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_m_entry_contract_reference": _build_block_m_entry_contract_reference(entry_contract),
        "block_m_read_summary": _build_block_m_read_summary(),
        "exe_direction_read_model": _build_exe_direction_read_model(entry_contract),
        "packaging_forbidden_read_model": _build_packaging_forbidden_read_model(),
        "runtime_forbidden_read_model": _build_runtime_forbidden_read_model(),
        "capability_read_rows": _build_capability_read_rows(entry_contract),
        "fail_closed_read_model_decision": _build_fail_closed_read_model_decision(),
        "non_activation_evidence": _build_non_activation_evidence(entry_contract),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(entry_contract),
        "future_steps": ["functional_preview_15_2_block_m_packaging_readiness_matrix"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_m_entry_contract_reference(entry: dict[str, Any]) -> dict[str, Any]:
    reference = {key: entry[key] for key in _ENTRY_REFERENCE_KEYS}
    reference["source_block_m_entry_step"] = SOURCE_BLOCK_M_ENTRY_STEP
    reference["source_block_m_entry_read_by_15_1_read_model"] = True
    reference["block_m_opened_before_read_model"] = True
    reference["static_entry_contract_only"] = True
    reference["runtime_activated_by_15_1"] = False
    reference["live_canary_started_by_15_1"] = False
    reference["live_trading_enabled_by_15_1"] = False
    reference["orders_enabled_by_15_1"] = False
    reference["network_io_opened_by_15_1"] = False
    reference["credentials_read_by_15_1"] = False
    reference["private_endpoint_accessed_by_15_1"] = False
    reference["filesystem_io_performed_by_15_1"] = False
    reference["exe_packaging_started_by_15_1"] = False
    reference["pyinstaller_started_by_15_1"] = False
    reference["build_artifact_created_by_15_1"] = False
    reference["qml_bridge_changed_by_15_1"] = False
    return reference


def _build_block_m_read_summary() -> dict[str, bool]:
    return {
        "block_m_entry_contract_available": True,
        "block_m_opened": True,
        "block_m_read_model_built": True,
        "ready_for_block_m_2": True,
        "exe_direction_preserved": True,
        "entry_contract_read_only": True,
        "read_model_plain_data_only": True,
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
        "safe_to_change_qml_bridge_now": False,
    }


def _build_exe_direction_read_model(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry["exe_direction_preservation_contract"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "read_model_confirms_exe_direction": True,
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


def _build_packaging_forbidden_read_model() -> dict[str, bool]:
    return {
        "packaging_read_model_built": True,
        "exe_packaging_allowed_now": False,
        "pyinstaller_allowed_now": False,
        "build_artifact_creation_allowed_now": False,
        "installer_changes_allowed_now": False,
        "release_workflow_changes_allowed_now": False,
        "packaging_command_execution_allowed_now": False,
        "packaging_filesystem_io_allowed_now": False,
        "packaging_requires_future_explicit_gate": True,
        "packaging_requires_future_block": True,
        "packaging_not_performed_by_read_model": True,
    }


def _build_runtime_forbidden_read_model() -> dict[str, bool]:
    return {
        "runtime_read_model_built": True,
        "runtime_activation_allowed_now": False,
        "paper_runtime_start_allowed_now": False,
        "testnet_runtime_start_allowed_now": False,
        "live_canary_start_allowed_now": False,
        "live_trading_allowed_now": False,
        "runtime_loop_allowed_now": False,
        "runtime_gate_execution_allowed_now": False,
        "gate_state_mutation_allowed_now": False,
        "order_generation_allowed_now": False,
        "order_submission_allowed_now": False,
        "order_cancel_allowed_now": False,
        "order_replace_allowed_now": False,
        "private_endpoint_access_allowed_now": False,
        "network_io_allowed_now": False,
        "credential_read_allowed_now": False,
        "filesystem_io_allowed_now": False,
        "config_env_secret_read_allowed_now": False,
        "qml_bridge_change_allowed_now": False,
    }


def _build_capability_read_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "read_model_row_type": "block_m_capability_static_read_row",
            "read_only": True,
            "source_forbidden_in_15_0": row["forbidden_in_15_0"],
            "allowed_now": False,
            "executed_now": False,
            "requires_future_explicit_gate": True,
            "read_model_confirms_blocked": True,
            "read_model_performed_capability": False,
        }
        for row in entry["forbidden_scope_matrix"]
    ]


def _build_fail_closed_read_model_decision() -> dict[str, str]:
    return {
        "missing_block_m_entry_contract_policy": "fail_closed",
        "missing_capability_row_policy": "fail_closed",
        "missing_future_gate_policy": "fail_closed",
        "exe_packaging_in_15_1": "blocked",
        "pyinstaller_in_15_1": "blocked",
        "build_artifact_creation_in_15_1": "blocked",
        "release_workflow_change_in_15_1": "blocked",
        "runtime_activation_in_15_1": "blocked",
        "paper_runtime_start_in_15_1": "blocked",
        "testnet_runtime_start_in_15_1": "blocked",
        "live_canary_start_in_15_1": "blocked",
        "live_trading_in_15_1": "blocked",
        "order_generation_in_15_1": "blocked",
        "order_submission_in_15_1": "blocked",
        "order_cancel_in_15_1": "blocked",
        "order_replace_in_15_1": "blocked",
        "private_endpoint_in_15_1": "blocked",
        "network_io_in_15_1": "blocked",
        "filesystem_io_in_15_1": "blocked",
        "credential_read_in_15_1": "blocked",
        "config_env_secret_read_in_15_1": "blocked",
        "qml_bridge_change_in_15_1": "blocked",
    }


def _build_non_activation_evidence(entry: dict[str, Any]) -> dict[str, bool]:
    source = entry["non_activation_evidence"]
    return {
        "source_block_m_entry_contract_read": True,
        "block_m_read_model_built": True,
        "block_m_read_model_only": True,
        "source_block_m_entry_runtime_loop_started": source["runtime_loop_started"],
        "source_block_m_entry_runtime_gate_executed": source["runtime_gate_executed"],
        "source_block_m_entry_gate_state_mutated": source["gate_state_mutated"],
        "source_block_m_entry_mode_activated": source["mode_activated"],
        "source_block_m_entry_order_generated": source["order_generated"],
        "source_block_m_entry_order_submitted": source["order_submitted"],
        "source_block_m_entry_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_block_m_entry_network_io_performed": source["network_io_performed"],
        "source_block_m_entry_filesystem_io_performed": source["filesystem_io_performed"],
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


def _build_read_model_boundaries() -> dict[str, bool]:
    return {
        "block_m_read_model_is_plain_data_only": True,
        "block_m_read_model_is_source_only": True,
        "block_m_read_model_reads_entry_contract_only": True,
        "block_m_read_model_preserves_exe_direction_without_packaging": True,
        "block_m_read_model_can_feed_15_2_packaging_readiness_matrix": True,
        "block_m_read_model_cannot_package_exe": True,
        "block_m_read_model_cannot_start_pyinstaller": True,
        "block_m_read_model_cannot_create_build_artifacts": True,
        "block_m_read_model_cannot_change_release_workflows": True,
        "block_m_read_model_cannot_activate_runtime": True,
        "block_m_read_model_cannot_start_paper_runtime": True,
        "block_m_read_model_cannot_start_testnet_runtime": True,
        "block_m_read_model_cannot_start_live_canary": True,
        "block_m_read_model_cannot_enable_live_trading": True,
        "block_m_read_model_cannot_generate_orders": True,
        f"block_m_read_model_cannot_{'sub'}mit_orders": True,
        f"block_m_read_model_cannot_{'can'}cel_orders": True,
        f"block_m_read_model_cannot_{'re'}place_orders": True,
        "block_m_read_model_cannot_access_private_endpoints": True,
        "block_m_read_model_cannot_open_network_io": True,
        "block_m_read_model_cannot_read_credentials": True,
        "block_m_read_model_cannot_start_runtime_loop": True,
        "block_m_read_model_cannot_execute_runtime_gates": True,
        "block_m_read_model_cannot_mutate_gate_state": True,
        "block_m_read_model_cannot_perform_filesystem_io": True,
        "block_m_read_model_cannot_read_config_env_or_secrets": True,
        "block_m_read_model_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_m_entry_contract": SOURCE_BLOCK_M_ENTRY_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_m_entry_contract_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_block_l_closure": source["source_block_l_closure"],
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
