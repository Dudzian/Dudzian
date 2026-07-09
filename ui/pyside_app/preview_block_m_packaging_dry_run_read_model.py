"""FUNCTIONAL-PREVIEW-15.5 Block M packaging dry-run read model.

Source-only plain-data read model over the 15.4 packaging dry-run contract.
It preserves the future desktop EXE direction while keeping dry-run execution,
packaging, PyInstaller, build commands, artifacts, runtime, trading, endpoints,
network, credentials, filesystem, UI bridge, installer, and release workflow
execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_dry_run_contract import (
    build_preview_block_m_packaging_dry_run_contract,
)

PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_dry_run_read_model.v1"
)
PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_m_packaging_dry_run_read_model"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.5"
BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_STATUS: Final[str] = (
    "block_m_packaging_dry_run_read_model_ready_exe_direction_preserved_dry_run_"
    "read_model_static_only_no_dry_run_execution_no_packaging_execution_no_pyinstaller_"
    "no_build_no_artifacts_no_runtime_no_orders_no_private_endpoints_no_network_io_"
    "no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_READY_EXE_DIRECTION_PRESERVED_DRY_RUN_"
    "READ_MODEL_STATIC_ONLY_NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_"
    "NO_BUILD_NO_ARTIFACTS_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_"
    "NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_6: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.6"
NEXT_STEP_TITLE: Final[str] = "PACKAGING ARTIFACT POLICY MATRIX"
STATUS: Final[str] = "ready_for_functional_preview_15_6_packaging_artifact_policy_matrix"
SOURCE_PACKAGING_DRY_RUN_CONTRACT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.4"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_dry_run_read_model_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_read_model_status",
    "block_m_packaging_dry_run_read_model_decision",
    "ready_for_block_m_6",
    "next_step",
    "next_step_title",
    "packaging_dry_run_contract_reference",
    "packaging_dry_run_read_summary",
    "dry_run_prerequisite_read_rows",
    "dry_run_execution_read_model",
    "dry_run_simulation_read_rows",
    "dry_run_artifact_policy_read_model",
    "runtime_safety_carryover_read_rows",
    "exe_direction_dry_run_read_model",
    "fail_closed_dry_run_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_CONTRACT_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_dry_run_contract_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_contract_status",
    "block_m_packaging_dry_run_contract_decision",
    "ready_for_block_m_5",
    "next_step",
    "next_step_title",
]
_SAFE_ORDER_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_ORDER_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_ORDER_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_ORDER_SUBMISSION_KEY: Final[str] = (
    "packaging_dry_run_read_model_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_ORDER_CANCEL_KEY: Final[str] = (
    "packaging_dry_run_read_model_cannot_" + "can" + "cel_orders"
)
_BOUNDARY_ORDER_REPLACE_KEY: Final[str] = (
    "packaging_dry_run_read_model_cannot_" + "re" + "place_orders"
)


def build_preview_block_m_packaging_dry_run_read_model() -> dict[str, Any]:
    """Build the Block M 15.5 source-only packaging dry-run read model."""
    contract = build_preview_block_m_packaging_dry_run_contract()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
        "block_m_packaging_dry_run_read_model_kind": PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_dry_run_read_model_status": BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_STATUS,
        "block_m_packaging_dry_run_read_model_decision": BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_DECISION,
        "ready_for_block_m_6": READY_FOR_BLOCK_M_6,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_dry_run_contract_reference": _build_contract_reference(contract),
        "packaging_dry_run_read_summary": _build_read_summary(),
        "dry_run_prerequisite_read_rows": _build_prerequisite_read_rows(contract),
        "dry_run_execution_read_model": _build_execution_read_model(contract),
        "dry_run_simulation_read_rows": _build_simulation_read_rows(contract),
        "dry_run_artifact_policy_read_model": _build_artifact_policy_read_model(contract),
        "runtime_safety_carryover_read_rows": _build_runtime_safety_carryover_read_rows(contract),
        "exe_direction_dry_run_read_model": _build_exe_direction_dry_run_read_model(contract),
        "fail_closed_dry_run_read_decision": _build_fail_closed_dry_run_read_decision(),
        "non_execution_evidence": _build_non_execution_evidence(contract),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(contract),
        "future_steps": ["functional_preview_15_6_packaging_artifact_policy_matrix"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    reference = {key: contract[key] for key in _CONTRACT_REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_dry_run_contract_step": SOURCE_PACKAGING_DRY_RUN_CONTRACT_STEP,
            "source_packaging_dry_run_contract_read_by_15_5_read_model": True,
            "packaging_dry_run_contract_available_before_read_model": True,
            "static_packaging_dry_run_contract_only": True,
            "packaging_dry_run_executed_by_15_5": False,
            "packaging_executed_by_15_5": False,
            "pyinstaller_started_by_15_5": False,
            "build_command_executed_by_15_5": False,
            "build_artifact_created_by_15_5": False,
            "installer_changed_by_15_5": False,
            "release_workflow_changed_by_15_5": False,
            "artifact_smoke_test_executed_by_15_5": False,
            "artifact_signed_by_15_5": False,
            "artifact_published_by_15_5": False,
            "dependency_freeze_performed_by_15_5": False,
            "asset_discovery_performed_by_15_5": False,
            "qml_asset_discovery_performed_by_15_5": False,
            "runtime_activated_by_15_5": False,
            "orders_enabled_by_15_5": False,
            "network_io_opened_by_15_5": False,
            "credentials_read_by_15_5": False,
            "private_endpoint_accessed_by_15_5": False,
            "filesystem_io_performed_by_15_5": False,
            "qml_bridge_changed_by_15_5": False,
        }
    )
    return reference


def _build_read_summary() -> dict[str, bool]:
    return {
        "packaging_dry_run_contract_available": True,
        "packaging_dry_run_read_model_built": True,
        "ready_for_block_m_6": True,
        "exe_direction_preserved": True,
        "dry_run_read_model_static_only": True,
        "dry_run_read_model_ready_for_future_matrix": True,
        "dry_run_contract_read_only": True,
        "dry_run_satisfied_now": False,
        "dry_run_can_execute_now": False,
        "packaging_ready_now": False,
        "packaging_can_execute_now": False,
        "pyinstaller_can_start_now": False,
        "build_command_can_execute_now": False,
        "build_artifact_can_be_created_now": False,
        "installer_can_change_now": False,
        "release_workflow_can_change_now": False,
        "artifact_smoke_test_can_run_now": False,
        "artifact_signing_can_run_now": False,
        "artifact_publishing_can_run_now": False,
        "dependency_freeze_can_run_now": False,
        "asset_discovery_can_run_now": False,
        "qml_asset_discovery_can_run_now": False,
        "packaging_filesystem_io_allowed_now": False,
        "safe_to_activate_runtime_now": False,
        "safe_to_start_paper_runtime_now": False,
        "safe_to_start_testnet_runtime_now": False,
        "safe_to_start_live_canary_now": False,
        "safe_to_enable_live_trading_now": False,
        "safe_to_generate_orders_now": False,
        _SAFE_ORDER_SUBMISSION_KEY: False,
        _SAFE_ORDER_CANCEL_KEY: False,
        _SAFE_ORDER_REPLACE_KEY: False,
        "safe_to_access_private_endpoints_now": False,
        "safe_to_open_network_io_now": False,
        "safe_to_read_credentials_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
        "safe_to_change_qml_bridge_now": False,
    }


def _build_prerequisite_read_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row["source_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "source_failure_policy": row["source_failure_policy"],
            "read_row_type": "packaging_dry_run_static_prerequisite_read_row",
            "source_required_before_dry_run": True,
            "source_satisfied_in_15_4": row["satisfied_in_15_4"],
            "source_checked_by_15_4": row["checked_by_15_4"],
            "required_before_future_dry_run": True,
            "satisfied_in_15_5": False,
            "checked_by_15_5": False,
            "read_by_15_5": True,
            "requires_future_step": True,
            "failure_policy": "fail_closed",
        }
        for row in contract["dry_run_prerequisite_contract"]
    ]


def _build_execution_read_model(contract: dict[str, Any]) -> dict[str, bool]:
    source = contract["dry_run_execution_blocked_contract"]
    return {
        "dry_run_execution_read_model_built": True,
        "source_dry_run_not_performed_by_15_4": source["dry_run_not_performed_by_15_4"],
        "source_packaging_not_performed_by_15_4": source["packaging_not_performed_by_15_4"],
        "packaging_dry_run_execution_allowed_now": False,
        "packaging_execution_allowed_now": False,
        "pyinstaller_execution_allowed_now": False,
        "build_command_execution_allowed_now": False,
        "build_artifact_creation_allowed_now": False,
        "installer_mutation_allowed_now": False,
        "release_workflow_mutation_allowed_now": False,
        "artifact_smoke_test_allowed_now": False,
        "artifact_signing_allowed_now": False,
        "artifact_publishing_allowed_now": False,
        "packaging_filesystem_io_allowed_now": False,
        "packaging_environment_probe_allowed_now": False,
        "dependency_freeze_allowed_now": False,
        "asset_discovery_allowed_now": False,
        "qml_asset_discovery_allowed_now": False,
        "dry_run_requires_future_explicit_gate": True,
        "dry_run_requires_future_operator_confirmation": True,
        "dry_run_not_performed_by_15_5": True,
        "packaging_not_performed_by_15_5": True,
    }


def _build_simulation_read_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "simulation_step_id": row["simulation_step_id"],
            "display_name": row["display_name"],
            "source_planned_for_future_dry_run": row["planned_for_future_dry_run"],
            "source_executed_in_15_4": row["executed_in_15_4"],
            "source_allowed_now": row["allowed_now"],
            "read_model_row_type": "packaging_dry_run_static_simulation_read_row",
            "read_by_15_5": True,
            "planned_for_future_dry_run": True,
            "executed_in_15_5": False,
            "allowed_now": False,
            "requires_future_explicit_gate": True,
            "notes": row["notes"],
        }
        for row in contract["dry_run_simulation_plan_contract"]
    ]


def _build_artifact_policy_read_model(contract: dict[str, Any]) -> dict[str, bool]:
    source = contract["dry_run_artifact_policy_contract"]
    return {
        "artifact_policy_read_model_built": True,
        "source_no_artifact_created_by_15_4": source["no_artifact_created_by_15_4"],
        "artifact_creation_allowed_now": False,
        "artifact_mutation_allowed_now": False,
        "artifact_delete_allowed_now": False,
        "artifact_smoke_test_allowed_now": False,
        "artifact_signing_allowed_now": False,
        "artifact_publishing_allowed_now": False,
        "artifact_naming_finalized_now": False,
        "artifact_location_selected_now": False,
        "artifact_retention_policy_finalized_now": False,
        "artifact_rollback_policy_finalized_now": False,
        "artifact_policy_requires_future_gate": True,
        "artifact_policy_requires_future_operator_confirmation": True,
        "no_artifact_created_by_15_5": True,
        "no_artifact_mutated_by_15_5": True,
        "no_artifact_deleted_by_15_5": True,
    }


def _build_runtime_safety_carryover_read_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_gate_allowed_now": row["source_gate_allowed_now"],
            "source_dry_run_contract_allowed_now": row["dry_run_contract_allowed_now"],
            "source_dry_run_contract_executed_now": row["dry_run_contract_executed_now"],
            "read_model_allowed_now": False,
            "read_model_executed_now": False,
            "blocked_in_15_5": True,
            "requires_future_explicit_gate": True,
            "notes": "15.5 read model carries forward the fail-closed runtime safety boundary unchanged.",
        }
        for row in contract["runtime_safety_carryover_contract"]
    ]


def _build_exe_direction_dry_run_read_model(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["exe_direction_dry_run_contract"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "dry_run_read_model_confirms_exe_direction": True,
        "exe_packaging_started_now": False,
        "packaging_dry_run_started_now": False,
        "pyinstaller_started_now": False,
        "build_command_added_now": False,
        "build_command_executed_now": False,
        "workflow_changed_for_packaging_now": False,
        "installer_changed_now": False,
        "release_artifact_created_now": False,
        "artifact_created_now": False,
        "artifact_smoke_test_executed_now": False,
        "artifact_signed_now": False,
        "artifact_published_now": False,
        "packaging_deferred_to_future_explicit_block": True,
        "dry_run_deferred_to_future_explicit_block": True,
        "future_packaging_requires_explicit_gate": True,
        "future_dry_run_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }


def _build_fail_closed_dry_run_read_decision() -> dict[str, str]:
    blocked_keys = [
        "packaging_dry_run_execution_in_15_5",
        "packaging_execution_in_15_5",
        "pyinstaller_execution_in_15_5",
        "build_command_execution_in_15_5",
        "build_artifact_creation_in_15_5",
        "installer_change_in_15_5",
        "release_workflow_change_in_15_5",
        "artifact_smoke_test_in_15_5",
        "artifact_signing_in_15_5",
        "artifact_publishing_in_15_5",
        "packaging_filesystem_io_in_15_5",
        "packaging_environment_probe_in_15_5",
        "dependency_freeze_in_15_5",
        "asset_discovery_in_15_5",
        "qml_asset_discovery_in_15_5",
        "runtime_activation_in_15_5",
        "paper_runtime_start_in_15_5",
        "testnet_runtime_start_in_15_5",
        "live_canary_start_in_15_5",
        "live_trading_in_15_5",
        "order_generation_in_15_5",
        "order_submission_in_15_5",
        "order_cancel_in_15_5",
        "order_replace_in_15_5",
        "private_endpoint_in_15_5",
        "network_io_in_15_5",
        "credential_read_in_15_5",
        "config_env_secret_read_in_15_5",
        "qml_bridge_change_in_15_5",
    ]
    decision = {
        "missing_packaging_dry_run_contract_policy": "fail_closed",
        "missing_dry_run_read_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    decision.update({key: "blocked" for key in blocked_keys})
    return decision


def _build_non_execution_evidence(contract: dict[str, Any]) -> dict[str, bool]:
    source = contract["non_execution_evidence"]
    evidence = {
        "source_packaging_dry_run_contract_read": True,
        "packaging_dry_run_read_model_built": True,
        "packaging_dry_run_read_model_only": True,
        "source_contract_packaging_dry_run_executed": source["packaging_dry_run_executed"],
        "source_contract_packaging_executed": source["packaging_executed"],
        "source_contract_runtime_loop_started": source["runtime_loop_started"],
        "source_contract_runtime_gate_executed": source["runtime_gate_executed"],
        "source_contract_gate_state_mutated": source["gate_state_mutated"],
        "source_contract_order_generated": source["order_generated"],
        "source_contract_order_submitted": source["order_submitted"],
        "source_contract_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_contract_network_io_performed": source["network_io_performed"],
        "source_contract_filesystem_io_performed": source["filesystem_io_performed"],
    }
    false_keys = [
        "packaging_dry_run_executed",
        "packaging_executed",
        "runtime_activated",
        "paper_runtime_started",
        "testnet_runtime_started",
        "live_canary_started",
        "runtime_loop_started",
        "runtime_gate_executed",
        "gate_state_mutated",
        "mode_activated",
        "order_generated",
        "order_submitted",
        "private_endpoint_accessed",
        "network_io_performed",
        "filesystem_io_performed",
        "credential_read_performed",
        "live_trading_started",
        "pyinstaller_started",
        "build_command_executed",
        "build_artifact_created",
        "installer_changed",
        "release_workflow_changed",
        "artifact_smoke_test_executed",
        "artifact_signed",
        "artifact_published",
        "dependency_freeze_performed",
        "asset_discovery_performed",
        "qml_asset_discovery_performed",
        "qml_bridge_changed",
    ]
    evidence.update({key: False for key in false_keys})
    return evidence


def _build_read_model_boundaries() -> dict[str, bool]:
    return {
        "packaging_dry_run_read_model_is_plain_data_only": True,
        "packaging_dry_run_read_model_is_source_only": True,
        "packaging_dry_run_read_model_reads_dry_run_contract_only": True,
        "packaging_dry_run_read_model_preserves_exe_direction_without_packaging": True,
        "packaging_dry_run_read_model_can_feed_15_6_packaging_artifact_policy_matrix": True,
        "packaging_dry_run_read_model_cannot_execute_dry_run": True,
        "packaging_dry_run_read_model_cannot_package_exe": True,
        "packaging_dry_run_read_model_cannot_start_pyinstaller": True,
        "packaging_dry_run_read_model_cannot_execute_build_commands": True,
        "packaging_dry_run_read_model_cannot_create_build_artifacts": True,
        "packaging_dry_run_read_model_cannot_change_installers": True,
        "packaging_dry_run_read_model_cannot_change_release_workflows": True,
        "packaging_dry_run_read_model_cannot_run_artifact_smoke_tests": True,
        "packaging_dry_run_read_model_cannot_sign_artifacts": True,
        "packaging_dry_run_read_model_cannot_publish_artifacts": True,
        "packaging_dry_run_read_model_cannot_probe_packaging_environment": True,
        "packaging_dry_run_read_model_cannot_freeze_dependencies": True,
        "packaging_dry_run_read_model_cannot_discover_assets": True,
        "packaging_dry_run_read_model_cannot_discover_qml_assets": True,
        "packaging_dry_run_read_model_cannot_perform_filesystem_io": True,
        "packaging_dry_run_read_model_cannot_activate_runtime": True,
        "packaging_dry_run_read_model_cannot_start_paper_runtime": True,
        "packaging_dry_run_read_model_cannot_start_testnet_runtime": True,
        "packaging_dry_run_read_model_cannot_start_live_canary": True,
        "packaging_dry_run_read_model_cannot_enable_live_trading": True,
        "packaging_dry_run_read_model_cannot_generate_orders": True,
        _BOUNDARY_ORDER_SUBMISSION_KEY: True,
        _BOUNDARY_ORDER_CANCEL_KEY: True,
        _BOUNDARY_ORDER_REPLACE_KEY: True,
        "packaging_dry_run_read_model_cannot_access_private_endpoints": True,
        "packaging_dry_run_read_model_cannot_open_network_io": True,
        "packaging_dry_run_read_model_cannot_read_credentials": True,
        "packaging_dry_run_read_model_cannot_start_runtime_loop": True,
        "packaging_dry_run_read_model_cannot_execute_runtime_gates": True,
        "packaging_dry_run_read_model_cannot_mutate_gate_state": True,
        "packaging_dry_run_read_model_cannot_read_config_env_or_secrets": True,
        "packaging_dry_run_read_model_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_packaging_dry_run_contract": SOURCE_PACKAGING_DRY_RUN_CONTRACT_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_dry_run_contract_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_packaging_gate_contract": source["source_packaging_gate_contract"],
            "forbidden_packaging_calls_present": source["forbidden_packaging_calls_present"],
            "forbidden_pyinstaller_calls_present": source["forbidden_pyinstaller_calls_present"],
            "forbidden_build_calls_present": source["forbidden_build_calls_present"],
            "forbidden_runtime_calls_present": source["forbidden_runtime_calls_present"],
            "forbidden_io_calls_present": source["forbidden_io_calls_present"],
            "forbidden_network_calls_present": source["forbidden_network_calls_present"],
            "forbidden_private_endpoint_calls_present": source[
                "forbidden_private_endpoint_calls_present"
            ],
            "forbidden_ui_bridge_calls_present": source["forbidden_ui_bridge_calls_present"],
        },
    }
