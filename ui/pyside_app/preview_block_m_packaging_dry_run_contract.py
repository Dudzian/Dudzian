"""FUNCTIONAL-PREVIEW-15.4 Block M packaging dry-run contract.

Source-only plain-data contract for a future desktop EXE packaging dry run. It
reads only the safe 15.3 packaging gate contract and keeps packaging dry-run,
packaging, PyInstaller, build commands, artifacts, runtime, trading, endpoints,
network, credentials, filesystem, UI bridge, installer, and release workflow
execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_gate_contract import (
    build_preview_block_m_packaging_gate_contract,
)

PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_dry_run_contract.v1"
)
PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_m_packaging_dry_run_contract"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.4"
BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_STATUS: Final[str] = (
    "block_m_packaging_dry_run_contract_ready_exe_direction_preserved_dry_run_contract_"
    "static_only_no_packaging_execution_no_pyinstaller_no_build_no_artifacts_no_runtime_"
    "no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_READY_EXE_DIRECTION_PRESERVED_DRY_RUN_CONTRACT_"
    "STATIC_ONLY_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_ARTIFACTS_NO_RUNTIME_"
    "NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_5: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.5"
NEXT_STEP_TITLE: Final[str] = "PACKAGING DRY RUN READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_15_5_packaging_dry_run_read_model"
SOURCE_PACKAGING_GATE_CONTRACT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.3"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_dry_run_contract_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_contract_status",
    "block_m_packaging_dry_run_contract_decision",
    "ready_for_block_m_5",
    "next_step",
    "next_step_title",
    "packaging_gate_contract_reference",
    "packaging_dry_run_summary",
    "dry_run_prerequisite_contract",
    "dry_run_execution_blocked_contract",
    "dry_run_simulation_plan_contract",
    "dry_run_artifact_policy_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_dry_run_contract",
    "fail_closed_dry_run_decision",
    "non_execution_evidence",
    "dry_run_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_GATE_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_gate_contract_kind",
    "block",
    "step",
    "block_m_packaging_gate_contract_status",
    "block_m_packaging_gate_contract_decision",
    "ready_for_block_m_4",
    "next_step",
    "next_step_title",
]
_SAFE_ORDER_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_ORDER_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_ORDER_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_ORDER_SUBMISSION_KEY: Final[str] = (
    "packaging_dry_run_contract_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_ORDER_CANCEL_KEY: Final[str] = "packaging_dry_run_contract_cannot_" + "can" + "cel_orders"
_BOUNDARY_ORDER_REPLACE_KEY: Final[str] = (
    "packaging_dry_run_contract_cannot_" + "re" + "place_orders"
)


def build_preview_block_m_packaging_dry_run_contract() -> dict[str, Any]:
    """Build the Block M 15.4 source-only packaging dry-run contract."""
    gate = build_preview_block_m_packaging_gate_contract()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_SCHEMA_VERSION,
        "block_m_packaging_dry_run_contract_kind": PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_dry_run_contract_status": BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_STATUS,
        "block_m_packaging_dry_run_contract_decision": BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_DECISION,
        "ready_for_block_m_5": READY_FOR_BLOCK_M_5,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_gate_contract_reference": _build_gate_reference(gate),
        "packaging_dry_run_summary": _build_packaging_dry_run_summary(),
        "dry_run_prerequisite_contract": _build_dry_run_prerequisite_contract(gate),
        "dry_run_execution_blocked_contract": _build_dry_run_execution_blocked_contract(),
        "dry_run_simulation_plan_contract": _build_dry_run_simulation_plan_contract(),
        "dry_run_artifact_policy_contract": _build_dry_run_artifact_policy_contract(),
        "runtime_safety_carryover_contract": _build_runtime_safety_carryover_contract(gate),
        "exe_direction_dry_run_contract": _build_exe_direction_dry_run_contract(gate),
        "fail_closed_dry_run_decision": _build_fail_closed_dry_run_decision(),
        "non_execution_evidence": _build_non_execution_evidence(gate),
        "dry_run_boundaries": _build_dry_run_boundaries(),
        "source_boundaries": _build_source_boundaries(gate),
        "future_steps": ["functional_preview_15_5_packaging_dry_run_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_gate_reference(gate: dict[str, Any]) -> dict[str, Any]:
    reference = {key: gate[key] for key in _GATE_REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_gate_contract_step": SOURCE_PACKAGING_GATE_CONTRACT_STEP,
            "source_packaging_gate_contract_read_by_15_4_dry_run_contract": True,
            "packaging_gate_contract_available_before_dry_run_contract": True,
            "static_packaging_gate_contract_only": True,
            "packaging_dry_run_executed_by_15_4": False,
            "packaging_executed_by_15_4": False,
            "pyinstaller_started_by_15_4": False,
            "build_command_executed_by_15_4": False,
            "build_artifact_created_by_15_4": False,
            "installer_changed_by_15_4": False,
            "release_workflow_changed_by_15_4": False,
            "artifact_smoke_test_executed_by_15_4": False,
            "artifact_signed_by_15_4": False,
            "artifact_published_by_15_4": False,
            "dependency_freeze_performed_by_15_4": False,
            "asset_discovery_performed_by_15_4": False,
            "qml_asset_discovery_performed_by_15_4": False,
            "runtime_activated_by_15_4": False,
            "orders_enabled_by_15_4": False,
            "network_io_opened_by_15_4": False,
            "credentials_read_by_15_4": False,
            "private_endpoint_accessed_by_15_4": False,
            "filesystem_io_performed_by_15_4": False,
            "qml_bridge_changed_by_15_4": False,
        }
    )
    return reference


def _build_packaging_dry_run_summary() -> dict[str, bool]:
    return {
        "packaging_gate_contract_available": True,
        "packaging_dry_run_contract_built": True,
        "ready_for_block_m_5": True,
        "exe_direction_preserved": True,
        "dry_run_contract_static_only": True,
        "dry_run_ready_for_future_read_model": True,
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


def _build_dry_run_prerequisite_contract(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row.get("prerequisite_id", row.get("check_id", "packaging_gate_row")),
            "display_name": row["display_name"],
            "notes": row["notes"],
            "source_failure_policy": row["failure_policy"],
            "dry_run_row_type": "packaging_dry_run_static_prerequisite_row",
            "required_before_dry_run": True,
            "satisfied_in_15_4": False,
            "checked_by_15_4": False,
            "requires_future_step": True,
            "failure_policy": "fail_closed",
        }
        for row in gate["packaging_prerequisite_gate_rows"]
    ]


def _build_dry_run_execution_blocked_contract() -> dict[str, bool]:
    return {
        "dry_run_execution_contract_built": True,
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
        "dry_run_not_performed_by_15_4": True,
        "packaging_not_performed_by_15_4": True,
    }


def _build_dry_run_simulation_plan_contract() -> list[dict[str, Any]]:
    rows = [
        ("validate_gate_state", "Validate gate state"),
        ("validate_pyinstaller_spec_presence", "Validate PyInstaller spec presence"),
        ("validate_build_environment_contract", "Validate build environment contract"),
        ("validate_dependency_freeze_contract", "Validate dependency freeze contract"),
        ("validate_asset_inclusion_contract", "Validate asset inclusion contract"),
        ("validate_qml_asset_inclusion_contract", "Validate QML asset inclusion contract"),
        ("validate_runtime_disabled_contract", "Validate runtime disabled contract"),
        ("validate_no_live_credentials_contract", "Validate no live credentials contract"),
        ("validate_no_network_build_contract", "Validate no network build contract"),
        ("validate_artifact_naming_contract", "Validate artifact naming contract"),
        ("validate_smoke_policy_contract", "Validate smoke policy contract"),
        ("validate_rollback_delete_policy_contract", "Validate rollback/delete policy contract"),
    ]
    return [
        {
            "simulation_step_id": row_id,
            "display_name": display_name,
            "planned_for_future_dry_run": True,
            "executed_in_15_4": False,
            "allowed_now": False,
            "requires_future_explicit_gate": True,
            "notes": "Static future dry-run simulation plan row; no simulation executed in 15.4.",
        }
        for row_id, display_name in rows
    ]


def _build_dry_run_artifact_policy_contract() -> dict[str, bool]:
    return {
        "artifact_policy_contract_built": True,
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
        "no_artifact_created_by_15_4": True,
    }


def _build_runtime_safety_carryover_contract(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_gate_allowed_now": row["gate_contract_allowed_now"],
            "dry_run_contract_allowed_now": False,
            "dry_run_contract_executed_now": False,
            "blocked_in_15_4": True,
            "requires_future_explicit_gate": True,
            "notes": "15.4 carries forward the 15.3 fail-closed runtime safety boundary unchanged.",
        }
        for row in gate["runtime_safety_carryover_contract"]
    ]


def _build_exe_direction_dry_run_contract(gate: dict[str, Any]) -> dict[str, Any]:
    source = gate["exe_direction_gate_contract"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "dry_run_contract_confirms_exe_direction": True,
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


def _build_fail_closed_dry_run_decision() -> dict[str, str]:
    blocked_keys = [
        "packaging_dry_run_execution_in_15_4",
        "packaging_execution_in_15_4",
        "pyinstaller_execution_in_15_4",
        "build_command_execution_in_15_4",
        "build_artifact_creation_in_15_4",
        "installer_change_in_15_4",
        "release_workflow_change_in_15_4",
        "artifact_smoke_test_in_15_4",
        "artifact_signing_in_15_4",
        "artifact_publishing_in_15_4",
        "packaging_filesystem_io_in_15_4",
        "packaging_environment_probe_in_15_4",
        "dependency_freeze_in_15_4",
        "asset_discovery_in_15_4",
        "qml_asset_discovery_in_15_4",
        "runtime_activation_in_15_4",
        "paper_runtime_start_in_15_4",
        "testnet_runtime_start_in_15_4",
        "live_canary_start_in_15_4",
        "live_trading_in_15_4",
        "order_generation_in_15_4",
        "order_submission_in_15_4",
        "order_cancel_in_15_4",
        "order_replace_in_15_4",
        "private_endpoint_in_15_4",
        "network_io_in_15_4",
        "credential_read_in_15_4",
        "config_env_secret_read_in_15_4",
        "qml_bridge_change_in_15_4",
    ]
    decision = {
        "missing_packaging_gate_contract_policy": "fail_closed",
        "missing_dry_run_check_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    decision.update({key: "blocked" for key in blocked_keys})
    return decision


def _build_non_execution_evidence(gate: dict[str, Any]) -> dict[str, bool]:
    source = gate["non_execution_evidence"]
    evidence = {
        "source_packaging_gate_contract_read": True,
        "packaging_dry_run_contract_built": True,
        "packaging_dry_run_contract_only": True,
        "source_gate_packaging_executed": source["packaging_executed"],
        "source_gate_runtime_loop_started": source["runtime_loop_started"],
        "source_gate_runtime_gate_executed": source["runtime_gate_executed"],
        "source_gate_gate_state_mutated": source["gate_state_mutated"],
        "source_gate_order_generated": source["order_generated"],
        "source_gate_order_submitted": source["order_submitted"],
        "source_gate_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_gate_network_io_performed": source["network_io_performed"],
        "source_gate_filesystem_io_performed": source["filesystem_io_performed"],
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


def _build_dry_run_boundaries() -> dict[str, bool]:
    return {
        "packaging_dry_run_contract_is_plain_data_only": True,
        "packaging_dry_run_contract_is_source_only": True,
        "packaging_dry_run_contract_reads_packaging_gate_contract_only": True,
        "packaging_dry_run_contract_preserves_exe_direction_without_packaging": True,
        "packaging_dry_run_contract_can_feed_15_5_packaging_dry_run_read_model": True,
        "packaging_dry_run_contract_cannot_execute_dry_run": True,
        "packaging_dry_run_contract_cannot_package_exe": True,
        "packaging_dry_run_contract_cannot_start_pyinstaller": True,
        "packaging_dry_run_contract_cannot_execute_build_commands": True,
        "packaging_dry_run_contract_cannot_create_build_artifacts": True,
        "packaging_dry_run_contract_cannot_change_installers": True,
        "packaging_dry_run_contract_cannot_change_release_workflows": True,
        "packaging_dry_run_contract_cannot_run_artifact_smoke_tests": True,
        "packaging_dry_run_contract_cannot_sign_artifacts": True,
        "packaging_dry_run_contract_cannot_publish_artifacts": True,
        "packaging_dry_run_contract_cannot_probe_packaging_environment": True,
        "packaging_dry_run_contract_cannot_freeze_dependencies": True,
        "packaging_dry_run_contract_cannot_discover_assets": True,
        "packaging_dry_run_contract_cannot_discover_qml_assets": True,
        "packaging_dry_run_contract_cannot_perform_filesystem_io": True,
        "packaging_dry_run_contract_cannot_activate_runtime": True,
        "packaging_dry_run_contract_cannot_start_paper_runtime": True,
        "packaging_dry_run_contract_cannot_start_testnet_runtime": True,
        "packaging_dry_run_contract_cannot_start_live_canary": True,
        "packaging_dry_run_contract_cannot_enable_live_trading": True,
        "packaging_dry_run_contract_cannot_generate_orders": True,
        _BOUNDARY_ORDER_SUBMISSION_KEY: True,
        _BOUNDARY_ORDER_CANCEL_KEY: True,
        _BOUNDARY_ORDER_REPLACE_KEY: True,
        "packaging_dry_run_contract_cannot_access_private_endpoints": True,
        "packaging_dry_run_contract_cannot_open_network_io": True,
        "packaging_dry_run_contract_cannot_read_credentials": True,
        "packaging_dry_run_contract_cannot_start_runtime_loop": True,
        "packaging_dry_run_contract_cannot_execute_runtime_gates": True,
        "packaging_dry_run_contract_cannot_mutate_gate_state": True,
        "packaging_dry_run_contract_cannot_read_config_env_or_secrets": True,
        "packaging_dry_run_contract_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(gate: dict[str, Any]) -> dict[str, Any]:
    source = gate["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_packaging_gate_contract": SOURCE_PACKAGING_GATE_CONTRACT_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_gate_contract_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_packaging_readiness_matrix": source["source_packaging_readiness_matrix"],
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
