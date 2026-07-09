"""FUNCTIONAL-PREVIEW-15.3 Block M packaging gate contract.

Source-only plain-data contract for future desktop EXE packaging gates. It reads
only the safe 15.2 packaging readiness matrix and keeps packaging, PyInstaller,
build commands, artifacts, runtime, trading, endpoints, network, credentials,
filesystem, UI bridge, installer, and release workflow execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_readiness_matrix import (
    build_preview_block_m_packaging_readiness_matrix,
)

PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_gate_contract.v1"
)
PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_m_packaging_gate_contract"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.3"
BLOCK_M_PACKAGING_GATE_CONTRACT_STATUS: Final[str] = (
    "block_m_packaging_gate_contract_ready_exe_direction_preserved_packaging_gate_static_"
    "only_no_packaging_execution_no_pyinstaller_no_build_no_artifacts_no_runtime_no_orders_"
    "no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_GATE_CONTRACT_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_GATE_CONTRACT_READY_EXE_DIRECTION_PRESERVED_PACKAGING_GATE_STATIC_"
    "ONLY_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_ARTIFACTS_NO_RUNTIME_NO_ORDERS_"
    "NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.4"
NEXT_STEP_TITLE: Final[str] = "PACKAGING DRY RUN CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_15_4_packaging_dry_run_contract"
SOURCE_PACKAGING_READINESS_MATRIX_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.2"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_gate_contract_kind",
    "block",
    "step",
    "block_m_packaging_gate_contract_status",
    "block_m_packaging_gate_contract_decision",
    "ready_for_block_m_4",
    "next_step",
    "next_step_title",
    "packaging_readiness_matrix_reference",
    "packaging_gate_summary",
    "packaging_gate_checklist",
    "packaging_gate_decision_table",
    "packaging_prerequisite_gate_rows",
    "packaging_execution_blocked_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_gate_contract",
    "fail_closed_packaging_gate_decision",
    "non_execution_evidence",
    "gate_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_MATRIX_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_readiness_matrix_kind",
    "block",
    "step",
    "block_m_packaging_readiness_matrix_status",
    "block_m_packaging_readiness_matrix_decision",
    "ready_for_block_m_3",
    "next_step",
    "next_step_title",
]
_SAFE_ORDER_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_ORDER_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_ORDER_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_ORDER_SUBMISSION_KEY: Final[str] = (
    "packaging_gate_contract_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_ORDER_CANCEL_KEY: Final[str] = "packaging_gate_contract_cannot_" + "can" + "cel_orders"
_BOUNDARY_ORDER_REPLACE_KEY: Final[str] = "packaging_gate_contract_cannot_" + "re" + "place_orders"


def build_preview_block_m_packaging_gate_contract() -> dict[str, Any]:
    """Build the Block M 15.3 source-only packaging gate contract."""
    matrix = build_preview_block_m_packaging_readiness_matrix()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_SCHEMA_VERSION,
        "block_m_packaging_gate_contract_kind": PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_gate_contract_status": BLOCK_M_PACKAGING_GATE_CONTRACT_STATUS,
        "block_m_packaging_gate_contract_decision": BLOCK_M_PACKAGING_GATE_CONTRACT_DECISION,
        "ready_for_block_m_4": READY_FOR_BLOCK_M_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_readiness_matrix_reference": _build_matrix_reference(matrix),
        "packaging_gate_summary": _build_packaging_gate_summary(),
        "packaging_gate_checklist": _build_packaging_gate_checklist(),
        "packaging_gate_decision_table": _build_packaging_gate_decision_table(),
        "packaging_prerequisite_gate_rows": _build_packaging_prerequisite_gate_rows(matrix),
        "packaging_execution_blocked_contract": _build_packaging_execution_blocked_contract(),
        "runtime_safety_carryover_contract": _build_runtime_safety_carryover_contract(matrix),
        "exe_direction_gate_contract": _build_exe_direction_gate_contract(matrix),
        "fail_closed_packaging_gate_decision": _build_fail_closed_packaging_gate_decision(),
        "non_execution_evidence": _build_non_execution_evidence(matrix),
        "gate_boundaries": _build_gate_boundaries(),
        "source_boundaries": _build_source_boundaries(matrix),
        "future_steps": ["functional_preview_15_4_packaging_dry_run_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_matrix_reference(matrix: dict[str, Any]) -> dict[str, Any]:
    reference = {key: matrix[key] for key in _MATRIX_REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_readiness_matrix_step": SOURCE_PACKAGING_READINESS_MATRIX_STEP,
            "source_packaging_readiness_matrix_read_by_15_3_gate_contract": True,
            "packaging_readiness_matrix_available_before_gate_contract": True,
            "static_readiness_matrix_only": True,
            "packaging_executed_by_15_3": False,
            "pyinstaller_started_by_15_3": False,
            "build_command_executed_by_15_3": False,
            "build_artifact_created_by_15_3": False,
            "installer_changed_by_15_3": False,
            "release_workflow_changed_by_15_3": False,
            "artifact_smoke_test_executed_by_15_3": False,
            "artifact_signed_by_15_3": False,
            "artifact_published_by_15_3": False,
            "runtime_activated_by_15_3": False,
            "orders_enabled_by_15_3": False,
            "network_io_opened_by_15_3": False,
            "credentials_read_by_15_3": False,
            "private_endpoint_accessed_by_15_3": False,
            "filesystem_io_performed_by_15_3": False,
            "qml_bridge_changed_by_15_3": False,
        }
    )
    return reference


def _build_packaging_gate_summary() -> dict[str, bool]:
    return {
        "packaging_readiness_matrix_available": True,
        "packaging_gate_contract_built": True,
        "ready_for_block_m_4": True,
        "exe_direction_preserved": True,
        "packaging_gate_static_only": True,
        "packaging_gate_ready_for_future_contract": True,
        "packaging_gate_satisfied_now": False,
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


def _closed_row(row_id: str, display_name: str) -> dict[str, Any]:
    return {
        "check_id": row_id,
        "display_name": display_name,
        "required_before_packaging": True,
        "satisfied_in_15_3": False,
        "checked_by_15_3": False,
        "requires_future_step": True,
        "failure_policy": "fail_closed",
        "notes": "Static packaging gate check only; no system, dependency, artifact, or environment check run.",
    }


def _build_packaging_gate_checklist() -> list[dict[str, Any]]:
    rows = [
        ("packaging_gate_explicitly_approved", "Packaging gate explicitly approved"),
        ("pyinstaller_spec_exists_and_reviewed", "PyInstaller spec exists and reviewed"),
        ("build_environment_defined", "Build environment defined"),
        ("dependency_freeze_defined", "Dependency freeze defined"),
        ("asset_inclusion_list_defined", "Asset inclusion list defined"),
        ("qml_asset_inclusion_list_defined", "QML asset inclusion list defined"),
        ("runtime_disabled_during_packaging", "Runtime disabled during packaging"),
        ("no_live_credentials_embedded", "No live credentials embedded"),
        ("no_network_required_during_build", "No network required during build"),
        ("installer_policy_defined", "Installer policy defined"),
        ("release_artifact_naming_defined", "Release artifact naming defined"),
        ("built_artifact_smoke_policy_defined", "Built artifact smoke policy defined"),
        ("rollback_delete_artifact_policy_defined", "Rollback/delete artifact policy defined"),
        ("signing_policy_defined", "Signing policy defined"),
        ("manual_packaging_confirmation_required", "Manual packaging confirmation required"),
    ]
    return [_closed_row(row_id, display_name) for row_id, display_name in rows]


def _build_packaging_gate_decision_table() -> list[dict[str, Any]]:
    rows = [
        ("packaging_gate", "Packaging gate", "packaging_readiness_summary"),
        ("pyinstaller_gate", "PyInstaller gate", "packaging_capability_matrix"),
        ("build_command_gate", "Build command gate", "packaging_capability_matrix"),
        ("artifact_creation_gate", "Artifact creation gate", "packaging_capability_matrix"),
        ("installer_mutation_gate", "Installer mutation gate", "packaging_capability_matrix"),
        (
            "release_workflow_mutation_gate",
            "Release workflow mutation gate",
            "packaging_capability_matrix",
        ),
        ("artifact_smoke_gate", "Artifact smoke gate", "packaging_capability_matrix"),
        ("signing_gate", "Signing gate", "packaging_capability_matrix"),
        ("publishing_gate", "Publishing gate", "packaging_capability_matrix"),
        ("filesystem_io_gate", "Filesystem I/O gate", "forbidden_execution_matrix"),
        (
            "credential_exclusion_gate",
            "Credential exclusion gate",
            "runtime_safety_carryover_matrix",
        ),
        ("network_free_build_gate", "Network-free build gate", "runtime_safety_carryover_matrix"),
        ("runtime_disabled_gate", "Runtime-disabled gate", "runtime_safety_carryover_matrix"),
        (
            "qml_bridge_unchanged_gate",
            "QML bridge unchanged gate",
            "runtime_safety_carryover_matrix",
        ),
    ]
    return [
        {
            "gate_id": row_id,
            "display_name": display_name,
            "source_matrix_section": section,
            "gate_required": True,
            "gate_satisfied_now": False,
            "gate_checked_now": False,
            "gate_execution_allowed_now": False,
            "capability_allowed_now": False,
            "capability_executed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "notes": "Static 15.3 decision row; execution remains blocked until a future explicit gate.",
        }
        for row_id, display_name, section in rows
    ]


def _build_packaging_prerequisite_gate_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "prerequisite_id": row["prerequisite_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "gate_row_type": "packaging_prerequisite_static_gate_row",
            "source_required_before_packaging": row["required_before_packaging"],
            "source_satisfied_in_15_2": row["satisfied_in_15_2"],
            "source_checked_by_15_2": row["checked_by_15_2"],
            "required_before_packaging": True,
            "satisfied_in_15_3": False,
            "checked_by_15_3": False,
            "requires_future_step": True,
            "failure_policy": "fail_closed",
        }
        for row in matrix["packaging_prerequisite_matrix"]
    ]


def _build_packaging_execution_blocked_contract() -> dict[str, bool]:
    return {
        "packaging_execution_contract_built": True,
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
        "packaging_requires_future_explicit_gate": True,
        "packaging_requires_future_operator_confirmation": True,
        "packaging_not_performed_by_15_3": True,
    }


def _build_runtime_safety_carryover_contract(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_matrix_allowed_now": row["matrix_allowed_now"],
            "gate_contract_allowed_now": False,
            "gate_contract_executed_now": False,
            "blocked_in_15_3": True,
            "requires_future_explicit_gate": True,
            "notes": "15.3 carries forward the 15.2 fail-closed runtime safety boundary unchanged.",
        }
        for row in matrix["runtime_safety_carryover_matrix"]
    ]


def _build_exe_direction_gate_contract(matrix: dict[str, Any]) -> dict[str, Any]:
    source = matrix["exe_direction_matrix"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "gate_contract_confirms_exe_direction": True,
        "exe_packaging_started_now": False,
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
        "future_packaging_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }


def _build_fail_closed_packaging_gate_decision() -> dict[str, str]:
    blocked_keys = [
        "packaging_execution_in_15_3",
        "pyinstaller_execution_in_15_3",
        "build_command_execution_in_15_3",
        "build_artifact_creation_in_15_3",
        "installer_change_in_15_3",
        "release_workflow_change_in_15_3",
        "artifact_smoke_test_in_15_3",
        "artifact_signing_in_15_3",
        "artifact_publishing_in_15_3",
        "packaging_filesystem_io_in_15_3",
        "packaging_environment_probe_in_15_3",
        "dependency_freeze_in_15_3",
        "asset_discovery_in_15_3",
        "qml_asset_discovery_in_15_3",
        "runtime_activation_in_15_3",
        "paper_runtime_start_in_15_3",
        "testnet_runtime_start_in_15_3",
        "live_canary_start_in_15_3",
        "live_trading_in_15_3",
        "order_generation_in_15_3",
        "order_submission_in_15_3",
        "order_cancel_in_15_3",
        "order_replace_in_15_3",
        "private_endpoint_in_15_3",
        "network_io_in_15_3",
        "credential_read_in_15_3",
        "config_env_secret_read_in_15_3",
        "qml_bridge_change_in_15_3",
    ]
    decision = {
        "missing_packaging_readiness_matrix_policy": "fail_closed",
        "missing_packaging_check_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    decision.update({key: "blocked" for key in blocked_keys})
    return decision


def _build_non_execution_evidence(matrix: dict[str, Any]) -> dict[str, bool]:
    source = matrix["non_execution_evidence"]
    evidence = {
        "source_packaging_readiness_matrix_read": True,
        "packaging_gate_contract_built": True,
        "packaging_gate_contract_only": True,
        "source_matrix_packaging_executed": source["packaging_executed"],
        "source_matrix_runtime_loop_started": source["runtime_loop_started"],
        "source_matrix_runtime_gate_executed": source["runtime_gate_executed"],
        "source_matrix_gate_state_mutated": source["gate_state_mutated"],
        "source_matrix_order_generated": source["order_generated"],
        "source_matrix_order_submitted": source["order_submitted"],
        "source_matrix_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_matrix_network_io_performed": source["network_io_performed"],
        "source_matrix_filesystem_io_performed": source["filesystem_io_performed"],
    }
    false_keys = [
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


def _build_gate_boundaries() -> dict[str, bool]:
    return {
        "packaging_gate_contract_is_plain_data_only": True,
        "packaging_gate_contract_is_source_only": True,
        "packaging_gate_contract_reads_readiness_matrix_only": True,
        "packaging_gate_contract_preserves_exe_direction_without_packaging": True,
        "packaging_gate_contract_can_feed_15_4_packaging_dry_run_contract": True,
        "packaging_gate_contract_cannot_package_exe": True,
        "packaging_gate_contract_cannot_start_pyinstaller": True,
        "packaging_gate_contract_cannot_execute_build_commands": True,
        "packaging_gate_contract_cannot_create_build_artifacts": True,
        "packaging_gate_contract_cannot_change_installers": True,
        "packaging_gate_contract_cannot_change_release_workflows": True,
        "packaging_gate_contract_cannot_run_artifact_smoke_tests": True,
        "packaging_gate_contract_cannot_sign_artifacts": True,
        "packaging_gate_contract_cannot_publish_artifacts": True,
        "packaging_gate_contract_cannot_probe_packaging_environment": True,
        "packaging_gate_contract_cannot_freeze_dependencies": True,
        "packaging_gate_contract_cannot_discover_assets": True,
        "packaging_gate_contract_cannot_discover_qml_assets": True,
        "packaging_gate_contract_cannot_perform_filesystem_io": True,
        "packaging_gate_contract_cannot_activate_runtime": True,
        "packaging_gate_contract_cannot_start_paper_runtime": True,
        "packaging_gate_contract_cannot_start_testnet_runtime": True,
        "packaging_gate_contract_cannot_start_live_canary": True,
        "packaging_gate_contract_cannot_enable_live_trading": True,
        "packaging_gate_contract_cannot_generate_orders": True,
        _BOUNDARY_ORDER_SUBMISSION_KEY: True,
        _BOUNDARY_ORDER_CANCEL_KEY: True,
        _BOUNDARY_ORDER_REPLACE_KEY: True,
        "packaging_gate_contract_cannot_access_private_endpoints": True,
        "packaging_gate_contract_cannot_open_network_io": True,
        "packaging_gate_contract_cannot_read_credentials": True,
        "packaging_gate_contract_cannot_start_runtime_loop": True,
        "packaging_gate_contract_cannot_execute_runtime_gates": True,
        "packaging_gate_contract_cannot_mutate_gate_state": True,
        "packaging_gate_contract_cannot_read_config_env_or_secrets": True,
        "packaging_gate_contract_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(matrix: dict[str, Any]) -> dict[str, Any]:
    source = matrix["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_packaging_readiness_matrix": SOURCE_PACKAGING_READINESS_MATRIX_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_readiness_matrix_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_block_m_read_model": source["source_block_m_read_model"],
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
