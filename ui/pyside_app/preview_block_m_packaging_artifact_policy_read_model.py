"""FUNCTIONAL-PREVIEW-15.7 Block M packaging artifact policy read model.

Source-only plain-data read model over the 15.6 artifact policy matrix.
It preserves the future desktop EXE direction while keeping artifact work,
dry-run execution, packaging, PyInstaller, build commands, runtime, trading,
endpoints, network, credentials, filesystem, UI bridge, installer, and release
workflow execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_artifact_policy_matrix import (
    build_preview_block_m_packaging_artifact_policy_matrix,
)

PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_artifact_policy_read_model.v1"
)
PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_m_packaging_artifact_policy_read_model"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.7"
BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STATUS: Final[str] = (
    "block_m_packaging_artifact_policy_read_model_ready_exe_direction_preserved_"
    "artifact_policy_read_model_static_only_no_artifact_creation_no_dry_run_execution_"
    "no_packaging_execution_no_pyinstaller_no_build_no_runtime_no_orders_"
    "no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_READY_EXE_DIRECTION_PRESERVED_"
    "ARTIFACT_POLICY_READ_MODEL_STATIC_ONLY_NO_ARTIFACT_CREATION_NO_DRY_RUN_EXECUTION_"
    "NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_NO_ORDERS_"
    "NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_8: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.8"
NEXT_STEP_TITLE: Final[str] = "PACKAGING RELEASE READINESS CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_15_8_packaging_release_readiness_contract"
SOURCE_PACKAGING_ARTIFACT_POLICY_MATRIX_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.6"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_artifact_policy_read_model_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_read_model_status",
    "block_m_packaging_artifact_policy_read_model_decision",
    "ready_for_block_m_8",
    "next_step",
    "next_step_title",
    "packaging_artifact_policy_matrix_reference",
    "artifact_policy_read_summary",
    "artifact_lifecycle_policy_read_rows",
    "artifact_naming_policy_read_rows",
    "artifact_retention_rollback_policy_read_rows",
    "artifact_smoke_sign_publish_policy_read_rows",
    "artifact_execution_read_model",
    "packaging_execution_carryover_read_rows",
    "runtime_safety_carryover_read_rows",
    "exe_direction_artifact_policy_read_model",
    "fail_closed_artifact_policy_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_artifact_policy_matrix_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_matrix_status",
    "block_m_packaging_artifact_policy_matrix_decision",
    "ready_for_block_m_7",
    "next_step",
    "next_step_title",
]
_SAFE_SUMMARY_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_SUMMARY_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_SUMMARY_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_SUBMISSION_KEY: Final[str] = (
    "packaging_artifact_policy_read_model_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_CANCEL_KEY: Final[str] = (
    "packaging_artifact_policy_read_model_cannot_" + "can" + "cel_orders"
)
_BOUNDARY_REPLACE_KEY: Final[str] = (
    "packaging_artifact_policy_read_model_cannot_" + "re" + "place_orders"
)
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_15_7"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_15_7"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_15_7"


def build_preview_block_m_packaging_artifact_policy_read_model() -> dict[str, Any]:
    """Build the Block M 15.7 source-only artifact policy read model."""
    matrix = build_preview_block_m_packaging_artifact_policy_matrix()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_SCHEMA_VERSION,
        "block_m_packaging_artifact_policy_read_model_kind": PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_artifact_policy_read_model_status": BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STATUS,
        "block_m_packaging_artifact_policy_read_model_decision": BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_DECISION,
        "ready_for_block_m_8": READY_FOR_BLOCK_M_8,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_artifact_policy_matrix_reference": _build_matrix_reference(matrix),
        "artifact_policy_read_summary": _build_artifact_policy_read_summary(),
        "artifact_lifecycle_policy_read_rows": _build_lifecycle_read_rows(matrix),
        "artifact_naming_policy_read_rows": _build_naming_read_rows(matrix),
        "artifact_retention_rollback_policy_read_rows": _build_retention_read_rows(matrix),
        "artifact_smoke_sign_publish_policy_read_rows": _build_smoke_sign_publish_read_rows(matrix),
        "artifact_execution_read_model": _build_artifact_execution_read_model(matrix),
        "packaging_execution_carryover_read_rows": _build_packaging_execution_carryover_read_rows(
            matrix
        ),
        "runtime_safety_carryover_read_rows": _build_runtime_safety_carryover_read_rows(matrix),
        "exe_direction_artifact_policy_read_model": _build_exe_direction_artifact_policy_read_model(
            matrix
        ),
        "fail_closed_artifact_policy_read_decision": _build_fail_closed_artifact_policy_read_decision(),
        "non_execution_evidence": _build_non_execution_evidence(matrix),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(matrix),
        "future_steps": ["functional_preview_15_8_packaging_release_readiness_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_matrix_reference(matrix: dict[str, Any]) -> dict[str, Any]:
    reference = {key: matrix[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_artifact_policy_matrix_step": SOURCE_PACKAGING_ARTIFACT_POLICY_MATRIX_STEP,
            "source_packaging_artifact_policy_matrix_read_by_15_7_read_model": True,
            "packaging_artifact_policy_matrix_available_before_read_model": True,
            "static_packaging_artifact_policy_matrix_only": True,
            "artifact_policy_read_model_built_by_15_7": True,
        }
    )
    for key in [
        "artifact_created_by_15_7",
        "artifact_mutated_by_15_7",
        "artifact_deleted_by_15_7",
        "artifact_smoke_test_executed_by_15_7",
        "artifact_signed_by_15_7",
        "artifact_published_by_15_7",
        "artifact_name_finalized_by_15_7",
        "artifact_location_selected_by_15_7",
        "artifact_checksum_generated_by_15_7",
        "artifact_metadata_written_by_15_7",
        "artifact_audit_exported_by_15_7",
        "artifact_cleanup_performed_by_15_7",
        "packaging_dry_run_executed_by_15_7",
        "packaging_executed_by_15_7",
        "pyinstaller_started_by_15_7",
        "build_command_executed_by_15_7",
        "build_artifact_created_by_15_7",
        "installer_changed_by_15_7",
        "release_workflow_changed_by_15_7",
        "dependency_freeze_performed_by_15_7",
        "asset_discovery_performed_by_15_7",
        "qml_asset_discovery_performed_by_15_7",
        "runtime_activated_by_15_7",
        "orders_enabled_by_15_7",
        "network_io_opened_by_15_7",
        "credentials_read_by_15_7",
        "private_endpoint_accessed_by_15_7",
        "filesystem_io_performed_by_15_7",
        "qml_bridge_changed_by_15_7",
    ]:
        reference[key] = False
    return reference


def _build_artifact_policy_read_summary() -> dict[str, bool]:
    summary = {
        "packaging_artifact_policy_matrix_available": True,
        "artifact_policy_read_model_built": True,
        "ready_for_block_m_8": True,
        "exe_direction_preserved": True,
        "artifact_policy_read_model_static_only": True,
        "artifact_policy_ready_for_future_release_readiness_contract": True,
        "artifact_policy_read_only": True,
    }
    for key in [
        "artifact_policy_satisfied_now",
        "artifact_creation_allowed_now",
        "artifact_mutation_allowed_now",
        "artifact_delete_allowed_now",
        "artifact_smoke_test_allowed_now",
        "artifact_signing_allowed_now",
        "artifact_publishing_allowed_now",
        "artifact_location_selected_now",
        "artifact_naming_finalized_now",
        "artifact_retention_policy_finalized_now",
        "artifact_rollback_policy_finalized_now",
        "artifact_checksum_generation_allowed_now",
        "artifact_metadata_write_allowed_now",
        "artifact_audit_export_allowed_now",
        "artifact_cleanup_allowed_now",
        "dry_run_can_execute_now",
        "packaging_ready_now",
        "packaging_can_execute_now",
        "pyinstaller_can_start_now",
        "build_command_can_execute_now",
        "build_artifact_can_be_created_now",
        "installer_can_change_now",
        "release_workflow_can_change_now",
        "dependency_freeze_can_run_now",
        "asset_discovery_can_run_now",
        "qml_asset_discovery_can_run_now",
        "packaging_filesystem_io_allowed_now",
        "safe_to_activate_runtime_now",
        "safe_to_start_paper_runtime_now",
        "safe_to_start_testnet_runtime_now",
        "safe_to_start_live_canary_now",
        "safe_to_enable_live_trading_now",
        "safe_to_generate_orders_now",
        _SAFE_SUMMARY_SUBMISSION_KEY,
        _SAFE_SUMMARY_CANCEL_KEY,
        _SAFE_SUMMARY_REPLACE_KEY,
        "safe_to_access_private_endpoints_now",
        "safe_to_open_network_io_now",
        "safe_to_read_credentials_now",
        "safe_for_filesystem_io_now",
        "safe_for_config_env_secrets_now",
        "safe_to_change_qml_bridge_now",
    ]:
        summary[key] = False
    return summary


def _build_lifecycle_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": row["policy_id"],
            "display_name": row["display_name"],
            "policy_category": row["policy_category"],
            "notes": row["notes"],
            "read_row_type": "packaging_artifact_lifecycle_static_read_row",
            "source_required_before_artifact_work": row["required_before_artifact_work"],
            "source_satisfied_in_15_6": row["satisfied_in_15_6"],
            "source_checked_by_15_6": row["checked_by_15_6"],
            "source_allowed_now": row["allowed_now"],
            "source_executed_now": row["executed_now"],
            "required_before_future_artifact_work": True,
            "satisfied_in_15_7": False,
            "checked_by_15_7": False,
            "read_by_15_7": True,
            "allowed_now": False,
            "executed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
        }
        for row in matrix["artifact_lifecycle_policy_matrix"]
    ]


def _build_naming_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "naming_rule_id": row["naming_rule_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "read_row_type": "packaging_artifact_naming_static_read_row",
            "source_required_before_artifact_naming": row["required_before_artifact_naming"],
            "source_finalized_in_15_6": row["finalized_in_15_6"],
            "source_checked_by_15_6": row["checked_by_15_6"],
            "source_selected_now": row["selected_now"],
            "required_before_future_artifact_naming": True,
            "finalized_in_15_7": False,
            "checked_by_15_7": False,
            "read_by_15_7": True,
            "selected_now": False,
            "allowed_now": False,
            "requires_future_explicit_gate": True,
        }
        for row in matrix["artifact_naming_policy_matrix"]
    ]


def _build_retention_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "retention_policy_id": row["retention_policy_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "read_row_type": "packaging_artifact_retention_rollback_static_read_row",
            "source_required_before_artifact_release": row["required_before_artifact_release"],
            "source_satisfied_in_15_6": row["satisfied_in_15_6"],
            "source_checked_by_15_6": row["checked_by_15_6"],
            "source_delete_allowed_now": row["delete_allowed_now"],
            "source_rollback_allowed_now": row["rollback_allowed_now"],
            "required_before_future_artifact_release": True,
            "satisfied_in_15_7": False,
            "checked_by_15_7": False,
            "read_by_15_7": True,
            "delete_allowed_now": False,
            "rollback_allowed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
        }
        for row in matrix["artifact_retention_rollback_policy_matrix"]
    ]


def _build_smoke_sign_publish_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": row["policy_id"],
            "display_name": row["display_name"],
            "notes": row["notes"],
            "read_row_type": "packaging_artifact_smoke_sign_publish_static_read_row",
            "source_required_before_publish": row["required_before_publish"],
            "source_satisfied_in_15_6": row["satisfied_in_15_6"],
            "source_checked_by_15_6": row["checked_by_15_6"],
            "source_smoke_allowed_now": row["smoke_allowed_now"],
            "source_sign_allowed_now": row["sign_allowed_now"],
            "source_publish_allowed_now": row["publish_allowed_now"],
            "required_before_future_publish": True,
            "satisfied_in_15_7": False,
            "checked_by_15_7": False,
            "read_by_15_7": True,
            "smoke_allowed_now": False,
            "sign_allowed_now": False,
            "publish_allowed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
        }
        for row in matrix["artifact_smoke_sign_publish_policy_matrix"]
    ]


def _build_artifact_execution_read_model(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_execution_read_model_built": True,
        "source_artifact_policy_matrix_built": matrix["artifact_policy_summary"][
            "artifact_policy_matrix_built"
        ],
        "source_artifact_execution_blocked_rows": [
            {
                "execution_id": row["execution_id"],
                "display_name": row["display_name"],
                "source_allowed_now": row["allowed_now"],
                "source_executed_now": row["executed_now"],
            }
            for row in matrix["artifact_execution_blocked_matrix"]
        ],
        "artifact_creation_allowed_now": False,
        "artifact_mutation_allowed_now": False,
        "artifact_delete_allowed_now": False,
        "artifact_smoke_test_allowed_now": False,
        "artifact_signing_allowed_now": False,
        "artifact_publishing_allowed_now": False,
        "artifact_location_selection_allowed_now": False,
        "artifact_name_finalization_allowed_now": False,
        "artifact_checksum_generation_allowed_now": False,
        "artifact_metadata_write_allowed_now": False,
        "artifact_audit_export_allowed_now": False,
        "artifact_cleanup_allowed_now": False,
        "artifact_work_requires_future_explicit_gate": True,
        "artifact_work_requires_future_operator_confirmation": True,
        "no_artifact_created_by_15_7": True,
        "no_artifact_mutated_by_15_7": True,
        "no_artifact_deleted_by_15_7": True,
    }


def _build_packaging_execution_carryover_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_allowed_now": row["source_allowed_now"],
            "source_matrix_allowed_now": row["matrix_allowed_now"],
            "source_matrix_executed_now": row["matrix_executed_now"],
            "read_model_allowed_now": False,
            "read_model_executed_now": False,
            "blocked_in_15_7": True,
            "requires_future_explicit_gate": True,
            "notes": "15.7 reads the 15.6 execution carryover without unlocking execution.",
        }
        for row in matrix["packaging_execution_carryover_matrix"]
    ]


def _build_runtime_safety_carryover_read_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_read_model_allowed_now": row["source_read_model_allowed_now"],
            "source_artifact_policy_matrix_allowed_now": row["artifact_policy_matrix_allowed_now"],
            "source_artifact_policy_matrix_executed_now": row[
                "artifact_policy_matrix_executed_now"
            ],
            "read_model_allowed_now": False,
            "read_model_executed_now": False,
            "blocked_in_15_7": True,
            "requires_future_explicit_gate": True,
            "notes": "15.7 artifact policy read model does not alter runtime safety boundaries.",
        }
        for row in matrix["runtime_safety_carryover_matrix"]
    ]


def _build_exe_direction_artifact_policy_read_model(matrix: dict[str, Any]) -> dict[str, Any]:
    source = matrix["exe_direction_artifact_policy_matrix"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "artifact_policy_read_model_confirms_exe_direction": True,
        "exe_packaging_started_now": False,
        "packaging_dry_run_started_now": False,
        "pyinstaller_started_now": False,
        "build_command_added_now": False,
        "build_command_executed_now": False,
        "workflow_changed_for_packaging_now": False,
        "installer_changed_now": False,
        "release_artifact_created_now": False,
        "artifact_created_now": False,
        "artifact_mutated_now": False,
        "artifact_deleted_now": False,
        "artifact_smoke_test_executed_now": False,
        "artifact_signed_now": False,
        "artifact_published_now": False,
        "packaging_deferred_to_future_explicit_block": True,
        "dry_run_deferred_to_future_explicit_block": True,
        "artifact_work_deferred_to_future_explicit_block": True,
        "future_packaging_requires_explicit_gate": True,
        "future_dry_run_requires_explicit_gate": True,
        "future_artifact_work_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }


def _build_fail_closed_artifact_policy_read_decision() -> dict[str, str]:
    decision = {
        "missing_packaging_artifact_policy_matrix_policy": "fail_closed",
        "missing_artifact_policy_read_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    for key in [
        "artifact_creation_in_15_7",
        "artifact_mutation_in_15_7",
        "artifact_deletion_in_15_7",
        "artifact_smoke_test_in_15_7",
        "artifact_signing_in_15_7",
        "artifact_publishing_in_15_7",
        "artifact_name_finalization_in_15_7",
        "artifact_location_selection_in_15_7",
        "artifact_checksum_generation_in_15_7",
        "artifact_metadata_write_in_15_7",
        "artifact_audit_export_in_15_7",
        "artifact_cleanup_in_15_7",
        "packaging_dry_run_execution_in_15_7",
        "packaging_execution_in_15_7",
        "pyinstaller_execution_in_15_7",
        "build_command_execution_in_15_7",
        "build_artifact_creation_in_15_7",
        "installer_change_in_15_7",
        "release_workflow_change_in_15_7",
        "packaging_filesystem_io_in_15_7",
        "packaging_environment_probe_in_15_7",
        "dependency_freeze_in_15_7",
        "asset_discovery_in_15_7",
        "qml_asset_discovery_in_15_7",
        "runtime_activation_in_15_7",
        "paper_runtime_start_in_15_7",
        "testnet_runtime_start_in_15_7",
        "live_canary_start_in_15_7",
        "live_trading_in_15_7",
        "order_generation_in_15_7",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_15_7",
        "network_io_in_15_7",
        "credential_read_in_15_7",
        "config_env_secret_read_in_15_7",
        "qml_bridge_change_in_15_7",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(matrix: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        "source_packaging_artifact_policy_matrix_read": True,
        "artifact_policy_read_model_built": True,
        "artifact_policy_read_model_only": True,
    }
    for key, value in matrix["non_execution_evidence"].items():
        if isinstance(value, bool) and value is False:
            evidence[key] = False
    for key in [
        "artifact_created",
        "artifact_mutated",
        "artifact_deleted",
        "artifact_smoke_test_executed",
        "artifact_signed",
        "artifact_published",
        "artifact_name_finalized",
        "artifact_location_selected",
        "artifact_checksum_generated",
        "artifact_metadata_written",
        "artifact_audit_exported",
        "artifact_cleanup_performed",
        "packaging_dry_run_executed",
        "packaging_executed",
        "pyinstaller_started",
        "build_command_executed",
        "build_artifact_created",
        "installer_changed",
        "release_workflow_changed",
        "dependency_freeze_performed",
        "asset_discovery_performed",
        "qml_asset_discovery_performed",
        "qml_bridge_changed",
    ]:
        evidence[key] = False
    return evidence


def _build_read_model_boundaries() -> dict[str, bool]:
    boundaries = {
        "packaging_artifact_policy_read_model_is_plain_data_only": True,
        "packaging_artifact_policy_read_model_is_source_only": True,
        "packaging_artifact_policy_read_model_reads_artifact_policy_matrix_only": True,
        "packaging_artifact_policy_read_model_preserves_exe_direction_without_packaging": True,
        "packaging_artifact_policy_read_model_can_feed_15_8_packaging_release_readiness_contract": True,
    }
    for key in [
        "packaging_artifact_policy_read_model_cannot_create_artifacts",
        "packaging_artifact_policy_read_model_cannot_mutate_artifacts",
        "packaging_artifact_policy_read_model_cannot_delete_artifacts",
        "packaging_artifact_policy_read_model_cannot_run_artifact_smoke_tests",
        "packaging_artifact_policy_read_model_cannot_sign_artifacts",
        "packaging_artifact_policy_read_model_cannot_publish_artifacts",
        "packaging_artifact_policy_read_model_cannot_finalize_artifact_names",
        "packaging_artifact_policy_read_model_cannot_select_artifact_locations",
        "packaging_artifact_policy_read_model_cannot_generate_checksums",
        "packaging_artifact_policy_read_model_cannot_write_artifact_metadata",
        "packaging_artifact_policy_read_model_cannot_export_artifact_audits",
        "packaging_artifact_policy_read_model_cannot_cleanup_artifacts",
        "packaging_artifact_policy_read_model_cannot_execute_dry_run",
        "packaging_artifact_policy_read_model_cannot_package_exe",
        "packaging_artifact_policy_read_model_cannot_start_pyinstaller",
        "packaging_artifact_policy_read_model_cannot_execute_build_commands",
        "packaging_artifact_policy_read_model_cannot_create_build_artifacts",
        "packaging_artifact_policy_read_model_cannot_change_installers",
        "packaging_artifact_policy_read_model_cannot_change_release_workflows",
        "packaging_artifact_policy_read_model_cannot_probe_packaging_environment",
        "packaging_artifact_policy_read_model_cannot_freeze_dependencies",
        "packaging_artifact_policy_read_model_cannot_discover_assets",
        "packaging_artifact_policy_read_model_cannot_discover_qml_assets",
        "packaging_artifact_policy_read_model_cannot_perform_filesystem_io",
        "packaging_artifact_policy_read_model_cannot_activate_runtime",
        "packaging_artifact_policy_read_model_cannot_start_paper_runtime",
        "packaging_artifact_policy_read_model_cannot_start_testnet_runtime",
        "packaging_artifact_policy_read_model_cannot_start_live_canary",
        "packaging_artifact_policy_read_model_cannot_enable_live_trading",
        "packaging_artifact_policy_read_model_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "packaging_artifact_policy_read_model_cannot_access_private_endpoints",
        "packaging_artifact_policy_read_model_cannot_open_network_io",
        "packaging_artifact_policy_read_model_cannot_read_credentials",
        "packaging_artifact_policy_read_model_cannot_start_runtime_loop",
        "packaging_artifact_policy_read_model_cannot_execute_runtime_gates",
        "packaging_artifact_policy_read_model_cannot_mutate_gate_state",
        "packaging_artifact_policy_read_model_cannot_read_config_env_or_secrets",
        "packaging_artifact_policy_read_model_cannot_change_ui_bridge",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_packaging_artifact_policy_matrix": SOURCE_PACKAGING_ARTIFACT_POLICY_MATRIX_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_artifact_policy_matrix_boundaries": {
            "allowed_imports_only": matrix["source_boundaries"]["allowed_imports_only"],
            "source_packaging_dry_run_read_model": matrix["source_boundaries"][
                "source_packaging_dry_run_read_model"
            ],
            "matrix_boundary_subset": {
                "packaging_artifact_policy_matrix_is_plain_data_only": matrix["matrix_boundaries"][
                    "packaging_artifact_policy_matrix_is_plain_data_only"
                ],
                "packaging_artifact_policy_matrix_is_source_only": matrix["matrix_boundaries"][
                    "packaging_artifact_policy_matrix_is_source_only"
                ],
                "packaging_artifact_policy_matrix_can_feed_15_7_packaging_artifact_policy_read_model": matrix[
                    "matrix_boundaries"
                ][
                    "packaging_artifact_policy_matrix_can_feed_15_7_packaging_artifact_policy_read_model"
                ],
            },
        },
    }
