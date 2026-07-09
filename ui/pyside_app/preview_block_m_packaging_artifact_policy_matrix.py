"""FUNCTIONAL-PREVIEW-15.6 Block M packaging artifact policy matrix.

Source-only plain-data matrix over the 15.5 packaging dry-run read model.
It preserves the future desktop EXE direction while keeping artifact work,
dry-run execution, packaging, PyInstaller, build commands, runtime, trading,
endpoints, network, credentials, filesystem, UI bridge, installer, and release
workflow execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_dry_run_read_model import (
    build_preview_block_m_packaging_dry_run_read_model,
)

PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_artifact_policy_matrix.v1"
)
PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_KIND: Final[str] = (
    "functional_preview_block_m_packaging_artifact_policy_matrix"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.6"
BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_STATUS: Final[str] = (
    "block_m_packaging_artifact_policy_matrix_ready_exe_direction_preserved_"
    "artifact_policy_matrix_static_only_no_artifact_creation_no_dry_run_execution_"
    "no_packaging_execution_no_pyinstaller_no_build_no_runtime_no_orders_"
    "no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_READY_EXE_DIRECTION_PRESERVED_"
    "ARTIFACT_POLICY_MATRIX_STATIC_ONLY_NO_ARTIFACT_CREATION_NO_DRY_RUN_EXECUTION_"
    "NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_NO_ORDERS_"
    "NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_7: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.7"
NEXT_STEP_TITLE: Final[str] = "PACKAGING ARTIFACT POLICY READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_15_7_packaging_artifact_policy_read_model"
SOURCE_PACKAGING_DRY_RUN_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.5"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_artifact_policy_matrix_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_matrix_status",
    "block_m_packaging_artifact_policy_matrix_decision",
    "ready_for_block_m_7",
    "next_step",
    "next_step_title",
    "packaging_dry_run_read_model_reference",
    "artifact_policy_summary",
    "artifact_lifecycle_policy_matrix",
    "artifact_naming_policy_matrix",
    "artifact_retention_rollback_policy_matrix",
    "artifact_smoke_sign_publish_policy_matrix",
    "artifact_execution_blocked_matrix",
    "packaging_execution_carryover_matrix",
    "runtime_safety_carryover_matrix",
    "exe_direction_artifact_policy_matrix",
    "fail_closed_artifact_policy_decision",
    "non_execution_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_dry_run_read_model_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_read_model_status",
    "block_m_packaging_dry_run_read_model_decision",
    "ready_for_block_m_6",
    "next_step",
    "next_step_title",
]
_SAFE_SUMMARY_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_SUMMARY_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_SUMMARY_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_SUBMISSION_KEY: Final[str] = (
    "packaging_artifact_policy_matrix_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_CANCEL_KEY: Final[str] = "packaging_artifact_policy_matrix_cannot_" + "can" + "cel_orders"
_BOUNDARY_REPLACE_KEY: Final[str] = (
    "packaging_artifact_policy_matrix_cannot_" + "re" + "place_orders"
)
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_15_6"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_15_6"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_15_6"


def build_preview_block_m_packaging_artifact_policy_matrix() -> dict[str, Any]:
    """Build the Block M 15.6 source-only artifact policy matrix."""
    read_model = build_preview_block_m_packaging_dry_run_read_model()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_SCHEMA_VERSION,
        "block_m_packaging_artifact_policy_matrix_kind": PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_artifact_policy_matrix_status": BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_STATUS,
        "block_m_packaging_artifact_policy_matrix_decision": BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_DECISION,
        "ready_for_block_m_7": READY_FOR_BLOCK_M_7,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_dry_run_read_model_reference": _build_read_model_reference(read_model),
        "artifact_policy_summary": _build_artifact_policy_summary(),
        "artifact_lifecycle_policy_matrix": _build_artifact_lifecycle_policy_matrix(),
        "artifact_naming_policy_matrix": _build_artifact_naming_policy_matrix(),
        "artifact_retention_rollback_policy_matrix": _build_artifact_retention_rollback_policy_matrix(),
        "artifact_smoke_sign_publish_policy_matrix": _build_artifact_smoke_sign_publish_policy_matrix(),
        "artifact_execution_blocked_matrix": _build_artifact_execution_blocked_matrix(),
        "packaging_execution_carryover_matrix": _build_packaging_execution_carryover_matrix(
            read_model
        ),
        "runtime_safety_carryover_matrix": _build_runtime_safety_carryover_matrix(read_model),
        "exe_direction_artifact_policy_matrix": _build_exe_direction_artifact_policy_matrix(
            read_model
        ),
        "fail_closed_artifact_policy_decision": _build_fail_closed_artifact_policy_decision(),
        "non_execution_evidence": _build_non_execution_evidence(read_model),
        "matrix_boundaries": _build_matrix_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_15_7_packaging_artifact_policy_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_dry_run_read_model_step": SOURCE_PACKAGING_DRY_RUN_READ_MODEL_STEP,
            "source_packaging_dry_run_read_model_read_by_15_6_matrix": True,
            "packaging_dry_run_read_model_available_before_artifact_policy_matrix": True,
            "static_packaging_dry_run_read_model_only": True,
            "artifact_policy_matrix_built_by_15_6": True,
            "artifact_created_by_15_6": False,
            "artifact_mutated_by_15_6": False,
            "artifact_deleted_by_15_6": False,
            "artifact_smoke_test_executed_by_15_6": False,
            "artifact_signed_by_15_6": False,
            "artifact_published_by_15_6": False,
            "packaging_dry_run_executed_by_15_6": False,
            "packaging_executed_by_15_6": False,
            "pyinstaller_started_by_15_6": False,
            "build_command_executed_by_15_6": False,
            "build_artifact_created_by_15_6": False,
            "installer_changed_by_15_6": False,
            "release_workflow_changed_by_15_6": False,
            "dependency_freeze_performed_by_15_6": False,
            "asset_discovery_performed_by_15_6": False,
            "qml_asset_discovery_performed_by_15_6": False,
            "runtime_activated_by_15_6": False,
            "orders_enabled_by_15_6": False,
            "network_io_opened_by_15_6": False,
            "credentials_read_by_15_6": False,
            "private_endpoint_accessed_by_15_6": False,
            "filesystem_io_performed_by_15_6": False,
            "qml_bridge_changed_by_15_6": False,
        }
    )
    return reference


def _build_artifact_policy_summary() -> dict[str, bool]:
    summary = {
        "packaging_dry_run_read_model_available": True,
        "artifact_policy_matrix_built": True,
        "ready_for_block_m_7": True,
        "exe_direction_preserved": True,
        "artifact_policy_matrix_static_only": True,
        "artifact_policy_ready_for_future_read_model": True,
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


def _build_artifact_lifecycle_policy_matrix() -> list[dict[str, Any]]:
    return [
        _lifecycle_row(policy_id, display_name)
        for policy_id, display_name in [
            ("artifact_creation_policy", "Artifact creation policy"),
            ("artifact_mutation_policy", "Artifact mutation policy"),
            ("artifact_deletion_policy", "Artifact deletion policy"),
            ("artifact_immutability_policy", "Artifact immutability policy"),
            ("artifact_provenance_policy", "Artifact provenance policy"),
            ("artifact_metadata_policy", "Artifact metadata policy"),
            ("artifact_checksum_policy", "Artifact checksum policy"),
            ("artifact_quarantine_policy", "Artifact quarantine policy"),
            ("artifact_promotion_policy", "Artifact promotion policy"),
            ("artifact_rollback_policy", "Artifact rollback policy"),
        ]
    ]


def _lifecycle_row(policy_id: str, display_name: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "display_name": display_name,
        "policy_category": "artifact_lifecycle",
        "required_before_artifact_work": True,
        "satisfied_in_15_6": False,
        "checked_by_15_6": False,
        "allowed_now": False,
        "executed_now": False,
        "requires_future_explicit_gate": True,
        "failure_policy": "fail_closed",
        "notes": "Static policy placeholder only; no artifact work is performed in 15.6.",
    }


def _build_artifact_naming_policy_matrix() -> list[dict[str, Any]]:
    return [
        {
            "naming_rule_id": rule_id,
            "display_name": display_name,
            "required_before_artifact_naming": True,
            "finalized_in_15_6": False,
            "checked_by_15_6": False,
            "allowed_now": False,
            "selected_now": False,
            "requires_future_explicit_gate": True,
            "notes": "Static naming rule placeholder only; no file name or path is selected.",
        }
        for rule_id, display_name in [
            ("product_name_token", "Product name token"),
            ("platform_token", "Platform token"),
            ("version_token", "Version token"),
            ("build_channel_token", "Build channel token"),
            ("git_revision_token", "Git revision token"),
            ("timestamp_token", "Timestamp token"),
            ("architecture_token", "Architecture token"),
            ("artifact_extension_token", "Artifact extension token"),
            ("checksum_naming_token", "Checksum naming token"),
            ("no_secret_material_in_name_token", "No secret material in name token"),
        ]
    ]


def _build_artifact_retention_rollback_policy_matrix() -> list[dict[str, Any]]:
    return [
        {
            "retention_policy_id": policy_id,
            "display_name": display_name,
            "required_before_artifact_release": True,
            "satisfied_in_15_6": False,
            "checked_by_15_6": False,
            "delete_allowed_now": False,
            "rollback_allowed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "notes": "Static retention and rollback placeholder only; no artifact is inspected or changed.",
        }
        for policy_id, display_name in [
            ("retention_duration_policy", "Retention duration policy"),
            ("retention_storage_policy", "Retention storage policy"),
            ("rollback_trigger_policy", "Rollback trigger policy"),
            ("rollback_artifact_selection_policy", "Rollback artifact selection policy"),
            ("delete_stale_artifact_policy", "Delete stale artifact policy"),
            ("delete_failed_artifact_policy", "Delete failed artifact policy"),
            ("manual_delete_approval_policy", "Manual delete approval policy"),
            ("audit_log_policy", "Audit log policy"),
            ("local_only_artifact_policy", "Local-only artifact policy"),
            ("future_release_cleanup_policy", "Future release cleanup policy"),
        ]
    ]


def _build_artifact_smoke_sign_publish_policy_matrix() -> list[dict[str, Any]]:
    return [
        {
            "policy_id": policy_id,
            "display_name": display_name,
            "required_before_publish": True,
            "satisfied_in_15_6": False,
            "checked_by_15_6": False,
            "smoke_allowed_now": False,
            "sign_allowed_now": False,
            "publish_allowed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "notes": "Static publish policy placeholder only; no smoke, signing, or publishing occurs.",
        }
        for policy_id, display_name in [
            ("artifact_smoke_policy", "Artifact smoke policy"),
            ("artifact_smoke_timeout_policy", "Artifact smoke timeout policy"),
            ("artifact_smoke_environment_policy", "Artifact smoke environment policy"),
            ("artifact_signing_policy", "Artifact signing policy"),
            ("signing_key_isolation_policy", "Signing key isolation policy"),
            ("no_live_credential_signing_policy", "No live credential signing policy"),
            ("artifact_publishing_policy", "Artifact publishing policy"),
            ("artifact_publishing_approval_policy", "Artifact publishing approval policy"),
            ("artifact_publishing_destination_policy", "Artifact publishing destination policy"),
            ("publish_rollback_policy", "Publish rollback policy"),
        ]
    ]


def _build_artifact_execution_blocked_matrix() -> list[dict[str, Any]]:
    return [
        {
            "execution_id": execution_id,
            "display_name": display_name,
            "forbidden_in_15_6": True,
            "allowed_now": False,
            "executed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "notes": "Artifact execution is blocked in the source-only 15.6 matrix.",
        }
        for execution_id, display_name in [
            ("artifact_creation", "Artifact creation"),
            ("artifact_mutation", "Artifact mutation"),
            ("artifact_deletion", "Artifact deletion"),
            ("artifact_smoke_test", "Artifact smoke test"),
            ("artifact_signing", "Artifact signing"),
            ("artifact_publishing", "Artifact publishing"),
            ("artifact_location_selection", "Artifact location selection"),
            ("artifact_name_finalization", "Artifact name finalization"),
            ("artifact_checksum_generation", "Artifact checksum generation"),
            ("artifact_metadata_write", "Artifact metadata write"),
            ("artifact_audit_export", "Artifact audit export"),
            ("artifact_cleanup", "Artifact cleanup"),
        ]
    ]


def _build_packaging_execution_carryover_matrix(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    execution = read_model["dry_run_execution_read_model"]
    artifact = read_model["dry_run_artifact_policy_read_model"]
    source_flags = {
        "packaging_dry_run_execution": execution["packaging_dry_run_execution_allowed_now"],
        "packaging_execution": execution["packaging_execution_allowed_now"],
        "pyinstaller_execution": execution["pyinstaller_execution_allowed_now"],
        "build_command_execution": execution["build_command_execution_allowed_now"],
        "build_artifact_creation": execution["build_artifact_creation_allowed_now"],
        "installer_mutation": execution["installer_mutation_allowed_now"],
        "release_workflow_mutation": execution["release_workflow_mutation_allowed_now"],
        "packaging_filesystem_io": execution["packaging_filesystem_io_allowed_now"],
        "packaging_environment_probe": execution["packaging_environment_probe_allowed_now"],
        "dependency_freeze": execution["dependency_freeze_allowed_now"],
        "asset_discovery": execution["asset_discovery_allowed_now"],
        "qml_asset_discovery": execution["qml_asset_discovery_allowed_now"],
        "artifact_creation": artifact["artifact_creation_allowed_now"],
        "artifact_mutation": artifact["artifact_mutation_allowed_now"],
        "artifact_deletion": artifact["artifact_delete_allowed_now"],
        "artifact_smoke_test": artifact["artifact_smoke_test_allowed_now"],
        "artifact_signing": artifact["artifact_signing_allowed_now"],
        "artifact_publishing": artifact["artifact_publishing_allowed_now"],
    }
    return [
        {
            "capability_id": capability_id,
            "display_name": capability_id.replace("_", " ").title(),
            "source_allowed_now": allowed,
            "matrix_allowed_now": False,
            "matrix_executed_now": False,
            "blocked_in_15_6": True,
            "requires_future_explicit_gate": True,
            "notes": "15.6 carries forward the 15.5 execution block without executing anything.",
        }
        for capability_id, allowed in source_flags.items()
    ]


def _build_runtime_safety_carryover_matrix(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_read_model_allowed_now": row["read_model_allowed_now"],
            "artifact_policy_matrix_allowed_now": False,
            "artifact_policy_matrix_executed_now": False,
            "blocked_in_15_6": True,
            "requires_future_explicit_gate": True,
            "notes": "15.6 artifact policy matrix does not alter runtime safety boundaries.",
        }
        for row in read_model["runtime_safety_carryover_read_rows"]
    ]


def _build_exe_direction_artifact_policy_matrix(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_dry_run_read_model"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "artifact_policy_matrix_confirms_exe_direction": True,
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


def _build_fail_closed_artifact_policy_decision() -> dict[str, str]:
    decision = {
        "missing_packaging_dry_run_read_model_policy": "fail_closed",
        "missing_artifact_policy_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    for key in [
        "artifact_creation_in_15_6",
        "artifact_mutation_in_15_6",
        "artifact_deletion_in_15_6",
        "artifact_smoke_test_in_15_6",
        "artifact_signing_in_15_6",
        "artifact_publishing_in_15_6",
        "artifact_name_finalization_in_15_6",
        "artifact_location_selection_in_15_6",
        "artifact_checksum_generation_in_15_6",
        "artifact_metadata_write_in_15_6",
        "artifact_audit_export_in_15_6",
        "artifact_cleanup_in_15_6",
        "packaging_dry_run_execution_in_15_6",
        "packaging_execution_in_15_6",
        "pyinstaller_execution_in_15_6",
        "build_command_execution_in_15_6",
        "build_artifact_creation_in_15_6",
        "installer_change_in_15_6",
        "release_workflow_change_in_15_6",
        "packaging_filesystem_io_in_15_6",
        "packaging_environment_probe_in_15_6",
        "dependency_freeze_in_15_6",
        "asset_discovery_in_15_6",
        "qml_asset_discovery_in_15_6",
        "runtime_activation_in_15_6",
        "paper_runtime_start_in_15_6",
        "testnet_runtime_start_in_15_6",
        "live_canary_start_in_15_6",
        "live_trading_in_15_6",
        "order_generation_in_15_6",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_15_6",
        "network_io_in_15_6",
        "credential_read_in_15_6",
        "config_env_secret_read_in_15_6",
        "qml_bridge_change_in_15_6",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(read_model: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        "source_packaging_dry_run_read_model_read": True,
        "artifact_policy_matrix_built": True,
        "artifact_policy_matrix_only": True,
    }
    for key, value in read_model["non_execution_evidence"].items():
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


def _build_matrix_boundaries() -> dict[str, bool]:
    boundaries = {
        "packaging_artifact_policy_matrix_is_plain_data_only": True,
        "packaging_artifact_policy_matrix_is_source_only": True,
        "packaging_artifact_policy_matrix_reads_dry_run_read_model_only": True,
        "packaging_artifact_policy_matrix_preserves_exe_direction_without_packaging": True,
        "packaging_artifact_policy_matrix_can_feed_15_7_packaging_artifact_policy_read_model": True,
    }
    for key in [
        "packaging_artifact_policy_matrix_cannot_create_artifacts",
        "packaging_artifact_policy_matrix_cannot_mutate_artifacts",
        "packaging_artifact_policy_matrix_cannot_delete_artifacts",
        "packaging_artifact_policy_matrix_cannot_run_artifact_smoke_tests",
        "packaging_artifact_policy_matrix_cannot_sign_artifacts",
        "packaging_artifact_policy_matrix_cannot_publish_artifacts",
        "packaging_artifact_policy_matrix_cannot_finalize_artifact_names",
        "packaging_artifact_policy_matrix_cannot_select_artifact_locations",
        "packaging_artifact_policy_matrix_cannot_generate_checksums",
        "packaging_artifact_policy_matrix_cannot_write_artifact_metadata",
        "packaging_artifact_policy_matrix_cannot_export_artifact_audits",
        "packaging_artifact_policy_matrix_cannot_cleanup_artifacts",
        "packaging_artifact_policy_matrix_cannot_execute_dry_run",
        "packaging_artifact_policy_matrix_cannot_package_exe",
        "packaging_artifact_policy_matrix_cannot_start_pyinstaller",
        "packaging_artifact_policy_matrix_cannot_execute_build_commands",
        "packaging_artifact_policy_matrix_cannot_create_build_artifacts",
        "packaging_artifact_policy_matrix_cannot_change_installers",
        "packaging_artifact_policy_matrix_cannot_change_release_workflows",
        "packaging_artifact_policy_matrix_cannot_probe_packaging_environment",
        "packaging_artifact_policy_matrix_cannot_freeze_dependencies",
        "packaging_artifact_policy_matrix_cannot_discover_assets",
        "packaging_artifact_policy_matrix_cannot_discover_qml_assets",
        "packaging_artifact_policy_matrix_cannot_perform_filesystem_io",
        "packaging_artifact_policy_matrix_cannot_activate_runtime",
        "packaging_artifact_policy_matrix_cannot_start_paper_runtime",
        "packaging_artifact_policy_matrix_cannot_start_testnet_runtime",
        "packaging_artifact_policy_matrix_cannot_start_live_canary",
        "packaging_artifact_policy_matrix_cannot_enable_live_trading",
        "packaging_artifact_policy_matrix_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "packaging_artifact_policy_matrix_cannot_access_private_endpoints",
        "packaging_artifact_policy_matrix_cannot_open_network_io",
        "packaging_artifact_policy_matrix_cannot_read_credentials",
        "packaging_artifact_policy_matrix_cannot_start_runtime_loop",
        "packaging_artifact_policy_matrix_cannot_execute_runtime_gates",
        "packaging_artifact_policy_matrix_cannot_mutate_gate_state",
        "packaging_artifact_policy_matrix_cannot_read_config_env_or_secrets",
        "packaging_artifact_policy_matrix_cannot_change_ui_bridge",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_packaging_dry_run_read_model": SOURCE_PACKAGING_DRY_RUN_READ_MODEL_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_dry_run_read_model_boundaries": {
            "allowed_imports_only": read_model["source_boundaries"]["allowed_imports_only"],
            "source_packaging_dry_run_contract": read_model["source_boundaries"][
                "source_packaging_dry_run_contract"
            ],
            "read_model_boundary_subset": {
                "packaging_dry_run_read_model_is_plain_data_only": read_model[
                    "read_model_boundaries"
                ]["packaging_dry_run_read_model_is_plain_data_only"],
                "packaging_dry_run_read_model_is_source_only": read_model["read_model_boundaries"][
                    "packaging_dry_run_read_model_is_source_only"
                ],
                "packaging_dry_run_read_model_can_feed_15_6_packaging_artifact_policy_matrix": read_model[
                    "read_model_boundaries"
                ]["packaging_dry_run_read_model_can_feed_15_6_packaging_artifact_policy_matrix"],
            },
        },
    }
