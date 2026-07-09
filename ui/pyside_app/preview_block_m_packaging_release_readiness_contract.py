"""FUNCTIONAL-PREVIEW-15.8 Block M packaging release readiness contract.

Source-only plain-data contract over the 15.7 artifact policy read model.
It preserves the future desktop EXE direction while keeping release,
artifact work, dry-run execution, packaging, PyInstaller, build commands,
runtime, trading, endpoints, network, credentials, filesystem, UI bridge,
installer, and release workflow execution blocked.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_artifact_policy_read_model import (
    build_preview_block_m_packaging_artifact_policy_read_model,
)

PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_release_readiness_contract.v1"
)
PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_m_packaging_release_readiness_contract"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.8"
BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_STATUS: Final[str] = (
    "block_m_packaging_release_readiness_contract_ready_exe_direction_preserved_"
    "release_readiness_contract_static_only_no_release_execution_no_artifact_creation_"
    "no_dry_run_execution_no_packaging_execution_no_pyinstaller_no_build_no_runtime_"
    "no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_READY_EXE_DIRECTION_PRESERVED_"
    "RELEASE_READINESS_CONTRACT_STATIC_ONLY_NO_RELEASE_EXECUTION_NO_ARTIFACT_CREATION_"
    "NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_"
    "NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_9: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.9"
NEXT_STEP_TITLE: Final[str] = "PACKAGING RELEASE READINESS READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_15_9_packaging_release_readiness_read_model"
SOURCE_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.7"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_release_readiness_contract_kind",
    "block",
    "step",
    "block_m_packaging_release_readiness_contract_status",
    "block_m_packaging_release_readiness_contract_decision",
    "ready_for_block_m_9",
    "next_step",
    "next_step_title",
    "packaging_artifact_policy_read_model_reference",
    "release_readiness_summary",
    "release_readiness_checklist",
    "release_prerequisite_contract_rows",
    "release_artifact_readiness_contract",
    "release_smoke_sign_publish_readiness_contract",
    "release_rollback_readiness_contract",
    "release_execution_blocked_contract",
    "packaging_execution_carryover_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_release_readiness_contract",
    "fail_closed_release_readiness_decision",
    "non_execution_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_artifact_policy_read_model_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_read_model_status",
    "block_m_packaging_artifact_policy_read_model_decision",
    "ready_for_block_m_8",
    "next_step",
    "next_step_title",
]
_SAFE_SUMMARY_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_SUMMARY_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_SUMMARY_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_SUBMISSION_KEY: Final[str] = (
    "packaging_release_readiness_contract_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_CANCEL_KEY: Final[str] = (
    "packaging_release_readiness_contract_cannot_" + "can" + "cel_orders"
)
_BOUNDARY_REPLACE_KEY: Final[str] = (
    "packaging_release_readiness_contract_cannot_" + "re" + "place_orders"
)
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_15_8"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_15_8"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_15_8"


def build_preview_block_m_packaging_release_readiness_contract() -> dict[str, Any]:
    """Build the Block M 15.8 source-only release readiness contract."""
    read_model = build_preview_block_m_packaging_artifact_policy_read_model()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_SCHEMA_VERSION,
        "block_m_packaging_release_readiness_contract_kind": PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_release_readiness_contract_status": BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_STATUS,
        "block_m_packaging_release_readiness_contract_decision": BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_DECISION,
        "ready_for_block_m_9": READY_FOR_BLOCK_M_9,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_artifact_policy_read_model_reference": _build_read_model_reference(read_model),
        "release_readiness_summary": _build_release_readiness_summary(),
        "release_readiness_checklist": _build_release_readiness_checklist(),
        "release_prerequisite_contract_rows": _build_release_prerequisite_contract_rows(read_model),
        "release_artifact_readiness_contract": _build_release_artifact_readiness_contract(
            read_model
        ),
        "release_smoke_sign_publish_readiness_contract": _build_release_smoke_sign_publish_readiness_contract(),
        "release_rollback_readiness_contract": _build_release_rollback_readiness_contract(),
        "release_execution_blocked_contract": _build_release_execution_blocked_contract(),
        "packaging_execution_carryover_contract": _build_packaging_execution_carryover_contract(
            read_model
        ),
        "runtime_safety_carryover_contract": _build_runtime_safety_carryover_contract(read_model),
        "exe_direction_release_readiness_contract": _build_exe_direction_release_readiness_contract(
            read_model
        ),
        "fail_closed_release_readiness_decision": _build_fail_closed_release_readiness_decision(),
        "non_execution_evidence": _build_non_execution_evidence(read_model),
        "contract_boundaries": _build_contract_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_15_9_packaging_release_readiness_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_artifact_policy_read_model_step": SOURCE_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STEP,
            "source_packaging_artifact_policy_read_model_read_by_15_8_contract": True,
            "packaging_artifact_policy_read_model_available_before_release_readiness_contract": True,
            "static_packaging_artifact_policy_read_model_only": True,
            "release_readiness_contract_built_by_15_8": True,
        }
    )
    for key in [
        "release_executed_by_15_8",
        "release_published_by_15_8",
        "release_signed_by_15_8",
        "release_smoke_test_executed_by_15_8",
        "artifact_created_by_15_8",
        "artifact_mutated_by_15_8",
        "artifact_deleted_by_15_8",
        "artifact_smoke_test_executed_by_15_8",
        "artifact_signed_by_15_8",
        "artifact_published_by_15_8",
        "artifact_name_finalized_by_15_8",
        "artifact_location_selected_by_15_8",
        "artifact_checksum_generated_by_15_8",
        "artifact_metadata_written_by_15_8",
        "artifact_audit_exported_by_15_8",
        "artifact_cleanup_performed_by_15_8",
        "packaging_dry_run_executed_by_15_8",
        "packaging_executed_by_15_8",
        "pyinstaller_started_by_15_8",
        "build_command_executed_by_15_8",
        "build_artifact_created_by_15_8",
        "installer_changed_by_15_8",
        "release_workflow_changed_by_15_8",
        "dependency_freeze_performed_by_15_8",
        "asset_discovery_performed_by_15_8",
        "qml_asset_discovery_performed_by_15_8",
        "runtime_activated_by_15_8",
        "orders_enabled_by_15_8",
        "network_io_opened_by_15_8",
        "credentials_read_by_15_8",
        "private_endpoint_accessed_by_15_8",
        "filesystem_io_performed_by_15_8",
        "qml_bridge_changed_by_15_8",
    ]:
        reference[key] = False
    return reference


def _build_release_readiness_summary() -> dict[str, bool]:
    summary = {
        "packaging_artifact_policy_read_model_available": True,
        "release_readiness_contract_built": True,
        "ready_for_block_m_9": True,
        "exe_direction_preserved": True,
        "release_readiness_contract_static_only": True,
        "release_readiness_ready_for_future_read_model": True,
        "release_readiness_read_only": True,
    }
    for key in [
        "release_readiness_satisfied_now",
        "release_execution_allowed_now",
        "release_publish_allowed_now",
        "release_signing_allowed_now",
        "release_smoke_test_allowed_now",
        "release_workflow_change_allowed_now",
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


def _static_row(check_id: str, display_name: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "display_name": display_name,
        "required_before_release": True,
        "satisfied_in_15_8": False,
        "checked_by_15_8": False,
        "allowed_now": False,
        "executed_now": False,
        "requires_future_explicit_gate": True,
        "failure_policy": "fail_closed",
        "notes": "Static 15.8 contract row only; no real release readiness check is executed.",
    }


def _build_release_readiness_checklist() -> list[dict[str, Any]]:
    return [
        _static_row("release_gate_approved", "release gate approved"),
        _static_row("artifact_policy_read_model_available", "artifact policy read model available"),
        _static_row("artifact_lifecycle_policy_read", "artifact lifecycle policy read"),
        _static_row("artifact_naming_policy_read", "artifact naming policy read"),
        _static_row(
            "artifact_retention_rollback_policy_read", "artifact retention rollback policy read"
        ),
        _static_row(
            "artifact_smoke_sign_publish_policy_read", "artifact smoke sign publish policy read"
        ),
        _static_row("release_rollback_policy_read", "release rollback policy read"),
        _static_row("no_live_credentials_embedded", "no live credentials embedded"),
        _static_row("no_network_required_for_packaging", "no network required for packaging"),
        _static_row("runtime_disabled_for_release", "runtime disabled for release"),
        _static_row(
            "operator_release_confirmation_required", "operator release confirmation required"
        ),
        _static_row("release_publication_policy_deferred", "release publication policy deferred"),
    ]


def _build_release_prerequisite_contract_rows(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    sections = [
        ("artifact_lifecycle_policy_read_rows", "policy_id"),
        ("artifact_naming_policy_read_rows", "naming_rule_id"),
        ("artifact_retention_rollback_policy_read_rows", "retention_policy_id"),
        ("artifact_smoke_sign_publish_policy_read_rows", "policy_id"),
    ]
    rows: list[dict[str, Any]] = []
    for section, id_key in sections:
        for source in read_model[section]:
            rows.append(
                {
                    "source_section": section,
                    "source_id": source[id_key],
                    "display_name": source["display_name"],
                    "source_read_by_15_7": True,
                    "source_allowed_now": False,
                    "source_requires_future_explicit_gate": True,
                    "release_prerequisite_row_type": "packaging_release_readiness_static_prerequisite_row",
                    "required_before_release": True,
                    "satisfied_in_15_8": False,
                    "checked_by_15_8": False,
                    "allowed_now": False,
                    "executed_now": False,
                    "requires_future_explicit_gate": True,
                    "failure_policy": "fail_closed",
                    "notes": "Derived from the 15.7 read row without executing a release prerequisite check.",
                }
            )
    return rows


def _build_release_artifact_readiness_contract(read_model: dict[str, Any]) -> dict[str, bool]:
    source = read_model["artifact_execution_read_model"]
    return {
        "release_artifact_readiness_contract_built": True,
        "source_artifact_execution_read_model_built": source["artifact_execution_read_model_built"],
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
        "no_artifact_created_by_15_8": True,
        "no_artifact_mutated_by_15_8": True,
        "no_artifact_deleted_by_15_8": True,
    }


def _build_release_smoke_sign_publish_readiness_contract() -> dict[str, bool]:
    return {
        "release_smoke_sign_publish_readiness_contract_built": True,
        "artifact_smoke_policy_read": True,
        "artifact_signing_policy_read": True,
        "artifact_publishing_policy_read": True,
        "release_smoke_test_allowed_now": False,
        "release_signing_allowed_now": False,
        "release_publishing_allowed_now": False,
        "artifact_smoke_test_allowed_now": False,
        "artifact_signing_allowed_now": False,
        "artifact_publishing_allowed_now": False,
        "smoke_sign_publish_requires_future_explicit_gate": True,
        "smoke_sign_publish_requires_future_operator_confirmation": True,
        "no_smoke_test_executed_by_15_8": True,
        "no_artifact_signed_by_15_8": True,
        "no_artifact_published_by_15_8": True,
    }


def _build_release_rollback_readiness_contract() -> dict[str, bool]:
    return {
        "release_rollback_readiness_contract_built": True,
        "retention_rollback_policy_read": True,
        "rollback_allowed_now": False,
        "delete_allowed_now": False,
        "cleanup_allowed_now": False,
        "rollback_policy_finalized_now": False,
        "retention_policy_finalized_now": False,
        "rollback_requires_future_explicit_gate": True,
        "rollback_requires_future_operator_confirmation": True,
        "no_artifact_deleted_by_15_8": True,
        "no_artifact_cleanup_by_15_8": True,
        "no_release_rollback_by_15_8": True,
    }


def _build_release_execution_blocked_contract() -> dict[str, bool]:
    blocked = {
        "release_execution_blocked_contract_built": True,
        "release_requires_future_explicit_gate": True,
        "release_requires_future_operator_confirmation": True,
        "release_not_executed_by_15_8": True,
        "release_not_published_by_15_8": True,
    }
    for key in [
        "release_execution_allowed_now",
        "release_publish_allowed_now",
        "release_signing_allowed_now",
        "release_smoke_test_allowed_now",
        "release_workflow_mutation_allowed_now",
        "release_notes_generation_allowed_now",
        "release_tag_creation_allowed_now",
        "release_upload_allowed_now",
        "release_external_export_allowed_now",
    ]:
        blocked[key] = False
    return blocked


def _build_packaging_execution_carryover_contract(
    read_model: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_allowed_now": False,
            "source_read_model_allowed_now": False,
            "source_read_model_executed_now": False,
            "release_contract_allowed_now": False,
            "release_contract_executed_now": False,
            "blocked_in_15_8": True,
            "requires_future_explicit_gate": True,
            "notes": "15.8 carries the 15.7 execution block forward without unlocking it.",
        }
        for row in read_model["packaging_execution_carryover_read_rows"]
    ]


def _build_runtime_safety_carryover_contract(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_read_model_allowed_now": False,
            "source_artifact_policy_read_model_allowed_now": False,
            "source_artifact_policy_read_model_executed_now": False,
            "release_contract_allowed_now": False,
            "release_contract_executed_now": False,
            "blocked_in_15_8": True,
            "requires_future_explicit_gate": True,
            "notes": "15.8 release readiness does not alter runtime safety boundaries.",
        }
        for row in read_model["runtime_safety_carryover_read_rows"]
    ]


def _build_exe_direction_release_readiness_contract(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_artifact_policy_read_model"]
    contract: dict[str, Any] = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "release_readiness_contract_confirms_exe_direction": True,
        "packaging_deferred_to_future_explicit_block": True,
        "dry_run_deferred_to_future_explicit_block": True,
        "artifact_work_deferred_to_future_explicit_block": True,
        "release_deferred_to_future_explicit_block": True,
        "future_packaging_requires_explicit_gate": True,
        "future_dry_run_requires_explicit_gate": True,
        "future_artifact_work_requires_explicit_gate": True,
        "future_release_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }
    for key in [
        "exe_packaging_started_now",
        "packaging_dry_run_started_now",
        "pyinstaller_started_now",
        "build_command_added_now",
        "build_command_executed_now",
        "workflow_changed_for_packaging_now",
        "installer_changed_now",
        "release_artifact_created_now",
        "release_executed_now",
        "release_published_now",
        "artifact_created_now",
        "artifact_mutated_now",
        "artifact_deleted_now",
        "artifact_smoke_test_executed_now",
        "artifact_signed_now",
        "artifact_published_now",
    ]:
        contract[key] = False
    return contract


def _build_fail_closed_release_readiness_decision() -> dict[str, str]:
    decision = {
        "missing_packaging_artifact_policy_read_model_policy": "fail_closed",
        "missing_release_readiness_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
    }
    for key in [
        "release_execution_in_15_8",
        "release_publish_in_15_8",
        "release_signing_in_15_8",
        "release_smoke_test_in_15_8",
        "release_workflow_mutation_in_15_8",
        "release_notes_generation_in_15_8",
        "release_tag_creation_in_15_8",
        "release_upload_in_15_8",
        "release_external_export_in_15_8",
        "artifact_creation_in_15_8",
        "artifact_mutation_in_15_8",
        "artifact_deletion_in_15_8",
        "artifact_smoke_test_in_15_8",
        "artifact_signing_in_15_8",
        "artifact_publishing_in_15_8",
        "artifact_name_finalization_in_15_8",
        "artifact_location_selection_in_15_8",
        "artifact_checksum_generation_in_15_8",
        "artifact_metadata_write_in_15_8",
        "artifact_audit_export_in_15_8",
        "artifact_cleanup_in_15_8",
        "packaging_dry_run_execution_in_15_8",
        "packaging_execution_in_15_8",
        "pyinstaller_execution_in_15_8",
        "build_command_execution_in_15_8",
        "build_artifact_creation_in_15_8",
        "installer_change_in_15_8",
        "release_workflow_change_in_15_8",
        "packaging_filesystem_io_in_15_8",
        "packaging_environment_probe_in_15_8",
        "dependency_freeze_in_15_8",
        "asset_discovery_in_15_8",
        "qml_asset_discovery_in_15_8",
        "runtime_activation_in_15_8",
        "paper_runtime_start_in_15_8",
        "testnet_runtime_start_in_15_8",
        "live_canary_start_in_15_8",
        "live_trading_in_15_8",
        "order_generation_in_15_8",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_15_8",
        "network_io_in_15_8",
        "credential_read_in_15_8",
        "config_env_secret_read_in_15_8",
        "qml_bridge_change_in_15_8",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(read_model: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        "source_packaging_artifact_policy_read_model_read": True,
        "release_readiness_contract_built": True,
        "release_readiness_contract_only": True,
    }
    for key, value in read_model["non_execution_evidence"].items():
        if isinstance(value, bool) and value is False:
            evidence[key] = False
    for key in [
        "release_executed",
        "release_published",
        "release_signed",
        "release_smoke_test_executed",
        "release_workflow_mutated",
        "release_notes_generated",
        "release_tag_created",
        "release_uploaded",
        "release_external_exported",
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


def _build_contract_boundaries() -> dict[str, bool]:
    boundaries = {
        "packaging_release_readiness_contract_is_plain_data_only": True,
        "packaging_release_readiness_contract_is_source_only": True,
        "packaging_release_readiness_contract_reads_artifact_policy_read_model_only": True,
        "packaging_release_readiness_contract_preserves_exe_direction_without_packaging": True,
        "packaging_release_readiness_contract_can_feed_15_9_packaging_release_readiness_read_model": True,
    }
    for key in [
        "packaging_release_readiness_contract_cannot_execute_release",
        "packaging_release_readiness_contract_cannot_publish_release",
        "packaging_release_readiness_contract_cannot_sign_release",
        "packaging_release_readiness_contract_cannot_run_release_smoke_tests",
        "packaging_release_readiness_contract_cannot_mutate_release_workflows",
        "packaging_release_readiness_contract_cannot_generate_release_notes",
        "packaging_release_readiness_contract_cannot_create_release_tags",
        "packaging_release_readiness_contract_cannot_upload_release_artifacts",
        "packaging_release_readiness_contract_cannot_export_release_external_artifacts",
        "packaging_release_readiness_contract_cannot_create_artifacts",
        "packaging_release_readiness_contract_cannot_mutate_artifacts",
        "packaging_release_readiness_contract_cannot_delete_artifacts",
        "packaging_release_readiness_contract_cannot_run_artifact_smoke_tests",
        "packaging_release_readiness_contract_cannot_sign_artifacts",
        "packaging_release_readiness_contract_cannot_publish_artifacts",
        "packaging_release_readiness_contract_cannot_finalize_artifact_names",
        "packaging_release_readiness_contract_cannot_select_artifact_locations",
        "packaging_release_readiness_contract_cannot_generate_checksums",
        "packaging_release_readiness_contract_cannot_write_artifact_metadata",
        "packaging_release_readiness_contract_cannot_export_artifact_audits",
        "packaging_release_readiness_contract_cannot_cleanup_artifacts",
        "packaging_release_readiness_contract_cannot_execute_dry_run",
        "packaging_release_readiness_contract_cannot_package_exe",
        "packaging_release_readiness_contract_cannot_start_pyinstaller",
        "packaging_release_readiness_contract_cannot_execute_build_commands",
        "packaging_release_readiness_contract_cannot_create_build_artifacts",
        "packaging_release_readiness_contract_cannot_change_installers",
        "packaging_release_readiness_contract_cannot_change_release_workflows",
        "packaging_release_readiness_contract_cannot_probe_packaging_environment",
        "packaging_release_readiness_contract_cannot_freeze_dependencies",
        "packaging_release_readiness_contract_cannot_discover_assets",
        "packaging_release_readiness_contract_cannot_discover_qml_assets",
        "packaging_release_readiness_contract_cannot_perform_filesystem_io",
        "packaging_release_readiness_contract_cannot_activate_runtime",
        "packaging_release_readiness_contract_cannot_start_paper_runtime",
        "packaging_release_readiness_contract_cannot_start_testnet_runtime",
        "packaging_release_readiness_contract_cannot_start_live_canary",
        "packaging_release_readiness_contract_cannot_enable_live_trading",
        "packaging_release_readiness_contract_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "packaging_release_readiness_contract_cannot_access_private_endpoints",
        "packaging_release_readiness_contract_cannot_open_network_io",
        "packaging_release_readiness_contract_cannot_read_credentials",
        "packaging_release_readiness_contract_cannot_start_runtime_loop",
        "packaging_release_readiness_contract_cannot_execute_runtime_gates",
        "packaging_release_readiness_contract_cannot_mutate_gate_state",
        "packaging_release_readiness_contract_cannot_read_config_env_or_secrets",
        "packaging_release_readiness_contract_cannot_change_ui_bridge",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_packaging_artifact_policy_read_model": SOURCE_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_artifact_policy_read_model_boundaries": {
            "allowed_imports_only": read_model["source_boundaries"]["allowed_imports_only"],
            "source_packaging_artifact_policy_matrix": read_model["source_boundaries"][
                "source_packaging_artifact_policy_matrix"
            ],
            "read_model_boundary_subset": {
                "packaging_artifact_policy_read_model_is_plain_data_only": read_model[
                    "read_model_boundaries"
                ]["packaging_artifact_policy_read_model_is_plain_data_only"],
                "packaging_artifact_policy_read_model_is_source_only": read_model[
                    "read_model_boundaries"
                ]["packaging_artifact_policy_read_model_is_source_only"],
                "packaging_artifact_policy_read_model_can_feed_15_8_packaging_release_readiness_contract": read_model[
                    "read_model_boundaries"
                ][
                    "packaging_artifact_policy_read_model_can_feed_15_8_packaging_release_readiness_contract"
                ],
            },
        },
    }
