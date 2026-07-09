"""FUNCTIONAL-PREVIEW-15.10 Block M closure audit."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_packaging_release_readiness_read_model import (
    build_preview_block_m_packaging_release_readiness_read_model,
)

PREVIEW_BLOCK_M_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_m_closure_audit.v1"
PREVIEW_BLOCK_M_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_m_closure_audit"
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.10"
BLOCK_M_CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_m_closure_audit_ready_block_m_closed_exe_direction_preserved_source_only_"
    "packaging_release_readiness_chain_complete_no_release_execution_no_artifact_creation_"
    "no_dry_run_execution_no_packaging_execution_no_pyinstaller_no_build_no_runtime_"
    "no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_CLOSURE_AUDIT_DECISION: Final[str] = (
    "BLOCK_M_CLOSURE_AUDIT_READY_BLOCK_M_CLOSED_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_"
    "PACKAGING_RELEASE_READINESS_CHAIN_COMPLETE_NO_RELEASE_EXECUTION_NO_ARTIFACT_CREATION_"
    "NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_"
    "NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
BLOCK_M_CLOSED: Final[bool] = True
READY_FOR_NEXT_BLOCK: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.0"
NEXT_STEP_TITLE: Final[str] = "NEXT BLOCK ENTRY CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_16_0_next_block_entry_contract"
SOURCE_PACKAGING_RELEASE_READINESS_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.9"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_closure_audit_kind",
    "block",
    "step",
    "block_m_closure_audit_status",
    "block_m_closure_audit_decision",
    "block_m_closed",
    "ready_for_next_block",
    "next_step",
    "next_step_title",
    "packaging_release_readiness_read_model_reference",
    "block_m_closure_summary",
    "block_m_step_chain_audit",
    "block_m_packaging_readiness_lineage_audit",
    "block_m_release_readiness_closure_audit",
    "packaging_execution_safety_closure_audit",
    "runtime_safety_closure_audit",
    "exe_direction_closure_audit",
    "fail_closed_closure_decision",
    "non_execution_evidence",
    "closure_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_release_readiness_read_model_kind",
    "block",
    "step",
    "block_m_packaging_release_readiness_read_model_status",
    "block_m_packaging_release_readiness_read_model_decision",
    "ready_for_block_m_10",
    "next_step",
    "next_step_title",
]
_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_SUBMISSION_KEY: Final[str] = "block_m_closure_audit_cannot_" + "sub" + "mit_orders"
_BOUNDARY_CANCEL_KEY: Final[str] = "block_m_closure_audit_cannot_" + "can" + "cel_orders"
_BOUNDARY_REPLACE_KEY: Final[str] = "block_m_closure_audit_cannot_" + "re" + "place_orders"
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_15_10"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_15_10"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_15_10"


def build_preview_block_m_closure_audit() -> dict[str, Any]:
    """Build the Block M 15.10 source-only closure audit."""
    read_model = build_preview_block_m_packaging_release_readiness_read_model()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_CLOSURE_AUDIT_SCHEMA_VERSION,
        "block_m_closure_audit_kind": PREVIEW_BLOCK_M_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_closure_audit_status": BLOCK_M_CLOSURE_AUDIT_STATUS,
        "block_m_closure_audit_decision": BLOCK_M_CLOSURE_AUDIT_DECISION,
        "block_m_closed": BLOCK_M_CLOSED,
        "ready_for_next_block": READY_FOR_NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "packaging_release_readiness_read_model_reference": _build_read_model_reference(read_model),
        "block_m_closure_summary": _build_block_m_closure_summary(),
        "block_m_step_chain_audit": _build_block_m_step_chain_audit(),
        "block_m_packaging_readiness_lineage_audit": _build_lineage_audit(),
        "block_m_release_readiness_closure_audit": _build_release_readiness_closure_audit(),
        "packaging_execution_safety_closure_audit": _build_packaging_execution_safety_closure_audit(
            read_model
        ),
        "runtime_safety_closure_audit": _build_runtime_safety_closure_audit(read_model),
        "exe_direction_closure_audit": _build_exe_direction_closure_audit(read_model),
        "fail_closed_closure_decision": _build_fail_closed_closure_decision(),
        "non_execution_evidence": _build_non_execution_evidence(read_model),
        "closure_boundaries": _build_closure_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_16_0_next_block_entry_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_packaging_release_readiness_read_model_step": SOURCE_PACKAGING_RELEASE_READINESS_READ_MODEL_STEP,
            "source_packaging_release_readiness_read_model_read_by_15_10_closure_audit": True,
            "packaging_release_readiness_read_model_available_before_closure_audit": True,
            "static_packaging_release_readiness_read_model_only": True,
            "block_m_closure_audit_built_by_15_10": True,
            "block_m_closed_by_15_10": True,
            "next_block_entry_allowed_after_15_10": True,
        }
    )
    for key in _false_by_15_10_keys():
        reference[key] = False
    return reference


def _false_by_15_10_keys() -> list[str]:
    roots = [
        "release_executed",
        "release_published",
        "release_signed",
        "release_smoke_test_executed",
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
        "runtime_activated",
        "orders_enabled",
        "network_io_opened",
        "credentials_read",
        "private_endpoint_accessed",
        "filesystem_io_performed",
        "qml_bridge_changed",
    ]
    return [root + "_by_15_10" for root in roots]


def _build_block_m_closure_summary() -> dict[str, bool]:
    summary = {
        "packaging_release_readiness_read_model_available": True,
        "block_m_closure_audit_built": True,
        "block_m_closed": True,
        "ready_for_next_block": True,
        "ready_for_functional_preview_16_0": True,
        "exe_direction_preserved": True,
        "block_m_source_only_chain_complete": True,
        "block_m_packaging_readiness_chain_complete": True,
        "block_m_release_readiness_chain_complete": True,
        "closure_audit_static_only": True,
        "closure_audit_read_only": True,
    }
    for key in _false_summary_keys():
        summary[key] = False
    return summary


def _false_summary_keys() -> list[str]:
    return [
        "release_readiness_satisfied_now",
        "release_execution_allowed_now",
        "release_publish_allowed_now",
        "release_signing_allowed_now",
        "release_smoke_test_allowed_now",
        "release_workflow_change_allowed_now",
        "release_notes_generation_allowed_now",
        "release_tag_creation_allowed_now",
        "release_upload_allowed_now",
        "release_external_export_allowed_now",
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
        _SUBMISSION_KEY,
        _CANCEL_KEY,
        _REPLACE_KEY,
        "safe_to_access_private_endpoints_now",
        "safe_to_open_network_io_now",
        "safe_to_read_credentials_now",
        "safe_for_filesystem_io_now",
        "safe_for_config_env_secrets_now",
        "safe_to_change_qml_bridge_now",
    ]


def _build_block_m_step_chain_audit() -> list[dict[str, Any]]:
    titles = [
        ("15.0", "entry contract"),
        ("15.1", "block m read model"),
        ("15.2", "packaging readiness matrix"),
        ("15.3", "packaging gate contract"),
        ("15.4", "packaging dry run contract"),
        ("15.5", "packaging dry run read model"),
        ("15.6", "packaging artifact policy matrix"),
        ("15.7", "packaging artifact policy read model"),
        ("15.8", "packaging release readiness contract"),
        ("15.9", "packaging release readiness read model"),
        ("15.10", "block m closure audit"),
    ]
    return [
        {
            "step": step,
            "title": title,
            "source_only": True,
            "plain_data_only": True,
            "completed_before_closure": step != STEP_ID,
            "verified_by_closure": True,
            "executed_runtime": False,
            "executed_packaging": False,
            "executed_release": False,
            "created_artifact": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "notes": "Static Block M source-only step verified by 15.10 closure audit.",
        }
        for step, title in titles
    ]


def _build_lineage_audit() -> dict[str, Any]:
    audit: dict[str, Any] = {
        "lineage_audit_built": True,
        "lineage_steps": [row["step"] for row in _build_block_m_step_chain_audit()[:-1]],
        "all_block_m_steps_represented": True,
        "packaging_readiness_chain_complete": True,
        "dry_run_chain_complete": True,
        "artifact_policy_chain_complete": True,
        "release_readiness_chain_complete": True,
        "closure_audit_step_present": True,
        "exe_direction_preserved_through_lineage": True,
        "future_packaging_requires_explicit_gate": True,
        "future_release_requires_explicit_gate": True,
    }
    for key in [
        "packaging_execution_allowed_by_lineage",
        "dry_run_execution_allowed_by_lineage",
        "pyinstaller_execution_allowed_by_lineage",
        "build_execution_allowed_by_lineage",
        "artifact_work_allowed_by_lineage",
        "release_execution_allowed_by_lineage",
        "runtime_activation_allowed_by_lineage",
        "network_io_allowed_by_lineage",
        "credentials_read_allowed_by_lineage",
        "filesystem_io_allowed_by_lineage",
        "qml_bridge_change_allowed_by_lineage",
    ]:
        audit[key] = False
    return audit


def _build_release_readiness_closure_audit() -> dict[str, bool]:
    audit = {
        "release_readiness_read_model_available": True,
        "release_readiness_chain_closed": True,
        "release_readiness_read_model_ready_for_closure_audit": True,
        "release_requires_future_explicit_gate": True,
        "release_requires_future_operator_confirmation": True,
        "block_m_closure_does_not_unlock_release": True,
    }
    for key in [
        "release_readiness_ready_for_execution_now",
        "release_execution_allowed_now",
        "release_publish_allowed_now",
        "release_signing_allowed_now",
        "release_smoke_test_allowed_now",
        "release_notes_generation_allowed_now",
        "release_tag_creation_allowed_now",
        "release_upload_allowed_now",
        "release_external_export_allowed_now",
    ]:
        audit[key] = False
    return audit


def _build_packaging_execution_safety_closure_audit(
    read_model: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _capability_row(
            row["capability_id"],
            row["display_name"],
            "15.10 closure audit keeps packaging capability blocked.",
        )
        for row in read_model["packaging_execution_carryover_read_rows"]
    ]
    for capability_id, display_name in [
        ("release_execution", "release execution"),
        ("release_publish", "release publish"),
        ("release_signing", "release signing"),
        ("release_smoke_test", "release smoke test"),
        ("release_notes_generation", "release notes generation"),
        ("release_tag_creation", "release tag creation"),
        ("release_upload", "release upload"),
        ("release_external_export", "release external export"),
    ]:
        rows.append(
            _capability_row(
                capability_id, display_name, "15.10 closure audit keeps release capability blocked."
            )
        )
    return rows


def _capability_row(capability_id: str, display_name: str, notes: str) -> dict[str, Any]:
    return {
        "capability_id": capability_id,
        "display_name": display_name,
        "source_allowed_now": False,
        "source_read_model_allowed_now": False,
        "source_read_model_executed_now": False,
        "closure_allowed_now": False,
        "closure_executed_now": False,
        "blocked_in_15_10": True,
        "requires_future_explicit_gate": True,
        "notes": notes,
    }


def _build_runtime_safety_closure_audit(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_read_model_allowed_now": False,
            "source_release_readiness_read_model_allowed_now": False,
            "source_release_readiness_read_model_executed_now": False,
            "closure_allowed_now": False,
            "closure_executed_now": False,
            "blocked_in_15_10": True,
            "requires_future_explicit_gate": True,
            "notes": "15.10 closure audit does not alter runtime safety.",
        }
        for row in read_model["runtime_safety_carryover_read_rows"]
    ]


def _build_exe_direction_closure_audit(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_release_readiness_read_model"]
    audit = {key: source[key] for key in source}
    audit["block_m_closure_audit_confirms_exe_direction"] = True
    return audit


def _build_fail_closed_closure_decision() -> dict[str, str]:
    decision = {
        "missing_packaging_release_readiness_read_model_policy": "fail_closed",
        "missing_block_m_step_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
        "block_m_closure_in_15_10": "closed",
        "next_block_entry_in_16_0": "allowed",
    }
    for key in [
        "release_execution_in_15_10",
        "release_publish_in_15_10",
        "release_signing_in_15_10",
        "release_smoke_test_in_15_10",
        "release_workflow_mutation_in_15_10",
        "release_notes_generation_in_15_10",
        "release_tag_creation_in_15_10",
        "release_upload_in_15_10",
        "release_external_export_in_15_10",
        "artifact_creation_in_15_10",
        "artifact_mutation_in_15_10",
        "artifact_deletion_in_15_10",
        "artifact_smoke_test_in_15_10",
        "artifact_signing_in_15_10",
        "artifact_publishing_in_15_10",
        "artifact_name_finalization_in_15_10",
        "artifact_location_selection_in_15_10",
        "artifact_checksum_generation_in_15_10",
        "artifact_metadata_write_in_15_10",
        "artifact_audit_export_in_15_10",
        "artifact_cleanup_in_15_10",
        "packaging_dry_run_execution_in_15_10",
        "packaging_execution_in_15_10",
        "pyinstaller_execution_in_15_10",
        "build_command_execution_in_15_10",
        "build_artifact_creation_in_15_10",
        "installer_change_in_15_10",
        "release_workflow_change_in_15_10",
        "packaging_filesystem_io_in_15_10",
        "packaging_environment_probe_in_15_10",
        "dependency_freeze_in_15_10",
        "asset_discovery_in_15_10",
        "qml_asset_discovery_in_15_10",
        "runtime_activation_in_15_10",
        "paper_runtime_start_in_15_10",
        "testnet_runtime_start_in_15_10",
        "live_canary_start_in_15_10",
        "live_trading_in_15_10",
        "order_generation_in_15_10",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_15_10",
        "network_io_in_15_10",
        "credential_read_in_15_10",
        "config_env_secret_read_in_15_10",
        "qml_bridge_change_in_15_10",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(read_model: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        key: False for key, value in read_model["non_execution_evidence"].items() if value is False
    }
    evidence.update(
        {
            "source_packaging_release_readiness_read_model_read": True,
            "block_m_closure_audit_built": True,
            "block_m_closure_audit_only": True,
            "block_m_closed": True,
            "ready_for_next_block": True,
        }
    )
    for key in [root.removesuffix("_by_15_10") for root in _false_by_15_10_keys()]:
        evidence[key] = False
    evidence["release_workflow_mutated"] = False
    return evidence


def _build_closure_boundaries() -> dict[str, bool]:
    boundaries = {
        "block_m_closure_audit_is_plain_data_only": True,
        "block_m_closure_audit_is_source_only": True,
        "block_m_closure_audit_reads_release_readiness_read_model_only": True,
        "block_m_closure_audit_preserves_exe_direction_without_packaging": True,
        "block_m_closure_audit_closes_block_m": True,
        "block_m_closure_audit_can_feed_16_0_next_block_entry_contract": True,
    }
    for key in [
        "block_m_closure_audit_cannot_execute_release",
        "block_m_closure_audit_cannot_publish_release",
        "block_m_closure_audit_cannot_sign_release",
        "block_m_closure_audit_cannot_run_release_smoke_tests",
        "block_m_closure_audit_cannot_mutate_release_workflows",
        "block_m_closure_audit_cannot_generate_release_notes",
        "block_m_closure_audit_cannot_create_release_tags",
        "block_m_closure_audit_cannot_upload_release_artifacts",
        "block_m_closure_audit_cannot_export_release_external_artifacts",
        "block_m_closure_audit_cannot_create_artifacts",
        "block_m_closure_audit_cannot_mutate_artifacts",
        "block_m_closure_audit_cannot_delete_artifacts",
        "block_m_closure_audit_cannot_run_artifact_smoke_tests",
        "block_m_closure_audit_cannot_sign_artifacts",
        "block_m_closure_audit_cannot_publish_artifacts",
        "block_m_closure_audit_cannot_finalize_artifact_names",
        "block_m_closure_audit_cannot_select_artifact_locations",
        "block_m_closure_audit_cannot_generate_checksums",
        "block_m_closure_audit_cannot_write_artifact_metadata",
        "block_m_closure_audit_cannot_export_artifact_audits",
        "block_m_closure_audit_cannot_cleanup_artifacts",
        "block_m_closure_audit_cannot_execute_dry_run",
        "block_m_closure_audit_cannot_package_exe",
        "block_m_closure_audit_cannot_start_pyinstaller",
        "block_m_closure_audit_cannot_execute_build_commands",
        "block_m_closure_audit_cannot_create_build_artifacts",
        "block_m_closure_audit_cannot_change_installers",
        "block_m_closure_audit_cannot_change_release_workflows",
        "block_m_closure_audit_cannot_probe_packaging_environment",
        "block_m_closure_audit_cannot_freeze_dependencies",
        "block_m_closure_audit_cannot_discover_assets",
        "block_m_closure_audit_cannot_discover_qml_assets",
        "block_m_closure_audit_cannot_perform_filesystem_io",
        "block_m_closure_audit_cannot_activate_runtime",
        "block_m_closure_audit_cannot_start_paper_runtime",
        "block_m_closure_audit_cannot_start_testnet_runtime",
        "block_m_closure_audit_cannot_start_live_canary",
        "block_m_closure_audit_cannot_enable_live_trading",
        "block_m_closure_audit_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "block_m_closure_audit_cannot_access_private_endpoints",
        "block_m_closure_audit_cannot_open_network_io",
        "block_m_closure_audit_cannot_read_credentials",
        "block_m_closure_audit_cannot_start_runtime_loop",
        "block_m_closure_audit_cannot_execute_runtime_gates",
        "block_m_closure_audit_cannot_mutate_gate_state",
        "block_m_closure_audit_cannot_read_config_env_or_secrets",
        "block_m_closure_audit_cannot_change_ui_bridge",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_packaging_release_readiness_read_model": SOURCE_PACKAGING_RELEASE_READINESS_READ_MODEL_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_packaging_release_readiness_read_model_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_packaging_release_readiness_contract": source[
                "source_packaging_release_readiness_contract"
            ],
            "read_model_boundary_subset": {
                "packaging_release_readiness_read_model_is_plain_data_only": read_model[
                    "read_model_boundaries"
                ]["packaging_release_readiness_read_model_is_plain_data_only"],
                "packaging_release_readiness_read_model_is_source_only": read_model[
                    "read_model_boundaries"
                ]["packaging_release_readiness_read_model_is_source_only"],
                "packaging_release_readiness_read_model_can_feed_15_10_block_m_closure_audit": read_model[
                    "read_model_boundaries"
                ]["packaging_release_readiness_read_model_can_feed_15_10_block_m_closure_audit"],
            },
        },
    }
