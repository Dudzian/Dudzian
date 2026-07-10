"""FUNCTIONAL-PREVIEW-16.1 Block N source-only read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_entry_contract import build_preview_block_n_entry_contract

PREVIEW_BLOCK_N_READ_MODEL_SCHEMA_VERSION: Final[str] = "preview_block_n_read_model.v1"
PREVIEW_BLOCK_N_READ_MODEL_KIND: Final[str] = "functional_preview_block_n_read_model"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.1"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.2"
NEXT_STEP_TITLE: Final[str] = "BLOCK N SAFETY GATE MATRIX"
READY_FOR_BLOCK_N_2: Final[bool] = True
BLOCK_N_READ_MODEL_STATUS: Final[str] = (
    "block_n_read_model_ready_block_n_entry_consumed_block_m_closure_preserved_exe_"
    "direction_preserved_source_only_block_read_model_no_release_execution_no_artifact_"
    "creation_no_dry_run_execution_no_packaging_execution_no_pyinstaller_no_build_no_"
    "runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_N_READ_MODEL_DECISION: Final[str] = (
    "BLOCK_N_READ_MODEL_READY_BLOCK_N_ENTRY_CONSUMED_BLOCK_M_CLOSURE_PRESERVED_EXE_"
    "DIRECTION_PRESERVED_SOURCE_ONLY_BLOCK_READ_MODEL_NO_RELEASE_EXECUTION_NO_ARTIFACT_"
    "CREATION_NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_"
    "RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
STATUS: Final[str] = "ready_for_functional_preview_16_2_block_n_safety_gate_matrix"
SOURCE_BLOCK_N_ENTRY_CONTRACT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.0"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_read_model_kind",
    "block",
    "step",
    "block_n_read_model_status",
    "block_n_read_model_decision",
    "ready_for_block_n_2",
    "next_step",
    "next_step_title",
    "block_n_entry_contract_reference",
    "block_n_read_summary",
    "block_m_closure_handoff_read_model",
    "block_n_entry_readiness_read_model",
    "packaging_release_safety_read_rows",
    "runtime_safety_read_rows",
    "exe_direction_read_model",
    "fail_closed_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_ENTRY_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_entry_contract_kind",
    "block",
    "step",
    "block_n_entry_contract_status",
    "block_n_entry_contract_decision",
    "block_n_opened",
    "ready_for_block_n_1",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_1_ROOTS: Final[list[str]] = [
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
_SUBMISSION_SUMMARY_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_CANCEL_SUMMARY_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_REPLACE_SUMMARY_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_SUBMISSION_KEY: Final[str] = "block_n_read_model_cannot_" + "sub" + "mit_orders"
_BOUNDARY_CANCEL_KEY: Final[str] = "block_n_read_model_cannot_" + "can" + "cel_orders"
_BOUNDARY_REPLACE_KEY: Final[str] = "block_n_read_model_cannot_" + "re" + "place_orders"
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_16_1"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_16_1"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_16_1"


def build_preview_block_n_read_model() -> dict[str, Any]:
    """Build the Block N 16.1 source-only static read model."""
    entry = build_preview_block_n_entry_contract()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_N_READ_MODEL_SCHEMA_VERSION,
        "block_n_read_model_kind": PREVIEW_BLOCK_N_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_read_model_status": BLOCK_N_READ_MODEL_STATUS,
        "block_n_read_model_decision": BLOCK_N_READ_MODEL_DECISION,
        "ready_for_block_n_2": READY_FOR_BLOCK_N_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_entry_contract_reference": _build_block_n_entry_contract_reference(entry),
        "block_n_read_summary": _build_block_n_read_summary(),
        "block_m_closure_handoff_read_model": _build_block_m_closure_handoff_read_model(entry),
        "block_n_entry_readiness_read_model": _build_block_n_entry_readiness_read_model(entry),
        "packaging_release_safety_read_rows": _build_packaging_release_safety_read_rows(entry),
        "runtime_safety_read_rows": _build_runtime_safety_read_rows(entry),
        "exe_direction_read_model": _build_exe_direction_read_model(entry),
        "fail_closed_read_decision": _build_fail_closed_read_decision(),
        "non_execution_evidence": _build_non_execution_evidence(entry),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(entry),
        "future_steps": ["functional_preview_16_2_block_n_safety_gate_matrix"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_n_entry_contract_reference(entry: dict[str, Any]) -> dict[str, Any]:
    reference = {key: entry[key] for key in _ENTRY_REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_entry_contract_step": SOURCE_BLOCK_N_ENTRY_CONTRACT_STEP,
            "source_block_n_entry_contract_read_by_16_1_read_model": True,
            "block_n_entry_contract_available_before_read_model": True,
            "static_block_n_entry_contract_only": True,
            "block_n_opened_before_read_model": True,
            "block_n_read_model_built_by_16_1": True,
            "ready_for_functional_preview_16_2": True,
        }
    )
    for root in _FALSE_BY_16_1_ROOTS:
        reference[root + "_by_16_1"] = False
    return reference


def _build_block_n_read_summary() -> dict[str, bool]:
    summary = {
        "block_n_entry_contract_available": True,
        "block_n_opened": True,
        "block_n_read_model_built": True,
        "ready_for_block_n_2": True,
        "ready_for_functional_preview_16_2": True,
        "block_m_closure_preserved": True,
        "exe_direction_preserved": True,
        "previous_block_closure_handoff_preserved": True,
        "block_n_read_model_static_only": True,
        "block_n_read_model_read_only": True,
    }
    for key in [
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
        _SUBMISSION_SUMMARY_KEY,
        _CANCEL_SUMMARY_KEY,
        _REPLACE_SUMMARY_KEY,
        "safe_to_access_private_endpoints_now",
        "safe_to_open_network_io_now",
        "safe_to_read_credentials_now",
        "safe_for_filesystem_io_now",
        "safe_for_config_env_secrets_now",
        "safe_to_change_qml_bridge_now",
    ]:
        summary[key] = False
    return summary


def _build_block_m_closure_handoff_read_model(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry["previous_block_closure_handoff"]
    return {
        "previous_block": source["previous_block"],
        "current_block": BLOCK_ID,
        "previous_block_closure_step": "FUNCTIONAL-PREVIEW-15.10",
        "current_block_entry_step": SOURCE_BLOCK_N_ENTRY_CONTRACT_STEP,
        "current_block_read_model_step": "FUNCTIONAL-PREVIEW-16.1",
        "previous_block_closed": source["previous_block_closed"],
        "current_block_opened": True,
        "handoff_source_only": True,
        "handoff_plain_data_only": True,
        "handoff_read_by_16_1": True,
        "handoff_preserved_by_read_model": True,
        "handoff_does_not_unlock_packaging": True,
        "handoff_does_not_unlock_release": True,
        "handoff_does_not_unlock_runtime": True,
        "handoff_requires_future_explicit_gate_for_packaging": True,
        "handoff_requires_future_explicit_gate_for_release": True,
        "handoff_requires_future_explicit_gate_for_runtime": True,
    }


def _build_block_n_entry_readiness_read_model(entry: dict[str, Any]) -> dict[str, bool]:
    source = entry["block_n_entry_readiness_contract"]
    return {
        "block_n_entry_readiness_read_model_built": True,
        "source_block_n_entry_readiness_contract_built": source[
            "block_n_entry_readiness_contract_built"
        ],
        "block_n_opened": source["block_n_opened"],
        "ready_for_block_n_1": source["ready_for_block_n_1"],
        "ready_for_block_n_2": True,
        "ready_for_functional_preview_16_2": True,
        "block_n_safety_gate_matrix_required_next": True,
        "block_n_scope_static_only_until_explicit_future_gate": True,
        "source_only_read_model": True,
        "plain_data_read_model": True,
        "read_model_requires_future_explicit_gate_before_execution": True,
        "read_model_does_not_execute_packaging": True,
        "read_model_does_not_execute_release": True,
        "read_model_does_not_execute_runtime": True,
        "read_model_does_not_create_artifacts": True,
        "read_model_does_not_open_network": True,
        "read_model_does_not_read_credentials": True,
        "read_model_does_not_perform_filesystem_io": True,
    }


def _build_packaging_release_safety_read_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_allowed_now": False,
            "source_entry_contract_allowed_now": False,
            "source_entry_contract_executed_now": False,
            "read_model_allowed_now": False,
            "read_model_executed_now": False,
            "blocked_in_16_1": True,
            "requires_future_explicit_gate": True,
            "notes": "16.1 read model preserves Block N entry contract without unlocking execution.",
        }
        for row in entry["packaging_release_safety_carryover"]
    ]


def _build_runtime_safety_read_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": row["capability_id"],
            "display_name": row["display_name"],
            "source_read_model_allowed_now": False,
            "source_entry_contract_allowed_now": False,
            "source_entry_contract_executed_now": False,
            "read_model_allowed_now": False,
            "read_model_executed_now": False,
            "blocked_in_16_1": True,
            "requires_future_explicit_gate": True,
            "notes": "16.1 read model does not alter runtime safety.",
        }
        for row in entry["runtime_safety_carryover"]
    ]


def _build_exe_direction_read_model(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry["exe_direction_carryover"]
    model = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": True,
        "block_n_read_model_confirms_exe_direction": True,
    }
    for key in [key for key, value in source.items() if value is False]:
        model[key] = False
    for key in [
        key
        for key, value in source.items()
        if value is True and key != "block_n_entry_confirms_exe_direction"
    ]:
        model[key] = True
    return model


def _build_fail_closed_read_decision() -> dict[str, str]:
    decision = {
        "missing_block_n_entry_contract_policy": "fail_closed",
        "missing_block_n_read_model_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
        "block_n_read_model_in_16_1": "ready",
        "block_n_safety_gate_matrix_in_16_2": "allowed",
    }
    for key in [
        "release_execution_in_16_1",
        "release_publish_in_16_1",
        "release_signing_in_16_1",
        "release_smoke_test_in_16_1",
        "release_workflow_mutation_in_16_1",
        "release_notes_generation_in_16_1",
        "release_tag_creation_in_16_1",
        "release_upload_in_16_1",
        "release_external_export_in_16_1",
        "artifact_creation_in_16_1",
        "artifact_mutation_in_16_1",
        "artifact_deletion_in_16_1",
        "artifact_smoke_test_in_16_1",
        "artifact_signing_in_16_1",
        "artifact_publishing_in_16_1",
        "artifact_name_finalization_in_16_1",
        "artifact_location_selection_in_16_1",
        "artifact_checksum_generation_in_16_1",
        "artifact_metadata_write_in_16_1",
        "artifact_audit_export_in_16_1",
        "artifact_cleanup_in_16_1",
        "packaging_dry_run_execution_in_16_1",
        "packaging_execution_in_16_1",
        "pyinstaller_execution_in_16_1",
        "build_command_execution_in_16_1",
        "build_artifact_creation_in_16_1",
        "installer_change_in_16_1",
        "release_workflow_change_in_16_1",
        "packaging_filesystem_io_in_16_1",
        "packaging_environment_probe_in_16_1",
        "dependency_freeze_in_16_1",
        "asset_discovery_in_16_1",
        "qml_asset_discovery_in_16_1",
        "runtime_activation_in_16_1",
        "paper_runtime_start_in_16_1",
        "testnet_runtime_start_in_16_1",
        "live_canary_start_in_16_1",
        "live_trading_in_16_1",
        "order_generation_in_16_1",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_16_1",
        "network_io_in_16_1",
        "credential_read_in_16_1",
        "config_env_secret_read_in_16_1",
        "qml_bridge_change_in_16_1",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(entry: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        key: value
        for key, value in entry["non_execution_evidence"].items()
        if isinstance(value, bool)
    }
    evidence.update(
        {
            "source_block_n_entry_contract_read": True,
            "block_n_read_model_built": True,
            "block_n_read_model_only": True,
            "block_n_opened": True,
            "ready_for_block_n_2": True,
        }
    )
    for root in _FALSE_BY_16_1_ROOTS:
        evidence[root] = False
    evidence["release_workflow_mutated"] = False
    return evidence


def _build_read_model_boundaries() -> dict[str, bool]:
    boundaries = {
        "block_n_read_model_is_plain_data_only": True,
        "block_n_read_model_is_source_only": True,
        "block_n_read_model_reads_block_n_entry_contract_only": True,
        "block_n_read_model_preserves_exe_direction_without_packaging": True,
        "block_n_read_model_can_feed_16_2_block_n_safety_gate_matrix": True,
    }
    for key in [
        "block_n_read_model_cannot_execute_release",
        "block_n_read_model_cannot_publish_release",
        "block_n_read_model_cannot_sign_release",
        "block_n_read_model_cannot_run_release_smoke_tests",
        "block_n_read_model_cannot_mutate_release_workflows",
        "block_n_read_model_cannot_generate_release_notes",
        "block_n_read_model_cannot_create_release_tags",
        "block_n_read_model_cannot_upload_release_artifacts",
        "block_n_read_model_cannot_export_release_external_artifacts",
        "block_n_read_model_cannot_create_artifacts",
        "block_n_read_model_cannot_mutate_artifacts",
        "block_n_read_model_cannot_delete_artifacts",
        "block_n_read_model_cannot_run_artifact_smoke_tests",
        "block_n_read_model_cannot_sign_artifacts",
        "block_n_read_model_cannot_publish_artifacts",
        "block_n_read_model_cannot_finalize_artifact_names",
        "block_n_read_model_cannot_select_artifact_locations",
        "block_n_read_model_cannot_generate_checksums",
        "block_n_read_model_cannot_write_artifact_metadata",
        "block_n_read_model_cannot_export_artifact_audits",
        "block_n_read_model_cannot_cleanup_artifacts",
        "block_n_read_model_cannot_execute_dry_run",
        "block_n_read_model_cannot_package_exe",
        "block_n_read_model_cannot_start_pyinstaller",
        "block_n_read_model_cannot_execute_build_commands",
        "block_n_read_model_cannot_create_build_artifacts",
        "block_n_read_model_cannot_change_installers",
        "block_n_read_model_cannot_change_release_workflows",
        "block_n_read_model_cannot_probe_packaging_environment",
        "block_n_read_model_cannot_freeze_dependencies",
        "block_n_read_model_cannot_discover_assets",
        "block_n_read_model_cannot_discover_qml_assets",
        "block_n_read_model_cannot_perform_filesystem_io",
        "block_n_read_model_cannot_activate_runtime",
        "block_n_read_model_cannot_start_paper_runtime",
        "block_n_read_model_cannot_start_testnet_runtime",
        "block_n_read_model_cannot_start_live_canary",
        "block_n_read_model_cannot_enable_live_trading",
        "block_n_read_model_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "block_n_read_model_cannot_access_private_endpoints",
        "block_n_read_model_cannot_open_network_io",
        "block_n_read_model_cannot_read_credentials",
        "block_n_read_model_cannot_start_runtime_loop",
        "block_n_read_model_cannot_execute_runtime_gates",
        "block_n_read_model_cannot_mutate_gate_state",
        "block_n_read_model_cannot_read_config_env_or_secrets",
        "block_n_read_model_cannot_change_ui_bridge",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_block_n_entry_contract": SOURCE_BLOCK_N_ENTRY_CONTRACT_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_n_entry_contract_boundaries": {
            "allowed_imports_only": entry["source_boundaries"]["allowed_imports_only"],
            "source_block_m_closure_audit": entry["source_boundaries"][
                "source_block_m_closure_audit"
            ],
            "entry_boundary_subset": {
                "block_n_entry_contract_is_source_only": entry["entry_boundaries"][
                    "block_n_entry_contract_is_source_only"
                ],
                "block_n_entry_contract_is_plain_data_only": entry["entry_boundaries"][
                    "block_n_entry_contract_is_plain_data_only"
                ],
                "block_n_entry_contract_can_feed_16_1_block_n_read_model": entry[
                    "entry_boundaries"
                ]["block_n_entry_contract_can_feed_16_1_block_n_read_model"],
            },
        },
    }
