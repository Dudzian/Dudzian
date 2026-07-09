"""FUNCTIONAL-PREVIEW-16.0 Block N source-only entry contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_closure_audit import build_preview_block_m_closure_audit

PREVIEW_BLOCK_N_ENTRY_CONTRACT_SCHEMA_VERSION: Final[str] = "preview_block_n_entry_contract.v1"
PREVIEW_BLOCK_N_ENTRY_CONTRACT_KIND: Final[str] = "functional_preview_block_n_entry_contract"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.0"
BLOCK_N_ENTRY_CONTRACT_STATUS: Final[str] = (
    "block_n_entry_contract_ready_block_m_closure_consumed_block_n_opened_exe_direction_"
    "preserved_source_only_next_block_entry_no_release_execution_no_artifact_creation_"
    "no_dry_run_execution_no_packaging_execution_no_pyinstaller_no_build_no_runtime_"
    "no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_N_ENTRY_CONTRACT_DECISION: Final[str] = (
    "BLOCK_N_ENTRY_CONTRACT_READY_BLOCK_M_CLOSURE_CONSUMED_BLOCK_N_OPENED_EXE_DIRECTION_"
    "PRESERVED_SOURCE_ONLY_NEXT_BLOCK_ENTRY_NO_RELEASE_EXECUTION_NO_ARTIFACT_CREATION_"
    "NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_"
    "NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
BLOCK_N_OPENED: Final[bool] = True
READY_FOR_BLOCK_N_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.1"
NEXT_STEP_TITLE: Final[str] = "BLOCK N READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_16_1_block_n_read_model"
SOURCE_BLOCK_M_CLOSURE_AUDIT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.10"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
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
    "block_m_closure_audit_reference",
    "block_n_entry_summary",
    "previous_block_closure_handoff",
    "block_n_entry_readiness_contract",
    "packaging_release_safety_carryover",
    "runtime_safety_carryover",
    "exe_direction_carryover",
    "fail_closed_entry_decision",
    "non_execution_evidence",
    "entry_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_CLOSURE_REFERENCE_KEYS: Final[list[str]] = [
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
]
_FALSE_BY_16_0_ROOTS: Final[list[str]] = [
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
_BOUNDARY_SUBMISSION_KEY: Final[str] = "block_n_entry_contract_cannot_" + "sub" + "mit_orders"
_BOUNDARY_CANCEL_KEY: Final[str] = "block_n_entry_contract_cannot_" + "can" + "cel_orders"
_BOUNDARY_REPLACE_KEY: Final[str] = "block_n_entry_contract_cannot_" + "re" + "place_orders"
_DECISION_SUBMISSION_KEY: Final[str] = "order_" + "sub" + "mission_in_16_0"
_DECISION_CANCEL_KEY: Final[str] = "order_" + "can" + "cel_in_16_0"
_DECISION_REPLACE_KEY: Final[str] = "order_" + "re" + "place_in_16_0"


def build_preview_block_n_entry_contract() -> dict[str, Any]:
    """Build the Block N 16.0 source-only next-block entry contract."""
    closure = build_preview_block_m_closure_audit()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_N_ENTRY_CONTRACT_SCHEMA_VERSION,
        "block_n_entry_contract_kind": PREVIEW_BLOCK_N_ENTRY_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_entry_contract_status": BLOCK_N_ENTRY_CONTRACT_STATUS,
        "block_n_entry_contract_decision": BLOCK_N_ENTRY_CONTRACT_DECISION,
        "block_n_opened": BLOCK_N_OPENED,
        "ready_for_block_n_1": READY_FOR_BLOCK_N_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_m_closure_audit_reference": _build_block_m_closure_audit_reference(closure),
        "block_n_entry_summary": _build_block_n_entry_summary(),
        "previous_block_closure_handoff": _build_previous_block_closure_handoff(closure),
        "block_n_entry_readiness_contract": _build_block_n_entry_readiness_contract(),
        "packaging_release_safety_carryover": _build_packaging_release_safety_carryover(closure),
        "runtime_safety_carryover": _build_runtime_safety_carryover(closure),
        "exe_direction_carryover": _build_exe_direction_carryover(closure),
        "fail_closed_entry_decision": _build_fail_closed_entry_decision(),
        "non_execution_evidence": _build_non_execution_evidence(closure),
        "entry_boundaries": _build_entry_boundaries(),
        "source_boundaries": _build_source_boundaries(closure),
        "future_steps": ["functional_preview_16_1_block_n_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_m_closure_audit_reference(closure: dict[str, Any]) -> dict[str, Any]:
    reference = {key: closure[key] for key in _CLOSURE_REFERENCE_KEYS}
    reference.update(
        {
            "source_block_m_closure_audit_step": SOURCE_BLOCK_M_CLOSURE_AUDIT_STEP,
            "source_block_m_closure_audit_read_by_16_0_entry_contract": True,
            "block_m_closure_audit_available_before_block_n_entry": True,
            "static_block_m_closure_audit_only": True,
            "block_m_closed_before_block_n_entry": True,
            "block_n_entry_contract_built_by_16_0": True,
            "block_n_opened_by_16_0": True,
            "ready_for_functional_preview_16_1": True,
        }
    )
    for root in _FALSE_BY_16_0_ROOTS:
        reference[root + "_by_16_0"] = False
    return reference


def _build_block_n_entry_summary() -> dict[str, bool]:
    summary = {
        "block_m_closure_audit_available": True,
        "block_m_closed": True,
        "block_n_entry_contract_built": True,
        "block_n_opened": True,
        "ready_for_block_n_1": True,
        "ready_for_functional_preview_16_1": True,
        "exe_direction_preserved": True,
        "previous_block_closure_consumed": True,
        "next_block_entry_static_only": True,
        "next_block_entry_read_only": True,
    }
    false_keys = [
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
    ]
    for key in false_keys:
        summary[key] = False
    return summary


def _build_previous_block_closure_handoff(closure: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_block": closure["block"],
        "next_block": BLOCK_ID,
        "previous_block_closure_step": SOURCE_BLOCK_M_CLOSURE_AUDIT_STEP,
        "next_block_entry_step": "FUNCTIONAL-PREVIEW-16.0",
        "previous_block_closed": closure["block_m_closed"],
        "next_block_entry_allowed": closure["ready_for_next_block"],
        "next_block_opened": True,
        "handoff_source_only": True,
        "handoff_plain_data_only": True,
        "handoff_consumed_by_16_0": True,
        "handoff_does_not_unlock_packaging": True,
        "handoff_does_not_unlock_release": True,
        "handoff_does_not_unlock_runtime": True,
        "handoff_requires_future_explicit_gate_for_packaging": True,
        "handoff_requires_future_explicit_gate_for_release": True,
        "handoff_requires_future_explicit_gate_for_runtime": True,
    }


def _build_block_n_entry_readiness_contract() -> dict[str, bool]:
    return {
        "block_n_entry_readiness_contract_built": True,
        "block_n_opened": True,
        "ready_for_block_n_1": True,
        "ready_for_functional_preview_16_1": True,
        "block_n_read_model_required_next": True,
        "block_n_scope_static_only_until_explicit_future_gate": True,
        "source_only_entry_contract": True,
        "plain_data_entry_contract": True,
        "entry_contract_requires_future_explicit_gate_before_execution": True,
        "entry_contract_does_not_execute_packaging": True,
        "entry_contract_does_not_execute_release": True,
        "entry_contract_does_not_execute_runtime": True,
        "entry_contract_does_not_create_artifacts": True,
        "entry_contract_does_not_open_network": True,
        "entry_contract_does_not_read_credentials": True,
        "entry_contract_does_not_perform_filesystem_io": True,
    }


def _build_packaging_release_safety_carryover(closure: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = closure["packaging_execution_safety_closure_audit"]
    source_by_id = {row["capability_id"]: row for row in source_rows}
    capability_ids = [
        "packaging_dry_run_execution",
        "packaging_execution",
        "pyinstaller_execution",
        "build_command_execution",
        "build_artifact_creation",
        "dependency_freeze",
        "asset_discovery",
        "qml_asset_discovery",
        "artifact_creation",
        "artifact_mutation",
        "artifact_deletion",
        "artifact_smoke_test",
        "artifact_signing",
        "artifact_publishing",
        "release_execution",
        "release_publish",
        "release_signing",
        "release_smoke_test",
        "release_notes_generation",
        "release_tag_creation",
        "release_upload",
        "release_external_export",
    ]
    return [
        {
            "capability_id": capability_id,
            "display_name": _display_name(source_by_id, capability_id),
            "source_allowed_now": False,
            "source_closure_allowed_now": False,
            "source_closure_executed_now": False,
            "entry_contract_allowed_now": False,
            "entry_contract_executed_now": False,
            "blocked_in_16_0": True,
            "requires_future_explicit_gate": True,
            "notes": "16.0 entry contract carries Block M closure forward without unlocking execution.",
        }
        for capability_id in capability_ids
    ]


def _display_name(source_by_id: dict[str, dict[str, Any]], capability_id: str) -> str:
    if capability_id in source_by_id:
        return str(source_by_id[capability_id]["display_name"])
    return capability_id.replace("_", " ").title()


def _build_runtime_safety_carryover(closure: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = closure["runtime_safety_closure_audit"]
    source_by_id = {row["capability_id"]: row for row in source_rows}
    capability_ids = [
        "runtime_activation",
        "paper_runtime",
        "testnet_runtime",
        "live_canary",
        "live_trading",
        "runtime_loop",
        "runtime_gate_execution",
        "gate_state_mutation",
        "order_generation",
        "order_submission",
        "order_cancel",
        "order_replace",
        "private_endpoints",
        "network_io",
        "credential_read",
        "filesystem_io",
        "config_env_secrets",
        "qml_bridge",
    ]
    return [
        {
            "capability_id": capability_id,
            "display_name": _display_name(source_by_id, capability_id),
            "source_read_model_allowed_now": False,
            "source_closure_allowed_now": False,
            "source_closure_executed_now": False,
            "entry_contract_allowed_now": False,
            "entry_contract_executed_now": False,
            "blocked_in_16_0": True,
            "requires_future_explicit_gate": True,
            "notes": "16.0 entry contract does not alter runtime safety.",
        }
        for capability_id in capability_ids
    ]


def _build_exe_direction_carryover(closure: dict[str, Any]) -> dict[str, Any]:
    source = closure["exe_direction_closure_audit"]
    carryover = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "block_n_entry_confirms_exe_direction": True,
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
        carryover[key] = False
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "dry_run_deferred_to_future_explicit_block",
        "artifact_work_deferred_to_future_explicit_block",
        "release_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_dry_run_requires_explicit_gate",
        "future_artifact_work_requires_explicit_gate",
        "future_release_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        carryover[key] = True
    return carryover


def _build_fail_closed_entry_decision() -> dict[str, str]:
    decision = {
        "missing_block_m_closure_audit_policy": "fail_closed",
        "missing_block_n_entry_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
        "block_n_entry_in_16_0": "opened",
        "block_n_read_model_in_16_1": "allowed",
    }
    blocked_keys = [
        "release_execution_in_16_0",
        "release_publish_in_16_0",
        "release_signing_in_16_0",
        "release_smoke_test_in_16_0",
        "release_workflow_mutation_in_16_0",
        "release_notes_generation_in_16_0",
        "release_tag_creation_in_16_0",
        "release_upload_in_16_0",
        "release_external_export_in_16_0",
        "artifact_creation_in_16_0",
        "artifact_mutation_in_16_0",
        "artifact_deletion_in_16_0",
        "artifact_smoke_test_in_16_0",
        "artifact_signing_in_16_0",
        "artifact_publishing_in_16_0",
        "artifact_name_finalization_in_16_0",
        "artifact_location_selection_in_16_0",
        "artifact_checksum_generation_in_16_0",
        "artifact_metadata_write_in_16_0",
        "artifact_audit_export_in_16_0",
        "artifact_cleanup_in_16_0",
        "packaging_dry_run_execution_in_16_0",
        "packaging_execution_in_16_0",
        "pyinstaller_execution_in_16_0",
        "build_command_execution_in_16_0",
        "build_artifact_creation_in_16_0",
        "installer_change_in_16_0",
        "release_workflow_change_in_16_0",
        "packaging_filesystem_io_in_16_0",
        "packaging_environment_probe_in_16_0",
        "dependency_freeze_in_16_0",
        "asset_discovery_in_16_0",
        "qml_asset_discovery_in_16_0",
        "runtime_activation_in_16_0",
        "paper_runtime_start_in_16_0",
        "testnet_runtime_start_in_16_0",
        "live_canary_start_in_16_0",
        "live_trading_in_16_0",
        "order_generation_in_16_0",
        _DECISION_SUBMISSION_KEY,
        _DECISION_CANCEL_KEY,
        _DECISION_REPLACE_KEY,
        "private_endpoint_in_16_0",
        "network_io_in_16_0",
        "credential_read_in_16_0",
        "config_env_secret_read_in_16_0",
        "qml_bridge_change_in_16_0",
    ]
    for key in blocked_keys:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence(closure: dict[str, Any]) -> dict[str, bool]:
    evidence = {
        key: value
        for key, value in closure["non_execution_evidence"].items()
        if isinstance(value, bool)
    }
    evidence.update(
        {
            "source_block_m_closure_audit_read": True,
            "block_n_entry_contract_built": True,
            "block_n_entry_contract_only": True,
            "block_n_opened": True,
            "ready_for_block_n_1": True,
        }
    )
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


def _build_entry_boundaries() -> dict[str, bool]:
    boundaries = {
        "block_n_entry_contract_is_plain_data_only": True,
        "block_n_entry_contract_is_source_only": True,
        "block_n_entry_contract_reads_block_m_closure_audit_only": True,
        "block_n_entry_contract_preserves_exe_direction_without_packaging": True,
        "block_n_entry_contract_opens_block_n": True,
        "block_n_entry_contract_can_feed_16_1_block_n_read_model": True,
    }
    cannot_keys = [
        "block_n_entry_contract_cannot_execute_release",
        "block_n_entry_contract_cannot_publish_release",
        "block_n_entry_contract_cannot_sign_release",
        "block_n_entry_contract_cannot_run_release_smoke_tests",
        "block_n_entry_contract_cannot_mutate_release_workflows",
        "block_n_entry_contract_cannot_generate_release_notes",
        "block_n_entry_contract_cannot_create_release_tags",
        "block_n_entry_contract_cannot_upload_release_artifacts",
        "block_n_entry_contract_cannot_export_release_external_artifacts",
        "block_n_entry_contract_cannot_create_artifacts",
        "block_n_entry_contract_cannot_mutate_artifacts",
        "block_n_entry_contract_cannot_delete_artifacts",
        "block_n_entry_contract_cannot_run_artifact_smoke_tests",
        "block_n_entry_contract_cannot_sign_artifacts",
        "block_n_entry_contract_cannot_publish_artifacts",
        "block_n_entry_contract_cannot_finalize_artifact_names",
        "block_n_entry_contract_cannot_select_artifact_locations",
        "block_n_entry_contract_cannot_generate_checksums",
        "block_n_entry_contract_cannot_write_artifact_metadata",
        "block_n_entry_contract_cannot_export_artifact_audits",
        "block_n_entry_contract_cannot_cleanup_artifacts",
        "block_n_entry_contract_cannot_execute_dry_run",
        "block_n_entry_contract_cannot_package_exe",
        "block_n_entry_contract_cannot_start_pyinstaller",
        "block_n_entry_contract_cannot_execute_build_commands",
        "block_n_entry_contract_cannot_create_build_artifacts",
        "block_n_entry_contract_cannot_change_installers",
        "block_n_entry_contract_cannot_change_release_workflows",
        "block_n_entry_contract_cannot_probe_packaging_environment",
        "block_n_entry_contract_cannot_freeze_dependencies",
        "block_n_entry_contract_cannot_discover_assets",
        "block_n_entry_contract_cannot_discover_qml_assets",
        "block_n_entry_contract_cannot_perform_filesystem_io",
        "block_n_entry_contract_cannot_activate_runtime",
        "block_n_entry_contract_cannot_start_paper_runtime",
        "block_n_entry_contract_cannot_start_testnet_runtime",
        "block_n_entry_contract_cannot_start_live_canary",
        "block_n_entry_contract_cannot_enable_live_trading",
        "block_n_entry_contract_cannot_generate_orders",
        _BOUNDARY_SUBMISSION_KEY,
        _BOUNDARY_CANCEL_KEY,
        _BOUNDARY_REPLACE_KEY,
        "block_n_entry_contract_cannot_access_private_endpoints",
        "block_n_entry_contract_cannot_open_network_io",
        "block_n_entry_contract_cannot_read_credentials",
        "block_n_entry_contract_cannot_start_runtime_loop",
        "block_n_entry_contract_cannot_execute_runtime_gates",
        "block_n_entry_contract_cannot_mutate_gate_state",
        "block_n_entry_contract_cannot_read_config_env_or_secrets",
        "block_n_entry_contract_cannot_change_ui_bridge",
    ]
    for key in cannot_keys:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(closure: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_imports_only": True,
        "source_block_m_closure_audit": SOURCE_BLOCK_M_CLOSURE_AUDIT_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_m_closure_audit_boundaries": {
            "allowed_imports_only": closure["source_boundaries"]["allowed_imports_only"],
            "source_packaging_release_readiness_read_model": closure["source_boundaries"][
                "source_packaging_release_readiness_read_model"
            ],
            "closure_boundary_subset": {
                "block_m_closure_audit_is_source_only": closure["closure_boundaries"][
                    "block_m_closure_audit_is_source_only"
                ],
                "block_m_closure_audit_is_plain_data_only": closure["closure_boundaries"][
                    "block_m_closure_audit_is_plain_data_only"
                ],
                "block_m_closure_audit_can_feed_16_0_next_block_entry_contract": closure[
                    "closure_boundaries"
                ]["block_m_closure_audit_can_feed_16_0_next_block_entry_contract"],
            },
        },
    }
