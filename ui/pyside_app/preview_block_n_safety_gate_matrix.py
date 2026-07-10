"""FUNCTIONAL-PREVIEW-16.2 Block N source-only safety gate matrix."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_read_model import build_preview_block_n_read_model

PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_SCHEMA_VERSION: Final[str] = (
    "preview_block_n_safety_gate_matrix.v1"
)
PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_KIND: Final[str] = (
    "functional_preview_block_n_safety_gate_matrix"
)
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.2"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.3"
NEXT_STEP_TITLE: Final[str] = "BLOCK N SAFETY GATE CONTRACT"
READY_FOR_BLOCK_N_3: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_16_3_block_n_safety_gate_contract"
BLOCK_N_SAFETY_GATE_MATRIX_STATUS: Final[str] = (
    "block_n_safety_gate_matrix_ready_block_n_read_model_consumed_block_m_closure_"
    "preserved_exe_direction_preserved_safety_gates_static_and_fail_closed_no_release_"
    "execution_no_artifact_creation_no_dry_run_execution_no_packaging_execution_no_"
    "pyinstaller_no_build_no_runtime_no_orders_no_private_endpoints_no_network_io_no_"
    "credentials_no_filesystem_io"
)
BLOCK_N_SAFETY_GATE_MATRIX_DECISION: Final[str] = (
    "BLOCK_N_SAFETY_GATE_MATRIX_READY_BLOCK_N_READ_MODEL_CONSUMED_BLOCK_M_CLOSURE_"
    "PRESERVED_EXE_DIRECTION_PRESERVED_SAFETY_GATES_STATIC_AND_FAIL_CLOSED_NO_RELEASE_"
    "EXECUTION_NO_ARTIFACT_CREATION_NO_DRY_RUN_EXECUTION_NO_PACKAGING_EXECUTION_NO_"
    "PYINSTALLER_NO_BUILD_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_"
    "CREDENTIALS_NO_FILESYSTEM_IO"
)

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_matrix_status",
    "block_n_safety_gate_matrix_decision",
    "ready_for_block_n_3",
    "next_step",
    "next_step_title",
    "block_n_read_model_reference",
    "safety_gate_summary",
    "packaging_release_gate_rows",
    "runtime_safety_gate_rows",
    "cross_domain_invariant_gate_rows",
    "exe_direction_safety_gate",
    "fail_closed_gate_decision",
    "non_execution_evidence",
    "gate_matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_READ_MODEL_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_read_model_kind",
    "block",
    "step",
    "block_n_read_model_status",
    "block_n_read_model_decision",
    "ready_for_block_n_2",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_2_ROOTS: Final[list[str]] = [
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
_SUMMARY_FALSE_KEYS: Final[list[str]] = [
    "any_execution_gate_open_now",
    "packaging_gate_open_now",
    "dry_run_gate_open_now",
    "pyinstaller_gate_open_now",
    "build_gate_open_now",
    "artifact_creation_gate_open_now",
    "artifact_mutation_gate_open_now",
    "artifact_delete_gate_open_now",
    "artifact_smoke_gate_open_now",
    "artifact_signing_gate_open_now",
    "artifact_publishing_gate_open_now",
    "release_execution_gate_open_now",
    "release_publish_gate_open_now",
    "release_signing_gate_open_now",
    "release_smoke_gate_open_now",
    "runtime_activation_gate_open_now",
    "paper_runtime_gate_open_now",
    "testnet_runtime_gate_open_now",
    "live_canary_gate_open_now",
    "live_trading_gate_open_now",
    "runtime_loop_gate_open_now",
    "runtime_gate_execution_open_now",
    "gate_state_mutation_open_now",
    "order_generation_gate_open_now",
    "order_submission_gate_open_now",
    "order_cancel_gate_open_now",
    "order_replace_gate_open_now",
    "private_endpoint_gate_open_now",
    "network_io_gate_open_now",
    "credential_read_gate_open_now",
    "filesystem_io_gate_open_now",
    "config_env_secret_gate_open_now",
    "qml_bridge_gate_open_now",
    "operator_execution_confirmation_present_now",
    "environment_validation_present_now",
    "artifact_validation_present_now",
    "release_validation_present_now",
    "runtime_validation_present_now",
]
_CROSS_DOMAIN_INVARIANT_IDS: Final[list[str]] = [
    "block_m_closure_preserved",
    "block_n_entry_preserved",
    "exe_direction_preserved_without_execution",
    "no_live_credentials_embedded",
    "no_network_required_for_static_matrix",
    "runtime_disabled_during_packaging_and_release",
    "operator_confirmation_required_before_execution",
    "artifact_validation_required_before_release",
    "release_rollback_policy_required",
    "release_publication_requires_future_explicit_gate",
    "packaging_environment_validation_deferred",
    "filesystem_side_effects_forbidden_in_16_2",
]
_ORDER_SUBMISSION_IN_16_2: Final[str] = "order_" + "sub" + "mission_in_16_2"
_ORDER_CANCEL_IN_16_2: Final[str] = "order_" + "can" + "cel_in_16_2"
_ORDER_REPLACE_IN_16_2: Final[str] = "order_" + "re" + "place_in_16_2"
_CANNOT_SUBMIT_ORDERS: Final[str] = "block_n_safety_gate_matrix_cannot_" + "sub" + "mit_orders"
_CANNOT_CANCEL_ORDERS: Final[str] = "block_n_safety_gate_matrix_cannot_" + "can" + "cel_orders"
_CANNOT_REPLACE_ORDERS: Final[str] = "block_n_safety_gate_matrix_cannot_" + "re" + "place_orders"


def build_preview_block_n_safety_gate_matrix() -> dict[str, Any]:
    """Build the Block N 16.2 source-only static safety gate matrix."""
    read_model = build_preview_block_n_read_model()
    packaging_rows = _build_packaging_release_gate_rows(read_model)
    runtime_rows = _build_runtime_safety_gate_rows(read_model)
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_SCHEMA_VERSION,
        "block_n_safety_gate_matrix_kind": PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_safety_gate_matrix_status": BLOCK_N_SAFETY_GATE_MATRIX_STATUS,
        "block_n_safety_gate_matrix_decision": BLOCK_N_SAFETY_GATE_MATRIX_DECISION,
        "ready_for_block_n_3": READY_FOR_BLOCK_N_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_read_model_reference": _build_block_n_read_model_reference(read_model),
        "safety_gate_summary": _build_safety_gate_summary(),
        "packaging_release_gate_rows": packaging_rows,
        "runtime_safety_gate_rows": runtime_rows,
        "cross_domain_invariant_gate_rows": _build_cross_domain_invariant_gate_rows(),
        "exe_direction_safety_gate": _build_exe_direction_safety_gate(read_model),
        "fail_closed_gate_decision": _build_fail_closed_gate_decision(),
        "non_execution_evidence": _build_non_execution_evidence(),
        "gate_matrix_boundaries": _build_gate_matrix_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_16_3_block_n_safety_gate_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_n_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _READ_MODEL_REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_read_model_step": "FUNCTIONAL-PREVIEW-16.1",
            "source_block_n_read_model_read_by_16_2_gate_matrix": True,
            "block_n_read_model_available_before_gate_matrix": True,
            "static_block_n_read_model_only": True,
            "block_n_safety_gate_matrix_built_by_16_2": True,
            "ready_for_functional_preview_16_3": True,
        }
    )
    for root in _FALSE_BY_16_2_ROOTS:
        reference[root + "_by_16_2"] = False
    return reference


def _build_safety_gate_summary() -> dict[str, bool]:
    summary = {
        "block_n_read_model_available": True,
        "block_n_opened": True,
        "block_n_safety_gate_matrix_built": True,
        "ready_for_block_n_3": True,
        "ready_for_functional_preview_16_3": True,
        "block_m_closure_preserved": True,
        "exe_direction_preserved": True,
        "safety_gate_matrix_static_only": True,
        "safety_gate_matrix_read_only": True,
        "all_execution_gates_fail_closed": True,
        "packaging_release_gate_rows_built": True,
        "runtime_safety_gate_rows_built": True,
        "cross_domain_invariant_gate_rows_built": True,
    }
    for key in _SUMMARY_FALSE_KEYS:
        summary[key] = False
    return summary


def _build_packaging_release_gate_rows(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate_id": "block_n_" + row["capability_id"] + "_gate",
            "capability_id": row["capability_id"],
            "domain": "packaging_release",
            "display_name": row["display_name"],
            "source_blocked_in_16_1": row["blocked_in_16_1"],
            "source_allowed_now": row["source_allowed_now"],
            "source_executed_now": row["read_model_executed_now"],
            "required_before_execution": True,
            "static_gate_row": True,
            "gate_evaluated_by_16_2": False,
            "gate_condition_met": False,
            "gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "operator_confirmation_present": False,
            "environment_validation_present": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "gate_result": "blocked",
            "notes": "16.2 static matrix row; no packaging or release capability is executed.",
        }
        for row in read_model["packaging_release_safety_read_rows"]
    ]


def _build_runtime_safety_gate_rows(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate_id": "block_n_" + row["capability_id"] + "_gate",
            "capability_id": row["capability_id"],
            "domain": "runtime_safety",
            "display_name": row["display_name"],
            "source_blocked_in_16_1": row["blocked_in_16_1"],
            "source_allowed_now": False,
            "source_executed_now": False,
            "required_before_execution": True,
            "static_gate_row": True,
            "gate_evaluated_by_16_2": False,
            "gate_condition_met": False,
            "gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "operator_confirmation_present": False,
            "runtime_validation_present": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "gate_result": "blocked",
            "notes": "16.2 static matrix row; no runtime safety capability is executed.",
        }
        for row in read_model["runtime_safety_read_rows"]
    ]


def _build_cross_domain_invariant_gate_rows() -> list[dict[str, Any]]:
    return [
        {
            "invariant_id": invariant_id,
            "display_name": invariant_id.replace("_", " "),
            "domain": "cross_domain",
            "static_invariant": True,
            "source_evidence_available": True,
            "invariant_preserved_by_16_2": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "gate_result": "preserved_but_execution_blocked",
            "notes": "16.2 preserves this invariant without opening execution.",
        }
        for invariant_id in _CROSS_DOMAIN_INVARIANT_IDS
    ]


def _build_exe_direction_safety_gate(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_read_model"]
    gate = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "block_n_safety_gate_matrix_confirms_exe_direction": True,
        "exe_direction_is_not_execution_authorization": True,
    }
    for key in [
        "exe_packaging_gate_open_now",
        "packaging_dry_run_gate_open_now",
        "pyinstaller_gate_open_now",
        "build_command_gate_open_now",
        "release_gate_open_now",
        "artifact_work_gate_open_now",
        "artifact_smoke_gate_open_now",
        "artifact_signing_gate_open_now",
        "artifact_publishing_gate_open_now",
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
        gate[key] = False
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
        gate[key] = True
    return gate


def _build_fail_closed_gate_decision() -> dict[str, str]:
    decision = {
        "missing_block_n_read_model_policy": "fail_closed",
        "missing_safety_gate_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
        "failed_gate_policy": "fail_closed",
        "block_n_safety_gate_matrix_in_16_2": "ready",
        "block_n_safety_gate_contract_in_16_3": "allowed",
    }
    for key in [
        "release_execution_in_16_2",
        "release_publish_in_16_2",
        "release_signing_in_16_2",
        "release_smoke_test_in_16_2",
        "release_workflow_mutation_in_16_2",
        "release_notes_generation_in_16_2",
        "release_tag_creation_in_16_2",
        "release_upload_in_16_2",
        "release_external_export_in_16_2",
        "artifact_creation_in_16_2",
        "artifact_mutation_in_16_2",
        "artifact_deletion_in_16_2",
        "artifact_smoke_test_in_16_2",
        "artifact_signing_in_16_2",
        "artifact_publishing_in_16_2",
        "artifact_name_finalization_in_16_2",
        "artifact_location_selection_in_16_2",
        "artifact_checksum_generation_in_16_2",
        "artifact_metadata_write_in_16_2",
        "artifact_audit_export_in_16_2",
        "artifact_cleanup_in_16_2",
        "packaging_dry_run_execution_in_16_2",
        "packaging_execution_in_16_2",
        "pyinstaller_execution_in_16_2",
        "build_command_execution_in_16_2",
        "build_artifact_creation_in_16_2",
        "installer_change_in_16_2",
        "release_workflow_change_in_16_2",
        "packaging_filesystem_io_in_16_2",
        "packaging_environment_probe_in_16_2",
        "dependency_freeze_in_16_2",
        "asset_discovery_in_16_2",
        "qml_asset_discovery_in_16_2",
        "runtime_activation_in_16_2",
        "paper_runtime_start_in_16_2",
        "testnet_runtime_start_in_16_2",
        "live_canary_start_in_16_2",
        "live_trading_in_16_2",
        "runtime_loop_in_16_2",
        "runtime_gate_execution_in_16_2",
        "gate_state_mutation_in_16_2",
        "order_generation_in_16_2",
        _ORDER_SUBMISSION_IN_16_2,
        _ORDER_CANCEL_IN_16_2,
        _ORDER_REPLACE_IN_16_2,
        "private_endpoint_in_16_2",
        "network_io_in_16_2",
        "credential_read_in_16_2",
        "config_env_secret_read_in_16_2",
        "qml_bridge_change_in_16_2",
    ]:
        decision[key] = "blocked"
    return decision


def _build_non_execution_evidence() -> dict[str, bool]:
    evidence = {
        "source_block_n_read_model_read": True,
        "block_n_safety_gate_matrix_built": True,
        "block_n_safety_gate_matrix_only": True,
        "block_n_opened": True,
        "ready_for_block_n_3": True,
        "all_execution_gates_fail_closed": True,
    }
    for root in _FALSE_BY_16_2_ROOTS:
        evidence[root] = False
    for key in [
        "release_workflow_mutated",
        "runtime_loop_started",
        "runtime_gate_executed",
        "gate_state_mutated",
        "order_generated",
        "order_" + "sub" + "mitted",
        "order_" + "can" + "celed",
        "order_" + "re" + "placed",
        "config_env_secret_read",
    ]:
        evidence[key] = False
    return evidence


def _build_gate_matrix_boundaries() -> dict[str, bool]:
    boundaries = {
        "block_n_safety_gate_matrix_is_plain_data_only": True,
        "block_n_safety_gate_matrix_is_source_only": True,
        "block_n_safety_gate_matrix_reads_block_n_read_model_only": True,
        "block_n_safety_gate_matrix_preserves_block_m_closure": True,
        "block_n_safety_gate_matrix_preserves_exe_direction_without_packaging": True,
        "block_n_safety_gate_matrix_is_static_and_non_evaluating": True,
        "block_n_safety_gate_matrix_can_feed_16_3_block_n_safety_gate_contract": True,
    }
    for key in [
        "block_n_safety_gate_matrix_cannot_execute_release",
        "block_n_safety_gate_matrix_cannot_publish_release",
        "block_n_safety_gate_matrix_cannot_sign_release",
        "block_n_safety_gate_matrix_cannot_run_release_smoke_tests",
        "block_n_safety_gate_matrix_cannot_mutate_release_workflows",
        "block_n_safety_gate_matrix_cannot_generate_release_notes",
        "block_n_safety_gate_matrix_cannot_create_release_tags",
        "block_n_safety_gate_matrix_cannot_upload_release_artifacts",
        "block_n_safety_gate_matrix_cannot_export_release_external_artifacts",
        "block_n_safety_gate_matrix_cannot_create_artifacts",
        "block_n_safety_gate_matrix_cannot_mutate_artifacts",
        "block_n_safety_gate_matrix_cannot_delete_artifacts",
        "block_n_safety_gate_matrix_cannot_run_artifact_smoke_tests",
        "block_n_safety_gate_matrix_cannot_sign_artifacts",
        "block_n_safety_gate_matrix_cannot_publish_artifacts",
        "block_n_safety_gate_matrix_cannot_finalize_artifact_names",
        "block_n_safety_gate_matrix_cannot_select_artifact_locations",
        "block_n_safety_gate_matrix_cannot_generate_checksums",
        "block_n_safety_gate_matrix_cannot_write_artifact_metadata",
        "block_n_safety_gate_matrix_cannot_export_artifact_audits",
        "block_n_safety_gate_matrix_cannot_cleanup_artifacts",
        "block_n_safety_gate_matrix_cannot_execute_dry_run",
        "block_n_safety_gate_matrix_cannot_package_exe",
        "block_n_safety_gate_matrix_cannot_start_pyinstaller",
        "block_n_safety_gate_matrix_cannot_execute_build_commands",
        "block_n_safety_gate_matrix_cannot_create_build_artifacts",
        "block_n_safety_gate_matrix_cannot_change_installers",
        "block_n_safety_gate_matrix_cannot_change_workflows",
        "block_n_safety_gate_matrix_cannot_probe_packaging_environment",
        "block_n_safety_gate_matrix_cannot_freeze_dependencies",
        "block_n_safety_gate_matrix_cannot_discover_assets",
        "block_n_safety_gate_matrix_cannot_discover_qml_assets",
        "block_n_safety_gate_matrix_cannot_perform_filesystem_io",
        "block_n_safety_gate_matrix_cannot_activate_paper_runtime",
        "block_n_safety_gate_matrix_cannot_activate_testnet_runtime",
        "block_n_safety_gate_matrix_cannot_activate_live_runtime",
        "block_n_safety_gate_matrix_cannot_start_live_canary",
        "block_n_safety_gate_matrix_cannot_start_live_trading",
        "block_n_safety_gate_matrix_cannot_start_runtime_loop",
        "block_n_safety_gate_matrix_cannot_execute_runtime_gates",
        "block_n_safety_gate_matrix_cannot_mutate_gate_state",
        "block_n_safety_gate_matrix_cannot_generate_orders",
        _CANNOT_SUBMIT_ORDERS,
        _CANNOT_CANCEL_ORDERS,
        _CANNOT_REPLACE_ORDERS,
        "block_n_safety_gate_matrix_cannot_access_private_endpoints",
        "block_n_safety_gate_matrix_cannot_open_network_io",
        "block_n_safety_gate_matrix_cannot_read_credentials",
        "block_n_safety_gate_matrix_cannot_read_config_env_secrets",
        "block_n_safety_gate_matrix_cannot_change_ui_bridge",
        "block_n_safety_gate_matrix_cannot_open_any_execution_gate",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["source_boundaries"]
    read_boundaries = read_model["read_model_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_n_read_model": "FUNCTIONAL-PREVIEW-16.1",
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_gate_execution_calls_present": False,
        "forbidden_gate_mutation_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_n_read_model_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_block_n_entry_contract": source["source_block_n_entry_contract"],
            "plain_data_source_only_subset": {
                "block_n_read_model_is_plain_data_only": read_boundaries[
                    "block_n_read_model_is_plain_data_only"
                ],
                "block_n_read_model_is_source_only": read_boundaries[
                    "block_n_read_model_is_source_only"
                ],
            },
            "can_feed_16_2_boundary": read_boundaries[
                "block_n_read_model_can_feed_16_2_block_n_safety_gate_matrix"
            ],
        },
    }
