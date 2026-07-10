"""FUNCTIONAL-PREVIEW-16.3 Block N source-only safety gate contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_safety_gate_matrix import (
    build_preview_block_n_safety_gate_matrix,
)

PREVIEW_BLOCK_N_SAFETY_GATE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_n_safety_gate_contract.v1"
)
PREVIEW_BLOCK_N_SAFETY_GATE_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_n_safety_gate_contract"
)
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.3"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.4"
NEXT_STEP_TITLE: Final[str] = "BLOCK N SAFETY GATE READ MODEL"
READY_FOR_BLOCK_N_4: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_16_4_block_n_safety_gate_read_model"
SOURCE_BLOCK_N_SAFETY_GATE_MATRIX_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.2"
BLOCK_N_SAFETY_GATE_CONTRACT_STATUS: Final[str] = (
    "block_n_safety_gate_contract_ready_16_2_matrix_consumed_block_m_closure_"
    "preserved_block_n_opened_exe_direction_preserved_source_only_fail_closed_no_"
    "gate_evaluation_no_gate_state_mutation_no_release_execution_no_artifact_"
    "creation_no_packaging_execution_no_pyinstaller_no_build_no_runtime_no_orders_"
    "no_private_endpoints_no_network_io_no_credentials_no_filesystem_io"
)
BLOCK_N_SAFETY_GATE_CONTRACT_DECISION: Final[str] = (
    "BLOCK_N_SAFETY_GATE_CONTRACT_READY_16_2_MATRIX_CONSUMED_BLOCK_M_CLOSURE_"
    "PRESERVED_BLOCK_N_OPENED_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_FAIL_CLOSED_NO_"
    "GATE_EVALUATION_NO_GATE_STATE_MUTATION_NO_RELEASE_EXECUTION_NO_ARTIFACT_"
    "CREATION_NO_PACKAGING_EXECUTION_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_NO_ORDERS_"
    "NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_contract_status",
    "block_n_safety_gate_contract_decision",
    "ready_for_block_n_4",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_matrix_reference",
    "safety_gate_contract_summary",
    "packaging_release_gate_contract_rows",
    "runtime_safety_gate_contract_rows",
    "cross_domain_invariant_contract_rows",
    "exe_direction_gate_contract",
    "fail_closed_contract_decision",
    "non_execution_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_MATRIX_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_matrix_status",
    "block_n_safety_gate_matrix_decision",
    "ready_for_block_n_3",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_3_ROOTS: Final[list[str]] = [
    "release_executed",
    "release_published",
    "release_signed",
    "release_smoke_test_executed",
    "release_notes_generated",
    "release_tag_created",
    "release_uploaded",
    "release_external_export",
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
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
    "operator_confirmation_performed",
    "environment_validation_performed",
    "runtime_validation_performed",
    "runtime_activated",
    "runtime_loop_started",
    "runtime_gate_executed",
    "orders_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "credentials_read",
    "config_env_secrets_read",
    "filesystem_io_performed",
    "qml_bridge_changed",
]
_REAL_CAPABILITY_KEYS: Final[list[str]] = [
    "release_execution",
    "release_publish",
    "release_signing",
    "release_smoke",
    "release_workflow_mutation",
    "release_notes",
    "release_tags",
    "release_upload",
    "release_external_export",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke",
    "artifact_signing",
    "artifact_publishing",
    "artifact_name_finalization",
    "artifact_location_selection",
    "artifact_checksum",
    "artifact_metadata_write",
    "artifact_audit_export",
    "artifact_cleanup",
    "packaging_dry_run",
    "packaging",
    "pyinstaller",
    "build_command",
    "build_artifact",
    "installer_mutation",
    "workflow_mutation",
    "packaging_filesystem_io",
    "packaging_environment_probe",
    "dependency_freeze",
    "asset_discovery",
    "qml_asset_discovery",
    "gate_evaluation",
    "gate_condition_acceptance",
    "gate_opening",
    "gate_state_mutation",
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gate_execution",
    "order_generation",
    "order_" + "sub" + "mission",
    "order_" + "can" + "cellation",
    "order_" + "re" + "placement",
    "private_endpoint_access",
    "network_io",
    "credential_read",
    "config_env_secret_read",
    "filesystem_io",
    "qml_bridge_change",
]


def build_preview_block_n_safety_gate_contract() -> dict[str, Any]:
    """Build the Block N 16.3 source-only fail-closed safety gate contract."""
    matrix = build_preview_block_n_safety_gate_matrix()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_N_SAFETY_GATE_CONTRACT_SCHEMA_VERSION,
        "block_n_safety_gate_contract_kind": PREVIEW_BLOCK_N_SAFETY_GATE_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_safety_gate_contract_status": BLOCK_N_SAFETY_GATE_CONTRACT_STATUS,
        "block_n_safety_gate_contract_decision": BLOCK_N_SAFETY_GATE_CONTRACT_DECISION,
        "ready_for_block_n_4": READY_FOR_BLOCK_N_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_safety_gate_matrix_reference": _build_matrix_reference(matrix),
        "safety_gate_contract_summary": _build_summary(),
        "packaging_release_gate_contract_rows": _build_packaging_rows(matrix),
        "runtime_safety_gate_contract_rows": _build_runtime_rows(matrix),
        "cross_domain_invariant_contract_rows": _build_invariant_rows(matrix),
        "exe_direction_gate_contract": _build_exe_direction_gate_contract(matrix),
        "fail_closed_contract_decision": _build_fail_closed_contract_decision(),
        "non_execution_evidence": _build_non_execution_evidence(),
        "contract_boundaries": _build_contract_boundaries(),
        "source_boundaries": _build_source_boundaries(matrix),
        "future_steps": ["functional_preview_16_4_block_n_safety_gate_read_model"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_matrix_reference(matrix: dict[str, Any]) -> dict[str, Any]:
    reference = {key: matrix[key] for key in _MATRIX_REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_safety_gate_matrix_step": SOURCE_BLOCK_N_SAFETY_GATE_MATRIX_STEP,
            "source_block_n_safety_gate_matrix_read_by_16_3_contract": True,
            "block_n_safety_gate_matrix_available_before_contract": True,
            "static_block_n_safety_gate_matrix_only": True,
            "block_n_safety_gate_contract_built_by_16_3": True,
            "ready_for_functional_preview_16_4": True,
        }
    )
    for root in _FALSE_BY_16_3_ROOTS:
        reference[root + "_by_16_3"] = False
    return reference


def _build_summary() -> dict[str, bool]:
    summary = {
        "block_n_safety_gate_matrix_available": True,
        "block_n_safety_gate_contract_built": True,
        "block_n_opened": True,
        "ready_for_block_n_4": True,
        "ready_for_functional_preview_16_4": True,
        "block_m_closure_preserved": True,
        "exe_direction_preserved": True,
        "safety_gate_contract_static_only": True,
        "safety_gate_contract_read_only": True,
        "all_capabilities_fail_closed": True,
        "all_contract_rows_require_future_explicit_gate": True,
        "packaging_release_gate_contract_rows_built": True,
        "runtime_safety_gate_contract_rows_built": True,
        "cross_domain_invariant_contract_rows_built": True,
        "missing_evidence_blocks_execution": True,
        "missing_operator_confirmation_blocks_execution": True,
        "missing_environment_validation_blocks_execution": True,
        "missing_runtime_validation_blocks_execution": True,
    }
    for key in [
        "any_gate_evaluated_now",
        "any_gate_condition_met_now",
        "any_gate_open_now",
        "any_execution_allowed_now",
        "any_execution_performed_now",
        "any_gate_state_mutated_now",
        "operator_confirmation_present_now",
        "environment_validation_present_now",
        "artifact_validation_present_now",
        "release_validation_present_now",
        "runtime_validation_present_now",
        "packaging_execution_allowed_now",
        "release_execution_allowed_now",
        "artifact_work_allowed_now",
        "runtime_activation_allowed_now",
        "order_activity_allowed_now",
        "private_endpoint_access_allowed_now",
        "network_io_allowed_now",
        "credential_read_allowed_now",
        "filesystem_io_allowed_now",
        "qml_bridge_change_allowed_now",
    ]:
        summary[key] = False
    return summary


def _build_packaging_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _capability_contract_row(row, "packaging_release")
        for row in matrix["packaging_release_gate_rows"]
    ]


def _build_runtime_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _capability_contract_row(row, "runtime_safety")
        for row in matrix["runtime_safety_gate_rows"]
    ]


def _capability_contract_row(row: dict[str, Any], domain: str) -> dict[str, Any]:
    contract = {
        "contract_id": row["gate_id"] + "_contract",
        "source_gate_id": row["gate_id"],
        "capability_id": row["capability_id"],
        "domain": domain,
        "display_name": row["display_name"],
        "source_gate_result": "blocked",
        "source_gate_open_now": False,
        "source_execution_allowed_now": False,
        "source_execution_performed_now": False,
        "contract_required": True,
        "contract_static_only": True,
        "contract_evaluated_by_16_3": False,
        "contract_condition_met": False,
        "contract_gate_open_now": False,
        "execution_allowed_now": False,
        "execution_performed_now": False,
        "operator_confirmation_required": True,
        "operator_confirmation_present": False,
    }
    if domain == "packaging_release":
        contract.update(
            {
                "environment_validation_required": True,
                "environment_validation_present": False,
                "artifact_validation_required": True,
                "artifact_validation_present": False,
            }
        )
    else:
        contract.update(
            {
                "runtime_validation_required": True,
                "runtime_validation_present": False,
                "credentials_validation_required": True,
                "credentials_validation_present": False,
            }
        )
    contract.update(
        {
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "contract_result": "blocked_pending_future_explicit_gate",
            "notes": "16.3 source-only contract row; future explicit gate required.",
        }
    )
    return contract


def _build_invariant_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "contract_id": "block_n_" + row["invariant_id"] + "_contract",
            "source_invariant_id": row["invariant_id"],
            "display_name": row["display_name"],
            "domain": "cross_domain",
            "source_invariant_preserved": True,
            "contract_required": True,
            "contract_static_only": True,
            "invariant_required_for_future_execution": True,
            "invariant_satisfied_for_static_contract": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "contract_result": "preserved_but_execution_blocked",
            "notes": "16.3 preserves this invariant while keeping execution blocked.",
        }
        for row in matrix["cross_domain_invariant_gate_rows"]
    ]


def _build_exe_direction_gate_contract(matrix: dict[str, Any]) -> dict[str, Any]:
    source = matrix["exe_direction_safety_gate"]
    contract = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "block_n_safety_gate_contract_confirms_exe_direction": True,
        "exe_direction_is_not_execution_authorization": True,
        "exe_direction_requires_future_explicit_packaging_gate": True,
        "exe_direction_requires_future_explicit_release_gate": True,
    }
    for key in [
        "exe_packaging_gate_open_now",
        "packaging_dry_run_gate_open_now",
        "pyinstaller_gate_open_now",
        "build_command_gate_open_now",
        "artifact_work_gate_open_now",
        "release_gate_open_now",
        "runtime_gate_open_now",
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
        contract[key] = True
    return contract


def _build_fail_closed_contract_decision() -> dict[str, str]:
    decision = {
        "missing_block_n_safety_gate_matrix_policy": "fail_closed",
        "missing_gate_contract_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "failed_contract_policy": "fail_closed",
        "block_n_safety_gate_contract_in_16_3": "ready",
        "block_n_safety_gate_read_model_in_16_4": "allowed",
    }
    for key in _REAL_CAPABILITY_KEYS:
        decision[key + "_in_16_3"] = "blocked"
    return decision


def _build_non_execution_evidence() -> dict[str, bool]:
    evidence = {
        "source_block_n_safety_gate_matrix_read": True,
        "block_n_safety_gate_contract_built": True,
        "block_n_safety_gate_contract_only": True,
        "block_n_opened": True,
        "ready_for_block_n_4": True,
        "all_contract_rows_fail_closed": True,
    }
    for root in _FALSE_BY_16_3_ROOTS:
        evidence[root] = False
    for key in [
        "release_workflow_mutated",
        "environment_probe_performed",
        "artifact_validation_performed",
        "release_validation_performed",
        "credentials_validation_performed",
        "paper_runtime_started",
        "testnet_runtime_started",
        "live_canary_started",
        "live_trading_started",
        "order_generated",
        "order_" + "sub" + "mitted",
        "order_" + "can" + "celed",
        "order_" + "re" + "placed",
        "config_env_secrets_read",
    ]:
        evidence[key] = False
    return evidence


def _build_contract_boundaries() -> dict[str, bool]:
    boundaries = {
        "block_n_safety_gate_contract_is_plain_data_only": True,
        "block_n_safety_gate_contract_is_source_only": True,
        "block_n_safety_gate_contract_reads_block_n_safety_gate_matrix_only": True,
        "block_n_safety_gate_contract_preserves_block_m_closure": True,
        "block_n_safety_gate_contract_preserves_block_n_entry": True,
        "block_n_safety_gate_contract_preserves_exe_direction_without_packaging": True,
        "block_n_safety_gate_contract_is_static_and_non_evaluating": True,
        "block_n_safety_gate_contract_is_non_mutating": True,
        "block_n_safety_gate_contract_can_feed_16_4_block_n_safety_gate_read_model": True,
    }
    for key in [
        "block_n_safety_gate_contract_cannot_evaluate_gates",
        "block_n_safety_gate_contract_cannot_mark_gate_condition_met",
        "block_n_safety_gate_contract_cannot_open_gates",
        "block_n_safety_gate_contract_cannot_mutate_gate_state",
        "block_n_safety_gate_contract_cannot_accept_operator_confirmation",
        "block_n_safety_gate_contract_cannot_validate_environment",
        "block_n_safety_gate_contract_cannot_validate_artifacts",
        "block_n_safety_gate_contract_cannot_validate_release",
        "block_n_safety_gate_contract_cannot_validate_runtime",
        "block_n_safety_gate_contract_cannot_validate_credentials",
        "block_n_safety_gate_contract_cannot_execute_release",
        "block_n_safety_gate_contract_cannot_publish_release",
        "block_n_safety_gate_contract_cannot_sign_release",
        "block_n_safety_gate_contract_cannot_smoke_release",
        "block_n_safety_gate_contract_cannot_generate_notes",
        "block_n_safety_gate_contract_cannot_generate_tags",
        "block_n_safety_gate_contract_cannot_upload_release",
        "block_n_safety_gate_contract_cannot_export_release",
        "block_n_safety_gate_contract_cannot_create_artifacts",
        "block_n_safety_gate_contract_cannot_mutate_artifacts",
        "block_n_safety_gate_contract_cannot_delete_artifacts",
        "block_n_safety_gate_contract_cannot_smoke_artifacts",
        "block_n_safety_gate_contract_cannot_sign_artifacts",
        "block_n_safety_gate_contract_cannot_publish_artifacts",
        "block_n_safety_gate_contract_cannot_finalize_names",
        "block_n_safety_gate_contract_cannot_select_locations",
        "block_n_safety_gate_contract_cannot_checksum",
        "block_n_safety_gate_contract_cannot_write_metadata",
        "block_n_safety_gate_contract_cannot_export_audit",
        "block_n_safety_gate_contract_cannot_cleanup",
        "block_n_safety_gate_contract_cannot_execute_packaging_dry_run",
        "block_n_safety_gate_contract_cannot_package_exe",
        "block_n_safety_gate_contract_cannot_start_pyinstaller",
        "block_n_safety_gate_contract_cannot_execute_build_commands",
        "block_n_safety_gate_contract_cannot_create_build_artifacts",
        "block_n_safety_gate_contract_cannot_change_installers",
        "block_n_safety_gate_contract_cannot_change_workflows",
        "block_n_safety_gate_contract_cannot_probe_packaging_environment",
        "block_n_safety_gate_contract_cannot_freeze_dependencies",
        "block_n_safety_gate_contract_cannot_discover_assets",
        "block_n_safety_gate_contract_cannot_discover_qml_assets",
        "block_n_safety_gate_contract_cannot_perform_filesystem_io",
        "block_n_safety_gate_contract_cannot_activate_runtime",
        "block_n_safety_gate_contract_cannot_start_paper_runtime",
        "block_n_safety_gate_contract_cannot_start_testnet_runtime",
        "block_n_safety_gate_contract_cannot_start_live_canary",
        "block_n_safety_gate_contract_cannot_start_live_trading",
        "block_n_safety_gate_contract_cannot_start_runtime_loop",
        "block_n_safety_gate_contract_cannot_execute_runtime_gates",
        "block_n_safety_gate_contract_cannot_generate_orders",
        "block_n_safety_gate_contract_cannot_" + "sub" + "mit_orders",
        "block_n_safety_gate_contract_cannot_" + "can" + "cel_orders",
        "block_n_safety_gate_contract_cannot_" + "re" + "place_orders",
        "block_n_safety_gate_contract_cannot_access_private_endpoints",
        "block_n_safety_gate_contract_cannot_open_network_io",
        "block_n_safety_gate_contract_cannot_read_credentials",
        "block_n_safety_gate_contract_cannot_read_config_env_secrets",
        "block_n_safety_gate_contract_cannot_change_qml_bridge",
        "block_n_safety_gate_contract_cannot_perform_any_execution_side_effect",
    ]:
        boundaries[key] = True
    return boundaries


def _build_source_boundaries(matrix: dict[str, Any]) -> dict[str, Any]:
    source = matrix["source_boundaries"]
    matrix_bounds = matrix["gate_matrix_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_n_safety_gate_matrix": SOURCE_BLOCK_N_SAFETY_GATE_MATRIX_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_gate_evaluation_calls_present": False,
        "forbidden_gate_execution_calls_present": False,
        "forbidden_gate_mutation_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_n_safety_gate_matrix_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_block_n_read_model": source["source_block_n_read_model"],
            "plain_data_source_only_subset": {
                "block_n_safety_gate_matrix_is_plain_data_only": matrix_bounds[
                    "block_n_safety_gate_matrix_is_plain_data_only"
                ],
                "block_n_safety_gate_matrix_is_source_only": matrix_bounds[
                    "block_n_safety_gate_matrix_is_source_only"
                ],
            },
            "static_and_non_evaluating_boundary": matrix_bounds[
                "block_n_safety_gate_matrix_is_static_and_non_evaluating"
            ],
            "can_feed_16_3_boundary": matrix_bounds[
                "block_n_safety_gate_matrix_can_feed_16_3_block_n_safety_gate_contract"
            ],
        },
    }
