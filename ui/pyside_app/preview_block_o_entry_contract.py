"""FUNCTIONAL-PREVIEW-17.0 Block O source-only entry contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_closure_audit import (
    build_preview_block_n_closure_audit,
)

SCHEMA_VERSION: Final[str] = "preview_block_o_entry_contract.v1"
KIND: Final[str] = "functional_preview_block_o_entry_contract"
BLOCK_ID: Final[str] = "O"
STEP_ID: Final[str] = "17.0"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.1"
NEXT_STEP_TITLE: Final[str] = "BLOCK O READ MODEL"
BLOCK_O_OPENED: Final[bool] = True
READY_FOR_BLOCK_O_1: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_17_1_block_o_read_model"
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_17_1_block_o_entry_source_not_accepted"
SOURCE_BLOCK_N_CLOSURE_AUDIT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.8"
ENTRY_CONTRACT_STATUS: Final[str] = (
    "block_o_entry_contract_ready_block_n_closure_audit_consumed_block_n_closed_block_o_opened_"
    "steps_16_0_through_16_8_preserved_block_m_closure_preserved_desktop_exe_direction_preserved_"
    "source_only_plain_data_static_contract_only_all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_all_requirements_missing_all_invariants_preserved_"
    "all_execution_unauthorized_all_gates_closed_no_readiness_recalculation_no_gate_evaluation_"
    "no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_"
    "no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_"
    "only_source_only_17_1_handoff_allowed"
)
ENTRY_CONTRACT_DECISION: Final[str] = (
    "BLOCK_O_ENTRY_CONTRACT_READY_BLOCK_N_CLOSURE_AUDIT_CONSUMED_BLOCK_N_CLOSED_BLOCK_O_OPENED_"
    "STEPS_16_0_THROUGH_16_8_PRESERVED_BLOCK_M_CLOSURE_PRESERVED_DESKTOP_EXE_DIRECTION_PRESERVED_"
    "SOURCE_ONLY_PLAIN_DATA_STATIC_CONTRACT_ONLY_ALL_EXECUTION_CAPABILITIES_NOT_READY_"
    "ALL_EXECUTION_CAPABILITIES_BLOCKED_ALL_REQUIREMENTS_MISSING_ALL_INVARIANTS_PRESERVED_"
    "ALL_EXECUTION_UNAUTHORIZED_ALL_GATES_CLOSED_NO_READINESS_RECALCULATION_NO_GATE_EVALUATION_"
    "NO_VALIDATION_NO_CONFIRMATION_ACCEPTANCE_NO_AUTHORIZATION_NO_PACKAGING_NO_BUILD_NO_RELEASE_"
    "NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_"
    "ONLY_SOURCE_ONLY_17_1_HANDOFF_ALLOWED"
)
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_o_entry_contract_kind",
    "block",
    "step",
    "block_o_entry_contract_status",
    "block_o_entry_contract_decision",
    "block_o_opened",
    "ready_for_block_o_1",
    "next_step",
    "next_step_title",
    "block_n_closure_audit_reference",
    "entry_contract_summary",
    "inherited_block_n_closure_summary",
    "inherited_capability_state",
    "inherited_invariant_state",
    "inherited_requirement_state",
    "exe_direction_entry_contract",
    "fail_closed_entry_decision",
    "non_execution_entry_evidence",
    "entry_contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_SOURCE_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_closure_audit_kind",
    "block",
    "step",
    "block_n_closure_audit_status",
    "block_n_closure_audit_decision",
    "block_n_closed",
    "ready_for_block_o_0",
    "next_step",
    "next_step_title",
]
_FALSE_BY_17_0_ROOTS: Final[list[str]] = [
    "readiness_recalculated_from_environment",
    "closure_recalculated",
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
    "execution_authorized",
    "operator_confirmation_accepted",
    "environment_validation_performed",
    "artifact_validation_performed",
    "release_validation_performed",
    "runtime_validation_performed",
    "credentials_validation_performed",
    "dependency_validation_performed",
    "future_explicit_gate_opened",
    "packaging_dry_run_executed",
    "packaging_executed",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "artifact_created",
    "artifact_mutated",
    "artifact_deleted",
    "artifact_smoke_tested",
    "artifact_signed",
    "artifact_published",
    "release_executed",
    "release_published",
    "release_signed",
    "release_smoke_tested",
    "release_notes_generated",
    "release_tag_created",
    "release_uploaded",
    "release_external_export",
    "runtime_activated",
    "paper_runtime_started",
    "testnet_runtime_started",
    "live_canary_started",
    "live_trading_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "order_activity_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "credentials_read",
    "config_env_secrets_read",
    "filesystem_io_performed",
    "qml_bridge_changed",
    "installer_changed",
    "workflow_changed",
]


_EXPECTED_SOURCE_SCHEMA_VERSION: Final[str] = "preview_block_n_closure_audit.v1"
_EXPECTED_SOURCE_KIND: Final[str] = "functional_preview_block_n_closure_audit"
_EXPECTED_SOURCE_BLOCK: Final[str] = "N"
_EXPECTED_SOURCE_STEP: Final[str] = "16.8"
_EXPECTED_SOURCE_STATUS: Final[str] = (
    "block_n_closed_ready_for_functional_preview_17_0_block_o_entry_contract"
)
_EXPECTED_SOURCE_CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_n_closure_audit_complete_16_7_readiness_read_model_consumed_"
    "steps_16_0_through_16_7_complete_block_m_closure_preserved_"
    "block_n_closed_exe_direction_preserved_source_only_plain_data_"
    "static_audit_only_all_capability_rows_read_all_requirements_missing_"
    "all_invariants_preserved_all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_all_execution_unauthorized_"
    "all_gates_closed_no_readiness_recalculation_no_gate_evaluation_"
    "no_validation_no_confirmation_acceptance_no_authorization_"
    "no_packaging_no_build_no_release_no_runtime_no_orders_"
    "no_private_endpoints_no_network_io_no_credentials_"
    "no_filesystem_io_only_source_only_block_o_handoff_allowed"
)
_EXPECTED_SOURCE_FUTURE_STEPS: Final[list[str]] = ["functional_preview_17_0_block_o_entry_contract"]


_EXPECTED_SOURCE_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_closure_audit_kind",
    "block",
    "step",
    "block_n_closure_audit_status",
    "block_n_closure_audit_decision",
    "block_n_closed",
    "ready_for_block_o_0",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_read_model_reference",
    "closure_audit_summary",
    "block_n_step_closure_rows",
    "packaging_release_closure_summary",
    "runtime_safety_closure_summary",
    "cross_domain_invariant_closure_summary",
    "validation_requirement_closure_summary",
    "exe_direction_closure_audit",
    "fail_closed_closure_decision",
    "non_execution_closure_evidence",
    "closure_audit_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

_EXPECTED_BLOCK_N_STEP_DATA: Final[list[tuple[str, str, str]]] = [
    ("FUNCTIONAL-PREVIEW-16.0", "BLOCK N ENTRY CONTRACT", "entry_contract"),
    ("FUNCTIONAL-PREVIEW-16.1", "BLOCK N READ MODEL", "read_model"),
    ("FUNCTIONAL-PREVIEW-16.2", "BLOCK N SAFETY GATE MATRIX", "safety_gate_matrix"),
    ("FUNCTIONAL-PREVIEW-16.3", "BLOCK N SAFETY GATE CONTRACT", "safety_gate_contract"),
    ("FUNCTIONAL-PREVIEW-16.4", "BLOCK N SAFETY GATE READ MODEL", "safety_gate_read_model"),
    (
        "FUNCTIONAL-PREVIEW-16.5",
        "BLOCK N SAFETY GATE READINESS MATRIX",
        "safety_gate_readiness_matrix",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.6",
        "BLOCK N SAFETY GATE READINESS CONTRACT",
        "safety_gate_readiness_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.7",
        "BLOCK N SAFETY GATE READINESS READ MODEL",
        "safety_gate_readiness_read_model",
    ),
]

_EXPECTED_BLOCK_N_ROW_FIELDS: Final[list[str]] = [
    "closure_row_id",
    "step",
    "title",
    "artifact_kind",
    "step_complete",
    "source_only",
    "plain_data",
    "execution_authorized",
    "real_capabilities_opened",
    "closure_status",
    "closure_result",
    "notes",
]

_EXPECTED_SOURCE_CLOSURE_SUMMARY_TRUE_FIELDS: Final[list[str]] = [
    "block_n_safety_gate_readiness_read_model_available",
    "block_n_closure_audit_built",
    "block_n_opened",
    "block_n_closed",
    "ready_for_block_o_0",
    "ready_for_functional_preview_17_0",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "closure_audit_source_only",
    "closure_audit_plain_data_only",
    "closure_audit_static_only",
    "closure_audit_read_only",
    "closure_audit_non_evaluating",
    "closure_audit_non_mutating",
    "closure_audit_non_authorizing",
    "all_block_n_steps_complete",
    "all_capability_rows_read",
    "all_requirement_rows_read",
    "all_invariant_rows_read",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_preserved",
    "all_domains_not_ready",
    "all_domains_execution_unauthorized",
    "only_source_only_block_o_handoff_allowed",
]

_EXPECTED_SOURCE_CLOSURE_SUMMARY_FALSE_FIELDS: Final[list[str]] = [
    "any_readiness_recalculated_from_environment_now",
    "any_gate_evaluated_now",
    "any_gate_condition_met_now",
    "any_gate_open_now",
    "any_gate_state_mutated_now",
    "any_execution_authorized_now",
    "any_execution_allowed_now",
    "any_execution_performed_now",
    "any_validation_completed_now",
    "any_requirement_present_now",
    "any_requirement_satisfied_now",
    "any_capability_ready_now",
    "packaging_release_domain_ready_now",
    "runtime_safety_domain_ready_now",
    "exe_build_ready_now",
    "exe_packaging_ready_now",
    "exe_release_ready_now",
    "runtime_enabled_by_closure",
    "packaging_enabled_by_closure",
    "release_enabled_by_closure",
    "orders_enabled_by_closure",
]

_EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS: Final[list[str]] = [
    "source_block_n_readiness_read_model_read",
    "block_n_closure_audit_built",
    "block_n_closure_audit_only",
    "block_n_opened",
    "block_n_closed",
    "ready_for_block_o_0",
    "all_block_n_steps_complete",
    "all_capability_rows_read",
    "all_capability_rows_not_ready",
    "all_invariant_rows_preserved",
    "all_requirement_rows_missing",
    "all_execution_authorization_false",
    "all_capabilities_fail_closed",
]

_EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS: Final[list[str]] = [
    "readiness_recalculated_from_environment",
    "gate_evaluation_performed",
    "gate_condition_accepted",
    "gate_opened",
    "gate_mutated",
    "confirmation_accepted",
    "validation_performed",
    "authorization_performed",
    "execution_performed",
    "packaging_performed",
    "build_performed",
    "release_performed",
    "runtime_performed",
    "orders_performed",
    "network_io_performed",
    "filesystem_io_performed",
    "private_endpoint_accessed",
    "credentials_read",
    "config_env_secrets_read",
    "real_capabilities_opened_by_closure",
]

_EXPECTED_SOURCE_CLOSURE_BOUNDARY_FIELDS: Final[list[str]] = [
    "block_n_closure_audit_is_plain_data_only",
    "block_n_closure_audit_is_source_only",
    "block_n_closure_audit_reads_16_7_only",
    "block_n_closure_audit_preserves_block_m_closure",
    "block_n_closure_audit_preserves_block_n_entry",
    "block_n_closure_audit_preserves_exe_direction_without_packaging",
    "block_n_closure_audit_is_static_and_non_evaluating",
    "block_n_closure_audit_is_non_mutating",
    "block_n_closure_audit_is_non_authorizing",
    "block_n_closure_audit_can_close_block_n",
    "block_n_closure_audit_can_feed_17_0_entry_contract",
    "cannot_recalculate_readiness_from_environment",
    "cannot_evaluate",
    "cannot_accept_condition",
    "cannot_open_gate",
    "cannot_mutate_gate",
    "cannot_accept_confirmations",
    "cannot_perform_validations",
    "cannot_authorize",
    "cannot_package",
    "cannot_build",
    "cannot_release",
    "cannot_perform_artifact_work",
    "cannot_run_runtime",
    "cannot_generate_orders",
    "cannot_" + "sub" + "mit_orders",
    "cannot_" + "can" + "cel_orders",
    "cannot_" + "re" + "place_orders",
    "cannot_use_network",
    "cannot_use_filesystem",
    "cannot_access_private_endpoints",
    "cannot_read_credentials",
    "cannot_read_config_env_secrets",
    "cannot_change_qml_or_bridge",
    "cannot_create_execution_side_effects",
]


_EXPECTED_SOURCE_FAIL_CLOSED_DECISION_FIELDS: Final[list[str]] = [
    "missing_block_n_readiness_read_model_policy",
    "missing_block_n_step_policy",
    "missing_capability_read_row_policy",
    "missing_requirement_read_row_policy",
    "missing_invariant_read_row_policy",
    "missing_operator_confirmation_policy",
    "missing_environment_validation_policy",
    "missing_artifact_validation_policy",
    "missing_release_validation_policy",
    "missing_runtime_validation_policy",
    "missing_credentials_validation_policy",
    "missing_future_explicit_gate_policy",
    "failed_closure_audit_policy",
    "block_n_closure_audit_in_16_8",
    "block_o_entry_contract_in_17_0",
    "only_source_only_17_0_handoff_allowed",
    "real_capability_status",
]

_EXPECTED_SOURCE_READINESS_REFERENCE_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_readiness_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_read_model_status",
    "block_n_safety_gate_readiness_read_model_decision",
    "ready_for_block_n_8",
    "next_step",
    "next_step_title",
    "source_block_n_safety_gate_readiness_read_model_step",
    "source_block_n_safety_gate_readiness_read_model_read_by_16_8",
    "block_n_safety_gate_readiness_read_model_available_before_closure_audit",
    "static_block_n_safety_gate_readiness_read_model_only",
    "block_n_closure_audit_built_by_16_8",
    "ready_for_functional_preview_17_0",
    "readiness_recalculated_from_environment_by_16_8",
    "gate_evaluated_by_16_8",
    "gate_condition_met_by_16_8",
    "gate_opened_by_16_8",
    "gate_state_mutated_by_16_8",
    "execution_authorized_by_16_8",
    "operator_confirmation_accepted_by_16_8",
    "environment_validation_performed_by_16_8",
    "artifact_validation_performed_by_16_8",
    "release_validation_performed_by_16_8",
    "runtime_validation_performed_by_16_8",
    "credentials_validation_performed_by_16_8",
    "dependency_validation_performed_by_16_8",
    "future_explicit_gate_opened_by_16_8",
    "packaging_dry_run_executed_by_16_8",
    "packaging_executed_by_16_8",
    "pyinstaller_started_by_16_8",
    "build_command_executed_by_16_8",
    "build_artifact_created_by_16_8",
    "artifact_created_by_16_8",
    "artifact_mutated_by_16_8",
    "artifact_deleted_by_16_8",
    "artifact_smoke_tested_by_16_8",
    "artifact_signed_by_16_8",
    "artifact_published_by_16_8",
    "release_executed_by_16_8",
    "release_published_by_16_8",
    "release_signed_by_16_8",
    "release_smoke_tested_by_16_8",
    "release_notes_generated_by_16_8",
    "release_tag_created_by_16_8",
    "release_uploaded_by_16_8",
    "release_external_export_by_16_8",
    "runtime_activated_by_16_8",
    "paper_runtime_started_by_16_8",
    "testnet_runtime_started_by_16_8",
    "live_canary_started_by_16_8",
    "live_trading_started_by_16_8",
    "runtime_loop_started_by_16_8",
    "runtime_gate_executed_by_16_8",
    "order_activity_enabled_by_16_8",
    "private_endpoint_accessed_by_16_8",
    "network_io_opened_by_16_8",
    "credentials_read_by_16_8",
    "config_env_secrets_read_by_16_8",
    "filesystem_io_performed_by_16_8",
    "qml_bridge_changed_by_16_8",
    "installer_changed_by_16_8",
    "workflow_changed_by_16_8",
]

_EXPECTED_SOURCE_READINESS_READ_MODEL_STATUS: Final[str] = (
    "readiness_read_model_ready_16_6_readiness_contract_consumed_"
    "block_m_closure_preserved_block_n_opened_exe_direction_preserved_"
    "source_only_plain_data_static_read_projection_only_"
    "all_capability_contracts_read_all_requirement_contracts_read_"
    "all_invariant_contracts_read_all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_all_requirements_missing_"
    "all_execution_unauthorized_all_gates_closed_invariants_preserved_"
    "no_readiness_recalculation_no_gate_evaluation_no_validation_"
    "no_confirmation_acceptance_no_authorization_no_packaging_"
    "no_build_no_release_no_runtime_no_orders_no_private_endpoints_"
    "no_network_io_no_credentials_no_filesystem_io"
)

_EXPECTED_SOURCE_EXE_DIRECTION_FIELDS: Final[list[str]] = [
    "final_product_direction",
    "exe_direction_preserved",
    "block_n_safety_gate_read_model_confirms_exe_direction",
    "exe_direction_is_not_execution_authorization",
    "exe_direction_requires_future_explicit_packaging_gate",
    "exe_direction_requires_future_explicit_release_gate",
    "packaging_requirements_complete",
    "release_requirements_complete",
    "ready_to_build_exe_now",
    "ready_to_package_exe_now",
    "ready_to_release_exe_now",
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
    "block_n_safety_gate_readiness_matrix_confirms_exe_direction",
    "build_readiness_classification",
    "packaging_readiness_classification",
    "release_readiness_classification",
    "build_authorized_now",
    "packaging_authorized_now",
    "release_authorized_now",
    "future_packaging_gate_required",
    "future_release_gate_required",
    "future_explicit_step_required",
    "failure_policy",
    "matrix_result",
    "block_n_safety_gate_readiness_contract_confirms_exe_direction",
    "readiness_matrix_source_preserved",
    "contract_result",
    "block_n_safety_gate_readiness_read_model_confirms_exe_direction",
    "readiness_contract_source_preserved",
    "read_model_is_not_execution_authorization",
    "read_result",
    "block_n_closure_audit_confirms_exe_direction",
    "readiness_read_model_source_preserved",
    "closure_is_not_execution_authorization",
    "closure_result",
]

_EXPECTED_SOURCE_EXE_TRUE_FIELDS: Final[list[str]] = [
    "exe_direction_preserved",
    "block_n_safety_gate_read_model_confirms_exe_direction",
    "exe_direction_is_not_execution_authorization",
    "exe_direction_requires_future_explicit_packaging_gate",
    "exe_direction_requires_future_explicit_release_gate",
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
    "block_n_safety_gate_readiness_matrix_confirms_exe_direction",
    "future_packaging_gate_required",
    "future_release_gate_required",
    "future_explicit_step_required",
    "block_n_safety_gate_readiness_contract_confirms_exe_direction",
    "readiness_matrix_source_preserved",
    "block_n_safety_gate_readiness_read_model_confirms_exe_direction",
    "readiness_contract_source_preserved",
    "read_model_is_not_execution_authorization",
    "block_n_closure_audit_confirms_exe_direction",
    "readiness_read_model_source_preserved",
    "closure_is_not_execution_authorization",
]
_EXPECTED_SOURCE_EXE_FALSE_FIELDS: Final[list[str]] = [
    "packaging_requirements_complete",
    "release_requirements_complete",
    "ready_to_build_exe_now",
    "ready_to_package_exe_now",
    "ready_to_release_exe_now",
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
    "build_authorized_now",
    "packaging_authorized_now",
    "release_authorized_now",
]
_EXPECTED_SOURCE_EXE_STRING_VALUES: Final[dict[str, str]] = {
    "final_product_direction": "desktop_exe",
    "build_readiness_classification": "not_ready",
    "packaging_readiness_classification": "not_ready",
    "release_readiness_classification": "not_ready",
    "failure_policy": "fail_closed",
    "matrix_result": "exe_direction_preserved_execution_not_ready",
    "contract_result": "exe_direction_contracted_execution_not_ready",
    "read_result": "exe_direction_read_execution_not_ready",
    "closure_result": "exe_direction_preserved_block_n_closed_execution_not_ready",
}

_EXPECTED_SOURCE_BOUNDARY_FIELDS: Final[list[str]] = [
    "allowed_imports_only",
    "source_block_n_safety_gate_readiness_contract",
    "source_block_n_safety_gate_readiness_matrix",
    "source_block_n_safety_gate_read_model",
    "source_block_n_safety_gate_readiness_contract_boundaries",
    "forbidden_packaging_calls_present",
    "forbidden_pyinstaller_calls_present",
    "forbidden_build_calls_present",
    "forbidden_release_calls_present",
    "forbidden_runtime_calls_present",
    "forbidden_gate_evaluation_calls_present",
    "forbidden_gate_execution_calls_present",
    "forbidden_gate_mutation_calls_present",
    "forbidden_validation_calls_present",
    "forbidden_confirmation_calls_present",
    "forbidden_authorization_calls_present",
    "forbidden_readiness_recalculation_calls_present",
    "forbidden_io_calls_present",
    "forbidden_network_calls_present",
    "forbidden_private_endpoint_calls_present",
    "forbidden_ui_bridge_calls_present",
    "source_block_n_safety_gate_readiness_read_model",
    "can_feed_16_8",
    "can_close_block_n",
    "can_feed_17_0",
    "forbidden_git_calls_present",
]
_EXPECTED_SOURCE_NESTED_BOUNDARY_FIELDS: Final[list[str]] = [
    "allowed_imports_only",
    "source_block_n_safety_gate_readiness_matrix",
    "source_block_n_safety_gate_read_model",
    "plain_data_source_only",
    "static_non_evaluating",
    "non_mutating",
    "non_authorizing",
    "can_feed_16_7",
    "can_feed_16_8",
]

_EXPECTED_SOURCE_INVARIANT_SUMMARY_FIELDS: Final[list[str]] = [
    "source_invariant_read_rows",
    "invariant_count",
    "preserved_invariant_count",
    "failed_invariant_count",
    "all_invariants_read",
    "all_invariants_preserved",
    "all_invariants_require_future_explicit_gate",
    "execution_gate_open_now",
    "execution_allowed_now",
    "execution_performed_now",
    "failure_policy",
    "closure_result",
]
_INVARIANT_STATE_17_0_FIELDS: Final[list[str]] = [
    "inherited_by_block_o_entry",
    "revalidated_by_block_o_entry",
    "entry_result",
]
_EXPECTED_SOURCE_REQUIREMENT_SUMMARY_FIELDS: Final[list[str]] = [
    "source_requirement_read_rows",
    "requirement_count",
    "required_requirement_count",
    "present_requirement_count",
    "completed_requirement_count",
    "satisfied_requirement_count",
    "missing_requirement_count",
    "all_requirements_read",
    "all_requirements_required",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_requirements_require_future_explicit_step",
    "failure_policy",
    "closure_result",
]
_REQUIREMENT_STATE_17_0_FIELDS: Final[list[str]] = [
    "inherited_by_block_o_entry",
    "validated_by_block_o_entry",
    "entry_result",
]
_EXPECTED_SOURCE_INVARIANT_CLOSURE_RESULT: Final[str] = (
    "closed_invariants_preserved_execution_blocked"
)
_EXPECTED_SOURCE_REQUIREMENT_CLOSURE_RESULT: Final[str] = (
    "closed_requirements_missing_execution_blocked"
)

_EXPECTED_SOURCE_INVARIANT_ROW_FIELDS: Final[list[str]] = [
    "read_row_id",
    "source_contract_row_id",
    "source_readiness_row_id",
    "source_read_row_id",
    "source_contract_id",
    "invariant_id",
    "domain",
    "display_name",
    "source_contract_result",
    "source_contract_readiness_classification",
    "source_invariant_preserved",
    "read_invariant_preserved",
    "invariant_required_for_future_execution",
    "execution_gate_open_now",
    "execution_allowed_now",
    "execution_performed_now",
    "requires_future_explicit_gate",
    "readiness_classification",
    "failure_policy",
    "read_result",
    "notes",
]
_EXPECTED_SOURCE_INVARIANT_IDS: Final[list[str]] = [
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
_EXPECTED_SOURCE_REQUIREMENT_ROW_FIELDS: Final[list[str]] = [
    "read_row_id",
    "source_contract_row_id",
    "requirement_id",
    "display_name",
    "source_required",
    "source_present",
    "source_completed",
    "source_satisfied",
    "required",
    "present",
    "completed",
    "satisfied",
    "applicable_domains",
    "missing_blocks_execution",
    "requires_future_explicit_step",
    "failure_policy",
    "read_result",
    "notes",
]
_EXPECTED_SOURCE_REQUIREMENT_IDS: Final[list[str]] = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]
_EXPECTED_REQUIREMENT_APPLICABLE_DOMAINS: Final[dict[str, list[str]]] = {
    "operator_confirmation": ["packaging_release", "runtime_safety"],
    "environment_validation": ["packaging_release"],
    "artifact_validation": ["packaging_release"],
    "release_validation": ["packaging_release"],
    "runtime_validation": ["runtime_safety"],
    "credentials_validation": ["runtime_safety"],
    "future_explicit_gate": ["packaging_release", "runtime_safety", "cross_domain"],
}
_EXPECTED_REQUIREMENT_DISPLAY_NAMES: Final[dict[str, str]] = {
    "operator_confirmation": "Operator Confirmation",
    "environment_validation": "Environment Validation",
    "artifact_validation": "Artifact Validation",
    "release_validation": "Release Validation",
    "runtime_validation": "Runtime Validation",
    "credentials_validation": "Credentials Validation",
    "future_explicit_gate": "Future Explicit Gate",
}

_EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS: Final[list[str]] = [
    "domain",
    "capability_count",
    "read_capability_count",
    "ready_capability_count",
    "blocked_capability_count",
    "required_requirement_ids",
    "satisfied_requirement_ids",
    "missing_requirement_ids",
    "requirements_complete",
    "domain_ready",
    "execution_authorized",
    "all_capabilities_read",
    "all_capabilities_not_ready",
    "all_capabilities_blocked",
    "failure_policy",
    "domain_closed_in_block_n",
    "domain_enabled_by_closure",
    "closure_result",
]
_CAPABILITY_DOMAIN_17_0_FIELDS: Final[list[str]] = [
    "inherited_by_block_o_entry",
    "enabled_by_block_o_entry",
]
_EXPECTED_CAPABILITY_DOMAIN_FIELDS: Final[list[str]] = (
    _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS + _CAPABILITY_DOMAIN_17_0_FIELDS
)
_EXPECTED_PACKAGING_RELEASE_REQUIREMENT_IDS: Final[list[str]] = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "future_explicit_gate",
]
_EXPECTED_RUNTIME_SAFETY_REQUIREMENT_IDS: Final[list[str]] = [
    "operator_confirmation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]

_EXPECTED_BLOCK_N_STEPS: Final[list[str]] = [
    "FUNCTIONAL-PREVIEW-16.0",
    "FUNCTIONAL-PREVIEW-16.1",
    "FUNCTIONAL-PREVIEW-16.2",
    "FUNCTIONAL-PREVIEW-16.3",
    "FUNCTIONAL-PREVIEW-16.4",
    "FUNCTIONAL-PREVIEW-16.5",
    "FUNCTIONAL-PREVIEW-16.6",
    "FUNCTIONAL-PREVIEW-16.7",
]


_EXPECTED_REAL_CAPABILITY_KEYS: Final[list[str]] = [
    "release_execution",
    "release_publish",
    "release_sign",
    "release_smoke",
    "release_workflow",
    "release_notes",
    "release_tag",
    "release_upload",
    "release_export",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke",
    "artifact_sign",
    "artifact_publish",
    "artifact_name",
    "artifact_location",
    "artifact_checksum",
    "artifact_metadata",
    "artifact_audit",
    "artifact_cleanup",
    "packaging_dry_run",
    "packaging",
    "pyinstaller",
    "build",
    "build_artifact",
    "installer",
    "workflow",
    "environment",
    "dependency",
    "asset",
    "qml_asset",
    "filesystem",
    "gate_evaluation",
    "gate_condition",
    "gate_opening",
    "gate_mutation",
    "confirmation_acceptance",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "dependency_validation",
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gates",
    "order_generation",
    "create_" + "order",
    "sub" + "mit_order",
    "can" + "cel_order",
    "re" + "place_order",
    "fetch" + "_balance",
    "private_endpoint",
    "network",
    "credentials",
    "config_env_secrets",
    "qml_bridge",
    "cc" + "xt",
]

_REQUIRED_INHERITED_FORBIDDEN_FIELDS: Final[list[str]] = [
    "forbidden_packaging_calls_present",
    "forbidden_pyinstaller_calls_present",
    "forbidden_build_calls_present",
    "forbidden_release_calls_present",
    "forbidden_runtime_calls_present",
    "forbidden_gate_evaluation_calls_present",
    "forbidden_gate_execution_calls_present",
    "forbidden_gate_mutation_calls_present",
    "forbidden_validation_calls_present",
    "forbidden_confirmation_calls_present",
    "forbidden_authorization_calls_present",
    "forbidden_readiness_recalculation_calls_present",
    "forbidden_io_calls_present",
    "forbidden_network_calls_present",
    "forbidden_private_endpoint_calls_present",
    "forbidden_ui_bridge_calls_present",
    "forbidden_git_calls_present",
]


def build_preview_block_o_entry_contract() -> dict[str, Any]:
    """Build the 17.0 source-only Block O entry contract from the 16.8 closure audit."""
    closure_audit = build_preview_block_n_closure_audit()
    safe_source_exe_direction = _plain_dict_section(
        closure_audit,
        "exe_direction_closure_audit",
    )
    block_n_summary = _build_block_n_summary(closure_audit)
    capability_state = _build_capability_state(closure_audit)
    invariant_state = _build_invariant_state(closure_audit)
    requirement_state = _build_requirement_state(closure_audit)
    source_exe_direction = dict(safe_source_exe_direction)
    entry_source_accepted = _build_entry_source_acceptance(
        closure_audit=closure_audit,
        block_n_summary=block_n_summary,
        capability_state=capability_state,
        invariant_state=invariant_state,
        requirement_state=requirement_state,
        exe_direction=source_exe_direction,
    )
    exe_direction = _build_exe_direction(
        source_exe_direction=source_exe_direction,
        block_n_summary=block_n_summary,
        entry_source_accepted=entry_source_accepted,
    )
    block_o_opened = BLOCK_O_OPENED and entry_source_accepted
    ready_for_block_o_1 = READY_FOR_BLOCK_O_1 and entry_source_accepted
    entry_status = ENTRY_CONTRACT_STATUS if entry_source_accepted else BLOCKED_STATUS
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_o_entry_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_o_entry_contract_status": entry_status,
        "block_o_entry_contract_decision": entry_status.upper(),
        "block_o_opened": block_o_opened,
        "ready_for_block_o_1": ready_for_block_o_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_closure_audit_reference": _build_reference(closure_audit, entry_source_accepted),
        "entry_contract_summary": _build_summary(
            closure_audit=closure_audit,
            block_n_summary=block_n_summary,
            capability_state=capability_state,
            invariant_state=invariant_state,
            requirement_state=requirement_state,
            exe_direction=exe_direction,
            entry_source_accepted=entry_source_accepted,
        ),
        "inherited_block_n_closure_summary": block_n_summary,
        "inherited_capability_state": capability_state,
        "inherited_invariant_state": invariant_state,
        "inherited_requirement_state": requirement_state,
        "exe_direction_entry_contract": exe_direction,
        "fail_closed_entry_decision": _build_fail_closed_decision(
            closure_audit, entry_source_accepted
        ),
        "non_execution_entry_evidence": _build_non_execution_evidence(
            block_n_summary=block_n_summary,
            capability_state=capability_state,
            invariant_state=invariant_state,
            requirement_state=requirement_state,
            entry_source_accepted=entry_source_accepted,
        ),
        "entry_contract_boundaries": _build_entry_boundaries(),
        "source_boundaries": _build_source_boundaries(
            closure_audit, block_n_summary, entry_source_accepted
        ),
        "future_steps": ["functional_preview_17_1_block_o_read_model"],
        "status": STATUS if entry_source_accepted else BLOCKED_STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _block_n_closure_source_identity_is_expected(closure_audit: dict[str, Any]) -> bool:
    return (
        list(closure_audit) == _EXPECTED_SOURCE_TOP_LEVEL_FIELDS
        and closure_audit["schema_version"] == _EXPECTED_SOURCE_SCHEMA_VERSION
        and closure_audit["block_n_closure_audit_kind"] == _EXPECTED_SOURCE_KIND
        and closure_audit["block"] == _EXPECTED_SOURCE_BLOCK
        and closure_audit["step"] == _EXPECTED_SOURCE_STEP
        and closure_audit["status"] == _EXPECTED_SOURCE_STATUS
        and closure_audit["block_n_closure_audit_status"] == _EXPECTED_SOURCE_CLOSURE_AUDIT_STATUS
        and closure_audit["block_n_closure_audit_decision"]
        == _EXPECTED_SOURCE_CLOSURE_AUDIT_STATUS.upper()
        and closure_audit["future_steps"] == _EXPECTED_SOURCE_FUTURE_STEPS
    )


def _real_capability_status_is_exactly_blocked(status: Any) -> bool:
    if type(status) is not dict:
        return False
    return list(status) == _EXPECTED_REAL_CAPABILITY_KEYS and all(
        value == "blocked" for value in status.values()
    )


def _plain_dict_section(source: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in source:
        return {}
    value = source[key]
    if type(value) is not dict:
        return {}
    return value


def _plain_dict_section_is_present(source: dict[str, Any], key: str) -> bool:
    return key in source and type(source[key]) is dict


def _plain_dict_section_has_exact_fields(
    source: dict[str, Any],
    key: str,
    expected_fields: list[str],
) -> bool:
    return _plain_dict_section_is_present(source, key) and list(source[key]) == expected_fields


def _plain_list_section(source: dict[str, Any], key: str) -> list[Any]:
    if key not in source:
        return []
    value = source[key]
    if type(value) is not list:
        return []
    return value


def _exe_direction_source_is_expected(exe_direction: dict[str, Any]) -> bool:
    if list(exe_direction) != _EXPECTED_SOURCE_EXE_DIRECTION_FIELDS:
        return False
    if not all(exe_direction[key] is True for key in _EXPECTED_SOURCE_EXE_TRUE_FIELDS):
        return False
    if not all(exe_direction[key] is False for key in _EXPECTED_SOURCE_EXE_FALSE_FIELDS):
        return False
    return all(
        exe_direction[key] == expected
        for key, expected in _EXPECTED_SOURCE_EXE_STRING_VALUES.items()
    )


def _block_n_closure_rows_are_expected(rows: list[Any]) -> bool:
    if len(rows) != len(_EXPECTED_BLOCK_N_STEP_DATA):
        return False
    if not all(type(row) is dict for row in rows):
        return False
    for index, row in enumerate(rows):
        step, title, artifact_kind = _EXPECTED_BLOCK_N_STEP_DATA[index]
        expected_row_id = step.lower().replace("functional-preview-", "functional_preview_")
        expected_row_id = expected_row_id.replace(".", "_") + "_closure"
        if (
            list(row) != _EXPECTED_BLOCK_N_ROW_FIELDS
            or row["closure_row_id"] != expected_row_id
            or row["step"] != step
            or row["title"] != title
            or row["artifact_kind"] != artifact_kind
            or row["step_complete"] is not True
            or row["source_only"] is not True
            or row["plain_data"] is not True
            or row["execution_authorized"] is not False
            or row["real_capabilities_opened"] is not False
            or row["closure_status"] != "complete"
            or row["closure_result"] != "closed_source_only_execution_blocked"
            or type(row["notes"]) is not str
        ):
            return False
    return True


def _invariant_state_is_self_consistent(state: dict[str, Any]) -> bool:
    if list(state) != _EXPECTED_SOURCE_INVARIANT_SUMMARY_FIELDS + _INVARIANT_STATE_17_0_FIELDS:
        return False
    rows = state["source_invariant_read_rows"]
    if type(rows) is not list:
        return False
    if len(rows) != 12:
        return False
    if not all(
        type(row) is dict and list(row) == _EXPECTED_SOURCE_INVARIANT_ROW_FIELDS for row in rows
    ):
        return False
    if [row["invariant_id"] for row in rows] != _EXPECTED_SOURCE_INVARIANT_IDS:
        return False
    for row in rows:
        for key in [
            "source_invariant_preserved",
            "read_invariant_preserved",
            "invariant_required_for_future_execution",
            "execution_gate_open_now",
            "execution_allowed_now",
            "execution_performed_now",
            "requires_future_explicit_gate",
        ]:
            if type(row[key]) is not bool:
                return False
        invariant_id = row["invariant_id"]
        expected_prefix = "block_n_" + invariant_id + "_contract"
        if (
            row["read_row_id"] != expected_prefix + "_read_model_readiness_matrix_contract_read"
            or row["source_contract_row_id"]
            != expected_prefix + "_read_model_readiness_matrix_contract"
            or row["source_readiness_row_id"] != expected_prefix + "_read_model_readiness_matrix"
            or row["source_read_row_id"] != expected_prefix + "_read_model"
            or row["source_contract_id"] != expected_prefix
            or row["domain"] != "cross_domain"
            or row["display_name"] != invariant_id.replace("_", " ")
            or row["source_contract_result"] != "contracted_invariant_preserved_execution_blocked"
            or row["source_contract_readiness_classification"]
            != "invariant_preserved_execution_not_ready"
            or row["source_invariant_preserved"] is not True
            or row["read_invariant_preserved"] is not True
            or row["invariant_required_for_future_execution"] is not True
            or row["execution_gate_open_now"] is not False
            or row["execution_allowed_now"] is not False
            or row["execution_performed_now"] is not False
            or row["requires_future_explicit_gate"] is not True
            or row["readiness_classification"] != "invariant_preserved_execution_not_ready"
            or row["failure_policy"] != "fail_closed"
            or row["read_result"] != "read_invariant_preserved_execution_blocked"
            or type(row["notes"]) is not str
        ):
            return False
    preserved_count = sum(row["read_invariant_preserved"] is True for row in rows)
    failed_count = len(rows) - preserved_count
    return (
        type(state["invariant_count"]) is int
        and type(state["preserved_invariant_count"]) is int
        and type(state["failed_invariant_count"]) is int
        and state["invariant_count"] == len(rows) == 12
        and state["preserved_invariant_count"] == preserved_count
        and state["failed_invariant_count"] == failed_count
        and state["all_invariants_read"] is True
        and state["all_invariants_preserved"]
        == all(row["read_invariant_preserved"] is True for row in rows)
        and state["all_invariants_require_future_explicit_gate"]
        == all(row["requires_future_explicit_gate"] is True for row in rows)
        and state["execution_gate_open_now"]
        == any(row["execution_gate_open_now"] is True for row in rows)
        and state["execution_allowed_now"]
        == any(row["execution_allowed_now"] is True for row in rows)
        and state["execution_performed_now"]
        == any(row["execution_performed_now"] is True for row in rows)
        and state["failure_policy"] == "fail_closed"
        and state["closure_result"] == _EXPECTED_SOURCE_INVARIANT_CLOSURE_RESULT
    )


def _requirement_state_is_self_consistent(state: dict[str, Any]) -> bool:
    if list(state) != _EXPECTED_SOURCE_REQUIREMENT_SUMMARY_FIELDS + _REQUIREMENT_STATE_17_0_FIELDS:
        return False
    rows = state["source_requirement_read_rows"]
    if type(rows) is not list:
        return False
    if len(rows) != 7:
        return False
    if not all(
        type(row) is dict and list(row) == _EXPECTED_SOURCE_REQUIREMENT_ROW_FIELDS for row in rows
    ):
        return False
    if [row["requirement_id"] for row in rows] != _EXPECTED_SOURCE_REQUIREMENT_IDS:
        return False
    for row in rows:
        for key in [
            "source_required",
            "source_present",
            "source_completed",
            "source_satisfied",
            "required",
            "present",
            "completed",
            "satisfied",
            "missing_blocks_execution",
            "requires_future_explicit_step",
        ]:
            if type(row[key]) is not bool:
                return False
        requirement_id = row["requirement_id"]
        if (
            row["read_row_id"] != requirement_id + "_readiness_contract_read"
            or row["source_contract_row_id"] != requirement_id + "_readiness_contract"
            or row["display_name"] != _EXPECTED_REQUIREMENT_DISPLAY_NAMES[requirement_id]
            or row["applicable_domains"] != _EXPECTED_REQUIREMENT_APPLICABLE_DOMAINS[requirement_id]
            or row["read_result"] != "read_missing_execution_blocked"
            or row["source_required"] is not True
            or row["source_present"] is not False
            or row["source_completed"] is not False
            or row["source_satisfied"] is not False
            or row["failure_policy"] != "fail_closed"
            or type(row["notes"]) is not str
        ):
            return False
        if row["required"] is not True:
            return False
        if row["present"] is False and (
            row["completed"] is not False or row["satisfied"] is not False
        ):
            return False
        if (
            row["present"] is not False
            or row["completed"] is not False
            or row["satisfied"] is not False
            or row["missing_blocks_execution"] is not True
            or row["requires_future_explicit_step"] is not True
        ):
            return False
    return (
        type(state["requirement_count"]) is int
        and type(state["required_requirement_count"]) is int
        and type(state["present_requirement_count"]) is int
        and type(state["completed_requirement_count"]) is int
        and type(state["satisfied_requirement_count"]) is int
        and type(state["missing_requirement_count"]) is int
        and state["requirement_count"] == len(rows) == 7
        and state["required_requirement_count"] == sum(row["required"] is True for row in rows)
        and state["present_requirement_count"] == sum(row["present"] is True for row in rows)
        and state["completed_requirement_count"] == sum(row["completed"] is True for row in rows)
        and state["satisfied_requirement_count"] == sum(row["satisfied"] is True for row in rows)
        and state["missing_requirement_count"] == sum(row["present"] is False for row in rows)
        and state["all_requirements_read"] is True
        and state["all_requirements_required"] == all(row["required"] is True for row in rows)
        and state["all_requirements_missing"] == all(row["present"] is False for row in rows)
        and state["all_requirements_block_execution"]
        == all(row["missing_blocks_execution"] is True for row in rows)
        and state["all_requirements_require_future_explicit_step"]
        == all(row["requires_future_explicit_step"] is True for row in rows)
        and state["failure_policy"] == "fail_closed"
        and state["closure_result"] == _EXPECTED_SOURCE_REQUIREMENT_CLOSURE_RESULT
    )


def _capability_domain_is_expected(
    domain: dict[str, Any],
    *,
    expected_domain: str,
    expected_count: int,
    expected_requirement_ids: list[str],
) -> bool:
    return (
        list(domain) == _EXPECTED_CAPABILITY_DOMAIN_FIELDS
        and domain["domain"] == expected_domain
        and type(domain["capability_count"]) is int
        and type(domain["read_capability_count"]) is int
        and type(domain["ready_capability_count"]) is int
        and type(domain["blocked_capability_count"]) is int
        and domain["capability_count"] == expected_count
        and domain["read_capability_count"] == expected_count
        and domain["ready_capability_count"] == 0
        and domain["blocked_capability_count"] == expected_count
        and domain["required_requirement_ids"] == expected_requirement_ids
        and domain["satisfied_requirement_ids"] == []
        and domain["missing_requirement_ids"] == expected_requirement_ids
        and domain["requirements_complete"] is False
        and domain["domain_ready"] is False
        and domain["execution_authorized"] is False
        and domain["all_capabilities_read"] is True
        and domain["all_capabilities_not_ready"] is True
        and domain["all_capabilities_blocked"] is True
        and domain["failure_policy"] == "fail_closed"
        and domain["domain_closed_in_block_n"] is True
        and domain["domain_enabled_by_closure"] is False
        and domain["closure_result"] == "closed_source_only_execution_blocked"
        and domain["inherited_by_block_o_entry"] is True
        and domain["enabled_by_block_o_entry"] is False
    )


def _source_closure_summary_is_expected(summary: dict[str, Any]) -> bool:
    return (
        list(summary)
        == _EXPECTED_SOURCE_CLOSURE_SUMMARY_TRUE_FIELDS
        + _EXPECTED_SOURCE_CLOSURE_SUMMARY_FALSE_FIELDS
        and all(summary[key] is True for key in _EXPECTED_SOURCE_CLOSURE_SUMMARY_TRUE_FIELDS)
        and all(summary[key] is False for key in _EXPECTED_SOURCE_CLOSURE_SUMMARY_FALSE_FIELDS)
    )


def _source_non_execution_evidence_is_expected(evidence: dict[str, Any]) -> bool:
    return (
        list(evidence)
        == _EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS + _EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS
        and all(evidence[key] is True for key in _EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS)
        and all(evidence[key] is False for key in _EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS)
    )


def _source_closure_boundaries_are_expected(boundaries: dict[str, Any]) -> bool:
    return list(boundaries) == _EXPECTED_SOURCE_CLOSURE_BOUNDARY_FIELDS and all(
        boundaries[key] is True for key in _EXPECTED_SOURCE_CLOSURE_BOUNDARY_FIELDS
    )


def _source_fail_closed_decision_is_expected(decision: dict[str, Any]) -> bool:
    if list(decision) != _EXPECTED_SOURCE_FAIL_CLOSED_DECISION_FIELDS:
        return False
    policy_fields_ok = all(
        decision[key] == "fail_closed"
        for key in _EXPECTED_SOURCE_FAIL_CLOSED_DECISION_FIELDS
        if key.endswith("_policy")
    )
    return (
        policy_fields_ok
        and decision["block_n_closure_audit_in_16_8"] == "closed"
        and decision["block_o_entry_contract_in_17_0"] == "allowed"
        and decision["only_source_only_17_0_handoff_allowed"] is True
        and _real_capability_status_is_exactly_blocked(decision["real_capability_status"])
    )


def _source_readiness_reference_is_expected(reference: dict[str, Any]) -> bool:
    false_by_16_8 = [
        key
        for key in _EXPECTED_SOURCE_READINESS_REFERENCE_FIELDS
        if key.endswith("_by_16_8")
        and key
        not in [
            "source_block_n_safety_gate_readiness_read_model_read_by_16_8",
            "block_n_closure_audit_built_by_16_8",
        ]
    ]
    return (
        list(reference) == _EXPECTED_SOURCE_READINESS_REFERENCE_FIELDS
        and reference["schema_version"] == "preview_block_n_safety_gate_readiness_read_model.v1"
        and reference["block_n_safety_gate_readiness_read_model_kind"]
        == "functional_preview_block_n_safety_gate_readiness_read_model"
        and reference["block"] == "N"
        and reference["step"] == "16.7"
        and reference["block_n_safety_gate_readiness_read_model_status"]
        == _EXPECTED_SOURCE_READINESS_READ_MODEL_STATUS
        and reference["block_n_safety_gate_readiness_read_model_decision"]
        == _EXPECTED_SOURCE_READINESS_READ_MODEL_STATUS.upper()
        and reference["ready_for_block_n_8"] is True
        and reference["next_step"] == "FUNCTIONAL-PREVIEW-16.8"
        and reference["next_step_title"] == "BLOCK N CLOSURE AUDIT"
        and reference["source_block_n_safety_gate_readiness_read_model_step"]
        == "FUNCTIONAL-PREVIEW-16.7"
        and reference["source_block_n_safety_gate_readiness_read_model_read_by_16_8"] is True
        and reference["block_n_safety_gate_readiness_read_model_available_before_closure_audit"]
        is True
        and reference["static_block_n_safety_gate_readiness_read_model_only"] is True
        and reference["block_n_closure_audit_built_by_16_8"] is True
        and reference["ready_for_functional_preview_17_0"] is True
        and all(reference[key] is False for key in false_by_16_8)
    )


def _source_boundaries_are_expected(boundaries: dict[str, Any]) -> bool:
    if list(boundaries) != _EXPECTED_SOURCE_BOUNDARY_FIELDS:
        return False
    nested = boundaries["source_block_n_safety_gate_readiness_contract_boundaries"]
    if type(nested) is not dict:
        return False
    if list(nested) != _EXPECTED_SOURCE_NESTED_BOUNDARY_FIELDS:
        return False
    source_forbidden_fields = [key for key in boundaries if key.startswith("forbidden_")]
    return (
        boundaries["allowed_imports_only"] is True
        and boundaries["source_block_n_safety_gate_readiness_contract"] == "FUNCTIONAL-PREVIEW-16.6"
        and boundaries["source_block_n_safety_gate_readiness_matrix"] == "FUNCTIONAL-PREVIEW-16.5"
        and boundaries["source_block_n_safety_gate_read_model"] == "FUNCTIONAL-PREVIEW-16.4"
        and nested["allowed_imports_only"] is True
        and nested["source_block_n_safety_gate_readiness_matrix"] == "FUNCTIONAL-PREVIEW-16.5"
        and nested["source_block_n_safety_gate_read_model"] == "FUNCTIONAL-PREVIEW-16.4"
        and nested["plain_data_source_only"] is True
        and nested["static_non_evaluating"] is True
        and nested["non_mutating"] is True
        and nested["non_authorizing"] is True
        and nested["can_feed_16_7"] is True
        and nested["can_feed_16_8"] is True
        and boundaries["source_block_n_safety_gate_readiness_read_model"]
        == "FUNCTIONAL-PREVIEW-16.7"
        and boundaries["can_feed_16_8"] is True
        and boundaries["can_close_block_n"] is True
        and boundaries["can_feed_17_0"] is True
        and source_forbidden_fields == _REQUIRED_INHERITED_FORBIDDEN_FIELDS
        and all(boundaries[key] is False for key in _REQUIRED_INHERITED_FORBIDDEN_FIELDS)
    )


def _build_reference(closure_audit: dict[str, Any], entry_source_accepted: bool) -> dict[str, Any]:
    reference = {
        key: (closure_audit[key] if key in closure_audit else None)
        for key in _REFERENCE_SOURCE_FIELDS
    }
    reference.update(
        {
            "source_block_n_closure_audit_step": SOURCE_BLOCK_N_CLOSURE_AUDIT_STEP,
            "source_block_n_closure_audit_read_by_17_0": True,
            "block_n_closure_audit_available_before_block_o_entry": True,
            "static_block_n_closure_audit_only": True,
            "block_o_entry_contract_built_by_17_0": True,
            "block_o_opened_by_17_0": entry_source_accepted,
            "ready_for_functional_preview_17_1": entry_source_accepted,
        }
    )
    for root in _FALSE_BY_17_0_ROOTS:
        reference[root + "_by_17_0"] = False
    return reference


def _build_summary(
    *,
    closure_audit: dict[str, Any],
    block_n_summary: dict[str, Any],
    capability_state: dict[str, Any],
    invariant_state: dict[str, Any],
    requirement_state: dict[str, Any],
    exe_direction: dict[str, Any],
    entry_source_accepted: bool,
) -> dict[str, bool]:
    overall = capability_state["overall"]
    source_decision = _plain_dict_section(closure_audit, "fail_closed_closure_decision")
    closure_summary = _plain_dict_section(closure_audit, "closure_audit_summary")
    source_real_capability_status = (
        source_decision["real_capability_status"]
        if _source_fail_closed_decision_is_expected(source_decision)
        else {}
    )
    return {
        "block_n_closure_audit_available": True,
        "block_n_closed": (
            "block_n_closed" in closure_audit and closure_audit["block_n_closed"] is True
        ),
        "block_o_entry_contract_built": True,
        "block_o_opened": entry_source_accepted,
        "ready_for_block_o_1": entry_source_accepted,
        "ready_for_functional_preview_17_1": entry_source_accepted,
        "block_m_closure_preserved": (
            _source_closure_summary_is_expected(closure_summary)
            and closure_summary["block_m_closure_preserved"] is True
        ),
        "exe_direction_preserved": (
            exe_direction["block_o_entry_contract_confirms_exe_direction"] is True
        ),
        "entry_contract_source_only": True,
        "entry_contract_plain_data_only": True,
        "entry_contract_static_only": True,
        "entry_contract_read_only": True,
        "entry_contract_non_evaluating": True,
        "entry_contract_non_mutating": True,
        "entry_contract_non_authorizing": True,
        "all_block_n_steps_preserved": block_n_summary["block_n_closure_preserved"] is True,
        "all_capabilities_inherited": overall["all_capabilities_inherited"] is True,
        "all_execution_capabilities_fail_closed": (
            overall["failure_policy"] == "fail_closed"
            and _real_capability_status_is_exactly_blocked(source_real_capability_status)
        ),
        "all_execution_capabilities_not_ready": overall["all_capabilities_not_ready"] is True,
        "all_execution_capabilities_blocked": overall["all_capabilities_blocked"] is True,
        "all_requirements_inherited": requirement_state["inherited_by_block_o_entry"] is True,
        "all_requirements_missing": (
            "all_requirements_missing" in requirement_state
            and requirement_state["all_requirements_missing"] is True
        ),
        "all_requirements_block_execution": (
            "all_requirements_block_execution" in requirement_state
            and requirement_state["all_requirements_block_execution"] is True
        ),
        "all_invariants_inherited": invariant_state["inherited_by_block_o_entry"] is True,
        "all_invariants_preserved": (
            "all_invariants_preserved" in invariant_state
            and invariant_state["all_invariants_preserved"] is True
        ),
        "all_domains_not_ready": overall["ready_capability_count"] == 0,
        "all_domains_execution_unauthorized": overall["execution_authorized"] is False,
        "only_source_only_17_1_handoff_allowed": entry_source_accepted,
        "any_closure_recalculated_now": False,
        "any_readiness_recalculated_from_environment_now": False,
        "any_gate_evaluated_now": False,
        "any_gate_condition_met_now": False,
        "any_gate_open_now": False,
        "any_gate_state_mutated_now": False,
        "any_execution_authorized_now": False,
        "any_execution_allowed_now": False,
        "any_execution_performed_now": False,
        "any_validation_completed_now": False,
        "any_requirement_present_now": False,
        "any_requirement_satisfied_now": False,
        "any_capability_ready_now": False,
        "packaging_release_domain_ready_now": False,
        "runtime_safety_domain_ready_now": False,
        "exe_build_ready_now": False,
        "exe_packaging_ready_now": False,
        "exe_release_ready_now": False,
        "runtime_enabled_by_block_o_entry": False,
        "packaging_enabled_by_block_o_entry": False,
        "release_enabled_by_block_o_entry": False,
        "orders_enabled_by_block_o_entry": False,
    }


def _build_block_n_summary(closure_audit: dict[str, Any]) -> dict[str, Any]:
    rows = _plain_list_section(closure_audit, "block_n_step_closure_rows")
    row_count = len(rows)
    rows_have_expected_shape = all(
        type(row) is dict and list(row) == _EXPECTED_BLOCK_N_ROW_FIELDS for row in rows
    )
    source_steps = [row["step"] for row in rows] if rows_have_expected_shape else []
    expected_step_chain_present = source_steps == _EXPECTED_BLOCK_N_STEPS
    exact_rows = _block_n_closure_rows_are_expected(rows)
    exact_step_count = row_count == len(_EXPECTED_BLOCK_N_STEPS)
    source_block_n_closed = (
        closure_audit["block_n_closed"] if "block_n_closed" in closure_audit else None
    )
    source_ready_for_block_o_0 = (
        closure_audit["ready_for_block_o_0"] if "ready_for_block_o_0" in closure_audit else None
    )
    source_next_step = closure_audit["next_step"] if "next_step" in closure_audit else None
    source_next_step_title = (
        closure_audit["next_step_title"] if "next_step_title" in closure_audit else None
    )
    all_steps_complete = (
        exact_step_count
        and expected_step_chain_present
        and exact_rows
        and all(row["step_complete"] is True for row in rows)
    )
    all_steps_source_only = (
        exact_step_count
        and expected_step_chain_present
        and exact_rows
        and all(row["source_only"] is True for row in rows)
    )
    all_steps_plain_data = (
        exact_step_count
        and expected_step_chain_present
        and exact_rows
        and all(row["plain_data"] is True for row in rows)
    )
    all_steps_execution_unauthorized = (
        exact_step_count
        and expected_step_chain_present
        and exact_rows
        and all(row["execution_authorized"] is False for row in rows)
    )
    all_real_capabilities_closed = (
        exact_step_count
        and expected_step_chain_present
        and exact_rows
        and all(row["real_capabilities_opened"] is False for row in rows)
    )
    block_n_closure_preserved = (
        source_block_n_closed is True
        and all_steps_complete
        and all_steps_source_only
        and all_steps_plain_data
        and all_steps_execution_unauthorized
        and all_real_capabilities_closed
    )
    closure_result = (
        "block_n_closure_inherited_execution_blocked"
        if block_n_closure_preserved
        else "block_n_closure_not_preserved_execution_blocked"
    )
    return {
        "source_block_n_closed": source_block_n_closed,
        "source_ready_for_block_o_0": source_ready_for_block_o_0,
        "source_next_step": source_next_step,
        "source_next_step_title": source_next_step_title,
        "block_n_step_count": row_count,
        "completed_block_n_step_count": (
            sum(row["step_complete"] is True for row in rows) if rows_have_expected_shape else 0
        ),
        "all_block_n_steps_complete": all_steps_complete,
        "all_block_n_steps_source_only": all_steps_source_only,
        "all_block_n_steps_plain_data": all_steps_plain_data,
        "all_block_n_steps_execution_unauthorized": all_steps_execution_unauthorized,
        "all_block_n_steps_real_capabilities_closed": all_real_capabilities_closed,
        "block_n_closure_preserved": block_n_closure_preserved,
        "block_o_entry_does_not_reopen_block_n": (
            source_block_n_closed is True
            and exact_step_count
            and expected_step_chain_present
            and all_steps_execution_unauthorized
            and all_real_capabilities_closed
        ),
        "closure_result": closure_result,
    }


def _build_capability_state(closure_audit: dict[str, Any]) -> dict[str, Any]:
    packaging_release_inherited = _plain_dict_section_is_present(
        closure_audit,
        "packaging_release_closure_summary",
    )
    runtime_safety_inherited = _plain_dict_section_is_present(
        closure_audit,
        "runtime_safety_closure_summary",
    )
    packaging_release_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "packaging_release_closure_summary",
        _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS,
    )
    runtime_safety_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "runtime_safety_closure_summary",
        _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS,
    )
    source_packaging_release = _plain_dict_section(
        closure_audit,
        "packaging_release_closure_summary",
    )
    source_runtime_safety = _plain_dict_section(
        closure_audit,
        "runtime_safety_closure_summary",
    )
    packaging_release = dict(source_packaging_release)
    runtime_safety = dict(source_runtime_safety)
    packaging_release.update(
        {
            "inherited_by_block_o_entry": packaging_release_inherited,
            "enabled_by_block_o_entry": False,
        }
    )
    runtime_safety.update(
        {
            "inherited_by_block_o_entry": runtime_safety_inherited,
            "enabled_by_block_o_entry": False,
        }
    )
    domains = [packaging_release, runtime_safety]
    domains_are_expected = (
        packaging_release_source_shape_ok
        and runtime_safety_source_shape_ok
        and _capability_domain_is_expected(
            packaging_release,
            expected_domain="packaging_release",
            expected_count=22,
            expected_requirement_ids=_EXPECTED_PACKAGING_RELEASE_REQUIREMENT_IDS,
        )
        and _capability_domain_is_expected(
            runtime_safety,
            expected_domain="runtime_safety",
            expected_count=18,
            expected_requirement_ids=_EXPECTED_RUNTIME_SAFETY_REQUIREMENT_IDS,
        )
    )
    overall = (
        {
            "total_capability_count": sum(domain["capability_count"] for domain in domains),
            "read_capability_count": sum(domain["read_capability_count"] for domain in domains),
            "ready_capability_count": sum(domain["ready_capability_count"] for domain in domains),
            "blocked_capability_count": sum(
                domain["blocked_capability_count"] for domain in domains
            ),
            "all_capabilities_inherited": (
                packaging_release_inherited and runtime_safety_inherited
            ),
            "all_capabilities_read": all(
                domain["all_capabilities_read"] is True for domain in domains
            ),
            "all_capabilities_not_ready": all(
                domain["all_capabilities_not_ready"] is True for domain in domains
            ),
            "all_capabilities_blocked": all(
                domain["all_capabilities_blocked"] is True for domain in domains
            ),
            "execution_authorized": any(
                domain["execution_authorized"] is True for domain in domains
            ),
            "enabled_by_block_o_entry": False,
            "failure_policy": "fail_closed",
            "entry_result": "inherited_not_ready_execution_blocked",
        }
        if domains_are_expected
        else {
            "total_capability_count": 0,
            "read_capability_count": 0,
            "ready_capability_count": 0,
            "blocked_capability_count": 0,
            "all_capabilities_inherited": False,
            "all_capabilities_read": False,
            "all_capabilities_not_ready": False,
            "all_capabilities_blocked": False,
            "execution_authorized": False,
            "enabled_by_block_o_entry": False,
            "failure_policy": "fail_closed",
            "entry_result": "inherited_source_invalid_execution_blocked",
        }
    )
    return {
        "packaging_release": packaging_release,
        "runtime_safety": runtime_safety,
        "overall": overall,
    }


def _build_invariant_state(closure_audit: dict[str, Any]) -> dict[str, Any]:
    source_inherited = _plain_dict_section_is_present(
        closure_audit,
        "cross_domain_invariant_closure_summary",
    )
    state = dict(_plain_dict_section(closure_audit, "cross_domain_invariant_closure_summary"))
    state.update(
        {
            "inherited_by_block_o_entry": source_inherited,
            "revalidated_by_block_o_entry": False,
            "entry_result": (
                "invariants_inherited_execution_blocked"
                if source_inherited
                else "invariant_source_unavailable_execution_blocked"
            ),
        }
    )
    return state


def _build_requirement_state(closure_audit: dict[str, Any]) -> dict[str, Any]:
    source_inherited = _plain_dict_section_is_present(
        closure_audit,
        "validation_requirement_closure_summary",
    )
    state = dict(_plain_dict_section(closure_audit, "validation_requirement_closure_summary"))
    state.update(
        {
            "inherited_by_block_o_entry": source_inherited,
            "validated_by_block_o_entry": False,
            "entry_result": (
                "requirements_inherited_missing_execution_blocked"
                if source_inherited
                else "requirement_source_unavailable_execution_blocked"
            ),
        }
    )
    return state


def _build_exe_direction(
    *,
    source_exe_direction: dict[str, Any],
    block_n_summary: dict[str, Any],
    entry_source_accepted: bool,
) -> dict[str, Any]:
    exe = dict(source_exe_direction)
    exe_source_expected = _exe_direction_source_is_expected(source_exe_direction)
    exe["block_o_entry_contract_confirms_exe_direction"] = exe_source_expected
    exe["block_n_closure_source_preserved"] = block_n_summary["block_n_closure_preserved"]
    exe["entry_contract_is_not_execution_authorization"] = True
    if entry_source_accepted:
        entry_result = "exe_direction_inherited_block_o_opened_execution_not_ready"
    elif exe_source_expected:
        entry_result = "exe_direction_inherited_block_o_entry_blocked"
    else:
        entry_result = "exe_direction_not_confirmed_block_o_entry_blocked"
    exe["entry_result"] = entry_result
    return exe


def _build_entry_source_acceptance(
    *,
    closure_audit: dict[str, Any],
    block_n_summary: dict[str, Any],
    capability_state: dict[str, Any],
    invariant_state: dict[str, Any],
    requirement_state: dict[str, Any],
    exe_direction: dict[str, Any],
) -> bool:
    overall = capability_state["overall"]
    packaging_release = capability_state["packaging_release"]
    runtime_safety = capability_state["runtime_safety"]
    source_identity_ok = _block_n_closure_source_identity_is_expected(closure_audit)
    closure_decision = _plain_dict_section(closure_audit, "fail_closed_closure_decision")
    closure_summary = _plain_dict_section(closure_audit, "closure_audit_summary")
    source_fail_closed_decision_ok = _source_fail_closed_decision_is_expected(closure_decision)
    real_capability_status = (
        closure_decision["real_capability_status"] if source_fail_closed_decision_ok else {}
    )
    source_readiness_reference_ok = _source_readiness_reference_is_expected(
        _plain_dict_section(
            closure_audit,
            "block_n_safety_gate_readiness_read_model_reference",
        )
    )
    full_source_boundaries_ok = _source_boundaries_are_expected(
        _plain_dict_section(closure_audit, "source_boundaries")
    )
    closure_summary_ok = _source_closure_summary_is_expected(closure_summary)
    source_evidence_ok = _source_non_execution_evidence_is_expected(
        _plain_dict_section(closure_audit, "non_execution_closure_evidence")
    )
    source_closure_boundaries_ok = _source_closure_boundaries_are_expected(
        _plain_dict_section(closure_audit, "closure_audit_boundaries")
    )
    packaging_release_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "packaging_release_closure_summary",
        _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS,
    )
    runtime_safety_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "runtime_safety_closure_summary",
        _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS,
    )
    invariant_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "cross_domain_invariant_closure_summary",
        _EXPECTED_SOURCE_INVARIANT_SUMMARY_FIELDS,
    )
    requirement_source_shape_ok = _plain_dict_section_has_exact_fields(
        closure_audit,
        "validation_requirement_closure_summary",
        _EXPECTED_SOURCE_REQUIREMENT_SUMMARY_FIELDS,
    )
    source_handoff_ok = (
        source_identity_ok
        and source_fail_closed_decision_ok
        and closure_summary_ok
        and closure_audit["block_n_closed"] is True
        and closure_audit["ready_for_block_o_0"] is True
        and closure_audit["next_step"] == "FUNCTIONAL-PREVIEW-17.0"
        and closure_audit["next_step_title"] == "BLOCK O ENTRY CONTRACT"
        and closure_decision["block_n_closure_audit_in_16_8"] == "closed"
        and closure_decision["block_o_entry_contract_in_17_0"] == "allowed"
        and closure_decision["only_source_only_17_0_handoff_allowed"] is True
        and closure_summary["block_m_closure_preserved"] is True
        and closure_summary["ready_for_functional_preview_17_0"] is True
    )
    block_n_ok = (
        block_n_summary["block_n_closure_preserved"] is True
        and block_n_summary["block_o_entry_does_not_reopen_block_n"] is True
        and block_n_summary["block_n_step_count"] == 8
        and block_n_summary["completed_block_n_step_count"] == 8
        and block_n_summary["all_block_n_steps_complete"] is True
        and block_n_summary["all_block_n_steps_source_only"] is True
        and block_n_summary["all_block_n_steps_plain_data"] is True
        and block_n_summary["all_block_n_steps_execution_unauthorized"] is True
        and block_n_summary["all_block_n_steps_real_capabilities_closed"] is True
    )
    capability_domains_ok = _capability_domain_is_expected(
        packaging_release,
        expected_domain="packaging_release",
        expected_count=22,
        expected_requirement_ids=_EXPECTED_PACKAGING_RELEASE_REQUIREMENT_IDS,
    ) and _capability_domain_is_expected(
        runtime_safety,
        expected_domain="runtime_safety",
        expected_count=18,
        expected_requirement_ids=_EXPECTED_RUNTIME_SAFETY_REQUIREMENT_IDS,
    )
    capability_ok = (
        overall["total_capability_count"] == 40
        and overall["read_capability_count"] == 40
        and overall["ready_capability_count"] == 0
        and overall["blocked_capability_count"] == 40
        and overall["all_capabilities_inherited"] is True
        and overall["all_capabilities_read"] is True
        and overall["all_capabilities_not_ready"] is True
        and overall["all_capabilities_blocked"] is True
        and overall["execution_authorized"] is False
        and overall["enabled_by_block_o_entry"] is False
        and overall["failure_policy"] == "fail_closed"
        and packaging_release_source_shape_ok
        and runtime_safety_source_shape_ok
        and capability_domains_ok
        and _real_capability_status_is_exactly_blocked(real_capability_status)
    )
    invariant_ok = (
        invariant_source_shape_ok
        and _invariant_state_is_self_consistent(invariant_state)
        and invariant_state["invariant_count"] == 12
        and invariant_state["preserved_invariant_count"] == 12
        and invariant_state["failed_invariant_count"] == 0
        and invariant_state["all_invariants_read"] is True
        and invariant_state["all_invariants_preserved"] is True
        and invariant_state["all_invariants_require_future_explicit_gate"] is True
        and invariant_state["execution_gate_open_now"] is False
        and invariant_state["execution_allowed_now"] is False
        and invariant_state["execution_performed_now"] is False
        and invariant_state["failure_policy"] == "fail_closed"
    )
    requirement_ok = (
        requirement_source_shape_ok
        and _requirement_state_is_self_consistent(requirement_state)
        and requirement_state["requirement_count"] == 7
        and requirement_state["required_requirement_count"] == 7
        and requirement_state["present_requirement_count"] == 0
        and requirement_state["completed_requirement_count"] == 0
        and requirement_state["satisfied_requirement_count"] == 0
        and requirement_state["missing_requirement_count"] == 7
        and requirement_state["all_requirements_read"] is True
        and requirement_state["all_requirements_required"] is True
        and requirement_state["all_requirements_missing"] is True
        and requirement_state["all_requirements_block_execution"] is True
        and requirement_state["all_requirements_require_future_explicit_step"] is True
        and requirement_state["failure_policy"] == "fail_closed"
    )
    exe_ok = _exe_direction_source_is_expected(exe_direction)
    return (
        source_identity_ok
        and source_handoff_ok
        and source_fail_closed_decision_ok
        and source_readiness_reference_ok
        and full_source_boundaries_ok
        and closure_summary_ok
        and block_n_ok
        and capability_ok
        and invariant_ok
        and requirement_ok
        and exe_ok
        and source_evidence_ok
        and source_closure_boundaries_ok
    )


def _build_fail_closed_decision(
    closure_audit: dict[str, Any], entry_source_accepted: bool
) -> dict[str, Any]:
    source_decision = _plain_dict_section(closure_audit, "fail_closed_closure_decision")
    source_real_capability_status_inherited = (
        "real_capability_status" in source_decision
        and type(source_decision["real_capability_status"]) is dict
    )
    source_real_capability_status = (
        source_decision["real_capability_status"] if source_real_capability_status_inherited else {}
    )
    return {
        "missing_block_n_closure_audit_policy": "fail_closed",
        "missing_block_n_step_closure_policy": "fail_closed",
        "missing_inherited_capability_state_policy": "fail_closed",
        "missing_inherited_requirement_state_policy": "fail_closed",
        "missing_inherited_invariant_state_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_block_o_entry_contract_policy": "fail_closed",
        "block_n_closure_audit_in_16_8": (
            "preserved" if entry_source_accepted else "not_preserved"
        ),
        "block_o_entry_contract_in_17_0": "opened" if entry_source_accepted else "blocked",
        "block_o_read_model_in_17_1": "allowed" if entry_source_accepted else "blocked",
        "only_source_only_17_1_handoff_allowed": entry_source_accepted,
        "real_capability_status": dict(source_real_capability_status),
        "real_capability_status_inherited_from_16_8": source_real_capability_status_inherited,
        "real_capability_status_modified_by_17_0": False,
    }


def _build_non_execution_evidence(
    *,
    block_n_summary: dict[str, Any],
    capability_state: dict[str, Any],
    invariant_state: dict[str, Any],
    requirement_state: dict[str, Any],
    entry_source_accepted: bool,
) -> dict[str, bool]:
    overall = capability_state["overall"]
    source_claims = {
        "source_block_n_closure_audit_read": True,
        "block_n_closure_preserved": entry_source_accepted
        and block_n_summary["block_n_closure_preserved"] is True,
        "block_o_entry_contract_built": True,
        "block_o_entry_contract_only": True,
        "block_o_opened": entry_source_accepted,
        "ready_for_block_o_1": entry_source_accepted,
        "all_block_n_steps_inherited": entry_source_accepted
        and block_n_summary["block_n_closure_preserved"] is True,
        "all_capability_states_inherited": entry_source_accepted
        and overall["all_capabilities_inherited"] is True,
        "all_capability_states_not_ready": entry_source_accepted
        and overall["all_capabilities_not_ready"] is True,
        "all_invariant_states_inherited": entry_source_accepted
        and invariant_state["inherited_by_block_o_entry"] is True,
        "all_invariant_states_preserved": entry_source_accepted
        and invariant_state["all_invariants_preserved"] is True,
        "all_requirement_states_inherited": entry_source_accepted
        and requirement_state["inherited_by_block_o_entry"] is True,
        "all_requirement_states_missing": entry_source_accepted
        and requirement_state["all_requirements_missing"] is True,
        "all_execution_authorization_false": entry_source_accepted
        and overall["execution_authorized"] is False,
        "all_capabilities_fail_closed": entry_source_accepted
        and overall["failure_policy"] == "fail_closed",
    }
    false_keys = [
        "closure_recalculated",
        "readiness_recalculated_from_environment",
        "gate_evaluation_performed",
        "gate_condition_accepted",
        "gate_opened",
        "gate_mutated",
        "confirmation_accepted",
        "validation_performed",
        "authorization_performed",
        "execution_performed",
        "packaging_performed",
        "build_performed",
        "release_performed",
        "runtime_performed",
        "orders_performed",
        "network_io_performed",
        "filesystem_io_performed",
        "private_endpoint_accessed",
        "credentials_read",
        "config_env_secrets_read",
        "real_capabilities_opened_by_block_o_entry",
    ]
    return {**source_claims, **{key: False for key in false_keys}}


def _build_entry_boundaries() -> dict[str, bool]:
    keys = [
        "block_o_entry_contract_is_plain_data_only",
        "block_o_entry_contract_is_source_only",
        "block_o_entry_contract_reads_16_8_only",
        "block_o_entry_contract_preserves_block_m_closure",
        "block_o_entry_contract_preserves_block_n_closure",
        "block_o_entry_contract_preserves_exe_direction_without_packaging",
        "block_o_entry_contract_is_static_and_non_evaluating",
        "block_o_entry_contract_is_non_mutating",
        "block_o_entry_contract_is_non_authorizing",
        "block_o_entry_contract_can_open_block_o",
        "block_o_entry_contract_can_feed_17_1_read_model",
        "cannot_recalculate_block_n_closure",
        "cannot_recalculate_readiness_from_environment",
        "cannot_evaluate",
        "cannot_accept_condition",
        "cannot_open_real_gate",
        "cannot_mutate_gate",
        "cannot_accept_confirmations",
        "cannot_perform_validations",
        "cannot_authorize",
        "cannot_package",
        "cannot_build",
        "cannot_release",
        "cannot_perform_artifact_work",
        "cannot_run_runtime",
        "cannot_generate_orders",
        "cannot_" + "sub" + "mit_orders",
        "cannot_" + "can" + "cel_orders",
        "cannot_" + "re" + "place_orders",
        "cannot_use_network",
        "cannot_use_filesystem",
        "cannot_access_private_endpoints",
        "cannot_read_credentials",
        "cannot_read_config_env_secrets",
        "cannot_change_qml_or_bridge",
        "cannot_create_execution_side_effects",
    ]
    return {key: True for key in keys}


def _build_source_boundaries(
    closure_audit: dict[str, Any],
    block_n_summary: dict[str, Any],
    entry_source_accepted: bool,
) -> dict[str, Any]:
    source = dict(_plain_dict_section(closure_audit, "source_boundaries"))
    source["source_block_n_closure_audit"] = SOURCE_BLOCK_N_CLOSURE_AUDIT_STEP
    source["block_n_closure_audit_source_preserved"] = (
        entry_source_accepted and block_n_summary["block_n_closure_preserved"] is True
    )
    source["can_open_block_o"] = entry_source_accepted
    source["can_feed_17_1"] = entry_source_accepted
    return source
