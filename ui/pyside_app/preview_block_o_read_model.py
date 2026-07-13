"""FUNCTIONAL-PREVIEW-17.1 Block O source-only read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_o_entry_contract import (
    build_preview_block_o_entry_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_o_read_model.v1"
KIND: Final[str] = "functional_preview_block_o_read_model"
BLOCK_ID: Final[str] = "O"
STEP_ID: Final[str] = "17.1"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.2"
NEXT_STEP_TITLE: Final[str] = "BLOCK O EXECUTION AUTHORIZATION MATRIX"
READ_MODEL_STATUS: Final[str] = (
    "block_o_read_model_ready_17_0_entry_contract_consumed_"
    "block_o_opened_block_n_closed_block_m_closure_preserved_"
    "desktop_exe_direction_preserved_source_only_plain_data_"
    "static_read_projection_only_all_capability_domains_read_"
    "all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_"
    "all_requirements_read_all_requirements_missing_"
    "all_requirements_block_execution_"
    "all_invariants_read_all_invariants_preserved_"
    "all_execution_unauthorized_all_gates_closed_"
    "no_source_state_recalculation_no_gate_evaluation_"
    "no_validation_no_confirmation_acceptance_"
    "no_authorization_no_packaging_no_build_no_release_"
    "no_runtime_no_orders_no_private_endpoints_"
    "no_network_io_no_credentials_no_filesystem_io_"
    "ready_for_functional_preview_17_2_"
    "execution_authorization_matrix"
)
READ_MODEL_DECISION: Final[str] = READ_MODEL_STATUS.upper()
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_17_2_block_o_read_model_source_not_accepted"
)
STATUS: Final[str] = "ready_for_functional_preview_17_2_block_o_execution_authorization_matrix"
FUTURE_STEPS: Final[list[str]] = ["functional_preview_17_2_block_o_execution_authorization_matrix"]
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_o_read_model_kind",
    "block",
    "step",
    "block_o_read_model_status",
    "block_o_read_model_decision",
    "block_o_read_model_ready",
    "ready_for_block_o_2",
    "next_step",
    "next_step_title",
    "block_o_entry_contract_reference",
    "read_model_summary",
    "block_n_closure_read_state",
    "capability_read_state",
    "invariant_read_state",
    "requirement_read_state",
    "exe_direction_read_state",
    "fail_closed_read_decision",
    "non_execution_read_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_EXPECTED_SOURCE_SCHEMA_VERSION: Final[str] = "preview_block_o_entry_contract.v1"
_EXPECTED_SOURCE_KIND: Final[str] = "functional_preview_block_o_entry_contract"
_EXPECTED_SOURCE_BLOCK: Final[str] = "O"
_EXPECTED_SOURCE_STEP: Final[str] = "17.0"
_EXPECTED_SOURCE_ENTRY_STATUS: Final[str] = (
    "block_o_entry_contract_ready_block_n_closure_audit_consumed_block_n_closed_block_o_opened_"
    "steps_16_0_through_16_8_preserved_block_m_closure_preserved_desktop_exe_direction_preserved_"
    "source_only_plain_data_static_contract_only_all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_all_requirements_missing_all_invariants_preserved_"
    "all_execution_unauthorized_all_gates_closed_no_readiness_recalculation_no_gate_evaluation_"
    "no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_"
    "no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_"
    "only_source_only_17_1_handoff_allowed"
)
_EXPECTED_SOURCE_STATUS: Final[str] = "ready_for_functional_preview_17_1_block_o_read_model"
_EXPECTED_SOURCE_FUTURE_STEPS: Final[list[str]] = ["functional_preview_17_1_block_o_read_model"]
_EXPECTED_SOURCE_TOP_LEVEL_FIELDS: Final[list[str]] = [
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
_REFERENCE_SOURCE_FIELDS: Final[list[str]] = _EXPECTED_SOURCE_TOP_LEVEL_FIELDS[:10]
_REFERENCE_HANDOFF_FIELDS: Final[list[str]] = [
    "source_block_o_entry_contract_step",
    "source_block_o_entry_contract_read_by_17_1",
    "block_o_entry_contract_available_before_read_model",
    "static_block_o_entry_contract_only",
    "block_o_read_model_built_by_17_1",
    "block_o_read_model_ready_by_17_1",
    "ready_for_functional_preview_17_2",
]
_FALSE_BY_17_1_ROOTS: Final[list[str]] = [
    "source_state_recalculated",
    "block_n_closure_recalculated",
    "capability_state_recalculated",
    "invariant_state_recalculated",
    "requirement_state_recalculated",
    "exe_direction_recalculated",
    "readiness_recalculated_from_environment",
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
_FALSE_BY_17_1_FIELDS: Final[list[str]] = [f"{root}_by_17_1" for root in _FALSE_BY_17_1_ROOTS]
_EXE_READ_STATE_17_1_FIELDS: Final[list[str]] = [
    "read_by_block_o_read_model",
    "recalculated_by_block_o_read_model",
    "block_o_read_model_confirms_exe_direction",
    "block_o_read_model_is_not_execution_authorization",
    "block_o_read_model_result",
]

_EXPECTED_SOURCE_ENTRY_SUMMARY_TRUE_FIELDS: Final[list[str]] = [
    "block_n_closure_audit_available",
    "block_n_closed",
    "block_o_entry_contract_built",
    "block_o_opened",
    "ready_for_block_o_1",
    "ready_for_functional_preview_17_1",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "entry_contract_source_only",
    "entry_contract_plain_data_only",
    "entry_contract_static_only",
    "entry_contract_read_only",
    "entry_contract_non_evaluating",
    "entry_contract_non_mutating",
    "entry_contract_non_authorizing",
    "all_block_n_steps_preserved",
    "all_capabilities_inherited",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_requirements_inherited",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_inherited",
    "all_invariants_preserved",
    "all_domains_not_ready",
    "all_domains_execution_unauthorized",
    "only_source_only_17_1_handoff_allowed",
]
_EXPECTED_SOURCE_ENTRY_SUMMARY_FALSE_FIELDS: Final[list[str]] = [
    "any_closure_recalculated_now",
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
    "runtime_enabled_by_block_o_entry",
    "packaging_enabled_by_block_o_entry",
    "release_enabled_by_block_o_entry",
    "orders_enabled_by_block_o_entry",
]
_READ_MODEL_SUMMARY_TRUE_FIELDS: Final[list[str]] = [
    "block_o_entry_contract_available",
    "block_o_entry_contract_source_accepted",
    "block_o_opened",
    "block_o_read_model_built",
    "block_o_read_model_ready",
    "ready_for_block_o_2",
    "ready_for_functional_preview_17_2",
    "block_n_closed",
    "block_n_closure_preserved",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "read_model_source_only",
    "read_model_plain_data_only",
    "read_model_static_only",
    "read_model_read_only",
    "read_model_non_evaluating",
    "read_model_non_mutating",
    "read_model_non_authorizing",
    "all_capability_domains_read",
    "all_capabilities_not_ready",
    "all_capabilities_blocked",
    "all_execution_capabilities_fail_closed",
    "all_requirements_read",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_read",
    "all_invariants_preserved",
    "all_execution_unauthorized",
    "all_gates_closed",
    "only_source_only_17_2_handoff_allowed",
]
_READ_MODEL_SUMMARY_FALSE_FIELDS: Final[list[str]] = [
    "any_source_state_recalculated_now",
    "any_readiness_recalculated_from_environment_now",
    "any_gate_evaluated_now",
    "any_gate_condition_met_now",
    "any_gate_open_now",
    "any_gate_state_mutated_now",
    "any_execution_authorization_computed_now",
    "any_execution_authorized_now",
    "any_execution_allowed_now",
    "any_execution_performed_now",
    "any_validation_completed_now",
    "any_requirement_present_now",
    "any_requirement_completed_now",
    "any_requirement_satisfied_now",
    "any_capability_ready_now",
    "any_capability_enabled_by_read_model",
    "packaging_release_domain_ready_now",
    "runtime_safety_domain_ready_now",
    "exe_build_ready_now",
    "exe_packaging_ready_now",
    "exe_release_ready_now",
    "runtime_enabled_by_block_o_read_model",
    "packaging_enabled_by_block_o_read_model",
    "release_enabled_by_block_o_read_model",
    "orders_enabled_by_block_o_read_model",
]
_READ_MODEL_BOUNDARY_FIELDS: Final[list[str]] = [
    "block_o_read_model_is_plain_data_only",
    "block_o_read_model_is_source_only",
    "block_o_read_model_reads_17_0_only",
    "block_o_read_model_preserves_block_m_closure",
    "block_o_read_model_preserves_block_n_closure",
    "block_o_read_model_preserves_block_o_entry",
    "block_o_read_model_preserves_exe_direction",
    "block_o_read_model_is_static_read_projection_only",
    "block_o_read_model_is_non_evaluating",
    "block_o_read_model_is_non_mutating",
    "block_o_read_model_is_non_authorizing",
    "block_o_read_model_can_feed_17_2_source_only_matrix",
    "cannot_recalculate_source_state",
    "cannot_recalculate_readiness_from_environment",
    "cannot_evaluate_gate",
    "cannot_accept_condition",
    "cannot_open_real_gate",
    "cannot_mutate_gate",
    "cannot_accept_confirmations",
    "cannot_perform_validations",
    "cannot_compute_real_execution_authorization",
    "cannot_authorize",
    "cannot_package",
    "cannot_build",
    "cannot_release",
    "cannot_perform_artifact_work",
    "cannot_run_runtime",
    "cannot_generate_orders",
    "cannot_submit_orders",
    "cannot_cancel_orders",
    "cannot_replace_orders",
    "cannot_use_network",
    "cannot_use_filesystem",
    "cannot_access_private_endpoints",
    "cannot_read_credentials",
    "cannot_read_config_env_secrets",
    "cannot_change_qml_or_bridge",
    "cannot_create_execution_side_effects",
]
_FAIL_CLOSED_READ_DECISION_FIELDS: Final[list[str]] = [
    "missing_block_o_entry_contract_policy",
    "missing_block_n_closure_state_policy",
    "missing_capability_state_policy",
    "missing_invariant_state_policy",
    "missing_requirement_state_policy",
    "missing_exe_direction_state_policy",
    "missing_operator_confirmation_policy",
    "missing_environment_validation_policy",
    "missing_artifact_validation_policy",
    "missing_release_validation_policy",
    "missing_runtime_validation_policy",
    "missing_credentials_validation_policy",
    "missing_future_explicit_gate_policy",
    "failed_block_o_read_model_policy",
    "block_o_entry_contract_in_17_0",
    "block_o_read_model_in_17_1",
    "block_o_execution_authorization_matrix_in_17_2",
    "only_source_only_17_2_handoff_allowed",
    "real_capability_status",
    "real_capability_status_inherited_from_17_0",
    "real_capability_status_modified_by_17_1",
]
_EXPECTED_SOURCE_REFERENCE_SOURCE_FIELDS: Final[list[str]] = [
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
_EXPECTED_SOURCE_REFERENCE_HANDOFF_FIELDS: Final[list[str]] = [
    "source_block_n_closure_audit_step",
    "source_block_n_closure_audit_read_by_17_0",
    "block_n_closure_audit_available_before_block_o_entry",
    "static_block_n_closure_audit_only",
    "block_o_entry_contract_built_by_17_0",
    "block_o_opened_by_17_0",
    "ready_for_functional_preview_17_1",
]
_EXPECTED_SOURCE_REFERENCE_FALSE_BY_17_0_FIELDS: Final[list[str]] = [
    f"{root}_by_17_0"
    for root in [
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
]
_EXPECTED_SOURCE_REFERENCE_FIELDS: Final[list[str]] = (
    _EXPECTED_SOURCE_REFERENCE_SOURCE_FIELDS
    + _EXPECTED_SOURCE_REFERENCE_HANDOFF_FIELDS
    + _EXPECTED_SOURCE_REFERENCE_FALSE_BY_17_0_FIELDS
)
_EXPECTED_SOURCE_BLOCK_N_STATE_FIELDS: Final[list[str]] = [
    "source_block_n_closed",
    "source_ready_for_block_o_0",
    "source_next_step",
    "source_next_step_title",
    "block_n_step_count",
    "completed_block_n_step_count",
    "all_block_n_steps_complete",
    "all_block_n_steps_source_only",
    "all_block_n_steps_plain_data",
    "all_block_n_steps_execution_unauthorized",
    "all_block_n_steps_real_capabilities_closed",
    "block_n_closure_preserved",
    "block_o_entry_does_not_reopen_block_n",
    "closure_result",
]
_EXPECTED_SOURCE_CAPABILITY_STATE_FIELDS: Final[list[str]] = [
    "packaging_release",
    "runtime_safety",
    "overall",
]
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
    "inherited_by_block_o_entry",
    "enabled_by_block_o_entry",
]
_EXPECTED_SOURCE_CAPABILITY_OVERALL_FIELDS: Final[list[str]] = [
    "total_capability_count",
    "read_capability_count",
    "ready_capability_count",
    "blocked_capability_count",
    "all_capabilities_inherited",
    "all_capabilities_read",
    "all_capabilities_not_ready",
    "all_capabilities_blocked",
    "execution_authorized",
    "enabled_by_block_o_entry",
    "failure_policy",
    "entry_result",
]
_PACKAGING_REQUIREMENT_IDS: Final[list[str]] = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "future_explicit_gate",
]
_RUNTIME_REQUIREMENT_IDS: Final[list[str]] = [
    "operator_confirmation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]
_EXPECTED_SOURCE_INVARIANT_STATE_FIELDS: Final[list[str]] = [
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
    "inherited_by_block_o_entry",
    "revalidated_by_block_o_entry",
    "entry_result",
]
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
_EXPECTED_SOURCE_REQUIREMENT_STATE_FIELDS: Final[list[str]] = [
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
    "inherited_by_block_o_entry",
    "validated_by_block_o_entry",
    "entry_result",
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
_EXPECTED_REQUIREMENT_DISPLAY_NAMES: Final[dict[str, str]] = {
    "operator_confirmation": "Operator Confirmation",
    "environment_validation": "Environment Validation",
    "artifact_validation": "Artifact Validation",
    "release_validation": "Release Validation",
    "runtime_validation": "Runtime Validation",
    "credentials_validation": "Credentials Validation",
    "future_explicit_gate": "Future Explicit Gate",
}
_EXPECTED_REQUIREMENT_APPLICABLE_DOMAINS: Final[dict[str, list[str]]] = {
    "operator_confirmation": ["packaging_release", "runtime_safety"],
    "environment_validation": ["packaging_release"],
    "artifact_validation": ["packaging_release"],
    "release_validation": ["packaging_release"],
    "runtime_validation": ["runtime_safety"],
    "credentials_validation": ["runtime_safety"],
    "future_explicit_gate": ["packaging_release", "runtime_safety", "cross_domain"],
}
_EXPECTED_SOURCE_EXE_FIELDS: Final[list[str]] = [
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
    "block_o_entry_contract_confirms_exe_direction",
    "block_n_closure_source_preserved",
    "entry_contract_is_not_execution_authorization",
    "entry_result",
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
    "block_o_entry_contract_confirms_exe_direction",
    "block_n_closure_source_preserved",
    "entry_contract_is_not_execution_authorization",
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
    "entry_result": "exe_direction_inherited_block_o_opened_execution_not_ready",
}
_EXPECTED_SOURCE_FAIL_CLOSED_FIELDS: Final[list[str]] = [
    "missing_block_n_closure_audit_policy",
    "missing_block_n_step_closure_policy",
    "missing_inherited_capability_state_policy",
    "missing_inherited_requirement_state_policy",
    "missing_inherited_invariant_state_policy",
    "missing_operator_confirmation_policy",
    "missing_environment_validation_policy",
    "missing_artifact_validation_policy",
    "missing_release_validation_policy",
    "missing_runtime_validation_policy",
    "missing_credentials_validation_policy",
    "missing_future_explicit_gate_policy",
    "failed_block_o_entry_contract_policy",
    "block_n_closure_audit_in_16_8",
    "block_o_entry_contract_in_17_0",
    "block_o_read_model_in_17_1",
    "only_source_only_17_1_handoff_allowed",
    "real_capability_status",
    "real_capability_status_inherited_from_16_8",
    "real_capability_status_modified_by_17_0",
]
_EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS: Final[list[str]] = [
    "source_block_n_closure_audit_read",
    "block_n_closure_preserved",
    "block_o_entry_contract_built",
    "block_o_entry_contract_only",
    "block_o_opened",
    "ready_for_block_o_1",
    "all_block_n_steps_inherited",
    "all_capability_states_inherited",
    "all_capability_states_not_ready",
    "all_invariant_states_inherited",
    "all_invariant_states_preserved",
    "all_requirement_states_inherited",
    "all_requirement_states_missing",
    "all_execution_authorization_false",
    "all_capabilities_fail_closed",
]
_EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS: Final[list[str]] = [
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
_EXPECTED_SOURCE_EVIDENCE_FIELDS: Final[list[str]] = (
    _EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS + _EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS
)
_EXPECTED_SOURCE_ENTRY_BOUNDARY_FIELDS: Final[list[str]] = [
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
    "cannot_submit_orders",
    "cannot_cancel_orders",
    "cannot_replace_orders",
    "cannot_use_network",
    "cannot_use_filesystem",
    "cannot_access_private_endpoints",
    "cannot_read_credentials",
    "cannot_read_config_env_secrets",
    "cannot_change_qml_or_bridge",
    "cannot_create_execution_side_effects",
]
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
    "source_block_n_closure_audit",
    "block_n_closure_audit_source_preserved",
    "can_open_block_o",
    "can_feed_17_1",
]
_EXPECTED_SOURCE_BOUNDARY_17_0_FIELDS: Final[list[str]] = [
    "source_block_o_entry_contract",
    "block_o_entry_contract_source_preserved",
    "can_build_block_o_read_model",
    "can_feed_17_2",
]
_EXPECTED_SOURCE_BOUNDARY_NESTED_FIELDS: Final[list[str]] = [
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

_REAL_CAPABILITY_KEYS: Final[list[str]] = [
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
    "create" + "_" + "order",
    "submit_order",
    "cancel_order",
    "replace_order",
    "fetch" + "_" + "balance",
    "private_endpoint",
    "network",
    "credentials",
    "config_env_secrets",
    "qml_bridge",
    "c" + "cxt",
]


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    return value


def _plain_dict_section(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    if type(value) is dict:
        return _copy_plain(value)
    return {}


def _plain_list_section(source: dict[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    if type(value) is list:
        return _copy_plain(value)
    return []


def _plain_dict_section_is_present(source: dict[str, Any], key: str) -> bool:
    return type(source.get(key)) is dict


def _plain_dict_section_has_exact_fields(
    source: dict[str, Any], key: str, expected_fields: list[str]
) -> bool:
    return type(source.get(key)) is dict and list(source[key]) == expected_fields


def _all_plain_json(value: Any) -> bool:
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is list:
        return all(_all_plain_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _all_plain_json(item) for key, item in value.items())
    return False


def _block_o_entry_source_identity_is_expected(source: dict[str, Any]) -> bool:
    return (
        type(source) is dict
        and list(source) == _EXPECTED_SOURCE_TOP_LEVEL_FIELDS
        and source.get("schema_version") == _EXPECTED_SOURCE_SCHEMA_VERSION
        and source.get("block_o_entry_contract_kind") == _EXPECTED_SOURCE_KIND
        and source.get("block") == _EXPECTED_SOURCE_BLOCK
        and source.get("step") == _EXPECTED_SOURCE_STEP
        and source.get("block_o_entry_contract_status") == _EXPECTED_SOURCE_ENTRY_STATUS
        and source.get("block_o_entry_contract_decision") == _EXPECTED_SOURCE_ENTRY_STATUS.upper()
        and source.get("block_o_opened") is True
        and source.get("ready_for_block_o_1") is True
        and source.get("next_step") == "FUNCTIONAL-PREVIEW-17.1"
        and source.get("next_step_title") == "BLOCK O READ MODEL"
        and source.get("status") == _EXPECTED_SOURCE_STATUS
        and source.get("future_steps") == _EXPECTED_SOURCE_FUTURE_STEPS
    )


def _source_reference_is_expected(reference: dict[str, Any]) -> bool:
    return (
        type(reference) is dict
        and list(reference) == _EXPECTED_SOURCE_REFERENCE_FIELDS
        and reference.get("schema_version") == "preview_block_n_closure_audit.v1"
        and reference.get("block_n_closure_audit_kind")
        == "functional_preview_block_n_closure_audit"
        and reference.get("block") == "N"
        and reference.get("step") == "16.8"
        and reference.get("block_n_closure_audit_status")
        == "block_n_closure_audit_complete_16_7_readiness_read_model_consumed_steps_16_0_through_16_7_complete_block_m_closure_preserved_block_n_closed_exe_direction_preserved_source_only_plain_data_static_audit_only_all_capability_rows_read_all_requirements_missing_all_invariants_preserved_all_execution_capabilities_not_ready_all_execution_capabilities_blocked_all_execution_unauthorized_all_gates_closed_no_readiness_recalculation_no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_only_source_only_block_o_handoff_allowed"
        and reference.get("block_n_closure_audit_decision")
        == "BLOCK_N_CLOSURE_AUDIT_COMPLETE_16_7_READINESS_READ_MODEL_CONSUMED_STEPS_16_0_THROUGH_16_7_COMPLETE_BLOCK_M_CLOSURE_PRESERVED_BLOCK_N_CLOSED_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_AUDIT_ONLY_ALL_CAPABILITY_ROWS_READ_ALL_REQUIREMENTS_MISSING_ALL_INVARIANTS_PRESERVED_ALL_EXECUTION_CAPABILITIES_NOT_READY_ALL_EXECUTION_CAPABILITIES_BLOCKED_ALL_EXECUTION_UNAUTHORIZED_ALL_GATES_CLOSED_NO_READINESS_RECALCULATION_NO_GATE_EVALUATION_NO_VALIDATION_NO_CONFIRMATION_ACCEPTANCE_NO_AUTHORIZATION_NO_PACKAGING_NO_BUILD_NO_RELEASE_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_ONLY_SOURCE_ONLY_BLOCK_O_HANDOFF_ALLOWED"
        and reference.get("block_n_closed") is True
        and reference.get("ready_for_block_o_0") is True
        and reference.get("next_step") == "FUNCTIONAL-PREVIEW-17.0"
        and reference.get("next_step_title") == "BLOCK O ENTRY CONTRACT"
        and reference.get("source_block_n_closure_audit_step") == "FUNCTIONAL-PREVIEW-16.8"
        and reference.get("source_block_n_closure_audit_read_by_17_0") is True
        and reference.get("block_n_closure_audit_available_before_block_o_entry") is True
        and reference.get("static_block_n_closure_audit_only") is True
        and reference.get("block_o_entry_contract_built_by_17_0") is True
        and reference.get("block_o_opened_by_17_0") is True
        and reference.get("ready_for_functional_preview_17_1") is True
        and all(
            reference.get(key) is False for key in _EXPECTED_SOURCE_REFERENCE_FALSE_BY_17_0_FIELDS
        )
    )


def _real_capability_status_is_exactly_blocked(status: Any) -> bool:
    if type(status) is not dict:
        return False
    return list(status) == _REAL_CAPABILITY_KEYS and all(
        status.get(key) == "blocked" for key in _REAL_CAPABILITY_KEYS
    )


def _entry_summary_is_expected(summary: dict[str, Any]) -> bool:
    return (
        list(summary)
        == _EXPECTED_SOURCE_ENTRY_SUMMARY_TRUE_FIELDS + _EXPECTED_SOURCE_ENTRY_SUMMARY_FALSE_FIELDS
        and all(summary.get(key) is True for key in _EXPECTED_SOURCE_ENTRY_SUMMARY_TRUE_FIELDS)
        and all(summary.get(key) is False for key in _EXPECTED_SOURCE_ENTRY_SUMMARY_FALSE_FIELDS)
    )


def _block_n_state_is_expected(state: dict[str, Any]) -> bool:
    return (
        type(state) is dict
        and list(state) == _EXPECTED_SOURCE_BLOCK_N_STATE_FIELDS
        and state.get("source_block_n_closed") is True
        and state.get("source_ready_for_block_o_0") is True
        and state.get("source_next_step") == "FUNCTIONAL-PREVIEW-17.0"
        and state.get("source_next_step_title") == "BLOCK O ENTRY CONTRACT"
        and _count_is(state.get("block_n_step_count"), 8)
        and _count_is(state.get("completed_block_n_step_count"), 8)
        and state.get("all_block_n_steps_complete") is True
        and state.get("all_block_n_steps_source_only") is True
        and state.get("all_block_n_steps_plain_data") is True
        and state.get("all_block_n_steps_execution_unauthorized") is True
        and state.get("all_block_n_steps_real_capabilities_closed") is True
        and state.get("block_n_closure_preserved") is True
        and state.get("block_o_entry_does_not_reopen_block_n") is True
        and state.get("closure_result") == "block_n_closure_inherited_execution_blocked"
    )


def _count_is(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _capability_domain_is_expected(
    domain: Any,
    *,
    name: str,
    count: int,
    requirement_ids: list[str],
) -> bool:
    return (
        type(domain) is dict
        and list(domain) == _EXPECTED_SOURCE_CAPABILITY_DOMAIN_FIELDS
        and domain.get("domain") == name
        and _count_is(domain.get("capability_count"), count)
        and _count_is(domain.get("read_capability_count"), count)
        and _count_is(domain.get("ready_capability_count"), 0)
        and _count_is(domain.get("blocked_capability_count"), count)
        and domain.get("required_requirement_ids") == requirement_ids
        and domain.get("satisfied_requirement_ids") == []
        and domain.get("missing_requirement_ids") == requirement_ids
        and domain.get("requirements_complete") is False
        and domain.get("domain_ready") is False
        and domain.get("execution_authorized") is False
        and domain.get("all_capabilities_read") is True
        and domain.get("all_capabilities_not_ready") is True
        and domain.get("all_capabilities_blocked") is True
        and domain.get("failure_policy") == "fail_closed"
        and domain.get("domain_closed_in_block_n") is True
        and domain.get("domain_enabled_by_closure") is False
        and domain.get("closure_result") == "closed_source_only_execution_blocked"
        and domain.get("inherited_by_block_o_entry") is True
        and domain.get("enabled_by_block_o_entry") is False
    )


def _capability_state_is_expected(state: dict[str, Any]) -> bool:
    if type(state) is not dict or list(state) != _EXPECTED_SOURCE_CAPABILITY_STATE_FIELDS:
        return False
    packaging = state.get("packaging_release")
    runtime = state.get("runtime_safety")
    overall = state.get("overall")
    if type(overall) is not dict or list(overall) != _EXPECTED_SOURCE_CAPABILITY_OVERALL_FIELDS:
        return False
    return (
        _capability_domain_is_expected(
            packaging,
            name="packaging_release",
            count=22,
            requirement_ids=_PACKAGING_REQUIREMENT_IDS,
        )
        and _capability_domain_is_expected(
            runtime, name="runtime_safety", count=18, requirement_ids=_RUNTIME_REQUIREMENT_IDS
        )
        and _count_is(overall.get("total_capability_count"), 40)
        and _count_is(overall.get("read_capability_count"), 40)
        and _count_is(overall.get("ready_capability_count"), 0)
        and _count_is(overall.get("blocked_capability_count"), 40)
        and overall.get("all_capabilities_inherited") is True
        and overall.get("all_capabilities_read") is True
        and overall.get("all_capabilities_not_ready") is True
        and overall.get("all_capabilities_blocked") is True
        and overall.get("execution_authorized") is False
        and overall.get("enabled_by_block_o_entry") is False
        and overall.get("failure_policy") == "fail_closed"
        and overall.get("entry_result") == "inherited_not_ready_execution_blocked"
    )


def _invariant_row_is_expected(row: Any, invariant_id: str) -> bool:
    return (
        type(row) is dict
        and list(row) == _EXPECTED_SOURCE_INVARIANT_ROW_FIELDS
        and row.get("invariant_id") == invariant_id
        and row.get("read_row_id")
        == f"block_n_{invariant_id}_contract_read_model_readiness_matrix_contract_read"
        and row.get("source_contract_row_id")
        == f"block_n_{invariant_id}_contract_read_model_readiness_matrix_contract"
        and row.get("source_readiness_row_id")
        == f"block_n_{invariant_id}_contract_read_model_readiness_matrix"
        and row.get("source_read_row_id") == f"block_n_{invariant_id}_contract_read_model"
        and row.get("source_contract_id") == f"block_n_{invariant_id}_contract"
        and row.get("domain") == "cross_domain"
        and row.get("display_name") == invariant_id.replace("_", " ")
        and row.get("source_contract_result") == "contracted_invariant_preserved_execution_blocked"
        and row.get("source_contract_readiness_classification")
        == "invariant_preserved_execution_not_ready"
        and row.get("source_invariant_preserved") is True
        and row.get("read_invariant_preserved") is True
        and row.get("invariant_required_for_future_execution") is True
        and row.get("execution_gate_open_now") is False
        and row.get("execution_allowed_now") is False
        and row.get("execution_performed_now") is False
        and row.get("requires_future_explicit_gate") is True
        and row.get("readiness_classification") == "invariant_preserved_execution_not_ready"
        and row.get("failure_policy") == "fail_closed"
        and row.get("read_result") == "read_invariant_preserved_execution_blocked"
        and type(row.get("notes")) is str
    )


def _invariant_state_is_expected(state: dict[str, Any]) -> bool:
    if type(state) is not dict or list(state) != _EXPECTED_SOURCE_INVARIANT_STATE_FIELDS:
        return False
    rows = state.get("source_invariant_read_rows")
    if type(rows) is not list or len(rows) != 12:
        return False
    if [
        row.get("invariant_id") if type(row) is dict else None for row in rows
    ] != _EXPECTED_SOURCE_INVARIANT_IDS:
        return False
    return (
        all(
            _invariant_row_is_expected(row, invariant_id)
            for row, invariant_id in zip(rows, _EXPECTED_SOURCE_INVARIANT_IDS)
        )
        and _count_is(state.get("invariant_count"), len(rows))
        and _count_is(state.get("preserved_invariant_count"), len(rows))
        and _count_is(state.get("failed_invariant_count"), 0)
        and state.get("all_invariants_read") is True
        and state.get("all_invariants_preserved") is True
        and state.get("all_invariants_require_future_explicit_gate") is True
        and state.get("execution_gate_open_now") is False
        and state.get("execution_allowed_now") is False
        and state.get("execution_performed_now") is False
        and state.get("failure_policy") == "fail_closed"
        and state.get("closure_result") == "closed_invariants_preserved_execution_blocked"
        and state.get("inherited_by_block_o_entry") is True
        and state.get("revalidated_by_block_o_entry") is False
        and state.get("entry_result") == "invariants_inherited_execution_blocked"
    )


def _requirement_row_is_expected(row: Any, requirement_id: str) -> bool:
    return (
        type(row) is dict
        and list(row) == _EXPECTED_SOURCE_REQUIREMENT_ROW_FIELDS
        and row.get("requirement_id") == requirement_id
        and row.get("read_row_id") == f"{requirement_id}_readiness_contract_read"
        and row.get("source_contract_row_id") == f"{requirement_id}_readiness_contract"
        and row.get("display_name") == _EXPECTED_REQUIREMENT_DISPLAY_NAMES[requirement_id]
        and row.get("source_required") is True
        and row.get("source_present") is False
        and row.get("source_completed") is False
        and row.get("source_satisfied") is False
        and row.get("required") is True
        and row.get("present") is False
        and row.get("completed") is False
        and row.get("satisfied") is False
        and row.get("applicable_domains")
        == _EXPECTED_REQUIREMENT_APPLICABLE_DOMAINS[requirement_id]
        and row.get("missing_blocks_execution") is True
        and row.get("requires_future_explicit_step") is True
        and row.get("failure_policy") == "fail_closed"
        and row.get("read_result") == "read_missing_execution_blocked"
        and type(row.get("notes")) is str
    )


def _requirement_state_is_expected(state: dict[str, Any]) -> bool:
    if type(state) is not dict or list(state) != _EXPECTED_SOURCE_REQUIREMENT_STATE_FIELDS:
        return False
    rows = state.get("source_requirement_read_rows")
    if type(rows) is not list or len(rows) != 7:
        return False
    if [
        row.get("requirement_id") if type(row) is dict else None for row in rows
    ] != _EXPECTED_SOURCE_REQUIREMENT_IDS:
        return False
    return (
        all(
            _requirement_row_is_expected(row, requirement_id)
            for row, requirement_id in zip(rows, _EXPECTED_SOURCE_REQUIREMENT_IDS)
        )
        and _count_is(state.get("requirement_count"), len(rows))
        and _count_is(state.get("required_requirement_count"), len(rows))
        and _count_is(state.get("present_requirement_count"), 0)
        and _count_is(state.get("completed_requirement_count"), 0)
        and _count_is(state.get("satisfied_requirement_count"), 0)
        and _count_is(state.get("missing_requirement_count"), len(rows))
        and state.get("all_requirements_read") is True
        and state.get("all_requirements_required") is True
        and state.get("all_requirements_missing") is True
        and state.get("all_requirements_block_execution") is True
        and state.get("all_requirements_require_future_explicit_step") is True
        and state.get("failure_policy") == "fail_closed"
        and state.get("closure_result") == "closed_requirements_missing_execution_blocked"
        and state.get("inherited_by_block_o_entry") is True
        and state.get("validated_by_block_o_entry") is False
        and state.get("entry_result") == "requirements_inherited_missing_execution_blocked"
    )


def _exe_state_is_expected(state: dict[str, Any]) -> bool:
    return (
        type(state) is dict
        and list(state) == _EXPECTED_SOURCE_EXE_FIELDS
        and all(state.get(key) is True for key in _EXPECTED_SOURCE_EXE_TRUE_FIELDS)
        and all(state.get(key) is False for key in _EXPECTED_SOURCE_EXE_FALSE_FIELDS)
        and all(
            state.get(key) == value for key, value in _EXPECTED_SOURCE_EXE_STRING_VALUES.items()
        )
    )


def _fail_closed_source_is_expected(source: dict[str, Any]) -> bool:
    decision = source.get("fail_closed_entry_decision")
    if type(decision) is not dict or list(decision) != _EXPECTED_SOURCE_FAIL_CLOSED_FIELDS:
        return False
    return (
        all(decision.get(key) == "fail_closed" for key in _EXPECTED_SOURCE_FAIL_CLOSED_FIELDS[:13])
        and decision.get("block_n_closure_audit_in_16_8") == "preserved"
        and decision.get("block_o_entry_contract_in_17_0") == "opened"
        and decision.get("block_o_read_model_in_17_1") == "allowed"
        and decision.get("only_source_only_17_1_handoff_allowed") is True
        and _real_capability_status_is_exactly_blocked(decision.get("real_capability_status"))
        and decision.get("real_capability_status_inherited_from_16_8") is True
        and decision.get("real_capability_status_modified_by_17_0") is False
    )


def _evidence_is_expected(evidence: dict[str, Any]) -> bool:
    return (
        type(evidence) is dict
        and list(evidence) == _EXPECTED_SOURCE_EVIDENCE_FIELDS
        and all(evidence.get(key) is True for key in _EXPECTED_SOURCE_EVIDENCE_TRUE_FIELDS)
        and all(evidence.get(key) is False for key in _EXPECTED_SOURCE_EVIDENCE_FALSE_FIELDS)
    )


def _entry_boundaries_are_expected(boundaries: dict[str, Any]) -> bool:
    return (
        type(boundaries) is dict
        and list(boundaries) == _EXPECTED_SOURCE_ENTRY_BOUNDARY_FIELDS
        and all(boundaries.get(key) is True for key in _EXPECTED_SOURCE_ENTRY_BOUNDARY_FIELDS)
    )


def _source_boundaries_is_expected(boundaries: dict[str, Any]) -> bool:
    if type(boundaries) is not dict or list(boundaries) != _EXPECTED_SOURCE_BOUNDARY_FIELDS:
        return False
    nested = boundaries.get("source_block_n_safety_gate_readiness_contract_boundaries")
    return (
        type(nested) is dict
        and list(nested) == _EXPECTED_SOURCE_BOUNDARY_NESTED_FIELDS
        and nested.get("allowed_imports_only") is True
        and nested.get("source_block_n_safety_gate_readiness_matrix") == "FUNCTIONAL-PREVIEW-16.5"
        and nested.get("source_block_n_safety_gate_read_model") == "FUNCTIONAL-PREVIEW-16.4"
        and all(
            nested.get(key) is True
            for key in _EXPECTED_SOURCE_BOUNDARY_NESTED_FIELDS
            if key
            not in (
                "source_block_n_safety_gate_readiness_matrix",
                "source_block_n_safety_gate_read_model",
            )
        )
        and boundaries.get("allowed_imports_only") is True
        and boundaries.get("source_block_n_safety_gate_readiness_contract")
        == "FUNCTIONAL-PREVIEW-16.6"
        and boundaries.get("source_block_n_safety_gate_readiness_matrix")
        == "FUNCTIONAL-PREVIEW-16.5"
        and boundaries.get("source_block_n_safety_gate_read_model") == "FUNCTIONAL-PREVIEW-16.4"
        and boundaries.get("source_block_n_safety_gate_readiness_read_model")
        == "FUNCTIONAL-PREVIEW-16.7"
        and boundaries.get("source_block_n_closure_audit") == "FUNCTIONAL-PREVIEW-16.8"
        and boundaries.get("can_feed_16_8") is True
        and boundaries.get("can_close_block_n") is True
        and boundaries.get("can_feed_17_0") is True
        and boundaries.get("block_n_closure_audit_source_preserved") is True
        and boundaries.get("can_open_block_o") is True
        and boundaries.get("can_feed_17_1") is True
        and all(
            boundaries.get(key) is False
            for key in _EXPECTED_SOURCE_BOUNDARY_FIELDS
            if key.startswith("forbidden_")
        )
        and all(key not in boundaries for key in _EXPECTED_SOURCE_BOUNDARY_17_0_FIELDS)
    )


def _build_read_model_source_acceptance(
    *,
    source: dict[str, Any],
    block_n_state: dict[str, Any],
    capability_state: dict[str, Any],
    invariant_state: dict[str, Any],
    requirement_state: dict[str, Any],
    exe_state: dict[str, Any],
    block_n_source_valid: bool,
    capability_source_valid: bool,
    invariant_source_valid: bool,
    requirement_source_valid: bool,
    exe_source_valid: bool,
) -> bool:
    source_identity_ok = _block_o_entry_source_identity_is_expected(source)
    source_reference_ok = _plain_dict_section_has_exact_fields(
        source, "block_n_closure_audit_reference", _EXPECTED_SOURCE_REFERENCE_FIELDS
    ) and _source_reference_is_expected(source.get("block_n_closure_audit_reference"))
    entry_summary_ok = _plain_dict_section_has_exact_fields(
        source,
        "entry_contract_summary",
        _EXPECTED_SOURCE_ENTRY_SUMMARY_TRUE_FIELDS + _EXPECTED_SOURCE_ENTRY_SUMMARY_FALSE_FIELDS,
    ) and _entry_summary_is_expected(source.get("entry_contract_summary"))
    block_n_state_ok = block_n_source_valid
    capability_state_ok = capability_source_valid
    invariant_state_ok = invariant_source_valid
    requirement_state_ok = requirement_source_valid
    exe_state_ok = exe_source_valid
    fail_closed_decision_ok = _plain_dict_section_has_exact_fields(
        source, "fail_closed_entry_decision", _EXPECTED_SOURCE_FAIL_CLOSED_FIELDS
    ) and _fail_closed_source_is_expected(source)
    evidence_ok = _plain_dict_section_has_exact_fields(
        source, "non_execution_entry_evidence", _EXPECTED_SOURCE_EVIDENCE_FIELDS
    ) and _evidence_is_expected(source.get("non_execution_entry_evidence"))
    entry_boundaries_ok = _plain_dict_section_has_exact_fields(
        source, "entry_contract_boundaries", _EXPECTED_SOURCE_ENTRY_BOUNDARY_FIELDS
    ) and _entry_boundaries_are_expected(source.get("entry_contract_boundaries"))
    source_boundaries_ok = _plain_dict_section_has_exact_fields(
        source, "source_boundaries", _EXPECTED_SOURCE_BOUNDARY_FIELDS
    ) and _source_boundaries_is_expected(source.get("source_boundaries"))
    plain_json_ok = _all_plain_json(source)
    return (
        source_identity_ok
        and source_reference_ok
        and entry_summary_ok
        and block_n_state_ok
        and capability_state_ok
        and invariant_state_ok
        and requirement_state_ok
        and exe_state_ok
        and fail_closed_decision_ok
        and evidence_ok
        and entry_boundaries_ok
        and source_boundaries_ok
        and plain_json_ok
    )


def _with_read_fields(
    state: dict[str, Any],
    valid: bool,
    result_ok: str,
    result_bad: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    copied = _copy_plain(state)
    copied["read_by_block_o_read_model"] = valid
    copied["recalculated_by_block_o_read_model"] = False
    if extra:
        for key, value in extra.items():
            copied[key] = value
    copied["read_result"] = result_ok if valid else result_bad
    return copied


def _build_exe_read_state(source_state: dict[str, Any], *, source_valid: bool) -> dict[str, Any]:
    copied = _copy_plain(source_state)
    copied["read_by_block_o_read_model"] = source_valid
    copied["recalculated_by_block_o_read_model"] = False
    copied["block_o_read_model_confirms_exe_direction"] = source_valid
    copied["block_o_read_model_is_not_execution_authorization"] = True
    copied["block_o_read_model_result"] = (
        "exe_direction_read_preserved_execution_not_ready"
        if source_valid
        else "exe_direction_source_invalid_execution_blocked"
    )
    return copied


def _build_reference(source: dict[str, Any], accepted: bool) -> dict[str, Any]:
    reference = {
        key: _copy_plain(source.get(key)) if key in source else None
        for key in _REFERENCE_SOURCE_FIELDS
    }
    reference.update(
        {
            "source_block_o_entry_contract_step": "FUNCTIONAL-PREVIEW-17.0",
            "source_block_o_entry_contract_read_by_17_1": True,
            "block_o_entry_contract_available_before_read_model": True,
            "static_block_o_entry_contract_only": True,
            "block_o_read_model_built_by_17_1": True,
            "block_o_read_model_ready_by_17_1": accepted,
            "ready_for_functional_preview_17_2": accepted,
        }
    )
    reference.update({key: False for key in _FALSE_BY_17_1_FIELDS})
    return reference


def _build_summary(accepted: bool) -> dict[str, Any]:
    constant_true = {
        "read_model_source_only",
        "read_model_plain_data_only",
        "read_model_static_only",
        "read_model_read_only",
        "read_model_non_evaluating",
        "read_model_non_mutating",
        "read_model_non_authorizing",
        "block_o_read_model_built",
    }
    summary = {
        key: (True if key in constant_true else accepted) for key in _READ_MODEL_SUMMARY_TRUE_FIELDS
    }
    summary.update({key: False for key in _READ_MODEL_SUMMARY_FALSE_FIELDS})
    return summary


def _build_fail_closed(source: dict[str, Any], accepted: bool) -> dict[str, Any]:
    source_decision = source.get("fail_closed_entry_decision")
    status = (
        source_decision.get("real_capability_status") if type(source_decision) is dict else None
    )
    status_copy = _copy_plain(status) if type(status) is dict else {}
    result = {key: "fail_closed" for key in _FAIL_CLOSED_READ_DECISION_FIELDS[:14]}
    result.update(
        {
            "block_o_entry_contract_in_17_0": "preserved" if accepted else "not_preserved",
            "block_o_read_model_in_17_1": "ready" if accepted else "blocked",
            "block_o_execution_authorization_matrix_in_17_2": "allowed" if accepted else "blocked",
            "only_source_only_17_2_handoff_allowed": accepted,
            "real_capability_status": status_copy,
            "real_capability_status_inherited_from_17_0": type(status) is dict,
            "real_capability_status_modified_by_17_1": False,
        }
    )
    return result


def _build_evidence(
    *,
    accepted: bool,
    block_n_source_valid: bool,
    capability_source_valid: bool,
    invariant_source_valid: bool,
    requirement_source_valid: bool,
    exe_source_valid: bool,
) -> dict[str, Any]:
    evidence = {
        "source_block_o_entry_contract_read": True,
        "source_block_o_entry_contract_accepted": accepted,
        "block_o_read_model_built": True,
        "block_o_remains_open": accepted,
        "block_n_closure_read": block_n_source_valid,
        "all_capability_states_read": capability_source_valid,
        "all_capability_states_blocked": capability_source_valid,
        "all_requirement_states_read": requirement_source_valid,
        "all_requirement_states_missing": requirement_source_valid,
        "all_invariant_states_read": invariant_source_valid,
        "all_invariant_states_preserved": invariant_source_valid,
        "exe_direction_read": exe_source_valid,
        "exe_direction_preserved": exe_source_valid,
        "all_execution_authorization_false": accepted,
        "all_capabilities_fail_closed": accepted,
    }
    evidence.update({f"{root}_by_17_1": False for root in _FALSE_BY_17_1_ROOTS})
    evidence["real_capabilities_opened_by_block_o_read_model"] = False
    return evidence


def _build_source_boundaries(source: dict[str, Any], accepted: bool) -> dict[str, Any]:
    boundaries = _plain_dict_section(source, "source_boundaries")
    boundaries["source_block_o_entry_contract"] = "FUNCTIONAL-PREVIEW-17.0"
    boundaries["block_o_entry_contract_source_preserved"] = accepted
    boundaries["can_build_block_o_read_model"] = accepted
    boundaries["can_feed_17_2"] = accepted
    return boundaries


def build_preview_block_o_read_model() -> dict[str, Any]:
    source = build_preview_block_o_entry_contract()
    if type(source) is not dict:
        source = {}
    block_n_state = _plain_dict_section(source, "inherited_block_n_closure_summary")
    capability_state = _plain_dict_section(source, "inherited_capability_state")
    invariant_state = _plain_dict_section(source, "inherited_invariant_state")
    requirement_state = _plain_dict_section(source, "inherited_requirement_state")
    exe_state = _plain_dict_section(source, "exe_direction_entry_contract")
    block_n_source_valid = _plain_dict_section_has_exact_fields(
        source, "inherited_block_n_closure_summary", _EXPECTED_SOURCE_BLOCK_N_STATE_FIELDS
    ) and _block_n_state_is_expected(block_n_state)
    capability_source_valid = _plain_dict_section_has_exact_fields(
        source, "inherited_capability_state", _EXPECTED_SOURCE_CAPABILITY_STATE_FIELDS
    ) and _capability_state_is_expected(capability_state)
    invariant_source_valid = _plain_dict_section_has_exact_fields(
        source, "inherited_invariant_state", _EXPECTED_SOURCE_INVARIANT_STATE_FIELDS
    ) and _invariant_state_is_expected(invariant_state)
    requirement_source_valid = _plain_dict_section_has_exact_fields(
        source, "inherited_requirement_state", _EXPECTED_SOURCE_REQUIREMENT_STATE_FIELDS
    ) and _requirement_state_is_expected(requirement_state)
    exe_source_valid = _plain_dict_section_has_exact_fields(
        source, "exe_direction_entry_contract", _EXPECTED_SOURCE_EXE_FIELDS
    ) and _exe_state_is_expected(exe_state)
    accepted = _build_read_model_source_acceptance(
        source=source,
        block_n_state=block_n_state,
        capability_state=capability_state,
        invariant_state=invariant_state,
        requirement_state=requirement_state,
        exe_state=exe_state,
        block_n_source_valid=block_n_source_valid,
        capability_source_valid=capability_source_valid,
        invariant_source_valid=invariant_source_valid,
        requirement_source_valid=requirement_source_valid,
        exe_source_valid=exe_source_valid,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_o_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_o_read_model_status": READ_MODEL_STATUS if accepted else BLOCKED_STATUS,
        "block_o_read_model_decision": READ_MODEL_DECISION if accepted else BLOCKED_STATUS.upper(),
        "block_o_read_model_ready": accepted,
        "ready_for_block_o_2": accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_o_entry_contract_reference": _build_reference(source, accepted),
        "read_model_summary": _build_summary(accepted),
        "block_n_closure_read_state": _with_read_fields(
            block_n_state,
            block_n_source_valid,
            "block_n_closure_read_preserved_execution_blocked",
            "block_n_closure_source_invalid_execution_blocked",
        ),
        "capability_read_state": _with_read_fields(
            capability_state,
            capability_source_valid,
            "capability_state_read_all_blocked_execution_unauthorized",
            "capability_source_invalid_execution_blocked",
            {"enabled_by_block_o_read_model": False},
        ),
        "invariant_read_state": _with_read_fields(
            invariant_state,
            invariant_source_valid,
            "invariants_read_all_preserved_execution_blocked",
            "invariant_source_invalid_execution_blocked",
        ),
        "requirement_read_state": _with_read_fields(
            requirement_state,
            requirement_source_valid,
            "requirements_read_all_missing_execution_blocked",
            "requirement_source_invalid_execution_blocked",
        ),
        "exe_direction_read_state": _build_exe_read_state(exe_state, source_valid=exe_source_valid),
        "fail_closed_read_decision": _build_fail_closed(source, accepted),
        "non_execution_read_evidence": _build_evidence(
            accepted=accepted,
            block_n_source_valid=block_n_source_valid,
            capability_source_valid=capability_source_valid,
            invariant_source_valid=invariant_source_valid,
            requirement_source_valid=requirement_source_valid,
            exe_source_valid=exe_source_valid,
        ),
        "read_model_boundaries": {key: True for key in _READ_MODEL_BOUNDARY_FIELDS},
        "source_boundaries": _build_source_boundaries(source, accepted),
        "future_steps": list(FUTURE_STEPS),
        "status": STATUS if accepted else BLOCKED_STATUS,
    }
    return {key: payload[key] for key in _TOP_LEVEL_FIELDS}
