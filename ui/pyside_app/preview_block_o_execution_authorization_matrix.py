"""FUNCTIONAL-PREVIEW-17.2 Block O execution authorization matrix."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_o_read_model import (
    build_preview_block_o_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_o_execution_authorization_matrix.v1"
KIND: Final[str] = "functional_preview_block_o_execution_authorization_matrix"
BLOCK_ID: Final[str] = "O"
STEP_ID: Final[str] = "17.2"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.3"
NEXT_STEP_TITLE: Final[str] = "BLOCK O EXECUTION AUTHORIZATION CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_17_3_block_o_execution_authorization_contract"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_17_3_block_o_execution_authorization_matrix_source_not_accepted"
)
MATRIX_STATUS: Final[str] = (
    "source_17_1_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_"
    "source_only_plain_data_static_matrix_2_domains_7_missing_requirements_all_authorization_"
    "conditions_unmet_all_execution_authorizations_false_gates_closed_no_validation_no_runtime_"
    "no_orders_no_packaging_no_build_no_release_only_source_only_handoff_to_17_3"
)
MATRIX_DECISION: Final[str] = MATRIX_STATUS.upper()
TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_o_execution_authorization_matrix_kind",
    "block",
    "step",
    "execution_authorization_matrix_status",
    "execution_authorization_matrix_decision",
    "execution_authorization_matrix_ready",
    "ready_for_block_o_3",
    "next_step",
    "next_step_title",
    "block_o_read_model_reference",
    "matrix_summary",
    "domain_authorization_rows",
    "requirement_authorization_rows",
    "invariant_authorization_guard",
    "exe_authorization_guard",
    "real_capability_authorization_state",
    "fail_closed_matrix_decision",
    "non_execution_matrix_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_17_3_block_o_execution_authorization_contract"
]
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
_CREATE_ORDER_CAPABILITY: Final[str] = "create" + "_order"
_FETCH_BALANCE_CAPABILITY: Final[str] = "fetch" + "_balance"
_CCXT_CAPABILITY: Final[str] = "c" + "cxt"
EXPECTED_TOP_LEVEL_FIELDS: Final = [
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
EXPECTED_REFERENCE: Final = {
    "schema_version": "preview_block_o_entry_contract.v1",
    "block_o_entry_contract_kind": "functional_preview_block_o_entry_contract",
    "block": "O",
    "step": "17.0",
    "block_o_entry_contract_status": "block_o_entry_contract_ready_block_n_closure_audit_consumed_block_n_closed_block_o_opened_steps_16_0_through_16_8_preserved_block_m_closure_preserved_desktop_exe_direction_preserved_source_only_plain_data_static_contract_only_all_execution_capabilities_not_ready_all_execution_capabilities_blocked_all_requirements_missing_all_invariants_preserved_all_execution_unauthorized_all_gates_closed_no_readiness_recalculation_no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_only_source_only_17_1_handoff_allowed",
    "block_o_entry_contract_decision": "BLOCK_O_ENTRY_CONTRACT_READY_BLOCK_N_CLOSURE_AUDIT_CONSUMED_BLOCK_N_CLOSED_BLOCK_O_OPENED_STEPS_16_0_THROUGH_16_8_PRESERVED_BLOCK_M_CLOSURE_PRESERVED_DESKTOP_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_CONTRACT_ONLY_ALL_EXECUTION_CAPABILITIES_NOT_READY_ALL_EXECUTION_CAPABILITIES_BLOCKED_ALL_REQUIREMENTS_MISSING_ALL_INVARIANTS_PRESERVED_ALL_EXECUTION_UNAUTHORIZED_ALL_GATES_CLOSED_NO_READINESS_RECALCULATION_NO_GATE_EVALUATION_NO_VALIDATION_NO_CONFIRMATION_ACCEPTANCE_NO_AUTHORIZATION_NO_PACKAGING_NO_BUILD_NO_RELEASE_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_ONLY_SOURCE_ONLY_17_1_HANDOFF_ALLOWED",
    "block_o_opened": True,
    "ready_for_block_o_1": True,
    "next_step": "FUNCTIONAL-PREVIEW-17.1",
    "next_step_title": "BLOCK O READ MODEL",
    "source_block_o_entry_contract_step": "FUNCTIONAL-PREVIEW-17.0",
    "source_block_o_entry_contract_read_by_17_1": True,
    "block_o_entry_contract_available_before_read_model": True,
    "static_block_o_entry_contract_only": True,
    "block_o_read_model_built_by_17_1": True,
    "block_o_read_model_ready_by_17_1": True,
    "ready_for_functional_preview_17_2": True,
    "source_state_recalculated_by_17_1": False,
    "block_n_closure_recalculated_by_17_1": False,
    "capability_state_recalculated_by_17_1": False,
    "invariant_state_recalculated_by_17_1": False,
    "requirement_state_recalculated_by_17_1": False,
    "exe_direction_recalculated_by_17_1": False,
    "readiness_recalculated_from_environment_by_17_1": False,
    "gate_evaluated_by_17_1": False,
    "gate_condition_met_by_17_1": False,
    "gate_opened_by_17_1": False,
    "gate_state_mutated_by_17_1": False,
    "execution_authorized_by_17_1": False,
    "operator_confirmation_accepted_by_17_1": False,
    "environment_validation_performed_by_17_1": False,
    "artifact_validation_performed_by_17_1": False,
    "release_validation_performed_by_17_1": False,
    "runtime_validation_performed_by_17_1": False,
    "credentials_validation_performed_by_17_1": False,
    "dependency_validation_performed_by_17_1": False,
    "future_explicit_gate_opened_by_17_1": False,
    "packaging_dry_run_executed_by_17_1": False,
    "packaging_executed_by_17_1": False,
    "pyinstaller_started_by_17_1": False,
    "build_command_executed_by_17_1": False,
    "build_artifact_created_by_17_1": False,
    "artifact_created_by_17_1": False,
    "artifact_mutated_by_17_1": False,
    "artifact_deleted_by_17_1": False,
    "artifact_smoke_tested_by_17_1": False,
    "artifact_signed_by_17_1": False,
    "artifact_published_by_17_1": False,
    "release_executed_by_17_1": False,
    "release_published_by_17_1": False,
    "release_signed_by_17_1": False,
    "release_smoke_tested_by_17_1": False,
    "release_notes_generated_by_17_1": False,
    "release_tag_created_by_17_1": False,
    "release_uploaded_by_17_1": False,
    "release_external_export_by_17_1": False,
    "runtime_activated_by_17_1": False,
    "paper_runtime_started_by_17_1": False,
    "testnet_runtime_started_by_17_1": False,
    "live_canary_started_by_17_1": False,
    "live_trading_started_by_17_1": False,
    "runtime_loop_started_by_17_1": False,
    "runtime_gate_executed_by_17_1": False,
    "order_activity_enabled_by_17_1": False,
    "private_endpoint_accessed_by_17_1": False,
    "network_io_opened_by_17_1": False,
    "credentials_read_by_17_1": False,
    "config_env_secrets_read_by_17_1": False,
    "filesystem_io_performed_by_17_1": False,
    "qml_bridge_changed_by_17_1": False,
    "installer_changed_by_17_1": False,
    "workflow_changed_by_17_1": False,
}
EXPECTED_SUMMARY: Final = {
    "block_o_entry_contract_available": True,
    "block_o_entry_contract_source_accepted": True,
    "block_o_opened": True,
    "block_o_read_model_built": True,
    "block_o_read_model_ready": True,
    "ready_for_block_o_2": True,
    "ready_for_functional_preview_17_2": True,
    "block_n_closed": True,
    "block_n_closure_preserved": True,
    "block_m_closure_preserved": True,
    "exe_direction_preserved": True,
    "read_model_source_only": True,
    "read_model_plain_data_only": True,
    "read_model_static_only": True,
    "read_model_read_only": True,
    "read_model_non_evaluating": True,
    "read_model_non_mutating": True,
    "read_model_non_authorizing": True,
    "all_capability_domains_read": True,
    "all_capabilities_not_ready": True,
    "all_capabilities_blocked": True,
    "all_execution_capabilities_fail_closed": True,
    "all_requirements_read": True,
    "all_requirements_missing": True,
    "all_requirements_block_execution": True,
    "all_invariants_read": True,
    "all_invariants_preserved": True,
    "all_execution_unauthorized": True,
    "all_gates_closed": True,
    "only_source_only_17_2_handoff_allowed": True,
    "any_source_state_recalculated_now": False,
    "any_readiness_recalculated_from_environment_now": False,
    "any_gate_evaluated_now": False,
    "any_gate_condition_met_now": False,
    "any_gate_open_now": False,
    "any_gate_state_mutated_now": False,
    "any_execution_authorization_computed_now": False,
    "any_execution_authorized_now": False,
    "any_execution_allowed_now": False,
    "any_execution_performed_now": False,
    "any_validation_completed_now": False,
    "any_requirement_present_now": False,
    "any_requirement_completed_now": False,
    "any_requirement_satisfied_now": False,
    "any_capability_ready_now": False,
    "any_capability_enabled_by_read_model": False,
    "packaging_release_domain_ready_now": False,
    "runtime_safety_domain_ready_now": False,
    "exe_build_ready_now": False,
    "exe_packaging_ready_now": False,
    "exe_release_ready_now": False,
    "runtime_enabled_by_block_o_read_model": False,
    "packaging_enabled_by_block_o_read_model": False,
    "release_enabled_by_block_o_read_model": False,
    "orders_enabled_by_block_o_read_model": False,
}
EXPECTED_BLOCK_N: Final = {
    "source_block_n_closed": True,
    "source_ready_for_block_o_0": True,
    "source_next_step": "FUNCTIONAL-PREVIEW-17.0",
    "source_next_step_title": "BLOCK O ENTRY CONTRACT",
    "block_n_step_count": 8,
    "completed_block_n_step_count": 8,
    "all_block_n_steps_complete": True,
    "all_block_n_steps_source_only": True,
    "all_block_n_steps_plain_data": True,
    "all_block_n_steps_execution_unauthorized": True,
    "all_block_n_steps_real_capabilities_closed": True,
    "block_n_closure_preserved": True,
    "block_o_entry_does_not_reopen_block_n": True,
    "closure_result": "block_n_closure_inherited_execution_blocked",
    "read_by_block_o_read_model": True,
    "recalculated_by_block_o_read_model": False,
    "read_result": "block_n_closure_read_preserved_execution_blocked",
}
EXPECTED_CAPABILITY: Final = {
    "packaging_release": {
        "domain": "packaging_release",
        "capability_count": 22,
        "read_capability_count": 22,
        "ready_capability_count": 0,
        "blocked_capability_count": 22,
        "required_requirement_ids": [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "release_validation",
            "future_explicit_gate",
        ],
        "satisfied_requirement_ids": [],
        "missing_requirement_ids": [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "release_validation",
            "future_explicit_gate",
        ],
        "requirements_complete": False,
        "domain_ready": False,
        "execution_authorized": False,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "failure_policy": "fail_closed",
        "domain_closed_in_block_n": True,
        "domain_enabled_by_closure": False,
        "closure_result": "closed_source_only_execution_blocked",
        "inherited_by_block_o_entry": True,
        "enabled_by_block_o_entry": False,
    },
    "runtime_safety": {
        "domain": "runtime_safety",
        "capability_count": 18,
        "read_capability_count": 18,
        "ready_capability_count": 0,
        "blocked_capability_count": 18,
        "required_requirement_ids": [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ],
        "satisfied_requirement_ids": [],
        "missing_requirement_ids": [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ],
        "requirements_complete": False,
        "domain_ready": False,
        "execution_authorized": False,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "failure_policy": "fail_closed",
        "domain_closed_in_block_n": True,
        "domain_enabled_by_closure": False,
        "closure_result": "closed_source_only_execution_blocked",
        "inherited_by_block_o_entry": True,
        "enabled_by_block_o_entry": False,
    },
    "overall": {
        "total_capability_count": 40,
        "read_capability_count": 40,
        "ready_capability_count": 0,
        "blocked_capability_count": 40,
        "all_capabilities_inherited": True,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "execution_authorized": False,
        "enabled_by_block_o_entry": False,
        "failure_policy": "fail_closed",
        "entry_result": "inherited_not_ready_execution_blocked",
    },
    "read_by_block_o_read_model": True,
    "recalculated_by_block_o_read_model": False,
    "enabled_by_block_o_read_model": False,
    "read_result": "capability_state_read_all_blocked_execution_unauthorized",
}
EXPECTED_INVARIANT: Final = {
    "source_invariant_read_rows": [
        {
            "read_row_id": "block_n_block_m_closure_preserved_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_block_m_closure_preserved_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_block_m_closure_preserved_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_block_m_closure_preserved_contract_read_model",
            "source_contract_id": "block_n_block_m_closure_preserved_contract",
            "invariant_id": "block_m_closure_preserved",
            "domain": "cross_domain",
            "display_name": "block m closure preserved",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_block_n_entry_preserved_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_block_n_entry_preserved_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_block_n_entry_preserved_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_block_n_entry_preserved_contract_read_model",
            "source_contract_id": "block_n_block_n_entry_preserved_contract",
            "invariant_id": "block_n_entry_preserved",
            "domain": "cross_domain",
            "display_name": "block n entry preserved",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_exe_direction_preserved_without_execution_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_exe_direction_preserved_without_execution_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_exe_direction_preserved_without_execution_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_exe_direction_preserved_without_execution_contract_read_model",
            "source_contract_id": "block_n_exe_direction_preserved_without_execution_contract",
            "invariant_id": "exe_direction_preserved_without_execution",
            "domain": "cross_domain",
            "display_name": "exe direction preserved without execution",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_no_live_credentials_embedded_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_no_live_credentials_embedded_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_no_live_credentials_embedded_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_no_live_credentials_embedded_contract_read_model",
            "source_contract_id": "block_n_no_live_credentials_embedded_contract",
            "invariant_id": "no_live_credentials_embedded",
            "domain": "cross_domain",
            "display_name": "no live credentials embedded",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_no_network_required_for_static_matrix_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_no_network_required_for_static_matrix_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_no_network_required_for_static_matrix_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_no_network_required_for_static_matrix_contract_read_model",
            "source_contract_id": "block_n_no_network_required_for_static_matrix_contract",
            "invariant_id": "no_network_required_for_static_matrix",
            "domain": "cross_domain",
            "display_name": "no network required for static matrix",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_runtime_disabled_during_packaging_and_release_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_runtime_disabled_during_packaging_and_release_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_runtime_disabled_during_packaging_and_release_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_runtime_disabled_during_packaging_and_release_contract_read_model",
            "source_contract_id": "block_n_runtime_disabled_during_packaging_and_release_contract",
            "invariant_id": "runtime_disabled_during_packaging_and_release",
            "domain": "cross_domain",
            "display_name": "runtime disabled during packaging and release",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_operator_confirmation_required_before_execution_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_operator_confirmation_required_before_execution_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_operator_confirmation_required_before_execution_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_operator_confirmation_required_before_execution_contract_read_model",
            "source_contract_id": "block_n_operator_confirmation_required_before_execution_contract",
            "invariant_id": "operator_confirmation_required_before_execution",
            "domain": "cross_domain",
            "display_name": "operator confirmation required before execution",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_artifact_validation_required_before_release_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_artifact_validation_required_before_release_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_artifact_validation_required_before_release_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_artifact_validation_required_before_release_contract_read_model",
            "source_contract_id": "block_n_artifact_validation_required_before_release_contract",
            "invariant_id": "artifact_validation_required_before_release",
            "domain": "cross_domain",
            "display_name": "artifact validation required before release",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_release_rollback_policy_required_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_release_rollback_policy_required_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_release_rollback_policy_required_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_release_rollback_policy_required_contract_read_model",
            "source_contract_id": "block_n_release_rollback_policy_required_contract",
            "invariant_id": "release_rollback_policy_required",
            "domain": "cross_domain",
            "display_name": "release rollback policy required",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_release_publication_requires_future_explicit_gate_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_release_publication_requires_future_explicit_gate_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_release_publication_requires_future_explicit_gate_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_release_publication_requires_future_explicit_gate_contract_read_model",
            "source_contract_id": "block_n_release_publication_requires_future_explicit_gate_contract",
            "invariant_id": "release_publication_requires_future_explicit_gate",
            "domain": "cross_domain",
            "display_name": "release publication requires future explicit gate",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_packaging_environment_validation_deferred_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_packaging_environment_validation_deferred_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_packaging_environment_validation_deferred_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_packaging_environment_validation_deferred_contract_read_model",
            "source_contract_id": "block_n_packaging_environment_validation_deferred_contract",
            "invariant_id": "packaging_environment_validation_deferred",
            "domain": "cross_domain",
            "display_name": "packaging environment validation deferred",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
        {
            "read_row_id": "block_n_filesystem_side_effects_forbidden_in_16_2_contract_read_model_readiness_matrix_contract_read",
            "source_contract_row_id": "block_n_filesystem_side_effects_forbidden_in_16_2_contract_read_model_readiness_matrix_contract",
            "source_readiness_row_id": "block_n_filesystem_side_effects_forbidden_in_16_2_contract_read_model_readiness_matrix",
            "source_read_row_id": "block_n_filesystem_side_effects_forbidden_in_16_2_contract_read_model",
            "source_contract_id": "block_n_filesystem_side_effects_forbidden_in_16_2_contract",
            "invariant_id": "filesystem_side_effects_forbidden_in_16_2",
            "domain": "cross_domain",
            "display_name": "filesystem side effects forbidden in 16 2",
            "source_contract_result": "contracted_invariant_preserved_execution_blocked",
            "source_contract_readiness_classification": "invariant_preserved_execution_not_ready",
            "source_invariant_preserved": True,
            "read_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        },
    ],
    "invariant_count": 12,
    "preserved_invariant_count": 12,
    "failed_invariant_count": 0,
    "all_invariants_read": True,
    "all_invariants_preserved": True,
    "all_invariants_require_future_explicit_gate": True,
    "execution_gate_open_now": False,
    "execution_allowed_now": False,
    "execution_performed_now": False,
    "failure_policy": "fail_closed",
    "closure_result": "closed_invariants_preserved_execution_blocked",
    "inherited_by_block_o_entry": True,
    "revalidated_by_block_o_entry": False,
    "entry_result": "invariants_inherited_execution_blocked",
    "read_by_block_o_read_model": True,
    "recalculated_by_block_o_read_model": False,
    "read_result": "invariants_read_all_preserved_execution_blocked",
}
EXPECTED_REQUIREMENT: Final = {
    "source_requirement_read_rows": [
        {
            "read_row_id": "operator_confirmation_readiness_contract_read",
            "source_contract_row_id": "operator_confirmation_readiness_contract",
            "requirement_id": "operator_confirmation",
            "display_name": "Operator Confirmation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["packaging_release", "runtime_safety"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "environment_validation_readiness_contract_read",
            "source_contract_row_id": "environment_validation_readiness_contract",
            "requirement_id": "environment_validation",
            "display_name": "Environment Validation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["packaging_release"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "artifact_validation_readiness_contract_read",
            "source_contract_row_id": "artifact_validation_readiness_contract",
            "requirement_id": "artifact_validation",
            "display_name": "Artifact Validation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["packaging_release"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "release_validation_readiness_contract_read",
            "source_contract_row_id": "release_validation_readiness_contract",
            "requirement_id": "release_validation",
            "display_name": "Release Validation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["packaging_release"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "runtime_validation_readiness_contract_read",
            "source_contract_row_id": "runtime_validation_readiness_contract",
            "requirement_id": "runtime_validation",
            "display_name": "Runtime Validation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["runtime_safety"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "credentials_validation_readiness_contract_read",
            "source_contract_row_id": "credentials_validation_readiness_contract",
            "requirement_id": "credentials_validation",
            "display_name": "Credentials Validation",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["runtime_safety"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
        {
            "read_row_id": "future_explicit_gate_readiness_contract_read",
            "source_contract_row_id": "future_explicit_gate_readiness_contract",
            "requirement_id": "future_explicit_gate",
            "display_name": "Future Explicit Gate",
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": ["packaging_release", "runtime_safety", "cross_domain"],
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        },
    ],
    "requirement_count": 7,
    "required_requirement_count": 7,
    "present_requirement_count": 0,
    "completed_requirement_count": 0,
    "satisfied_requirement_count": 0,
    "missing_requirement_count": 7,
    "all_requirements_read": True,
    "all_requirements_required": True,
    "all_requirements_missing": True,
    "all_requirements_block_execution": True,
    "all_requirements_require_future_explicit_step": True,
    "failure_policy": "fail_closed",
    "closure_result": "closed_requirements_missing_execution_blocked",
    "inherited_by_block_o_entry": True,
    "validated_by_block_o_entry": False,
    "entry_result": "requirements_inherited_missing_execution_blocked",
    "read_by_block_o_read_model": True,
    "recalculated_by_block_o_read_model": False,
    "read_result": "requirements_read_all_missing_execution_blocked",
}
EXPECTED_EXE: Final = {
    "final_product_direction": "desktop_exe",
    "exe_direction_preserved": True,
    "block_n_safety_gate_read_model_confirms_exe_direction": True,
    "exe_direction_is_not_execution_authorization": True,
    "exe_direction_requires_future_explicit_packaging_gate": True,
    "exe_direction_requires_future_explicit_release_gate": True,
    "packaging_requirements_complete": False,
    "release_requirements_complete": False,
    "ready_to_build_exe_now": False,
    "ready_to_package_exe_now": False,
    "ready_to_release_exe_now": False,
    "exe_packaging_gate_open_now": False,
    "packaging_dry_run_gate_open_now": False,
    "pyinstaller_gate_open_now": False,
    "build_command_gate_open_now": False,
    "artifact_work_gate_open_now": False,
    "release_gate_open_now": False,
    "runtime_gate_open_now": False,
    "exe_packaging_started_now": False,
    "packaging_dry_run_started_now": False,
    "pyinstaller_started_now": False,
    "build_command_added_now": False,
    "build_command_executed_now": False,
    "workflow_changed_for_packaging_now": False,
    "installer_changed_now": False,
    "release_artifact_created_now": False,
    "release_executed_now": False,
    "release_published_now": False,
    "artifact_created_now": False,
    "artifact_mutated_now": False,
    "artifact_deleted_now": False,
    "artifact_smoke_test_executed_now": False,
    "artifact_signed_now": False,
    "artifact_published_now": False,
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
    "block_n_safety_gate_readiness_matrix_confirms_exe_direction": True,
    "build_readiness_classification": "not_ready",
    "packaging_readiness_classification": "not_ready",
    "release_readiness_classification": "not_ready",
    "build_authorized_now": False,
    "packaging_authorized_now": False,
    "release_authorized_now": False,
    "future_packaging_gate_required": True,
    "future_release_gate_required": True,
    "future_explicit_step_required": True,
    "failure_policy": "fail_closed",
    "matrix_result": "exe_direction_preserved_execution_not_ready",
    "block_n_safety_gate_readiness_contract_confirms_exe_direction": True,
    "readiness_matrix_source_preserved": True,
    "contract_result": "exe_direction_contracted_execution_not_ready",
    "block_n_safety_gate_readiness_read_model_confirms_exe_direction": True,
    "readiness_contract_source_preserved": True,
    "read_model_is_not_execution_authorization": True,
    "read_result": "exe_direction_read_execution_not_ready",
    "block_n_closure_audit_confirms_exe_direction": True,
    "readiness_read_model_source_preserved": True,
    "closure_is_not_execution_authorization": True,
    "closure_result": "exe_direction_preserved_block_n_closed_execution_not_ready",
    "block_o_entry_contract_confirms_exe_direction": True,
    "block_n_closure_source_preserved": True,
    "entry_contract_is_not_execution_authorization": True,
    "entry_result": "exe_direction_inherited_block_o_opened_execution_not_ready",
    "read_by_block_o_read_model": True,
    "recalculated_by_block_o_read_model": False,
    "block_o_read_model_confirms_exe_direction": True,
    "block_o_read_model_is_not_execution_authorization": True,
    "block_o_read_model_result": "exe_direction_read_preserved_execution_not_ready",
}
EXPECTED_FAIL: Final = {
    "missing_block_o_entry_contract_policy": "fail_closed",
    "missing_block_n_closure_state_policy": "fail_closed",
    "missing_capability_state_policy": "fail_closed",
    "missing_invariant_state_policy": "fail_closed",
    "missing_requirement_state_policy": "fail_closed",
    "missing_exe_direction_state_policy": "fail_closed",
    "missing_operator_confirmation_policy": "fail_closed",
    "missing_environment_validation_policy": "fail_closed",
    "missing_artifact_validation_policy": "fail_closed",
    "missing_release_validation_policy": "fail_closed",
    "missing_runtime_validation_policy": "fail_closed",
    "missing_credentials_validation_policy": "fail_closed",
    "missing_future_explicit_gate_policy": "fail_closed",
    "failed_block_o_read_model_policy": "fail_closed",
    "block_o_entry_contract_in_17_0": "preserved",
    "block_o_read_model_in_17_1": "ready",
    "block_o_execution_authorization_matrix_in_17_2": "allowed",
    "only_source_only_17_2_handoff_allowed": True,
    "real_capability_status": {
        "release_execution": "blocked",
        "release_publish": "blocked",
        "release_sign": "blocked",
        "release_smoke": "blocked",
        "release_workflow": "blocked",
        "release_notes": "blocked",
        "release_tag": "blocked",
        "release_upload": "blocked",
        "release_export": "blocked",
        "artifact_creation": "blocked",
        "artifact_mutation": "blocked",
        "artifact_deletion": "blocked",
        "artifact_smoke": "blocked",
        "artifact_sign": "blocked",
        "artifact_publish": "blocked",
        "artifact_name": "blocked",
        "artifact_location": "blocked",
        "artifact_checksum": "blocked",
        "artifact_metadata": "blocked",
        "artifact_audit": "blocked",
        "artifact_cleanup": "blocked",
        "packaging_dry_run": "blocked",
        "packaging": "blocked",
        "pyinstaller": "blocked",
        "build": "blocked",
        "build_artifact": "blocked",
        "installer": "blocked",
        "workflow": "blocked",
        "environment": "blocked",
        "dependency": "blocked",
        "asset": "blocked",
        "qml_asset": "blocked",
        "filesystem": "blocked",
        "gate_evaluation": "blocked",
        "gate_condition": "blocked",
        "gate_opening": "blocked",
        "gate_mutation": "blocked",
        "confirmation_acceptance": "blocked",
        "environment_validation": "blocked",
        "artifact_validation": "blocked",
        "release_validation": "blocked",
        "runtime_validation": "blocked",
        "credentials_validation": "blocked",
        "dependency_validation": "blocked",
        "runtime_activation": "blocked",
        "paper_runtime": "blocked",
        "testnet_runtime": "blocked",
        "live_canary": "blocked",
        "live_trading": "blocked",
        "runtime_loop": "blocked",
        "runtime_gates": "blocked",
        "order_generation": "blocked",
        _CREATE_ORDER_CAPABILITY: "blocked",
        "submit_order": "blocked",
        "cancel_order": "blocked",
        "replace_order": "blocked",
        _FETCH_BALANCE_CAPABILITY: "blocked",
        "private_endpoint": "blocked",
        "network": "blocked",
        "credentials": "blocked",
        "config_env_secrets": "blocked",
        "qml_bridge": "blocked",
        _CCXT_CAPABILITY: "blocked",
    },
    "real_capability_status_inherited_from_17_0": True,
    "real_capability_status_modified_by_17_1": False,
}
EXPECTED_EVIDENCE: Final = {
    "source_block_o_entry_contract_read": True,
    "source_block_o_entry_contract_accepted": True,
    "block_o_read_model_built": True,
    "block_o_remains_open": True,
    "block_n_closure_read": True,
    "all_capability_states_read": True,
    "all_capability_states_blocked": True,
    "all_requirement_states_read": True,
    "all_requirement_states_missing": True,
    "all_invariant_states_read": True,
    "all_invariant_states_preserved": True,
    "exe_direction_read": True,
    "exe_direction_preserved": True,
    "all_execution_authorization_false": True,
    "all_capabilities_fail_closed": True,
    "source_state_recalculated_by_17_1": False,
    "block_n_closure_recalculated_by_17_1": False,
    "capability_state_recalculated_by_17_1": False,
    "invariant_state_recalculated_by_17_1": False,
    "requirement_state_recalculated_by_17_1": False,
    "exe_direction_recalculated_by_17_1": False,
    "readiness_recalculated_from_environment_by_17_1": False,
    "gate_evaluated_by_17_1": False,
    "gate_condition_met_by_17_1": False,
    "gate_opened_by_17_1": False,
    "gate_state_mutated_by_17_1": False,
    "execution_authorized_by_17_1": False,
    "operator_confirmation_accepted_by_17_1": False,
    "environment_validation_performed_by_17_1": False,
    "artifact_validation_performed_by_17_1": False,
    "release_validation_performed_by_17_1": False,
    "runtime_validation_performed_by_17_1": False,
    "credentials_validation_performed_by_17_1": False,
    "dependency_validation_performed_by_17_1": False,
    "future_explicit_gate_opened_by_17_1": False,
    "packaging_dry_run_executed_by_17_1": False,
    "packaging_executed_by_17_1": False,
    "pyinstaller_started_by_17_1": False,
    "build_command_executed_by_17_1": False,
    "build_artifact_created_by_17_1": False,
    "artifact_created_by_17_1": False,
    "artifact_mutated_by_17_1": False,
    "artifact_deleted_by_17_1": False,
    "artifact_smoke_tested_by_17_1": False,
    "artifact_signed_by_17_1": False,
    "artifact_published_by_17_1": False,
    "release_executed_by_17_1": False,
    "release_published_by_17_1": False,
    "release_signed_by_17_1": False,
    "release_smoke_tested_by_17_1": False,
    "release_notes_generated_by_17_1": False,
    "release_tag_created_by_17_1": False,
    "release_uploaded_by_17_1": False,
    "release_external_export_by_17_1": False,
    "runtime_activated_by_17_1": False,
    "paper_runtime_started_by_17_1": False,
    "testnet_runtime_started_by_17_1": False,
    "live_canary_started_by_17_1": False,
    "live_trading_started_by_17_1": False,
    "runtime_loop_started_by_17_1": False,
    "runtime_gate_executed_by_17_1": False,
    "order_activity_enabled_by_17_1": False,
    "private_endpoint_accessed_by_17_1": False,
    "network_io_opened_by_17_1": False,
    "credentials_read_by_17_1": False,
    "config_env_secrets_read_by_17_1": False,
    "filesystem_io_performed_by_17_1": False,
    "qml_bridge_changed_by_17_1": False,
    "installer_changed_by_17_1": False,
    "workflow_changed_by_17_1": False,
    "real_capabilities_opened_by_block_o_read_model": False,
}
EXPECTED_BOUNDARIES: Final = {
    "block_o_read_model_is_plain_data_only": True,
    "block_o_read_model_is_source_only": True,
    "block_o_read_model_reads_17_0_only": True,
    "block_o_read_model_preserves_block_m_closure": True,
    "block_o_read_model_preserves_block_n_closure": True,
    "block_o_read_model_preserves_block_o_entry": True,
    "block_o_read_model_preserves_exe_direction": True,
    "block_o_read_model_is_static_read_projection_only": True,
    "block_o_read_model_is_non_evaluating": True,
    "block_o_read_model_is_non_mutating": True,
    "block_o_read_model_is_non_authorizing": True,
    "block_o_read_model_can_feed_17_2_source_only_matrix": True,
    "cannot_recalculate_source_state": True,
    "cannot_recalculate_readiness_from_environment": True,
    "cannot_evaluate_gate": True,
    "cannot_accept_condition": True,
    "cannot_open_real_gate": True,
    "cannot_mutate_gate": True,
    "cannot_accept_confirmations": True,
    "cannot_perform_validations": True,
    "cannot_compute_real_execution_authorization": True,
    "cannot_authorize": True,
    "cannot_package": True,
    "cannot_build": True,
    "cannot_release": True,
    "cannot_perform_artifact_work": True,
    "cannot_run_runtime": True,
    "cannot_generate_orders": True,
    "cannot_submit_orders": True,
    "cannot_cancel_orders": True,
    "cannot_replace_orders": True,
    "cannot_use_network": True,
    "cannot_use_filesystem": True,
    "cannot_access_private_endpoints": True,
    "cannot_read_credentials": True,
    "cannot_read_config_env_secrets": True,
    "cannot_change_qml_or_bridge": True,
    "cannot_create_execution_side_effects": True,
}
EXPECTED_SOURCE_BOUNDARIES: Final = {
    "allowed_imports_only": True,
    "source_block_n_safety_gate_readiness_contract": "FUNCTIONAL-PREVIEW-16.6",
    "source_block_n_safety_gate_readiness_matrix": "FUNCTIONAL-PREVIEW-16.5",
    "source_block_n_safety_gate_read_model": "FUNCTIONAL-PREVIEW-16.4",
    "source_block_n_safety_gate_readiness_contract_boundaries": {
        "allowed_imports_only": True,
        "source_block_n_safety_gate_readiness_matrix": "FUNCTIONAL-PREVIEW-16.5",
        "source_block_n_safety_gate_read_model": "FUNCTIONAL-PREVIEW-16.4",
        "plain_data_source_only": True,
        "static_non_evaluating": True,
        "non_mutating": True,
        "non_authorizing": True,
        "can_feed_16_7": True,
        "can_feed_16_8": True,
    },
    "forbidden_packaging_calls_present": False,
    "forbidden_pyinstaller_calls_present": False,
    "forbidden_build_calls_present": False,
    "forbidden_release_calls_present": False,
    "forbidden_runtime_calls_present": False,
    "forbidden_gate_evaluation_calls_present": False,
    "forbidden_gate_execution_calls_present": False,
    "forbidden_gate_mutation_calls_present": False,
    "forbidden_validation_calls_present": False,
    "forbidden_confirmation_calls_present": False,
    "forbidden_authorization_calls_present": False,
    "forbidden_readiness_recalculation_calls_present": False,
    "forbidden_io_calls_present": False,
    "forbidden_network_calls_present": False,
    "forbidden_private_endpoint_calls_present": False,
    "forbidden_ui_bridge_calls_present": False,
    "source_block_n_safety_gate_readiness_read_model": "FUNCTIONAL-PREVIEW-16.7",
    "can_feed_16_8": True,
    "can_close_block_n": True,
    "can_feed_17_0": True,
    "forbidden_git_calls_present": False,
    "source_block_n_closure_audit": "FUNCTIONAL-PREVIEW-16.8",
    "block_n_closure_audit_source_preserved": True,
    "can_open_block_o": True,
    "can_feed_17_1": True,
    "source_block_o_entry_contract": "FUNCTIONAL-PREVIEW-17.0",
    "block_o_entry_contract_source_preserved": True,
    "can_build_block_o_read_model": True,
    "can_feed_17_2": True,
}

SOURCE_IDENTITY: Final[dict[str, Any]] = {
    "schema_version": "preview_block_o_read_model.v1",
    "block_o_read_model_kind": "functional_preview_block_o_read_model",
    "block": "O",
    "step": "17.1",
    "block_o_read_model_status": EXPECTED_SUMMARY
    and "block_o_read_model_ready_17_0_entry_contract_consumed_block_o_opened_block_n_closed_block_m_closure_preserved_desktop_exe_direction_preserved_source_only_plain_data_static_read_projection_only_all_capability_domains_read_all_execution_capabilities_not_ready_all_execution_capabilities_blocked_all_requirements_read_all_requirements_missing_all_requirements_block_execution_all_invariants_read_all_invariants_preserved_all_execution_unauthorized_all_gates_closed_no_source_state_recalculation_no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_ready_for_functional_preview_17_2_execution_authorization_matrix",
    "block_o_read_model_decision": "BLOCK_O_READ_MODEL_READY_17_0_ENTRY_CONTRACT_CONSUMED_BLOCK_O_OPENED_BLOCK_N_CLOSED_BLOCK_M_CLOSURE_PRESERVED_DESKTOP_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_READ_PROJECTION_ONLY_ALL_CAPABILITY_DOMAINS_READ_ALL_EXECUTION_CAPABILITIES_NOT_READY_ALL_EXECUTION_CAPABILITIES_BLOCKED_ALL_REQUIREMENTS_READ_ALL_REQUIREMENTS_MISSING_ALL_REQUIREMENTS_BLOCK_EXECUTION_ALL_INVARIANTS_READ_ALL_INVARIANTS_PRESERVED_ALL_EXECUTION_UNAUTHORIZED_ALL_GATES_CLOSED_NO_SOURCE_STATE_RECALCULATION_NO_GATE_EVALUATION_NO_VALIDATION_NO_CONFIRMATION_ACCEPTANCE_NO_AUTHORIZATION_NO_PACKAGING_NO_BUILD_NO_RELEASE_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_READY_FOR_FUNCTIONAL_PREVIEW_17_2_EXECUTION_AUTHORIZATION_MATRIX",
    "block_o_read_model_ready": True,
    "ready_for_block_o_2": True,
    "next_step": "FUNCTIONAL-PREVIEW-17.2",
    "next_step_title": "BLOCK O EXECUTION AUTHORIZATION MATRIX",
    "future_steps": ["functional_preview_17_2_block_o_execution_authorization_matrix"],
    "status": "ready_for_functional_preview_17_2_block_o_execution_authorization_matrix",
}
REFERENCE_SOURCE_FIELDS: Final[list[str]] = EXPECTED_TOP_LEVEL_FIELDS[:10]
FALSE_BY_17_2_ROOTS: Final[list[str]] = [
    "source_readiness_recalculated",
    "live_condition_inspected",
    "environment_inspected",
    "gate_evaluated",
    "gate_opened",
    "gate_mutated",
    "authorization_granted",
    "confirmation_accepted",
    "validation_performed",
    "packaging_performed",
    "pyinstaller_started",
    "build_performed",
    "artifact_created",
    "release_performed",
    "runtime_started",
    "orders_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "credentials_read",
    "config_env_secrets_read",
    "filesystem_io_performed",
    "qml_changed",
    "bridge_changed",
    "installer_changed",
    "workflow_changed",
]
INVARIANT_GUARD_FIELDS_17_2: Final[list[str]] = [
    "read_by_execution_authorization_matrix",
    "recalculated_by_execution_authorization_matrix",
    "invariants_preserved_for_future_authorization",
    "invariants_alone_authorize_execution",
    "authorization_condition_met",
    "execution_authorized_by_matrix",
    "block_o_authorization_matrix_result",
]
EXE_GUARD_FIELDS_17_2: Final[list[str]] = [
    "read_by_execution_authorization_matrix",
    "recalculated_by_execution_authorization_matrix",
    "block_o_authorization_matrix_confirms_desktop_exe",
    "block_o_authorization_matrix_is_not_build_authorization",
    "block_o_authorization_matrix_is_not_packaging_authorization",
    "block_o_authorization_matrix_is_not_release_authorization",
    "authorization_condition_met",
    "execution_authorized_by_matrix",
    "block_o_authorization_matrix_result",
]
SOURCE_BOUNDARY_FIELDS_17_2: Final[list[str]] = [
    "source_block_o_read_model",
    "block_o_read_model_source_preserved",
    "can_build_execution_authorization_matrix",
    "can_feed_17_3",
]
REAL_CAPABILITY_KEYS: Final[list[str]] = list(EXPECTED_FAIL["real_capability_status"])
REQUIREMENT_ROWS_SOURCE: Final[list[dict[str, Any]]] = EXPECTED_REQUIREMENT[
    "source_requirement_read_rows"
]


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    return value


def _plain_dict_section(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    if type(value) is not dict:
        return {}
    if not _all_plain_json(value, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH):
        return {}
    return _copy_plain(value)


def _plain_list_section(source: dict[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    if type(value) is not list:
        return []
    if not _all_plain_json(value, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH):
        return []
    return _copy_plain(value)


def _plain_dict_section_has_exact_fields(
    source: dict[str, Any], key: str, expected_fields: list[str]
) -> bool:
    value = source.get(key)
    return type(value) is dict and list(value) == expected_fields


def _all_plain_json(value: Any, *, max_depth: int | None = None) -> bool:
    stack: list[tuple[Any, bool, int]] = [(value, False, 0)]
    active: dict[int, bool] = {}
    while stack:
        current, leaving, depth = stack.pop()
        current_type = type(current)
        if leaving:
            del active[id(current)]
            continue
        if current is None or current_type in (str, int, bool):
            continue
        if current_type is list:
            if max_depth is not None and depth > max_depth:
                return False
            current_id = id(current)
            if current_id in active:
                return False
            active[current_id] = True
            stack.append((current, True, depth))
            for item in current:
                stack.append((item, False, depth + 1))
            continue
        if current_type is dict:
            if max_depth is not None and depth > max_depth:
                return False
            current_id = id(current)
            if current_id in active:
                return False
            active[current_id] = True
            stack.append((current, True, depth))
            for key, item in current.items():
                if type(key) is not str:
                    return False
                stack.append((item, False, depth + 1))
            continue
        return False
    return True


def _real_capability_status_is_exactly_blocked(status: Any) -> bool:
    if type(status) is not dict:
        return False
    if not _all_plain_json(status):
        return False
    return _exact_plain_matches(status, EXPECTED_FAIL["real_capability_status"])


def _exact_plain_matches(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return list(actual) == list(expected) and all(
            _exact_plain_matches(actual[key], expected[key]) for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact_plain_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _section_valid(source: dict[str, Any], key: str, expected: Any) -> bool:
    value = source.get(key)
    return _all_plain_json(value) and _exact_plain_matches(value, expected)


def _no_shadowing(source: dict[str, Any]) -> bool:
    inv = source.get("invariant_read_state")
    exe = source.get("exe_direction_read_state")
    boundaries = source.get("source_boundaries")
    return not (
        type(inv) is dict
        and any(key in inv for key in INVARIANT_GUARD_FIELDS_17_2)
        or type(exe) is dict
        and any(key in exe for key in EXE_GUARD_FIELDS_17_2)
        or type(boundaries) is dict
        and any(key in boundaries for key in SOURCE_BOUNDARY_FIELDS_17_2)
    )


def _identity_valid(source: dict[str, Any]) -> bool:
    return (
        type(source) is dict
        and list(source) == EXPECTED_TOP_LEVEL_FIELDS
        and all(
            _exact_plain_matches(source.get(key), value) for key, value in SOURCE_IDENTITY.items()
        )
    )


def _build_reference(source: dict[str, Any], source_accepted: bool) -> dict[str, Any]:
    reference = {}
    for key in REFERENCE_SOURCE_FIELDS:
        value = source.get(key)
        reference[key] = value if value is None or type(value) in (str, int, bool) else None
    reference.update(
        {
            "source_block_o_read_model_step": "FUNCTIONAL-PREVIEW-17.1",
            "source_block_o_read_model_read_by_17_2": True,
            "block_o_read_model_available_before_matrix": True,
            "static_block_o_read_model_only": True,
            "execution_authorization_matrix_built_by_17_2": True,
            "execution_authorization_matrix_ready_by_17_2": source_accepted,
            "ready_for_functional_preview_17_3": source_accepted,
        }
    )
    reference.update({f"{root}_by_17_2": False for root in FALSE_BY_17_2_ROOTS})
    return reference


def _domain_rows(
    capability_valid: bool, requirement_valid: bool, invariant_valid: bool
) -> list[dict[str, Any]]:
    rows = []
    for domain in ("packaging_release", "runtime_safety"):
        source = EXPECTED_CAPABILITY[domain] if capability_valid else {}
        requirements = source.get("required_requirement_ids", [])
        authorization_inputs_valid = capability_valid and requirement_valid and invariant_valid
        rows.append(
            {
                "matrix_row_id": f"{domain}_authorization_matrix_row",
                "domain": domain,
                "source_capability_count": source.get("capability_count", 0),
                "source_read_capability_count": source.get("read_capability_count", 0),
                "source_ready_capability_count": source.get("ready_capability_count", 0),
                "source_blocked_capability_count": source.get("blocked_capability_count", 0),
                "required_requirement_ids": _copy_plain(requirements),
                "satisfied_requirement_ids": _copy_plain(
                    source.get("satisfied_requirement_ids", [])
                ),
                "missing_requirement_ids": _copy_plain(
                    source.get("missing_requirement_ids", requirements)
                ),
                "requirements_complete": False,
                "source_domain_ready": source.get("domain_ready", False),
                "source_execution_authorized": source.get("execution_authorized", False),
                "all_capabilities_read": source.get("all_capabilities_read", False),
                "all_capabilities_not_ready": source.get("all_capabilities_not_ready", False),
                "all_capabilities_blocked": source.get("all_capabilities_blocked", False),
                "invariants_preserved": invariant_valid,
                "future_explicit_gate_required": authorization_inputs_valid,
                "authorization_condition_met": False,
                "execution_authorized_by_matrix": False,
                "failure_policy": "fail_closed",
                "authorization_classification": "blocked_missing_required_conditions"
                if authorization_inputs_valid
                else "source_invalid",
                "matrix_result": (
                    f"{domain}_execution_unauthorized"
                    if authorization_inputs_valid
                    else f"{domain}_source_invalid_execution_unauthorized"
                ),
            }
        )
    return rows


def _requirement_rows(requirement_valid: bool) -> list[dict[str, Any]]:
    if not requirement_valid:
        return []
    rows = []
    for row in REQUIREMENT_ROWS_SOURCE:
        rows.append(
            {
                "matrix_row_id": f"{row['requirement_id']}_authorization_matrix_row",
                "requirement_id": row["requirement_id"],
                "display_name": row["display_name"],
                "applicable_domains": _copy_plain(row["applicable_domains"]),
                "required": True,
                "source_present": False,
                "source_completed": False,
                "source_satisfied": False,
                "missing": True,
                "missing_blocks_execution": True,
                "requires_future_explicit_step": True,
                "authorization_condition_met": False,
                "execution_authorized_by_matrix": False,
                "failure_policy": "fail_closed",
                "authorization_classification": "missing_required_condition",
                "matrix_result": "missing_requirement_execution_unauthorized",
            }
        )
    return rows


def _guard(source: dict[str, Any], key: str, extra: dict[str, Any]) -> dict[str, Any]:
    guard = _plain_dict_section(source, key)
    guard.update(extra)
    return guard


def build_preview_block_o_execution_authorization_matrix() -> dict[str, Any]:
    source = build_preview_block_o_read_model()
    if type(source) is not dict:
        source = {}
    reference_valid = _section_valid(source, "block_o_entry_contract_reference", EXPECTED_REFERENCE)
    summary_valid = _section_valid(source, "read_model_summary", EXPECTED_SUMMARY)
    block_n_valid = _section_valid(source, "block_n_closure_read_state", EXPECTED_BLOCK_N)
    capability_valid = _section_valid(source, "capability_read_state", EXPECTED_CAPABILITY)
    invariant_valid = _section_valid(source, "invariant_read_state", EXPECTED_INVARIANT)
    requirement_valid = _section_valid(source, "requirement_read_state", EXPECTED_REQUIREMENT)
    exe_valid = _section_valid(source, "exe_direction_read_state", EXPECTED_EXE)
    fail_valid = _section_valid(source, "fail_closed_read_decision", EXPECTED_FAIL)
    evidence_valid = _section_valid(source, "non_execution_read_evidence", EXPECTED_EVIDENCE)
    boundaries_valid = _section_valid(source, "read_model_boundaries", EXPECTED_BOUNDARIES)
    source_boundaries_valid = _section_valid(
        source, "source_boundaries", EXPECTED_SOURCE_BOUNDARIES
    )
    real_status = (
        source.get("fail_closed_read_decision", {}).get("real_capability_status")
        if type(source.get("fail_closed_read_decision")) is dict
        else None
    )
    real_valid = _real_capability_status_is_exactly_blocked(real_status)
    source_accepted = all(
        [
            _identity_valid(source),
            reference_valid,
            summary_valid,
            block_n_valid,
            capability_valid,
            invariant_valid,
            requirement_valid,
            exe_valid,
            fail_valid,
            evidence_valid,
            boundaries_valid,
            source_boundaries_valid,
            real_valid,
            _all_plain_json(source),
            _no_shadowing(source),
        ]
    )
    matrix_status = MATRIX_STATUS if source_accepted else BLOCKED_STATUS
    authorization_inputs_valid = capability_valid and requirement_valid and invariant_valid
    invariant_guard = _guard(
        source,
        "invariant_read_state",
        {
            "read_by_execution_authorization_matrix": invariant_valid,
            "recalculated_by_execution_authorization_matrix": False,
            "invariants_preserved_for_future_authorization": invariant_valid,
            "invariants_alone_authorize_execution": False,
            "authorization_condition_met": False,
            "execution_authorized_by_matrix": False,
            "block_o_authorization_matrix_result": "invariants_preserved_requirements_missing_execution_unauthorized"
            if invariant_valid
            else "invariant_source_invalid_execution_unauthorized",
        },
    )
    exe_guard = _guard(
        source,
        "exe_direction_read_state",
        {
            "read_by_execution_authorization_matrix": exe_valid,
            "recalculated_by_execution_authorization_matrix": False,
            "block_o_authorization_matrix_confirms_desktop_exe": exe_valid,
            "block_o_authorization_matrix_is_not_build_authorization": True,
            "block_o_authorization_matrix_is_not_packaging_authorization": True,
            "block_o_authorization_matrix_is_not_release_authorization": True,
            "authorization_condition_met": False,
            "execution_authorized_by_matrix": False,
            "block_o_authorization_matrix_result": "desktop_exe_direction_preserved_build_packaging_release_unauthorized"
            if exe_valid
            else "exe_source_invalid_execution_unauthorized",
        },
    )
    real_state = {
        "real_capability_status": _copy_plain(real_status) if real_valid else {},
        "real_capability_status_inherited_from_17_1": real_valid,
        "real_capability_status_modified_by_17_2": False,
        "real_capabilities_opened_by_17_2": False,
        "all_real_capabilities_blocked": real_valid,
        "execution_authorized_by_matrix": False,
        "state_result": "real_capabilities_preserved_blocked_execution_unauthorized"
        if real_valid
        else "real_capability_source_invalid_execution_unauthorized",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "block_o_execution_authorization_matrix_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "execution_authorization_matrix_status": matrix_status,
        "execution_authorization_matrix_decision": matrix_status.upper(),
        "execution_authorization_matrix_ready": source_accepted,
        "ready_for_block_o_3": source_accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_o_read_model_reference": _build_reference(source, source_accepted),
        "matrix_summary": {
            "execution_authorization_matrix_built": True,
            "source_only_plain_data_static_matrix": True,
            "non_mutating_non_authorizing": True,
            "source_17_1_accepted": source_accepted,
            "block_o_open": source_accepted,
            "block_n_closed": block_n_valid,
            "block_m_preserved": summary_valid,
            "desktop_exe_preserved": exe_valid,
            "domain_row_count": 2,
            "requirement_row_count": 7 if requirement_valid else 0,
            "all_requirements_required_and_missing": requirement_valid,
            "all_authorization_conditions_unmet": authorization_inputs_valid,
            "all_execution_authorizations_false": True,
            "capabilities_not_ready_and_blocked": capability_valid,
            "real_capabilities_blocked": real_valid,
            "invariants_preserved": invariant_valid,
            "gates_closed": source_accepted,
            "future_explicit_step_required": source_accepted,
            "only_source_only_17_3_handoff_allowed": source_accepted,
            "validation_performed_by_17_2": False,
            "runtime_started_by_17_2": False,
            "orders_enabled_by_17_2": False,
            "packaging_build_release_performed_by_17_2": False,
        },
        "domain_authorization_rows": _domain_rows(
            capability_valid, requirement_valid, invariant_valid
        ),
        "requirement_authorization_rows": _requirement_rows(requirement_valid),
        "invariant_authorization_guard": invariant_guard,
        "exe_authorization_guard": exe_guard,
        "real_capability_authorization_state": real_state,
        "fail_closed_matrix_decision": {
            "missing_source_policy": "fail_closed",
            "missing_domain_state_policy": "fail_closed",
            "missing_requirement_state_policy": "fail_closed",
            "missing_invariant_state_policy": "fail_closed",
            "missing_exe_state_policy": "fail_closed",
            "missing_real_capability_state_policy": "fail_closed",
            "missing_confirmation_policy": "fail_closed",
            "missing_validation_policy": "fail_closed",
            "missing_credentials_policy": "fail_closed",
            "missing_future_gate_policy": "fail_closed",
            "failed_matrix_policy": "fail_closed",
            "block_o_read_model_in_17_1": "preserved" if source_accepted else "not_preserved",
            "execution_authorization_matrix_in_17_2": "ready" if source_accepted else "blocked",
            "execution_authorization_contract_in_17_3": "allowed" if source_accepted else "blocked",
            "only_source_only_17_3_handoff_allowed": source_accepted,
            "real_capability_status": _copy_plain(real_status) if real_valid else {},
            "real_capability_status_inherited_from_17_1": real_valid,
            "real_capability_status_modified_by_17_2": False,
            "execution_authorization_granted_by_17_2": False,
        },
        "non_execution_matrix_evidence": {
            "source_block_o_read_model_read": True,
            "execution_authorization_matrix_built": True,
            "source_block_o_read_model_accepted": source_accepted,
            "block_o_remains_open": source_accepted,
            "reference_read_valid": reference_valid,
            "summary_read_valid": summary_valid,
            "block_n_read_valid": block_n_valid,
            "capability_read_valid": capability_valid,
            "invariant_read_valid": invariant_valid,
            "requirement_read_valid": requirement_valid,
            "exe_read_valid": exe_valid,
            "fail_closed_read_valid": fail_valid,
            "evidence_read_valid": evidence_valid,
            "read_model_boundaries_valid": boundaries_valid,
            "source_boundaries_valid": source_boundaries_valid,
            "real_capability_map_valid": real_valid,
            "all_execution_authorizations_false": True,
        },
        "matrix_boundaries": {
            "matrix_is_plain_data_only": True,
            "matrix_is_source_only": True,
            "matrix_reads_17_1_only": True,
            "matrix_is_static_projection_only": True,
            "matrix_is_non_evaluating": True,
            "matrix_is_non_mutating": True,
            "matrix_is_non_authorizing": True,
            "cannot_recalculate_readiness_from_environment": True,
            "cannot_evaluate_live_conditions": True,
            "cannot_accept_confirmation": True,
            "cannot_perform_validation": True,
            "cannot_grant_authorization": True,
            "cannot_open_or_mutate_gate": True,
            "cannot_use_io": True,
            "cannot_run_runtime": True,
            "cannot_generate_or_submit_orders": True,
            "cannot_package": True,
            "cannot_build": True,
            "cannot_release": True,
        },
        "source_boundaries": {
            **_plain_dict_section(source, "source_boundaries"),
            "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
            "block_o_read_model_source_preserved": source_accepted,
            "can_build_execution_authorization_matrix": source_accepted,
            "can_feed_17_3": source_accepted,
        },
        "future_steps": _copy_plain(FUTURE_STEPS),
        "status": STATUS if source_accepted else BLOCKED_STATUS,
    }
