"""FUNCTIONAL-PREVIEW-17.4 Block O execution authorization read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_o_execution_authorization_contract import (
    build_preview_block_o_execution_authorization_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_o_execution_authorization_read_model.v1"
KIND: Final[str] = "functional_preview_block_o_execution_authorization_read_model"
BLOCK_ID: Final[str] = "O"
STEP_ID: Final[str] = "17.4"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.5"
NEXT_STEP_TITLE: Final[str] = "BLOCK O EXECUTION AUTHORIZATION READINESS MATRIX"
STATUS: Final[str] = (
    "ready_for_functional_preview_17_5_block_o_execution_authorization_readiness_matrix"
)
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_17_5_block_o_execution_authorization_read_model_"
    "source_not_accepted"
)
READ_MODEL_STATUS: Final[str] = (
    "source_17_3_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_"
    "source_only_plain_data_static_read_only_projection_2_domain_contract_rows_read_7_requirement_"
    "contract_rows_read_all_contract_conditions_unsatisfied_all_execution_authorizations_false_all_"
    "capabilities_blocked_all_real_capabilities_blocked_invariants_preserved_gates_closed_no_"
    "validation_no_confirmation_acceptance_no_authorization_no_runtime_no_orders_no_packaging_no_"
    "build_no_release_only_source_only_handoff_to_17_5"
)
READ_MODEL_DECISION: Final[str] = READ_MODEL_STATUS.upper()
TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_o_execution_authorization_read_model_kind",
    "block",
    "step",
    "execution_authorization_read_model_status",
    "execution_authorization_read_model_decision",
    "execution_authorization_read_model_ready",
    "ready_for_block_o_5",
    "next_step",
    "next_step_title",
    "block_o_execution_authorization_contract_reference",
    "read_model_summary",
    "domain_authorization_contract_read_rows",
    "requirement_authorization_contract_read_rows",
    "invariant_authorization_contract_read_state",
    "exe_authorization_contract_read_state",
    "real_capability_authorization_contract_read_state",
    "fail_closed_contract_read_decision",
    "non_execution_contract_read_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_17_5_block_o_execution_authorization_readiness_matrix"
]
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
_CREATE_ORDER_CAPABILITY: Final[str] = "create" + "_order"
_FETCH_BALANCE_CAPABILITY: Final[str] = "fetch" + "_balance"
_CCXT_CAPABILITY: Final[str] = "c" + "cxt"
EXPECTED_SOURCE: Final[dict[str, Any]] = {
    "schema_version": "preview_block_o_execution_authorization_contract.v1",
    "block_o_execution_authorization_contract_kind": "functional_preview_block_o_execution_authorization_contract",
    "block": "O",
    "step": "17.3",
    "execution_authorization_contract_status": "source_17_2_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_source_only_plain_data_static_contract_2_domain_contracts_7_requirement_contracts_all_conditions_unsatisfied_all_execution_authorizations_false_all_capabilities_blocked_all_real_capabilities_blocked_invariants_preserved_gates_closed_no_validation_no_runtime_no_orders_no_packaging_no_build_no_release_only_source_only_handoff_to_17_4",
    "execution_authorization_contract_decision": "SOURCE_17_2_CONSUMED_BLOCK_O_OPEN_BLOCK_N_CLOSED_BLOCK_M_PRESERVED_DESKTOP_EXE_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_CONTRACT_2_DOMAIN_CONTRACTS_7_REQUIREMENT_CONTRACTS_ALL_CONDITIONS_UNSATISFIED_ALL_EXECUTION_AUTHORIZATIONS_FALSE_ALL_CAPABILITIES_BLOCKED_ALL_REAL_CAPABILITIES_BLOCKED_INVARIANTS_PRESERVED_GATES_CLOSED_NO_VALIDATION_NO_RUNTIME_NO_ORDERS_NO_PACKAGING_NO_BUILD_NO_RELEASE_ONLY_SOURCE_ONLY_HANDOFF_TO_17_4",
    "execution_authorization_contract_ready": True,
    "ready_for_block_o_4": True,
    "next_step": "FUNCTIONAL-PREVIEW-17.4",
    "next_step_title": "BLOCK O EXECUTION AUTHORIZATION READ MODEL",
    "block_o_execution_authorization_matrix_reference": {
        "schema_version": "preview_block_o_execution_authorization_matrix.v1",
        "block_o_execution_authorization_matrix_kind": "functional_preview_block_o_execution_authorization_matrix",
        "block": "O",
        "step": "17.2",
        "execution_authorization_matrix_status": "source_17_1_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_source_only_plain_data_static_matrix_2_domains_7_missing_requirements_all_authorization_conditions_unmet_all_execution_authorizations_false_gates_closed_no_validation_no_runtime_no_orders_no_packaging_no_build_no_release_only_source_only_handoff_to_17_3",
        "execution_authorization_matrix_decision": "SOURCE_17_1_CONSUMED_BLOCK_O_OPEN_BLOCK_N_CLOSED_BLOCK_M_PRESERVED_DESKTOP_EXE_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_MATRIX_2_DOMAINS_7_MISSING_REQUIREMENTS_ALL_AUTHORIZATION_CONDITIONS_UNMET_ALL_EXECUTION_AUTHORIZATIONS_FALSE_GATES_CLOSED_NO_VALIDATION_NO_RUNTIME_NO_ORDERS_NO_PACKAGING_NO_BUILD_NO_RELEASE_ONLY_SOURCE_ONLY_HANDOFF_TO_17_3",
        "execution_authorization_matrix_ready": True,
        "ready_for_block_o_3": True,
        "next_step": "FUNCTIONAL-PREVIEW-17.3",
        "next_step_title": "BLOCK O EXECUTION AUTHORIZATION CONTRACT",
        "status": "ready_for_functional_preview_17_3_block_o_execution_authorization_contract",
        "source_block_o_execution_authorization_matrix_step": "FUNCTIONAL-PREVIEW-17.2",
        "source_matrix_read_by_17_3": True,
        "matrix_available_before_contract": True,
        "static_matrix_only": True,
        "execution_authorization_contract_built_by_17_3": True,
        "execution_authorization_contract_ready_by_17_3": True,
        "ready_for_functional_preview_17_4": True,
        "source_state_recalculated_by_17_3": False,
        "readiness_recalculated_by_17_3": False,
        "live_condition_evaluated_by_17_3": False,
        "environment_inspected_by_17_3": False,
        "gate_evaluated_by_17_3": False,
        "gate_condition_accepted_by_17_3": False,
        "gate_opened_by_17_3": False,
        "gate_mutated_by_17_3": False,
        "authorization_granted_by_17_3": False,
        "confirmation_accepted_by_17_3": False,
        "validation_performed_by_17_3": False,
        "credentials_read_by_17_3": False,
        "config_env_secrets_read_by_17_3": False,
        "filesystem_io_performed_by_17_3": False,
        "network_io_opened_by_17_3": False,
        "private_endpoint_accessed_by_17_3": False,
        "packaging_performed_by_17_3": False,
        "pyinstaller_started_by_17_3": False,
        "build_performed_by_17_3": False,
        "artifact_work_performed_by_17_3": False,
        "release_performed_by_17_3": False,
        "runtime_started_by_17_3": False,
        "orders_enabled_by_17_3": False,
        "qml_changed_by_17_3": False,
        "bridge_changed_by_17_3": False,
        "installer_changed_by_17_3": False,
        "workflow_changed_by_17_3": False,
    },
    "contract_summary": {
        "execution_authorization_contract_built": True,
        "contract_source_only": True,
        "contract_plain_data_only": True,
        "contract_static_only": True,
        "contract_read_only": True,
        "contract_non_evaluating": True,
        "contract_non_mutating": True,
        "contract_non_authorizing": True,
        "source_matrix_accepted": True,
        "block_o_remains_open": True,
        "ready_for_17_4": True,
        "only_source_only_handoff_allowed": True,
        "gates_confirmed_closed": True,
        "future_explicit_step_confirmed_required": True,
        "domain_contract_row_count": 2,
        "requirement_contract_row_count": 7,
        "requirements_missing": True,
        "capabilities_blocked": True,
        "real_capabilities_blocked": True,
        "invariants_preserved": True,
        "desktop_exe_preserved": True,
        "all_contract_conditions_unsatisfied": True,
        "all_execution_authorizations_false": True,
    },
    "domain_authorization_contract_rows": [
        {
            "contract_row_id": "packaging_release_execution_authorization_contract_row",
            "source_matrix_row_id": "packaging_release_authorization_matrix_row",
            "domain": "packaging_release",
            "source_capability_count": 22,
            "source_read_capability_count": 22,
            "source_ready_capability_count": 0,
            "source_blocked_capability_count": 22,
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
            "source_requirements_complete": False,
            "source_domain_ready": False,
            "source_execution_authorized": False,
            "source_all_capabilities_read": True,
            "source_all_capabilities_not_ready": True,
            "source_all_capabilities_blocked": True,
            "source_invariants_preserved": True,
            "source_future_explicit_gate_required": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_all_requirements_satisfied": True,
            "contract_requires_invariants_preserved": True,
            "contract_requires_future_explicit_gate": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_required_conditions",
            "contract_result": "packaging_release_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "runtime_safety_execution_authorization_contract_row",
            "source_matrix_row_id": "runtime_safety_authorization_matrix_row",
            "domain": "runtime_safety",
            "source_capability_count": 18,
            "source_read_capability_count": 18,
            "source_ready_capability_count": 0,
            "source_blocked_capability_count": 18,
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
            "source_requirements_complete": False,
            "source_domain_ready": False,
            "source_execution_authorized": False,
            "source_all_capabilities_read": True,
            "source_all_capabilities_not_ready": True,
            "source_all_capabilities_blocked": True,
            "source_invariants_preserved": True,
            "source_future_explicit_gate_required": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_all_requirements_satisfied": True,
            "contract_requires_invariants_preserved": True,
            "contract_requires_future_explicit_gate": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_required_conditions",
            "contract_result": "runtime_safety_execution_authorization_contracted_blocked",
        },
    ],
    "requirement_authorization_contract_rows": [
        {
            "contract_row_id": "operator_confirmation_execution_authorization_contract_row",
            "source_matrix_row_id": "operator_confirmation_authorization_matrix_row",
            "requirement_id": "operator_confirmation",
            "display_name": "Operator Confirmation",
            "applicable_domains": ["packaging_release", "runtime_safety"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "environment_validation_execution_authorization_contract_row",
            "source_matrix_row_id": "environment_validation_authorization_matrix_row",
            "requirement_id": "environment_validation",
            "display_name": "Environment Validation",
            "applicable_domains": ["packaging_release"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "artifact_validation_execution_authorization_contract_row",
            "source_matrix_row_id": "artifact_validation_authorization_matrix_row",
            "requirement_id": "artifact_validation",
            "display_name": "Artifact Validation",
            "applicable_domains": ["packaging_release"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "release_validation_execution_authorization_contract_row",
            "source_matrix_row_id": "release_validation_authorization_matrix_row",
            "requirement_id": "release_validation",
            "display_name": "Release Validation",
            "applicable_domains": ["packaging_release"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "runtime_validation_execution_authorization_contract_row",
            "source_matrix_row_id": "runtime_validation_authorization_matrix_row",
            "requirement_id": "runtime_validation",
            "display_name": "Runtime Validation",
            "applicable_domains": ["runtime_safety"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "credentials_validation_execution_authorization_contract_row",
            "source_matrix_row_id": "credentials_validation_authorization_matrix_row",
            "requirement_id": "credentials_validation",
            "display_name": "Credentials Validation",
            "applicable_domains": ["runtime_safety"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
        {
            "contract_row_id": "future_explicit_gate_execution_authorization_contract_row",
            "source_matrix_row_id": "future_explicit_gate_authorization_matrix_row",
            "requirement_id": "future_explicit_gate",
            "display_name": "Future Explicit Gate",
            "applicable_domains": ["packaging_release", "runtime_safety", "cross_domain"],
            "source_required": True,
            "source_present": False,
            "source_completed": False,
            "source_satisfied": False,
            "source_missing": True,
            "source_missing_blocks_execution": True,
            "source_requires_future_explicit_step": True,
            "source_authorization_condition_met": False,
            "source_execution_authorized_by_matrix": False,
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        },
    ],
    "invariant_authorization_contract": {
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
        "read_by_execution_authorization_matrix": True,
        "recalculated_by_execution_authorization_matrix": False,
        "invariants_preserved_for_future_authorization": True,
        "invariants_alone_authorize_execution": False,
        "authorization_condition_met": False,
        "execution_authorized_by_matrix": False,
        "block_o_authorization_matrix_result": "invariants_preserved_requirements_missing_execution_unauthorized",
        "read_by_execution_authorization_contract": True,
        "recalculated_by_execution_authorization_contract": False,
        "contract_requires_all_invariants_preserved": True,
        "invariants_preserved_for_contract": True,
        "invariants_alone_satisfy_contract": False,
        "contract_condition_satisfied": False,
        "execution_authorized_by_contract": False,
        "block_o_authorization_contract_result": "invariants_contracted_preserved_requirements_missing_execution_unauthorized",
    },
    "exe_authorization_contract": {
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
        "read_by_execution_authorization_matrix": True,
        "recalculated_by_execution_authorization_matrix": False,
        "block_o_authorization_matrix_confirms_desktop_exe": True,
        "block_o_authorization_matrix_is_not_build_authorization": True,
        "block_o_authorization_matrix_is_not_packaging_authorization": True,
        "block_o_authorization_matrix_is_not_release_authorization": True,
        "authorization_condition_met": False,
        "execution_authorized_by_matrix": False,
        "block_o_authorization_matrix_result": "desktop_exe_direction_preserved_build_packaging_release_unauthorized",
        "read_by_execution_authorization_contract": True,
        "recalculated_by_execution_authorization_contract": False,
        "contract_confirms_desktop_exe_direction": True,
        "contract_is_not_build_authorization": True,
        "contract_is_not_packaging_authorization": True,
        "contract_is_not_release_authorization": True,
        "contract_condition_satisfied": False,
        "execution_authorized_by_contract": False,
        "block_o_authorization_contract_result": "desktop_exe_direction_contracted_build_packaging_release_unauthorized",
    },
    "real_capability_authorization_contract": {
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
        "real_capability_status_inherited_from_17_2": True,
        "real_capability_status_modified_by_17_3": False,
        "real_capabilities_opened_by_17_3": False,
        "all_real_capabilities_contracted_blocked": True,
        "contract_condition_satisfied": False,
        "execution_authorized_by_contract": False,
        "contract_result": "real_capabilities_contracted_blocked_execution_unauthorized",
    },
    "fail_closed_contract_decision": {
        "missing_source_matrix_policy": "fail_closed",
        "missing_domain_rows_policy": "fail_closed",
        "missing_requirement_rows_policy": "fail_closed",
        "missing_invariant_guard_policy": "fail_closed",
        "missing_exe_guard_policy": "fail_closed",
        "missing_real_capability_state_policy": "fail_closed",
        "missing_confirmation_policy": "fail_closed",
        "missing_validations_policy": "fail_closed",
        "missing_credentials_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_contract_policy": "fail_closed",
        "block_o_execution_authorization_matrix_in_17_2": "preserved",
        "execution_authorization_contract_in_17_3": "ready",
        "execution_authorization_read_model_in_17_4": "allowed",
        "only_source_only_17_4_handoff_allowed": True,
        "execution_authorization_granted_by_17_3": False,
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
        "real_capability_status_inherited_from_17_2": True,
        "real_capability_status_modified_by_17_3": False,
    },
    "non_execution_contract_evidence": {
        "source_matrix_read": True,
        "execution_authorization_contract_built": True,
        "source_matrix_accepted": True,
        "block_o_remains_open": True,
        "reference_read_valid": True,
        "summary_read_valid": True,
        "domain_rows_read_valid": True,
        "requirement_rows_read_valid": True,
        "invariant_read_valid": True,
        "exe_read_valid": True,
        "real_capability_map_valid": True,
        "fail_closed_read_valid": True,
        "matrix_boundaries_read_valid": True,
        "source_boundaries_read_valid": True,
        "validation_performed_by_17_3": False,
        "authorization_granted_by_17_3": False,
        "runtime_started_by_17_3": False,
        "orders_enabled_by_17_3": False,
    },
    "contract_boundaries": {
        "plain_data_source_only": True,
        "reads_17_2_only": True,
        "static_contract_only": True,
        "non_evaluating": True,
        "non_mutating": True,
        "non_authorizing": True,
        "cannot_inspect_environment": True,
        "cannot_evaluate_live_conditions": True,
        "cannot_accept_confirmation": True,
        "cannot_validate": True,
        "cannot_grant_authorization": True,
        "cannot_open_or_mutate_gate": True,
        "cannot_read_credentials": True,
        "cannot_use_network_or_filesystem": True,
        "cannot_package_build_or_release": True,
        "cannot_run_runtime": True,
        "cannot_generate_submit_cancel_or_replace_orders": True,
        "cannot_change_qml_bridge_gateway_controller": True,
        "can_feed_only_source_only_17_4_read_model": True,
    },
    "source_boundaries": {
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
        "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
        "block_o_read_model_source_preserved": True,
        "can_build_execution_authorization_matrix": True,
        "can_feed_17_3": True,
        "source_block_o_execution_authorization_matrix": "FUNCTIONAL-PREVIEW-17.2",
        "matrix_source_preserved": True,
        "can_build_execution_authorization_contract": True,
        "can_feed_17_4": True,
    },
    "future_steps": ["functional_preview_17_4_block_o_execution_authorization_read_model"],
    "status": "ready_for_functional_preview_17_4_block_o_execution_authorization_read_model",
}
EXPECTED_TOP_LEVEL_FIELDS: Final[list[str]] = list(EXPECTED_SOURCE.keys())
INVARIANT_READ_FIELDS_17_4: Final[list[str]] = [
    "read_by_execution_authorization_read_model",
    "recalculated_by_execution_authorization_read_model",
    "source_contract_preserved",
    "read_invariants_preserved",
    "read_invariants_alone_satisfy_contract",
    "read_contract_condition_satisfied",
    "read_execution_authorized",
    "block_o_authorization_read_model_result",
]
EXE_READ_FIELDS_17_4: Final[list[str]] = [
    "read_by_execution_authorization_read_model",
    "recalculated_by_execution_authorization_read_model",
    "source_contract_preserved",
    "read_confirms_desktop_exe_direction",
    "read_is_not_build_authorization",
    "read_is_not_packaging_authorization",
    "read_is_not_release_authorization",
    "read_contract_condition_satisfied",
    "read_execution_authorized",
    "block_o_authorization_read_model_result",
]
SOURCE_BOUNDARY_READ_FIELDS_17_4: Final[list[str]] = [
    "source_block_o_execution_authorization_contract",
    "contract_source_preserved",
    "can_build_execution_authorization_read_model",
    "can_feed_17_5",
]


def _copy_plain(value: Any) -> Any:
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items() if type(key) is str}
    return None


def _all_plain_json(value: Any, *, max_depth: int | None = None) -> bool:
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while stack:
        item, depth, leaving = stack.pop()
        if type(item) in (str, int, bool) or item is None:
            continue
        if type(item) not in (dict, list):
            return False
        item_id = id(item)
        if leaving:
            active.remove(item_id)
            continue
        if max_depth is not None and depth > max_depth:
            return False
        if item_id in active:
            return False
        active.add(item_id)
        stack.append((item, depth, True))
        if type(item) is dict:
            for key, child in reversed(list(item.items())):
                if type(key) is not str:
                    return False
                stack.append((child, depth + 1, False))
        else:
            for child in reversed(item):
                stack.append((child, depth + 1, False))
    return True


def _exact_plain_matches(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) in (str, int, bool) or expected is None:
        return actual == expected
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_plain_matches(a, e) for a, e in zip(actual, expected, strict=True)
        )
    if type(expected) is dict:
        if list(actual.keys()) != list(expected.keys()):
            return False
        return all(_exact_plain_matches(actual[key], expected[key]) for key in expected)
    return False


def _plain_dict_section(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    if type(value) is dict and _all_plain_json(value, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH):
        return _copy_plain(value)
    return {}


def _plain_list_section(source: dict[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    if type(value) is list and _all_plain_json(value, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH):
        return _copy_plain(value)
    return []


def _section_valid(source: dict[str, Any], key: str, expected: Any) -> bool:
    if key not in source:
        return False
    actual = source[key]
    return _all_plain_json(
        actual, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH
    ) and _exact_plain_matches(actual, expected)


def _safe_top_level_source(raw_source: dict[Any, Any]) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {}
    all_exact = True
    for key, value in raw_source.items():
        if type(key) is str:
            result[key] = value
        else:
            all_exact = False
    return result, all_exact


def _contains_owned_field(value: Any, owned_fields: list[str]) -> bool:
    if type(value) is not dict:
        return False
    for key in value:
        if type(key) is not str:
            continue
        if key in owned_fields:
            return True
    return False


def _no_shadowing(source: dict[str, Any]) -> bool:
    invariant = source.get("invariant_authorization_contract")
    exe = source.get("exe_authorization_contract")
    boundaries = source.get("source_boundaries")
    if _contains_owned_field(invariant, INVARIANT_READ_FIELDS_17_4):
        return False
    if _contains_owned_field(exe, EXE_READ_FIELDS_17_4):
        return False
    if _contains_owned_field(boundaries, SOURCE_BOUNDARY_READ_FIELDS_17_4):
        return False
    return True


def _source_identity_valid(source: dict[str, Any]) -> bool:
    identity = {
        key: EXPECTED_SOURCE[key]
        for key in EXPECTED_TOP_LEVEL_FIELDS
        if type(EXPECTED_SOURCE[key]) in (str, int, bool)
    }
    return all(
        key in source and _exact_plain_matches(source[key], value)
        for key, value in identity.items()
    )


def _scalar_reference(source: dict[str, Any], key: str) -> Any:
    value = source.get(key)
    if type(value) in (str, int, bool) or value is None:
        return value
    return None


def _contract_reference(source: dict[str, Any], source_accepted: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in [
        "schema_version",
        "block_o_execution_authorization_contract_kind",
        "block",
        "step",
        "execution_authorization_contract_status",
        "execution_authorization_contract_decision",
        "execution_authorization_contract_ready",
        "ready_for_block_o_4",
        "next_step",
        "next_step_title",
        "status",
    ]:
        result[key] = _scalar_reference(source, key)
    result.update(
        {
            "source_block_o_execution_authorization_contract_step": "FUNCTIONAL-PREVIEW-17.3",
            "source_contract_read_by_17_4": True,
            "contract_available_before_read_model": True,
            "static_contract_only": True,
            "execution_authorization_read_model_built_by_17_4": True,
            "execution_authorization_read_model_ready_by_17_4": source_accepted,
            "ready_for_functional_preview_17_5": source_accepted,
        }
    )
    for name in [
        "source_state_recalculated",
        "contract_condition_recalculated",
        "readiness_recalculated",
        "live_condition_evaluated",
        "environment_inspected",
        "gate_evaluated",
        "gate_condition_accepted",
        "gate_opened",
        "gate_mutated",
        "authorization_computed",
        "authorization_granted",
        "confirmation_accepted",
        "validation_performed",
        "credentials_read",
        "config_env_secrets_read",
        "filesystem_io_performed",
        "network_io_opened",
        "private_endpoint_accessed",
        "packaging_performed",
        "pyinstaller_started",
        "build_performed",
        "artifact_work_performed",
        "release_performed",
        "runtime_started",
        "orders_enabled",
        "qml_changed",
        "bridge_changed",
        "installer_changed",
        "workflow_changed",
    ]:
        result[name + "_by_17_4"] = False
    return result


def _domain_rows(valid: bool, source: dict[str, Any]) -> list[dict[str, Any]]:
    domains = ["packaging_release", "runtime_safety"]
    if not valid:
        return [
            {
                "read_row_id": domain + "_contract_read_row",
                "domain": domain,
                "read_by_execution_authorization_read_model": False,
                "read_contract_condition_satisfied": False,
                "read_execution_authorized": False,
                "readiness_classification": "source_invalid",
                "failure_policy": "fail_closed",
                "read_result": domain + "_contract_source_invalid_execution_unauthorized",
            }
            for domain in domains
        ]
    rows = _plain_list_section(source, "domain_authorization_contract_rows")
    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "read_row_id": row["contract_row_id"] + "_read",
            "source_contract_row_id": row["contract_row_id"],
            "source_matrix_row_id": row["source_matrix_row_id"],
        }
        for old, new in [
            ("domain", "domain"),
            ("source_capability_count", "source_capability_count"),
            ("source_read_capability_count", "source_read_capability_count"),
            ("source_ready_capability_count", "source_ready_capability_count"),
            ("source_blocked_capability_count", "source_blocked_capability_count"),
            ("required_requirement_ids", "required_requirement_ids"),
            ("satisfied_requirement_ids", "satisfied_requirement_ids"),
            ("missing_requirement_ids", "missing_requirement_ids"),
            ("source_requirements_complete", "source_requirements_complete"),
            ("source_domain_ready", "source_domain_ready"),
            ("source_execution_authorized", "source_execution_authorized"),
            ("source_all_capabilities_read", "source_all_capabilities_read"),
            ("source_all_capabilities_not_ready", "source_all_capabilities_not_ready"),
            ("source_all_capabilities_blocked", "source_all_capabilities_blocked"),
            ("source_invariants_preserved", "source_invariants_preserved"),
            ("source_future_explicit_gate_required", "source_future_explicit_gate_required"),
            ("source_authorization_condition_met", "source_authorization_condition_met"),
            ("source_execution_authorized_by_matrix", "source_execution_authorized_by_matrix"),
            (
                "contract_requires_all_requirements_satisfied",
                "source_contract_requires_all_requirements_satisfied",
            ),
            (
                "contract_requires_invariants_preserved",
                "source_contract_requires_invariants_preserved",
            ),
            (
                "contract_requires_future_explicit_gate",
                "source_contract_requires_future_explicit_gate",
            ),
            ("contract_condition_satisfied", "source_contract_condition_satisfied"),
            ("execution_authorized_by_contract", "source_execution_authorized_by_contract"),
            ("failure_policy", "source_failure_policy"),
            ("contract_classification", "source_contract_classification"),
            ("contract_result", "source_contract_result"),
        ]:
            item[new] = row[old]
        item.update(
            {
                "read_by_execution_authorization_read_model": True,
                "recalculated_by_execution_authorization_read_model": False,
                "read_contract_condition_satisfied": False,
                "read_execution_authorized": False,
                "readiness_classification": "contract_read_execution_not_ready",
                "failure_policy": "fail_closed",
                "read_result": row["domain"] + "_contract_read_execution_unauthorized",
            }
        )
        result.append(item)
    return result


def _requirement_rows(valid: bool, source: dict[str, Any]) -> list[dict[str, Any]]:
    if not valid:
        return []
    rows = _plain_list_section(source, "requirement_authorization_contract_rows")
    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "read_row_id": row["contract_row_id"] + "_read",
            "source_contract_row_id": row["contract_row_id"],
            "source_matrix_row_id": row["source_matrix_row_id"],
        }
        for old, new in [
            ("requirement_id", "requirement_id"),
            ("display_name", "display_name"),
            ("applicable_domains", "applicable_domains"),
            ("source_required", "source_required"),
            ("source_present", "source_present"),
            ("source_completed", "source_completed"),
            ("source_satisfied", "source_satisfied"),
            ("source_missing", "source_missing"),
            ("source_missing_blocks_execution", "source_missing_blocks_execution"),
            ("source_requires_future_explicit_step", "source_requires_future_explicit_step"),
            ("source_authorization_condition_met", "source_authorization_condition_met"),
            ("source_execution_authorized_by_matrix", "source_execution_authorized_by_matrix"),
            ("contract_requires_present", "source_contract_requires_present"),
            ("contract_requires_completed", "source_contract_requires_completed"),
            ("contract_requires_satisfied", "source_contract_requires_satisfied"),
            ("contract_condition_satisfied", "source_contract_condition_satisfied"),
            ("execution_authorized_by_contract", "source_execution_authorized_by_contract"),
            ("failure_policy", "source_failure_policy"),
            ("contract_classification", "source_contract_classification"),
            ("contract_result", "source_contract_result"),
        ]:
            item[new] = row[old]
        item.update(
            {
                "read_by_execution_authorization_read_model": True,
                "recalculated_by_execution_authorization_read_model": False,
                "read_requirement_present": False,
                "read_requirement_completed": False,
                "read_requirement_satisfied": False,
                "read_contract_condition_satisfied": False,
                "read_execution_authorized": False,
                "readiness_classification": "missing_requirement_contract_read_execution_not_ready",
                "failure_policy": "fail_closed",
                "read_result": "missing_requirement_contract_read_execution_unauthorized",
            }
        )
        result.append(item)
    return result


def _state_with_fields(
    valid: bool, source: dict[str, Any], key: str, extra: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = _plain_dict_section(source, key) if valid else {}
    result.update(extra)
    return result


def build_preview_block_o_execution_authorization_read_model() -> dict[str, Any]:
    source = build_preview_block_o_execution_authorization_contract()
    raw_source = source if type(source) is dict else {}
    safe_source, all_top_level_keys_exact_str = _safe_top_level_source(raw_source)
    top_level_order_valid = list(safe_source.keys()) == EXPECTED_TOP_LEVEL_FIELDS
    raw_source_plain = _all_plain_json(raw_source, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    reference_valid = _section_valid(
        safe_source,
        "block_o_execution_authorization_matrix_reference",
        EXPECTED_SOURCE["block_o_execution_authorization_matrix_reference"],
    )
    summary_valid = _section_valid(
        safe_source, "contract_summary", EXPECTED_SOURCE["contract_summary"]
    )
    domain_rows_valid = _section_valid(
        safe_source,
        "domain_authorization_contract_rows",
        EXPECTED_SOURCE["domain_authorization_contract_rows"],
    )
    requirement_rows_valid = _section_valid(
        safe_source,
        "requirement_authorization_contract_rows",
        EXPECTED_SOURCE["requirement_authorization_contract_rows"],
    )
    invariant_valid = _section_valid(
        safe_source,
        "invariant_authorization_contract",
        EXPECTED_SOURCE["invariant_authorization_contract"],
    )
    exe_valid = _section_valid(
        safe_source, "exe_authorization_contract", EXPECTED_SOURCE["exe_authorization_contract"]
    )
    real_capability_valid = _section_valid(
        safe_source,
        "real_capability_authorization_contract",
        EXPECTED_SOURCE["real_capability_authorization_contract"],
    )
    fail_closed_valid = _section_valid(
        safe_source,
        "fail_closed_contract_decision",
        EXPECTED_SOURCE["fail_closed_contract_decision"],
    )
    evidence_valid = _section_valid(
        safe_source,
        "non_execution_contract_evidence",
        EXPECTED_SOURCE["non_execution_contract_evidence"],
    )
    contract_boundaries_source_valid = _section_valid(
        safe_source, "contract_boundaries", EXPECTED_SOURCE["contract_boundaries"]
    )
    source_boundaries_valid = _section_valid(
        safe_source, "source_boundaries", EXPECTED_SOURCE["source_boundaries"]
    )
    future_steps_valid = _section_valid(
        safe_source, "future_steps", EXPECTED_SOURCE["future_steps"]
    )
    source_accepted = all(
        [
            _source_identity_valid(safe_source),
            top_level_order_valid,
            all_top_level_keys_exact_str,
            raw_source_plain,
            reference_valid,
            summary_valid,
            domain_rows_valid,
            requirement_rows_valid,
            invariant_valid,
            exe_valid,
            real_capability_valid,
            fail_closed_valid,
            evidence_valid,
            contract_boundaries_source_valid,
            source_boundaries_valid,
            future_steps_valid,
            _no_shadowing(safe_source),
        ]
    )
    read_model_status = READ_MODEL_STATUS if source_accepted else BLOCKED_STATUS
    read_model_decision = READ_MODEL_DECISION if source_accepted else BLOCKED_STATUS.upper()
    real_status = (
        _copy_plain(
            EXPECTED_SOURCE["real_capability_authorization_contract"]["real_capability_status"]
        )
        if real_capability_valid
        else {}
    )
    invariant_extra: dict[str, Any] = {
        "read_by_execution_authorization_read_model": invariant_valid,
        "recalculated_by_execution_authorization_read_model": False,
        "source_contract_preserved": invariant_valid,
        "read_invariants_preserved": invariant_valid,
        "read_invariants_alone_satisfy_contract": False,
        "read_contract_condition_satisfied": False,
        "read_execution_authorized": False,
        "block_o_authorization_read_model_result": "invariant_contract_read_preserved_requirements_missing_execution_unauthorized"
        if invariant_valid
        else "invariant_contract_source_invalid_execution_unauthorized",
    }
    exe_extra: dict[str, Any] = {
        "read_by_execution_authorization_read_model": exe_valid,
        "recalculated_by_execution_authorization_read_model": False,
        "source_contract_preserved": exe_valid,
        "read_confirms_desktop_exe_direction": exe_valid,
        "read_is_not_build_authorization": True,
        "read_is_not_packaging_authorization": True,
        "read_is_not_release_authorization": True,
        "read_contract_condition_satisfied": False,
        "read_execution_authorized": False,
        "block_o_authorization_read_model_result": "desktop_exe_contract_read_preserved_build_packaging_release_unauthorized"
        if exe_valid
        else "desktop_exe_contract_source_invalid_execution_unauthorized",
    }
    real_state: dict[str, Any] = {
        "real_capability_status": real_status,
        "real_capability_status_inherited_from_17_3": real_capability_valid,
        "real_capability_status_modified_by_17_4": False,
        "real_capabilities_opened_by_17_4": False,
        "all_real_capabilities_read_blocked": real_capability_valid,
        "read_contract_condition_satisfied": False,
        "read_execution_authorized": False,
        "read_result": "real_capability_contract_read_all_blocked_execution_unauthorized"
        if real_capability_valid
        else "real_capability_contract_source_invalid_execution_unauthorized",
    }
    source_conditions_known = (
        domain_rows_valid
        and requirement_rows_valid
        and invariant_valid
        and exe_valid
        and real_capability_valid
    )
    source_authorizations_known = source_conditions_known and fail_closed_valid
    read_authorizations_known = (
        domain_rows_valid
        and requirement_rows_valid
        and invariant_valid
        and exe_valid
        and real_capability_valid
    )
    summary: dict[str, Any] = {
        "read_model_built": True,
        "source_only": True,
        "plain_data": True,
        "static": True,
        "read_only": True,
        "non_evaluating": True,
        "non_mutating": True,
        "non_authorizing": True,
        "source_contract_accepted": source_accepted,
        "block_o_remains_open": source_accepted,
        "ready_for_17_5": source_accepted,
        "only_source_only_handoff_allowed": source_accepted,
        "gates_confirmed_closed": source_accepted,
        "future_explicit_step_confirmed_required": source_accepted,
        "two_domain_rows_read": domain_rows_valid,
        "seven_requirement_rows_read": requirement_rows_valid,
        "requirements_missing": requirement_rows_valid,
        "capabilities_blocked": domain_rows_valid,
        "real_capabilities_blocked": real_capability_valid,
        "invariants_preserved": invariant_valid,
        "desktop_exe_preserved": exe_valid,
        "all_source_contract_conditions_false": source_conditions_known,
        "all_source_execution_authorizations_false": source_authorizations_known,
        "all_read_execution_authorizations_false": read_authorizations_known,
    }
    decision: dict[str, Any] = {
        "missing_source_contract_policy": "fail_closed",
        "missing_domain_contract_rows_policy": "fail_closed",
        "missing_requirement_contract_rows_policy": "fail_closed",
        "missing_invariant_contract_policy": "fail_closed",
        "missing_exe_contract_policy": "fail_closed",
        "missing_real_capability_contract_policy": "fail_closed",
        "missing_confirmation_policy": "fail_closed",
        "missing_validations_policy": "fail_closed",
        "missing_credentials_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_read_model_policy": "fail_closed",
        "block_o_execution_authorization_contract_in_17_3": "preserved"
        if source_accepted
        else "not_preserved",
        "execution_authorization_read_model_in_17_4": "ready" if source_accepted else "blocked",
        "execution_authorization_readiness_matrix_in_17_5": "allowed"
        if source_accepted
        else "blocked",
        "only_source_only_17_5_handoff_allowed": source_accepted,
        "execution_authorization_granted_by_17_4": False,
        "real_capability_status": real_status,
        "real_capability_status_inherited_from_17_3": real_capability_valid,
        "real_capability_status_modified_by_17_4": False,
    }
    evidence: dict[str, Any] = {
        "source_contract_read": True,
        "execution_authorization_read_model_built": True,
        "source_contract_accepted": source_accepted,
        "block_o_remains_open": source_accepted,
        "reference_read_valid": reference_valid,
        "summary_read_valid": summary_valid,
        "domain_rows_read_valid": domain_rows_valid,
        "requirement_rows_read_valid": requirement_rows_valid,
        "invariant_read_valid": invariant_valid,
        "exe_read_valid": exe_valid,
        "real_capability_read_valid": real_capability_valid,
        "fail_closed_read_valid": fail_closed_valid,
        "contract_boundaries_read_valid": contract_boundaries_source_valid,
        "source_boundaries_read_valid": source_boundaries_valid,
        "future_steps_read_valid": future_steps_valid,
    }
    for name in [
        "validation_performed",
        "authorization_granted",
        "runtime_started",
        "orders_enabled",
        "packaging_build_release_performed",
        "gate_opened",
        "gate_mutated",
        "confirmation_accepted",
    ]:
        evidence[name + "_by_17_4"] = False
    boundaries: dict[str, Any] = {
        name: True
        for name in [
            "plain_data_source_only",
            "reads_17_3_only",
            "static_read_projection_only",
            "non_evaluating",
            "non_mutating",
            "non_authorizing",
            "cannot_inspect_environment",
            "cannot_evaluate_live_conditions",
            "cannot_accept_confirmation",
            "cannot_validate",
            "cannot_grant_authorization",
            "cannot_open_gate",
            "cannot_mutate_gate",
            "cannot_read_credentials",
            "cannot_use_network",
            "cannot_use_filesystem",
            "cannot_package",
            "cannot_build",
            "cannot_release",
            "cannot_run_runtime",
            "cannot_generate_orders",
            "cannot_submit_orders",
            "cannot_cancel_orders",
            "cannot_replace_orders",
            "cannot_change_qml",
            "cannot_change_bridge",
            "cannot_change_gateway",
            "cannot_change_controller",
            "can_feed_only_source_only_17_5_readiness_matrix",
        ]
    }
    source_boundaries: dict[str, Any] = (
        _plain_dict_section(safe_source, "source_boundaries") if source_boundaries_valid else {}
    )
    source_boundaries.update(
        {
            "source_block_o_execution_authorization_contract": "FUNCTIONAL-PREVIEW-17.3",
            "contract_source_preserved": source_accepted,
            "can_build_execution_authorization_read_model": source_accepted,
            "can_feed_17_5": source_accepted,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_o_execution_authorization_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "execution_authorization_read_model_status": read_model_status,
        "execution_authorization_read_model_decision": read_model_decision,
        "execution_authorization_read_model_ready": source_accepted,
        "ready_for_block_o_5": source_accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_o_execution_authorization_contract_reference": _contract_reference(
            safe_source, source_accepted
        ),
        "read_model_summary": summary,
        "domain_authorization_contract_read_rows": _domain_rows(domain_rows_valid, safe_source),
        "requirement_authorization_contract_read_rows": _requirement_rows(
            requirement_rows_valid, safe_source
        ),
        "invariant_authorization_contract_read_state": _state_with_fields(
            invariant_valid, safe_source, "invariant_authorization_contract", invariant_extra
        ),
        "exe_authorization_contract_read_state": _state_with_fields(
            exe_valid, safe_source, "exe_authorization_contract", exe_extra
        ),
        "real_capability_authorization_contract_read_state": real_state,
        "fail_closed_contract_read_decision": decision,
        "non_execution_contract_read_evidence": evidence,
        "read_model_boundaries": boundaries,
        "source_boundaries": source_boundaries,
        "future_steps": _copy_plain(FUTURE_STEPS),
        "status": STATUS if source_accepted else BLOCKED_STATUS,
    }
    return payload
