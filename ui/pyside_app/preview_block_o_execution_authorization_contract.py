"""FUNCTIONAL-PREVIEW-17.3 Block O execution authorization contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_o_execution_authorization_matrix import (
    build_preview_block_o_execution_authorization_matrix,
)

SCHEMA_VERSION: Final[str] = "preview_block_o_execution_authorization_contract.v1"
KIND: Final[str] = "functional_preview_block_o_execution_authorization_contract"
BLOCK_ID: Final[str] = "O"
STEP_ID: Final[str] = "17.3"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.4"
NEXT_STEP_TITLE: Final[str] = "BLOCK O EXECUTION AUTHORIZATION READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_17_4_block_o_execution_authorization_read_model"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_17_4_block_o_execution_authorization_contract_source_not_accepted"
)
CONTRACT_STATUS: Final[str] = (
    "source_17_2_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_"
    "source_only_plain_data_static_contract_2_domain_contracts_7_requirement_contracts_all_"
    "conditions_unsatisfied_all_execution_authorizations_false_all_capabilities_blocked_all_real_"
    "capabilities_blocked_invariants_preserved_gates_closed_no_validation_no_runtime_no_orders_no_"
    "packaging_no_build_no_release_only_source_only_handoff_to_17_4"
)
CONTRACT_DECISION: Final[str] = CONTRACT_STATUS.upper()
TOP_LEVEL_FIELDS: Final[list[str]] = [
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
    "block_o_execution_authorization_matrix_reference",
    "contract_summary",
    "domain_authorization_contract_rows",
    "requirement_authorization_contract_rows",
    "invariant_authorization_contract",
    "exe_authorization_contract",
    "real_capability_authorization_contract",
    "fail_closed_contract_decision",
    "non_execution_contract_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_17_4_block_o_execution_authorization_read_model"
]
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
EXPECTED_SOURCE_TOP_LEVEL_FIELDS: Final[list[str]] = [
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
EXPECTED_IDENTITY: Final[dict[str, Any]] = {
    "block": "O",
    "block_o_execution_authorization_matrix_kind": "functional_preview_block_o_execution_authorization_matrix",
    "execution_authorization_matrix_decision": "SOURCE_17_1_CONSUMED_BLOCK_O_OPEN_BLOCK_N_CLOSED_BLOCK_M_PRESERVED_DESKTOP_EXE_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_MATRIX_2_DOMAINS_7_MISSING_REQUIREMENTS_ALL_AUTHORIZATION_CONDITIONS_UNMET_ALL_EXECUTION_AUTHORIZATIONS_FALSE_GATES_CLOSED_NO_VALIDATION_NO_RUNTIME_NO_ORDERS_NO_PACKAGING_NO_BUILD_NO_RELEASE_ONLY_SOURCE_ONLY_HANDOFF_TO_17_3",
    "execution_authorization_matrix_ready": True,
    "execution_authorization_matrix_status": "source_17_1_consumed_block_o_open_block_n_closed_block_m_preserved_desktop_exe_preserved_source_only_plain_data_static_matrix_2_domains_7_missing_requirements_all_authorization_conditions_unmet_all_execution_authorizations_false_gates_closed_no_validation_no_runtime_no_orders_no_packaging_no_build_no_release_only_source_only_handoff_to_17_3",
    "future_steps": ["functional_preview_17_3_block_o_execution_authorization_contract"],
    "next_step": "FUNCTIONAL-PREVIEW-17.3",
    "next_step_title": "BLOCK O EXECUTION AUTHORIZATION CONTRACT",
    "ready_for_block_o_3": True,
    "schema_version": "preview_block_o_execution_authorization_matrix.v1",
    "status": "ready_for_functional_preview_17_3_block_o_execution_authorization_contract",
    "step": "17.2",
}
EXPECTED_DOMAINS: Final[list[dict[str, Any]]] = [
    {
        "matrix_row_id": "packaging_release_authorization_matrix_row",
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
        "requirements_complete": False,
        "source_domain_ready": False,
        "source_execution_authorized": False,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "invariants_preserved": True,
        "future_explicit_gate_required": True,
        "authorization_condition_met": False,
        "execution_authorized_by_matrix": False,
        "failure_policy": "fail_closed",
        "authorization_classification": "blocked_missing_required_conditions",
        "matrix_result": "packaging_release_execution_unauthorized",
    },
    {
        "matrix_row_id": "runtime_safety_authorization_matrix_row",
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
        "requirements_complete": False,
        "source_domain_ready": False,
        "source_execution_authorized": False,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "invariants_preserved": True,
        "future_explicit_gate_required": True,
        "authorization_condition_met": False,
        "execution_authorized_by_matrix": False,
        "failure_policy": "fail_closed",
        "authorization_classification": "blocked_missing_required_conditions",
        "matrix_result": "runtime_safety_execution_unauthorized",
    },
]
REQ_IDS: Final[list[tuple[str, str, list[str]]]] = [
    ("operator_confirmation", "Operator Confirmation", ["packaging_release", "runtime_safety"]),
    ("environment_validation", "Environment Validation", ["packaging_release"]),
    ("artifact_validation", "Artifact Validation", ["packaging_release"]),
    ("release_validation", "Release Validation", ["packaging_release"]),
    ("runtime_validation", "Runtime Validation", ["runtime_safety"]),
    ("credentials_validation", "Credentials Validation", ["runtime_safety"]),
    (
        "future_explicit_gate",
        "Future Explicit Gate",
        ["packaging_release", "runtime_safety", "cross_domain"],
    ),
]
EXPECTED_REQUIREMENTS: Final[list[dict[str, Any]]] = [
    {
        "matrix_row_id": f"{rid}_authorization_matrix_row",
        "requirement_id": rid,
        "display_name": name,
        "applicable_domains": domains,
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
    for rid, name, domains in REQ_IDS
]
REAL_CAPABILITY_STATUS: Final[dict[str, str]] = {
    k: "blocked"
    for k in [
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
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "fetch_balance",
        "private_endpoint",
        "network",
        "credentials",
        "config_env_secrets",
        "qml_bridge",
        "ccxt",
    ]
}
EXPECTED_REFERENCE: Final = {
    "schema_version": "preview_block_o_read_model.v1",
    "block_o_read_model_kind": "functional_preview_block_o_read_model",
    "block": "O",
    "step": "17.1",
    "block_o_read_model_status": "block_o_read_model_ready_17_0_entry_contract_consumed_block_o_opened_block_n_closed_block_m_closure_preserved_desktop_exe_direction_preserved_source_only_plain_data_static_read_projection_only_all_capability_domains_read_all_execution_capabilities_not_ready_all_execution_capabilities_blocked_all_requirements_read_all_requirements_missing_all_requirements_block_execution_all_invariants_read_all_invariants_preserved_all_execution_unauthorized_all_gates_closed_no_source_state_recalculation_no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_ready_for_functional_preview_17_2_execution_authorization_matrix",
    "block_o_read_model_decision": "BLOCK_O_READ_MODEL_READY_17_0_ENTRY_CONTRACT_CONSUMED_BLOCK_O_OPENED_BLOCK_N_CLOSED_BLOCK_M_CLOSURE_PRESERVED_DESKTOP_EXE_DIRECTION_PRESERVED_SOURCE_ONLY_PLAIN_DATA_STATIC_READ_PROJECTION_ONLY_ALL_CAPABILITY_DOMAINS_READ_ALL_EXECUTION_CAPABILITIES_NOT_READY_ALL_EXECUTION_CAPABILITIES_BLOCKED_ALL_REQUIREMENTS_READ_ALL_REQUIREMENTS_MISSING_ALL_REQUIREMENTS_BLOCK_EXECUTION_ALL_INVARIANTS_READ_ALL_INVARIANTS_PRESERVED_ALL_EXECUTION_UNAUTHORIZED_ALL_GATES_CLOSED_NO_SOURCE_STATE_RECALCULATION_NO_GATE_EVALUATION_NO_VALIDATION_NO_CONFIRMATION_ACCEPTANCE_NO_AUTHORIZATION_NO_PACKAGING_NO_BUILD_NO_RELEASE_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO_READY_FOR_FUNCTIONAL_PREVIEW_17_2_EXECUTION_AUTHORIZATION_MATRIX",
    "block_o_read_model_ready": True,
    "ready_for_block_o_2": True,
    "next_step": "FUNCTIONAL-PREVIEW-17.2",
    "next_step_title": "BLOCK O EXECUTION AUTHORIZATION MATRIX",
    "source_block_o_read_model_step": "FUNCTIONAL-PREVIEW-17.1",
    "source_block_o_read_model_read_by_17_2": True,
    "block_o_read_model_available_before_matrix": True,
    "static_block_o_read_model_only": True,
    "execution_authorization_matrix_built_by_17_2": True,
    "execution_authorization_matrix_ready_by_17_2": True,
    "ready_for_functional_preview_17_3": True,
    "source_readiness_recalculated_by_17_2": False,
    "live_condition_inspected_by_17_2": False,
    "environment_inspected_by_17_2": False,
    "gate_evaluated_by_17_2": False,
    "gate_opened_by_17_2": False,
    "gate_mutated_by_17_2": False,
    "authorization_granted_by_17_2": False,
    "confirmation_accepted_by_17_2": False,
    "validation_performed_by_17_2": False,
    "packaging_performed_by_17_2": False,
    "pyinstaller_started_by_17_2": False,
    "build_performed_by_17_2": False,
    "artifact_created_by_17_2": False,
    "release_performed_by_17_2": False,
    "runtime_started_by_17_2": False,
    "orders_enabled_by_17_2": False,
    "private_endpoint_accessed_by_17_2": False,
    "network_io_opened_by_17_2": False,
    "credentials_read_by_17_2": False,
    "config_env_secrets_read_by_17_2": False,
    "filesystem_io_performed_by_17_2": False,
    "qml_changed_by_17_2": False,
    "bridge_changed_by_17_2": False,
    "installer_changed_by_17_2": False,
    "workflow_changed_by_17_2": False,
}

EXPECTED_SUMMARY: Final = {
    "execution_authorization_matrix_built": True,
    "source_only_plain_data_static_matrix": True,
    "non_mutating_non_authorizing": True,
    "source_17_1_accepted": True,
    "block_o_open": True,
    "block_n_closed": True,
    "block_m_preserved": True,
    "desktop_exe_preserved": True,
    "domain_row_count": 2,
    "requirement_row_count": 7,
    "all_requirements_required_and_missing": True,
    "all_authorization_conditions_unmet": True,
    "all_execution_authorizations_false": True,
    "capabilities_not_ready_and_blocked": True,
    "real_capabilities_blocked": True,
    "invariants_preserved": True,
    "gates_closed": True,
    "future_explicit_step_required": True,
    "only_source_only_17_3_handoff_allowed": True,
    "validation_performed_by_17_2": False,
    "runtime_started_by_17_2": False,
    "orders_enabled_by_17_2": False,
    "packaging_build_release_performed_by_17_2": False,
}

EXPECTED_INVARIANT_GUARD: Final = {
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
}

EXPECTED_EXE_GUARD: Final = {
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
}

EXPECTED_REAL_CAPABILITY_STATE: Final = {
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
        "create_order": "blocked",
        "submit_order": "blocked",
        "cancel_order": "blocked",
        "replace_order": "blocked",
        "fetch_balance": "blocked",
        "private_endpoint": "blocked",
        "network": "blocked",
        "credentials": "blocked",
        "config_env_secrets": "blocked",
        "qml_bridge": "blocked",
        "ccxt": "blocked",
    },
    "real_capability_status_inherited_from_17_1": True,
    "real_capability_status_modified_by_17_2": False,
    "real_capabilities_opened_by_17_2": False,
    "all_real_capabilities_blocked": True,
    "execution_authorized_by_matrix": False,
    "state_result": "real_capabilities_preserved_blocked_execution_unauthorized",
}

EXPECTED_FAIL_CLOSED_DECISION: Final = {
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
    "block_o_read_model_in_17_1": "preserved",
    "execution_authorization_matrix_in_17_2": "ready",
    "execution_authorization_contract_in_17_3": "allowed",
    "only_source_only_17_3_handoff_allowed": True,
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
        "create_order": "blocked",
        "submit_order": "blocked",
        "cancel_order": "blocked",
        "replace_order": "blocked",
        "fetch_balance": "blocked",
        "private_endpoint": "blocked",
        "network": "blocked",
        "credentials": "blocked",
        "config_env_secrets": "blocked",
        "qml_bridge": "blocked",
        "ccxt": "blocked",
    },
    "real_capability_status_inherited_from_17_1": True,
    "real_capability_status_modified_by_17_2": False,
    "execution_authorization_granted_by_17_2": False,
}

EXPECTED_EVIDENCE: Final = {
    "source_block_o_read_model_read": True,
    "execution_authorization_matrix_built": True,
    "source_block_o_read_model_accepted": True,
    "block_o_remains_open": True,
    "reference_read_valid": True,
    "summary_read_valid": True,
    "block_n_read_valid": True,
    "capability_read_valid": True,
    "invariant_read_valid": True,
    "requirement_read_valid": True,
    "exe_read_valid": True,
    "fail_closed_read_valid": True,
    "evidence_read_valid": True,
    "read_model_boundaries_valid": True,
    "source_boundaries_valid": True,
    "real_capability_map_valid": True,
    "all_execution_authorizations_false": True,
}

EXPECTED_MATRIX_BOUNDARIES: Final = {
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
    "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
    "block_o_read_model_source_preserved": True,
    "can_build_execution_authorization_matrix": True,
    "can_feed_17_3": True,
}

EXPECTED_FUTURE_STEPS: Final = ["functional_preview_17_3_block_o_execution_authorization_contract"]

INVARIANT_CONTRACT_FIELDS_17_3: Final[list[str]] = [
    "read_by_execution_authorization_contract",
    "recalculated_by_execution_authorization_contract",
    "contract_requires_all_invariants_preserved",
    "invariants_preserved_for_contract",
    "invariants_alone_satisfy_contract",
    "contract_condition_satisfied",
    "execution_authorized_by_contract",
    "block_o_authorization_contract_result",
]
EXE_CONTRACT_FIELDS_17_3: Final[list[str]] = [
    "read_by_execution_authorization_contract",
    "recalculated_by_execution_authorization_contract",
    "contract_confirms_desktop_exe_direction",
    "contract_is_not_build_authorization",
    "contract_is_not_packaging_authorization",
    "contract_is_not_release_authorization",
    "contract_condition_satisfied",
    "execution_authorized_by_contract",
    "block_o_authorization_contract_result",
]
SOURCE_BOUNDARY_FIELDS_17_3: Final[list[str]] = [
    "source_block_o_execution_authorization_matrix",
    "matrix_source_preserved",
    "can_build_execution_authorization_contract",
    "can_feed_17_4",
]


def _copy_plain(value: Any) -> Any:
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    if type(value) is dict:
        return {str(key): _copy_plain(item) for key, item in value.items() if type(key) is str}
    return None


def _all_plain_json(value: Any, *, max_depth: int | None = None) -> bool:
    stack: list[tuple[Any, bool, int]] = [(value, False, 0)]
    active_container_ids: set[int] = set()
    while stack:
        item, leaving, depth = stack.pop()
        if type(item) in (str, int, bool) or item is None:
            continue
        if type(item) not in (dict, list):
            return False
        ident = id(item)
        if leaving:
            active_container_ids.remove(ident)
            continue
        if max_depth is not None and depth > max_depth:
            return False
        if ident in active_container_ids:
            return False
        active_container_ids.add(ident)
        stack.append((item, True, depth))
        if type(item) is dict:
            child_items: list[Any] = []
            for key, child in item.items():
                if type(key) is not str:
                    return False
                child_items.append(child)
            for child in reversed(child_items):
                stack.append((child, False, depth + 1))
        else:
            for child in reversed(item):
                stack.append((child, False, depth + 1))
    return True


def _exact_plain_matches(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if list(actual.keys()) != list(expected.keys()):
            return False
        return all(_exact_plain_matches(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        if len(actual) != len(expected):
            return False
        return all(_exact_plain_matches(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected


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
    value = source.get(key)
    return _all_plain_json(
        value, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH
    ) and _exact_plain_matches(value, expected)


def _safe_top_level_source(raw_source: dict[Any, Any]) -> tuple[dict[str, Any], bool]:
    safe_source: dict[str, Any] = {}
    all_top_level_keys_exact_str = True
    for key, value in raw_source.items():
        if type(key) is not str:
            all_top_level_keys_exact_str = False
            continue
        safe_source[key] = value
    return safe_source, all_top_level_keys_exact_str


def _no_shadowing(source: dict[str, Any]) -> bool:
    invariant = source.get("invariant_authorization_guard")
    exe = source.get("exe_authorization_guard")
    boundaries = source.get("source_boundaries")
    if type(invariant) is dict and any(key in invariant for key in INVARIANT_CONTRACT_FIELDS_17_3):
        return False
    if type(exe) is dict and any(key in exe for key in EXE_CONTRACT_FIELDS_17_3):
        return False
    if type(boundaries) is dict and any(key in boundaries for key in SOURCE_BOUNDARY_FIELDS_17_3):
        return False
    return True


def _source_identity_valid(source: dict[str, Any]) -> bool:
    return list(source.keys()) == EXPECTED_SOURCE_TOP_LEVEL_FIELDS and all(
        _exact_plain_matches(source.get(k), v) for k, v in EXPECTED_IDENTITY.items()
    )


def _matrix_reference(source: dict[str, Any], accepted: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in [
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
        "status",
    ]:
        value = source.get(key)
        result[key] = value if type(value) in (str, int, bool) or value is None else None
    result["source_block_o_execution_authorization_matrix_step"] = "FUNCTIONAL-PREVIEW-17.2"
    result["source_matrix_read_by_17_3"] = True
    result["matrix_available_before_contract"] = True
    result["static_matrix_only"] = True
    result["execution_authorization_contract_built_by_17_3"] = True
    result["execution_authorization_contract_ready_by_17_3"] = accepted
    result["ready_for_functional_preview_17_4"] = accepted
    for name in [
        "source_state_recalculated",
        "readiness_recalculated",
        "live_condition_evaluated",
        "environment_inspected",
        "gate_evaluated",
        "gate_condition_accepted",
        "gate_opened",
        "gate_mutated",
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
        result[f"{name}_by_17_3"] = False
    return result


def _domain_contract_rows(
    domain_source_valid: bool, authorization_inputs_valid: bool, source: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = (
        _plain_list_section(source, "domain_authorization_rows")
        if domain_source_valid
        else EXPECTED_DOMAINS
    )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:2]):
        domain = (
            row.get("domain")
            if type(row) is dict and type(row.get("domain")) is str
            else ["packaging_release", "runtime_safety"][index]
        )
        out: dict[str, Any] = {
            "contract_row_id": f"{domain}_execution_authorization_contract_row",
            "source_matrix_row_id": row.get("matrix_row_id")
            if domain_source_valid and type(row) is dict
            else None,
            "domain": domain,
        }
        for source_key, target_key in [
            ("source_capability_count", "source_capability_count"),
            ("source_read_capability_count", "source_read_capability_count"),
            ("source_ready_capability_count", "source_ready_capability_count"),
            ("source_blocked_capability_count", "source_blocked_capability_count"),
            ("required_requirement_ids", "required_requirement_ids"),
            ("satisfied_requirement_ids", "satisfied_requirement_ids"),
            ("missing_requirement_ids", "missing_requirement_ids"),
            ("requirements_complete", "source_requirements_complete"),
            ("source_domain_ready", "source_domain_ready"),
            ("source_execution_authorized", "source_execution_authorized"),
            ("all_capabilities_read", "source_all_capabilities_read"),
            ("all_capabilities_not_ready", "source_all_capabilities_not_ready"),
            ("all_capabilities_blocked", "source_all_capabilities_blocked"),
            ("invariants_preserved", "source_invariants_preserved"),
            ("future_explicit_gate_required", "source_future_explicit_gate_required"),
            ("authorization_condition_met", "source_authorization_condition_met"),
            ("execution_authorized_by_matrix", "source_execution_authorized_by_matrix"),
        ]:
            out[target_key] = (
                _copy_plain(row.get(source_key))
                if domain_source_valid and type(row) is dict
                else (
                    [] if target_key.endswith("ids") else False if "count" not in target_key else 0
                )
            )
        out.update(
            {
                "contract_requires_all_requirements_satisfied": True,
                "contract_requires_invariants_preserved": True,
                "contract_requires_future_explicit_gate": True,
                "contract_condition_satisfied": False,
                "execution_authorized_by_contract": False,
                "failure_policy": "fail_closed",
                "contract_classification": "contracted_missing_required_conditions"
                if authorization_inputs_valid
                else "source_invalid",
                "contract_result": f"{domain}_execution_authorization_contracted_blocked"
                if authorization_inputs_valid
                else f"{domain}_source_invalid_execution_unauthorized",
            }
        )
        result.append(out)
    return result


def _requirement_contract_rows(valid: bool, source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _plain_list_section(source, "requirement_authorization_rows") if valid else []
    result: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {
            "contract_row_id": f"{row['requirement_id']}_execution_authorization_contract_row",
            "source_matrix_row_id": row["matrix_row_id"],
            "requirement_id": row["requirement_id"],
            "display_name": row["display_name"],
            "applicable_domains": _copy_plain(row["applicable_domains"]),
            "source_required": row["required"],
            "source_present": row["source_present"],
            "source_completed": row["source_completed"],
            "source_satisfied": row["source_satisfied"],
            "source_missing": row["missing"],
            "source_missing_blocks_execution": row["missing_blocks_execution"],
            "source_requires_future_explicit_step": row["requires_future_explicit_step"],
            "source_authorization_condition_met": row["authorization_condition_met"],
            "source_execution_authorized_by_matrix": row["execution_authorized_by_matrix"],
            "contract_requires_present": True,
            "contract_requires_completed": True,
            "contract_requires_satisfied": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "failure_policy": "fail_closed",
            "contract_classification": "contracted_missing_requirement",
            "contract_result": "missing_requirement_execution_authorization_contracted_blocked",
        }
        result.append(out)
    return result


def _invariant_contract(valid: bool, source: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = (
        _plain_dict_section(source, "invariant_authorization_guard") if valid else {}
    )
    result.update(
        {
            "read_by_execution_authorization_contract": valid,
            "recalculated_by_execution_authorization_contract": False,
            "contract_requires_all_invariants_preserved": True,
            "invariants_preserved_for_contract": valid,
            "invariants_alone_satisfy_contract": False,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "block_o_authorization_contract_result": "invariants_contracted_preserved_requirements_missing_execution_unauthorized"
            if valid
            else "invariant_source_invalid_execution_unauthorized",
        }
    )
    return result


def _exe_contract(valid: bool, source: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = _plain_dict_section(source, "exe_authorization_guard") if valid else {}
    result.update(
        {
            "read_by_execution_authorization_contract": valid,
            "recalculated_by_execution_authorization_contract": False,
            "contract_confirms_desktop_exe_direction": valid,
            "contract_is_not_build_authorization": True,
            "contract_is_not_packaging_authorization": True,
            "contract_is_not_release_authorization": True,
            "contract_condition_satisfied": False,
            "execution_authorized_by_contract": False,
            "block_o_authorization_contract_result": "desktop_exe_direction_contracted_build_packaging_release_unauthorized"
            if valid
            else "exe_source_invalid_build_packaging_release_unauthorized",
        }
    )
    return result


def _real_contract(valid: bool, source: dict[str, Any]) -> dict[str, Any]:
    status = (
        _copy_plain(source["real_capability_authorization_state"]["real_capability_status"])
        if valid
        else {}
    )
    return {
        "real_capability_status": status,
        "real_capability_status_inherited_from_17_2": valid,
        "real_capability_status_modified_by_17_3": False,
        "real_capabilities_opened_by_17_3": False,
        "all_real_capabilities_contracted_blocked": valid,
        "contract_condition_satisfied": False,
        "execution_authorized_by_contract": False,
        "contract_result": "real_capabilities_contracted_blocked_execution_unauthorized"
        if valid
        else "real_capabilities_source_invalid_execution_unauthorized",
    }


def build_preview_block_o_execution_authorization_contract() -> dict[str, Any]:
    source = build_preview_block_o_execution_authorization_matrix()
    raw_source = source if type(source) is dict else {}
    safe_source, all_top_level_keys_exact_str = _safe_top_level_source(raw_source)
    top_level_order_valid = (
        all_top_level_keys_exact_str and list(safe_source) == EXPECTED_SOURCE_TOP_LEVEL_FIELDS
    )
    raw_source_plain = _all_plain_json(raw_source, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    reference_valid = _section_valid(
        safe_source, "block_o_read_model_reference", EXPECTED_REFERENCE
    )
    summary_valid = _section_valid(safe_source, "matrix_summary", EXPECTED_SUMMARY)
    domain_rows_valid = _section_valid(safe_source, "domain_authorization_rows", EXPECTED_DOMAINS)
    requirement_rows_valid = _section_valid(
        safe_source, "requirement_authorization_rows", EXPECTED_REQUIREMENTS
    )
    invariant_valid = _section_valid(
        safe_source, "invariant_authorization_guard", EXPECTED_INVARIANT_GUARD
    )
    exe_valid = _section_valid(safe_source, "exe_authorization_guard", EXPECTED_EXE_GUARD)
    real_capability_valid = _section_valid(
        safe_source, "real_capability_authorization_state", EXPECTED_REAL_CAPABILITY_STATE
    )
    fail_closed_valid = _section_valid(
        safe_source, "fail_closed_matrix_decision", EXPECTED_FAIL_CLOSED_DECISION
    )
    evidence_valid = _section_valid(safe_source, "non_execution_matrix_evidence", EXPECTED_EVIDENCE)
    contract_boundaries_source_valid = _section_valid(
        safe_source, "matrix_boundaries", EXPECTED_MATRIX_BOUNDARIES
    )
    source_boundaries_valid = _section_valid(
        safe_source, "source_boundaries", EXPECTED_SOURCE_BOUNDARIES
    )
    future_steps_valid = _section_valid(safe_source, "future_steps", EXPECTED_FUTURE_STEPS)
    source_accepted = (
        _source_identity_valid(safe_source)
        and all_top_level_keys_exact_str
        and top_level_order_valid
        and raw_source_plain
        and reference_valid
        and summary_valid
        and domain_rows_valid
        and requirement_rows_valid
        and invariant_valid
        and exe_valid
        and real_capability_valid
        and fail_closed_valid
        and evidence_valid
        and contract_boundaries_source_valid
        and source_boundaries_valid
        and future_steps_valid
        and _no_shadowing(safe_source)
    )
    authorization_inputs_valid = domain_rows_valid and requirement_rows_valid and invariant_valid
    contract_status = CONTRACT_STATUS if source_accepted else BLOCKED_STATUS
    contract_decision = CONTRACT_DECISION if source_accepted else BLOCKED_STATUS.upper()
    real_contract = _real_contract(real_capability_valid, safe_source)
    fail: dict[str, Any] = {
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
        "block_o_execution_authorization_matrix_in_17_2": "preserved"
        if source_accepted
        else "not_preserved",
        "execution_authorization_contract_in_17_3": "ready" if source_accepted else "blocked",
        "execution_authorization_read_model_in_17_4": "allowed" if source_accepted else "blocked",
        "only_source_only_17_4_handoff_allowed": source_accepted,
        "execution_authorization_granted_by_17_3": False,
        "real_capability_status": _copy_plain(real_contract["real_capability_status"]),
        "real_capability_status_inherited_from_17_2": real_contract[
            "real_capability_status_inherited_from_17_2"
        ],
        "real_capability_status_modified_by_17_3": False,
    }
    evidence: dict[str, Any] = {
        "source_matrix_read": True,
        "execution_authorization_contract_built": True,
        "source_matrix_accepted": source_accepted,
        "block_o_remains_open": source_accepted,
        "reference_read_valid": reference_valid,
        "summary_read_valid": summary_valid,
        "domain_rows_read_valid": domain_rows_valid,
        "requirement_rows_read_valid": requirement_rows_valid,
        "invariant_read_valid": invariant_valid,
        "exe_read_valid": exe_valid,
        "real_capability_map_valid": real_capability_valid,
        "fail_closed_read_valid": fail_closed_valid,
        "matrix_boundaries_read_valid": contract_boundaries_source_valid,
        "source_boundaries_read_valid": source_boundaries_valid,
        "validation_performed_by_17_3": False,
        "authorization_granted_by_17_3": False,
        "runtime_started_by_17_3": False,
        "orders_enabled_by_17_3": False,
    }
    summary: dict[str, Any] = {
        "execution_authorization_contract_built": True,
        "contract_source_only": True,
        "contract_plain_data_only": True,
        "contract_static_only": True,
        "contract_read_only": True,
        "contract_non_evaluating": True,
        "contract_non_mutating": True,
        "contract_non_authorizing": True,
        "source_matrix_accepted": source_accepted,
        "block_o_remains_open": source_accepted,
        "ready_for_17_4": source_accepted,
        "only_source_only_handoff_allowed": source_accepted,
        "gates_confirmed_closed": source_accepted,
        "future_explicit_step_confirmed_required": source_accepted,
        "domain_contract_row_count": 2 if domain_rows_valid else 0,
        "requirement_contract_row_count": 7 if requirement_rows_valid else 0,
        "requirements_missing": requirement_rows_valid,
        "capabilities_blocked": domain_rows_valid,
        "real_capabilities_blocked": real_capability_valid,
        "invariants_preserved": invariant_valid,
        "desktop_exe_preserved": exe_valid,
        "all_contract_conditions_unsatisfied": domain_rows_valid
        and requirement_rows_valid
        and invariant_valid,
        "all_execution_authorizations_false": domain_rows_valid
        and requirement_rows_valid
        and invariant_valid
        and exe_valid
        and real_capability_valid,
    }
    boundaries: dict[str, Any] = {
        key: True
        for key in [
            "plain_data_source_only",
            "reads_17_2_only",
            "static_contract_only",
            "non_evaluating",
            "non_mutating",
            "non_authorizing",
            "cannot_inspect_environment",
            "cannot_evaluate_live_conditions",
            "cannot_accept_confirmation",
            "cannot_validate",
            "cannot_grant_authorization",
            "cannot_open_or_mutate_gate",
            "cannot_read_credentials",
            "cannot_use_network_or_filesystem",
            "cannot_package_build_or_release",
            "cannot_run_runtime",
            "cannot_generate_submit_cancel_or_replace_orders",
            "cannot_change_qml_bridge_gateway_controller",
            "can_feed_only_source_only_17_4_read_model",
        ]
    }
    src_boundaries: dict[str, Any] = (
        _plain_dict_section(safe_source, "source_boundaries") if source_boundaries_valid else {}
    )
    src_boundaries.update(
        {
            "source_block_o_execution_authorization_matrix": "FUNCTIONAL-PREVIEW-17.2",
            "matrix_source_preserved": source_accepted,
            "can_build_execution_authorization_contract": source_accepted,
            "can_feed_17_4": source_accepted,
        }
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_o_execution_authorization_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "execution_authorization_contract_status": contract_status,
        "execution_authorization_contract_decision": contract_decision,
        "execution_authorization_contract_ready": source_accepted,
        "ready_for_block_o_4": source_accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_o_execution_authorization_matrix_reference": _matrix_reference(
            safe_source, source_accepted
        ),
        "contract_summary": summary,
        "domain_authorization_contract_rows": _domain_contract_rows(
            domain_rows_valid, authorization_inputs_valid, safe_source
        ),
        "requirement_authorization_contract_rows": _requirement_contract_rows(
            requirement_rows_valid, safe_source
        ),
        "invariant_authorization_contract": _invariant_contract(invariant_valid, safe_source),
        "exe_authorization_contract": _exe_contract(exe_valid, safe_source),
        "real_capability_authorization_contract": real_contract,
        "fail_closed_contract_decision": fail,
        "non_execution_contract_evidence": evidence,
        "contract_boundaries": boundaries,
        "source_boundaries": src_boundaries,
        "future_steps": _copy_plain(FUTURE_STEPS),
        "status": STATUS if source_accepted else BLOCKED_STATUS,
    }
    return result
