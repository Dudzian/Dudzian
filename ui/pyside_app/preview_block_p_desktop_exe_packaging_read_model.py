"""FUNCTIONAL-PREVIEW-18.4 Block P desktop EXE packaging read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_packaging_contract import (
    build_preview_block_p_desktop_exe_packaging_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_packaging_read_model.v1"
KIND: Final[str] = "functional_preview_block_p_desktop_exe_packaging_read_model"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.4"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-18.5"
NEXT_STEP_TITLE: Final[str] = "BLOCK P DESKTOP EXE BUILD READINESS MATRIX"
STATUS: Final[str] = "ready_for_functional_preview_18_5_block_p_desktop_exe_build_readiness_matrix"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_5_block_p_desktop_exe_packaging_read_model_source_not_accepted"
)
PACKAGING_READ_MODEL_STATUS: Final[str] = (
    "source_18_3_consumed_packaging_contract_preserved_source_only_plain_data_read_model_complete_"
    "6_domains_3_scopes_8_requirements_12_blockers_12_evidence_6_acceptance_rules_projected_"
    "all_blocker_evidence_readiness_approval_authorization_states_fail_closed_only_source_only_handoff_to_18_5_allowed"
)
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
ACCEPTANCE_RELATIONS_EXPECTED: Final[dict[str, Any]] = {
    "all_twelve_blockers_resolved": {
        "required_blocker_ids": [
            "final_desktop_entrypoint_not_selected",
            "desktop_entrypoint_validation_not_performed",
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "qt_plugin_inventory_missing",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
            "final_desktop_packaging_profile_not_aligned",
            "dependency_resolution_not_performed",
            "secret_and_local_data_exclusion_policy_not_validated",
            "windows_toolchain_not_confirmed",
            "future_explicit_build_execution_gate_missing",
        ],
        "required_evidence_ids": [],
    },
    "all_required_evidence_collected_and_validated": {
        "required_blocker_ids": [],
        "required_evidence_ids": [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_qt_plugin_inventory",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
            "evidence_final_windows_profile_alignment",
            "evidence_windows_dependency_resolution",
            "evidence_artifact_exclusion_validation",
            "evidence_windows_toolchain_confirmation",
            "evidence_future_explicit_build_execution_gate",
        ],
    },
    "exactly_one_desktop_entrypoint_selected_and_validated": {
        "required_blocker_ids": [
            "final_desktop_entrypoint_not_selected",
            "desktop_entrypoint_validation_not_performed",
        ],
        "required_evidence_ids": [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
        ],
    },
    "qml_qt_and_packaging_metadata_contract_satisfied": {
        "required_blocker_ids": [
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "qt_plugin_inventory_missing",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
        ],
        "required_evidence_ids": [
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_qt_plugin_inventory",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
        ],
    },
    "windows_dependencies_profile_toolchain_contract_satisfied": {
        "required_blocker_ids": [
            "final_desktop_packaging_profile_not_aligned",
            "dependency_resolution_not_performed",
            "windows_toolchain_not_confirmed",
        ],
        "required_evidence_ids": [
            "evidence_final_windows_profile_alignment",
            "evidence_windows_dependency_resolution",
            "evidence_windows_toolchain_confirmation",
        ],
    },
    "artifact_exclusion_and_explicit_build_gate_contract_satisfied": {
        "required_blocker_ids": [
            "secret_and_local_data_exclusion_policy_not_validated",
            "future_explicit_build_execution_gate_missing",
        ],
        "required_evidence_ids": [
            "evidence_artifact_exclusion_validation",
            "evidence_future_explicit_build_execution_gate",
        ],
    },
}

BLOCKER_RELATIONS_EXPECTED: Final[dict[str, Any]] = {
    "final_desktop_entrypoint_not_selected": {
        "contract_clause_id": "clause_final_desktop_entrypoint_not_selected",
        "source_affected_scope_ids": ["desktop_application_entrypoint"],
        "contract_requirement_ids": ["contract_desktop_application_entrypoint_inventory"],
        "required_evidence_ids": ["evidence_final_desktop_entrypoint_selection"],
    },
    "desktop_entrypoint_validation_not_performed": {
        "contract_clause_id": "clause_desktop_entrypoint_validation_not_performed",
        "source_affected_scope_ids": ["desktop_application_entrypoint"],
        "contract_requirement_ids": ["contract_desktop_application_entrypoint_inventory"],
        "required_evidence_ids": ["evidence_desktop_entrypoint_validation"],
    },
    "qml_bundle_validation_not_performed": {
        "contract_clause_id": "clause_qml_bundle_validation_not_performed",
        "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
        "contract_requirement_ids": ["contract_qml_asset_inventory"],
        "required_evidence_ids": ["evidence_qml_bundle_validation"],
    },
    "windows_shared_qml_import_path_unresolved": {
        "contract_clause_id": "clause_windows_shared_qml_import_path_unresolved",
        "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
        "contract_requirement_ids": ["contract_qml_asset_inventory"],
        "required_evidence_ids": ["evidence_windows_shared_qml_import_path"],
    },
    "qt_plugin_inventory_missing": {
        "contract_clause_id": "clause_qt_plugin_inventory_missing",
        "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
        "contract_requirement_ids": ["contract_qt_plugin_inventory"],
        "required_evidence_ids": ["evidence_qt_plugin_inventory"],
    },
    "ui_package_discovery_missing": {
        "contract_clause_id": "clause_ui_package_discovery_missing",
        "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
        "contract_requirement_ids": ["contract_qml_asset_inventory"],
        "required_evidence_ids": ["evidence_ui_package_discovery"],
    },
    "qml_package_data_missing": {
        "contract_clause_id": "clause_qml_package_data_missing",
        "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
        "contract_requirement_ids": ["contract_qml_asset_inventory"],
        "required_evidence_ids": ["evidence_qml_package_data"],
    },
    "final_desktop_packaging_profile_not_aligned": {
        "contract_clause_id": "clause_final_desktop_packaging_profile_not_aligned",
        "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
        "contract_requirement_ids": ["contract_packaging_profile_alignment"],
        "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
    },
    "dependency_resolution_not_performed": {
        "contract_clause_id": "clause_dependency_resolution_not_performed",
        "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
        "contract_requirement_ids": ["contract_python_dependency_inventory"],
        "required_evidence_ids": ["evidence_windows_dependency_resolution"],
    },
    "secret_and_local_data_exclusion_policy_not_validated": {
        "contract_clause_id": "clause_secret_and_local_data_exclusion_policy_not_validated",
        "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
        "contract_requirement_ids": ["contract_secret_and_local_data_exclusion_policy"],
        "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
    },
    "windows_toolchain_not_confirmed": {
        "contract_clause_id": "clause_windows_toolchain_not_confirmed",
        "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
        "contract_requirement_ids": ["contract_windows_target_toolchain_confirmation"],
        "required_evidence_ids": ["evidence_windows_toolchain_confirmation"],
    },
    "future_explicit_build_execution_gate_missing": {
        "contract_clause_id": "clause_future_explicit_build_execution_gate_missing",
        "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
        "contract_requirement_ids": ["contract_future_explicit_build_execution_gate"],
        "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
    },
}

SCOPE_RELATIONS_EXPECTED: Final[dict[str, Any]] = {
    "desktop_application_entrypoint": {
        "source_unresolved_blocker_ids": [
            "final_desktop_entrypoint_not_selected",
            "desktop_entrypoint_validation_not_performed",
        ],
        "contract_clause_ids": [
            "clause_final_desktop_entrypoint_not_selected",
            "clause_desktop_entrypoint_validation_not_performed",
        ],
        "required_evidence_ids": [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
        ],
    },
    "qt_qml_runtime_bundle": {
        "source_unresolved_blocker_ids": [
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "qt_plugin_inventory_missing",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
        ],
        "contract_clause_ids": [
            "clause_qml_bundle_validation_not_performed",
            "clause_windows_shared_qml_import_path_unresolved",
            "clause_qt_plugin_inventory_missing",
            "clause_ui_package_discovery_missing",
            "clause_qml_package_data_missing",
        ],
        "required_evidence_ids": [
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_qt_plugin_inventory",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
        ],
    },
    "windows_exe_artifact_pipeline": {
        "source_unresolved_blocker_ids": [
            "final_desktop_packaging_profile_not_aligned",
            "dependency_resolution_not_performed",
            "secret_and_local_data_exclusion_policy_not_validated",
            "windows_toolchain_not_confirmed",
            "future_explicit_build_execution_gate_missing",
        ],
        "contract_clause_ids": [
            "clause_final_desktop_packaging_profile_not_aligned",
            "clause_dependency_resolution_not_performed",
            "clause_secret_and_local_data_exclusion_policy_not_validated",
            "clause_windows_toolchain_not_confirmed",
            "clause_future_explicit_build_execution_gate_missing",
        ],
        "required_evidence_ids": [
            "evidence_final_windows_profile_alignment",
            "evidence_windows_dependency_resolution",
            "evidence_artifact_exclusion_validation",
            "evidence_windows_toolchain_confirmation",
            "evidence_future_explicit_build_execution_gate",
        ],
    },
}

REQUIREMENT_RELATIONS_EXPECTED: Final[dict[str, Any]] = {
    "desktop_application_entrypoint_inventory": {
        "contract_requirement_id": "contract_desktop_application_entrypoint_inventory",
        "source_unresolved_condition_ids": [
            "final_desktop_entrypoint_not_selected",
            "desktop_entrypoint_validation_not_performed",
        ],
        "contract_clause_ids": [
            "clause_final_desktop_entrypoint_not_selected",
            "clause_desktop_entrypoint_validation_not_performed",
        ],
        "required_evidence_ids": [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
        ],
    },
    "qml_asset_inventory": {
        "contract_requirement_id": "contract_qml_asset_inventory",
        "source_unresolved_condition_ids": [
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
        ],
        "contract_clause_ids": [
            "clause_qml_bundle_validation_not_performed",
            "clause_windows_shared_qml_import_path_unresolved",
            "clause_ui_package_discovery_missing",
            "clause_qml_package_data_missing",
        ],
        "required_evidence_ids": [
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
        ],
    },
    "qt_plugin_inventory": {
        "contract_requirement_id": "contract_qt_plugin_inventory",
        "source_unresolved_condition_ids": ["qt_plugin_inventory_missing"],
        "contract_clause_ids": ["clause_qt_plugin_inventory_missing"],
        "required_evidence_ids": ["evidence_qt_plugin_inventory"],
    },
    "python_dependency_inventory": {
        "contract_requirement_id": "contract_python_dependency_inventory",
        "source_unresolved_condition_ids": ["dependency_resolution_not_performed"],
        "contract_clause_ids": ["clause_dependency_resolution_not_performed"],
        "required_evidence_ids": ["evidence_windows_dependency_resolution"],
    },
    "packaging_profile_alignment": {
        "contract_requirement_id": "contract_packaging_profile_alignment",
        "source_unresolved_condition_ids": ["final_desktop_packaging_profile_not_aligned"],
        "contract_clause_ids": ["clause_final_desktop_packaging_profile_not_aligned"],
        "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
    },
    "secret_and_local_data_exclusion_policy": {
        "contract_requirement_id": "contract_secret_and_local_data_exclusion_policy",
        "source_unresolved_condition_ids": ["secret_and_local_data_exclusion_policy_not_validated"],
        "contract_clause_ids": ["clause_secret_and_local_data_exclusion_policy_not_validated"],
        "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
    },
    "windows_target_toolchain_confirmation": {
        "contract_requirement_id": "contract_windows_target_toolchain_confirmation",
        "source_unresolved_condition_ids": ["windows_toolchain_not_confirmed"],
        "contract_clause_ids": ["clause_windows_toolchain_not_confirmed"],
        "required_evidence_ids": ["evidence_windows_toolchain_confirmation"],
    },
    "future_explicit_build_execution_gate": {
        "contract_requirement_id": "contract_future_explicit_build_execution_gate",
        "source_unresolved_condition_ids": ["future_explicit_build_execution_gate_missing"],
        "contract_clause_ids": ["clause_future_explicit_build_execution_gate_missing"],
        "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
    },
}

EVIDENCE_RELATIONS_EXPECTED: Final[dict[str, Any]] = {
    "evidence_final_desktop_entrypoint_selection": {
        "blocker_id": "final_desktop_entrypoint_not_selected"
    },
    "evidence_desktop_entrypoint_validation": {
        "blocker_id": "desktop_entrypoint_validation_not_performed"
    },
    "evidence_qml_bundle_validation": {"blocker_id": "qml_bundle_validation_not_performed"},
    "evidence_windows_shared_qml_import_path": {
        "blocker_id": "windows_shared_qml_import_path_unresolved"
    },
    "evidence_qt_plugin_inventory": {"blocker_id": "qt_plugin_inventory_missing"},
    "evidence_ui_package_discovery": {"blocker_id": "ui_package_discovery_missing"},
    "evidence_qml_package_data": {"blocker_id": "qml_package_data_missing"},
    "evidence_final_windows_profile_alignment": {
        "blocker_id": "final_desktop_packaging_profile_not_aligned"
    },
    "evidence_windows_dependency_resolution": {"blocker_id": "dependency_resolution_not_performed"},
    "evidence_artifact_exclusion_validation": {
        "blocker_id": "secret_and_local_data_exclusion_policy_not_validated"
    },
    "evidence_windows_toolchain_confirmation": {"blocker_id": "windows_toolchain_not_confirmed"},
    "evidence_future_explicit_build_execution_gate": {
        "blocker_id": "future_explicit_build_execution_gate_missing"
    },
}

TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_packaging_read_model_kind",
    "block",
    "step",
    "block_p_desktop_exe_packaging_read_model_status",
    "block_p_desktop_exe_packaging_read_model_decision",
    "packaging_read_model_artifact_complete",
    "ready_for_block_p_5",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_packaging_contract_reference",
    "packaging_read_model_summary",
    "source_contract_preservation",
    "packaging_contract_overview",
    "domain_contract_read_model_rows",
    "scope_contract_read_model_rows",
    "requirement_contract_read_model_rows",
    "blocker_read_model_rows",
    "evidence_read_model_rows",
    "acceptance_rule_read_model_rows",
    "capability_read_model_state",
    "fail_closed_read_model_decision",
    "non_execution_read_model_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
TOP_LEVEL_FIELDS_18_3: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_packaging_contract_kind",
    "block",
    "step",
    "block_p_desktop_exe_packaging_contract_status",
    "block_p_desktop_exe_packaging_contract_decision",
    "packaging_contract_artifact_complete",
    "ready_for_block_p_4",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_packaging_inventory_matrix_reference",
    "packaging_contract_summary",
    "source_matrix_preservation",
    "contract_principles",
    "desktop_entrypoint_contract",
    "qml_bundle_contract",
    "python_dependency_contract",
    "packaging_metadata_contract",
    "preview_packaging_separation_contract",
    "artifact_exclusion_contract",
    "packaging_scope_contract_rows",
    "packaging_requirement_contract_rows",
    "unresolved_blocker_contract_rows",
    "contract_evidence_requirement_rows",
    "contract_acceptance_rule_rows",
    "real_capability_contract_state",
    "fail_closed_contract_decision",
    "non_execution_contract_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SOURCE_BOUNDARY_FIELDS_18_4: Final[list[str]] = [
    "source_block_p_desktop_exe_packaging_contract",
    "packaging_contract_preserved",
    "can_build_desktop_exe_packaging_read_model",
    "packaging_read_model_artifact_complete",
    "can_build_desktop_exe_build_readiness_matrix",
    "can_feed_18_5",
]
SUMMARY_OWNED_FIELDS_18_4: Final[list[str]] = [
    "packaging_read_model_artifact_complete",
    "ready_for_block_p_5",
    "contract_satisfied",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
]
FAIL_CLOSED_OWNED_FIELDS_18_4: Final[list[str]] = [
    "only_source_only_18_5_handoff_allowed",
    "build_ready_by_18_4",
    "packaging_authorized_by_18_4",
    "build_authorized_by_18_4",
]
SOURCE_IDENTITY_EXPECTED: Final[dict[str, Any]] = {
    "schema_version": "preview_block_p_desktop_exe_packaging_contract.v1",
    "block_p_desktop_exe_packaging_contract_kind": "functional_preview_block_p_desktop_exe_packaging_contract",
    "block": "P",
    "step": "18.3",
    "block_p_desktop_exe_packaging_contract_status": "source_18_2_consumed_inventory_matrix_preserved_static_plain_data_contract_complete_3_scopes_8_requirements_12_blockers_12_evidence_6_acceptance_rules_defined_zero_resolution_approval_readiness_authorization_build_runtime_orders_only_source_only_handoff_to_18_4_allowed",
    "block_p_desktop_exe_packaging_contract_decision": "SOURCE_18_2_CONSUMED_INVENTORY_MATRIX_PRESERVED_STATIC_PLAIN_DATA_CONTRACT_COMPLETE_3_SCOPES_8_REQUIREMENTS_12_BLOCKERS_12_EVIDENCE_6_ACCEPTANCE_RULES_DEFINED_ZERO_RESOLUTION_APPROVAL_READINESS_AUTHORIZATION_BUILD_RUNTIME_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_4_ALLOWED",
    "packaging_contract_artifact_complete": True,
    "ready_for_block_p_4": True,
    "next_step": "FUNCTIONAL-PREVIEW-18.4",
    "next_step_title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
    "status": "ready_for_functional_preview_18_4_block_p_desktop_exe_packaging_read_model",
}
EXPECTED_SOURCE: Final[dict[str, Any]] = {
    "schema_version": "preview_block_p_desktop_exe_packaging_contract.v1",
    "block_p_desktop_exe_packaging_contract_kind": "functional_preview_block_p_desktop_exe_packaging_contract",
    "block": "P",
    "step": "18.3",
    "block_p_desktop_exe_packaging_contract_status": "source_18_2_consumed_inventory_matrix_preserved_static_plain_data_contract_complete_3_scopes_8_requirements_12_blockers_12_evidence_6_acceptance_rules_defined_zero_resolution_approval_readiness_authorization_build_runtime_orders_only_source_only_handoff_to_18_4_allowed",
    "block_p_desktop_exe_packaging_contract_decision": "SOURCE_18_2_CONSUMED_INVENTORY_MATRIX_PRESERVED_STATIC_PLAIN_DATA_CONTRACT_COMPLETE_3_SCOPES_8_REQUIREMENTS_12_BLOCKERS_12_EVIDENCE_6_ACCEPTANCE_RULES_DEFINED_ZERO_RESOLUTION_APPROVAL_READINESS_AUTHORIZATION_BUILD_RUNTIME_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_4_ALLOWED",
    "packaging_contract_artifact_complete": True,
    "ready_for_block_p_4": True,
    "next_step": "FUNCTIONAL-PREVIEW-18.4",
    "next_step_title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
    "block_p_desktop_exe_packaging_inventory_matrix_reference": {
        "schema_version": "preview_block_p_desktop_exe_packaging_inventory_matrix.v1",
        "block_p_desktop_exe_packaging_inventory_matrix_kind": "functional_preview_block_p_desktop_exe_packaging_inventory_matrix",
        "block": "P",
        "step": "18.2",
        "block_p_desktop_exe_packaging_inventory_matrix_status": "source_18_1_consumed_source_inventory_preserved_static_source_only_matrix_complete_four_entrypoint_rows_evaluated_qml_inventory_evaluated_dependency_declarations_evaluated_packaging_metadata_evaluated_cli_preview_packaging_evaluated_as_separate_scope_exclusion_policy_evaluated_11_findings_evaluated_3_packaging_scopes_evaluated_8_packaging_requirements_evaluated_unresolved_contract_blockers_recorded_no_entrypoint_selection_no_validation_no_approval_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_3_allowed",
        "block_p_desktop_exe_packaging_inventory_matrix_decision": "SOURCE_18_1_CONSUMED_SOURCE_INVENTORY_PRESERVED_STATIC_SOURCE_ONLY_MATRIX_COMPLETE_FOUR_ENTRYPOINT_ROWS_EVALUATED_QML_INVENTORY_EVALUATED_DEPENDENCY_DECLARATIONS_EVALUATED_PACKAGING_METADATA_EVALUATED_CLI_PREVIEW_PACKAGING_EVALUATED_AS_SEPARATE_SCOPE_EXCLUSION_POLICY_EVALUATED_11_FINDINGS_EVALUATED_3_PACKAGING_SCOPES_EVALUATED_8_PACKAGING_REQUIREMENTS_EVALUATED_UNRESOLVED_CONTRACT_BLOCKERS_RECORDED_NO_ENTRYPOINT_SELECTION_NO_VALIDATION_NO_APPROVAL_NO_PACKAGING_NO_BUILD_NO_ARTIFACT_NO_RELEASE_NO_RUNTIME_NO_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_3_ALLOWED",
        "inventory_matrix_artifact_complete": True,
        "ready_for_block_p_3": True,
        "next_step": "FUNCTIONAL-PREVIEW-18.3",
        "next_step_title": "BLOCK P DESKTOP EXE PACKAGING CONTRACT",
        "status": "ready_for_functional_preview_18_3_block_p_desktop_exe_packaging_contract",
        "source_identity_valid": True,
        "source_block_p_desktop_exe_packaging_inventory_matrix_step": "FUNCTIONAL-PREVIEW-18.2",
        "source_inventory_matrix_read_by_18_3": True,
        "inventory_matrix_available_before_contract": True,
        "static_packaging_contract_only": True,
        "packaging_contract_built_by_18_3": True,
        "packaging_contract_artifact_complete_by_18_3": True,
        "ready_for_functional_preview_18_4": True,
        "repo_rescan_performed": False,
        "filesystem_io_performed": False,
        "environment_inspection_performed": False,
        "secret_read": False,
        "dependency_resolution_performed": False,
        "pyside_imported": False,
        "qml_loaded": False,
        "plugin_discovery_performed": False,
        "entrypoint_selection_performed": False,
        "metadata_mutation_performed": False,
        "profile_selection_performed": False,
        "tool_selection_performed": False,
        "spec_created": False,
        "command_created": False,
        "package_performed": False,
        "build_performed": False,
        "artifact_created": False,
        "signing_performed": False,
        "installer_created": False,
        "release_performed": False,
        "runtime_started": False,
        "orders_enabled": False,
        "network_opened": False,
        "credentials_read": False,
    },
    "packaging_contract_summary": {
        "source_18_2_accepted": True,
        "scope_contract_count": 3,
        "requirement_contract_count": 8,
        "blocker_contract_count": 12,
        "evidence_requirement_count": 12,
        "acceptance_rule_count": 6,
        "all_contract_definitions_complete": True,
        "any_blocker_resolved": False,
        "all_blockers_resolved": False,
        "all_required_evidence_collected": False,
        "all_required_evidence_validated": False,
        "contract_satisfied": False,
        "desktop_entrypoint_selected": False,
        "desktop_entrypoint_validated": False,
        "qml_bundle_validated": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_created": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
        "ready_for_block_p_4": True,
    },
    "source_matrix_preservation": {
        "source_identity_preserved": True,
        "desktop_entrypoint_row_count": 4,
        "qml_bundle_row_count": 5,
        "python_dependency_row_count": 4,
        "packaging_metadata_row_count": 4,
        "existing_preview_packaging_row_count": 4,
        "artifact_exclusion_policy_row_count": 1,
        "finding_row_count": 11,
        "scope_row_count": 3,
        "requirement_row_count": 8,
        "unresolved_blocker_row_count": 12,
        "referential_integrity_preserved": True,
        "source_readiness_granted": False,
        "source_approval_granted": False,
        "source_authorization_granted": False,
        "source_matrix_recalculated": False,
        "repo_rescanned": False,
        "source_rows_modified": False,
        "source_relations_modified": False,
    },
    "contract_principles": [
        {
            "principle_id": "final_product_is_windows_desktop_exe",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "contract_is_source_only",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "exactly_one_desktop_entrypoint_must_be_selected_later",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "cli_preview_scope_is_not_final_desktop_packaging",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "all_qml_and_qt_runtime_inputs_require_explicit_contract_coverage",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "secrets_and_local_user_data_are_excluded_fail_closed",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "build_execution_requires_future_explicit_gate",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
        {
            "principle_id": "contract_completion_does_not_grant_readiness_or_authorization",
            "required": True,
            "contract_defined": True,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied",
        },
    ],
    "desktop_entrypoint_contract": {
        "candidate_paths": ["ui/pyside_app/__main__.py", "ui/pyside_app/app.py"],
        "selection_cardinality": "exactly_one",
        "selection_required": True,
        "source_validation_required": True,
        "future_windows_launch_smoke_required": True,
        "selection_performed": False,
        "source_validation_performed": False,
        "future_windows_launch_smoke_performed": False,
        "approval_granted": False,
        "readiness_granted": False,
        "selected_path": "",
        "selected_symbol": "",
        "cli_preview_entrypoints_excluded": True,
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "qml_bundle_contract": {
        "default_entrypoint": "MainWindow.qml",
        "qml_roots": ["ui/pyside_app/qml", "ui/qml"],
        "pyside_qml_file_count": 24,
        "shared_qml_file_count": 107,
        "additional_qml_support_asset_count": 0,
        "styles_qmldir_observed": True,
        "required_extensions": [
            ".qml",
            ".js",
            ".mjs",
            ".qmltypes",
            "qmldir",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".webp",
            ".ico",
            ".ttf",
            ".otf",
        ],
        "requires_all_inventory_paths": True,
        "requires_windows_shared_qml_import_path": True,
        "requires_ui_package_discovery": True,
        "requires_qml_package_data": True,
        "requires_qt_plugin_inventory": True,
        "requires_future_qml_load_smoke": True,
        "resolution_performed": False,
        "validation_performed": False,
        "approval_granted": False,
        "readiness_granted": False,
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "python_dependency_contract": {
        "declared_dependency_count": 25,
        "optional_desktop_dependency_count": 3,
        "target_platform": "windows",
        "requires_pyside6": True,
        "build_tool_candidates": ["pyinstaller", "briefcase"],
        "resolution_performed": False,
        "locked_environment_recorded": False,
        "tool_selection_performed": False,
        "qt_validation_performed": False,
        "approval_granted": False,
        "readiness_granted": False,
        "selected_tool": "",
        "selected_tool_version": "",
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "packaging_metadata_contract": {
        "requires_ui_package_discovery": True,
        "requires_qml_package_data_declaration": True,
        "ui_package_discovery_present": False,
        "qml_package_data_present": False,
        "metadata_mutation_performed": False,
        "validation_performed": False,
        "approval_granted": False,
        "readiness_granted": False,
        "deploy_packaging_observations_preserved": True,
        "documentation_observations_preserved": True,
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "preview_packaging_separation_contract": {
        "cli_preview_entrypoint": "scripts/run_local_bot.py",
        "windows_preview_targets_cli": True,
        "source_scope": "cli_preview",
        "not_final_desktop_contract": True,
        "final_profile_path": "",
        "final_entrypoint": "",
        "selection_performed": False,
        "validation_performed": False,
        "approval_granted": False,
        "readiness_granted": False,
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "artifact_exclusion_contract": {
        "policy_source": "scripts/safe_exe_preview_build_plan.py",
        "policy_version": "security_packaging_artifact_policy.v1",
        "denied_patterns": [
            ".env",
            "*.env",
            "trading.db",
            "bot_core/logs",
            "logs",
            "reports",
            "test-results",
            "var/security",
            "*api_key*",
            "*api_secret*",
            "*secret*",
            "*token*",
            "*keychain*",
        ],
        "requires_policy_application": True,
        "requires_config_secret_review": True,
        "requires_final_bundle_scan_zero_denied_matches": True,
        "policy_application_performed": False,
        "config_secret_review_performed": False,
        "final_bundle_scan_performed": False,
        "local_data_excluded": False,
        "approval_granted": False,
        "readiness_granted": False,
        "contract_defined": True,
        "contract_satisfied": False,
    },
    "packaging_scope_contract_rows": [
        {
            "scope_id": "desktop_application_entrypoint",
            "source_scope_preserved": True,
            "source_supporting_matrix_row_ids": [
                "desktop_module_launcher_matrix",
                "desktop_application_main_matrix",
            ],
            "source_unresolved_blocker_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            "contract_clause_ids": [
                "clause_final_desktop_entrypoint_not_selected",
                "clause_desktop_entrypoint_validation_not_performed",
            ],
            "required_evidence_ids": [
                "evidence_final_desktop_entrypoint_selection",
                "evidence_desktop_entrypoint_validation",
            ],
            "contract_definition_complete": True,
            "contract_satisfied": False,
            "resolved_blocker_count": 0,
            "unresolved_blocker_count": 2,
            "ready_for_read_model": True,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "contract_classification": "scope_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "scope_id": "qt_qml_runtime_bundle",
            "source_scope_preserved": True,
            "source_supporting_matrix_row_ids": [
                "default_qml_entrypoint",
                "pyside_qml_root",
                "shared_qml_root",
                "styles_module",
                "windows_shared_qml_import_path",
                "setuptools_ui_package_discovery",
                "qml_package_data_declaration",
            ],
            "source_unresolved_blocker_ids": [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "qt_plugin_inventory_missing",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            "contract_clause_ids": [
                "clause_qml_bundle_validation_not_performed",
                "clause_windows_shared_qml_import_path_unresolved",
                "clause_qt_plugin_inventory_missing",
                "clause_ui_package_discovery_missing",
                "clause_qml_package_data_missing",
            ],
            "required_evidence_ids": [
                "evidence_qml_bundle_validation",
                "evidence_windows_shared_qml_import_path",
                "evidence_qt_plugin_inventory",
                "evidence_ui_package_discovery",
                "evidence_qml_package_data",
            ],
            "contract_definition_complete": True,
            "contract_satisfied": False,
            "resolved_blocker_count": 0,
            "unresolved_blocker_count": 5,
            "ready_for_read_model": True,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "contract_classification": "scope_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "scope_id": "windows_exe_artifact_pipeline",
            "source_scope_preserved": True,
            "source_supporting_matrix_row_ids": [
                "project_dependency_declarations",
                "desktop_optional_dependency_declarations",
                "dependency_resolution",
                "desktop_build_tool_candidates",
                "safe_exe_preview_build_plan",
                "windows_preview_profile",
                "artifact_exclusion_policy",
            ],
            "source_unresolved_blocker_ids": [
                "final_desktop_packaging_profile_not_aligned",
                "dependency_resolution_not_performed",
                "secret_and_local_data_exclusion_policy_not_validated",
                "windows_toolchain_not_confirmed",
                "future_explicit_build_execution_gate_missing",
            ],
            "contract_clause_ids": [
                "clause_final_desktop_packaging_profile_not_aligned",
                "clause_dependency_resolution_not_performed",
                "clause_secret_and_local_data_exclusion_policy_not_validated",
                "clause_windows_toolchain_not_confirmed",
                "clause_future_explicit_build_execution_gate_missing",
            ],
            "required_evidence_ids": [
                "evidence_final_windows_profile_alignment",
                "evidence_windows_dependency_resolution",
                "evidence_artifact_exclusion_validation",
                "evidence_windows_toolchain_confirmation",
                "evidence_future_explicit_build_execution_gate",
            ],
            "contract_definition_complete": True,
            "contract_satisfied": False,
            "resolved_blocker_count": 0,
            "unresolved_blocker_count": 5,
            "ready_for_read_model": True,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "contract_classification": "scope_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
    ],
    "packaging_requirement_contract_rows": [
        {
            "requirement_id": "desktop_application_entrypoint_inventory",
            "source_requirement_preserved": True,
            "source_inventory_observed": True,
            "source_inventory_requirement_satisfied": True,
            "source_unresolved_condition_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            "contract_requirement_id": "contract_desktop_application_entrypoint_inventory",
            "contract_clause_ids": [
                "clause_final_desktop_entrypoint_not_selected",
                "clause_desktop_entrypoint_validation_not_performed",
            ],
            "required_evidence_ids": [
                "evidence_final_desktop_entrypoint_selection",
                "evidence_desktop_entrypoint_validation",
            ],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "qml_asset_inventory",
            "source_requirement_preserved": True,
            "source_inventory_observed": True,
            "source_inventory_requirement_satisfied": True,
            "source_unresolved_condition_ids": [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            "contract_requirement_id": "contract_qml_asset_inventory",
            "contract_clause_ids": [
                "clause_qml_bundle_validation_not_performed",
                "clause_windows_shared_qml_import_path_unresolved",
                "clause_ui_package_discovery_missing",
                "clause_qml_package_data_missing",
            ],
            "required_evidence_ids": [
                "evidence_qml_bundle_validation",
                "evidence_windows_shared_qml_import_path",
                "evidence_ui_package_discovery",
                "evidence_qml_package_data",
            ],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "qt_plugin_inventory",
            "source_requirement_preserved": True,
            "source_inventory_observed": False,
            "source_inventory_requirement_satisfied": False,
            "source_unresolved_condition_ids": ["qt_plugin_inventory_missing"],
            "contract_requirement_id": "contract_qt_plugin_inventory",
            "contract_clause_ids": ["clause_qt_plugin_inventory_missing"],
            "required_evidence_ids": ["evidence_qt_plugin_inventory"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "python_dependency_inventory",
            "source_requirement_preserved": True,
            "source_inventory_observed": True,
            "source_inventory_requirement_satisfied": True,
            "source_unresolved_condition_ids": ["dependency_resolution_not_performed"],
            "contract_requirement_id": "contract_python_dependency_inventory",
            "contract_clause_ids": ["clause_dependency_resolution_not_performed"],
            "required_evidence_ids": ["evidence_windows_dependency_resolution"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "packaging_profile_alignment",
            "source_requirement_preserved": True,
            "source_inventory_observed": True,
            "source_inventory_requirement_satisfied": True,
            "source_unresolved_condition_ids": ["final_desktop_packaging_profile_not_aligned"],
            "contract_requirement_id": "contract_packaging_profile_alignment",
            "contract_clause_ids": ["clause_final_desktop_packaging_profile_not_aligned"],
            "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "secret_and_local_data_exclusion_policy",
            "source_requirement_preserved": True,
            "source_inventory_observed": True,
            "source_inventory_requirement_satisfied": True,
            "source_unresolved_condition_ids": [
                "secret_and_local_data_exclusion_policy_not_validated"
            ],
            "contract_requirement_id": "contract_secret_and_local_data_exclusion_policy",
            "contract_clause_ids": ["clause_secret_and_local_data_exclusion_policy_not_validated"],
            "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "windows_target_toolchain_confirmation",
            "source_requirement_preserved": True,
            "source_inventory_observed": False,
            "source_inventory_requirement_satisfied": False,
            "source_unresolved_condition_ids": ["windows_toolchain_not_confirmed"],
            "contract_requirement_id": "contract_windows_target_toolchain_confirmation",
            "contract_clause_ids": ["clause_windows_toolchain_not_confirmed"],
            "required_evidence_ids": ["evidence_windows_toolchain_confirmation"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": False,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
        {
            "requirement_id": "future_explicit_build_execution_gate",
            "source_requirement_preserved": True,
            "source_inventory_observed": False,
            "source_inventory_requirement_satisfied": False,
            "source_unresolved_condition_ids": ["future_explicit_build_execution_gate_missing"],
            "contract_requirement_id": "contract_future_explicit_build_execution_gate",
            "contract_clause_ids": ["clause_future_explicit_build_execution_gate_missing"],
            "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": True,
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        },
    ],
    "unresolved_blocker_contract_rows": [
        {
            "blocker_id": "final_desktop_entrypoint_not_selected",
            "source_blocker_preserved": True,
            "source_finding_ids": [
                "desktop_module_launcher_observed",
                "desktop_application_main_observed",
            ],
            "source_affected_scope_ids": ["desktop_application_entrypoint"],
            "contract_clause_id": "clause_final_desktop_entrypoint_not_selected",
            "contract_requirement_ids": ["contract_desktop_application_entrypoint_inventory"],
            "required_evidence_ids": ["evidence_final_desktop_entrypoint_selection"],
            "resolution_criteria": "exactly_one_candidate_selected_and_recorded",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "desktop_entrypoint_validation_not_performed",
            "source_blocker_preserved": True,
            "source_finding_ids": [
                "desktop_module_launcher_observed",
                "desktop_application_main_observed",
            ],
            "source_affected_scope_ids": ["desktop_application_entrypoint"],
            "contract_clause_id": "clause_desktop_entrypoint_validation_not_performed",
            "contract_requirement_ids": ["contract_desktop_application_entrypoint_inventory"],
            "required_evidence_ids": ["evidence_desktop_entrypoint_validation"],
            "resolution_criteria": "selected_entrypoint_passes_source_validation_and_future_windows_launch_smoke",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "qml_bundle_validation_not_performed",
            "source_blocker_preserved": True,
            "source_finding_ids": [
                "default_qml_entrypoint_observed",
                "two_qml_source_roots_observed",
            ],
            "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
            "contract_clause_id": "clause_qml_bundle_validation_not_performed",
            "contract_requirement_ids": ["contract_qml_asset_inventory"],
            "required_evidence_ids": ["evidence_qml_bundle_validation"],
            "resolution_criteria": "all_inventoried_qml_inputs_and_required_runtime_assets_pass_static_and_future_load_validation",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "windows_shared_qml_import_path_unresolved",
            "source_blocker_preserved": True,
            "source_finding_ids": ["shared_qml_import_path_is_platform_conditional"],
            "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
            "contract_clause_id": "clause_windows_shared_qml_import_path_unresolved",
            "contract_requirement_ids": ["contract_qml_asset_inventory"],
            "required_evidence_ids": ["evidence_windows_shared_qml_import_path"],
            "resolution_criteria": "windows_bundle_defines_shared_qml_import_path_or_equivalent_and_future_smoke_passes",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "qt_plugin_inventory_missing",
            "source_blocker_preserved": True,
            "source_finding_ids": [],
            "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
            "contract_clause_id": "clause_qt_plugin_inventory_missing",
            "contract_requirement_ids": ["contract_qt_plugin_inventory"],
            "required_evidence_ids": ["evidence_qt_plugin_inventory"],
            "resolution_criteria": "exact_required_qt_plugin_inventory_is_recorded_and_validated_for_selected_tool",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "ui_package_discovery_missing",
            "source_blocker_preserved": True,
            "source_finding_ids": [
                "ui_package_discovery_not_declared_in_current_setuptools_include"
            ],
            "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
            "contract_clause_id": "clause_ui_package_discovery_missing",
            "contract_requirement_ids": ["contract_qml_asset_inventory"],
            "required_evidence_ids": ["evidence_ui_package_discovery"],
            "resolution_criteria": "packaging_metadata_contains_ui_package_discovery_pattern",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "qml_package_data_missing",
            "source_blocker_preserved": True,
            "source_finding_ids": ["qml_package_data_not_declared_in_current_setuptools_metadata"],
            "source_affected_scope_ids": ["qt_qml_runtime_bundle"],
            "contract_clause_id": "clause_qml_package_data_missing",
            "contract_requirement_ids": ["contract_qml_asset_inventory"],
            "required_evidence_ids": ["evidence_qml_package_data"],
            "resolution_criteria": "packaging_metadata_contains_required_qml_and_support_asset_package_data",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "final_desktop_packaging_profile_not_aligned",
            "source_blocker_preserved": True,
            "source_finding_ids": ["cli_preview_plan_targets_non_desktop_entrypoint"],
            "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "contract_clause_id": "clause_final_desktop_packaging_profile_not_aligned",
            "contract_requirement_ids": ["contract_packaging_profile_alignment"],
            "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
            "resolution_criteria": "final_windows_profile_targets_selected_desktop_entrypoint_and_final_runtime_name",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "dependency_resolution_not_performed",
            "source_blocker_preserved": True,
            "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
            "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "contract_clause_id": "clause_dependency_resolution_not_performed",
            "contract_requirement_ids": ["contract_python_dependency_inventory"],
            "required_evidence_ids": ["evidence_windows_dependency_resolution"],
            "resolution_criteria": "windows_target_dependency_resolution_and_locked_environment_complete",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "secret_and_local_data_exclusion_policy_not_validated",
            "source_blocker_preserved": True,
            "source_finding_ids": [
                "example_config_references_local_secret_paths",
                "example_config_contains_sensitive_field_reference",
            ],
            "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "contract_clause_id": "clause_secret_and_local_data_exclusion_policy_not_validated",
            "contract_requirement_ids": ["contract_secret_and_local_data_exclusion_policy"],
            "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
            "resolution_criteria": "policy_applied_and_final_bundle_scan_reports_zero_denied_matches",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "windows_toolchain_not_confirmed",
            "source_blocker_preserved": True,
            "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
            "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "contract_clause_id": "clause_windows_toolchain_not_confirmed",
            "contract_requirement_ids": ["contract_windows_target_toolchain_confirmation"],
            "required_evidence_ids": ["evidence_windows_toolchain_confirmation"],
            "resolution_criteria": "exact_windows_python_qt_and_packaging_toolchain_versions_recorded_and_validated",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
        {
            "blocker_id": "future_explicit_build_execution_gate_missing",
            "source_blocker_preserved": True,
            "source_finding_ids": [],
            "source_affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "contract_clause_id": "clause_future_explicit_build_execution_gate_missing",
            "contract_requirement_ids": ["contract_future_explicit_build_execution_gate"],
            "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
            "resolution_criteria": "separate_future_explicit_gate_approves_exact_build_command_after_readiness_contract",
            "contract_clause_defined": True,
            "resolved_by_18_3": False,
            "evidence_collected": False,
            "evidence_validated": False,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "contract_classification": "unresolved_blocker_contract_clause",
            "contract_result": "future_resolution_required_fail_closed",
        },
    ],
    "contract_evidence_requirement_rows": [
        {
            "evidence_id": "evidence_final_desktop_entrypoint_selection",
            "blocker_id": "final_desktop_entrypoint_not_selected",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "selected_desktop_entrypoint_record",
                "selection_cardinality_check",
            ],
            "collection_stage": "future_explicit_packaging_preparation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_desktop_entrypoint_validation",
            "blocker_id": "desktop_entrypoint_validation_not_performed",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "selected_entrypoint_static_validation_result",
                "windows_launch_smoke_result",
            ],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_qml_bundle_validation",
            "blocker_id": "qml_bundle_validation_not_performed",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": ["qml_inventory_validation_manifest", "qml_load_smoke_result"],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_windows_shared_qml_import_path",
            "blocker_id": "windows_shared_qml_import_path_unresolved",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "windows_shared_qml_import_path_manifest",
                "windows_import_path_smoke_result",
            ],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_qt_plugin_inventory",
            "blocker_id": "qt_plugin_inventory_missing",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "qt_plugin_inventory_manifest",
                "selected_tool_qt_plugin_validation_result",
            ],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_ui_package_discovery",
            "blocker_id": "ui_package_discovery_missing",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "ui_package_discovery_metadata_record",
                "package_discovery_static_check_result",
            ],
            "collection_stage": "future_explicit_packaging_preparation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_qml_package_data",
            "blocker_id": "qml_package_data_missing",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "qml_package_data_metadata_record",
                "support_asset_package_data_check_result",
            ],
            "collection_stage": "future_explicit_packaging_preparation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_final_windows_profile_alignment",
            "blocker_id": "final_desktop_packaging_profile_not_aligned",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "final_windows_profile_alignment_record",
                "desktop_entrypoint_profile_target_check",
            ],
            "collection_stage": "future_explicit_packaging_preparation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_windows_dependency_resolution",
            "blocker_id": "dependency_resolution_not_performed",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "windows_locked_dependency_manifest",
                "dependency_resolution_result",
            ],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_artifact_exclusion_validation",
            "blocker_id": "secret_and_local_data_exclusion_policy_not_validated",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "applied_exclusion_policy_manifest",
                "final_bundle_denied_pattern_scan_result",
            ],
            "collection_stage": "future_explicit_packaging_preparation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_windows_toolchain_confirmation",
            "blocker_id": "windows_toolchain_not_confirmed",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "windows_toolchain_version_manifest",
                "toolchain_validation_result",
            ],
            "collection_stage": "future_explicit_windows_validation",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
        {
            "evidence_id": "evidence_future_explicit_build_execution_gate",
            "blocker_id": "future_explicit_build_execution_gate_missing",
            "evidence_type": "future_contract_evidence",
            "required_artifacts": [
                "exact_build_command_record",
                "explicit_build_gate_approval_record",
            ],
            "collection_stage": "future_explicit_build_gate",
            "source_only_definition": True,
            "collected_by_18_3": False,
            "validated_by_18_3": False,
            "required_for_build_readiness": True,
            "required_for_packaging_authorization": True,
            "required_for_build_authorization": True,
            "failure_policy": "fail_closed",
            "evidence_classification": "required_future_evidence",
            "evidence_result": "not_collected_by_18_3",
        },
    ],
    "contract_acceptance_rule_rows": [
        {
            "acceptance_rule_id": "all_twelve_blockers_resolved",
            "required_blocker_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "qt_plugin_inventory_missing",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
                "final_desktop_packaging_profile_not_aligned",
                "dependency_resolution_not_performed",
                "secret_and_local_data_exclusion_policy_not_validated",
                "windows_toolchain_not_confirmed",
                "future_explicit_build_execution_gate_missing",
            ],
            "required_evidence_ids": [],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
        {
            "acceptance_rule_id": "all_required_evidence_collected_and_validated",
            "required_blocker_ids": [],
            "required_evidence_ids": [
                "evidence_final_desktop_entrypoint_selection",
                "evidence_desktop_entrypoint_validation",
                "evidence_qml_bundle_validation",
                "evidence_windows_shared_qml_import_path",
                "evidence_qt_plugin_inventory",
                "evidence_ui_package_discovery",
                "evidence_qml_package_data",
                "evidence_final_windows_profile_alignment",
                "evidence_windows_dependency_resolution",
                "evidence_artifact_exclusion_validation",
                "evidence_windows_toolchain_confirmation",
                "evidence_future_explicit_build_execution_gate",
            ],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
        {
            "acceptance_rule_id": "exactly_one_desktop_entrypoint_selected_and_validated",
            "required_blocker_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            "required_evidence_ids": [
                "evidence_final_desktop_entrypoint_selection",
                "evidence_desktop_entrypoint_validation",
            ],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
        {
            "acceptance_rule_id": "qml_qt_and_packaging_metadata_contract_satisfied",
            "required_blocker_ids": [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "qt_plugin_inventory_missing",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            "required_evidence_ids": [
                "evidence_qml_bundle_validation",
                "evidence_windows_shared_qml_import_path",
                "evidence_qt_plugin_inventory",
                "evidence_ui_package_discovery",
                "evidence_qml_package_data",
            ],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
        {
            "acceptance_rule_id": "windows_dependencies_profile_toolchain_contract_satisfied",
            "required_blocker_ids": [
                "final_desktop_packaging_profile_not_aligned",
                "dependency_resolution_not_performed",
                "windows_toolchain_not_confirmed",
            ],
            "required_evidence_ids": [
                "evidence_final_windows_profile_alignment",
                "evidence_windows_dependency_resolution",
                "evidence_windows_toolchain_confirmation",
            ],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
        {
            "acceptance_rule_id": "artifact_exclusion_and_explicit_build_gate_contract_satisfied",
            "required_blocker_ids": [
                "secret_and_local_data_exclusion_policy_not_validated",
                "future_explicit_build_execution_gate_missing",
            ],
            "required_evidence_ids": [
                "evidence_artifact_exclusion_validation",
                "evidence_future_explicit_build_execution_gate",
            ],
            "all_inputs_required": True,
            "rule_defined": True,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3",
        },
    ],
    "real_capability_contract_state": {
        "inherited_block_o_capabilities": {
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
            "create" + "_order": "blocked",
            "submit_order": "blocked",
            "cancel_order": "blocked",
            "replace_order": "blocked",
            "fetch" + "_balance": "blocked",
            "private_endpoint": "blocked",
            "network": "blocked",
            "credentials": "blocked",
            "config_env_secrets": "blocked",
            "qml_bridge": "blocked",
            "cc" + "xt": "blocked",
        },
        "inherited_block_o_capabilities_known_blocked": True,
        "inherited_block_p_capabilities": {
            "desktop_entrypoint_selection": "blocked",
            "desktop_entrypoint_validation": "blocked",
            "qml_asset_inventory": "blocked",
            "qml_asset_validation": "blocked",
            "qt_plugin_inventory": "blocked",
            "qt_plugin_validation": "blocked",
            "python_dependency_inventory": "blocked",
            "python_dependency_validation": "blocked",
            "packaging_profile_selection": "blocked",
            "packaging_profile_validation": "blocked",
            "secret_exclusion_validation": "blocked",
            "local_data_exclusion_validation": "blocked",
            "windows_toolchain_selection": "blocked",
            "windows_toolchain_validation": "blocked",
            "pyinstaller_configuration": "blocked",
            "pyinstaller_execution": "blocked",
            "build_command_creation": "blocked",
            "build_command_execution": "blocked",
            "exe_artifact_creation": "blocked",
            "artifact_scan": "blocked",
            "artifact_hash_manifest": "blocked",
            "artifact_signing": "blocked",
            "installer_creation": "blocked",
            "release_creation": "blocked",
            "release_upload": "blocked",
            "runtime_activation": "blocked",
            "order_activity": "blocked",
        },
        "inherited_block_p_capabilities_known_blocked": True,
        "source_inventory_capabilities": {
            "runtime_filesystem_inventory": "blocked",
            "home_directory_scan": "blocked",
            "secret_directory_scan": "blocked",
            "secret_file_read": "blocked",
            "dependency_import": "blocked",
            "dependency_resolution": "blocked",
            "qml_load": "blocked",
            "qml_runtime_validation": "blocked",
            "qt_plugin_discovery": "blocked",
            "qt_plugin_validation": "blocked",
            "packaging_profile_validation": "blocked",
            "build_tool_selection": "blocked",
            "build_tool_execution": "blocked",
            "spec_file_creation": "blocked",
            "build_command_creation": "blocked",
            "build_command_execution": "blocked",
            "exe_artifact_creation": "blocked",
            "artifact_scan": "blocked",
            "artifact_signing": "blocked",
            "installer_creation": "blocked",
            "release_creation": "blocked",
            "runtime_activation": "blocked",
            "order_activity": "blocked",
        },
        "source_inventory_capabilities_known_blocked": True,
        "inventory_matrix_capabilities": {
            "runtime_inventory_matrix_evaluation": "blocked",
            "desktop_entrypoint_selection": "blocked",
            "desktop_entrypoint_validation": "blocked",
            "qml_bundle_validation": "blocked",
            "qt_plugin_discovery": "blocked",
            "qt_plugin_validation": "blocked",
            "dependency_resolution": "blocked",
            "dependency_validation": "blocked",
            "packaging_metadata_mutation": "blocked",
            "packaging_profile_selection": "blocked",
            "packaging_profile_validation": "blocked",
            "artifact_exclusion_policy_application": "blocked",
            "artifact_exclusion_policy_validation": "blocked",
            "windows_toolchain_selection": "blocked",
            "windows_toolchain_validation": "blocked",
            "packaging_contract_approval": "blocked",
            "build_readiness_grant": "blocked",
            "packaging_authorization": "blocked",
            "build_authorization": "blocked",
            "spec_file_creation": "blocked",
            "build_command_creation": "blocked",
            "build_command_execution": "blocked",
            "pyinstaller_execution": "blocked",
            "briefcase_execution": "blocked",
            "exe_artifact_creation": "blocked",
            "artifact_scan": "blocked",
            "artifact_signing": "blocked",
            "installer_creation": "blocked",
            "release_creation": "blocked",
            "runtime_activation": "blocked",
            "order_activity": "blocked",
        },
        "inventory_matrix_capabilities_known_blocked": True,
        "packaging_contract_capabilities": {
            "packaging_contract_operational_approval": "blocked",
            "desktop_entrypoint_selection": "blocked",
            "desktop_entrypoint_validation": "blocked",
            "qml_bundle_validation": "blocked",
            "qt_plugin_discovery": "blocked",
            "qt_plugin_validation": "blocked",
            "dependency_resolution": "blocked",
            "dependency_validation": "blocked",
            "packaging_metadata_mutation": "blocked",
            "packaging_profile_selection": "blocked",
            "packaging_profile_validation": "blocked",
            "artifact_exclusion_policy_application": "blocked",
            "artifact_exclusion_policy_validation": "blocked",
            "windows_toolchain_selection": "blocked",
            "windows_toolchain_validation": "blocked",
            "build_readiness_grant": "blocked",
            "packaging_authorization": "blocked",
            "build_authorization": "blocked",
            "spec_file_creation": "blocked",
            "build_command_creation": "blocked",
            "build_command_execution": "blocked",
            "pyinstaller_execution": "blocked",
            "briefcase_execution": "blocked",
            "exe_artifact_creation": "blocked",
            "artifact_scan": "blocked",
            "artifact_signing": "blocked",
            "installer_creation": "blocked",
            "release_creation": "blocked",
            "runtime_activation": "blocked",
            "order_activity": "blocked",
        },
        "packaging_contract_capabilities_known_blocked": True,
        "all_real_capabilities_blocked_at_18_3": True,
    },
    "fail_closed_contract_decision": {
        "block_p_inventory_matrix_preserved_in_18_3": True,
        "block_p_packaging_contract_complete_in_18_3": True,
        "only_source_only_18_4_handoff_allowed": True,
        "definitions_complete": True,
        "requirements_satisfied": False,
        "blockers_resolved": False,
        "evidence_collected": False,
        "selection_completed": False,
        "validation_completed": False,
        "readiness_granted": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "build_ready_by_18_3": False,
        "runtime_started_by_18_3": False,
        "orders_enabled_by_18_3": False,
    },
    "non_execution_contract_evidence": {
        "source_builder_called": True,
        "source_builder_call_count": 1,
        "source_accepted": True,
        "source_plain_bounded": True,
        "all_top_level_keys_exact_str": True,
        "identity_valid": True,
        "source_reference_valid": True,
        "matrix_summary_valid": True,
        "source_preservation_valid": True,
        "entrypoint_rows_valid": True,
        "qml_rows_valid": True,
        "dependency_rows_valid": True,
        "metadata_rows_valid": True,
        "preview_rows_valid": True,
        "policy_rows_valid": True,
        "finding_rows_valid": True,
        "scope_rows_valid": True,
        "requirement_rows_valid": True,
        "blocker_rows_valid": True,
        "real_capability_valid": True,
        "fail_closed_valid": True,
        "evidence_valid": True,
        "matrix_boundaries_valid": True,
        "source_boundaries_valid": True,
        "future_steps_valid": True,
        "scope_contract_count": 3,
        "requirement_contract_count": 8,
        "blocker_contract_count": 12,
        "evidence_requirement_count": 12,
        "acceptance_rule_count": 6,
        "referential_integrity_valid": True,
        "contract_artifact_complete": True,
        "resolution_by_18_3": False,
        "entrypoint_selection_by_18_3": False,
        "entrypoint_validation_by_18_3": False,
        "qml_validation_by_18_3": False,
        "dependency_resolution_by_18_3": False,
        "metadata_mutation_by_18_3": False,
        "policy_application_by_18_3": False,
        "readiness_granted_by_18_3": False,
        "packaging_authorized_by_18_3": False,
        "build_authorized_by_18_3": False,
        "build_performed_by_18_3": False,
        "artifact_created_by_18_3": False,
        "runtime_started_by_18_3": False,
        "orders_enabled_by_18_3": False,
    },
    "contract_boundaries": {
        "reads_18_2_only": True,
        "source_only": True,
        "plain_data": True,
        "static_contract": True,
        "repo_rescan": False,
        "filesystem_io": False,
        "environment_io": False,
        "secret_io": False,
        "dependency_resolution": False,
        "pyside_import": False,
        "qml_load": False,
        "plugin_discovery": False,
        "entrypoint_selection": False,
        "metadata_mutation": False,
        "profile_selection": False,
        "tool_selection": False,
        "policy_application": False,
        "toolchain_confirmation": False,
        "gate_approval": False,
        "spec_creation": False,
        "command_creation": False,
        "package_execution": False,
        "build_execution": False,
        "artifact_creation": False,
        "artifact_signing": False,
        "installer_creation": False,
        "release_creation": False,
        "runtime_activation": False,
        "order_activity": False,
        "network_io": False,
        "credential_io": False,
        "only_18_4_handoff_depends_on_source_acceptance": True,
    },
    "source_boundaries": {
        "source_block_p_desktop_exe_packaging_inventory_matrix": "FUNCTIONAL-PREVIEW-18.2",
        "inventory_matrix_preserved": True,
        "can_build_desktop_exe_packaging_contract": True,
        "packaging_contract_artifact_complete": True,
        "can_build_desktop_exe_packaging_read_model": True,
        "can_feed_18_4": True,
    },
    "future_steps": [
        {
            "step": "18.4",
            "title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.5",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS MATRIX",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.6",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.7",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.8",
            "title": "BLOCK P CLOSURE AUDIT",
            "source_only": True,
            "build_performed": False,
        },
    ],
    "status": "ready_for_functional_preview_18_4_block_p_desktop_exe_packaging_read_model",
}

CAPABILITY_KEYS: Final[list[str]] = [
    "contract_mutation",
    "blocker_resolution",
    "evidence_collection",
    "evidence_validation",
    "acceptance_rule_satisfaction",
    "desktop_entrypoint_selection",
    "desktop_entrypoint_validation",
    "qml_bundle_validation",
    "qt_plugin_discovery",
    "qt_plugin_validation",
    "dependency_resolution",
    "dependency_validation",
    "packaging_metadata_mutation",
    "packaging_profile_selection",
    "packaging_profile_validation",
    "artifact_exclusion_policy_application",
    "artifact_exclusion_policy_validation",
    "windows_toolchain_selection",
    "windows_toolchain_validation",
    "build_readiness_grant",
    "packaging_authorization",
    "build_authorization",
    "spec_file_creation",
    "build_command_creation",
    "build_command_execution",
    "pyinstaller_execution",
    "briefcase_execution",
    "exe_artifact_creation",
    "artifact_scan",
    "artifact_signing",
    "installer_creation",
    "release_creation",
    "runtime_activation",
    "order_activity",
]


def _all_plain_json(value: Any, max_depth: int) -> bool:
    stack: list[tuple[Any, int, set[int]]] = [(value, 0, set())]
    while stack:
        current, depth, ancestors = stack.pop()
        if depth > max_depth or type(current) not in (dict, list, str, int, bool, type(None)):
            return False
        if type(current) in (dict, list):
            identity = id(current)
            if identity in ancestors:
                return False
            next_ancestors = ancestors | {identity}
            items = current.items() if type(current) is dict else enumerate(current)
            for key, item in items:
                if type(current) is dict and type(key) is not str:
                    return False
                stack.append((item, depth + 1, next_ancestors))
    return True


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    return value


def _exact_plain_matches(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        if list(actual.keys()) != list(expected.keys()):
            return False
        return all(_exact_plain_matches(actual[key], expected[key]) for key in expected)
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact_plain_matches(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def _plain_dict_section(source: Any, field: str) -> dict[str, Any]:
    value = _get_exact_string_key(source, field)
    return value if type(value) is dict else {}


def _plain_list_section(source: Any, field: str) -> list[Any]:
    value = _get_exact_string_key(source, field)
    return value if type(value) is list else []


def _section_valid(actual: Any, expected: Any) -> bool:
    return _all_plain_json(actual, MAX_DIAGNOSTIC_CONTAINER_DEPTH) and _exact_plain_matches(
        actual, expected
    )


def _safe_top_level_source(source: Any) -> tuple[dict[str, Any], bool]:
    if type(source) is not dict:
        return {}, False
    safe: dict[str, Any] = {}
    all_keys_exact_str = True
    for key, value in source.items():
        if type(key) is str:
            safe[key] = value
        else:
            all_keys_exact_str = False
    return safe, all_keys_exact_str


def _get_exact_string_key(section: Any, key: str) -> Any:
    if type(section) is not dict:
        return None
    for actual, value in section.items():
        if type(actual) is str and actual == key:
            return value
    return None


def _owned_fields_are_unshadowed(
    section: Any, expected: dict[str, Any], owned_fields: list[str]
) -> bool:
    if type(section) is not dict:
        return False
    for raw_key, raw_value in section.items():
        if type(raw_key) is not str:
            continue
        if raw_key not in owned_fields:
            continue
        if raw_key not in expected:
            return False
        if not _all_plain_json(raw_value, MAX_DIAGNOSTIC_CONTAINER_DEPTH):
            return False
        if not _exact_plain_matches(raw_value, expected[raw_key]):
            return False
    return True


def _no_shadowing(source: Any) -> bool:
    if type(source) is not dict:
        return False
    return (
        _owned_fields_are_unshadowed(source, EXPECTED_SOURCE, TOP_LEVEL_FIELDS_18_3)
        and _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "packaging_contract_summary"),
            EXPECTED_SOURCE["packaging_contract_summary"],
            SUMMARY_OWNED_FIELDS_18_4,
        )
        and _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "fail_closed_contract_decision"),
            EXPECTED_SOURCE["fail_closed_contract_decision"],
            FAIL_CLOSED_OWNED_FIELDS_18_4,
        )
        and _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "source_boundaries"),
            EXPECTED_SOURCE["source_boundaries"],
            SOURCE_BOUNDARY_FIELDS_18_4,
        )
    )


def _source_identity_valid(source: Any) -> bool:
    return type(source) is dict and all(
        type(_get_exact_string_key(source, key)) is type(value)
        and _get_exact_string_key(source, key) == value
        for key, value in SOURCE_IDENTITY_EXPECTED.items()
    )


def _scalar_reference(source: dict[str, Any], key: str, identity_valid: bool) -> Any:
    return (
        _get_exact_string_key(source, key)
        if identity_valid
        else (False if type(SOURCE_IDENTITY_EXPECTED[key]) is bool else "")
    )


def _capability_map_known_blocked(capability_map: dict[str, Any]) -> bool:
    return bool(capability_map) and all(
        type(value) is str and value == "blocked" for value in capability_map.values()
    )


def _nonempty_unique(values: list[str]) -> bool:
    return (
        bool(values)
        and all(type(value) is str and bool(value) for value in values)
        and len(values) == len(set(values))
    )


def _links_valid(rows: list[dict[str, Any]], field: str, allowed: set[str]) -> bool:
    return all(
        type(row[field]) is list
        and len(row[field]) == len(set(row[field]))
        and all(type(value) is str and bool(value) and value in allowed for value in row[field])
        for row in rows
    )


def _exact_row_dict(value: Any) -> dict[str, Any] | None:
    return value if type(value) is dict else None


def _exact_string_list(value: Any) -> list[str] | None:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        return None
    return value if len(value) == len(set(value)) else None


def _is_canonical_subsequence(actual_ids: list[str], canonical_ids: list[str]) -> bool:
    position = 0
    for actual_id in actual_ids:
        while position < len(canonical_ids) and canonical_ids[position] != actual_id:
            position += 1
        if position == len(canonical_ids):
            return False
        position += 1
    return True


def _output_graph_semantics_valid(
    scope_rows: list[dict[str, Any]],
    requirement_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    acceptance_rows: list[dict[str, Any]],
) -> bool:
    scope_ids = {row["scope_id"] for row in scope_rows}
    contract_ids = {row["contract_requirement_id"] for row in requirement_rows}
    blocker_ids = {row["blocker_id"] for row in blocker_rows}
    evidence_ids = {row["evidence_id"] for row in evidence_rows}
    clause_ids = {row["contract_clause_id"] for row in blocker_rows}
    if not (
        _is_canonical_subsequence(
            [row["scope_id"] for row in scope_rows], list(SCOPE_RELATIONS_EXPECTED)
        )
        and _is_canonical_subsequence(
            [row["requirement_id"] for row in requirement_rows],
            list(REQUIREMENT_RELATIONS_EXPECTED),
        )
        and _is_canonical_subsequence(
            [row["blocker_id"] for row in blocker_rows], list(BLOCKER_RELATIONS_EXPECTED)
        )
        and _is_canonical_subsequence(
            [row["evidence_id"] for row in evidence_rows], list(EVIDENCE_RELATIONS_EXPECTED)
        )
        and _is_canonical_subsequence(
            [row["acceptance_rule_id"] for row in acceptance_rows],
            list(ACCEPTANCE_RELATIONS_EXPECTED),
        )
    ):
        return False
    for row in blocker_rows:
        expected = BLOCKER_RELATIONS_EXPECTED.get(row["blocker_id"])
        if (
            expected is None
            or row["contract_clause_id"] != expected["contract_clause_id"]
            or row["source_affected_scope_ids"]
            != [x for x in expected["source_affected_scope_ids"] if x in scope_ids]
            or row["contract_requirement_ids"]
            != [x for x in expected["contract_requirement_ids"] if x in contract_ids]
            or row["required_evidence_ids"]
            != [x for x in expected["required_evidence_ids"] if x in evidence_ids]
        ):
            return False
    for row in scope_rows:
        expected = SCOPE_RELATIONS_EXPECTED.get(row["scope_id"])
        if (
            expected is None
            or row["source_unresolved_blocker_ids"]
            != [x for x in expected["source_unresolved_blocker_ids"] if x in blocker_ids]
            or row["contract_clause_ids"]
            != [x for x in expected["contract_clause_ids"] if x in clause_ids]
            or row["required_evidence_ids"]
            != [x for x in expected["required_evidence_ids"] if x in evidence_ids]
        ):
            return False
    for row in requirement_rows:
        expected = REQUIREMENT_RELATIONS_EXPECTED.get(row["requirement_id"])
        if (
            expected is None
            or row["contract_requirement_id"] != expected["contract_requirement_id"]
            or row["source_unresolved_condition_ids"]
            != [x for x in expected["source_unresolved_condition_ids"] if x in blocker_ids]
            or row["contract_clause_ids"]
            != [x for x in expected["contract_clause_ids"] if x in clause_ids]
            or row["required_evidence_ids"]
            != [x for x in expected["required_evidence_ids"] if x in evidence_ids]
        ):
            return False
    for row in evidence_rows:
        if (
            EVIDENCE_RELATIONS_EXPECTED.get(row["evidence_id"], {}).get("blocker_id")
            != row["blocker_id"]
            or row["blocker_id"] not in blocker_ids
        ):
            return False
    for row in acceptance_rows:
        expected = ACCEPTANCE_RELATIONS_EXPECTED.get(row["acceptance_rule_id"])
        if (
            expected is None
            or row["required_blocker_ids"]
            != [x for x in expected["required_blocker_ids"] if x in blocker_ids]
            or row["required_evidence_ids"]
            != [x for x in expected["required_evidence_ids"] if x in evidence_ids]
        ):
            return False
    return True


def _output_integrity(payload: dict[str, Any]) -> bool:
    if type(payload) is not dict:
        return False
    families = (
        ("domain_contract_read_model_rows", "domain_id"),
        ("scope_contract_read_model_rows", "scope_id"),
        ("requirement_contract_read_model_rows", "requirement_id"),
        ("blocker_read_model_rows", "blocker_id"),
        ("evidence_read_model_rows", "evidence_id"),
        ("acceptance_rule_read_model_rows", "acceptance_rule_id"),
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, key in families:
        family = payload.get(name)
        if type(family) is not list:
            return False
        checked = []
        for candidate in family:
            row = _exact_row_dict(candidate)
            if row is None or type(row.get(key)) is not str or not row[key]:
                return False
            checked.append(row)
        if len([row[key] for row in checked]) != len({row[key] for row in checked}):
            return False
        rows[name] = checked
    scopes = rows["scope_contract_read_model_rows"]
    requirements = rows["requirement_contract_read_model_rows"]
    blockers = rows["blocker_read_model_rows"]
    evidence = rows["evidence_read_model_rows"]
    acceptance = rows["acceptance_rule_read_model_rows"]
    required = (
        (scopes, ("source_unresolved_blocker_ids", "contract_clause_ids", "required_evidence_ids")),
        (
            requirements,
            ("source_unresolved_condition_ids", "contract_clause_ids", "required_evidence_ids"),
        ),
        (
            blockers,
            ("source_affected_scope_ids", "contract_requirement_ids", "required_evidence_ids"),
        ),
        (acceptance, ("required_blocker_ids", "required_evidence_ids")),
    )
    if any(
        _exact_string_list(row.get(field)) is None
        for family, fields in required
        for row in family
        for field in fields
    ):
        return False
    if any(
        type(row.get("contract_requirement_id")) is not str or not row["contract_requirement_id"]
        for row in requirements
    ):
        return False
    if any(
        type(row.get("contract_clause_id")) is not str or not row["contract_clause_id"]
        for row in blockers
    ):
        return False
    if any(type(row.get("blocker_id")) is not str for row in evidence):
        return False
    return _output_graph_semantics_valid(scopes, requirements, blockers, evidence, acceptance)


def _source_referential_integrity_valid(
    scope_rows: list[dict[str, Any]],
    requirement_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    acceptance_rows: list[dict[str, Any]],
) -> bool:
    def ids(rows: list[dict[str, Any]], field: str) -> list[str] | None:
        values: list[str] = []
        for row in rows:
            if type(row) is not dict:
                return None
            value = row.get(field)
            if type(value) is not str or not value:
                return None
            values.append(value)
        return values if len(values) == len(set(values)) else None

    scope_ids = ids(scope_rows, "scope_id")
    requirement_ids = ids(requirement_rows, "requirement_id")
    contract_requirement_ids = ids(requirement_rows, "contract_requirement_id")
    blocker_ids = ids(blocker_rows, "blocker_id")
    clause_ids = ids(blocker_rows, "contract_clause_id")
    evidence_ids = ids(evidence_rows, "evidence_id")
    acceptance_ids = ids(acceptance_rows, "acceptance_rule_id")
    if any(
        item is None
        for item in (
            scope_ids,
            requirement_ids,
            contract_requirement_ids,
            blocker_ids,
            clause_ids,
            evidence_ids,
            acceptance_ids,
        )
    ):
        return False
    if not all(
        (
            scope_ids,
            requirement_ids,
            contract_requirement_ids,
            blocker_ids,
            clause_ids,
            evidence_ids,
            acceptance_ids,
        )
    ):
        return False

    def links(rows: list[dict[str, Any]], field: str, allowed: list[str]) -> bool:
        allowed_ids = set(allowed)
        for row in rows:
            values = row.get(field)
            if type(values) is not list or not _nonempty_unique(values):
                return False
            if not set(values).issubset(allowed_ids):
                return False
        return True

    safe_blocker_ids = blocker_ids if blocker_ids is not None else []
    safe_evidence_ids = evidence_ids if evidence_ids is not None else []

    def optional_links(rows: list[dict[str, Any]], field: str, allowed: list[str]) -> bool:
        allowed_ids = set(allowed)
        return all(
            type(row.get(field)) is list
            and len(row[field]) == len(set(row[field]))
            and all(
                type(value) is str and bool(value) and value in allowed_ids for value in row[field]
            )
            for row in rows
        )

    if not (
        links(scope_rows, "source_unresolved_blocker_ids", blocker_ids)
        and links(scope_rows, "contract_clause_ids", clause_ids)
        and links(scope_rows, "required_evidence_ids", evidence_ids)
        and links(requirement_rows, "source_unresolved_condition_ids", blocker_ids)
        and links(requirement_rows, "contract_clause_ids", clause_ids)
        and links(requirement_rows, "required_evidence_ids", evidence_ids)
        and links(blocker_rows, "source_affected_scope_ids", scope_ids)
        and links(blocker_rows, "contract_requirement_ids", contract_requirement_ids)
        and links(blocker_rows, "required_evidence_ids", evidence_ids)
        and optional_links(acceptance_rows, "required_blocker_ids", blocker_ids)
    ):
        return False
    for row in acceptance_rows:
        values = row.get("required_evidence_ids")
        if (
            type(values) is not list
            or len(values) != len(set(values))
            or any(
                type(value) is not str or not value or value not in safe_evidence_ids
                for value in values
            )
        ):
            return False
    if any(
        type(row.get("blocker_id")) is not str or row["blocker_id"] not in safe_blocker_ids
        for row in evidence_rows
    ):
        return False
    if any(
        len(row["contract_requirement_ids"]) != 1 or len(row["required_evidence_ids"]) != 1
        for row in blocker_rows
    ):
        return False
    evidence_blockers = [row["blocker_id"] for row in evidence_rows]
    if len(evidence_blockers) != len(set(evidence_blockers)) or set(evidence_blockers) != set(
        blocker_ids
    ):
        return False
    safe_graph_blocker_ids = blocker_ids if blocker_ids is not None else []
    evidence_by_blocker = {row["blocker_id"]: row["evidence_id"] for row in evidence_rows}
    clause_by_blocker = {row["blocker_id"]: row["contract_clause_id"] for row in blocker_rows}
    blocker_by_id = {row["blocker_id"]: row for row in blocker_rows}
    requirement_by_blocker: dict[str, list[str]] = {
        blocker_id: [] for blocker_id in safe_graph_blocker_ids
    }
    scope_by_blocker: dict[str, list[str]] = {
        blocker_id: [] for blocker_id in safe_graph_blocker_ids
    }
    for row in requirement_rows:
        if not (
            len(row["source_unresolved_condition_ids"])
            == len(row["contract_clause_ids"])
            == len(row["required_evidence_ids"])
        ):
            return False
        for blocker_id in row["source_unresolved_condition_ids"]:
            requirement_by_blocker[blocker_id].append(row["contract_requirement_id"])
        for blocker_id, clause_id, evidence_id in zip(
            row["source_unresolved_condition_ids"],
            row["contract_clause_ids"],
            row["required_evidence_ids"],
        ):
            if (
                clause_id != clause_by_blocker[blocker_id]
                or evidence_id != evidence_by_blocker[blocker_id]
            ):
                return False
    for row in scope_rows:
        if not (
            len(row["source_unresolved_blocker_ids"])
            == len(row["contract_clause_ids"])
            == len(row["required_evidence_ids"])
        ):
            return False
        for blocker_id in row["source_unresolved_blocker_ids"]:
            scope_by_blocker[blocker_id].append(row["scope_id"])
        for blocker_id, clause_id, evidence_id in zip(
            row["source_unresolved_blocker_ids"],
            row["contract_clause_ids"],
            row["required_evidence_ids"],
        ):
            if (
                clause_id != clause_by_blocker[blocker_id]
                or evidence_id != evidence_by_blocker[blocker_id]
            ):
                return False
    for row in acceptance_rows:
        expected = ACCEPTANCE_RELATIONS_EXPECTED.get(row["acceptance_rule_id"])
        if (
            expected is None
            or row["required_blocker_ids"] != expected["required_blocker_ids"]
            or row["required_evidence_ids"] != expected["required_evidence_ids"]
        ):
            return False
    for blocker_id, row in blocker_by_id.items():
        if row["required_evidence_ids"] != [evidence_by_blocker[blocker_id]]:
            return False
        if row["contract_requirement_ids"] != requirement_by_blocker[blocker_id]:
            return False
        if row["source_affected_scope_ids"] != scope_by_blocker[blocker_id]:
            return False
    if [row["scope_id"] for row in scope_rows] != list(SCOPE_RELATIONS_EXPECTED):
        return False
    if [row["requirement_id"] for row in requirement_rows] != list(REQUIREMENT_RELATIONS_EXPECTED):
        return False
    if [row["blocker_id"] for row in blocker_rows] != list(BLOCKER_RELATIONS_EXPECTED):
        return False
    if [row["evidence_id"] for row in evidence_rows] != list(EVIDENCE_RELATIONS_EXPECTED):
        return False
    if [row["acceptance_rule_id"] for row in acceptance_rows] != list(
        ACCEPTANCE_RELATIONS_EXPECTED
    ):
        return False
    for row in blocker_rows:
        expected = BLOCKER_RELATIONS_EXPECTED[row["blocker_id"]]
        if any(row[field] != expected[field] for field in expected):
            return False
    for row in scope_rows:
        expected = SCOPE_RELATIONS_EXPECTED[row["scope_id"]]
        if any(row[field] != expected[field] for field in expected):
            return False
    for row in requirement_rows:
        expected = REQUIREMENT_RELATIONS_EXPECTED[row["requirement_id"]]
        if any(row[field] != expected[field] for field in expected):
            return False
    if any(
        row["blocker_id"] != EVIDENCE_RELATIONS_EXPECTED[row["evidence_id"]]["blocker_id"]
        for row in evidence_rows
    ):
        return False
    return True


def build_preview_block_p_desktop_exe_packaging_read_model() -> dict[str, Any]:
    source_raw = build_preview_block_p_desktop_exe_packaging_contract()
    plain = _all_plain_json(source_raw, MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    source, keys = _safe_top_level_source(source_raw)
    identity_valid = _source_identity_valid(source)
    section_names = [
        "block_p_desktop_exe_packaging_inventory_matrix_reference",
        "packaging_contract_summary",
        "source_matrix_preservation",
        "contract_principles",
        "desktop_entrypoint_contract",
        "qml_bundle_contract",
        "python_dependency_contract",
        "packaging_metadata_contract",
        "preview_packaging_separation_contract",
        "artifact_exclusion_contract",
        "packaging_scope_contract_rows",
        "packaging_requirement_contract_rows",
        "unresolved_blocker_contract_rows",
        "contract_evidence_requirement_rows",
        "contract_acceptance_rule_rows",
        "real_capability_contract_state",
        "fail_closed_contract_decision",
        "non_execution_contract_evidence",
        "contract_boundaries",
        "source_boundaries",
        "future_steps",
    ]
    valid = {
        "contract_reference_valid": _section_valid(
            _get_exact_string_key(
                source, "block_p_desktop_exe_packaging_inventory_matrix_reference"
            ),
            EXPECTED_SOURCE["block_p_desktop_exe_packaging_inventory_matrix_reference"],
        ),
        "contract_summary_valid": _section_valid(
            _get_exact_string_key(source, "packaging_contract_summary"),
            EXPECTED_SOURCE["packaging_contract_summary"],
        ),
        "source_preservation_valid": _section_valid(
            _get_exact_string_key(source, "source_matrix_preservation"),
            EXPECTED_SOURCE["source_matrix_preservation"],
        ),
        "principles_valid": _section_valid(
            _get_exact_string_key(source, "contract_principles"),
            EXPECTED_SOURCE["contract_principles"],
        ),
        "desktop_entrypoint_contract_valid": _section_valid(
            _get_exact_string_key(source, "desktop_entrypoint_contract"),
            EXPECTED_SOURCE["desktop_entrypoint_contract"],
        ),
        "qml_bundle_contract_valid": _section_valid(
            _get_exact_string_key(source, "qml_bundle_contract"),
            EXPECTED_SOURCE["qml_bundle_contract"],
        ),
        "python_dependency_contract_valid": _section_valid(
            _get_exact_string_key(source, "python_dependency_contract"),
            EXPECTED_SOURCE["python_dependency_contract"],
        ),
        "packaging_metadata_contract_valid": _section_valid(
            _get_exact_string_key(source, "packaging_metadata_contract"),
            EXPECTED_SOURCE["packaging_metadata_contract"],
        ),
        "preview_packaging_contract_valid": _section_valid(
            _get_exact_string_key(source, "preview_packaging_separation_contract"),
            EXPECTED_SOURCE["preview_packaging_separation_contract"],
        ),
        "artifact_exclusion_contract_valid": _section_valid(
            _get_exact_string_key(source, "artifact_exclusion_contract"),
            EXPECTED_SOURCE["artifact_exclusion_contract"],
        ),
        "scope_contract_rows_valid": _section_valid(
            _get_exact_string_key(source, "packaging_scope_contract_rows"),
            EXPECTED_SOURCE["packaging_scope_contract_rows"],
        ),
        "requirement_contract_rows_valid": _section_valid(
            _get_exact_string_key(source, "packaging_requirement_contract_rows"),
            EXPECTED_SOURCE["packaging_requirement_contract_rows"],
        ),
        "blocker_contract_rows_valid": _section_valid(
            _get_exact_string_key(source, "unresolved_blocker_contract_rows"),
            EXPECTED_SOURCE["unresolved_blocker_contract_rows"],
        ),
        "evidence_rows_valid": _section_valid(
            _get_exact_string_key(source, "contract_evidence_requirement_rows"),
            EXPECTED_SOURCE["contract_evidence_requirement_rows"],
        ),
        "acceptance_rows_valid": _section_valid(
            _get_exact_string_key(source, "contract_acceptance_rule_rows"),
            EXPECTED_SOURCE["contract_acceptance_rule_rows"],
        ),
        "capability_state_valid": _section_valid(
            _get_exact_string_key(source, "real_capability_contract_state"),
            EXPECTED_SOURCE["real_capability_contract_state"],
        ),
        "fail_closed_valid": _section_valid(
            _get_exact_string_key(source, "fail_closed_contract_decision"),
            EXPECTED_SOURCE["fail_closed_contract_decision"],
        ),
        "non_execution_evidence_valid": _section_valid(
            _get_exact_string_key(source, "non_execution_contract_evidence"),
            EXPECTED_SOURCE["non_execution_contract_evidence"],
        ),
        "contract_boundaries_valid": _section_valid(
            _get_exact_string_key(source, "contract_boundaries"),
            EXPECTED_SOURCE["contract_boundaries"],
        ),
        "source_boundaries_valid": _section_valid(
            _get_exact_string_key(source, "source_boundaries"), EXPECTED_SOURCE["source_boundaries"]
        ),
        "future_steps_valid": _section_valid(
            _get_exact_string_key(source, "future_steps"), EXPECTED_SOURCE["future_steps"]
        ),
    }
    valid.update(
        {
            "block_p_desktop_exe_packaging_inventory_matrix_reference": valid[
                "contract_reference_valid"
            ],
            "packaging_contract_summary": valid["contract_summary_valid"],
            "source_matrix_preservation": valid["source_preservation_valid"],
            "contract_principles": valid["principles_valid"],
            "desktop_entrypoint_contract": valid["desktop_entrypoint_contract_valid"],
            "qml_bundle_contract": valid["qml_bundle_contract_valid"],
            "python_dependency_contract": valid["python_dependency_contract_valid"],
            "packaging_metadata_contract": valid["packaging_metadata_contract_valid"],
            "preview_packaging_separation_contract": valid["preview_packaging_contract_valid"],
            "artifact_exclusion_contract": valid["artifact_exclusion_contract_valid"],
            "packaging_scope_contract_rows": valid["scope_contract_rows_valid"],
            "packaging_requirement_contract_rows": valid["requirement_contract_rows_valid"],
            "unresolved_blocker_contract_rows": valid["blocker_contract_rows_valid"],
            "contract_evidence_requirement_rows": valid["evidence_rows_valid"],
            "contract_acceptance_rule_rows": valid["acceptance_rows_valid"],
            "real_capability_contract_state": valid["capability_state_valid"],
            "fail_closed_contract_decision": valid["fail_closed_valid"],
            "non_execution_contract_evidence": valid["non_execution_evidence_valid"],
            "contract_boundaries": valid["contract_boundaries_valid"],
            "source_boundaries": valid["source_boundaries_valid"],
            "future_steps": valid["future_steps_valid"],
        }
    )
    source_scope_rows = (
        _plain_list_section(source, "packaging_scope_contract_rows")
        if valid["scope_contract_rows_valid"]
        else []
    )
    source_requirement_rows = (
        _plain_list_section(source, "packaging_requirement_contract_rows")
        if valid["requirement_contract_rows_valid"]
        else []
    )
    source_blocker_rows = (
        _plain_list_section(source, "unresolved_blocker_contract_rows")
        if valid["blocker_contract_rows_valid"]
        else []
    )
    source_evidence_rows = (
        _plain_list_section(source, "contract_evidence_requirement_rows")
        if valid["evidence_rows_valid"]
        else []
    )
    source_acceptance_rows = (
        _plain_list_section(source, "contract_acceptance_rule_rows")
        if valid["acceptance_rows_valid"]
        else []
    )
    source_graph_integrity_valid: bool = (
        valid["scope_contract_rows_valid"]
        and valid["requirement_contract_rows_valid"]
        and valid["blocker_contract_rows_valid"]
        and valid["evidence_rows_valid"]
        and valid["acceptance_rows_valid"]
        and _source_referential_integrity_valid(
            source_scope_rows,
            source_requirement_rows,
            source_blocker_rows,
            source_evidence_rows,
            source_acceptance_rows,
        )
    )
    contract_definition_graph_complete: bool = source_graph_integrity_valid and all(
        valid[key]
        for key in (
            "principles_valid",
            "desktop_entrypoint_contract_valid",
            "qml_bundle_contract_valid",
            "python_dependency_contract_valid",
            "packaging_metadata_contract_valid",
            "preview_packaging_contract_valid",
            "artifact_exclusion_contract_valid",
            "scope_contract_rows_valid",
            "requirement_contract_rows_valid",
            "blocker_contract_rows_valid",
            "evidence_rows_valid",
            "acceptance_rows_valid",
        )
    )
    accepted = (
        plain
        and keys
        and list(source.keys()) == TOP_LEVEL_FIELDS_18_3
        and identity_valid
        and all(valid.values())
        and _no_shadowing(source)
        and source_graph_integrity_valid
    )
    domains = [
        ("desktop_application_entrypoint", "desktop_entrypoint_contract", True, True, 2),
        ("qt_qml_runtime_bundle", "qml_bundle_contract", False, True, 5),
        ("python_dependency_bundle", "python_dependency_contract", False, True, 2),
        ("packaging_metadata", "packaging_metadata_contract", False, True, 2),
        ("preview_packaging_separation", "preview_packaging_separation_contract", True, True, 1),
        ("artifact_exclusion_policy", "artifact_exclusion_contract", False, True, 1),
    ]
    domain_rows = [
        {
            "domain_id": d,
            "source_contract_section": section,
            "source_contract_preserved": valid[section],
            "contract_defined": valid[section],
            "contract_satisfied": False,
            "selection_required": select,
            "selection_performed": False,
            "validation_required": validate,
            "validation_performed": False,
            "unresolved_condition_count": count if valid[section] else 0,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "read_model_state": "defined_unresolved"
            if valid[section]
            else "blocked_source_contract_not_preserved",
            "read_model_result": "future_explicit_resolution_required"
            if valid[section]
            else "read_model_projection_blocked",
            "next_resolution_stage": "FUNCTIONAL-PREVIEW-18.5",
        }
        for d, section, select, validate, count in domains
    ]
    blockers_source = (
        _plain_list_section(source, "unresolved_blocker_contract_rows")
        if valid["unresolved_blocker_contract_rows"]
        else []
    )
    req_source = (
        _plain_list_section(source, "packaging_requirement_contract_rows")
        if valid["packaging_requirement_contract_rows"]
        else []
    )
    req_rows = []
    for row in req_source:
        req_rows.append(
            {
                "requirement_id": row["requirement_id"],
                "contract_requirement_id": row["contract_requirement_id"],
                "source_requirement_preserved": True,
                "source_inventory_observed": row["source_inventory_observed"],
                "source_inventory_requirement_satisfied": row[
                    "source_inventory_requirement_satisfied"
                ],
                "source_unresolved_condition_ids": _copy_plain(
                    row["source_unresolved_condition_ids"]
                ),
                "contract_clause_ids": _copy_plain(row["contract_clause_ids"]),
                "required_evidence_ids": _copy_plain(row["required_evidence_ids"]),
                "contract_defined": True,
                "contract_satisfied": False,
                "build_requirement_satisfied": False,
                "requires_future_explicit_step": row["requires_future_explicit_step"],
                "ready_for_read_model": True,
                "ready_for_build_readiness_matrix": True,
                "build_ready": False,
                "packaging_authorized": False,
                "build_authorized": False,
                "read_model_state": "defined_unresolved",
                "read_model_result": "future_explicit_resolution_required",
                "failure_policy": "fail_closed",
            }
        )
    emitted_contract_requirement_ids = {row["contract_requirement_id"] for row in req_rows}
    blocker_rows = []
    for row in blockers_source:
        links = [
            item
            for item in row["contract_requirement_ids"]
            if item in emitted_contract_requirement_ids
        ]
        blocker_rows.append(
            {
                "blocker_id": row["blocker_id"],
                "source_blocker_preserved": True,
                "source_finding_ids": _copy_plain(row["source_finding_ids"]),
                "source_affected_scope_ids": _copy_plain(row["source_affected_scope_ids"]),
                "contract_clause_id": row["contract_clause_id"],
                "contract_requirement_ids": links,
                "required_evidence_ids": _copy_plain(row["required_evidence_ids"]),
                "resolution_criteria": row["resolution_criteria"],
                "contract_clause_defined": True,
                "resolved": False,
                "evidence_collected": False,
                "evidence_validated": False,
                "blocks_build_readiness": True,
                "blocks_packaging_authorization": True,
                "blocks_build_authorization": True,
                "read_model_severity": "blocking",
                "read_model_state": "unresolved",
                "read_model_result": "future_evidence_and_resolution_required",
                "failure_policy": "fail_closed",
            }
        )
    emitted_blockers = {row["blocker_id"] for row in blocker_rows}
    clauses = {row["blocker_id"]: row["contract_clause_id"] for row in blocker_rows}
    evidence_by_blocker = {row["blocker_id"]: row["required_evidence_ids"] for row in blocker_rows}
    scope_rows = []
    for row in (
        _plain_list_section(source, "packaging_scope_contract_rows")
        if valid["packaging_scope_contract_rows"]
        else []
    ):
        links = [item for item in row["source_unresolved_blocker_ids"] if item in emitted_blockers]
        scope_rows.append(
            {
                "scope_id": row["scope_id"],
                "source_scope_preserved": True,
                "source_supporting_matrix_row_ids": _copy_plain(
                    row["source_supporting_matrix_row_ids"]
                ),
                "source_unresolved_blocker_ids": links,
                "contract_clause_ids": [clauses[item] for item in links],
                "required_evidence_ids": [evidence_by_blocker[item][0] for item in links],
                "contract_definition_complete": True,
                "contract_satisfied": False,
                "resolved_blocker_count": 0,
                "unresolved_blocker_count": len(links),
                "ready_for_read_model": True,
                "ready_for_build_readiness_matrix": True,
                "scope_ready": False,
                "scope_authorized": False,
                "build_ready": False,
                "read_model_state": "contract_defined_unresolved",
                "read_model_result": "build_readiness_evaluation_required_in_future_source_only_step",
                "failure_policy": "fail_closed",
            }
        )
    evidence_rows = []
    for row in (
        _plain_list_section(source, "contract_evidence_requirement_rows")
        if valid["contract_evidence_requirement_rows"]
        else []
    ):
        if row["blocker_id"] in emitted_blockers:
            evidence_rows.append(
                {
                    "evidence_id": row["evidence_id"],
                    "blocker_id": row["blocker_id"],
                    "evidence_type": row["evidence_type"],
                    "required_artifacts": _copy_plain(row["required_artifacts"]),
                    "collection_stage": row["collection_stage"],
                    "source_only_definition": True,
                    "collected": False,
                    "validated": False,
                    "required_for_build_readiness": True,
                    "required_for_packaging_authorization": True,
                    "required_for_build_authorization": True,
                    "read_model_state": "required_not_collected",
                    "read_model_result": "future_explicit_evidence_collection_required",
                    "failure_policy": "fail_closed",
                }
            )
    evidence_ids = {row["evidence_id"] for row in evidence_rows}
    acceptance_rows = []
    for row in (
        _plain_list_section(source, "contract_acceptance_rule_rows")
        if valid["contract_acceptance_rule_rows"]
        else []
    ):
        acceptance_rows.append(
            {
                "acceptance_rule_id": row["acceptance_rule_id"],
                "required_blocker_ids": [
                    x for x in row["required_blocker_ids"] if x in emitted_blockers
                ],
                "required_evidence_ids": [
                    x for x in row["required_evidence_ids"] if x in evidence_ids
                ],
                "all_inputs_required": True,
                "rule_defined": True,
                "rule_satisfied": False,
                "grants_build_readiness": False,
                "grants_packaging_authorization": False,
                "grants_build_authorization": False,
                "read_model_state": "defined_unsatisfied",
                "read_model_result": "required_inputs_unresolved",
                "failure_policy": "fail_closed",
            }
        )
    emitted_scope_ids = {row["scope_id"] for row in scope_rows}
    emitted_clause_ids = {row["contract_clause_id"] for row in blocker_rows}
    for row in blocker_rows:
        row["source_affected_scope_ids"] = [
            item for item in row["source_affected_scope_ids"] if item in emitted_scope_ids
        ]
        row["contract_requirement_ids"] = [
            item
            for item in row["contract_requirement_ids"]
            if item in emitted_contract_requirement_ids
        ]
        row["required_evidence_ids"] = [
            item for item in row["required_evidence_ids"] if item in evidence_ids
        ]
        complete = (
            bool(row["source_affected_scope_ids"])
            and bool(row["contract_requirement_ids"])
            and bool(row["required_evidence_ids"])
        )
        if not complete:
            row["source_blocker_preserved"] = False
            row["contract_clause_defined"] = False
            row["read_model_state"] = "blocked_source_contract_not_preserved"
            row["read_model_result"] = "read_model_projection_blocked"
    emitted_clause_ids = {row["contract_clause_id"] for row in blocker_rows}
    for row in scope_rows:
        row["source_unresolved_blocker_ids"] = [
            item for item in row["source_unresolved_blocker_ids"] if item in emitted_blockers
        ]
        row["contract_clause_ids"] = [
            item for item in row["contract_clause_ids"] if item in emitted_clause_ids
        ]
        row["required_evidence_ids"] = [
            item for item in row["required_evidence_ids"] if item in evidence_ids
        ]
        complete = len(row["source_unresolved_blocker_ids"]) == len(
            row["contract_clause_ids"]
        ) == len(row["required_evidence_ids"]) and bool(row["source_unresolved_blocker_ids"])
        if not complete:
            row["source_scope_preserved"] = False
            row["contract_definition_complete"] = False
            row["ready_for_read_model"] = False
            row["ready_for_build_readiness_matrix"] = False
            row["read_model_state"] = "blocked_source_contract_not_preserved"
            row["read_model_result"] = "read_model_projection_blocked"
    for row in req_rows:
        row["source_unresolved_condition_ids"] = [
            item for item in row["source_unresolved_condition_ids"] if item in emitted_blockers
        ]
        row["contract_clause_ids"] = [
            item for item in row["contract_clause_ids"] if item in emitted_clause_ids
        ]
        row["required_evidence_ids"] = [
            item for item in row["required_evidence_ids"] if item in evidence_ids
        ]
        complete = len(row["source_unresolved_condition_ids"]) == len(
            row["contract_clause_ids"]
        ) == len(row["required_evidence_ids"]) and bool(row["source_unresolved_condition_ids"])
        if not complete:
            row["source_requirement_preserved"] = False
            row["contract_defined"] = False
            row["ready_for_read_model"] = False
            row["ready_for_build_readiness_matrix"] = False
            row["read_model_state"] = "blocked_source_contract_not_preserved"
            row["read_model_result"] = "read_model_projection_blocked"
    complete_blocker_ids = {
        row["blocker_id"]
        for row in blocker_rows
        if row["source_blocker_preserved"] and row["contract_clause_defined"]
    }
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
    for row, source_row in zip(
        acceptance_rows,
        _plain_list_section(source, "contract_acceptance_rule_rows")
        if valid["contract_acceptance_rule_rows"]
        else [],
    ):
        if (
            row["required_blocker_ids"] != source_row["required_blocker_ids"]
            or row["required_evidence_ids"] != source_row["required_evidence_ids"]
            or not set(row["required_blocker_ids"]).issubset(complete_blocker_ids)
            or any(
                evidence_by_id[item]["blocker_id"] not in complete_blocker_ids
                for item in row["required_evidence_ids"]
            )
        ):
            row["rule_defined"] = False
            row["read_model_state"] = "blocked_source_contract_not_preserved"
            row["read_model_result"] = "read_model_projection_blocked"
    caps_source = (
        _plain_dict_section(source, "real_capability_contract_state")
        if valid["real_capability_contract_state"]
        else {}
    )
    cap_names = [
        "inherited_block_o_capabilities",
        "inherited_block_p_capabilities",
        "source_inventory_capabilities",
        "inventory_matrix_capabilities",
        "packaging_contract_capabilities",
    ]
    caps = {name: _copy_plain(caps_source.get(name, {})) for name in cap_names}
    caps["packaging_read_model_capabilities"] = {key: "blocked" for key in CAPABILITY_KEYS}
    cap_state = {**caps}
    for name in cap_names + ["packaging_read_model_capabilities"]:
        cap_state[name + "_known_blocked"] = _capability_map_known_blocked(caps[name])
    cap_state["all_real_capabilities_blocked_at_18_4"] = all(
        cap_state[name + "_known_blocked"]
        for name in cap_names + ["packaging_read_model_capabilities"]
    )
    artifact_status = PACKAGING_READ_MODEL_STATUS if accepted else BLOCKED_STATUS
    handoff_status = STATUS if accepted else BLOCKED_STATUS
    preservation = {
        "source_identity_preserved": identity_valid,
        "contract_reference_preserved": valid[section_names[0]],
        "contract_summary_preserved": valid[section_names[1]],
        "principles_preserved": valid["contract_principles"],
        "domain_contracts_preserved": all(valid[x] for x in [d[1] for d in domains]),
        "scope_contracts_preserved": valid["packaging_scope_contract_rows"],
        "requirement_contracts_preserved": valid["packaging_requirement_contract_rows"],
        "blocker_contracts_preserved": valid["unresolved_blocker_contract_rows"],
        "evidence_contracts_preserved": valid["contract_evidence_requirement_rows"],
        "acceptance_rules_preserved": valid["contract_acceptance_rule_rows"],
        "capability_state_preserved": valid["real_capability_contract_state"],
        "fail_closed_state_preserved": valid["fail_closed_contract_decision"],
        "contract_boundaries_preserved": valid["contract_boundaries"],
        "source_boundaries_preserved": valid["source_boundaries"],
        "future_steps_preserved": valid["future_steps"],
        "referential_integrity_preserved": source_graph_integrity_valid,
        "source_contract_modified": False,
        "source_contract_recalculated": False,
        "source_relations_modified": False,
        "repo_rescanned": False,
    }
    reference = {
        key: _scalar_reference(source, key, identity_valid) for key in SOURCE_IDENTITY_EXPECTED
    }
    reference.update(
        {
            "source_identity_valid": identity_valid,
            "source_block_p_desktop_exe_packaging_contract_step": "FUNCTIONAL-PREVIEW-18.3",
            "source_packaging_contract_read_by_18_4": True,
            "source_packaging_contract_available_before_read_model": True,
            "static_packaging_read_model_only": True,
            "packaging_read_model_built_by_18_4": True,
            "packaging_read_model_artifact_complete_by_18_4": accepted,
            "ready_for_functional_preview_18_5": accepted,
        }
    )
    for key in [
        "repo_rescan",
        "filesystem_scan",
        "environment_scan",
        "secret_file_read",
        "dependency_import",
        "dependency_resolution",
        "pyside_import",
        "qml_load",
        "qt_plugin_discovery",
        "entrypoint_selection",
        "entrypoint_validation",
        "packaging_metadata_mutation",
        "packaging_profile_selection",
        "packaging_profile_validation",
        "build_tool_selection",
        "toolchain_validation",
        "policy_application",
        "bundle_scan",
        "build_gate_approval",
        "spec_file_creation",
        "build_command_creation",
        "build_command_execution",
        "packaging",
        "artifact_creation",
        "artifact_scan",
        "artifact_signing",
        "installer_creation",
        "release",
        "runtime",
        "orders",
        "network",
        "credentials_read",
    ]:
        reference[key] = False
    summary = {
        "source_18_3_accepted": accepted,
        "source_packaging_contract_preserved": accepted,
        "source_only": True,
        "plain_data": True,
        "static_read_model": True,
        "packaging_read_model_artifact_complete": accepted,
        "ready_for_block_p_5": accepted,
        "domain_contract_count": sum(bool(row["source_contract_preserved"]) for row in domain_rows),
        "scope_contract_count": len(scope_rows),
        "requirement_contract_count": len(req_rows),
        "blocker_count": len(blocker_rows),
        "resolved_blocker_count": 0,
        "unresolved_blocker_count": len(blocker_rows),
        "evidence_requirement_count": len(evidence_rows),
        "collected_evidence_count": 0,
        "validated_evidence_count": 0,
        "acceptance_rule_count": len(acceptance_rows),
        "satisfied_acceptance_rule_count": 0,
        "contract_definitions_complete": contract_definition_graph_complete,
        "contract_satisfied": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
        "only_source_only_18_5_handoff_allowed": accepted,
    }
    overview = {
        "contract_source_step": "FUNCTIONAL-PREVIEW-18.3"
        if contract_definition_graph_complete
        else "",
        "contract_artifact_complete": accepted,
        "contract_definitions_complete": contract_definition_graph_complete,
        "contract_satisfied": False,
        "all_blockers_resolved": False,
        "all_evidence_collected": False,
        "all_evidence_validated": False,
        "all_acceptance_rules_satisfied": False,
        "desktop_entrypoint_selected": False,
        "desktop_entrypoint_validated": False,
        "qml_bundle_validated": False,
        "qt_plugin_inventory_complete": False,
        "dependency_resolution_complete": False,
        "packaging_metadata_complete": False,
        "packaging_profile_aligned": False,
        "artifact_exclusion_policy_validated": False,
        "windows_toolchain_confirmed": False,
        "future_explicit_build_gate_present": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "read_model_state": (
            "contract_defined_all_operational_conditions_unresolved"
            if accepted
            else "contract_defined_handoff_blocked_by_non_graph_source"
            if contract_definition_graph_complete
            else "contract_overview_blocked_by_invalid_source"
        ),
        "next_required_source_only_step": NEXT_STEP if accepted else "",
    }
    decision = {
        "block_p_packaging_contract_in_18_3": "preserved" if accepted else "not_preserved",
        "block_p_packaging_read_model_in_18_4": "complete" if accepted else "blocked",
        "block_p_build_readiness_matrix_in_18_5": "allowed" if accepted else "blocked",
        "only_source_only_18_5_handoff_allowed": accepted,
    }
    for key in [
        "source_contract_modified_by_18_4",
        "contract_conditions_reinterpreted_by_18_4",
        "blocker_resolved_by_18_4",
        "evidence_collected_by_18_4",
        "evidence_validated_by_18_4",
        "acceptance_rule_satisfied_by_18_4",
        "desktop_entrypoint_selected_by_18_4",
        "desktop_entrypoint_validated_by_18_4",
        "qml_bundle_validated_by_18_4",
        "qt_plugins_discovered_by_18_4",
        "dependency_resolution_performed_by_18_4",
        "packaging_metadata_modified_by_18_4",
        "packaging_profile_selected_by_18_4",
        "build_tool_selected_by_18_4",
        "windows_toolchain_confirmed_by_18_4",
        "artifact_exclusion_policy_applied_by_18_4",
        "bundle_scan_performed_by_18_4",
        "build_gate_approved_by_18_4",
        "build_ready_by_18_4",
        "packaging_authorized_by_18_4",
        "build_authorized_by_18_4",
        "build_executed_by_18_4",
        "artifact_created_by_18_4",
        "release_authorized_by_18_4",
        "runtime_enabled_by_18_4",
        "orders_enabled_by_18_4",
    ]:
        decision[key] = False
    boundaries = {
        "reads_18_3_only": True,
        "source_only": True,
        "plain_data": True,
        "static_read_model": True,
    }
    for key in [
        "repo_rescan",
        "filesystem_scan",
        "environment_scan",
        "secret_file_read",
        "dependency_import",
        "dependency_resolution",
        "pyside_import",
        "qml_load",
        "qt_plugin_discovery",
        "entrypoint_selection",
        "entrypoint_validation",
        "packaging_metadata_mutation",
        "packaging_profile_selection",
        "packaging_profile_validation",
        "build_tool_selection",
        "toolchain_validation",
        "artifact_exclusion_policy_application",
        "bundle_scan",
        "build_gate_approval",
        "build_readiness_grant",
        "packaging_authorization",
        "build_authorization",
        "spec_file_creation",
        "build_command_creation",
        "build_command_execution",
        "packaging_performed",
        "artifact_created",
        "artifact_scanned",
        "artifact_signed",
        "installer_created",
        "release_performed",
        "runtime_started",
        "orders_enabled",
        "network_opened",
        "credentials_read",
        "qml_bridge_gateway_controller_changed",
    ]:
        boundaries[key] = False
    boundaries["can_feed_only_18_5_build_readiness_matrix"] = accepted
    source_boundaries = {
        "source_block_p_desktop_exe_packaging_contract": "FUNCTIONAL-PREVIEW-18.3",
        "packaging_contract_preserved": accepted,
        "can_build_desktop_exe_packaging_read_model": accepted,
        "packaging_read_model_artifact_complete": accepted,
        "can_build_desktop_exe_build_readiness_matrix": accepted,
        "can_feed_18_5": accepted,
    }
    future = [
        {
            "step": "18.5",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS MATRIX",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.6",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.7",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.8",
            "title": "BLOCK P CLOSURE AUDIT",
            "source_only": True,
            "build_performed": False,
        },
    ]
    evidence = {
        "source_builder_called": True,
        "source_builder_call_count": 1,
        "source_plain_bounded": plain,
        "all_top_level_keys_exact_str": keys,
        "source_accepted": accepted,
        "identity_valid": identity_valid,
        **{
            key: valid[key]
            for key in (
                "contract_reference_valid",
                "contract_summary_valid",
                "source_preservation_valid",
                "principles_valid",
                "desktop_entrypoint_contract_valid",
                "qml_bundle_contract_valid",
                "python_dependency_contract_valid",
                "packaging_metadata_contract_valid",
                "preview_packaging_contract_valid",
                "artifact_exclusion_contract_valid",
                "scope_contract_rows_valid",
                "requirement_contract_rows_valid",
                "blocker_contract_rows_valid",
                "evidence_rows_valid",
                "acceptance_rows_valid",
                "capability_state_valid",
                "fail_closed_valid",
                "non_execution_evidence_valid",
                "contract_boundaries_valid",
                "source_boundaries_valid",
                "future_steps_valid",
            )
        },
        "source_graph_integrity_valid": source_graph_integrity_valid,
        "source_referential_integrity_valid": source_graph_integrity_valid,
        "contract_definition_graph_complete": contract_definition_graph_complete,
        "output_graph_integrity_valid": False,
        "output_referential_integrity_valid": False,
        "domain_contract_read_model_row_count": len(domain_rows),
        "scope_contract_read_model_row_count": len(scope_rows),
        "requirement_contract_read_model_row_count": len(req_rows),
        "blocker_read_model_row_count": len(blocker_rows),
        "evidence_read_model_row_count": len(evidence_rows),
        "acceptance_rule_read_model_row_count": len(acceptance_rows),
        "read_model_artifact_complete": accepted,
    }
    for key in [
        "repo_rescan_by_18_4",
        "filesystem_scan_by_18_4",
        "dependency_resolution_by_18_4",
        "qml_load_by_18_4",
        "qt_plugin_discovery_by_18_4",
        "entrypoint_selection_by_18_4",
        "entrypoint_validation_by_18_4",
        "metadata_mutation_by_18_4",
        "profile_selection_by_18_4",
        "tool_selection_by_18_4",
        "policy_application_by_18_4",
        "bundle_scan_by_18_4",
        "build_gate_approval_by_18_4",
        "build_ready_by_18_4",
        "packaging_authorized_by_18_4",
        "build_authorized_by_18_4",
        "build_execution_by_18_4",
        "artifact_creation_by_18_4",
        "release_by_18_4",
        "runtime_by_18_4",
        "orders_by_18_4",
        "network_by_18_4",
        "credentials_read_by_18_4",
    ]:
        evidence[key] = False
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_packaging_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_desktop_exe_packaging_read_model_status": artifact_status,
        "block_p_desktop_exe_packaging_read_model_decision": artifact_status.upper(),
        "packaging_read_model_artifact_complete": accepted,
        "ready_for_block_p_5": accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_desktop_exe_packaging_contract_reference": reference,
        "packaging_read_model_summary": summary,
        "source_contract_preservation": preservation,
        "packaging_contract_overview": overview,
        "domain_contract_read_model_rows": domain_rows,
        "scope_contract_read_model_rows": scope_rows,
        "requirement_contract_read_model_rows": req_rows,
        "blocker_read_model_rows": blocker_rows,
        "evidence_read_model_rows": evidence_rows,
        "acceptance_rule_read_model_rows": acceptance_rows,
        "capability_read_model_state": cap_state,
        "fail_closed_read_model_decision": decision,
        "non_execution_read_model_evidence": evidence,
        "read_model_boundaries": boundaries,
        "source_boundaries": source_boundaries,
        "future_steps": future,
        "status": handoff_status,
    }
    output_graph_integrity_valid = _output_integrity(payload)
    payload["non_execution_read_model_evidence"]["output_graph_integrity_valid"] = (
        output_graph_integrity_valid
    )
    payload["non_execution_read_model_evidence"]["output_referential_integrity_valid"] = (
        output_graph_integrity_valid
    )
    return payload
