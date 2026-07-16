"""FUNCTIONAL-PREVIEW-18.3 Block P desktop EXE packaging contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_packaging_inventory_matrix import (
    build_preview_block_p_desktop_exe_packaging_inventory_matrix,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_packaging_contract.v1"
KIND: Final[str] = "functional_preview_block_p_desktop_exe_packaging_contract"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.3"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-18.4"
NEXT_STEP_TITLE: Final[str] = "BLOCK P DESKTOP EXE PACKAGING READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_18_4_block_p_desktop_exe_packaging_read_model"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_4_block_p_desktop_exe_packaging_contract_source_not_accepted"
)
PACKAGING_CONTRACT_STATUS: Final[str] = (
    "source_18_2_consumed_inventory_matrix_preserved_static_plain_data_contract_complete_3_scopes_8_requirements_12_blockers_12_evidence_6_acceptance_rules_defined_zero_resolution_approval_readiness_authorization_build_runtime_orders_only_source_only_handoff_to_18_4_allowed"
)
PACKAGING_CONTRACT_DECISION: Final[str] = PACKAGING_CONTRACT_STATUS.upper()
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
TOP_LEVEL_FIELDS: Final[list[str]] = [
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
TOP_LEVEL_FIELDS_18_2: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_packaging_inventory_matrix_kind",
    "block",
    "step",
    "block_p_desktop_exe_packaging_inventory_matrix_status",
    "block_p_desktop_exe_packaging_inventory_matrix_decision",
    "inventory_matrix_artifact_complete",
    "ready_for_block_p_3",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_packaging_source_inventory_reference",
    "inventory_matrix_summary",
    "source_inventory_preservation",
    "desktop_entrypoint_matrix_rows",
    "qml_bundle_matrix_rows",
    "python_dependency_matrix_rows",
    "packaging_metadata_matrix_rows",
    "existing_preview_packaging_matrix_rows",
    "artifact_exclusion_policy_matrix_rows",
    "inventory_finding_matrix_rows",
    "packaging_scope_matrix_rows",
    "packaging_requirement_matrix_rows",
    "unresolved_contract_blocker_rows",
    "real_capability_matrix_state",
    "fail_closed_matrix_decision",
    "non_execution_matrix_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SOURCE_BOUNDARY_FIELDS_18_3: Final[list[str]] = [
    "source_block_p_desktop_exe_packaging_inventory_matrix",
    "inventory_matrix_preserved",
    "can_build_desktop_exe_packaging_contract",
    "packaging_contract_artifact_complete",
    "can_build_desktop_exe_packaging_read_model",
    "can_feed_18_4",
]
SUMMARY_OWNED_FIELDS_18_3: Final[list[str]] = [
    "packaging_contract_artifact_complete",
    "contract_definitions_complete",
    "contract_satisfied",
    "ready_for_block_p_4",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
]
FAIL_CLOSED_OWNED_FIELDS_18_3: Final[list[str]] = [
    "block_p_inventory_matrix_preserved_in_18_3",
    "block_p_packaging_contract_complete_in_18_3",
    "only_source_only_18_4_handoff_allowed",
    "build_ready_by_18_3",
    "packaging_authorized_by_18_3",
    "build_authorized_by_18_3",
]
SOURCE_BOUNDARY_OWNED_FIELDS_18_3: Final[list[str]] = SOURCE_BOUNDARY_FIELDS_18_3
EXPECTED_SOURCE: Final[dict[str, Any]] = {
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
    "block_p_desktop_exe_packaging_source_inventory_reference": {
        "schema_version": "preview_block_p_desktop_exe_packaging_source_inventory.v1",
        "block_p_desktop_exe_packaging_source_inventory_kind": "functional_preview_block_p_desktop_exe_packaging_source_inventory",
        "block": "P",
        "step": "18.1",
        "block_p_desktop_exe_packaging_source_inventory_status": "source_18_0_consumed_block_o_closed_block_p_open_source_only_plain_data_static_inventory_current_repository_packaging_sources_inventoried_desktop_entrypoint_candidates_observed_qml_roots_and_assets_observed_dependency_declarations_observed_cli_preview_packaging_sources_observed_separately_package_discovery_observations_recorded_exclusion_policy_observed_not_applied_inventory_artifact_complete_for_18_1_no_approval_no_validation_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_2_allowed",
        "block_p_desktop_exe_packaging_source_inventory_decision": "SOURCE_18_0_CONSUMED_BLOCK_O_CLOSED_BLOCK_P_OPEN_SOURCE_ONLY_PLAIN_DATA_STATIC_INVENTORY_CURRENT_REPOSITORY_PACKAGING_SOURCES_INVENTORIED_DESKTOP_ENTRYPOINT_CANDIDATES_OBSERVED_QML_ROOTS_AND_ASSETS_OBSERVED_DEPENDENCY_DECLARATIONS_OBSERVED_CLI_PREVIEW_PACKAGING_SOURCES_OBSERVED_SEPARATELY_PACKAGE_DISCOVERY_OBSERVATIONS_RECORDED_EXCLUSION_POLICY_OBSERVED_NOT_APPLIED_INVENTORY_ARTIFACT_COMPLETE_FOR_18_1_NO_APPROVAL_NO_VALIDATION_NO_PACKAGING_NO_BUILD_NO_ARTIFACT_NO_RELEASE_NO_RUNTIME_NO_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_2_ALLOWED",
        "source_inventory_artifact_complete": True,
        "ready_for_block_p_2": True,
        "next_step": "FUNCTIONAL-PREVIEW-18.2",
        "next_step_title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
        "status": "ready_for_functional_preview_18_2_block_p_desktop_exe_packaging_inventory_matrix",
        "source_block_p_desktop_exe_packaging_source_inventory_step": "FUNCTIONAL-PREVIEW-18.1",
        "source_inventory_read_by_18_2": True,
        "source_inventory_available_before_matrix": True,
        "static_inventory_matrix_only": True,
        "inventory_matrix_built_by_18_2": True,
        "inventory_matrix_artifact_complete_by_18_2": True,
        "ready_for_functional_preview_18_3": True,
        "runtime_filesystem_scan": False,
        "repo_rescan": False,
        "environment_scan": False,
        "secret_file_read": False,
        "dependency_import": False,
        "dependency_resolution": False,
        "pyside_import": False,
        "qml_load": False,
        "qt_plugin_discovery": False,
        "packaging_profile_validation": False,
        "build_tool_selection": False,
        "build_tool_execution": False,
        "spec_file_creation": False,
        "build_command_creation": False,
        "build_command_execution": False,
        "packaging": False,
        "artifact_creation": False,
        "artifact_scan": False,
        "artifact_signing": False,
        "installer_creation": False,
        "release": False,
        "runtime": False,
        "orders": False,
        "network": False,
        "credentials_read": False,
    },
    "inventory_matrix_summary": {
        "source_18_1_accepted": True,
        "source_inventory_artifact_preserved": True,
        "source_only": True,
        "plain_data": True,
        "static_matrix": True,
        "inventory_matrix_artifact_complete": True,
        "inventory_matrix_evaluated": True,
        "desktop_entrypoint_matrix_row_count": 4,
        "qml_bundle_matrix_row_count": 5,
        "python_dependency_matrix_row_count": 4,
        "packaging_metadata_matrix_row_count": 4,
        "existing_preview_packaging_matrix_row_count": 4,
        "artifact_exclusion_policy_matrix_row_count": 1,
        "inventory_finding_matrix_row_count": 11,
        "packaging_scope_matrix_row_count": 3,
        "packaging_requirement_matrix_row_count": 8,
        "unresolved_contract_blocker_count": 12,
        "all_matrix_rows_evaluated": True,
        "unresolved_contract_blockers_present": True,
        "packaging_contract_conditions_satisfied": False,
        "desktop_entrypoint_selected": False,
        "desktop_entrypoint_approved": False,
        "qml_bundle_validated": False,
        "qt_plugin_inventory_complete": False,
        "dependency_bundle_validated": False,
        "packaging_profile_aligned": False,
        "artifact_exclusion_policy_validated": False,
        "windows_toolchain_confirmed": False,
        "future_explicit_build_execution_gate_present": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
        "only_source_only_18_3_handoff_allowed": True,
    },
    "source_inventory_preservation": {
        "source_inventory_identity": {
            "schema_version": "preview_block_p_desktop_exe_packaging_source_inventory.v1",
            "block_p_desktop_exe_packaging_source_inventory_kind": "functional_preview_block_p_desktop_exe_packaging_source_inventory",
            "block": "P",
            "step": "18.1",
            "block_p_desktop_exe_packaging_source_inventory_status": "source_18_0_consumed_block_o_closed_block_p_open_source_only_plain_data_static_inventory_current_repository_packaging_sources_inventoried_desktop_entrypoint_candidates_observed_qml_roots_and_assets_observed_dependency_declarations_observed_cli_preview_packaging_sources_observed_separately_package_discovery_observations_recorded_exclusion_policy_observed_not_applied_inventory_artifact_complete_for_18_1_no_approval_no_validation_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_2_allowed",
            "block_p_desktop_exe_packaging_source_inventory_decision": "SOURCE_18_0_CONSUMED_BLOCK_O_CLOSED_BLOCK_P_OPEN_SOURCE_ONLY_PLAIN_DATA_STATIC_INVENTORY_CURRENT_REPOSITORY_PACKAGING_SOURCES_INVENTORIED_DESKTOP_ENTRYPOINT_CANDIDATES_OBSERVED_QML_ROOTS_AND_ASSETS_OBSERVED_DEPENDENCY_DECLARATIONS_OBSERVED_CLI_PREVIEW_PACKAGING_SOURCES_OBSERVED_SEPARATELY_PACKAGE_DISCOVERY_OBSERVATIONS_RECORDED_EXCLUSION_POLICY_OBSERVED_NOT_APPLIED_INVENTORY_ARTIFACT_COMPLETE_FOR_18_1_NO_APPROVAL_NO_VALIDATION_NO_PACKAGING_NO_BUILD_NO_ARTIFACT_NO_RELEASE_NO_RUNTIME_NO_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_2_ALLOWED",
            "source_inventory_artifact_complete": True,
            "ready_for_block_p_2": True,
            "next_step": "FUNCTIONAL-PREVIEW-18.2",
            "next_step_title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
            "status": "ready_for_functional_preview_18_2_block_p_desktop_exe_packaging_inventory_matrix",
        },
        "desktop_entrypoint_row_count": 4,
        "pyside_qml_file_count": 24,
        "shared_qml_file_count": 107,
        "additional_qml_support_asset_count": 0,
        "deploy_packaging_source_file_count": 46,
        "deployment_documentation_file_count": 17,
        "config_reference_row_count": 6,
        "project_dependency_count": 25,
        "desktop_optional_dependency_count": 3,
        "inventory_finding_count": 11,
        "cli_preview_remains_separate": True,
        "all_source_approvals_false": True,
        "all_source_build_runtime_order_flags_false": True,
        "source_inventory_preserved": True,
        "source_inventory_recalculated": False,
        "repo_rescanned": False,
        "inventory_paths_modified": False,
        "inventory_findings_modified": False,
    },
    "desktop_entrypoint_matrix_rows": [
        {
            "matrix_row_id": "desktop_module_launcher_matrix",
            "source_inventory_row_id": "desktop_module_launcher",
            "path": "ui/pyside_app/__main__.py",
            "source_kind": "python_module_launcher",
            "observed_role": "desktop_module_launcher",
            "observed_symbol": "ui.pyside_app.app.main",
            "source_observed": True,
            "source_inventory_preserved": True,
            "desktop_entrypoint_candidate": True,
            "excluded_from_final_desktop_contract": False,
            "selection_required": True,
            "validation_required": True,
            "selected_as_final_desktop_entrypoint": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "desktop_launcher_candidate_requires_contract_selection",
            "matrix_result": "desktop_launcher_observed_selection_and_validation_pending",
        },
        {
            "matrix_row_id": "desktop_application_main_matrix",
            "source_inventory_row_id": "desktop_application_main",
            "path": "ui/pyside_app/app.py",
            "source_kind": "python_desktop_application_source",
            "observed_role": "pyside_qml_application_entrypoint",
            "observed_symbol": "main",
            "source_observed": True,
            "source_inventory_preserved": True,
            "desktop_entrypoint_candidate": True,
            "excluded_from_final_desktop_contract": False,
            "selection_required": True,
            "validation_required": True,
            "selected_as_final_desktop_entrypoint": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "desktop_application_candidate_requires_contract_selection",
            "matrix_result": "desktop_application_entrypoint_observed_selection_and_validation_pending",
        },
        {
            "matrix_row_id": "cli_preview_run_local_bot_matrix",
            "source_inventory_row_id": "cli_preview_run_local_bot",
            "path": "scripts/run_local_bot.py",
            "source_kind": "python_cli_preview_source",
            "observed_role": "cli_preview_entrypoint",
            "observed_symbol": "main",
            "source_observed": True,
            "source_inventory_preserved": True,
            "desktop_entrypoint_candidate": False,
            "excluded_from_final_desktop_contract": True,
            "selection_required": False,
            "validation_required": False,
            "selected_as_final_desktop_entrypoint": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_source_excluded_from_final_desktop_contract",
            "matrix_result": "cli_preview_source_preserved_as_separate_scope_not_final_desktop_entrypoint",
        },
        {
            "matrix_row_id": "cli_preview_operator_preview_bundle_matrix",
            "source_inventory_row_id": "cli_preview_operator_preview_bundle",
            "path": "scripts/operator_preview_bundle.py",
            "source_kind": "python_cli_preview_source",
            "observed_role": "cli_preview_operator_bundle",
            "observed_symbol": "main",
            "source_observed": True,
            "source_inventory_preserved": True,
            "desktop_entrypoint_candidate": False,
            "excluded_from_final_desktop_contract": True,
            "selection_required": False,
            "validation_required": False,
            "selected_as_final_desktop_entrypoint": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_source_excluded_from_final_desktop_contract",
            "matrix_result": "cli_preview_source_preserved_as_separate_scope_not_final_desktop_entrypoint",
        },
    ],
    "qml_bundle_matrix_rows": [
        {
            "matrix_row_id": "default_qml_entrypoint",
            "source_paths": ["ui/pyside_app/qml/MainWindow.qml"],
            "source_inventory_present": True,
            "source_inventory_complete": True,
            "source_inventory_preserved": True,
            "matrix_evaluated": True,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "qml_source_observed_requires_packaging_contract_validation",
            "matrix_result": "qml_source_observed_contract_validation_pending",
        },
        {
            "matrix_row_id": "pyside_qml_root",
            "source_paths": ["ui/pyside_app/qml"],
            "source_inventory_present": True,
            "source_inventory_complete": True,
            "source_inventory_preserved": True,
            "matrix_evaluated": True,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "qml_source_observed_requires_packaging_contract_validation",
            "matrix_result": "qml_source_observed_contract_validation_pending",
        },
        {
            "matrix_row_id": "shared_qml_root",
            "source_paths": ["ui/qml"],
            "source_inventory_present": True,
            "source_inventory_complete": True,
            "source_inventory_preserved": True,
            "matrix_evaluated": True,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "qml_source_observed_requires_packaging_contract_validation",
            "matrix_result": "qml_source_observed_contract_validation_pending",
        },
        {
            "matrix_row_id": "styles_module",
            "source_paths": [
                "ui/pyside_app/qml/Styles/qmldir",
                "ui/pyside_app/qml/Styles/DesignSystem.qml",
            ],
            "source_inventory_present": True,
            "source_inventory_complete": True,
            "source_inventory_preserved": True,
            "matrix_evaluated": True,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "qml_source_observed_requires_packaging_contract_validation",
            "matrix_result": "qml_source_observed_contract_validation_pending",
        },
        {
            "matrix_row_id": "windows_shared_qml_import_path",
            "source_paths": ["ui/qml"],
            "source_inventory_present": True,
            "source_inventory_complete": True,
            "source_inventory_preserved": True,
            "matrix_evaluated": True,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": True,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "windows_shared_qml_import_path_requires_resolution",
            "matrix_result": "shared_qml_root_platform_condition_observed_windows_bundle_contract_unresolved",
        },
    ],
    "python_dependency_matrix_rows": [
        {
            "matrix_row_id": "project_dependency_declarations",
            "source_inventory_present": True,
            "source_inventory_preserved": True,
            "declaration_inventory_complete": True,
            "resolution_required": True,
            "resolution_performed": False,
            "selection_required": False,
            "selection_performed": False,
            "validated": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "dependency_declarations_observed_resolution_pending",
            "matrix_result": "dependency_resolution_pending",
        },
        {
            "matrix_row_id": "desktop_optional_dependency_declarations",
            "source_inventory_present": True,
            "source_inventory_preserved": True,
            "declaration_inventory_complete": True,
            "resolution_required": True,
            "resolution_performed": False,
            "selection_required": False,
            "selection_performed": False,
            "validated": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "dependency_declarations_observed_resolution_pending",
            "matrix_result": "dependency_resolution_pending",
        },
        {
            "matrix_row_id": "dependency_resolution",
            "source_inventory_present": False,
            "source_inventory_preserved": False,
            "declaration_inventory_complete": False,
            "resolution_required": True,
            "resolution_performed": False,
            "selection_required": False,
            "selection_performed": False,
            "validated": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "dependency_resolution_not_performed",
            "matrix_result": "dependency_resolution_pending",
        },
        {
            "matrix_row_id": "desktop_build_tool_candidates",
            "source_inventory_present": True,
            "source_inventory_preserved": True,
            "declaration_inventory_complete": False,
            "resolution_required": False,
            "resolution_performed": False,
            "selection_required": True,
            "selection_performed": False,
            "validated": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "desktop_build_tool_candidates_observed_not_selected",
            "matrix_result": "dependency_resolution_pending",
        },
    ],
    "packaging_metadata_matrix_rows": [
        {
            "matrix_row_id": "setuptools_ui_package_discovery",
            "source_inventory_present": True,
            "required_declaration_present": False,
            "inventory_complete": None,
            "unresolved_condition_present": True,
            "validation_required": False,
            "validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "setuptools_ui_package_discovery_missing",
            "matrix_result": "setuptools_ui_package_discovery_missing_pending",
        },
        {
            "matrix_row_id": "qml_package_data_declaration",
            "source_inventory_present": True,
            "required_declaration_present": False,
            "inventory_complete": None,
            "unresolved_condition_present": True,
            "validation_required": False,
            "validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "qml_package_data_declaration_missing",
            "matrix_result": "qml_package_data_declaration_missing_pending",
        },
        {
            "matrix_row_id": "deploy_packaging_sources",
            "source_inventory_present": True,
            "required_declaration_present": None,
            "inventory_complete": True,
            "unresolved_condition_present": False,
            "validation_required": True,
            "validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "deployment_source_inventory_observed_requires_contract_selection",
            "matrix_result": "deployment_source_inventory_observed_requires_contract_selection_pending",
        },
        {
            "matrix_row_id": "deployment_documentation",
            "source_inventory_present": True,
            "required_declaration_present": None,
            "inventory_complete": True,
            "unresolved_condition_present": False,
            "validation_required": True,
            "validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "deployment_source_inventory_observed_requires_contract_selection",
            "matrix_result": "deployment_source_inventory_observed_requires_contract_selection_pending",
        },
    ],
    "existing_preview_packaging_matrix_rows": [
        {
            "matrix_row_id": "safe_exe_preview_build_plan",
            "source_inventory_present": True,
            "source_scope": "cli_preview",
            "targets_run_local_bot": True,
            "targets_final_desktop_entrypoint": False,
            "final_desktop_profile_aligned": False,
            "reusable_as_final_desktop_contract": False,
            "profile_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_packaging_source_not_aligned_to_final_desktop_contract",
            "matrix_result": "cli_preview_packaging_source_preserved_not_final_desktop_contract",
        },
        {
            "matrix_row_id": "windows_preview_profile",
            "source_inventory_present": True,
            "source_scope": "cli_preview",
            "targets_run_local_bot": True,
            "targets_final_desktop_entrypoint": False,
            "final_desktop_profile_aligned": False,
            "reusable_as_final_desktop_contract": False,
            "profile_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_packaging_source_not_aligned_to_final_desktop_contract",
            "matrix_result": "cli_preview_packaging_source_preserved_not_final_desktop_contract",
        },
        {
            "matrix_row_id": "linux_preview_profile",
            "source_inventory_present": True,
            "source_scope": "cli_preview",
            "targets_run_local_bot": True,
            "targets_final_desktop_entrypoint": False,
            "final_desktop_profile_aligned": False,
            "reusable_as_final_desktop_contract": False,
            "profile_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_packaging_source_not_aligned_to_final_desktop_contract",
            "matrix_result": "cli_preview_packaging_source_preserved_not_final_desktop_contract",
        },
        {
            "matrix_row_id": "macos_preview_profile",
            "source_inventory_present": True,
            "source_scope": "cli_preview",
            "targets_run_local_bot": True,
            "targets_final_desktop_entrypoint": False,
            "final_desktop_profile_aligned": False,
            "reusable_as_final_desktop_contract": False,
            "profile_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_packaging_source_not_aligned_to_final_desktop_contract",
            "matrix_result": "cli_preview_packaging_source_preserved_not_final_desktop_contract",
        },
    ],
    "artifact_exclusion_policy_matrix_rows": [
        {
            "matrix_row_id": "artifact_exclusion_policy",
            "policy_source": "scripts/safe_exe_preview_build_plan.py",
            "policy_version": "security_packaging_artifact_policy.v1",
            "policy_observed": True,
            "denied_patterns_inventory_preserved": True,
            "policy_application_required": True,
            "policy_applied": False,
            "desktop_bundle_validation_required": True,
            "desktop_bundle_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "artifact_exclusion_policy_observed_not_validated_for_desktop_bundle",
            "matrix_result": "artifact_exclusion_policy_requires_contract_application_and_validation",
        }
    ],
    "inventory_finding_matrix_rows": [
        {
            "finding_id": "desktop_module_launcher_observed",
            "source_paths": ["ui/pyside_app/__main__.py"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "entrypoint_candidate_requires_contract_selection",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "desktop_application_main_observed",
            "source_paths": ["ui/pyside_app/app.py"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "entrypoint_candidate_requires_contract_selection",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "default_qml_entrypoint_observed",
            "source_paths": ["ui/pyside_app/qml/MainWindow.qml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "qml_entrypoint_requires_contract_validation",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "two_qml_source_roots_observed",
            "source_paths": ["ui/pyside_app/qml", "ui/qml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "qml_roots_require_bundle_contract_definition",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "shared_qml_import_path_is_platform_conditional",
            "source_paths": ["ui/pyside_app/app.py", "ui/pyside_app/qml/MainWindow.qml", "ui/qml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "windows_qml_import_path_requires_resolution",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "cli_preview_plan_targets_non_desktop_entrypoint",
            "source_paths": [
                "scripts/safe_exe_preview_build_plan.py",
                "deploy/packaging/profiles/preview/windows.toml",
                "deploy/packaging/profiles/preview/linux.toml",
                "deploy/packaging/profiles/preview/macos.toml",
            ],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "cli_preview_packaging_not_final_desktop_contract",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "desktop_build_tools_declared_as_optional_dependencies",
            "source_paths": ["pyproject.toml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": False,
            "resolved": False,
            "approved": False,
            "matrix_classification": "build_tool_candidates_observed_selection_pending",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "ui_package_discovery_not_declared_in_current_setuptools_include",
            "source_paths": ["pyproject.toml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "required_packaging_metadata_missing",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "qml_package_data_not_declared_in_current_setuptools_metadata",
            "source_paths": ["pyproject.toml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "required_qml_package_data_missing",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "example_config_references_local_secret_paths",
            "source_paths": ["ui/config/example.yaml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "secret_path_exclusion_contract_required",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
        {
            "finding_id": "example_config_contains_sensitive_field_reference",
            "source_paths": ["ui/config/example.yaml"],
            "source_observation_preserved": True,
            "source_severity_classification": "inventory_observation",
            "matrix_evaluated": True,
            "requires_packaging_contract_action": True,
            "contract_blocker_present": True,
            "resolved": False,
            "approved": False,
            "matrix_classification": "sensitive_field_exclusion_contract_required",
            "matrix_result": "inventory_finding_evaluated_contract_action_pending",
        },
    ],
    "packaging_scope_matrix_rows": [
        {
            "scope_id": "desktop_application_entrypoint",
            "source_inventory_artifact_present": True,
            "inventory_matrix_evaluated": True,
            "supporting_matrix_row_ids": [
                "desktop_module_launcher_matrix",
                "desktop_application_main_matrix",
            ],
            "resolved_condition_count": 0,
            "unresolved_condition_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            "unresolved_condition_count": 2,
            "ready_for_packaging_contract": False,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "entrypoint_inventory_complete_contract_conditions_unresolved",
            "matrix_result": "scope_contract_conditions_unresolved",
        },
        {
            "scope_id": "qt_qml_runtime_bundle",
            "source_inventory_artifact_present": True,
            "inventory_matrix_evaluated": True,
            "supporting_matrix_row_ids": [
                "default_qml_entrypoint",
                "pyside_qml_root",
                "shared_qml_root",
                "styles_module",
                "windows_shared_qml_import_path",
                "setuptools_ui_package_discovery",
                "qml_package_data_declaration",
            ],
            "resolved_condition_count": 0,
            "unresolved_condition_ids": [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "qt_plugin_inventory_missing",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            "unresolved_condition_count": 5,
            "ready_for_packaging_contract": False,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "qml_inventory_complete_runtime_bundle_contract_conditions_unresolved",
            "matrix_result": "scope_contract_conditions_unresolved",
        },
        {
            "scope_id": "windows_exe_artifact_pipeline",
            "source_inventory_artifact_present": True,
            "inventory_matrix_evaluated": True,
            "supporting_matrix_row_ids": [
                "project_dependency_declarations",
                "desktop_optional_dependency_declarations",
                "dependency_resolution",
                "desktop_build_tool_candidates",
                "safe_exe_preview_build_plan",
                "windows_preview_profile",
                "artifact_exclusion_policy",
            ],
            "resolved_condition_count": 0,
            "unresolved_condition_ids": [
                "final_desktop_packaging_profile_not_aligned",
                "dependency_resolution_not_performed",
                "secret_and_local_data_exclusion_policy_not_validated",
                "windows_toolchain_not_confirmed",
                "future_explicit_build_execution_gate_missing",
            ],
            "unresolved_condition_count": 5,
            "ready_for_packaging_contract": False,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "packaging_source_inventory_complete_final_desktop_pipeline_unresolved",
            "matrix_result": "scope_contract_conditions_unresolved",
        },
    ],
    "packaging_requirement_matrix_rows": [
        {
            "requirement_id": "desktop_application_entrypoint_inventory",
            "required": True,
            "source_inventory_observed": True,
            "inventory_requirement_satisfied": True,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": False,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "qml_asset_inventory",
            "required": True,
            "source_inventory_observed": True,
            "inventory_requirement_satisfied": True,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": False,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "qt_plugin_inventory",
            "required": True,
            "source_inventory_observed": False,
            "inventory_requirement_satisfied": False,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": True,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["qt_plugin_inventory_missing"],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_missing_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "python_dependency_inventory",
            "required": True,
            "source_inventory_observed": True,
            "inventory_requirement_satisfied": True,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": False,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["dependency_resolution_not_performed"],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "packaging_profile_alignment",
            "required": True,
            "source_inventory_observed": True,
            "inventory_requirement_satisfied": True,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": False,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["final_desktop_packaging_profile_not_aligned"],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "secret_and_local_data_exclusion_policy",
            "required": True,
            "source_inventory_observed": True,
            "inventory_requirement_satisfied": True,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": False,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["secret_and_local_data_exclusion_policy_not_validated"],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "windows_target_toolchain_confirmation",
            "required": True,
            "source_inventory_observed": False,
            "inventory_requirement_satisfied": False,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": True,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["windows_toolchain_not_confirmed"],
            "requires_future_explicit_step": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_missing_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
        {
            "requirement_id": "future_explicit_build_execution_gate",
            "required": True,
            "source_inventory_observed": False,
            "inventory_requirement_satisfied": False,
            "matrix_evaluated": True,
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": True,
            "unresolved_for_contract": True,
            "unresolved_condition_ids": ["future_explicit_build_execution_gate_missing"],
            "requires_future_explicit_step": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_missing_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        },
    ],
    "unresolved_contract_blocker_rows": [
        {
            "blocker_id": "final_desktop_entrypoint_not_selected",
            "source_finding_ids": [
                "desktop_module_launcher_observed",
                "desktop_application_main_observed",
            ],
            "affected_scope_ids": ["desktop_application_entrypoint"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "desktop_entrypoint_validation_not_performed",
            "source_finding_ids": [
                "desktop_module_launcher_observed",
                "desktop_application_main_observed",
            ],
            "affected_scope_ids": ["desktop_application_entrypoint"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "qml_bundle_validation_not_performed",
            "source_finding_ids": [
                "default_qml_entrypoint_observed",
                "two_qml_source_roots_observed",
            ],
            "affected_scope_ids": ["qt_qml_runtime_bundle"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "windows_shared_qml_import_path_unresolved",
            "source_finding_ids": ["shared_qml_import_path_is_platform_conditional"],
            "affected_scope_ids": ["qt_qml_runtime_bundle"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "qt_plugin_inventory_missing",
            "source_finding_ids": [],
            "affected_scope_ids": ["qt_qml_runtime_bundle"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "ui_package_discovery_missing",
            "source_finding_ids": [
                "ui_package_discovery_not_declared_in_current_setuptools_include"
            ],
            "affected_scope_ids": ["qt_qml_runtime_bundle"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "qml_package_data_missing",
            "source_finding_ids": ["qml_package_data_not_declared_in_current_setuptools_metadata"],
            "affected_scope_ids": ["qt_qml_runtime_bundle"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "final_desktop_packaging_profile_not_aligned",
            "source_finding_ids": ["cli_preview_plan_targets_non_desktop_entrypoint"],
            "affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "dependency_resolution_not_performed",
            "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
            "affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "secret_and_local_data_exclusion_policy_not_validated",
            "source_finding_ids": [
                "example_config_references_local_secret_paths",
                "example_config_contains_sensitive_field_reference",
            ],
            "affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "windows_toolchain_not_confirmed",
            "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
            "affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
        {
            "blocker_id": "future_explicit_build_execution_gate_missing",
            "source_finding_ids": [],
            "affected_scope_ids": ["windows_exe_artifact_pipeline"],
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        },
    ],
    "real_capability_matrix_state": {
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
            "c" + "cxt": "blocked",
        },
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
        "inherited_block_o_capabilities_known_blocked": True,
        "inherited_block_p_capabilities_known_blocked": True,
        "source_inventory_capabilities_known_blocked": True,
        "matrix_capabilities_known_blocked": True,
        "all_real_capabilities_blocked_at_18_2": True,
    },
    "fail_closed_matrix_decision": {
        "block_p_source_inventory_in_18_1": "preserved",
        "block_p_inventory_matrix_in_18_2": "complete",
        "block_p_packaging_contract_in_18_3": "allowed",
        "only_source_only_18_3_handoff_allowed": True,
        "inventory_observations_modified_by_18_2": False,
        "desktop_entrypoint_selected_by_18_2": False,
        "desktop_entrypoint_approved_by_18_2": False,
        "qml_bundle_validated_by_18_2": False,
        "qt_plugins_discovered_by_18_2": False,
        "dependency_resolution_performed_by_18_2": False,
        "packaging_metadata_modified_by_18_2": False,
        "packaging_profile_selected_by_18_2": False,
        "build_tool_selected_by_18_2": False,
        "packaging_contract_approved_by_18_2": False,
        "build_ready_by_18_2": False,
        "packaging_authorized_by_18_2": False,
        "build_authorized_by_18_2": False,
        "build_executed_by_18_2": False,
        "artifact_created_by_18_2": False,
        "release_authorized_by_18_2": False,
        "runtime_enabled_by_18_2": False,
        "orders_enabled_by_18_2": False,
    },
    "non_execution_matrix_evidence": {
        "source_builder_called": True,
        "source_builder_call_count": 1,
        "source_accepted": True,
        "identity_valid": True,
        "reference_valid": True,
        "summary_valid": True,
        "entrypoint_rows_valid": True,
        "qml_inventory_valid": True,
        "config_reference_rows_valid": True,
        "python_dependency_inventory_valid": True,
        "packaging_metadata_valid": True,
        "preview_packaging_valid": True,
        "artifact_exclusion_policy_valid": True,
        "inventory_findings_valid": True,
        "real_capability_valid": True,
        "fail_closed_valid": True,
        "evidence_valid": True,
        "inventory_boundaries_valid": True,
        "source_boundaries_valid": True,
        "future_steps_valid": True,
        "exact_source_counts_preserved": True,
        "desktop_entrypoint_matrix_row_count": 4,
        "qml_bundle_matrix_row_count": 5,
        "python_dependency_matrix_row_count": 4,
        "packaging_metadata_matrix_row_count": 4,
        "existing_preview_packaging_matrix_row_count": 4,
        "artifact_exclusion_policy_matrix_row_count": 1,
        "inventory_finding_matrix_row_count": 11,
        "packaging_scope_matrix_row_count": 3,
        "packaging_requirement_matrix_row_count": 8,
        "unresolved_contract_blocker_count": 12,
        "matrix_artifact_complete": True,
        "inventory_observations_modified_by_18_2": False,
        "desktop_entrypoint_selected_by_18_2": False,
        "desktop_entrypoint_approved_by_18_2": False,
        "qml_bundle_validated_by_18_2": False,
        "qt_plugins_discovered_by_18_2": False,
        "dependency_resolution_performed_by_18_2": False,
        "packaging_metadata_modified_by_18_2": False,
        "packaging_profile_selected_by_18_2": False,
        "build_tool_selected_by_18_2": False,
        "packaging_contract_approved_by_18_2": False,
        "build_ready_by_18_2": False,
        "packaging_authorized_by_18_2": False,
        "build_authorized_by_18_2": False,
        "build_executed_by_18_2": False,
        "artifact_created_by_18_2": False,
        "release_authorized_by_18_2": False,
        "runtime_enabled_by_18_2": False,
        "orders_enabled_by_18_2": False,
    },
    "matrix_boundaries": {
        "reads_18_1_only": True,
        "source_only": True,
        "plain_data": True,
        "static": True,
        "repo_rescan": False,
        "filesystem_inventory": False,
        "environment_scan": False,
        "secret_file_read": False,
        "dependency_import": False,
        "dependency_resolution": False,
        "pyside_import": False,
        "qml_load": False,
        "qt_plugin_discovery": False,
        "packaging_metadata_mutation": False,
        "packaging_profile_validation": False,
        "build_tool_selection": False,
        "build_tool_execution": False,
        "spec_file_creation": False,
        "build_command_creation": False,
        "build_command_execution": False,
        "packaging_performed": False,
        "artifact_created": False,
        "artifact_scanned": False,
        "artifact_signed": False,
        "installer_created": False,
        "release_performed": False,
        "runtime_started": False,
        "orders_enabled": False,
        "network_opened": False,
        "credentials_read": False,
        "qml_bridge_gateway_controller_changed": False,
        "can_feed_only_18_3_packaging_contract": True,
    },
    "source_boundaries": {
        "source_block_p_desktop_exe_packaging_source_inventory": "FUNCTIONAL-PREVIEW-18.1",
        "source_inventory_preserved": True,
        "can_build_desktop_exe_packaging_inventory_matrix": True,
        "inventory_matrix_artifact_complete": True,
        "can_build_desktop_exe_packaging_contract": True,
        "can_feed_18_3": True,
    },
    "future_steps": [
        {
            "step": "18.3",
            "title": "BLOCK P DESKTOP EXE PACKAGING CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
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
    "status": "ready_for_functional_preview_18_3_block_p_desktop_exe_packaging_contract",
}
SOURCE_IDENTITY_EXPECTED: Final[dict[str, Any]] = {
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
}

DENIED_ARTIFACT_PATTERNS_18_3: Final[list[str]] = [
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
]
BLOCKER_TO_EVIDENCE: Final[dict[str, str]] = {
    "final_desktop_entrypoint_not_selected": "evidence_final_desktop_entrypoint_selection",
    "desktop_entrypoint_validation_not_performed": "evidence_desktop_entrypoint_validation",
    "qml_bundle_validation_not_performed": "evidence_qml_bundle_validation",
    "windows_shared_qml_import_path_unresolved": "evidence_windows_shared_qml_import_path",
    "qt_plugin_inventory_missing": "evidence_qt_plugin_inventory",
    "ui_package_discovery_missing": "evidence_ui_package_discovery",
    "qml_package_data_missing": "evidence_qml_package_data",
    "final_desktop_packaging_profile_not_aligned": "evidence_final_windows_profile_alignment",
    "dependency_resolution_not_performed": "evidence_windows_dependency_resolution",
    "secret_and_local_data_exclusion_policy_not_validated": "evidence_artifact_exclusion_validation",
    "windows_toolchain_not_confirmed": "evidence_windows_toolchain_confirmation",
    "future_explicit_build_execution_gate_missing": "evidence_future_explicit_build_execution_gate",
}
RESOLUTION_CRITERIA: Final[dict[str, str]] = {
    "final_desktop_entrypoint_not_selected": "exactly_one_candidate_selected_and_recorded",
    "desktop_entrypoint_validation_not_performed": "selected_entrypoint_passes_source_validation_and_future_windows_launch_smoke",
    "qml_bundle_validation_not_performed": "all_inventoried_qml_inputs_and_required_runtime_assets_pass_static_and_future_load_validation",
    "windows_shared_qml_import_path_unresolved": "windows_bundle_defines_shared_qml_import_path_or_equivalent_and_future_smoke_passes",
    "qt_plugin_inventory_missing": "exact_required_qt_plugin_inventory_is_recorded_and_validated_for_selected_tool",
    "ui_package_discovery_missing": "packaging_metadata_contains_ui_package_discovery_pattern",
    "qml_package_data_missing": "packaging_metadata_contains_required_qml_and_support_asset_package_data",
    "final_desktop_packaging_profile_not_aligned": "final_windows_profile_targets_selected_desktop_entrypoint_and_final_runtime_name",
    "dependency_resolution_not_performed": "windows_target_dependency_resolution_and_locked_environment_complete",
    "secret_and_local_data_exclusion_policy_not_validated": "policy_applied_and_final_bundle_scan_reports_zero_denied_matches",
    "windows_toolchain_not_confirmed": "exact_windows_python_qt_and_packaging_toolchain_versions_recorded_and_validated",
    "future_explicit_build_execution_gate_missing": "separate_future_explicit_gate_approves_exact_build_command_after_readiness_contract",
}
EVIDENCE_ARTIFACTS: Final[dict[str, list[str]]] = {
    "final_desktop_entrypoint_not_selected": [
        "selected_desktop_entrypoint_record",
        "selection_cardinality_check",
    ],
    "desktop_entrypoint_validation_not_performed": [
        "selected_entrypoint_static_validation_result",
        "windows_launch_smoke_result",
    ],
    "qml_bundle_validation_not_performed": [
        "qml_inventory_validation_manifest",
        "qml_load_smoke_result",
    ],
    "windows_shared_qml_import_path_unresolved": [
        "windows_shared_qml_import_path_manifest",
        "windows_import_path_smoke_result",
    ],
    "qt_plugin_inventory_missing": [
        "qt_plugin_inventory_manifest",
        "selected_tool_qt_plugin_validation_result",
    ],
    "ui_package_discovery_missing": [
        "ui_package_discovery_metadata_record",
        "package_discovery_static_check_result",
    ],
    "qml_package_data_missing": [
        "qml_package_data_metadata_record",
        "support_asset_package_data_check_result",
    ],
    "final_desktop_packaging_profile_not_aligned": [
        "final_windows_profile_alignment_record",
        "desktop_entrypoint_profile_target_check",
    ],
    "dependency_resolution_not_performed": [
        "windows_locked_dependency_manifest",
        "dependency_resolution_result",
    ],
    "secret_and_local_data_exclusion_policy_not_validated": [
        "applied_exclusion_policy_manifest",
        "final_bundle_denied_pattern_scan_result",
    ],
    "windows_toolchain_not_confirmed": [
        "windows_toolchain_version_manifest",
        "toolchain_validation_result",
    ],
    "future_explicit_build_execution_gate_missing": [
        "exact_build_command_record",
        "explicit_build_gate_approval_record",
    ],
}
ACCEPTANCE_RULE_LINKS: Final[dict[str, dict[str, list[str]]]] = {
    "all_twelve_blockers_resolved": {
        "blockers": list(BLOCKER_TO_EVIDENCE.keys()),
        "evidence": [],
    },
    "all_required_evidence_collected_and_validated": {
        "blockers": [],
        "evidence": list(BLOCKER_TO_EVIDENCE.values()),
    },
    "exactly_one_desktop_entrypoint_selected_and_validated": {
        "blockers": [
            "final_desktop_entrypoint_not_selected",
            "desktop_entrypoint_validation_not_performed",
        ],
        "evidence": [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
        ],
    },
    "qml_qt_and_packaging_metadata_contract_satisfied": {
        "blockers": [
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "qt_plugin_inventory_missing",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
        ],
        "evidence": [
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_qt_plugin_inventory",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
        ],
    },
    "windows_dependencies_profile_toolchain_contract_satisfied": {
        "blockers": [
            "final_desktop_packaging_profile_not_aligned",
            "dependency_resolution_not_performed",
            "windows_toolchain_not_confirmed",
        ],
        "evidence": [
            "evidence_final_windows_profile_alignment",
            "evidence_windows_dependency_resolution",
            "evidence_windows_toolchain_confirmation",
        ],
    },
    "artifact_exclusion_and_explicit_build_gate_contract_satisfied": {
        "blockers": [
            "secret_and_local_data_exclusion_policy_not_validated",
            "future_explicit_build_execution_gate_missing",
        ],
        "evidence": [
            "evidence_artifact_exclusion_validation",
            "evidence_future_explicit_build_execution_gate",
        ],
    },
}
CONTRACT_CAPABILITIES: Final[list[str]] = [
    "packaging_contract_operational_approval",
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
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.discard(id(item))
            continue
        if depth > max_depth:
            return False
        if type(item) in (str, int, bool) or item is None:
            continue
        if type(item) is dict:
            oid = id(item)
            if oid in active:
                return False
            active.add(oid)
            stack.append((item, depth, True))
            for k, v in item.items():
                if type(k) is not str:
                    return False
                stack.append((v, depth + 1, False))
            continue
        if type(item) is list:
            oid = id(item)
            if oid in active:
                return False
            active.add(oid)
            stack.append((item, depth, True))
            for v in item:
                stack.append((v, depth + 1, False))
            continue
        return False
    return True


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {k: _copy_plain(v) for k, v in value.items()}
    if type(value) is list:
        return [_copy_plain(v) for v in value]
    return value


def _exact_plain_matches(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if list(actual.keys()) != list(expected.keys()):
            return False
        for key, value in expected.items():
            if not _exact_plain_matches(actual[key], value):
                return False
        return True
    if type(expected) is list:
        if len(actual) != len(expected):
            return False
        return all(_exact_plain_matches(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _plain_dict_section(source: Any, field: str) -> dict[str, Any]:
    if (
        type(source) is dict
        and type(source.get(field)) is dict
        and _all_plain_json(source[field], MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    ):
        return source[field]
    return {}


def _plain_list_section(source: Any, field: str) -> list[Any]:
    if (
        type(source) is dict
        and type(source.get(field)) is list
        and _all_plain_json(source[field], MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    ):
        return source[field]
    return []


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
    for raw_key, raw_value in section.items():
        if type(raw_key) is str and raw_key == key:
            return raw_value
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
        _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "inventory_matrix_summary"),
            EXPECTED_SOURCE["inventory_matrix_summary"],
            SUMMARY_OWNED_FIELDS_18_3,
        )
        and _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "fail_closed_matrix_decision"),
            EXPECTED_SOURCE["fail_closed_matrix_decision"],
            FAIL_CLOSED_OWNED_FIELDS_18_3,
        )
        and _owned_fields_are_unshadowed(
            _get_exact_string_key(source, "source_boundaries"),
            EXPECTED_SOURCE["source_boundaries"],
            SOURCE_BOUNDARY_OWNED_FIELDS_18_3,
        )
    )


def _source_identity_valid(source: Any) -> bool:
    if type(source) is not dict:
        return False
    for key, value in SOURCE_IDENTITY_EXPECTED.items():
        actual = _get_exact_string_key(source, key)
        if not _all_plain_json(actual, MAX_DIAGNOSTIC_CONTAINER_DEPTH):
            return False
        if not _exact_plain_matches(actual, value):
            return False
    return True


def _scalar_reference(source: dict[str, Any], key: str, identity_valid: bool) -> Any:
    expected = SOURCE_IDENTITY_EXPECTED[key]
    if not identity_valid:
        if type(expected) is bool:
            return False
        return ""
    value = source.get(key)
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(expected) is bool:
        return False
    return ""


def _capability_map_known_blocked(capability_map: dict[str, Any]) -> bool:
    return bool(capability_map) and all(
        type(value) is str and value == "blocked" for value in capability_map.values()
    )


def _nonempty_unique(values: list[str]) -> bool:
    return all(type(value) is str and bool(value) for value in values) and len(values) == len(
        set(values)
    )


def _source_referential_integrity_valid(
    source: dict[str, Any],
    available_source_matrix_row_ids: set[str],
    finding_rows_valid: bool,
    scope_rows_valid: bool,
    requirement_rows_valid: bool,
    blocker_rows_valid: bool,
) -> bool:
    if not (
        finding_rows_valid and scope_rows_valid and requirement_rows_valid and blocker_rows_valid
    ):
        return False
    finding_rows = _plain_list_section(source, "inventory_finding_matrix_rows")
    scope_rows = _plain_list_section(source, "packaging_scope_matrix_rows")
    requirement_rows = _plain_list_section(source, "packaging_requirement_matrix_rows")
    blocker_rows = _plain_list_section(source, "unresolved_contract_blocker_rows")
    finding_ids = [row["finding_id"] for row in finding_rows]
    scope_ids = [row["scope_id"] for row in scope_rows]
    blocker_ids = [row["blocker_id"] for row in blocker_rows]
    requirement_ids = [row["requirement_id"] for row in requirement_rows]
    if not (
        _nonempty_unique(finding_ids)
        and _nonempty_unique(scope_ids)
        and _nonempty_unique(blocker_ids)
        and _nonempty_unique(requirement_ids)
    ):
        return False
    finding_id_set = set(finding_ids)
    scope_id_set = set(scope_ids)
    blocker_id_set = set(blocker_ids)
    for row in scope_rows:
        supporting_ids = row["supporting_matrix_row_ids"]
        unresolved_ids = row["unresolved_condition_ids"]
        if not _nonempty_unique(supporting_ids):
            return False
        if not set(supporting_ids).issubset(available_source_matrix_row_ids):
            return False
        if not _nonempty_unique(unresolved_ids):
            return False
        if not set(unresolved_ids).issubset(blocker_id_set):
            return False
    for row in requirement_rows:
        unresolved_ids = row["unresolved_condition_ids"]
        if not _nonempty_unique(unresolved_ids):
            return False
        if not set(unresolved_ids).issubset(blocker_id_set):
            return False
    for row in blocker_rows:
        affected_scope_ids = row["affected_scope_ids"]
        source_finding_ids = row["source_finding_ids"]
        if not _nonempty_unique(affected_scope_ids):
            return False
        if not set(affected_scope_ids).issubset(scope_id_set):
            return False
        if not set(source_finding_ids).issubset(finding_id_set):
            return False
        if len(source_finding_ids) != len(set(source_finding_ids)):
            return False
    return True


def _contract_principles(ok: bool) -> list[dict[str, Any]]:
    ids = [
        "final_product_is_windows_desktop_exe",
        "contract_is_source_only",
        "exactly_one_desktop_entrypoint_must_be_selected_later",
        "cli_preview_scope_is_not_final_desktop_packaging",
        "all_qml_and_qt_runtime_inputs_require_explicit_contract_coverage",
        "secrets_and_local_user_data_are_excluded_fail_closed",
        "build_execution_requires_future_explicit_gate",
        "contract_completion_does_not_grant_readiness_or_authorization",
    ]
    return [
        {
            "principle_id": principle_id,
            "required": True,
            "contract_defined": ok,
            "satisfied_by_18_3": False,
            "operational_effect_granted": False,
            "failure_policy": "fail_closed",
            "contract_result": "defined_not_satisfied" if ok else "blocked_source_not_accepted",
        }
        for principle_id in ids
    ]


def _blocker_contract_rows(
    rows: list[Any],
    requirement_rows: list[Any],
    requirement_contract_id_by_source_id: dict[str, str],
    ok: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not ok:
        return out
    for row in rows:
        blocker_id = row["blocker_id"]
        linked_source_requirement_ids = [
            req["requirement_id"]
            for req in requirement_rows
            if blocker_id in req["unresolved_condition_ids"]
        ]
        linked = [
            requirement_contract_id_by_source_id[requirement_id]
            for requirement_id in linked_source_requirement_ids
            if requirement_id in requirement_contract_id_by_source_id
        ]
        out.append(
            {
                "blocker_id": blocker_id,
                "source_blocker_preserved": True,
                "source_finding_ids": _copy_plain(row["source_finding_ids"]),
                "source_affected_scope_ids": _copy_plain(row["affected_scope_ids"]),
                "contract_clause_id": "clause_" + blocker_id,
                "contract_requirement_ids": linked,
                "required_evidence_ids": [BLOCKER_TO_EVIDENCE[blocker_id]],
                "resolution_criteria": RESOLUTION_CRITERIA[blocker_id],
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
            }
        )
    return out


def _scope_contract_rows(
    rows: list[Any],
    blockers: list[dict[str, Any]],
    ok: bool,
    available_source_matrix_row_ids: set[str],
    entrypoint_rows_valid: bool,
    qml_rows_valid: bool,
    dependency_rows_valid: bool,
    metadata_rows_valid: bool,
    preview_rows_valid: bool,
    policy_rows_valid: bool,
) -> list[dict[str, Any]]:
    if not ok:
        return []
    by_blocker = {blocker["blocker_id"]: blocker for blocker in blockers}
    out: list[dict[str, Any]] = []
    for row in rows:
        scope_id = row["scope_id"]
        if scope_id == "desktop_application_entrypoint":
            source_scope_preserved = entrypoint_rows_valid and bool(blockers)
        elif scope_id == "qt_qml_runtime_bundle":
            source_scope_preserved = qml_rows_valid and metadata_rows_valid and bool(blockers)
        else:
            source_scope_preserved = (
                dependency_rows_valid
                and preview_rows_valid
                and policy_rows_valid
                and bool(blockers)
            )
        preserved_supporting_ids = [
            row_id
            for row_id in row["supporting_matrix_row_ids"]
            if row_id in available_source_matrix_row_ids
        ]
        linked_blocker_ids = [
            blocker_id for blocker_id in row["unresolved_condition_ids"] if blocker_id in by_blocker
        ]
        out.append(
            {
                "scope_id": scope_id,
                "source_scope_preserved": source_scope_preserved,
                "source_supporting_matrix_row_ids": preserved_supporting_ids,
                "source_unresolved_blocker_ids": linked_blocker_ids,
                "contract_clause_ids": [
                    by_blocker[blocker_id]["contract_clause_id"]
                    for blocker_id in linked_blocker_ids
                ],
                "required_evidence_ids": [
                    by_blocker[blocker_id]["required_evidence_ids"][0]
                    for blocker_id in linked_blocker_ids
                ],
                "contract_definition_complete": source_scope_preserved,
                "contract_satisfied": False,
                "resolved_blocker_count": 0,
                "unresolved_blocker_count": len(linked_blocker_ids),
                "ready_for_read_model": source_scope_preserved,
                "scope_ready": False,
                "scope_authorized": False,
                "build_ready": False,
                "failure_policy": "fail_closed",
                "contract_classification": "scope_contract_defined_blocked"
                if source_scope_preserved
                else "scope_contract_source_support_not_fully_preserved",
                "contract_result": "fail_closed_unresolved"
                if source_scope_preserved
                else "scope_contract_definition_blocked_by_invalid_supporting_source",
            }
        )
    return out


def _requirement_contract_rows(
    rows: list[Any],
    blockers: list[dict[str, Any]],
    requirement_contract_id_by_source_id: dict[str, str],
    ok: bool,
) -> list[dict[str, Any]]:
    if not ok:
        return []
    by_blocker = {blocker["blocker_id"]: blocker for blocker in blockers}
    return [
        {
            "requirement_id": row["requirement_id"],
            "source_requirement_preserved": True,
            "source_inventory_observed": row["source_inventory_observed"],
            "source_inventory_requirement_satisfied": row["inventory_requirement_satisfied"],
            "source_unresolved_condition_ids": _copy_plain(row["unresolved_condition_ids"]),
            "contract_requirement_id": requirement_contract_id_by_source_id[row["requirement_id"]],
            "contract_clause_ids": [
                by_blocker[blocker_id]["contract_clause_id"]
                for blocker_id in row["unresolved_condition_ids"]
            ],
            "required_evidence_ids": [
                by_blocker[blocker_id]["required_evidence_ids"][0]
                for blocker_id in row["unresolved_condition_ids"]
            ],
            "contract_defined": True,
            "contract_satisfied": False,
            "build_requirement_satisfied": False,
            "requires_future_explicit_step": row["requires_future_explicit_step"],
            "ready_for_read_model": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "contract_classification": "requirement_contract_defined_blocked",
            "contract_result": "fail_closed_unresolved",
        }
        for row in rows
    ]


def _evidence_rows(blockers: list[dict[str, Any]], ok: bool) -> list[dict[str, Any]]:
    if not ok:
        return []
    out: list[dict[str, Any]] = []
    for blocker in blockers:
        blocker_id = blocker["blocker_id"]
        evidence_id = blocker["required_evidence_ids"][0]
        stage = (
            "future_explicit_build_gate"
            if blocker_id == "future_explicit_build_execution_gate_missing"
            else (
                "future_explicit_windows_validation"
                if blocker_id
                in {
                    "desktop_entrypoint_validation_not_performed",
                    "qml_bundle_validation_not_performed",
                    "windows_shared_qml_import_path_unresolved",
                    "qt_plugin_inventory_missing",
                    "dependency_resolution_not_performed",
                    "windows_toolchain_not_confirmed",
                }
                else "future_explicit_packaging_preparation"
            )
        )
        out.append(
            {
                "evidence_id": evidence_id,
                "blocker_id": blocker_id,
                "evidence_type": "future_contract_evidence",
                "required_artifacts": _copy_plain(EVIDENCE_ARTIFACTS[blocker_id]),
                "collection_stage": stage,
                "source_only_definition": True,
                "collected_by_18_3": False,
                "validated_by_18_3": False,
                "required_for_build_readiness": True,
                "required_for_packaging_authorization": True,
                "required_for_build_authorization": True,
                "failure_policy": "fail_closed",
                "evidence_classification": "required_future_evidence",
                "evidence_result": "not_collected_by_18_3",
            }
        )
    return out


def _acceptance_rows(ok: bool) -> list[dict[str, Any]]:
    return [
        {
            "acceptance_rule_id": rule_id,
            "required_blocker_ids": _copy_plain(links["blockers"]) if ok else [],
            "required_evidence_ids": _copy_plain(links["evidence"]) if ok else [],
            "all_inputs_required": True,
            "rule_defined": ok,
            "rule_satisfied_by_18_3": False,
            "grants_build_readiness": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "failure_policy": "fail_closed",
            "acceptance_classification": "fail_closed_acceptance_rule",
            "acceptance_result": "not_satisfied_by_18_3" if ok else "blocked_source_not_accepted",
        }
        for rule_id, links in ACCEPTANCE_RULE_LINKS.items()
    ]


def build_preview_block_p_desktop_exe_packaging_contract() -> dict[str, Any]:
    source_raw = build_preview_block_p_desktop_exe_packaging_inventory_matrix()
    source_plain_bounded = _all_plain_json(source_raw, MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    safe_source, all_top_level_keys_exact_str = _safe_top_level_source(source_raw)
    identity_valid = _source_identity_valid(safe_source)
    source_reference_valid = _section_valid(
        safe_source.get("block_p_desktop_exe_packaging_source_inventory_reference"),
        EXPECTED_SOURCE["block_p_desktop_exe_packaging_source_inventory_reference"],
    )
    matrix_summary_valid = _section_valid(
        _get_exact_string_key(safe_source, "inventory_matrix_summary"),
        EXPECTED_SOURCE["inventory_matrix_summary"],
    )
    source_preservation_valid = _section_valid(
        safe_source.get("source_inventory_preservation"),
        EXPECTED_SOURCE["source_inventory_preservation"],
    )
    entrypoint_rows_valid = _section_valid(
        safe_source.get("desktop_entrypoint_matrix_rows"),
        EXPECTED_SOURCE["desktop_entrypoint_matrix_rows"],
    )
    qml_rows_valid = _section_valid(
        safe_source.get("qml_bundle_matrix_rows"), EXPECTED_SOURCE["qml_bundle_matrix_rows"]
    )
    dependency_rows_valid = _section_valid(
        safe_source.get("python_dependency_matrix_rows"),
        EXPECTED_SOURCE["python_dependency_matrix_rows"],
    )
    metadata_rows_valid = _section_valid(
        safe_source.get("packaging_metadata_matrix_rows"),
        EXPECTED_SOURCE["packaging_metadata_matrix_rows"],
    )
    preview_rows_valid = _section_valid(
        safe_source.get("existing_preview_packaging_matrix_rows"),
        EXPECTED_SOURCE["existing_preview_packaging_matrix_rows"],
    )
    policy_rows_valid = _section_valid(
        safe_source.get("artifact_exclusion_policy_matrix_rows"),
        EXPECTED_SOURCE["artifact_exclusion_policy_matrix_rows"],
    )
    finding_rows_valid = _section_valid(
        safe_source.get("inventory_finding_matrix_rows"),
        EXPECTED_SOURCE["inventory_finding_matrix_rows"],
    )
    scope_rows_valid = _section_valid(
        safe_source.get("packaging_scope_matrix_rows"),
        EXPECTED_SOURCE["packaging_scope_matrix_rows"],
    )
    requirement_rows_valid = _section_valid(
        safe_source.get("packaging_requirement_matrix_rows"),
        EXPECTED_SOURCE["packaging_requirement_matrix_rows"],
    )
    blocker_rows_valid = _section_valid(
        safe_source.get("unresolved_contract_blocker_rows"),
        EXPECTED_SOURCE["unresolved_contract_blocker_rows"],
    )
    real_capability_valid = _section_valid(
        safe_source.get("real_capability_matrix_state"),
        EXPECTED_SOURCE["real_capability_matrix_state"],
    )
    fail_closed_valid = _section_valid(
        _get_exact_string_key(safe_source, "fail_closed_matrix_decision"),
        EXPECTED_SOURCE["fail_closed_matrix_decision"],
    )
    evidence_valid = _section_valid(
        safe_source.get("non_execution_matrix_evidence"),
        EXPECTED_SOURCE["non_execution_matrix_evidence"],
    )
    matrix_boundaries_valid = _section_valid(
        safe_source.get("matrix_boundaries"), EXPECTED_SOURCE["matrix_boundaries"]
    )
    source_boundaries_valid = _section_valid(
        _get_exact_string_key(safe_source, "source_boundaries"),
        EXPECTED_SOURCE["source_boundaries"],
    )
    future_steps_valid = _section_valid(
        safe_source.get("future_steps"), EXPECTED_SOURCE["future_steps"]
    )
    local_validity = {
        "identity_valid": identity_valid,
        "source_reference_valid": source_reference_valid,
        "matrix_summary_valid": matrix_summary_valid,
        "source_preservation_valid": source_preservation_valid,
        "entrypoint_rows_valid": entrypoint_rows_valid,
        "qml_rows_valid": qml_rows_valid,
        "dependency_rows_valid": dependency_rows_valid,
        "metadata_rows_valid": metadata_rows_valid,
        "preview_rows_valid": preview_rows_valid,
        "policy_rows_valid": policy_rows_valid,
        "finding_rows_valid": finding_rows_valid,
        "scope_rows_valid": scope_rows_valid,
        "requirement_rows_valid": requirement_rows_valid,
        "blocker_rows_valid": blocker_rows_valid,
        "real_capability_valid": real_capability_valid,
        "fail_closed_valid": fail_closed_valid,
        "evidence_valid": evidence_valid,
        "matrix_boundaries_valid": matrix_boundaries_valid,
        "source_boundaries_valid": source_boundaries_valid,
        "future_steps_valid": future_steps_valid,
    }
    entrypoint_source_row_ids = (
        [
            row["matrix_row_id"]
            for row in _plain_list_section(safe_source, "desktop_entrypoint_matrix_rows")
        ]
        if entrypoint_rows_valid
        else []
    )
    qml_source_row_ids = (
        [row["matrix_row_id"] for row in _plain_list_section(safe_source, "qml_bundle_matrix_rows")]
        if qml_rows_valid
        else []
    )
    dependency_source_row_ids = (
        [
            row["matrix_row_id"]
            for row in _plain_list_section(safe_source, "python_dependency_matrix_rows")
        ]
        if dependency_rows_valid
        else []
    )
    metadata_source_row_ids = (
        [
            row["matrix_row_id"]
            for row in _plain_list_section(safe_source, "packaging_metadata_matrix_rows")
        ]
        if metadata_rows_valid
        else []
    )
    preview_source_row_ids = (
        [
            row["matrix_row_id"]
            for row in _plain_list_section(safe_source, "existing_preview_packaging_matrix_rows")
        ]
        if preview_rows_valid
        else []
    )
    policy_source_row_ids = (
        [
            row["matrix_row_id"]
            for row in _plain_list_section(safe_source, "artifact_exclusion_policy_matrix_rows")
        ]
        if policy_rows_valid
        else []
    )
    available_source_matrix_row_ids = set(
        entrypoint_source_row_ids
        + qml_source_row_ids
        + dependency_source_row_ids
        + metadata_source_row_ids
        + preview_source_row_ids
        + policy_source_row_ids
    )
    source_requirement_rows = (
        _plain_list_section(safe_source, "packaging_requirement_matrix_rows")
        if requirement_rows_valid
        else []
    )
    requirement_contract_id_by_source_id = {
        row["requirement_id"]: "contract_" + row["requirement_id"]
        for row in source_requirement_rows
    }
    referential_integrity_valid = _source_referential_integrity_valid(
        safe_source,
        available_source_matrix_row_ids,
        finding_rows_valid,
        scope_rows_valid,
        requirement_rows_valid,
        blocker_rows_valid,
    )
    source_accepted = (
        source_plain_bounded
        and all_top_level_keys_exact_str
        and list(safe_source.keys()) == TOP_LEVEL_FIELDS_18_2
        and all(local_validity.values())
        and _no_shadowing(safe_source)
        and referential_integrity_valid
    )
    status = PACKAGING_CONTRACT_STATUS if source_accepted else BLOCKED_STATUS
    blocker_rows = _blocker_contract_rows(
        _plain_list_section(safe_source, "unresolved_contract_blocker_rows"),
        source_requirement_rows,
        requirement_contract_id_by_source_id,
        blocker_rows_valid and finding_rows_valid and scope_rows_valid,
    )
    scope_rows = _scope_contract_rows(
        _plain_list_section(safe_source, "packaging_scope_matrix_rows"),
        blocker_rows,
        scope_rows_valid,
        available_source_matrix_row_ids,
        entrypoint_rows_valid,
        qml_rows_valid,
        dependency_rows_valid,
        metadata_rows_valid,
        preview_rows_valid,
        policy_rows_valid,
    )
    req_rows = _requirement_contract_rows(
        source_requirement_rows,
        blocker_rows,
        requirement_contract_id_by_source_id,
        requirement_rows_valid and bool(blocker_rows),
    )
    evidence = _evidence_rows(blocker_rows, bool(blocker_rows))
    acceptance = _acceptance_rows(bool(blocker_rows) and bool(evidence))
    caps = (
        _plain_dict_section(safe_source, "real_capability_matrix_state")
        if real_capability_valid
        else {}
    )
    inherited_block_o_capabilities = _copy_plain(caps.get("inherited_block_o_capabilities", {}))
    inherited_block_p_capabilities = _copy_plain(caps.get("inherited_block_p_capabilities", {}))
    source_inventory_capabilities = _copy_plain(caps.get("source_inventory_capabilities", {}))
    inventory_matrix_capabilities = _copy_plain(caps.get("inventory_matrix_capabilities", {}))
    packaging_contract_capabilities = {key: "blocked" for key in CONTRACT_CAPABILITIES}
    inherited_block_o_claim = _capability_map_known_blocked(inherited_block_o_capabilities)
    inherited_block_p_claim = _capability_map_known_blocked(inherited_block_p_capabilities)
    source_inventory_claim = _capability_map_known_blocked(source_inventory_capabilities)
    inventory_matrix_claim = _capability_map_known_blocked(inventory_matrix_capabilities)
    packaging_contract_claim = _capability_map_known_blocked(packaging_contract_capabilities)
    all_capabilities_claim = (
        inherited_block_o_claim
        and inherited_block_p_claim
        and source_inventory_claim
        and inventory_matrix_claim
        and packaging_contract_claim
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_packaging_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_desktop_exe_packaging_contract_status": status,
        "block_p_desktop_exe_packaging_contract_decision": status.upper(),
        "packaging_contract_artifact_complete": source_accepted,
        "ready_for_block_p_4": source_accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_desktop_exe_packaging_inventory_matrix_reference": {
            "schema_version": _scalar_reference(safe_source, "schema_version", identity_valid),
            "block_p_desktop_exe_packaging_inventory_matrix_kind": _scalar_reference(
                safe_source,
                "block_p_desktop_exe_packaging_inventory_matrix_kind",
                identity_valid,
            ),
            "block": _scalar_reference(safe_source, "block", identity_valid),
            "step": _scalar_reference(safe_source, "step", identity_valid),
            "block_p_desktop_exe_packaging_inventory_matrix_status": _scalar_reference(
                safe_source,
                "block_p_desktop_exe_packaging_inventory_matrix_status",
                identity_valid,
            ),
            "block_p_desktop_exe_packaging_inventory_matrix_decision": _scalar_reference(
                safe_source,
                "block_p_desktop_exe_packaging_inventory_matrix_decision",
                identity_valid,
            ),
            "inventory_matrix_artifact_complete": _scalar_reference(
                safe_source, "inventory_matrix_artifact_complete", identity_valid
            ),
            "ready_for_block_p_3": _scalar_reference(
                safe_source, "ready_for_block_p_3", identity_valid
            ),
            "next_step": _scalar_reference(safe_source, "next_step", identity_valid),
            "next_step_title": _scalar_reference(safe_source, "next_step_title", identity_valid),
            "status": _scalar_reference(safe_source, "status", identity_valid),
            "source_identity_valid": identity_valid,
            "source_block_p_desktop_exe_packaging_inventory_matrix_step": "FUNCTIONAL-PREVIEW-18.2",
            "source_inventory_matrix_read_by_18_3": True,
            "inventory_matrix_available_before_contract": True,
            "static_packaging_contract_only": True,
            "packaging_contract_built_by_18_3": True,
            "packaging_contract_artifact_complete_by_18_3": source_accepted,
            "ready_for_functional_preview_18_4": source_accepted,
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
            "source_18_2_accepted": source_accepted,
            "scope_contract_count": len(scope_rows),
            "requirement_contract_count": len(req_rows),
            "blocker_contract_count": len(blocker_rows),
            "evidence_requirement_count": len(evidence),
            "acceptance_rule_count": len(acceptance),
            "all_contract_definitions_complete": source_accepted,
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
            "ready_for_block_p_4": source_accepted,
        },
        "source_matrix_preservation": {
            "source_identity_preserved": identity_valid,
            "desktop_entrypoint_row_count": 4 if entrypoint_rows_valid else 0,
            "qml_bundle_row_count": 5 if qml_rows_valid else 0,
            "python_dependency_row_count": 4 if dependency_rows_valid else 0,
            "packaging_metadata_row_count": 4 if metadata_rows_valid else 0,
            "existing_preview_packaging_row_count": 4 if preview_rows_valid else 0,
            "artifact_exclusion_policy_row_count": 1 if policy_rows_valid else 0,
            "finding_row_count": 11 if finding_rows_valid else 0,
            "scope_row_count": 3 if scope_rows_valid else 0,
            "requirement_row_count": 8 if requirement_rows_valid else 0,
            "unresolved_blocker_row_count": 12 if blocker_rows_valid else 0,
            "referential_integrity_preserved": referential_integrity_valid,
            "source_readiness_granted": False,
            "source_approval_granted": False,
            "source_authorization_granted": False,
            "source_matrix_recalculated": False,
            "repo_rescanned": False,
            "source_rows_modified": False,
            "source_relations_modified": False,
        },
        "contract_principles": _contract_principles(source_accepted),
        "desktop_entrypoint_contract": {
            "candidate_paths": ["ui/pyside_app/__main__.py", "ui/pyside_app/app.py"]
            if entrypoint_rows_valid
            else [],
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
            "contract_defined": entrypoint_rows_valid,
            "contract_satisfied": False,
        },
        "qml_bundle_contract": {
            "default_entrypoint": "MainWindow.qml" if qml_rows_valid else "",
            "qml_roots": ["ui/pyside_app/qml", "ui/qml"] if qml_rows_valid else [],
            "pyside_qml_file_count": 24 if qml_rows_valid else 0,
            "shared_qml_file_count": 107 if qml_rows_valid else 0,
            "additional_qml_support_asset_count": 0,
            "styles_qmldir_observed": qml_rows_valid,
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
            "contract_defined": qml_rows_valid,
            "contract_satisfied": False,
        },
        "python_dependency_contract": {
            "declared_dependency_count": 25 if dependency_rows_valid else 0,
            "optional_desktop_dependency_count": 3 if dependency_rows_valid else 0,
            "target_platform": "windows",
            "requires_pyside6": True,
            "build_tool_candidates": ["pyinstaller", "briefcase"] if dependency_rows_valid else [],
            "resolution_performed": False,
            "locked_environment_recorded": False,
            "tool_selection_performed": False,
            "qt_validation_performed": False,
            "approval_granted": False,
            "readiness_granted": False,
            "selected_tool": "",
            "selected_tool_version": "",
            "contract_defined": dependency_rows_valid,
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
            "deploy_packaging_observations_preserved": metadata_rows_valid,
            "documentation_observations_preserved": metadata_rows_valid,
            "contract_defined": metadata_rows_valid,
            "contract_satisfied": False,
        },
        "preview_packaging_separation_contract": {
            "cli_preview_entrypoint": "scripts/run_local_bot.py" if preview_rows_valid else "",
            "windows_preview_targets_cli": preview_rows_valid,
            "source_scope": "cli_preview" if preview_rows_valid else "",
            "not_final_desktop_contract": True,
            "final_profile_path": "",
            "final_entrypoint": "",
            "selection_performed": False,
            "validation_performed": False,
            "approval_granted": False,
            "readiness_granted": False,
            "contract_defined": preview_rows_valid,
            "contract_satisfied": False,
        },
        "artifact_exclusion_contract": {
            "policy_source": "scripts/safe_exe_preview_build_plan.py" if policy_rows_valid else "",
            "policy_version": "security_packaging_artifact_policy.v1" if policy_rows_valid else "",
            "denied_patterns": _copy_plain(DENIED_ARTIFACT_PATTERNS_18_3)
            if policy_rows_valid
            else [],
            "requires_policy_application": True,
            "requires_config_secret_review": True,
            "requires_final_bundle_scan_zero_denied_matches": True,
            "policy_application_performed": False,
            "config_secret_review_performed": False,
            "final_bundle_scan_performed": False,
            "local_data_excluded": False,
            "approval_granted": False,
            "readiness_granted": False,
            "contract_defined": policy_rows_valid,
            "contract_satisfied": False,
        },
        "packaging_scope_contract_rows": scope_rows,
        "packaging_requirement_contract_rows": req_rows,
        "unresolved_blocker_contract_rows": blocker_rows,
        "contract_evidence_requirement_rows": evidence,
        "contract_acceptance_rule_rows": acceptance,
        "real_capability_contract_state": {
            "inherited_block_o_capabilities": inherited_block_o_capabilities,
            "inherited_block_o_capabilities_known_blocked": inherited_block_o_claim,
            "inherited_block_p_capabilities": inherited_block_p_capabilities,
            "inherited_block_p_capabilities_known_blocked": inherited_block_p_claim,
            "source_inventory_capabilities": source_inventory_capabilities,
            "source_inventory_capabilities_known_blocked": source_inventory_claim,
            "inventory_matrix_capabilities": inventory_matrix_capabilities,
            "inventory_matrix_capabilities_known_blocked": inventory_matrix_claim,
            "packaging_contract_capabilities": packaging_contract_capabilities,
            "packaging_contract_capabilities_known_blocked": packaging_contract_claim,
            "all_real_capabilities_blocked_at_18_3": all_capabilities_claim,
        },
        "fail_closed_contract_decision": {
            "block_p_inventory_matrix_preserved_in_18_3": source_accepted,
            "block_p_packaging_contract_complete_in_18_3": source_accepted,
            "only_source_only_18_4_handoff_allowed": source_accepted,
            "definitions_complete": source_accepted,
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
            "source_accepted": source_accepted,
            "source_plain_bounded": source_plain_bounded,
            "all_top_level_keys_exact_str": all_top_level_keys_exact_str,
            **local_validity,
            "scope_contract_count": len(scope_rows),
            "requirement_contract_count": len(req_rows),
            "blocker_contract_count": len(blocker_rows),
            "evidence_requirement_count": len(evidence),
            "acceptance_rule_count": len(acceptance),
            "referential_integrity_valid": referential_integrity_valid,
            "contract_artifact_complete": source_accepted,
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
            "only_18_4_handoff_depends_on_source_acceptance": source_accepted,
        },
        "source_boundaries": {
            "source_block_p_desktop_exe_packaging_inventory_matrix": "FUNCTIONAL-PREVIEW-18.2",
            "inventory_matrix_preserved": source_accepted,
            "can_build_desktop_exe_packaging_contract": source_accepted,
            "packaging_contract_artifact_complete": source_accepted,
            "can_build_desktop_exe_packaging_read_model": source_accepted,
            "can_feed_18_4": source_accepted,
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
        "status": STATUS if source_accepted else BLOCKED_STATUS,
    }
    return _copy_plain(payload)
