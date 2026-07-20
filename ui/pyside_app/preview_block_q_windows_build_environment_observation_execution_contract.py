"""FUNCTIONAL-PREVIEW-19.6 Block Q Windows build observation execution contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_q_windows_build_environment_observation_plan import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_5,
    build_preview_block_q_windows_build_environment_observation_plan,
)

SCHEMA_VERSION: Final[str] = (
    "preview_block_q_windows_build_environment_observation_execution_contract.v1"
)
KIND: Final[str] = (
    "functional_preview_block_q_windows_build_environment_observation_execution_contract"
)
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.6"
TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT OBSERVATION EXECUTION CONTRACT"
NEXT_STEP: Final[str] = ""
NEXT_STEP_TITLE: Final[str] = ""
STATUS: Final[str] = (
    "source_19_5_accepted_11_row_execution_contract_defined_11_future_read_only_collectors_authorized_"
    "current_stage_execution_not_authorized_0_observations_performed_0_evidence_collected_source_only_"
    "environment_build_not_ready_handoff_blocked_next_step_contract_missing_no_subprocess_no_pyside_import_"
    "no_qml_load_no_build_no_packaging_no_artifact_no_runtime_no_network_no_credentials_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_6_source_19_5_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_observation_execution_contract_kind",
    "block",
    "step",
    "title",
    "block_q_windows_build_environment_observation_execution_contract_status",
    "source_19_5_accepted",
    "block_q_windows_build_environment_observation_execution_contract_decision",
    "observation_execution_contract_artifact_complete",
    "actual_environment_observation_performed",
    "current_stage_execution_authorized",
    "future_read_only_observation_authorized",
    "environment_build_ready",
    "next_step",
    "next_step_title",
    "next_step_contract_missing",
    "block_q_19_5_observation_plan_reference",
    "source_observation_plan_preservation",
    "observation_execution_contract_scope",
    "observation_execution_contract_rows",
    "observation_execution_contract_summary",
    "global_authorization",
    "non_execution_observation_execution_contract_evidence",
    "observation_execution_contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
ROW_FIELDS: Final[tuple[str, ...]] = (
    "execution_contract_id",
    "observation_plan_id",
    "read_model_id",
    "observation_target",
    "collector_kind",
    "collector_source",
    "result_type",
    "evidence_type",
    "future_read_authorized",
    "current_stage_execution_authorized",
    "mutation_authorized",
    "subprocess_authorized",
    "network_authorized",
    "credentials_authorized",
    "observation_performed",
    "evidence_collected",
    "contract_state",
    "source_only_definition",
)
SOURCE_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_observation_plan_kind",
    "block",
    "step",
    "title",
    "block_q_windows_build_environment_observation_plan_status",
    "source_19_4_accepted",
    "block_q_windows_build_environment_observation_plan_decision",
    "observation_plan_artifact_complete",
    "actual_environment_observation_performed",
    "environment_build_ready",
    "ready_for_block_q_6",
    "next_step",
    "next_step_title",
    "next_step_contract_missing",
    "block_q_19_4_read_model_reference",
    "source_read_model_preservation",
    "observation_plan_scope",
    "observation_plan_rows",
    "observation_plan_summary",
    "observation_authorization_state",
    "non_execution_observation_plan_evidence",
    "observation_plan_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)

SOURCE_19_5_SPEC: Final[Any] = (
    "dict",
    (
        ("schema_version", "preview_block_q_windows_build_environment_observation_plan.v1"),
        (
            "block_q_windows_build_environment_observation_plan_kind",
            "functional_preview_block_q_windows_build_environment_observation_plan",
        ),
        ("block", "Q"),
        ("step", "19.5"),
        ("title", "WINDOWS BUILD ENVIRONMENT OBSERVATION PLAN"),
        (
            "block_q_windows_build_environment_observation_plan_status",
            "source_19_4_accepted_11_row_observation_plan_defined_0_observations_authorized_0_observations_performed_0_evidence_environment_build_not_ready_source_only_handoff_blocked_next_step_contract_missing_no_build_no_packaging_no_artifact_no_runtime_no_orders",
        ),
        ("source_19_4_accepted", True),
        (
            "block_q_windows_build_environment_observation_plan_decision",
            "SOURCE_19_4_ACCEPTED_11_ROW_OBSERVATION_PLAN_DEFINED_0_OBSERVATIONS_AUTHORIZED_0_OBSERVATIONS_PERFORMED_0_EVIDENCE_ENVIRONMENT_BUILD_NOT_READY_SOURCE_ONLY_HANDOFF_BLOCKED_NEXT_STEP_CONTRACT_MISSING_NO_BUILD_NO_PACKAGING_NO_ARTIFACT_NO_RUNTIME_NO_ORDERS",
        ),
        ("observation_plan_artifact_complete", True),
        ("actual_environment_observation_performed", False),
        ("environment_build_ready", False),
        ("ready_for_block_q_6", False),
        ("next_step", ""),
        ("next_step_title", ""),
        ("next_step_contract_missing", True),
        (
            "block_q_19_4_read_model_reference",
            (
                "dict",
                (
                    ("schema_version", "preview_block_q_windows_build_environment_read_model.v1"),
                    ("kind", "functional_preview_block_q_windows_build_environment_read_model"),
                    ("block", "Q"),
                    ("step", "19.4"),
                    (
                        "status",
                        "source_19_3_accepted_11_row_environment_read_model_defined_0_actual_reads_0_evidence_environment_build_not_ready_handoff_blocked_next_step_contract_missing",
                    ),
                    (
                        "source_top_level_fields",
                        (
                            "list",
                            (
                                "schema_version",
                                "block_q_windows_build_environment_read_model_kind",
                                "block",
                                "step",
                                "block_q_windows_build_environment_read_model_status",
                                "source_19_3_accepted",
                                "block_q_windows_build_environment_read_model_decision",
                                "environment_read_model_artifact_complete",
                                "environment_observation_complete",
                                "environment_build_ready",
                                "ready_for_block_q_5",
                                "next_step",
                                "next_step_title",
                                "block_q_19_3_contract_reference",
                                "source_contract_preservation",
                                "environment_read_model_scope",
                                "environment_read_model_rows",
                                "environment_read_model_summary",
                                "build_execution_authorization_state",
                                "non_execution_read_model_evidence",
                                "read_model_boundaries",
                                "source_boundaries",
                                "future_steps",
                                "status",
                                "integrity_valid",
                            ),
                        ),
                    ),
                    ("source_read_model_row_count", 11),
                    ("integrity_valid", True),
                ),
            ),
        ),
        (
            "source_read_model_preservation",
            (
                "dict",
                (
                    ("preserves_19_4_payload", False),
                    ("preserves_all_11_read_model_rows", True),
                    ("preserves_read_model_order", True),
                    ("preserves_one_to_one_mapping", True),
                    ("source_read_model_modified", False),
                ),
            ),
        ),
        (
            "observation_plan_scope",
            (
                "dict",
                (
                    ("observation_plan_defined", True),
                    ("source_only_definition", True),
                    ("actual_environment_observed", False),
                    ("observation_results_included", False),
                    ("environment_build_ready", False),
                ),
            ),
        ),
        (
            "observation_plan_rows",
            (
                "list",
                (
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_windows_host"),
                            ("read_model_id", "read_model_windows_host"),
                            ("contract_id", "contract_windows_host"),
                            ("matrix_id", "matrix_windows_host"),
                            ("inventory_id", "windows_host"),
                            ("requirement_field", "windows_host_required"),
                            ("category", "platform"),
                            ("required", True),
                            ("observation_target", "windows_host_identity"),
                            ("observation_method", "windows_platform_identity_query"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_supported_windows_version"),
                            ("read_model_id", "read_model_supported_windows_version"),
                            ("contract_id", "contract_supported_windows_version"),
                            ("matrix_id", "matrix_supported_windows_version"),
                            ("inventory_id", "supported_windows_version"),
                            ("requirement_field", "supported_windows_version_must_be_confirmed"),
                            ("category", "platform"),
                            ("required", True),
                            ("observation_target", "windows_version"),
                            ("observation_method", "windows_version_query"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_python_version"),
                            ("read_model_id", "read_model_python_version"),
                            ("contract_id", "contract_python_version"),
                            ("matrix_id", "matrix_python_version"),
                            ("inventory_id", "python_version"),
                            ("requirement_field", "exact_python_version_must_be_confirmed"),
                            ("category", "runtime"),
                            ("required", True),
                            ("observation_target", "python_interpreter_version"),
                            ("observation_method", "python_version_query"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_pyside_version"),
                            ("read_model_id", "read_model_pyside_version"),
                            ("contract_id", "contract_pyside_version"),
                            ("matrix_id", "matrix_pyside_version"),
                            ("inventory_id", "pyside_version"),
                            ("requirement_field", "exact_pyside_version_must_be_confirmed"),
                            ("category", "framework"),
                            ("required", True),
                            ("observation_target", "pyside_package_version"),
                            ("observation_method", "python_package_metadata_query"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_packaging_tool_and_version"),
                            ("read_model_id", "read_model_packaging_tool_and_version"),
                            ("contract_id", "contract_packaging_tool_and_version"),
                            ("matrix_id", "matrix_packaging_tool_and_version"),
                            ("inventory_id", "packaging_tool_and_version"),
                            (
                                "requirement_field",
                                "exact_packaging_tool_and_version_must_be_selected",
                            ),
                            ("category", "packaging"),
                            ("required", True),
                            ("observation_target", "packaging_tool_and_version"),
                            ("observation_method", "packaging_configuration_query"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_desktop_entrypoint"),
                            ("read_model_id", "read_model_desktop_entrypoint"),
                            ("contract_id", "contract_desktop_entrypoint"),
                            ("matrix_id", "matrix_desktop_entrypoint"),
                            ("inventory_id", "desktop_entrypoint"),
                            ("requirement_field", "desktop_entrypoint_must_be_confirmed"),
                            ("category", "application"),
                            ("required", True),
                            ("observation_target", "desktop_entrypoint"),
                            ("observation_method", "repository_configuration_read"),
                            ("expected_value_type", "string"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_qml_assets"),
                            ("read_model_id", "read_model_qml_assets"),
                            ("contract_id", "contract_qml_assets"),
                            ("matrix_id", "matrix_qml_assets"),
                            ("inventory_id", "qml_assets"),
                            ("requirement_field", "qml_assets_must_be_confirmed"),
                            ("category", "assets"),
                            ("required", True),
                            ("observation_target", "qml_asset_manifest"),
                            ("observation_method", "repository_manifest_read"),
                            ("expected_value_type", "list"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_qt_plugins"),
                            ("read_model_id", "read_model_qt_plugins"),
                            ("contract_id", "contract_qt_plugins"),
                            ("matrix_id", "matrix_qt_plugins"),
                            ("inventory_id", "qt_plugins"),
                            ("requirement_field", "qt_plugins_must_be_confirmed"),
                            ("category", "plugins"),
                            ("required", True),
                            ("observation_target", "qt_plugin_requirements"),
                            ("observation_method", "repository_manifest_read"),
                            ("expected_value_type", "list"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            ("observation_plan_id", "observation_plan_dependency_lock"),
                            ("read_model_id", "read_model_dependency_lock"),
                            ("contract_id", "contract_dependency_lock"),
                            ("matrix_id", "matrix_dependency_lock"),
                            ("inventory_id", "dependency_lock"),
                            ("requirement_field", "dependency_lock_must_be_resolved"),
                            ("category", "dependencies"),
                            ("required", True),
                            ("observation_target", "dependency_lock_state"),
                            ("observation_method", "dependency_lock_read"),
                            ("expected_value_type", "object"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            (
                                "observation_plan_id",
                                "observation_plan_secret_and_local_data_exclusion",
                            ),
                            ("read_model_id", "read_model_secret_and_local_data_exclusion"),
                            ("contract_id", "contract_secret_and_local_data_exclusion"),
                            ("matrix_id", "matrix_secret_and_local_data_exclusion"),
                            ("inventory_id", "secret_and_local_data_exclusion"),
                            (
                                "requirement_field",
                                "secret_and_local_data_exclusion_must_be_confirmed",
                            ),
                            ("category", "security"),
                            ("required", True),
                            ("observation_target", "secret_and_local_data_exclusion_policy"),
                            ("observation_method", "repository_policy_read"),
                            ("expected_value_type", "object"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                    (
                        "dict",
                        (
                            (
                                "observation_plan_id",
                                "observation_plan_output_name_and_version_policy",
                            ),
                            ("read_model_id", "read_model_output_name_and_version_policy"),
                            ("contract_id", "contract_output_name_and_version_policy"),
                            ("matrix_id", "matrix_output_name_and_version_policy"),
                            ("inventory_id", "output_name_and_version_policy"),
                            (
                                "requirement_field",
                                "output_name_and_version_policy_must_be_confirmed",
                            ),
                            ("category", "output"),
                            ("required", True),
                            ("observation_target", "output_name_and_version_policy"),
                            ("observation_method", "repository_policy_read"),
                            ("expected_value_type", "object"),
                            ("evidence_required", True),
                            ("observation_authorized", False),
                            ("observation_status", "not_performed"),
                            ("evidence_status", "not_collected"),
                            ("plan_state", "defined_but_not_authorized"),
                            ("source_only_definition", True),
                        ),
                    ),
                ),
            ),
        ),
        (
            "observation_plan_summary",
            (
                "dict",
                (
                    ("observation_plan_row_count", 11),
                    ("required_count", 11),
                    ("authorized_observation_count", 0),
                    ("performed_observation_count", 0),
                    ("evidence_required_count", 11),
                    ("evidence_collected_count", 0),
                    ("plan_definition_complete", True),
                    ("actual_environment_observation_complete", False),
                    ("environment_build_ready", False),
                ),
            ),
        ),
        (
            "observation_authorization_state",
            (
                "dict",
                (
                    ("repo_runtime_scan_authorized", False),
                    ("filesystem_runtime_scan_authorized", False),
                    ("environment_scan_authorized", False),
                    ("windows_host_inspection_authorized", False),
                    ("windows_version_read_authorized", False),
                    ("interpreter_observation_authorized", False),
                    ("dependency_resolution_authorized", False),
                    ("pyside_import_authorized", False),
                    ("qml_load_authorized", False),
                    ("qt_plugin_discovery_authorized", False),
                    ("build_command_creation_authorized", False),
                    ("build_command_execution_authorized", False),
                    ("packaging_authorized", False),
                    ("artifact_creation_authorized", False),
                    ("artifact_scan_authorized", False),
                    ("artifact_signing_authorized", False),
                    ("installer_creation_authorized", False),
                    ("release_authorized", False),
                    ("application_runtime_authorized", False),
                    ("network_open_authorized", False),
                    ("credentials_read_authorized", False),
                    ("orders_authorized", False),
                    ("observation_plan_definition_authorized", True),
                    ("next_step_authorized", False),
                    ("next_step_contract_missing", True),
                ),
            ),
        ),
        (
            "non_execution_observation_plan_evidence",
            (
                "dict",
                (
                    ("source_read", True),
                    ("observation_plan_definition_built", True),
                    ("environment_evidence_collected", False),
                    ("environment_evidence_validated", False),
                    ("repo_runtime_scan_performed", False),
                    ("filesystem_runtime_scan_performed", False),
                    ("environment_scan_performed", False),
                    ("windows_host_inspection_performed", False),
                    ("windows_version_read_performed", False),
                    ("interpreter_observation_performed", False),
                    ("dependency_resolution_performed", False),
                    ("pyside_import_performed", False),
                    ("qml_load_performed", False),
                    ("qt_plugin_discovery_performed", False),
                    ("build_command_creation_performed", False),
                    ("build_command_execution_performed", False),
                    ("packaging_performed", False),
                    ("artifact_creation_performed", False),
                    ("artifact_scan_performed", False),
                    ("artifact_signing_performed", False),
                    ("installer_creation_performed", False),
                    ("release_performed", False),
                    ("application_runtime_performed", False),
                    ("network_performed", False),
                    ("credentials_performed", False),
                    ("orders_performed", False),
                ),
            ),
        ),
        (
            "observation_plan_boundaries",
            (
                "dict",
                (
                    ("reads_19_4_only", True),
                    ("one_to_one_read_model_mapping_required", True),
                    ("plain_data", True),
                    ("source_only", True),
                    ("no_runtime_execution", True),
                    ("no_environment_mutation", True),
                    ("no_source_mutation", True),
                    ("no_network", True),
                    ("no_credentials", True),
                    ("no_orders", True),
                    ("repo_runtime_scan", False),
                    ("filesystem_runtime_scan", False),
                    ("environment_scan", False),
                    ("windows_host_inspection", False),
                    ("windows_version_read", False),
                    ("interpreter_observation", False),
                    ("dependency_resolution", False),
                    ("pyside_import", False),
                    ("qml_load", False),
                    ("qt_plugin_discovery", False),
                    ("build_command_creation", False),
                    ("build_command_execution", False),
                    ("packaging", False),
                    ("artifact_creation", False),
                    ("artifact_scan", False),
                    ("artifact_signing", False),
                    ("installer_creation", False),
                    ("release", False),
                    ("application_runtime", False),
                    ("network", False),
                    ("credentials", False),
                    ("orders", False),
                ),
            ),
        ),
        (
            "source_boundaries",
            (
                "dict",
                (
                    ("source_step", "19.4"),
                    ("source_block", "Q"),
                    ("source_integrity_required", True),
                    ("source_mutation_allowed", False),
                    ("only_public_19_4_builder_read", True),
                    ("block_p_builders_read", False),
                    ("earlier_block_q_builders_read", False),
                ),
            ),
        ),
        ("future_steps", ("list", ())),
        (
            "status",
            "source_19_4_accepted_11_row_observation_plan_defined_0_observations_authorized_0_observations_performed_0_evidence_environment_build_not_ready_source_only_handoff_blocked_next_step_contract_missing_no_build_no_packaging_no_artifact_no_runtime_no_orders",
        ),
        ("integrity_valid", True),
    ),
)
EXECUTION_ROW_SPECS: Final[Any] = (
    "list",
    (
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_windows_host"),
                ("observation_plan_id", "observation_plan_windows_host"),
                ("read_model_id", "read_model_windows_host"),
                ("observation_target", "windows_host_identity"),
                ("collector_kind", "python_standard_library"),
                ("collector_source", "platform.system_platform.machine_platform.release"),
                ("result_type", "object"),
                ("evidence_type", "structured_platform_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_supported_windows_version"),
                ("observation_plan_id", "observation_plan_supported_windows_version"),
                ("read_model_id", "read_model_supported_windows_version"),
                ("observation_target", "windows_version"),
                ("collector_kind", "python_standard_library"),
                ("collector_source", "platform.release_platform.version"),
                ("result_type", "object"),
                ("evidence_type", "structured_windows_version_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_python_version"),
                ("observation_plan_id", "observation_plan_python_version"),
                ("read_model_id", "read_model_python_version"),
                ("observation_target", "python_interpreter_version"),
                ("collector_kind", "python_standard_library"),
                ("collector_source", "sys.version_info"),
                ("result_type", "object"),
                ("evidence_type", "structured_python_version_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_pyside_version"),
                ("observation_plan_id", "observation_plan_pyside_version"),
                ("read_model_id", "read_model_pyside_version"),
                ("observation_target", "pyside_package_version"),
                ("collector_kind", "installed_package_metadata"),
                ("collector_source", "importlib.metadata.version_PySide6"),
                ("result_type", "string"),
                ("evidence_type", "package_version_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_packaging_tool_and_version"),
                ("observation_plan_id", "observation_plan_packaging_tool_and_version"),
                ("read_model_id", "read_model_packaging_tool_and_version"),
                ("observation_target", "packaging_tool_and_version"),
                ("collector_kind", "bounded_repository_configuration"),
                (
                    "collector_source",
                    "pyproject_packaging_configuration_and_installed_package_metadata",
                ),
                ("result_type", "object"),
                ("evidence_type", "packaging_configuration_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_desktop_entrypoint"),
                ("observation_plan_id", "observation_plan_desktop_entrypoint"),
                ("read_model_id", "read_model_desktop_entrypoint"),
                ("observation_target", "desktop_entrypoint"),
                ("collector_kind", "bounded_repository_configuration"),
                ("collector_source", "pyproject_and_declared_desktop_entrypoint_configuration"),
                ("result_type", "string"),
                ("evidence_type", "entrypoint_configuration_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_qml_assets"),
                ("observation_plan_id", "observation_plan_qml_assets"),
                ("read_model_id", "read_model_qml_assets"),
                ("observation_target", "qml_asset_manifest"),
                ("collector_kind", "bounded_repository_file_listing"),
                ("collector_source", "ui_pyside_declared_qml_paths"),
                ("result_type", "list"),
                ("evidence_type", "qml_asset_path_manifest"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_qt_plugins"),
                ("observation_plan_id", "observation_plan_qt_plugins"),
                ("read_model_id", "read_model_qt_plugins"),
                ("observation_target", "qt_plugin_requirements"),
                ("collector_kind", "bounded_repository_configuration"),
                ("collector_source", "declared_qt_plugin_requirements_only"),
                ("result_type", "list"),
                ("evidence_type", "qt_plugin_requirement_manifest"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_dependency_lock"),
                ("observation_plan_id", "observation_plan_dependency_lock"),
                ("read_model_id", "read_model_dependency_lock"),
                ("observation_target", "dependency_lock_state"),
                ("collector_kind", "bounded_repository_file_metadata"),
                ("collector_source", "declared_dependency_and_lock_files"),
                ("result_type", "object"),
                ("evidence_type", "dependency_lock_file_metadata_and_hash"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_secret_and_local_data_exclusion"),
                ("observation_plan_id", "observation_plan_secret_and_local_data_exclusion"),
                ("read_model_id", "read_model_secret_and_local_data_exclusion"),
                ("observation_target", "secret_and_local_data_exclusion_policy"),
                ("collector_kind", "bounded_repository_policy_read"),
                ("collector_source", "gitignore_and_declared_exclusion_policy_files"),
                ("result_type", "object"),
                ("evidence_type", "exclusion_policy_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
        (
            "dict",
            (
                ("execution_contract_id", "execution_contract_output_name_and_version_policy"),
                ("observation_plan_id", "observation_plan_output_name_and_version_policy"),
                ("read_model_id", "read_model_output_name_and_version_policy"),
                ("observation_target", "output_name_and_version_policy"),
                ("collector_kind", "bounded_repository_configuration"),
                ("collector_source", "pyproject_and_declared_release_configuration"),
                ("result_type", "object"),
                ("evidence_type", "output_policy_metadata"),
                ("future_read_authorized", True),
                ("current_stage_execution_authorized", False),
                ("mutation_authorized", False),
                ("subprocess_authorized", False),
                ("network_authorized", False),
                ("credentials_authorized", False),
                ("observation_performed", False),
                ("evidence_collected", False),
                ("contract_state", "authorized_for_future_read_only_observation"),
                ("source_only_definition", True),
            ),
        ),
    ),
)
AUTHORIZATION_FIELD_SPECS: Final[Any] = (
    "dict",
    (
        ("current_stage_execution_authorized", False),
        ("future_read_only_observation_authorized", True),
        ("subprocess_authorized", False),
        ("shell_authorized", False),
        ("powershell_authorized", False),
        ("dependency_resolution_authorized", False),
        ("pyside_import_authorized", False),
        ("qml_load_authorized", False),
        ("qt_plugin_runtime_discovery_authorized", False),
        ("build_command_creation_authorized", False),
        ("build_command_execution_authorized", False),
        ("packaging_authorized", False),
        ("artifact_creation_authorized", False),
        ("artifact_scan_authorized", False),
        ("artifact_signing_authorized", False),
        ("installer_creation_authorized", False),
        ("release_authorized", False),
        ("application_runtime_authorized", False),
        ("network_open_authorized", False),
        ("credentials_read_authorized", False),
        ("secret_values_read_authorized", False),
        ("source_mutation_authorized", False),
        ("environment_mutation_authorized", False),
        ("orders_authorized", False),
    ),
)
SUMMARY_FIELD_SPECS: Final[Any] = (
    "dict",
    (
        ("execution_contract_row_count", 11),
        ("future_read_authorized_count", 11),
        ("current_stage_execution_authorized_count", 0),
        ("mutation_authorized_count", 0),
        ("subprocess_authorized_count", 0),
        ("network_authorized_count", 0),
        ("credentials_authorized_count", 0),
        ("performed_observation_count", 0),
        ("evidence_collected_count", 0),
        ("contract_definition_complete", True),
        ("actual_environment_observation_complete", False),
        ("environment_build_ready", False),
    ),
)
BOUNDARY_FIELD_SPECS: Final[Any] = (
    "dict",
    (
        ("reads_19_5_only", True),
        ("one_to_one_observation_plan_mapping_required", True),
        ("plain_data", True),
        ("source_only", True),
        ("no_current_stage_execution", True),
        ("future_execution_requires_separate_contract", True),
        ("bounded_read_only_collectors_only", True),
        ("no_subprocess", True),
        ("no_shell", True),
        ("no_dependency_resolution", True),
        ("no_pyside_import", True),
        ("no_qml_load", True),
        ("no_qt_plugin_runtime_discovery", True),
        ("no_build", True),
        ("no_packaging", True),
        ("no_artifact_creation", True),
        ("no_runtime", True),
        ("no_network", True),
        ("no_credentials", True),
        ("no_secret_value_reads", True),
        ("no_source_mutation", True),
        ("no_environment_mutation", True),
        ("no_orders", True),
    ),
)
SOURCE_BOUNDARY_FIELD_SPECS: Final[tuple[tuple[str, str | bool], ...]] = (
    ("source_step", "19.5"),
    ("source_block", "Q"),
    ("source_integrity_required", True),
    ("source_mutation_allowed", False),
    ("only_public_19_5_builder_read", True),
    ("block_p_builders_read", False),
    ("earlier_block_q_builders_read", False),
)


def _from_spec(data: Any) -> Any:
    if type(data) is tuple and len(data) == 2 and data[0] == "dict":
        return {k: _from_spec(v) for k, v in data[1]}
    if type(data) is tuple and len(data) == 2 and data[0] == "list":
        return [_from_spec(v) for v in data[1]]
    return data


def _trusted_source_19_5() -> dict[str, Any]:
    return _from_spec(SOURCE_19_5_SPEC)


def _manifest_rows() -> list[dict[str, Any]]:
    return _from_spec(EXECUTION_ROW_SPECS)


def _manifest_global_authorization() -> dict[str, bool]:
    return _from_spec(AUTHORIZATION_FIELD_SPECS)


def _manifest_summary() -> dict[str, Any]:
    return _from_spec(SUMMARY_FIELD_SPECS)


def _manifest_boundaries() -> dict[str, bool]:
    return _from_spec(BOUNDARY_FIELD_SPECS)


def _source_boundaries() -> dict[str, Any]:
    return {k: v for k, v in SOURCE_BOUNDARY_FIELD_SPECS}


def _scalar_same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if type(b) is bool or b is None:
        return a is b
    if type(b) is int:
        return int.__eq__(a, b) is True
    if type(b) is str:
        return str.__eq__(a, b) is True
    return False


def _exact_plain(a: Any, b: Any) -> bool:
    try:
        pending = [(a, b)]
        seen: list[tuple[int, int]] = []
        amap: list[tuple[int, int]] = []
        bmap: list[tuple[int, int]] = []
        while pending:
            x, y = pending.pop()
            if type(x) is not type(y):
                return False
            if type(y) is dict:
                xid, yid = id(x), id(y)
                for ax, by in amap:
                    if ax == xid and by != yid:
                        return False
                for by, ax in bmap:
                    if by == yid and ax != xid:
                        return False
                if (xid, yid) in seen:
                    continue
                seen.append((xid, yid))
                amap.append((xid, yid))
                bmap.append((yid, xid))
                lx = list(dict.keys(x))
                ly = list(dict.keys(y))
                if not all(type(k) is str for k in lx) or lx != ly:
                    return False
                vals_x = list(dict.values(x))
                vals_y = list(dict.values(y))
                if len(vals_x) != len(vals_y):
                    return False
                for i in range(len(vals_y) - 1, -1, -1):
                    pending.append((vals_x[i], vals_y[i]))
            elif type(y) is list:
                xid, yid = id(x), id(y)
                for ax, by in amap:
                    if ax == xid and by != yid:
                        return False
                for by, ax in bmap:
                    if by == yid and ax != xid:
                        return False
                if (xid, yid) in seen:
                    continue
                seen.append((xid, yid))
                amap.append((xid, yid))
                bmap.append((yid, xid))
                if list.__len__(x) != list.__len__(y):
                    return False
                for i in range(list.__len__(y) - 1, -1, -1):
                    pending.append((list.__getitem__(x, i), list.__getitem__(y, i)))
            elif not _scalar_same(x, y):
                return False
        return True
    except Exception:
        return False


def _reference(accepted: bool) -> dict[str, Any]:
    if accepted:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "kind": SOURCE_KIND,
            "block": "Q",
            "step": "19.5",
            "status": SOURCE_STATUS,
            "source_top_level_fields": list(SOURCE_TOP_LEVEL_FIELDS),
            "source_observation_plan_row_count": 11,
            "integrity_valid": True,
        }
    return {
        "schema_version": "",
        "kind": "",
        "block": "",
        "step": "",
        "status": "",
        "source_top_level_fields": [],
        "source_observation_plan_row_count": 0,
        "integrity_valid": False,
    }


def _preservation(accepted: bool) -> dict[str, bool]:
    return {
        "preserves_19_5_payload": False,
        "preserves_all_11_observation_plan_rows": accepted,
        "preserves_observation_plan_order": accepted,
        "preserves_one_to_one_mapping": accepted,
        "source_observation_plan_modified": False,
    }


def _scope(accepted: bool) -> dict[str, bool]:
    return {
        "execution_contract_defined": accepted,
        "source_only_definition": accepted,
        "future_read_only_observation_authorized": accepted,
        "actual_environment_observed": False,
        "current_stage_execution_authorized": False,
        "environment_build_ready": False,
    }


def _summary(accepted: bool) -> dict[str, Any]:
    s = _manifest_summary()
    if not accepted:
        for k in s:
            s[k] = False if type(s[k]) is bool else 0
    return s


def _global_auth(accepted: bool) -> dict[str, bool]:
    source = _manifest_global_authorization()
    if accepted:
        return source
    return {k: False for k in source}


def _evidence(accepted: bool) -> dict[str, bool]:
    data = {
        "source_read": accepted,
        "observation_execution_contract_definition_built": accepted,
        "actual_environment_observation_performed": False,
        "evidence_collected": False,
    }
    for name in (
        "subprocess",
        "shell",
        "powershell",
        "dependency_resolution",
        "pyside_import",
        "qml_load",
        "qt_plugin_runtime_discovery",
        "build_command_creation",
        "build_command_execution",
        "packaging",
        "artifact_creation",
        "artifact_scan",
        "artifact_signing",
        "installer_creation",
        "release",
        "application_runtime",
        "network",
        "credentials",
        "secret_values_read",
        "source_mutation",
        "environment_mutation",
        "orders",
    ):
        data[f"{name}_performed"] = False
    return data


def _boundaries(accepted: bool) -> dict[str, bool]:
    b = _manifest_boundaries()
    if not accepted:
        for k in (
            "reads_19_5_only",
            "one_to_one_observation_plan_mapping_required",
            "plain_data",
            "source_only",
            "bounded_read_only_collectors_only",
        ):
            b[k] = False
    return b


def _rows_from_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = dict.__getitem__(source, "observation_plan_rows")
    specs = _manifest_rows()
    if type(source_rows) is not list or list.__len__(source_rows) != list.__len__(specs):
        raise ValueError("source rows rejected")
    rows = []
    for i in range(list.__len__(specs)):
        src = list.__getitem__(source_rows, i)
        spec = list.__getitem__(specs, i)
        if type(src) is not dict or type(spec) is not dict:
            raise ValueError("row type rejected")
        for field in (
            "observation_plan_id",
            "read_model_id",
            "observation_target",
            "source_only_definition",
        ):
            if not _scalar_same(dict.__getitem__(src, field), dict.__getitem__(spec, field)):
                raise ValueError("row source mismatch")
        row = {
            "execution_contract_id": dict.__getitem__(spec, "execution_contract_id"),
            "observation_plan_id": dict.__getitem__(src, "observation_plan_id"),
            "read_model_id": dict.__getitem__(src, "read_model_id"),
            "observation_target": dict.__getitem__(src, "observation_target"),
            "collector_kind": dict.__getitem__(spec, "collector_kind"),
            "collector_source": dict.__getitem__(spec, "collector_source"),
            "result_type": dict.__getitem__(spec, "result_type"),
            "evidence_type": dict.__getitem__(spec, "evidence_type"),
            "future_read_authorized": True,
            "current_stage_execution_authorized": False,
            "mutation_authorized": False,
            "subprocess_authorized": False,
            "network_authorized": False,
            "credentials_authorized": False,
            "observation_performed": False,
            "evidence_collected": False,
            "contract_state": "authorized_for_future_read_only_observation",
            "source_only_definition": dict.__getitem__(src, "source_only_definition"),
        }
        if tuple(row) != ROW_FIELDS:
            raise ValueError("row order rejected")
        rows.append(row)
    return rows


def _payload(accepted: bool, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_observation_execution_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "title": TITLE,
        "block_q_windows_build_environment_observation_execution_contract_status": STATUS
        if accepted
        else BLOCKED_STATUS,
        "source_19_5_accepted": accepted,
        "block_q_windows_build_environment_observation_execution_contract_decision": DECISION
        if accepted
        else BLOCKED_DECISION,
        "observation_execution_contract_artifact_complete": accepted,
        "actual_environment_observation_performed": False,
        "current_stage_execution_authorized": False,
        "future_read_only_observation_authorized": accepted,
        "environment_build_ready": False,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "next_step_contract_missing": True,
        "block_q_19_5_observation_plan_reference": _reference(accepted),
        "source_observation_plan_preservation": _preservation(accepted),
        "observation_execution_contract_scope": _scope(accepted),
        "observation_execution_contract_rows": rows if rows is not None else [],
        "observation_execution_contract_summary": _summary(accepted),
        "global_authorization": _global_auth(accepted),
        "non_execution_observation_execution_contract_evidence": _evidence(accepted),
        "observation_execution_contract_boundaries": _boundaries(accepted),
        "source_boundaries": _source_boundaries(),
        "future_steps": [],
        "status": STATUS if accepted else BLOCKED_STATUS,
        "integrity_valid": True,
    }


def _canonical_nominal() -> dict[str, Any]:
    return _payload(True, _rows_from_source(_trusted_source_19_5()))


def _canonical_blocked() -> dict[str, Any]:
    return _payload(False, [])


def _source_accepted(source: Any) -> bool:
    if not _exact_plain(source, _trusted_source_19_5()):
        return False
    return _integrity_19_5(source) is True


def _integrity(payload: Any) -> bool:
    return _exact_plain(payload, _canonical_nominal()) or _exact_plain(
        payload, _canonical_blocked()
    )


def build_preview_block_q_windows_build_environment_observation_execution_contract() -> dict[
    str, Any
]:
    try:
        source = build_preview_block_q_windows_build_environment_observation_plan()
        if not _source_accepted(source):
            return _canonical_blocked()
        return _payload(True, _rows_from_source(source))
    except Exception:
        return _canonical_blocked()
