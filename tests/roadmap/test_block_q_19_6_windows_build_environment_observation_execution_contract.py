from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "docs/roadmap/block_q_19_6_windows_build_environment_observation_execution_contract.yaml"
)
SOURCE_19_5_PATH = (
    ROOT / "docs/roadmap/block_q_19_5_windows_build_environment_observation_plan.yaml"
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "kind",
    "block",
    "step",
    "title",
    "goal",
    "source_step",
    "source_kind",
    "source_builder",
    "source_only",
    "contract_artifact_complete",
    "actual_environment_observation_performed",
    "current_stage_execution_authorized",
    "future_read_only_observation_authorized",
    "environment_build_ready",
    "next_step",
    "next_step_title",
    "next_step_contract_missing",
    "next_step_authorized",
    "source_acceptance",
    "global_authorization",
    "observation_execution_contract_rows",
    "summary",
    "boundaries",
]

IDENTITY = {
    "schema_version": "preview_block_q_windows_build_environment_observation_execution_contract.v1",
    "kind": "functional_preview_block_q_windows_build_environment_observation_execution_contract",
    "block": "Q",
    "step": "19.6",
    "title": "WINDOWS BUILD ENVIRONMENT OBSERVATION EXECUTION CONTRACT",
    "goal": (
        "Authorize an exact future read-only observation contract for the 11 Block Q Windows "
        "build environment observation plan rows without performing observation in this stage."
    ),
    "source_step": "19.5",
    "source_kind": "functional_preview_block_q_windows_build_environment_observation_plan",
    "source_builder": "build_preview_block_q_windows_build_environment_observation_plan",
    "source_only": True,
    "contract_artifact_complete": True,
    "actual_environment_observation_performed": False,
    "current_stage_execution_authorized": False,
    "future_read_only_observation_authorized": True,
    "environment_build_ready": False,
    "next_step": "",
    "next_step_title": "",
    "next_step_contract_missing": True,
    "next_step_authorized": False,
}

SOURCE_ACCEPTANCE = {
    "requires_nominal_19_5": True,
    "requires_11_observation_plan_rows": True,
    "requires_source_only_19_5": True,
    "requires_zero_performed_observations": True,
    "requires_zero_collected_evidence": True,
    "requires_environment_build_not_ready": True,
    "requires_next_step_contract_missing": True,
    "source_rejected_blocked_payload_required": True,
}

GLOBAL_AUTHORIZATION = {
    "current_stage_execution_authorized": False,
    "future_read_only_observation_authorized": True,
    "subprocess_authorized": False,
    "shell_authorized": False,
    "powershell_authorized": False,
    "dependency_resolution_authorized": False,
    "pyside_import_authorized": False,
    "qml_load_authorized": False,
    "qt_plugin_runtime_discovery_authorized": False,
    "build_command_creation_authorized": False,
    "build_command_execution_authorized": False,
    "packaging_authorized": False,
    "artifact_creation_authorized": False,
    "artifact_scan_authorized": False,
    "artifact_signing_authorized": False,
    "installer_creation_authorized": False,
    "release_authorized": False,
    "application_runtime_authorized": False,
    "network_open_authorized": False,
    "credentials_read_authorized": False,
    "secret_values_read_authorized": False,
    "source_mutation_authorized": False,
    "environment_mutation_authorized": False,
    "orders_authorized": False,
}

ROW_FIELDS = [
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
]

EXPECTED_ROWS = [
    (
        "execution_contract_windows_host",
        "observation_plan_windows_host",
        "read_model_windows_host",
        "windows_host_identity",
        "python_standard_library",
        "platform.system_platform.machine_platform.release",
        "object",
        "structured_platform_metadata",
    ),
    (
        "execution_contract_supported_windows_version",
        "observation_plan_supported_windows_version",
        "read_model_supported_windows_version",
        "windows_version",
        "python_standard_library",
        "platform.release_platform.version",
        "object",
        "structured_windows_version_metadata",
    ),
    (
        "execution_contract_python_version",
        "observation_plan_python_version",
        "read_model_python_version",
        "python_interpreter_version",
        "python_standard_library",
        "sys.version_info",
        "object",
        "structured_python_version_metadata",
    ),
    (
        "execution_contract_pyside_version",
        "observation_plan_pyside_version",
        "read_model_pyside_version",
        "pyside_package_version",
        "installed_package_metadata",
        "importlib.metadata.version_PySide6",
        "string",
        "package_version_metadata",
    ),
    (
        "execution_contract_packaging_tool_and_version",
        "observation_plan_packaging_tool_and_version",
        "read_model_packaging_tool_and_version",
        "packaging_tool_and_version",
        "bounded_repository_configuration",
        "pyproject_packaging_configuration_and_installed_package_metadata",
        "object",
        "packaging_configuration_metadata",
    ),
    (
        "execution_contract_desktop_entrypoint",
        "observation_plan_desktop_entrypoint",
        "read_model_desktop_entrypoint",
        "desktop_entrypoint",
        "bounded_repository_configuration",
        "pyproject_and_declared_desktop_entrypoint_configuration",
        "string",
        "entrypoint_configuration_metadata",
    ),
    (
        "execution_contract_qml_assets",
        "observation_plan_qml_assets",
        "read_model_qml_assets",
        "qml_asset_manifest",
        "bounded_repository_file_listing",
        "ui_pyside_declared_qml_paths",
        "list",
        "qml_asset_path_manifest",
    ),
    (
        "execution_contract_qt_plugins",
        "observation_plan_qt_plugins",
        "read_model_qt_plugins",
        "qt_plugin_requirements",
        "bounded_repository_configuration",
        "declared_qt_plugin_requirements_only",
        "list",
        "qt_plugin_requirement_manifest",
    ),
    (
        "execution_contract_dependency_lock",
        "observation_plan_dependency_lock",
        "read_model_dependency_lock",
        "dependency_lock_state",
        "bounded_repository_file_metadata",
        "declared_dependency_and_lock_files",
        "object",
        "dependency_lock_file_metadata_and_hash",
    ),
    (
        "execution_contract_secret_and_local_data_exclusion",
        "observation_plan_secret_and_local_data_exclusion",
        "read_model_secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_policy",
        "bounded_repository_policy_read",
        "gitignore_and_declared_exclusion_policy_files",
        "object",
        "exclusion_policy_metadata",
    ),
    (
        "execution_contract_output_name_and_version_policy",
        "observation_plan_output_name_and_version_policy",
        "read_model_output_name_and_version_policy",
        "output_name_and_version_policy",
        "bounded_repository_configuration",
        "pyproject_and_declared_release_configuration",
        "object",
        "output_policy_metadata",
    ),
]

COMMON_ROW_VALUES = {
    "future_read_authorized": True,
    "current_stage_execution_authorized": False,
    "mutation_authorized": False,
    "subprocess_authorized": False,
    "network_authorized": False,
    "credentials_authorized": False,
    "observation_performed": False,
    "evidence_collected": False,
    "contract_state": "authorized_for_future_read_only_observation",
    "source_only_definition": True,
}

SUMMARY = {
    "execution_contract_row_count": 11,
    "future_read_authorized_count": 11,
    "current_stage_execution_authorized_count": 0,
    "mutation_authorized_count": 0,
    "subprocess_authorized_count": 0,
    "network_authorized_count": 0,
    "credentials_authorized_count": 0,
    "performed_observation_count": 0,
    "evidence_collected_count": 0,
    "contract_definition_complete": True,
    "actual_environment_observation_complete": False,
    "environment_build_ready": False,
}

BOUNDARIES = {
    "reads_19_5_only": True,
    "one_to_one_observation_plan_mapping_required": True,
    "plain_data": True,
    "source_only": True,
    "no_current_stage_execution": True,
    "future_execution_requires_separate_contract": True,
    "bounded_read_only_collectors_only": True,
    "no_subprocess": True,
    "no_shell": True,
    "no_dependency_resolution": True,
    "no_pyside_import": True,
    "no_qml_load": True,
    "no_qt_plugin_runtime_discovery": True,
    "no_build": True,
    "no_packaging": True,
    "no_artifact_creation": True,
    "no_runtime": True,
    "no_network": True,
    "no_credentials": True,
    "no_secret_value_reads": True,
    "no_source_mutation": True,
    "no_environment_mutation": True,
    "no_orders": True,
}

FORBIDDEN_TEXT_FRAGMENTS = (
    "subprocess.run(",
    "popen(",
    "os.system(",
    "exec(",
    "eval(",
    "powershell.exe",
    "cmd.exe",
    "bash -c",
    "sh -c",
    "import pyside6",
    "pyside6.qtqml",
    "qmlengine",
    "pluginpath",
    "/credentials",
    "\\credentials",
    ".credentials",
    "secret_path",
    "secret_value:",
    "private_key:",
    "api_key:",
    "orders_enabled: true",
    "place_order(",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_block_q_19_6_schema_identity_authorization_and_boundaries_are_exact() -> None:
    contract = _load_yaml(CONTRACT_PATH)

    assert list(contract) == TOP_LEVEL_FIELDS
    for key, value in IDENTITY.items():
        assert contract[key] == value
    assert list(contract["source_acceptance"]) == list(SOURCE_ACCEPTANCE)
    assert contract["source_acceptance"] == SOURCE_ACCEPTANCE
    assert list(contract["global_authorization"]) == list(GLOBAL_AUTHORIZATION)
    assert contract["global_authorization"] == GLOBAL_AUTHORIZATION
    assert contract["future_read_only_observation_authorized"] is True
    assert contract["current_stage_execution_authorized"] is False
    forbidden_values = [
        value
        for key, value in contract["global_authorization"].items()
        if key != "future_read_only_observation_authorized"
    ]
    assert all(value is False for value in forbidden_values)
    assert contract["summary"] == SUMMARY
    assert contract["boundaries"] == BOUNDARIES


def test_block_q_19_6_rows_are_exact_unique_and_map_one_to_one_to_19_5() -> None:
    contract = _load_yaml(CONTRACT_PATH)
    source = _load_yaml(SOURCE_19_5_PATH)
    rows = contract["observation_execution_contract_rows"]
    source_rows = source["observation_plan_rows"]

    assert len(rows) == 11
    assert len(source_rows) == 11
    assert [row["execution_contract_id"] for row in rows] == [row[0] for row in EXPECTED_ROWS]
    assert [row["observation_plan_id"] for row in rows] == [
        row["observation_plan_id"] for row in source_rows
    ]
    assert [row["read_model_id"] for row in rows] == [row["read_model_id"] for row in source_rows]
    assert [row["observation_target"] for row in rows] == [
        row["observation_target"] for row in source_rows
    ]

    for field in ("execution_contract_id", "observation_plan_id", "read_model_id"):
        values = [row[field] for row in rows]
        assert len(values) == len(set(values))

    for row, expected in zip(rows, EXPECTED_ROWS, strict=True):
        assert list(row) == ROW_FIELDS
        assert row == {
            "execution_contract_id": expected[0],
            "observation_plan_id": expected[1],
            "read_model_id": expected[2],
            "observation_target": expected[3],
            "collector_kind": expected[4],
            "collector_source": expected[5],
            "result_type": expected[6],
            "evidence_type": expected[7],
            **COMMON_ROW_VALUES,
        }
        for key, value in COMMON_ROW_VALUES.items():
            assert row[key] == value


def test_block_q_19_6_defines_no_next_stage_execution_snippets_or_sensitive_reads() -> None:
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8").lower()
    contract = _load_yaml(CONTRACT_PATH)

    assert "19.7" not in contract_text
    assert contract["next_step"] == ""
    assert contract["next_step_title"] == ""
    assert contract["next_step_contract_missing"] is True
    assert contract["next_step_authorized"] is False
    assert contract["actual_environment_observation_performed"] is False
    assert contract["summary"]["performed_observation_count"] == 0
    assert contract["summary"]["evidence_collected_count"] == 0
    assert contract["global_authorization"]["build_command_creation_authorized"] is False
    assert contract["global_authorization"]["build_command_execution_authorized"] is False
    assert contract["global_authorization"]["packaging_authorized"] is False
    assert contract["global_authorization"]["network_open_authorized"] is False
    assert contract["global_authorization"]["orders_authorized"] is False
    for fragment in FORBIDDEN_TEXT_FRAGMENTS:
        assert fragment not in contract_text
