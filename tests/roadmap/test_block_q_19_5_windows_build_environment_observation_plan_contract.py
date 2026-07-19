from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/roadmap/block_q_19_5_windows_build_environment_observation_plan.yaml"
)

IDENTITY = {
    "schema_version": "preview_block_q_windows_build_environment_observation_plan_contract.v1",
    "kind": "functional_preview_block_q_windows_build_environment_observation_plan_contract",
    "block": "Q",
    "step": "19.5",
    "title": "WINDOWS BUILD ENVIRONMENT OBSERVATION PLAN",
    "source_step": "19.4",
    "source_kind": "functional_preview_block_q_windows_build_environment_read_model",
    "source_builder": "build_preview_block_q_windows_build_environment_read_model",
    "source_only": True,
    "actual_environment_observation_performed": False,
    "environment_build_ready": False,
    "execution_authorized": False,
}

ROW_FIELDS = [
    "observation_plan_id",
    "read_model_id",
    "contract_id",
    "matrix_id",
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "observation_target",
    "observation_method",
    "expected_value_type",
    "evidence_required",
    "observation_authorized",
    "observation_status",
    "evidence_status",
    "plan_state",
    "source_only_definition",
]

EXPECTED_ROWS = [
    (
        "observation_plan_windows_host",
        "read_model_windows_host",
        "contract_windows_host",
        "matrix_windows_host",
        "windows_host",
        "windows_host_required",
        "platform",
        "windows_host_identity",
        "windows_platform_identity_query",
        "string",
    ),
    (
        "observation_plan_supported_windows_version",
        "read_model_supported_windows_version",
        "contract_supported_windows_version",
        "matrix_supported_windows_version",
        "supported_windows_version",
        "supported_windows_version_must_be_confirmed",
        "platform",
        "windows_version",
        "windows_version_query",
        "string",
    ),
    (
        "observation_plan_python_version",
        "read_model_python_version",
        "contract_python_version",
        "matrix_python_version",
        "python_version",
        "exact_python_version_must_be_confirmed",
        "runtime",
        "python_interpreter_version",
        "python_version_query",
        "string",
    ),
    (
        "observation_plan_pyside_version",
        "read_model_pyside_version",
        "contract_pyside_version",
        "matrix_pyside_version",
        "pyside_version",
        "exact_pyside_version_must_be_confirmed",
        "framework",
        "pyside_package_version",
        "python_package_metadata_query",
        "string",
    ),
    (
        "observation_plan_packaging_tool_and_version",
        "read_model_packaging_tool_and_version",
        "contract_packaging_tool_and_version",
        "matrix_packaging_tool_and_version",
        "packaging_tool_and_version",
        "exact_packaging_tool_and_version_must_be_selected",
        "packaging",
        "packaging_tool_and_version",
        "packaging_configuration_query",
        "string",
    ),
    (
        "observation_plan_desktop_entrypoint",
        "read_model_desktop_entrypoint",
        "contract_desktop_entrypoint",
        "matrix_desktop_entrypoint",
        "desktop_entrypoint",
        "desktop_entrypoint_must_be_confirmed",
        "application",
        "desktop_entrypoint",
        "repository_configuration_read",
        "string",
    ),
    (
        "observation_plan_qml_assets",
        "read_model_qml_assets",
        "contract_qml_assets",
        "matrix_qml_assets",
        "qml_assets",
        "qml_assets_must_be_confirmed",
        "assets",
        "qml_asset_manifest",
        "repository_manifest_read",
        "list",
    ),
    (
        "observation_plan_qt_plugins",
        "read_model_qt_plugins",
        "contract_qt_plugins",
        "matrix_qt_plugins",
        "qt_plugins",
        "qt_plugins_must_be_confirmed",
        "plugins",
        "qt_plugin_requirements",
        "repository_manifest_read",
        "list",
    ),
    (
        "observation_plan_dependency_lock",
        "read_model_dependency_lock",
        "contract_dependency_lock",
        "matrix_dependency_lock",
        "dependency_lock",
        "dependency_lock_must_be_resolved",
        "dependencies",
        "dependency_lock_state",
        "dependency_lock_read",
        "object",
    ),
    (
        "observation_plan_secret_and_local_data_exclusion",
        "read_model_secret_and_local_data_exclusion",
        "contract_secret_and_local_data_exclusion",
        "matrix_secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_must_be_confirmed",
        "security",
        "secret_and_local_data_exclusion_policy",
        "repository_policy_read",
        "object",
    ),
    (
        "observation_plan_output_name_and_version_policy",
        "read_model_output_name_and_version_policy",
        "contract_output_name_and_version_policy",
        "matrix_output_name_and_version_policy",
        "output_name_and_version_policy",
        "output_name_and_version_policy_must_be_confirmed",
        "output",
        "output_name_and_version_policy",
        "repository_policy_read",
        "object",
    ),
]

COMMON_VALUES = {
    "required": True,
    "evidence_required": True,
    "observation_authorized": False,
    "observation_status": "not_performed",
    "evidence_status": "not_collected",
    "plan_state": "defined_but_not_authorized",
    "source_only_definition": True,
}

SUMMARY = {
    "observation_plan_row_count": 11,
    "required_count": 11,
    "authorized_observation_count": 0,
    "performed_observation_count": 0,
    "evidence_required_count": 11,
    "evidence_collected_count": 0,
    "plan_definition_complete": True,
    "actual_environment_observation_complete": False,
    "environment_build_ready": False,
}

BOUNDARIES = {
    "reads_19_4_only": True,
    "one_to_one_read_model_mapping_required": True,
    "plain_data": True,
    "source_only": True,
    "no_runtime_execution": True,
    "no_environment_mutation": True,
    "no_source_mutation": True,
    "no_network": True,
    "no_credentials": True,
    "no_orders": True,
}

FORBIDDEN_VALUE_FRAGMENTS = (
    "subprocess",
    "os.system",
    "exec(",
    "eval(",
    "powershell",
    "cmd.exe",
    "bash",
    "sh -c",
    "pyside6",
    "qml load",
    "qt plugin discovery",
    "credentials/",
    ".credentials",
    "secret_path",
    "private_key",
    "api_key",
    "order placement",
    "orders_enabled: true",
    "place_order",
)


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def test_block_q_19_5_contract_identity_next_step_and_summary_are_exact() -> None:
    contract = _load_contract()

    for key, value in IDENTITY.items():
        assert contract[key] == value
    assert contract["next_step"] == ""
    assert contract["next_step_title"] == ""
    assert contract["next_step_contract_missing"] is True
    assert contract["next_step_authorized"] is False
    assert contract["summary"] == SUMMARY
    assert contract["boundaries"] == BOUNDARIES


def test_block_q_19_5_observation_rows_match_19_4_one_to_one_plain_data() -> None:
    rows = _load_contract()["observation_plan_rows"]

    assert len(rows) == 11
    assert [row["observation_plan_id"] for row in rows] == [row[0] for row in EXPECTED_ROWS]

    for field in (
        "observation_plan_id",
        "read_model_id",
        "contract_id",
        "matrix_id",
        "inventory_id",
        "requirement_field",
    ):
        values = [row[field] for row in rows]
        assert len(values) == len(set(values))

    for row, expected in zip(rows, EXPECTED_ROWS, strict=True):
        assert list(row) == ROW_FIELDS
        assert row == {
            "observation_plan_id": expected[0],
            "read_model_id": expected[1],
            "contract_id": expected[2],
            "matrix_id": expected[3],
            "inventory_id": expected[4],
            "requirement_field": expected[5],
            "category": expected[6],
            "required": True,
            "observation_target": expected[7],
            "observation_method": expected[8],
            "expected_value_type": expected[9],
            "evidence_required": True,
            "observation_authorized": False,
            "observation_status": "not_performed",
            "evidence_status": "not_collected",
            "plan_state": "defined_but_not_authorized",
            "source_only_definition": True,
        }
        for key, value in COMMON_VALUES.items():
            assert row[key] == value


def test_block_q_19_5_authorization_flags_forbid_real_execution() -> None:
    contract = _load_contract()

    assert contract["actual_environment_observation_performed"] is False
    assert contract["environment_build_ready"] is False
    assert contract["execution_authorized"] is False
    assert contract["next_step_authorized"] is False
    assert all(value is False for value in contract["forbidden_authorizations"].values())
    assert all(row["observation_authorized"] is False for row in contract["observation_plan_rows"])
    assert contract["summary"]["authorized_observation_count"] == 0
    assert contract["summary"]["performed_observation_count"] == 0


def test_block_q_19_5_manifest_remains_plain_data_without_commands_credentials_or_orders() -> None:
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8").lower()
    contract = _load_contract()

    assert contract["boundaries"]["plain_data"] is True
    assert contract["boundaries"]["no_runtime_execution"] is True
    assert contract["boundaries"]["no_network"] is True
    assert contract["boundaries"]["no_credentials"] is True
    assert contract["boundaries"]["no_orders"] is True
    assert "19.6" not in contract_text
    for fragment in FORBIDDEN_VALUE_FRAGMENTS:
        assert fragment not in contract_text


TOP_LEVEL_FIELDS = [
    "schema_version",
    "kind",
    "block",
    "step",
    "title",
    "source_step",
    "source_kind",
    "source_builder",
    "source_only",
    "actual_environment_observation_performed",
    "environment_build_ready",
    "execution_authorized",
    "next_step",
    "next_step_title",
    "next_step_contract_missing",
    "next_step_authorized",
    "forbidden_authorizations",
    "observation_plan_rows",
    "summary",
    "boundaries",
]

FORBIDDEN_AUTHORIZATION_FIELDS = [
    "repo_runtime_scan_authorized",
    "filesystem_runtime_scan_authorized",
    "environment_scan_authorized",
    "windows_host_inspection_authorized",
    "windows_version_read_authorized",
    "interpreter_observation_authorized",
    "dependency_resolution_authorized",
    "pyside_import_authorized",
    "qml_load_authorized",
    "qt_plugin_discovery_authorized",
    "build_command_creation_authorized",
    "build_command_execution_authorized",
    "packaging_authorized",
    "artifact_creation_authorized",
    "artifact_scan_authorized",
    "artifact_signing_authorized",
    "installer_creation_authorized",
    "release_authorized",
    "application_runtime_authorized",
    "network_open_authorized",
    "credentials_read_authorized",
    "orders_authorized",
]


def test_manifest_top_level_and_forbidden_authorization_order() -> None:
    data = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert list(data) == TOP_LEVEL_FIELDS
    assert list(data["forbidden_authorizations"]) == FORBIDDEN_AUTHORIZATION_FIELDS
    assert all(value is False for value in data["forbidden_authorizations"].values())
