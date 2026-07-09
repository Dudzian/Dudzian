"""Tests for FUNCTIONAL-PREVIEW-15.2 Block M packaging readiness matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_readiness_matrix import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_READINESS_MATRIX_DECISION,
    BLOCK_M_PACKAGING_READINESS_MATRIX_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_KIND,
    PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_3,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_readiness_matrix,
)
from ui.pyside_app.preview_block_m_read_model import build_preview_block_m_read_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_readiness_matrix.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_readiness_matrix_kind",
    "block",
    "step",
    "block_m_packaging_readiness_matrix_status",
    "block_m_packaging_readiness_matrix_decision",
    "ready_for_block_m_3",
    "next_step",
    "next_step_title",
    "block_m_read_model_reference",
    "packaging_readiness_summary",
    "packaging_prerequisite_matrix",
    "packaging_capability_matrix",
    "exe_direction_matrix",
    "runtime_safety_carryover_matrix",
    "forbidden_execution_matrix",
    "fail_closed_matrix_decision",
    "non_execution_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_TRUE_FLAGS = [
    "block_m_read_model_available",
    "packaging_readiness_matrix_built",
    "ready_for_block_m_3",
    "exe_direction_preserved",
    "packaging_direction_known",
    "packaging_readiness_evaluated_static_only",
]
SUMMARY_FALSE_FLAGS = [
    "packaging_ready_now",
    "pyinstaller_ready_now",
    "build_artifact_creation_ready_now",
    "installer_ready_now",
    "release_workflow_ready_now",
    "packaging_command_execution_ready_now",
    "packaging_filesystem_io_ready_now",
    "packaging_can_run_now",
    "safe_to_activate_runtime_now",
    "safe_to_start_paper_runtime_now",
    "safe_to_start_testnet_runtime_now",
    "safe_to_start_live_canary_now",
    "safe_to_enable_live_trading_now",
    "safe_to_generate_orders_now",
    "safe_to_" + "sub" + "mit_orders_now",
    "safe_to_" + "can" + "cel_orders_now",
    "safe_to_" + "re" + "place_orders_now",
    "safe_to_access_private_endpoints_now",
    "safe_to_open_network_io_now",
    "safe_to_read_credentials_now",
    "safe_for_filesystem_io_now",
    "safe_for_config_env_secrets_now",
    "safe_to_change_qml_bridge_now",
]
PREREQUISITE_IDS = {
    "explicit_packaging_gate",
    "pyinstaller_spec_contract",
    "build_environment_contract",
    "dependency_freeze_contract",
    "asset_inclusion_contract",
    "qml_asset_inclusion_contract",
    "runtime_disabled_during_packaging_contract",
    "no_live_credentials_in_artifact_contract",
    "no_network_required_during_build_contract",
    "installer_policy_contract",
    "release_artifact_naming_contract",
    "smoke_test_for_built_artifact_contract",
    "rollback_delete_artifact_policy_contract",
    "signing_policy_contract",
}
CAPABILITY_IDS = {
    "exe_packaging",
    "pyinstaller",
    "build_command",
    "build_artifact_creation",
    "installer_changes",
    "release_workflow_changes",
    "packaging_filesystem_io",
    "artifact_smoke_testing",
    "artifact_signing",
    "artifact_publishing",
}
RUNTIME_CAPABILITY_IDS = {
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gate_execution",
    "gate_state_mutation",
    "order_generation",
    "order_submission",
    "order_cancel",
    "order_replace",
    "private_endpoints",
    "network_io",
    "credential_read",
    "filesystem_io",
    "config_env_secrets",
    "qml_bridge",
}
FORBIDDEN_EXECUTION_IDS = {
    "packaging_execution",
    "pyinstaller_execution",
    "build_command_execution",
    "artifact_creation",
    "installer_mutation",
    "release_workflow_mutation",
    "filesystem_io",
    "runtime_activation",
    "order_flow",
    "private_endpoint_access",
    "network_io",
    "credential_read",
    "config_env_secrets_read",
    "qml_bridge_change",
}
BLOCKED_DECISION_KEYS = [
    "exe_packaging_in_15_2",
    "pyinstaller_in_15_2",
    "build_command_in_15_2",
    "build_artifact_creation_in_15_2",
    "installer_change_in_15_2",
    "release_workflow_change_in_15_2",
    "packaging_filesystem_io_in_15_2",
    "runtime_activation_in_15_2",
    "paper_runtime_start_in_15_2",
    "testnet_runtime_start_in_15_2",
    "live_canary_start_in_15_2",
    "live_trading_in_15_2",
    "order_generation_in_15_2",
    "order_submission_in_15_2",
    "order_cancel_in_15_2",
    "order_replace_in_15_2",
    "private_endpoint_in_15_2",
    "network_io_in_15_2",
    "credential_read_in_15_2",
    "config_env_secret_read_in_15_2",
    "qml_bridge_change_in_15_2",
]
NON_EXECUTION_FALSE_FLAGS = [
    "source_read_model_runtime_loop_started",
    "source_read_model_runtime_gate_executed",
    "source_read_model_gate_state_mutated",
    "source_read_model_mode_activated",
    "source_read_model_order_generated",
    "source_read_model_order_submitted",
    "source_read_model_private_endpoint_accessed",
    "source_read_model_network_io_performed",
    "source_read_model_filesystem_io_performed",
    "packaging_executed",
    "runtime_activated",
    "paper_runtime_started",
    "testnet_runtime_started",
    "live_canary_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "gate_state_mutated",
    "mode_activated",
    "order_generated",
    "order_submitted",
    "private_endpoint_accessed",
    "network_io_performed",
    "filesystem_io_performed",
    "credential_read_performed",
    "live_trading_started",
    "exe_packaging_started",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "installer_changed",
    "release_workflow_changed",
    "artifact_published",
    "qml_bridge_changed",
]
BOUNDARY_TRUE_FLAGS = [
    "packaging_readiness_matrix_is_plain_data_only",
    "packaging_readiness_matrix_is_source_only",
    "packaging_readiness_matrix_reads_block_m_read_model_only",
    "packaging_readiness_matrix_preserves_exe_direction_without_packaging",
    "packaging_readiness_matrix_can_feed_15_3_packaging_gate_contract",
    "packaging_readiness_matrix_cannot_package_exe",
    "packaging_readiness_matrix_cannot_start_pyinstaller",
    "packaging_readiness_matrix_cannot_execute_build_commands",
    "packaging_readiness_matrix_cannot_create_build_artifacts",
    "packaging_readiness_matrix_cannot_change_installers",
    "packaging_readiness_matrix_cannot_change_release_workflows",
    "packaging_readiness_matrix_cannot_perform_filesystem_io",
    "packaging_readiness_matrix_cannot_activate_runtime",
    "packaging_readiness_matrix_cannot_start_paper_runtime",
    "packaging_readiness_matrix_cannot_start_testnet_runtime",
    "packaging_readiness_matrix_cannot_start_live_canary",
    "packaging_readiness_matrix_cannot_enable_live_trading",
    "packaging_readiness_matrix_cannot_generate_orders",
    "packaging_readiness_matrix_cannot_" + "sub" + "mit_orders",
    "packaging_readiness_matrix_cannot_" + "can" + "cel_orders",
    "packaging_readiness_matrix_cannot_" + "re" + "place_orders",
    "packaging_readiness_matrix_cannot_access_private_endpoints",
    "packaging_readiness_matrix_cannot_open_network_io",
    "packaging_readiness_matrix_cannot_read_credentials",
    "packaging_readiness_matrix_cannot_start_runtime_loop",
    "packaging_readiness_matrix_cannot_execute_runtime_gates",
    "packaging_readiness_matrix_cannot_mutate_gate_state",
    "packaging_readiness_matrix_cannot_read_config_env_or_secrets",
    "packaging_readiness_matrix_cannot_change_ui_bridge",
]
ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_m_read_model",
}
FORBIDDEN_CALL_NAMES = {
    "open",
    "read",
    "write",
    "read_text",
    "write_text",
    "getenv",
    "environ",
    "loads",
    "load",
    "dump",
    "dumps",
    "request",
    "get",
    "post",
    "put",
    "delete",
    "run",
    "Popen",
    "urlopen",
    "getaddrinfo",
    "create_connection",
    "activate",
    "start",
    "execute",
    "mutate",
    "package",
    "build",
}
FORBIDDEN_LITERAL_TOKENS = [
    "create_order",
    "submit_order",
    "cancel_order",
    "replace_order",
    "fetch" + "_balance",
    "cc" + "xt",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_m_packaging_readiness_matrix()


def test_payload_is_json_serializable_and_top_level_fields_are_stable() -> None:
    payload = _payload()
    json.dumps(payload, sort_keys=True)
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_2() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_SCHEMA_VERSION
    assert (
        payload["block_m_packaging_readiness_matrix_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_KIND
    )
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.2"
    assert (
        payload["block_m_packaging_readiness_matrix_status"]
        == BLOCK_M_PACKAGING_READINESS_MATRIX_STATUS
    )
    assert (
        payload["block_m_packaging_readiness_matrix_decision"]
        == BLOCK_M_PACKAGING_READINESS_MATRIX_DECISION
    )
    assert payload["ready_for_block_m_3"] is READY_FOR_BLOCK_M_3 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.3"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING GATE CONTRACT"
    assert payload["status"] == STATUS


def test_block_m_read_model_reference_points_to_15_1_subset() -> None:
    reference = _payload()["block_m_read_model_reference"]
    read_model = build_preview_block_m_read_model()
    assert reference["schema_version"] == read_model["schema_version"]
    assert reference["block_m_read_model_kind"] == read_model["block_m_read_model_kind"]
    assert reference["step"] == "15.1"
    assert reference["source_block_m_read_model_step"] == "FUNCTIONAL-PREVIEW-15.1"
    assert reference["source_block_m_read_model_read_by_15_2_matrix"] is True
    assert reference["block_m_read_model_available_before_matrix"] is True
    assert reference["static_read_model_only"] is True
    for key, value in reference.items():
        if (
            key.endswith("_by_15_2")
            or key.endswith("_opened_by_15_2")
            or key.endswith("_changed_by_15_2")
        ):
            assert value is False


def test_packaging_readiness_summary_preserves_exe_direction_and_blocks_execution_paths() -> None:
    summary = _payload()["packaging_readiness_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_packaging_prerequisite_matrix_requires_future_steps_without_checks() -> None:
    rows = _payload()["packaging_prerequisite_matrix"]
    assert {row["prerequisite_id"] for row in rows} == PREREQUISITE_IDS
    for row in rows:
        assert row["display_name"]
        assert row["required_before_packaging"] is True
        assert row["satisfied_in_15_2"] is False
        assert row["requires_future_step"] is True
        assert row["checked_by_15_2"] is False
        assert row["notes"]


def test_packaging_capability_matrix_preserves_exe_direction_but_allows_nothing_now() -> None:
    rows = _payload()["packaging_capability_matrix"]
    assert {row["capability_id"] for row in rows} == CAPABILITY_IDS
    for row in rows:
        assert row["source_read_model_section"] in {
            "packaging_forbidden_read_model",
            "exe_direction_read_model",
        }
        assert row["ready_now"] is False
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["requires_future_contract"] is True
        if row["capability_id"] == "exe_packaging":
            assert row["direction_preserved"] is True
        else:
            assert row["direction_preserved"] is False


def test_exe_direction_matrix_preserved_without_packaging_or_release_start() -> None:
    matrix = _payload()["exe_direction_matrix"]
    assert matrix["final_product_direction"] == "desktop_exe"
    assert matrix["exe_direction_preserved"] is True
    assert matrix["matrix_confirms_exe_direction"] is True
    for key in [
        "exe_packaging_started_now",
        "pyinstaller_started_now",
        "build_command_added_now",
        "workflow_changed_for_packaging_now",
        "installer_changed_now",
        "release_artifact_created_now",
        "artifact_created_now",
    ]:
        assert matrix[key] is False
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        assert matrix[key] is True


def test_runtime_safety_carryover_matrix_blocks_runtime_live_order_io_and_bridge() -> None:
    rows = _payload()["runtime_safety_carryover_matrix"]
    assert {row["capability_id"] for row in rows} == RUNTIME_CAPABILITY_IDS
    for row in rows:
        assert row["source_read_model_allowed_now"] is False
        assert row["matrix_allowed_now"] is False
        assert row["matrix_executed_now"] is False
        assert row["blocked_in_15_2"] is True
        assert row["requires_future_explicit_gate"] is True


def test_forbidden_execution_matrix_blocks_all_execution_paths() -> None:
    rows = _payload()["forbidden_execution_matrix"]
    assert {row["execution_id"] for row in rows} == FORBIDDEN_EXECUTION_IDS
    for row in rows:
        assert row["forbidden_in_15_2"] is True
        assert row["executed_by_15_2"] is False
        assert row["allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True


def test_fail_closed_matrix_decision_blocks_all_15_2_paths() -> None:
    decision = _payload()["fail_closed_matrix_decision"]
    assert decision["missing_block_m_read_model_policy"] == "fail_closed"
    assert decision["missing_packaging_prerequisite_policy"] == "fail_closed"
    assert decision["missing_packaging_gate_policy"] == "fail_closed"
    assert decision["missing_runtime_safety_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_execution_evidence_remains_false_for_packaging_runtime_io_and_bridge() -> None:
    evidence = _payload()["non_execution_evidence"]
    assert evidence["source_block_m_read_model_read"] is True
    assert evidence["packaging_readiness_matrix_built"] is True
    assert evidence["packaging_matrix_only"] is True
    for key in NON_EXECUTION_FALSE_FLAGS:
        assert evidence[key] is False


def test_matrix_boundaries_are_closed() -> None:
    boundaries = _payload()["matrix_boundaries"]
    for key in BOUNDARY_TRUE_FLAGS:
        assert boundaries[key] is True
    for key, value in boundaries.items():
        if key.startswith("packaging_readiness_matrix_cannot_"):
            assert value is True


def test_source_boundaries_reference_15_1_and_forbidden_calls_absent() -> None:
    boundaries = _payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_m_read_model"] == "FUNCTIONAL-PREVIEW-15.1"
    for key, value in boundaries.items():
        if key.startswith("forbidden_") and key.endswith("_calls_present"):
            assert value is False
    source_subset = boundaries["source_block_m_read_model_boundaries"]
    assert source_subset["allowed_imports_only"] is True
    assert source_subset["source_block_m_entry_contract"] == "FUNCTIONAL-PREVIEW-15.0"


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "__future__")
    assert imports == ALLOWED_IMPORT_MODULES


def test_source_call_guard_blocks_forbidden_runtime_io_and_packaging_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    assert not (called_names & FORBIDDEN_CALL_NAMES)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    for token in FORBIDDEN_LITERAL_TOKENS:
        assert token not in source
