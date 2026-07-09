"""Tests for FUNCTIONAL-PREVIEW-15.1 Block M read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_entry_contract import build_preview_block_m_entry_contract
from ui.pyside_app.preview_block_m_read_model import (
    BLOCK_ID,
    BLOCK_M_READ_MODEL_DECISION,
    BLOCK_M_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_READ_MODEL_KIND,
    PREVIEW_BLOCK_M_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_2,
    STATUS,
    STEP_ID,
    build_preview_block_m_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_read_model_kind",
    "block",
    "step",
    "block_m_read_model_status",
    "block_m_read_model_decision",
    "ready_for_block_m_2",
    "next_step",
    "next_step_title",
    "block_m_entry_contract_reference",
    "block_m_read_summary",
    "exe_direction_read_model",
    "packaging_forbidden_read_model",
    "runtime_forbidden_read_model",
    "capability_read_rows",
    "fail_closed_read_model_decision",
    "non_activation_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

SUMMARY_TRUE_FLAGS = [
    "block_m_entry_contract_available",
    "block_m_opened",
    "block_m_read_model_built",
    "ready_for_block_m_2",
    "exe_direction_preserved",
    "entry_contract_read_only",
    "read_model_plain_data_only",
]
SUMMARY_FALSE_FLAGS = [
    "exe_packaging_in_scope_now",
    "pyinstaller_in_scope_now",
    "build_artifact_creation_in_scope_now",
    "release_packaging_in_scope_now",
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
RUNTIME_FALSE_FLAGS = [
    "runtime_activation_allowed_now",
    "paper_runtime_start_allowed_now",
    "testnet_runtime_start_allowed_now",
    "live_canary_start_allowed_now",
    "live_trading_allowed_now",
    "runtime_loop_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "filesystem_io_allowed_now",
    "config_env_secret_read_allowed_now",
    "qml_bridge_change_allowed_now",
]
BLOCKED_DECISION_KEYS = [
    "exe_packaging_in_15_1",
    "pyinstaller_in_15_1",
    "build_artifact_creation_in_15_1",
    "release_workflow_change_in_15_1",
    "runtime_activation_in_15_1",
    "paper_runtime_start_in_15_1",
    "testnet_runtime_start_in_15_1",
    "live_canary_start_in_15_1",
    "live_trading_in_15_1",
    "order_generation_in_15_1",
    "order_submission_in_15_1",
    "order_cancel_in_15_1",
    "order_replace_in_15_1",
    "private_endpoint_in_15_1",
    "network_io_in_15_1",
    "filesystem_io_in_15_1",
    "credential_read_in_15_1",
    "config_env_secret_read_in_15_1",
    "qml_bridge_change_in_15_1",
]
NON_ACTIVATION_FALSE_FLAGS = [
    "source_block_m_entry_runtime_loop_started",
    "source_block_m_entry_runtime_gate_executed",
    "source_block_m_entry_gate_state_mutated",
    "source_block_m_entry_mode_activated",
    "source_block_m_entry_order_generated",
    "source_block_m_entry_order_submitted",
    "source_block_m_entry_private_endpoint_accessed",
    "source_block_m_entry_network_io_performed",
    "source_block_m_entry_filesystem_io_performed",
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
    "build_artifact_created",
    "release_workflow_changed",
    "qml_bridge_changed",
]
BOUNDARY_TRUE_FLAGS = [
    "block_m_read_model_is_plain_data_only",
    "block_m_read_model_is_source_only",
    "block_m_read_model_reads_entry_contract_only",
    "block_m_read_model_preserves_exe_direction_without_packaging",
    "block_m_read_model_can_feed_15_2_packaging_readiness_matrix",
    "block_m_read_model_cannot_package_exe",
    "block_m_read_model_cannot_start_pyinstaller",
    "block_m_read_model_cannot_create_build_artifacts",
    "block_m_read_model_cannot_change_release_workflows",
    "block_m_read_model_cannot_activate_runtime",
    "block_m_read_model_cannot_start_paper_runtime",
    "block_m_read_model_cannot_start_testnet_runtime",
    "block_m_read_model_cannot_start_live_canary",
    "block_m_read_model_cannot_enable_live_trading",
    "block_m_read_model_cannot_generate_orders",
    "block_m_read_model_cannot_" + "sub" + "mit_orders",
    "block_m_read_model_cannot_" + "can" + "cel_orders",
    "block_m_read_model_cannot_" + "re" + "place_orders",
    "block_m_read_model_cannot_access_private_endpoints",
    "block_m_read_model_cannot_open_network_io",
    "block_m_read_model_cannot_read_credentials",
    "block_m_read_model_cannot_start_runtime_loop",
    "block_m_read_model_cannot_execute_runtime_gates",
    "block_m_read_model_cannot_mutate_gate_state",
    "block_m_read_model_cannot_perform_filesystem_io",
    "block_m_read_model_cannot_read_config_env_or_secrets",
    "block_m_read_model_cannot_change_ui_bridge",
]
ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_m_entry_contract",
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
    return build_preview_block_m_read_model()


def test_payload_is_json_serializable_and_top_level_fields_are_stable() -> None:
    payload = _payload()
    json.dumps(payload, sort_keys=True)
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_1() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_M_READ_MODEL_SCHEMA_VERSION
    assert payload["block_m_read_model_kind"] == PREVIEW_BLOCK_M_READ_MODEL_KIND
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.1"
    assert payload["block_m_read_model_status"] == BLOCK_M_READ_MODEL_STATUS
    assert payload["block_m_read_model_decision"] == BLOCK_M_READ_MODEL_DECISION
    assert payload["ready_for_block_m_2"] is READY_FOR_BLOCK_M_2 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.2"
    assert payload["next_step_title"] == NEXT_STEP_TITLE
    assert payload["status"] == STATUS


def test_entry_contract_reference_points_to_15_0_subset() -> None:
    payload = _payload()
    entry = build_preview_block_m_entry_contract()
    reference = payload["block_m_entry_contract_reference"]
    assert reference["schema_version"] == entry["schema_version"]
    assert reference["block_m_entry_contract_kind"] == entry["block_m_entry_contract_kind"]
    assert reference["step"] == "15.0"
    assert reference["source_block_m_entry_step"] == "FUNCTIONAL-PREVIEW-15.0"
    assert reference["source_block_m_entry_read_by_15_1_read_model"] is True
    assert reference["block_m_opened_before_read_model"] is True
    assert reference["static_entry_contract_only"] is True
    for key, value in reference.items():
        if key.endswith("_by_15_1") or key.endswith("_by_15_1_read_model"):
            if key != "source_block_m_entry_read_by_15_1_read_model":
                assert value is False


def test_read_summary_is_source_only_and_blocks_unsafe_paths() -> None:
    summary = _payload()["block_m_read_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_exe_direction_preserved_without_packaging_start() -> None:
    model = _payload()["exe_direction_read_model"]
    assert model["final_product_direction"] == "desktop_exe"
    assert model["exe_direction_preserved"] is True
    assert model["read_model_confirms_exe_direction"] is True
    for key in [
        "exe_packaging_started_now",
        "pyinstaller_started_now",
        "build_command_added_now",
        "workflow_changed_for_packaging_now",
        "installer_changed_now",
        "release_artifact_created_now",
    ]:
        assert model[key] is False
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        assert model[key] is True


def test_packaging_forbidden_read_model_blocks_build_scope() -> None:
    model = _payload()["packaging_forbidden_read_model"]
    assert model["packaging_read_model_built"] is True
    for key in [
        "exe_packaging_allowed_now",
        "pyinstaller_allowed_now",
        "build_artifact_creation_allowed_now",
        "installer_changes_allowed_now",
        "release_workflow_changes_allowed_now",
        "packaging_command_execution_allowed_now",
        "packaging_filesystem_io_allowed_now",
    ]:
        assert model[key] is False
    assert model["packaging_requires_future_explicit_gate"] is True
    assert model["packaging_requires_future_block"] is True
    assert model["packaging_not_performed_by_read_model"] is True


def test_runtime_forbidden_read_model_blocks_runtime_and_io_scope() -> None:
    model = _payload()["runtime_forbidden_read_model"]
    assert model["runtime_read_model_built"] is True
    for key in RUNTIME_FALSE_FLAGS:
        assert model[key] is False


def test_capability_rows_are_read_only_fail_closed_from_entry_matrix() -> None:
    rows = _payload()["capability_read_rows"]
    source_rows = build_preview_block_m_entry_contract()["forbidden_scope_matrix"]
    assert len(rows) == len(source_rows)
    assert {row["capability_id"] for row in rows} == {row["capability_id"] for row in source_rows}
    for row in rows:
        assert row["display_name"]
        assert row["notes"]
        assert row["read_model_row_type"] == "block_m_capability_static_read_row"
        assert row["read_only"] is True
        assert row["source_forbidden_in_15_0"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["read_model_confirms_blocked"] is True
        assert row["read_model_performed_capability"] is False


def test_fail_closed_decision_blocks_all_15_1_paths() -> None:
    decision = _payload()["fail_closed_read_model_decision"]
    assert decision["missing_block_m_entry_contract_policy"] == "fail_closed"
    assert decision["missing_capability_row_policy"] == "fail_closed"
    assert decision["missing_future_gate_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_activation_evidence_remains_false_for_unsafe_paths() -> None:
    evidence = _payload()["non_activation_evidence"]
    assert evidence["source_block_m_entry_contract_read"] is True
    assert evidence["block_m_read_model_built"] is True
    assert evidence["block_m_read_model_only"] is True
    for key in NON_ACTIVATION_FALSE_FLAGS:
        assert evidence[key] is False


def test_read_model_boundaries_are_closed() -> None:
    boundaries = _payload()["read_model_boundaries"]
    for key in BOUNDARY_TRUE_FLAGS:
        assert boundaries[key] is True
    for key, value in boundaries.items():
        if key.startswith("block_m_read_model_cannot_"):
            assert value is True


def test_source_boundaries_reference_15_0_and_forbidden_calls_absent() -> None:
    boundaries = _payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_m_entry_contract"] == "FUNCTIONAL-PREVIEW-15.0"
    for key, value in boundaries.items():
        if key.startswith("forbidden_") and key.endswith("_calls_present"):
            assert value is False
    source_subset = boundaries["source_block_m_entry_contract_boundaries"]
    assert source_subset["allowed_imports_only"] is True
    assert source_subset["source_block_l_closure"] == "FUNCTIONAL-PREVIEW-14.6"


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
