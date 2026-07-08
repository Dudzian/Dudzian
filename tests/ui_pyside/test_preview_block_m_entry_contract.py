"""Tests for FUNCTIONAL-PREVIEW-15.0 Block M entry contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_entry_contract import (
    BLOCK_ID,
    BLOCK_M_ENTRY_CONTRACT_DECISION,
    BLOCK_M_ENTRY_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_ENTRY_CONTRACT_KIND,
    PREVIEW_BLOCK_M_ENTRY_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_1,
    STATUS,
    STEP_ID,
    build_preview_block_m_entry_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_entry_contract.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_entry_contract_kind",
    "block",
    "step",
    "block_m_entry_contract_status",
    "block_m_entry_contract_decision",
    "ready_for_block_m_1",
    "next_step",
    "next_step_title",
    "block_l_closure_reference",
    "block_m_entry_summary",
    "block_m_scope_contract",
    "exe_direction_preservation_contract",
    "forbidden_scope_matrix",
    "fail_closed_entry_decision",
    "non_activation_evidence",
    "entry_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

SUMMARY_TRUE_FLAGS = [
    "block_l_closure_available",
    "block_l_closed",
    "block_m_entry_contract_built",
    "block_m_opened",
    "ready_for_block_m_1",
    "exe_direction_preserved",
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
]
SCOPE_TRUE_FLAGS = [
    "block_m_scope_is_entry_contract_only",
    "block_m_scope_is_source_only",
    "block_m_scope_is_plain_data_only",
    "block_m_scope_preserves_exe_direction",
    "block_m_scope_does_not_package_exe",
    "block_m_scope_does_not_change_build_system",
    "block_m_scope_does_not_change_runtime",
    "block_m_scope_does_not_change_ui_bridge",
    "block_m_scope_does_not_touch_credentials",
    "block_m_scope_does_not_touch_network",
    "block_m_scope_does_not_touch_private_endpoints",
    "block_m_scope_does_not_touch_orders",
    "block_m_scope_does_not_touch_filesystem",
]
CAPABILITY_IDS = {
    "exe_packaging",
    "pyinstaller",
    "build_artifact_creation",
    "release_workflow_changes",
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "gate_execution",
    "gate_mutation",
    "order_generation",
    "order_submission",
    "order_cancel",
    "order_replace",
    "private_endpoints",
    "network_io",
    "filesystem_io",
    "credentials",
    "config_env_secrets",
    "qml_bridge_changes",
}
BLOCKED_DECISION_KEYS = [
    "exe_packaging_in_15_0",
    "pyinstaller_in_15_0",
    "build_artifact_creation_in_15_0",
    "release_workflow_change_in_15_0",
    "runtime_activation_in_15_0",
    "paper_runtime_start_in_15_0",
    "testnet_runtime_start_in_15_0",
    "live_canary_start_in_15_0",
    "live_trading_in_15_0",
    "order_generation_in_15_0",
    "order_submission_in_15_0",
    "order_cancel_in_15_0",
    "order_replace_in_15_0",
    "private_endpoint_in_15_0",
    "network_io_in_15_0",
    "filesystem_io_in_15_0",
    "credential_read_in_15_0",
    "config_env_secret_read_in_15_0",
    "qml_bridge_change_in_15_0",
]
NON_ACTIVATION_FALSE_FLAGS = [
    "source_block_l_closure_live_canary_started",
    "source_block_l_closure_runtime_loop_started",
    "source_block_l_closure_runtime_gate_executed",
    "source_block_l_closure_gate_state_mutated",
    "source_block_l_closure_mode_activated",
    "source_block_l_closure_order_generated",
    "source_block_l_closure_order_submitted",
    "source_block_l_closure_private_endpoint_accessed",
    "source_block_l_closure_network_io_performed",
    "source_block_l_closure_filesystem_io_performed",
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
ENTRY_TRUE_FLAGS = [
    "block_m_entry_is_plain_data_only",
    "block_m_entry_is_source_only",
    "block_m_entry_opens_block_m",
    "block_m_entry_preserves_exe_direction_without_packaging",
    "block_m_entry_can_feed_15_1_read_model",
    "block_m_entry_cannot_package_exe",
    "block_m_entry_cannot_start_pyinstaller",
    "block_m_entry_cannot_create_build_artifacts",
    "block_m_entry_cannot_change_release_workflows",
    "block_m_entry_cannot_activate_runtime",
    "block_m_entry_cannot_start_paper_runtime",
    "block_m_entry_cannot_start_testnet_runtime",
    "block_m_entry_cannot_start_live_canary",
    "block_m_entry_cannot_enable_live_trading",
    "block_m_entry_cannot_generate_orders",
    "block_m_entry_cannot_" + "sub" + "mit_orders",
    "block_m_entry_cannot_" + "can" + "cel_orders",
    "block_m_entry_cannot_" + "re" + "place_orders",
    "block_m_entry_cannot_access_private_endpoints",
    "block_m_entry_cannot_open_network_io",
    "block_m_entry_cannot_read_credentials",
    "block_m_entry_cannot_start_runtime_loop",
    "block_m_entry_cannot_execute_runtime_gates",
    "block_m_entry_cannot_mutate_gate_state",
    "block_m_entry_cannot_perform_filesystem_io",
    "block_m_entry_cannot_read_config_env_or_secrets",
    "block_m_entry_cannot_change_ui_bridge",
]
ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_closure_audit",
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
    "dumps",
    "dump",
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
    "create_order",
    "submit_order",
    "cancel_order",
    "replace_order",
}
FORBIDDEN_SOURCE_TOKENS = [
    "balance" + "_fetch",
    "cc" + "xt",
    "create_order",
    "submit_order",
    "cancel_order",
    "replace_order",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_m_entry_contract()


def test_payload_is_json_serializable() -> None:
    json.dumps(_payload(), sort_keys=True)


def test_top_level_fields_are_exact_and_stable() -> None:
    assert list(_payload()) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_match_15_0() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_M_ENTRY_CONTRACT_SCHEMA_VERSION
    assert payload["block_m_entry_contract_kind"] == PREVIEW_BLOCK_M_ENTRY_CONTRACT_KIND
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.0"
    assert payload["block_m_entry_contract_status"] == BLOCK_M_ENTRY_CONTRACT_STATUS
    assert payload["block_m_entry_contract_decision"] == BLOCK_M_ENTRY_CONTRACT_DECISION
    assert payload["ready_for_block_m_1"] is READY_FOR_BLOCK_M_1 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.1"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "BLOCK M READ MODEL"
    assert payload["status"] == STATUS


def test_block_l_closure_reference_points_to_14_6_and_closure_line() -> None:
    reference = _payload()["block_l_closure_reference"]
    assert reference["schema_version"] == "preview_block_l_closure_audit.v1"
    assert reference["block_l_closure_audit_kind"] == "functional_preview_block_l_closure_audit"
    assert reference["block"] == "L"
    assert reference["step"] == "14.6"
    assert reference["ready_for_next_block"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-15.0"
    assert reference["next_step_title"] == "NEXT BLOCK ENTRY CONTRACT"
    assert reference["closure_line"] == "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"
    assert reference["source_block_l_closure_step"] == "FUNCTIONAL-PREVIEW-14.6"
    assert reference["source_block_l_closure_read_by_15_0_entry_contract"] is True
    assert reference["block_l_closed_before_block_m_entry"] is True
    assert reference["static_closure_audit_only"] is True
    for key in [
        "runtime_activated_by_15_0",
        "live_canary_started_by_15_0",
        "live_trading_enabled_by_15_0",
        "orders_enabled_by_15_0",
        "network_io_opened_by_15_0",
        "credentials_read_by_15_0",
        "private_endpoint_accessed_by_15_0",
        "filesystem_io_performed_by_15_0",
        "exe_packaging_started_by_15_0",
    ]:
        assert reference[key] is False


def test_block_m_entry_summary_opens_block_m_and_blocks_unsafe_paths() -> None:
    summary = _payload()["block_m_entry_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_block_m_scope_contract_is_source_only_and_plain_data() -> None:
    scope = _payload()["block_m_scope_contract"]
    for key in SCOPE_TRUE_FLAGS:
        assert scope[key] is True


def test_exe_direction_preservation_contract_does_not_start_packaging() -> None:
    contract = _payload()["exe_direction_preservation_contract"]
    assert contract["final_product_direction"] == "desktop_exe"
    assert contract["exe_direction_preserved"] is True
    for key in [
        "exe_packaging_started_now",
        "pyinstaller_started_now",
        "build_command_added_now",
        "workflow_changed_for_packaging_now",
        "installer_changed_now",
        "release_artifact_created_now",
    ]:
        assert contract[key] is False
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        assert contract[key] is True


def test_forbidden_scope_matrix_blocks_required_capabilities() -> None:
    matrix = _payload()["forbidden_scope_matrix"]
    assert {row["capability_id"] for row in matrix} == CAPABILITY_IDS
    for row in matrix:
        assert row["forbidden_in_15_0"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["notes"]


def test_fail_closed_entry_decision_blocks_all_forbidden_paths() -> None:
    decision = _payload()["fail_closed_entry_decision"]
    assert decision["missing_block_l_closure_policy"] == "fail_closed"
    assert decision["missing_block_m_scope_policy"] == "fail_closed"
    assert decision["missing_future_gate_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_activation_evidence_has_no_activation_packaging_io_or_bridge() -> None:
    evidence = _payload()["non_activation_evidence"]
    assert evidence["source_block_l_closure_read"] is True
    assert evidence["block_m_entry_contract_built"] is True
    assert evidence["block_m_opened"] is True
    for key in NON_ACTIVATION_FALSE_FLAGS:
        assert evidence[key] is False


def test_entry_boundaries_are_closed() -> None:
    boundaries = _payload()["entry_boundaries"]
    for key in ENTRY_TRUE_FLAGS:
        assert boundaries[key] is True


def test_source_boundaries_reference_14_6_and_forbidden_calls_absent() -> None:
    boundaries = _payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_l_closure"] == "FUNCTIONAL-PREVIEW-14.6"
    for key in [
        "forbidden_packaging_calls_present",
        "forbidden_pyinstaller_calls_present",
        "forbidden_build_calls_present",
        "forbidden_runtime_calls_present",
        "forbidden_io_calls_present",
        "forbidden_network_calls_present",
        "forbidden_private_endpoint_calls_present",
        "forbidden_ui_bridge_calls_present",
    ]:
        assert boundaries[key] is False
    assert boundaries["source_block_l_closure_boundaries"]["allowed_imports_only"] is True


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module or "__future__")
    assert imports == ALLOWED_IMPORT_MODULES


def test_source_call_guard_blocks_io_network_runtime_orders_private_config_and_packaging() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called_names.add(function.id)
            if isinstance(function, ast.Attribute):
                called_names.add(function.attr)
    assert not (called_names & FORBIDDEN_CALL_NAMES)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
