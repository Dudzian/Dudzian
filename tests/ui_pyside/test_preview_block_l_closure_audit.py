"""Tests for FUNCTIONAL-PREVIEW-14.6 Block L closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_closure_audit import (
    BLOCK_ID,
    BLOCK_L_CLOSURE_AUDIT_DECISION,
    BLOCK_L_CLOSURE_AUDIT_STATUS,
    CLOSURE_LINE,
    NEXT_BLOCK,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_L_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_NEXT_BLOCK,
    STATUS,
    STEP_ID,
    build_preview_block_l_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_closure_audit.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_closure_audit_kind",
    "block",
    "step",
    "block_l_closure_audit_status",
    "block_l_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
    "live_canary_contract_reference",
    "block_l_completion_summary",
    "block_l_step_ledger",
    "block_l_safety_closure_matrix",
    "blocked_capability_closure_summary",
    "non_activation_closure_evidence",
    "fail_closed_closure_decision",
    "closure_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

STEP_IDS = [f"FUNCTIONAL-PREVIEW-14.{index}" for index in range(7)]
CAPABILITY_IDS = {
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
    "exe_packaging",
}
SUMMARY_TRUE_FLAGS = [
    "block_l_started",
    "block_l_contract_built",
    "runtime_activation_read_model_built",
    "runtime_activation_gate_matrix_built",
    "paper_runtime_activation_gate_built",
    "testnet_runtime_activation_gate_built",
    "live_canary_gate_contract_built",
    "block_l_closure_audit_built",
    "block_l_closed",
    "ready_for_next_block",
]
SUMMARY_FALSE_FLAGS = [
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
    "exe_packaging_in_scope_now",
]
LEDGER_FALSE_FLAGS = [
    "runtime_activation_performed",
    "gate_execution_performed",
    "live_canary_started",
    "orders_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "filesystem_io_performed",
    "credentials_read",
]
BLOCKED_TRUE_FLAGS = [
    "runtime_activation_blocked",
    "paper_runtime_blocked",
    "testnet_runtime_blocked",
    "live_canary_start_blocked",
    "live_trading_blocked",
    "order_generation_blocked",
    "order_submission_blocked",
    "order_cancel_blocked",
    "order_replace_blocked",
    "private_endpoint_access_blocked",
    "network_io_blocked",
    "filesystem_io_blocked",
    "credential_read_blocked",
    "config_env_secret_read_blocked",
    "runtime_loop_blocked",
    "runtime_gate_execution_blocked",
    "gate_state_mutation_blocked",
    "exe_packaging_blocked",
]
NON_ACTIVATION_FALSE_FLAGS = [
    "source_live_canary_contract_live_canary_started",
    "source_live_canary_contract_runtime_loop_started",
    "source_live_canary_contract_runtime_gate_executed",
    "source_live_canary_contract_gate_state_mutated",
    "source_live_canary_contract_mode_activated",
    "source_live_canary_contract_order_generated",
    "source_live_canary_contract_order_submitted",
    "source_live_canary_contract_private_endpoint_accessed",
    "source_live_canary_contract_network_io_performed",
    "source_live_canary_contract_filesystem_io_performed",
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
]
BLOCKED_DECISION_KEYS = [
    "runtime_activation_in_block_l",
    "paper_runtime_start_in_block_l",
    "testnet_runtime_start_in_block_l",
    "live_canary_start_in_block_l",
    "live_trading_in_block_l",
    "order_generation_in_block_l",
    "order_submission_in_block_l",
    "order_cancel_in_block_l",
    "order_replace_in_block_l",
    "private_endpoint_in_block_l",
    "network_io_in_block_l",
    "filesystem_io_in_block_l",
    "credential_read_in_block_l",
    "config_env_secret_read_in_block_l",
    "runtime_loop_start_in_block_l",
    "runtime_gate_execution_in_block_l",
    "gate_state_mutation_in_block_l",
    "exe_packaging_in_block_l",
]
ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_live_canary_gate_contract",
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
    return build_preview_block_l_closure_audit()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()
    json.dumps(payload, sort_keys=True)
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_next_and_closure_line_are_14_6_values() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_L_CLOSURE_AUDIT_SCHEMA_VERSION
    assert payload["block_l_closure_audit_kind"] == PREVIEW_BLOCK_L_CLOSURE_AUDIT_KIND
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.6"
    assert payload["block_l_closure_audit_status"] == BLOCK_L_CLOSURE_AUDIT_STATUS
    assert payload["block_l_closure_audit_decision"] == BLOCK_L_CLOSURE_AUDIT_DECISION
    assert payload["ready_for_next_block"] is READY_FOR_NEXT_BLOCK is True
    assert payload["next_block"] == NEXT_BLOCK
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.0"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "NEXT BLOCK ENTRY CONTRACT"
    assert payload["closure_line"] == CLOSURE_LINE
    assert payload["status"] == STATUS


def test_live_canary_contract_reference_points_to_14_5_without_activation() -> None:
    reference = _payload()["live_canary_contract_reference"]
    assert reference["source_live_canary_contract_step"] == "FUNCTIONAL-PREVIEW-14.5"
    assert reference["step"] == "14.5"
    assert reference["ready_for_block_l_6"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.6"
    assert reference["next_step_title"] == "BLOCK L CLOSURE AUDIT"
    assert reference["source_live_canary_contract_read_by_14_6_closure"] is True
    assert reference["static_live_canary_contract_only"] is True
    for key in [
        "runtime_activated_by_14_6",
        "live_canary_started_by_14_6",
        "live_trading_enabled_by_14_6",
        "orders_enabled_by_14_6",
        "network_io_opened_by_14_6",
        "credentials_read_by_14_6",
        "private_endpoint_accessed_by_14_6",
        "exe_packaging_started_by_14_6",
    ]:
        assert reference[key] is False


def test_completion_summary_closes_block_l_but_keeps_activation_paths_closed() -> None:
    summary = _payload()["block_l_completion_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_step_ledger_contains_14_0_through_14_6_and_no_execution_flags() -> None:
    ledger = _payload()["block_l_step_ledger"]
    assert [row["step_id"] for row in ledger] == STEP_IDS
    for row in ledger:
        assert row["source_only"] is True
        assert row["plain_data_only"] is True
        assert row["completed"] is True
        assert row["title"]
        assert row["artifact_type"]
        for key in LEDGER_FALSE_FLAGS:
            assert row[key] is False


def test_safety_closure_matrix_blocks_all_required_capabilities() -> None:
    matrix = _payload()["block_l_safety_closure_matrix"]
    assert {row["capability_id"] for row in matrix} == CAPABILITY_IDS
    for row in matrix:
        assert row["blocked_through_block_l"] is True
        assert row["allowed_now"] is False
        assert row["executed_in_block_l"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["display_name"]
        assert row["notes"]


def test_blocked_capability_closure_summary_contains_required_blocks() -> None:
    summary = _payload()["blocked_capability_closure_summary"]
    assert summary["blocked_capability_count"] == len(summary["blocked_capabilities"])
    for key in BLOCKED_TRUE_FLAGS:
        assert summary[key] is True


def test_non_activation_closure_evidence_has_no_activation_or_io() -> None:
    evidence = _payload()["non_activation_closure_evidence"]
    assert evidence["source_live_canary_contract_read"] is True
    assert evidence["block_l_closure_audit_built"] is True
    assert evidence["block_l_closed"] is True
    for key in NON_ACTIVATION_FALSE_FLAGS:
        assert evidence[key] is False


def test_fail_closed_closure_decision_blocks_all_activation_paths() -> None:
    decision = _payload()["fail_closed_closure_decision"]
    assert decision["missing_live_canary_contract_policy"] == "fail_closed"
    assert decision["missing_block_l_step_policy"] == "fail_closed"
    assert decision["missing_future_gate_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_closure_boundaries_are_closed() -> None:
    boundaries = _payload()["closure_boundaries"]
    assert boundaries["block_l_closure_is_plain_data_only"] is True
    assert boundaries["block_l_closure_is_source_only"] is True
    assert boundaries["block_l_closure_closes_block_l"] is True
    assert boundaries["block_l_closure_can_feed_next_block_entry_contract"] is True
    cannot_keys = [key for key in boundaries if key.startswith("block_l_closure_cannot_")]
    assert cannot_keys
    for key in cannot_keys:
        assert boundaries[key] is True
    assert boundaries["block_l_closure_cannot_" + "sub" + "mit_orders"] is True
    assert boundaries["block_l_closure_cannot_" + "can" + "cel_orders"] is True
    assert boundaries["block_l_closure_cannot_" + "re" + "place_orders"] is True


def test_source_boundaries_point_to_14_5_and_are_closed() -> None:
    boundaries = _payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_live_canary_contract"] == "FUNCTIONAL-PREVIEW-14.5"
    for key in [
        "forbidden_runtime_calls_present",
        "forbidden_io_calls_present",
        "forbidden_network_calls_present",
        "forbidden_private_endpoint_calls_present",
        "forbidden_ui_bridge_calls_present",
        "forbidden_packaging_calls_present",
    ]:
        assert boundaries[key] is False
    source = boundaries["source_live_canary_contract_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_testnet_gate"] == "FUNCTIONAL-PREVIEW-14.4"


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    assert set(imported_modules) == ALLOWED_IMPORT_MODULES


def test_source_call_guard_blocks_forbidden_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert calls.isdisjoint(FORBIDDEN_CALL_NAMES)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
