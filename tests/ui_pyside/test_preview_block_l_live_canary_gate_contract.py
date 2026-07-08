"""Tests for FUNCTIONAL-PREVIEW-14.5 Block L live canary gate contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_live_canary_gate_contract import (
    BLOCK_ID,
    BLOCK_L_LIVE_CANARY_GATE_CONTRACT_DECISION,
    BLOCK_L_LIVE_CANARY_GATE_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_KIND,
    PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_6,
    STATUS,
    STEP_ID,
    build_preview_block_l_live_canary_gate_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_live_canary_gate_contract.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_live_canary_gate_contract_kind",
    "block",
    "step",
    "block_l_live_canary_gate_contract_status",
    "block_l_live_canary_gate_contract_decision",
    "ready_for_block_l_6",
    "next_step",
    "next_step_title",
    "testnet_gate_reference",
    "live_canary_gate_contract_readiness",
    "live_canary_candidate_static_checks",
    "live_canary_mode_decision",
    "live_canary_blocked_summary",
    "required_future_prerequisites",
    "fail_closed_live_canary_decision",
    "non_start_evidence",
    "live_canary_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

READINESS_FALSE_FLAGS = [
    "safe_to_start_live_canary_now",
    "safe_to_enable_live_trading_now",
    "safe_to_generate_orders_now",
    "safe_to_" + "sub" + "mit_orders_now",
    "safe_to_" + "can" + "cel_orders_now",
    "safe_to_" + "re" + "place_orders_now",
    "safe_to_access_private_endpoints_now",
    "safe_to_open_network_io_now",
    "safe_to_read_credentials_now",
    "safe_to_start_runtime_loop_now",
    "safe_to_execute_runtime_gate_now",
    "safe_to_mutate_gate_state_now",
    "safe_for_filesystem_io_now",
    "safe_for_config_env_secrets_now",
]

CHECK_FALSE_FLAGS = [
    "live_canary_gate_executed_by_14_5",
    "live_canary_start_allowed_now",
    "live_trading_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "runtime_loop_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "filesystem_io_allowed_now",
]

MODE_FALSE_FLAGS = [
    "live_canary_started",
    "live_canary_start_allowed_now",
    "live_trading_enabled",
    "live_trading_allowed_now",
    "runtime_loop_started",
    "runtime_loop_allowed_now",
    "network_io_opened",
    "network_io_allowed_now",
    "credential_read_performed",
    "credential_read_allowed_now",
    "private_endpoint_accessed",
    "private_endpoint_access_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "filesystem_io_allowed_now",
]

BLOCKED_TRUE_FLAGS = [
    "live_canary_start_blocked",
    "live_trading_blocked",
    "order_generation_blocked",
    "order_submission_blocked",
    "order_cancel_blocked",
    "order_replace_blocked",
    "private_endpoint_access_blocked",
    "network_io_blocked",
    "credential_read_blocked",
    "runtime_loop_blocked",
    "runtime_gate_execution_blocked",
    "gate_state_mutation_blocked",
    "filesystem_io_blocked",
    "config_env_secret_read_blocked",
]

BLOCKED_DECISION_KEYS = [
    "live_canary_start_in_14_5",
    "live_trading_in_14_5",
    "order_generation_in_14_5",
    "order_submission_in_14_5",
    "order_cancel_in_14_5",
    "order_replace_in_14_5",
    "private_endpoint_in_14_5",
    "network_io_in_14_5",
    "credential_read_in_14_5",
    "runtime_loop_start_in_14_5",
    "runtime_gate_execution_in_14_5",
    "gate_state_mutation_in_14_5",
    "filesystem_io_in_14_5",
    "config_env_secret_read_in_14_5",
]

NON_START_FALSE_FLAGS = [
    "source_testnet_gate_testnet_runtime_started",
    "source_testnet_gate_runtime_loop_started",
    "source_testnet_gate_runtime_gate_executed",
    "source_testnet_gate_gate_state_mutated",
    "source_testnet_gate_mode_activated",
    "source_testnet_gate_order_generated",
    "source_testnet_gate_order_submitted",
    "source_testnet_gate_private_endpoint_accessed",
    "source_testnet_gate_network_io_performed",
    "source_testnet_gate_filesystem_io_performed",
    "source_testnet_gate_live_canary_started",
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
]

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_testnet_runtime_activation_gate",
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
    "PySide",
    "QML",
    "TradingController",
    "DecisionEnvelope",
    "requests",
    "subprocess",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_l_live_canary_gate_contract()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()
    json.dumps(payload, sort_keys=True)
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_14_5_values() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_SCHEMA_VERSION
    assert (
        payload["block_l_live_canary_gate_contract_kind"]
        == PREVIEW_BLOCK_L_LIVE_CANARY_GATE_CONTRACT_KIND
    )
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.5"
    assert (
        payload["block_l_live_canary_gate_contract_status"]
        == BLOCK_L_LIVE_CANARY_GATE_CONTRACT_STATUS
    )
    assert (
        payload["block_l_live_canary_gate_contract_decision"]
        == BLOCK_L_LIVE_CANARY_GATE_CONTRACT_DECISION
    )
    assert payload["ready_for_block_l_6"] is READY_FOR_BLOCK_L_6 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-14.6"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "BLOCK L CLOSURE AUDIT"
    assert payload["status"] == STATUS


def test_testnet_gate_reference_points_to_14_4_without_activation() -> None:
    reference = _payload()["testnet_gate_reference"]
    assert reference["source_testnet_gate_step"] == "FUNCTIONAL-PREVIEW-14.4"
    assert reference["step"] == "14.4"
    assert reference["ready_for_block_l_5"] is True
    assert reference["source_testnet_gate_read_by_14_5_live_canary_contract"] is True
    assert reference["static_testnet_gate_only"] is True
    for key in [
        "live_canary_started_by_14_5",
        "live_trading_enabled_by_14_5",
        "orders_enabled_by_14_5",
        "network_io_opened_by_14_5",
        "credentials_read_by_14_5",
        "private_endpoint_accessed_by_14_5",
    ]:
        assert reference[key] is False


def test_readiness_only_allows_14_6_static_audit() -> None:
    readiness = _payload()["live_canary_gate_contract_readiness"]
    assert readiness["testnet_gate_available"] is True
    assert readiness["live_canary_contract_built"] is True
    assert readiness["ready_for_14_6_block_l_closure_audit"] is True
    for key in READINESS_FALSE_FLAGS:
        assert readiness[key] is False


def test_candidate_static_checks_are_read_only_and_not_executed() -> None:
    checks = _payload()["live_canary_candidate_static_checks"]
    assert checks
    for check in checks:
        assert check["live_canary_check_type"] == "static_read_only_candidate_gate_check"
        assert check["live_canary_static_check_included"] is True
        assert (
            check["live_canary_static_check_result"]
            == "not_executed_static_live_canary_contract_only"
        )
        assert check["safe_for_offline_tests"] is True
        assert check["required_for_future_activation"] is True
        for key in CHECK_FALSE_FLAGS:
            assert check[key] is False


def test_mode_decision_is_static_and_fail_closed() -> None:
    mode = _payload()["live_canary_mode_decision"]
    assert (
        mode["live_canary_mode_static_decision"]
        == "live_canary_candidate_requires_future_static_gate_not_started"
    )
    assert mode["live_canary_mode_discovered_from_static_chain"] is True
    assert mode["live_canary_mode_requires_future_gate"] is True
    for key in MODE_FALSE_FLAGS:
        assert mode[key] is False


def test_blocked_summary_contains_required_blocks() -> None:
    summary = _payload()["live_canary_blocked_summary"]
    assert summary["blocked_capability_count"] == len(summary["blocked_capabilities"])
    for key in BLOCKED_TRUE_FLAGS:
        assert summary[key] is True


def test_future_prerequisites_are_unsatisfied_in_14_5() -> None:
    prerequisites = _payload()["required_future_prerequisites"]
    assert len(prerequisites) >= 12
    for prerequisite in prerequisites:
        assert prerequisite["required_before_live_canary_start"] is True
        assert prerequisite["satisfied_in_14_5"] is False
        assert prerequisite["requires_future_step"] is True
        assert prerequisite["prerequisite_id"]
        assert prerequisite["display_name"]
        assert prerequisite["notes"]


def test_fail_closed_decision_blocks_all_activation_paths() -> None:
    decision = _payload()["fail_closed_live_canary_decision"]
    policy_keys = [key for key in decision if key.endswith("_policy")]
    assert policy_keys
    for key in policy_keys:
        assert decision[key] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_start_evidence_has_no_activation_or_io() -> None:
    evidence = _payload()["non_start_evidence"]
    assert evidence["source_testnet_gate_read"] is True
    assert evidence["live_canary_contract_built"] is True
    for key in NON_START_FALSE_FLAGS:
        assert evidence[key] is False


def test_live_canary_boundaries_are_closed() -> None:
    boundaries = _payload()["live_canary_boundaries"]
    assert boundaries["live_canary_contract_is_plain_data_only"] is True
    assert boundaries["live_canary_contract_is_source_only"] is True
    assert boundaries["live_canary_contract_can_feed_14_6_block_l_closure_audit"] is True
    cannot_keys = [key for key in boundaries if key.startswith("live_canary_contract_cannot_")]
    assert cannot_keys
    for key in cannot_keys:
        assert boundaries[key] is True
    assert boundaries["live_canary_contract_cannot_" + "sub" + "mit_orders"] is True
    assert boundaries["live_canary_contract_cannot_" + "can" + "cel_orders"] is True
    assert boundaries["live_canary_contract_cannot_" + "re" + "place_orders"] is True


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
