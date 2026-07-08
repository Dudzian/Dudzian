"""Tests for FUNCTIONAL-PREVIEW-14.4 Block L testnet runtime activation gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_testnet_runtime_activation_gate import (
    BLOCK_ID,
    BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_DECISION,
    BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_KIND,
    PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_5,
    STATUS,
    STEP_ID,
    build_preview_block_l_testnet_runtime_activation_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_testnet_runtime_activation_gate.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_testnet_runtime_activation_gate_kind",
    "block",
    "step",
    "block_l_testnet_runtime_activation_gate_status",
    "block_l_testnet_runtime_activation_gate_decision",
    "ready_for_block_l_5",
    "next_step",
    "next_step_title",
    "paper_gate_reference",
    "testnet_runtime_gate_readiness",
    "testnet_candidate_static_checks",
    "testnet_runtime_mode_decision",
    "testnet_activation_blocked_summary",
    "required_future_prerequisites",
    "fail_closed_testnet_gate_decision",
    "non_start_evidence",
    "testnet_gate_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

READINESS_FALSE_FLAGS = [
    "safe_to_start_testnet_runtime_now",
    "safe_to_open_network_io_now",
    "safe_to_read_credentials_now",
    "safe_to_access_private_endpoints_now",
    "safe_to_start_runtime_loop_now",
    "safe_to_execute_runtime_gate_now",
    "safe_to_mutate_gate_state_now",
    "safe_for_order_generation_now",
    "safe_for_order_submission_now",
    "safe_for_filesystem_io_now",
    "safe_for_config_env_secrets_now",
    "safe_for_live_canary_now",
    "safe_for_live_trading_now",
]

CHECK_FALSE_FLAGS = [
    "testnet_gate_executed_by_14_4",
    "testnet_runtime_start_allowed_now",
    "runtime_loop_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "filesystem_io_allowed_now",
]

MODE_FALSE_FLAGS = [
    "testnet_runtime_started",
    "testnet_runtime_start_allowed_now",
    "runtime_loop_started",
    "runtime_loop_allowed_now",
    "network_io_opened",
    "network_io_allowed_now",
    "credential_read_performed",
    "credential_read_allowed_now",
    "private_endpoint_accessed",
    "private_endpoint_access_allowed_now",
    "order_flow_allowed_now",
    "filesystem_io_allowed_now",
    "live_trading_allowed_now",
]

BLOCKED_TRUE_FLAGS = [
    "testnet_runtime_start_blocked",
    "network_io_blocked",
    "credential_read_blocked",
    "private_endpoint_access_blocked",
    "runtime_loop_blocked",
    "runtime_gate_execution_blocked",
    "gate_state_mutation_blocked",
    "order_generation_blocked",
    "order_submission_blocked",
    "filesystem_io_blocked",
    "config_env_secret_read_blocked",
    "live_canary_blocked",
    "live_trading_blocked",
]

BLOCKED_DECISION_KEYS = [
    "testnet_runtime_start_in_14_4",
    "network_io_in_14_4",
    "credential_read_in_14_4",
    "private_endpoint_in_14_4",
    "runtime_loop_start_in_14_4",
    "runtime_gate_execution_in_14_4",
    "gate_state_mutation_in_14_4",
    "order_generation_in_14_4",
    "order_submission_in_14_4",
    "filesystem_io_in_14_4",
    "config_env_secret_read_in_14_4",
    "live_canary_in_14_4",
    "live_trading_in_14_4",
]

NON_START_FALSE_FLAGS = [
    "source_paper_gate_paper_runtime_started",
    "source_paper_gate_runtime_loop_started",
    "source_paper_gate_runtime_gate_executed",
    "source_paper_gate_gate_state_mutated",
    "source_paper_gate_mode_activated",
    "source_paper_gate_order_generated",
    "source_paper_gate_order_submitted",
    "source_paper_gate_private_endpoint_accessed",
    "source_paper_gate_network_io_performed",
    "source_paper_gate_filesystem_io_performed",
    "source_paper_gate_live_canary_started",
    "paper_runtime_started",
    "testnet_runtime_started",
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
    "live_canary_started",
    "live_trading_started",
]

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_paper_runtime_activation_gate",
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
    return build_preview_block_l_testnet_runtime_activation_gate()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()

    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_14_4() -> None:
    payload = _payload()

    assert (
        payload["schema_version"] == PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION
    )
    assert (
        payload["block_l_testnet_runtime_activation_gate_kind"]
        == PREVIEW_BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_KIND
    )
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.4"
    assert (
        payload["block_l_testnet_runtime_activation_gate_status"]
        == BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_STATUS
    )
    assert (
        payload["block_l_testnet_runtime_activation_gate_decision"]
        == BLOCK_L_TESTNET_RUNTIME_ACTIVATION_GATE_DECISION
    )
    assert payload["ready_for_block_l_5"] is READY_FOR_BLOCK_L_5 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-14.5"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "LIVE CANARY GATE CONTRACT"
    assert payload["status"] == STATUS


def test_paper_gate_reference_points_to_safe_14_3_subset() -> None:
    reference = _payload()["paper_gate_reference"]

    assert reference["schema_version"] == "preview_block_l_paper_runtime_activation_gate.v1"
    assert reference["block_l_paper_runtime_activation_gate_kind"]
    assert reference["block"] == "L"
    assert reference["step"] == "14.3"
    assert reference["ready_for_block_l_4"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.4"
    assert reference["next_step_title"] == "TESTNET RUNTIME ACTIVATION GATE"
    assert reference["source_paper_gate_step"] == "FUNCTIONAL-PREVIEW-14.3"
    assert reference["source_paper_gate_read_by_14_4_testnet_gate"] is True
    assert reference["static_paper_gate_only"] is True
    assert reference["testnet_runtime_started_by_14_4"] is False
    assert reference["network_io_opened_by_14_4"] is False
    assert reference["credentials_read_by_14_4"] is False
    assert reference["private_endpoint_accessed_by_14_4"] is False


def test_readiness_is_ready_only_for_14_5_and_fail_closed_now() -> None:
    readiness = _payload()["testnet_runtime_gate_readiness"]

    assert readiness["paper_gate_available"] is True
    assert readiness["testnet_gate_built"] is True
    assert readiness["ready_for_14_5_live_canary_gate_contract"] is True
    for flag in READINESS_FALSE_FLAGS:
        assert readiness[flag] is False


def test_candidate_static_checks_are_static_read_only_and_do_not_allow_actions() -> None:
    checks = _payload()["testnet_candidate_static_checks"]

    assert checks
    for check in checks:
        assert check["runtime_activation_gate_id"]
        assert check["source_gate_id"]
        assert check["display_name"]
        assert check["gate_domain"]
        assert check["gate_type"]
        assert check["planned_source_step"]
        assert check["required_for_future_activation"] is True
        assert check["safe_for_offline_tests"] is True
        assert check["testnet_gate_check_type"] == "static_read_only_candidate_gate_check"
        assert check["testnet_gate_static_check_included"] is True
        assert check["testnet_gate_static_check_result"] == "not_executed_static_testnet_gate_only"
        for flag in CHECK_FALSE_FLAGS:
            assert check[flag] is False


def test_testnet_runtime_mode_decision_is_static_fail_closed_and_not_started() -> None:
    decision = _payload()["testnet_runtime_mode_decision"]

    assert decision["source_paper_gate_step"] == "FUNCTIONAL-PREVIEW-14.3"
    assert decision["source_paper_mode_static_decision"]
    assert decision["source_required_future_prerequisite_count"] == 10
    assert (
        decision["testnet_mode_static_decision"]
        == "testnet_runtime_candidate_requires_future_static_gate_not_started"
    )
    assert decision["testnet_mode_discovered_from_static_chain"] is True
    assert decision["testnet_mode_requires_future_gate"] is True
    for flag in MODE_FALSE_FLAGS:
        assert decision[flag] is False


def test_blocked_summary_contains_required_blocks() -> None:
    summary = _payload()["testnet_activation_blocked_summary"]

    assert summary["blocked_capability_count"] == len(summary["blocked_capabilities"])
    assert summary["blocked_capability_count"] >= len(BLOCKED_TRUE_FLAGS)
    for flag in BLOCKED_TRUE_FLAGS:
        assert summary[flag] is True


def test_required_future_prerequisites_are_unsatisfied_and_future_only() -> None:
    prerequisites = _payload()["required_future_prerequisites"]

    assert len(prerequisites) == 10
    for prerequisite in prerequisites:
        assert prerequisite["prerequisite_id"]
        assert prerequisite["display_name"]
        assert prerequisite["required_before_testnet_runtime_start"] is True
        assert prerequisite["satisfied_in_14_4"] is False
        assert prerequisite["requires_future_step"] is True
        assert prerequisite["notes"]


def test_fail_closed_testnet_gate_decision_blocks_all_14_4_activation_paths() -> None:
    decision = _payload()["fail_closed_testnet_gate_decision"]

    assert decision["missing_paper_gate_policy"] == "fail_closed"
    assert decision["missing_testnet_mode_policy"] == "fail_closed"
    assert decision["missing_testnet_credentials_policy"] == "fail_closed"
    assert decision["missing_network_gate_policy"] == "fail_closed"
    assert decision["missing_private_endpoint_gate_policy"] == "fail_closed"
    assert decision["missing_future_prerequisite_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_start_evidence_shows_no_activation_start_execution_or_io() -> None:
    evidence = _payload()["non_start_evidence"]

    assert evidence["source_paper_gate_read"] is True
    assert evidence["testnet_gate_built"] is True
    for flag in NON_START_FALSE_FLAGS:
        assert evidence[flag] is False


def test_testnet_gate_boundaries_are_closed() -> None:
    boundaries = _payload()["testnet_gate_boundaries"]

    assert boundaries["testnet_gate_is_plain_data_only"] is True
    assert boundaries["testnet_gate_is_source_only"] is True
    assert boundaries["testnet_gate_can_feed_14_5_live_canary_gate_contract"] is True
    cannot_flags = [key for key in boundaries if key.startswith("testnet_gate_cannot_")]
    assert cannot_flags
    for flag in cannot_flags:
        assert boundaries[flag] is True


def test_source_import_guard_allows_only_static_paper_gate_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    assert imports == ALLOWED_IMPORT_MODULES


def test_source_call_guard_blocks_io_network_runtime_and_ui_bridge_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)

    assert not (call_names & FORBIDDEN_CALL_NAMES)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
