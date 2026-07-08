"""Tests for FUNCTIONAL-PREVIEW-14.3 Block L paper runtime activation gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_paper_runtime_activation_gate import (
    BLOCK_ID,
    BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_DECISION,
    BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_KIND,
    PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_4,
    STATUS,
    STEP_ID,
    build_preview_block_l_paper_runtime_activation_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_paper_runtime_activation_gate.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_paper_runtime_activation_gate_kind",
    "block",
    "step",
    "block_l_paper_runtime_activation_gate_status",
    "block_l_paper_runtime_activation_gate_decision",
    "ready_for_block_l_4",
    "next_step",
    "next_step_title",
    "gate_matrix_reference",
    "paper_runtime_gate_readiness",
    "paper_runtime_candidate_gate_checks",
    "paper_runtime_mode_decision",
    "paper_runtime_activation_blocked_summary",
    "required_future_prerequisites",
    "fail_closed_paper_gate_decision",
    "non_start_evidence",
    "paper_gate_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

READINESS_FALSE_FLAGS = [
    "safe_to_start_paper_runtime_now",
    "safe_to_start_runtime_loop_now",
    "safe_to_execute_runtime_gate_now",
    "safe_to_mutate_gate_state_now",
    "safe_for_order_generation_now",
    "safe_for_order_submission_now",
    "safe_for_private_endpoints_now",
    "safe_for_network_io_now",
    "safe_for_filesystem_io_now",
    "safe_for_config_env_secrets_now",
    "safe_for_testnet_runtime_now",
    "safe_for_live_canary_now",
    "safe_for_live_trading_now",
]

CHECK_FALSE_FLAGS = [
    "paper_gate_executed_by_14_3",
    "paper_runtime_start_allowed_now",
    "runtime_loop_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
]

MODE_FALSE_FLAGS = [
    "paper_runtime_started",
    "paper_runtime_start_allowed_now",
    "runtime_loop_started",
    "runtime_loop_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "filesystem_io_allowed_now",
    "live_trading_allowed_now",
]

BLOCKED_TRUE_FLAGS = [
    "paper_runtime_start_blocked",
    "runtime_loop_blocked",
    "runtime_gate_execution_blocked",
    "gate_state_mutation_blocked",
    "order_generation_blocked",
    "order_submission_blocked",
    "private_endpoint_access_blocked",
    "network_io_blocked",
    "filesystem_io_blocked",
    "credential_read_blocked",
    "testnet_runtime_blocked",
    "live_canary_blocked",
    "live_trading_blocked",
]

BLOCKED_DECISION_KEYS = [
    "paper_runtime_start_in_14_3",
    "runtime_loop_start_in_14_3",
    "runtime_gate_execution_in_14_3",
    "gate_state_mutation_in_14_3",
    "order_generation_in_14_3",
    "order_submission_in_14_3",
    "private_endpoint_in_14_3",
    "network_io_in_14_3",
    "filesystem_io_in_14_3",
    "credential_read_in_14_3",
    "testnet_runtime_in_14_3",
    "live_canary_in_14_3",
    "live_trading_in_14_3",
]

NON_START_FALSE_FLAGS = [
    "source_gate_matrix_runtime_activation_started",
    "source_gate_matrix_runtime_gate_executed",
    "source_gate_matrix_gate_state_mutated",
    "source_gate_matrix_mode_activated",
    "source_gate_matrix_order_generated",
    "source_gate_matrix_order_submitted",
    "source_gate_matrix_private_endpoint_accessed",
    "source_gate_matrix_network_io_performed",
    "source_gate_matrix_filesystem_io_performed",
    "source_gate_matrix_live_canary_started",
    "paper_runtime_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "gate_state_mutated",
    "mode_activated",
    "order_generated",
    "order_submitted",
    "private_endpoint_accessed",
    "network_io_performed",
    "filesystem_io_performed",
    "testnet_runtime_started",
    "live_canary_started",
    "live_trading_started",
]

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_runtime_activation_gate_matrix",
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
    "cancel_order",
    "replace_order",
}

FORBIDDEN_SOURCE_TOKENS = [
    "balance" + "_fetch",
    "cc" + "xt",
    "create_order",
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
    return build_preview_block_l_paper_runtime_activation_gate()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()

    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_14_3() -> None:
    payload = _payload()

    assert payload["schema_version"] == PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_SCHEMA_VERSION
    assert (
        payload["block_l_paper_runtime_activation_gate_kind"]
        == PREVIEW_BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_KIND
    )
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.3"
    assert (
        payload["block_l_paper_runtime_activation_gate_status"]
        == BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_STATUS
    )
    assert (
        payload["block_l_paper_runtime_activation_gate_decision"]
        == BLOCK_L_PAPER_RUNTIME_ACTIVATION_GATE_DECISION
    )
    assert payload["ready_for_block_l_4"] is READY_FOR_BLOCK_L_4 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-14.4"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "TESTNET RUNTIME ACTIVATION GATE"
    assert payload["status"] == STATUS


def test_gate_matrix_reference_points_to_safe_14_2_subset() -> None:
    reference = _payload()["gate_matrix_reference"]

    assert reference["schema_version"] == "preview_block_l_runtime_activation_gate_matrix.v1"
    assert reference["block_l_runtime_activation_gate_matrix_kind"]
    assert reference["block"] == "L"
    assert reference["step"] == "14.2"
    assert reference["ready_for_block_l_3"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.3"
    assert reference["next_step_title"] == "PAPER RUNTIME ACTIVATION GATE"
    assert reference["source_gate_matrix_step"] == "FUNCTIONAL-PREVIEW-14.2"
    assert reference["source_gate_matrix_read_by_14_3_paper_gate"] is True
    assert reference["static_gate_matrix_only"] is True
    assert reference["paper_runtime_started_by_14_3"] is False
    assert reference["runtime_loop_started_by_14_3"] is False


def test_readiness_is_ready_only_for_14_4_and_fail_closed_now() -> None:
    readiness = _payload()["paper_runtime_gate_readiness"]

    assert readiness["gate_matrix_available"] is True
    assert readiness["paper_gate_built"] is True
    assert readiness["ready_for_14_4_testnet_runtime_activation_gate"] is True
    for flag in READINESS_FALSE_FLAGS:
        assert readiness[flag] is False


def test_candidate_gate_checks_are_static_read_only_and_do_not_allow_actions() -> None:
    checks = _payload()["paper_runtime_candidate_gate_checks"]

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
        assert check["paper_gate_check_type"] == "static_read_only_candidate_gate_check"
        assert check["paper_gate_static_check_included"] is True
        assert check["paper_gate_static_check_result"] == "not_executed_static_paper_gate_only"
        for flag in CHECK_FALSE_FLAGS:
            assert check[flag] is False


def test_paper_runtime_mode_decision_finds_paper_mode_but_does_not_start_it() -> None:
    decision = _payload()["paper_runtime_mode_decision"]

    assert decision["paper_mode_found"] is True
    assert decision["source_mode_id"] == "paper_runtime_candidate"
    assert decision["mode_classification"] == "paper_runtime"
    assert decision["requires_future_gate"] is True
    assert decision["safe_for_offline_tests"] is True
    assert (
        decision["paper_mode_static_decision"]
        == "paper_runtime_candidate_identified_but_not_started"
    )
    for flag in MODE_FALSE_FLAGS:
        assert decision[flag] is False


def test_blocked_summary_contains_required_blocks() -> None:
    summary = _payload()["paper_runtime_activation_blocked_summary"]

    assert summary["blocked_capability_count"] == len(summary["blocked_capabilities"])
    assert summary["blocked_capability_count"] >= len(BLOCKED_TRUE_FLAGS)
    for flag in BLOCKED_TRUE_FLAGS:
        assert summary[flag] is True


def test_required_future_prerequisites_are_unsatisfied_and_future_only() -> None:
    prerequisites = _payload()["required_future_prerequisites"]

    assert len(prerequisites) == 8
    for prerequisite in prerequisites:
        assert prerequisite["prerequisite_id"]
        assert prerequisite["display_name"]
        assert prerequisite["required_before_paper_runtime_start"] is True
        assert prerequisite["satisfied_in_14_3"] is False
        assert prerequisite["requires_future_step"] is True
        assert prerequisite["notes"]


def test_fail_closed_paper_gate_decision_blocks_all_14_3_activation_paths() -> None:
    decision = _payload()["fail_closed_paper_gate_decision"]

    assert decision["missing_gate_matrix_policy"] == "fail_closed"
    assert decision["missing_paper_mode_policy"] == "fail_closed"
    assert decision["missing_future_prerequisite_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_start_evidence_shows_no_activation_start_or_execution() -> None:
    evidence = _payload()["non_start_evidence"]

    assert evidence["source_gate_matrix_read"] is True
    assert evidence["paper_gate_built"] is True
    for flag in NON_START_FALSE_FLAGS:
        assert evidence[flag] is False


def test_paper_gate_boundaries_are_closed() -> None:
    boundaries = _payload()["paper_gate_boundaries"]

    assert boundaries["paper_gate_is_plain_data_only"] is True
    assert boundaries["paper_gate_is_source_only"] is True
    assert boundaries["paper_gate_can_feed_14_4_testnet_runtime_activation_gate"] is True
    cannot_flags = [key for key in boundaries if key.startswith("paper_gate_cannot_")]
    assert cannot_flags
    for flag in cannot_flags:
        assert boundaries[flag] is True


def test_source_import_guard_allows_only_static_gate_matrix_imports() -> None:
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
