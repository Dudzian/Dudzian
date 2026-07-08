"""Tests for FUNCTIONAL-PREVIEW-14.2 Block L runtime activation gate matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_runtime_activation_gate_matrix import (
    BLOCK_ID,
    BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_DECISION,
    BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_KIND,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_3,
    STATUS,
    STEP_ID,
    build_preview_block_l_runtime_activation_gate_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_runtime_activation_gate_matrix.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_runtime_activation_gate_matrix_kind",
    "block",
    "step",
    "block_l_runtime_activation_gate_matrix_status",
    "block_l_runtime_activation_gate_matrix_decision",
    "ready_for_block_l_3",
    "next_step",
    "next_step_title",
    "read_model_reference",
    "runtime_activation_gate_matrix_readiness",
    "runtime_activation_gate_matrix_rows",
    "runtime_activation_mode_matrix_rows",
    "runtime_activation_gate_to_mode_matrix",
    "blocked_capability_gate_matrix",
    "fail_closed_decision_matrix",
    "non_execution_evidence",
    "gate_matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

FALSE_READINESS_FLAGS = [
    "safe_to_execute_gates_now",
    "safe_to_mutate_gate_state_now",
    "safe_to_activate_runtime_now",
    "safe_for_paper_runtime_now",
    "safe_for_testnet_runtime_now",
    "safe_for_live_canary_now",
    "safe_for_orders_now",
    "safe_for_private_endpoints_now",
    "safe_for_network_io_now",
    "safe_for_filesystem_io_now",
    "safe_for_credentials_now",
    "safe_for_live_trading_now",
]

GATE_FALSE_FLAGS = [
    "gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "runtime_activation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
]

MODE_FALSE_FLAGS = [
    "activation_allowed_now",
    "runtime_activation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "live_trading_allowed_now",
]

REQUIRED_BLOCKS = [
    "runtime activation",
    "runtime gate execution",
    "gate state mutation",
    "live canary",
    "testnet runtime",
    "paper runtime activation",
    "order generation",
    "order submission",
    "private endpoint access",
    "filesystem I/O",
    "network I/O",
    "credential read",
]

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_runtime_activation_read_model",
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
    "submit_order",
    "cancel_order",
    "replace_order",
}

FORBIDDEN_SOURCE_TOKENS = [
    "balance" + "_fetch",
    "cc" + "xt",
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
    return build_preview_block_l_runtime_activation_gate_matrix()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()

    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_14_2() -> None:
    payload = _payload()

    assert (
        payload["schema_version"] == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_SCHEMA_VERSION
    )
    assert (
        payload["block_l_runtime_activation_gate_matrix_kind"]
        == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_KIND
    )
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.2"
    assert (
        payload["block_l_runtime_activation_gate_matrix_status"]
        == BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_STATUS
    )
    assert (
        payload["block_l_runtime_activation_gate_matrix_decision"]
        == BLOCK_L_RUNTIME_ACTIVATION_GATE_MATRIX_DECISION
    )
    assert payload["ready_for_block_l_3"] is READY_FOR_BLOCK_L_3 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-14.3"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "PAPER RUNTIME ACTIVATION GATE"
    assert payload["status"] == STATUS


def test_read_model_reference_points_to_safe_14_1_subset() -> None:
    reference = _payload()["read_model_reference"]

    assert reference["schema_version"] == "preview_block_l_runtime_activation_read_model.v1"
    assert (
        reference["block_l_runtime_activation_read_model_kind"]
        == "functional_preview_block_l_runtime_activation_read_model"
    )
    assert reference["block"] == "L"
    assert reference["step"] == "14.1"
    assert reference["ready_for_block_l_2"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.2"
    assert reference["next_step_title"] == "RUNTIME ACTIVATION GATE MATRIX"
    assert reference["source_read_model_step"] == "FUNCTIONAL-PREVIEW-14.1"
    assert reference["source_read_model_read_by_14_2_gate_matrix"] is True
    assert reference["static_read_model_only"] is True
    assert reference["gate_execution_performed_by_14_2"] is False


def test_readiness_is_ready_only_for_14_3_and_fail_closed_now() -> None:
    readiness = _payload()["runtime_activation_gate_matrix_readiness"]

    assert readiness["read_model_available"] is True
    assert readiness["gate_matrix_built"] is True
    assert readiness["ready_for_14_3_paper_runtime_activation_gate"] is True
    for flag in FALSE_READINESS_FLAGS:
        assert readiness[flag] is False


def test_gate_matrix_rows_are_read_only_matrix_only_and_fail_closed() -> None:
    rows = _payload()["runtime_activation_gate_matrix_rows"]

    assert len(rows) == 8
    for row in rows:
        assert row["read_only"] is True
        assert row["matrix_row_type"] == "runtime_activation_gate_static_matrix_row"
        assert row["matrix_included"] is True
        assert row["matrix_evaluated_by_14_2"] is True
        assert row["runtime_activation_gate_id"].startswith("runtime_activation_gate_")
        assert row["source_gate_id"]
        assert row["display_name"]
        assert row["gate_domain"]
        assert row["gate_type"]
        assert row["planned_source_step"].startswith("FUNCTIONAL-PREVIEW-")
        assert row["required_for_future_activation"] is True
        assert row["eligible_for_future_gate_matrix"] is True
        assert row["safe_for_offline_tests"] is True
        assert row["gate_execution_result"] == "not_executed_static_matrix_only"
        for flag in GATE_FALSE_FLAGS:
            assert row[flag] is False


def test_mode_matrix_rows_are_read_only_matrix_only_and_fail_closed() -> None:
    rows = _payload()["runtime_activation_mode_matrix_rows"]

    assert len(rows) == 6
    for row in rows:
        assert row["read_only"] is True
        assert row["matrix_row_type"] == "runtime_activation_mode_static_matrix_row"
        assert row["matrix_included"] is True
        assert row["matrix_evaluated_by_14_2"] is True
        assert row["runtime_activation_mode_id"].startswith("runtime_activation_mode_")
        assert row["source_mode_id"]
        assert row["display_name"]
        assert row["mode_classification"]
        assert row["activation_stage"]
        assert row["requires_future_gate"] is True
        assert row["safe_for_offline_tests"] is True
        assert row["activation_result"] == "not_activated_static_matrix_only"
        for flag in MODE_FALSE_FLAGS:
            assert row[flag] is False


def test_gate_to_mode_matrix_contains_static_future_gate_requirements() -> None:
    matrix = _payload()["runtime_activation_gate_to_mode_matrix"]

    assert len(matrix["gate_ids"]) == 8
    assert len(matrix["mode_ids"]) == 6
    assert matrix["all_modes_require_future_gate"] is True
    assert matrix["all_gates_not_executed_now"] is True
    assert matrix["all_modes_not_activated_now"] is True
    assert matrix["paper_mode_requires_future_gate"] is True
    assert matrix["testnet_mode_requires_future_gate"] is True
    assert matrix["live_canary_mode_requires_future_gate"] is True
    assert matrix["live_scaled_mode_requires_future_gate"] is True
    assert set(matrix["gate_domains_by_id"]) == set(matrix["gate_ids"])
    assert set(matrix["mode_classifications_by_id"]) == set(matrix["mode_ids"])
    assert set(matrix["mode_activation_stages_by_id"]) == set(matrix["mode_ids"])
    assert matrix["mode_classifications_by_id"]["paper_runtime_candidate"] == "paper_runtime"
    for mode_id in matrix["mode_ids"]:
        assert matrix["future_required_gate_ids_by_mode_id"][mode_id] == matrix["gate_ids"]


def test_blocked_capability_gate_matrix_contains_required_blocks() -> None:
    blocked = _payload()["blocked_capability_gate_matrix"]

    assert blocked["blocked_capability_count"] == len(blocked["blocked_capabilities"])
    for capability in REQUIRED_BLOCKS:
        assert capability in blocked["blocked_capabilities"]
    assert blocked["runtime_activation_blocked"] is True
    assert blocked["gate_execution_blocked"] is True
    assert blocked["order_flow_blocked"] is True
    assert blocked["private_endpoint_access_blocked"] is True
    assert blocked["network_io_blocked"] is True
    assert blocked["filesystem_io_blocked"] is True
    assert blocked["live_canary_blocked"] is True
    assert blocked["credential_read_blocked"] is True
    assert blocked["live_trading_blocked"] is True


def test_fail_closed_decision_matrix_blocks_all_activation_paths_in_14_2() -> None:
    matrix = _payload()["fail_closed_decision_matrix"]

    assert matrix["missing_gate_policy"] == "fail_closed"
    assert matrix["missing_read_model_policy"] == "fail_closed"
    assert matrix["missing_contract_policy"] == "fail_closed"
    for key, value in matrix.items():
        if key.endswith("_in_14_2") or key == "runtime_activation_without_future_gate":
            assert value == "blocked"


def test_non_execution_evidence_shows_no_activation_or_execution() -> None:
    evidence = _payload()["non_execution_evidence"]

    assert evidence["source_read_model_read"] is True
    assert evidence["gate_matrix_built"] is True
    for flag in [
        "runtime_activation_started",
        "runtime_gate_executed",
        "gate_state_mutated",
        "mode_activated",
        "order_generated",
        "order_submitted",
        "private_endpoint_accessed",
        "network_io_performed",
        "filesystem_io_performed",
        "live_canary_started",
        "testnet_runtime_started",
        "paper_runtime_activated",
    ]:
        assert evidence[flag] is False


def test_gate_matrix_boundaries_are_closed() -> None:
    boundaries = _payload()["gate_matrix_boundaries"]

    assert boundaries["gate_matrix_is_plain_data_only"] is True
    assert boundaries["gate_matrix_can_feed_14_3_paper_runtime_activation_gate"] is True
    for key, value in boundaries.items():
        if key.startswith("gate_matrix_cannot"):
            assert value is True


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert set(imported_modules) == ALLOWED_IMPORT_MODULES


def test_source_call_guard_blocks_io_network_runtime_orders_and_ui_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    call_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)

    assert not (call_names & FORBIDDEN_CALL_NAMES)


def test_forbidden_literal_tokens_do_not_appear_in_helper() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
