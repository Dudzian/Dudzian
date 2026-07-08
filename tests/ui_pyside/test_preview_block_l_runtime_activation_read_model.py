"""Tests for FUNCTIONAL-PREVIEW-14.1 Block L runtime activation read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_runtime_activation_read_model import (
    BLOCK_ID,
    BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_DECISION,
    BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_KIND,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_2,
    STATUS,
    STEP_ID,
    build_preview_block_l_runtime_activation_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_runtime_activation_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_runtime_activation_read_model_kind",
    "block",
    "step",
    "block_l_runtime_activation_read_model_status",
    "block_l_runtime_activation_read_model_decision",
    "ready_for_block_l_2",
    "next_step",
    "next_step_title",
    "contract_reference",
    "runtime_activation_readiness",
    "runtime_activation_gate_rows",
    "runtime_activation_mode_rows",
    "blocked_capability_read_model",
    "non_activation_evidence_read_model",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

FALSE_READINESS_FLAGS = [
    "safe_to_activate_runtime_now",
    "safe_to_execute_gates_now",
    "safe_to_mutate_gate_state_now",
    "safe_for_paper_runtime_now",
    "safe_for_testnet_runtime_now",
    "safe_for_live_canary_now",
    "safe_for_orders_now",
    "safe_for_private_endpoints_now",
    "safe_for_network_io_now",
    "safe_for_filesystem_io_now",
]

GATE_FALSE_FLAGS = [
    "runtime_activation_allowed_now",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
    "gate_executed_by_read_model",
]

MODE_FALSE_FLAGS = [
    "allowed_in_14_1",
    "runtime_activation_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "credential_read_allowed_now",
    "live_trading_allowed_now",
    "mode_activated_by_read_model",
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
]

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "typing",
    "ui.pyside_app.preview_block_l_runtime_activation_contract",
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
    return build_preview_block_l_runtime_activation_read_model()


def test_payload_is_json_serializable_and_has_stable_top_level_fields() -> None:
    payload = _payload()

    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_14_1() -> None:
    payload = _payload()

    assert payload["schema_version"] == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_SCHEMA_VERSION
    assert (
        payload["block_l_runtime_activation_read_model_kind"]
        == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_KIND
    )
    assert payload["block"] == BLOCK_ID == "L"
    assert payload["step"] == STEP_ID == "14.1"
    assert (
        payload["block_l_runtime_activation_read_model_status"]
        == BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_STATUS
    )
    assert (
        payload["block_l_runtime_activation_read_model_decision"]
        == BLOCK_L_RUNTIME_ACTIVATION_READ_MODEL_DECISION
    )
    assert payload["ready_for_block_l_2"] is READY_FOR_BLOCK_L_2 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-14.2"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "RUNTIME ACTIVATION GATE MATRIX"
    assert payload["status"] == STATUS


def test_contract_reference_points_to_safe_14_0_subset() -> None:
    reference = _payload()["contract_reference"]

    assert reference["schema_version"] == "preview_block_l_runtime_activation_contract.v1"
    assert (
        reference["block_l_runtime_activation_contract_kind"]
        == "functional_preview_block_l_runtime_activation_contract"
    )
    assert reference["block"] == "L"
    assert reference["step"] == "14.0"
    assert reference["ready_for_block_l_1"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.1"
    assert reference["next_step_title"] == "RUNTIME ACTIVATION READ MODEL"
    assert reference["source_contract_step"] == "FUNCTIONAL-PREVIEW-14.0"
    assert reference["source_contract_read_by_14_1_read_model"] is True


def test_gate_rows_are_read_only_and_fail_closed() -> None:
    rows = _payload()["runtime_activation_gate_rows"]

    assert len(rows) == 8
    for row in rows:
        assert row["read_only"] is True
        assert row["runtime_activation_gate_id"].startswith("runtime_activation_gate_")
        assert row["source_gate_id"]
        assert row["display_name"]
        assert row["gate_domain"]
        assert row["gate_type"]
        assert row["planned_source_step"].startswith("FUNCTIONAL-PREVIEW-")
        assert row["required_for_future_activation"] is True
        assert row["eligible_for_future_gate_matrix"] is True
        assert row["safe_for_offline_tests"] is True
        for flag in GATE_FALSE_FLAGS:
            assert row[flag] is False


def test_mode_rows_are_read_only_and_fail_closed() -> None:
    rows = _payload()["runtime_activation_mode_rows"]

    assert len(rows) == 6
    for row in rows:
        assert row["read_only"] is True
        assert row["runtime_activation_mode_id"].startswith("runtime_activation_mode_")
        assert row["source_mode_id"]
        assert row["display_name"]
        assert row["mode_classification"]
        assert row["activation_stage"]
        assert row["requires_future_gate"] is True
        assert row["safe_for_offline_tests"] is True
        for flag in MODE_FALSE_FLAGS:
            assert row[flag] is False


def test_readiness_is_ready_only_for_14_2_and_fail_closed_now() -> None:
    readiness = _payload()["runtime_activation_readiness"]

    assert readiness["contract_available"] is True
    assert readiness["read_model_built"] is True
    assert readiness["ready_for_14_2_gate_matrix"] is True
    for flag in FALSE_READINESS_FLAGS:
        assert readiness[flag] is False


def test_blocked_capability_read_model_contains_required_blocks() -> None:
    blocked = _payload()["blocked_capability_read_model"]

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


def test_non_activation_evidence_shows_no_activation() -> None:
    evidence = _payload()["non_activation_evidence_read_model"]

    assert evidence["source_contract_read"] is True
    assert evidence["read_model_built"] is True
    for flag in [
        "runtime_activation_started",
        "runtime_gate_executed",
        "gate_state_mutated",
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


def test_read_model_boundaries_are_closed() -> None:
    boundaries = _payload()["read_model_boundaries"]

    assert boundaries["read_model_is_plain_data_only"] is True
    assert boundaries["read_model_can_feed_14_2_gate_matrix"] is True
    for key, value in boundaries.items():
        if key.startswith("read_model_cannot"):
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
