"""Tests for FUNCTIONAL-PREVIEW-13.6 Block K closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_k_closure_audit import (
    BLOCK_ID,
    BLOCK_K_CLOSURE_AUDIT_DECISION,
    BLOCK_K_CLOSURE_AUDIT_STATUS,
    CLOSURE_LINE,
    NEXT_BLOCK,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_K_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_K_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_NEXT_BLOCK,
    STATUS,
    STEP_ID,
    build_preview_block_k_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_k_closure_audit.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_k_closure_audit_kind",
    "block",
    "step",
    "block_k_closure_audit_status",
    "block_k_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
    "block_k_contract_reference",
    "observability_read_model_reference",
    "audit_envelope_read_model_reference",
    "rollback_read_model_reference",
    "soak_read_model_reference",
    "gate_matrix_reference",
    "block_k_closure_scope",
    "block_k_completed_steps",
    "block_k_completion_matrix",
    "block_k_safety_summary",
    "observability_audit_rollback_soak_capability_closure",
    "runtime_order_private_network_closure",
    "blocked_block_k_capabilities",
    "block_k_source_boundaries",
    "non_activation_evidence",
    "next_block_entry_requirements",
    "future_blocks",
    "status",
]
COMPLETED_IDS = [
    "functional_preview_13_0_contract",
    "functional_preview_13_1_observability_read_model",
    "functional_preview_13_2_audit_envelope_read_model",
    "functional_preview_13_3_rollback_read_model",
    "functional_preview_13_4_soak_read_model",
    "functional_preview_13_5_gate_matrix",
    "functional_preview_13_6_closure_audit",
]
READY_FLAGS = [
    "ready_for_block_k_1",
    "ready_for_block_k_2",
    "ready_for_block_k_3",
    "ready_for_block_k_4",
    "ready_for_block_k_5",
    "ready_for_block_k_6",
    "ready_for_next_block",
]
FUTURE_BLOCKS = [
    "BLOK L — RUNTIME ACTIVATION / LIVE CANARY GATES",
    "BLOK M — FULL RUNTIME PRODUCTIZATION",
    "BLOK N — FULL UI PRODUCT COMPLETION",
    "BLOK O — SECURITY / CREDENTIALS / LICENSING",
    "BLOK P — EXE PACKAGING / INSTALLER / SIGNING",
    "BLOK Q — PRODUCTION READINESS / FINAL RC",
]
BLOCKED = [
    "Block L implementation",
    "runtime activation",
    "runtime gate execution",
    "gate state mutation",
    "live canary",
    "testnet runtime",
    "paper runtime activation",
    "observability runtime collection",
    "metrics collection",
    "metrics export",
    "audit writer",
    "audit export",
    "audit file read",
    "audit file write",
    "log file read",
    "log file write",
    "rollback execution",
    "runtime shutdown",
    "soak runtime",
    "soak scheduler",
    "runtime loop",
    "wall-clock runtime measurement",
    "stability probe",
    "state mutation",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account read",
    "balance read",
    "positions read",
    "orders read",
    "fills read",
    "market data read",
    "adapter instantiation",
    "adapter runtime wiring",
    "scheduler",
    "filesystem I/O",
    "network I/O",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "credential read",
    "secret read",
    "secure store read",
    "secure store write",
    "config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "TradingController change",
    "DecisionEnvelope change",
    "QML action dispatch",
    "bridge API changes",
    "PyInstaller/EXE packaging",
]
BOUNDARIES = [
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no scheduler import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no strategy/AI/scoring/recommendation import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no sandbox adapter import",
    "no exchange adapter runtime import",
    "no account module import",
    "no secrets module import",
    "no security store import",
    "no observability runtime import",
    "no logger/exporter runtime import",
    "no metrics exporter import",
    "no audit writer import",
    "no audit exporter import",
    "no rollback runner import",
    "no rollback executor import",
    "no soak runner import",
    "no soak scheduler import",
    "no filesystem I/O",
    "no audit file read",
    "no audit file write",
    "no log file read",
    "no log file write",
    "no audit write",
    "no audit export",
    "no runtime shutdown",
    "no state mutation",
    "no wall-clock runtime measurement",
    "no stability probe",
    "no config file read",
    "no config discovery",
    "no YAML parse",
    "no JSON parse",
    "no environment variable read",
    "no credential read",
    "no credential validation",
    "no secret material handling",
    "no secure store read",
    "no secure store write",
    "no real market data read",
    "no private endpoint access",
    "no account read",
    "no balance read",
    "no positions read",
    "no orders read",
    "no fills read",
    "no order generation",
    "no order submission",
    "no order cancel",
    "no order replace",
    "no position mutation",
    "no gate execution",
    "no gate state mutation",
    "no runtime activation",
    "no Block L implementation",
    "no live canary",
    "no testnet runtime",
    "no paper runtime activation",
    "no network I/O",
    "no DNS lookup",
    "no HTTP request",
    "no WebSocket connection",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_k_closure_audit()


def test_block_k_closure_audit_is_plain_json_serializable_dict() -> None:
    payload = _payload()
    assert isinstance(payload, dict)
    assert json.loads(json.dumps(payload)) == payload


def test_top_level_identity_status_decision_and_next_step_are_exact() -> None:
    payload = _payload()
    assert list(payload) == TOP_LEVEL_FIELDS
    assert payload["schema_version"] == PREVIEW_BLOCK_K_CLOSURE_AUDIT_SCHEMA_VERSION
    assert payload["block_k_closure_audit_kind"] == PREVIEW_BLOCK_K_CLOSURE_AUDIT_KIND
    assert payload["block"] == BLOCK_ID
    assert payload["step"] == STEP_ID
    assert payload["block_k_closure_audit_status"] == BLOCK_K_CLOSURE_AUDIT_STATUS
    assert payload["block_k_closure_audit_decision"] == BLOCK_K_CLOSURE_AUDIT_DECISION
    assert payload["ready_for_next_block"] is READY_FOR_NEXT_BLOCK
    assert payload["next_block"] == NEXT_BLOCK
    assert payload["next_step"] == NEXT_STEP
    assert payload["next_step_title"] == NEXT_STEP_TITLE
    assert payload["closure_line"] == CLOSURE_LINE
    assert payload["status"] == STATUS


def test_references_keep_only_safe_subsets_and_ready_flags() -> None:
    payload = _payload()
    expected = [
        (
            "block_k_contract_reference",
            [
                "schema_version",
                "observability_audit_rollback_soak_contract_kind",
                "observability_audit_rollback_soak_contract_status",
                "observability_audit_rollback_soak_contract_decision",
                "ready_for_block_k_1",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_1",
            "FUNCTIONAL-PREVIEW-13.1",
            None,
        ),
        (
            "observability_read_model_reference",
            [
                "schema_version",
                "observability_read_model_kind",
                "observability_read_model_status",
                "observability_read_model_decision",
                "ready_for_block_k_2",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_2",
            "FUNCTIONAL-PREVIEW-13.2",
            None,
        ),
        (
            "audit_envelope_read_model_reference",
            [
                "schema_version",
                "audit_envelope_read_model_kind",
                "audit_envelope_read_model_status",
                "audit_envelope_read_model_decision",
                "ready_for_block_k_3",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_3",
            "FUNCTIONAL-PREVIEW-13.3",
            None,
        ),
        (
            "rollback_read_model_reference",
            [
                "schema_version",
                "rollback_read_model_kind",
                "rollback_read_model_status",
                "rollback_read_model_decision",
                "ready_for_block_k_4",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_4",
            "FUNCTIONAL-PREVIEW-13.4",
            None,
        ),
        (
            "soak_read_model_reference",
            [
                "schema_version",
                "soak_read_model_kind",
                "soak_read_model_status",
                "soak_read_model_decision",
                "ready_for_block_k_5",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_5",
            "FUNCTIONAL-PREVIEW-13.5",
            None,
        ),
        (
            "gate_matrix_reference",
            [
                "schema_version",
                "observability_audit_rollback_soak_gate_matrix_kind",
                "observability_audit_rollback_soak_gate_matrix_status",
                "observability_audit_rollback_soak_gate_matrix_decision",
                "ready_for_block_k_6",
                "next_step",
                "next_step_title",
                "status",
            ],
            "ready_for_block_k_6",
            "FUNCTIONAL-PREVIEW-13.6",
            "BLOCK K CLOSURE AUDIT",
        ),
    ]
    for key, keys, ready, next_step, title in expected:
        ref = payload[key]
        assert list(ref) == keys
        assert ref[ready] is True
        assert ref["next_step"] == next_step
        if title is not None:
            assert ref["next_step_title"] == title


def test_scope_completion_and_summaries_are_exact() -> None:
    payload = _payload()
    scope = payload["block_k_closure_scope"]
    true_flags = {
        "closure_audit_only",
        "closes_block_k",
        "derived_from_block_k_contract_13_0",
        "derived_from_observability_read_model_13_1",
        "derived_from_audit_envelope_read_model_13_2",
        "derived_from_rollback_read_model_13_3",
        "derived_from_soak_read_model_13_4",
        "derived_from_gate_matrix_13_5",
        "next_block_is_block_l",
        "exe_direction_preserved",
    }
    assert scope["scope_name"] == "block_k_closure_audit"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_flags)
    steps = payload["block_k_completed_steps"]
    assert [step["completed_step_id"] for step in steps] == COMPLETED_IDS
    assert [step["ready_flag"] for step in steps] == READY_FLAGS
    for step in steps:
        assert step["completion_state"] == "complete"
        assert step["runtime_activation_allowed_now"] is False
        assert step["order_flow_allowed_now"] is False
        assert step["private_endpoint_access_allowed_now"] is False
        assert step["network_io_allowed_now"] is False
        assert step["safe_for_offline_tests"] is True
        assert step["notes"]
    assert payload["block_k_completion_matrix"] == {
        "completed_step_ids": COMPLETED_IDS,
        "source_step_ids": COMPLETED_IDS[:6],
        "closure_step_id": COMPLETED_IDS[-1],
        "ready_flags_in_order": READY_FLAGS,
        "source_steps_all_complete": True,
        "closure_audit_complete": True,
        "block_k_complete": True,
        "ready_for_next_block": True,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "closure_line": CLOSURE_LINE,
    }
    assert payload["block_k_safety_summary"] == {
        "block_k_complete": True,
        "block_k_runtime_activation_performed": False,
        "block_k_order_flow_performed": False,
        "block_k_private_endpoint_access_performed": False,
        "block_k_network_io_performed": False,
        "block_k_filesystem_io_performed": False,
        "block_k_qml_changed": False,
        "block_k_bridge_api_changed": False,
        "block_k_ready_for_next_block": True,
        "next_block_requires_new_explicit_contract": True,
        "safe_to_enter_block_l_contract_only": True,
        "safe_to_enter_block_l_runtime_activation": False,
        "safe_to_enter_live_canary": False,
    }


def test_closure_capabilities_boundaries_evidence_and_future_blocks_are_exact() -> None:
    payload = _payload()
    assert payload["observability_audit_rollback_soak_capability_closure"] == {
        "observability_contract_ready": True,
        "observability_read_model_ready": True,
        "audit_envelope_read_model_ready": True,
        "rollback_read_model_ready": True,
        "soak_read_model_ready": True,
        "gate_matrix_ready": True,
        "closure_audit_ready": True,
        "observability_runtime_enabled": False,
        "metrics_collection_enabled": False,
        "metrics_export_enabled": False,
        "audit_writer_enabled": False,
        "audit_export_enabled": False,
        "rollback_execution_enabled": False,
        "soak_runtime_enabled": False,
        "soak_scheduler_enabled": False,
        "runtime_loop_enabled": False,
    }
    assert payload["runtime_order_private_network_closure"] == {
        "runtime_activation_allowed_now": False,
        "runtime_gate_execution_allowed_now": False,
        "runtime_started": False,
        "order_generation_allowed_now": False,
        "order_submission_allowed_now": False,
        "order_cancel_allowed_now": False,
        "order_replace_allowed_now": False,
        "order_flow_performed": False,
        "private_endpoint_access_allowed_now": False,
        "private_endpoint_accessed": False,
        "account_read_allowed_now": False,
        "balance_read_allowed_now": False,
        "positions_read_allowed_now": False,
        "orders_read_allowed_now": False,
        "fills_read_allowed_now": False,
        "market_data_read_allowed_now": False,
        "network_io_allowed_now": False,
        "dns_lookup_allowed_now": False,
        "http_request_allowed_now": False,
        "websocket_allowed_now": False,
    }
    assert payload["blocked_block_k_capabilities"] == BLOCKED
    assert payload["block_k_source_boundaries"] == BOUNDARIES
    evidence = payload["non_activation_evidence"]
    for key in [
        "block_k_contract_13_0_read",
        "observability_read_model_13_1_read",
        "audit_envelope_read_model_13_2_read",
        "rollback_read_model_13_3_read",
        "soak_read_model_13_4_read",
        "gate_matrix_13_5_read",
        "block_k_closure_audit_built",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if key not in {
            "block_k_contract_13_0_read",
            "observability_read_model_13_1_read",
            "audit_envelope_read_model_13_2_read",
            "rollback_read_model_13_3_read",
            "soak_read_model_13_4_read",
            "gate_matrix_13_5_read",
            "block_k_closure_audit_built",
        }:
            assert value is False
    assert payload["next_block_entry_requirements"] == {
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "requires_new_explicit_contract": True,
        "block_l_contract_only_first": True,
        "runtime_activation_must_remain_blocked_until_block_l_gate": True,
        "live_canary_must_remain_blocked_until_explicit_gate": True,
        "testnet_runtime_must_remain_blocked_until_explicit_gate": True,
        "paper_runtime_activation_must_remain_blocked_until_explicit_gate": True,
        "risk_governor_runtime_must_remain_blocked_until_explicit_gate": True,
        "order_flow_must_remain_blocked_until_explicit_gate": True,
        "private_endpoint_must_remain_blocked_until_explicit_gate": True,
        "network_io_must_remain_blocked_until_explicit_gate": True,
        "credentials_must_remain_blocked_until_explicit_gate": True,
        "qml_bridge_changes_must_remain_blocked_until_explicit_gate": True,
    }
    assert payload["future_blocks"] == FUTURE_BLOCKS


def test_source_imports_are_limited_to_safe_preview_helpers() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(isinstance(node, ast.Import) for node in imports)
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom)}
    assert modules == {
        "__future__",
        "typing",
        "ui.pyside_app.preview_audit_envelope_read_model",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
        "ui.pyside_app.preview_observability_audit_rollback_soak_gate_matrix",
        "ui.pyside_app.preview_observability_read_model",
        "ui.pyside_app.preview_rollback_read_model",
        "ui.pyside_app.preview_soak_read_model",
    }


def test_source_has_no_forbidden_calls_or_literal_tokens() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "open",
        "read_text",
        "write_text",
        "getenv",
        "create_connection",
        "getaddrinfo",
        "start_runtime",
        "start_loop",
        "start_observability",
        "collect_metrics",
        "export_metrics",
        "write_log",
        "read_log",
        "write_audit",
        "export_audit",
        "execute_rollback",
        "shutdown_runtime",
        "mutate_state",
        "start_soak",
        "run_soak",
        "start_scheduler",
        "schedule",
        "sleep",
        "execute_gate",
        "activate_runtime",
        "implement_block_l",
        "start_live_canary",
        "start_testnet_runtime",
        "activate_paper_runtime",
        "mutate_gate",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "cancel_order",
        "replace_order",
        "withdraw",
        "transfer",
        "fetch_market_data",
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
    }
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert calls.isdisjoint(forbidden_calls)
    assert "fetch_" + "balance" not in source
    assert "cc" + "xt" not in source
