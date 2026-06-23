"""Tests for FUNCTIONAL-PREVIEW-10.8 Block H market data closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_read_only_market_data_closure_audit import (
    CLOSURE_LINE,
    NEXT_BLOCK,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_DECISION,
    READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_STATUS,
    STATUS,
    build_preview_read_only_market_data_closure_audit,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
EXPECTED_COMPLETED_STEPS = [
    "FUNCTIONAL-PREVIEW-10.0 — READ-ONLY MARKET DATA ADAPTER CONTRACT",
    "FUNCTIONAL-PREVIEW-10.1 — READ-ONLY MARKET DATA READ MODEL",
    "FUNCTIONAL-PREVIEW-10.2 — READ-ONLY MARKET DATA STATIC FIXTURE",
    "FUNCTIONAL-PREVIEW-10.3 — READ-ONLY MARKET DATA AUDIT ENVELOPE",
    "FUNCTIONAL-PREVIEW-10.4 — READ-ONLY MARKET DATA UI READ-ONLY SURFACE",
    "FUNCTIONAL-PREVIEW-10.5 — READ-ONLY MARKET DATA SELECTION GATE",
    "FUNCTIONAL-PREVIEW-10.6 — READ-ONLY MARKET DATA CONTROLLED REFRESH PREVIEW",
    "FUNCTIONAL-PREVIEW-10.7 — READ-ONLY MARKET DATA BRIDGE SNAPSHOT",
    "FUNCTIONAL-PREVIEW-10.8 — READ-ONLY MARKET DATA CLOSURE AUDIT",
]
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "market_data_closure_audit_kind",
    "block",
    "step",
    "market_data_closure_audit_status",
    "market_data_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "block_h_handoff_reference",
    "block_h_completed_steps",
    "block_h_completion_matrix",
    "read_only_market_data_preview_path_summary",
    "bridge_snapshot_closure_summary",
    "no_live_no_fetch_no_runtime_evidence",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "next_block_entry_requirements",
    "closure_decision",
    "status",
}
FALSE_COMPLETION_FLAGS = {
    "network_io_performed",
    "market_data_fetch_performed",
    "live_or_testnet_connection_performed",
    "account_or_private_data_accessed",
    "order_or_fill_action_performed",
    "runtime_execution_performed",
    "qml_action_added",
    "bridge_api_changed",
}
EXPECTED_BOUNDARY_TRUE = {
    "local_only",
    "block_h_closure_audit_only",
    "bridge_snapshot_reference_valid",
    "all_block_h_steps_complete",
    "read_only_market_data_scope_complete",
    "public_market_data_allowed_future_only",
    "recorded_replay_allowed_future_only",
    "testnet_mode_allowed_next_block_contract_only",
    "exe_direction_preserved",
    "ready_for_next_block",
}
EXPECTED_BOUNDARY_FALSE = {
    "adapter_implemented_now",
    "refresh_execution_allowed_now",
    "refresh_performed_now",
    "controlled_refresh_allowed_now",
    "controlled_refresh_performed_now",
    "network_io_allowed_now",
    "market_data_fetch_allowed_now",
    "live_market_data_fetch_allowed_now",
    "exchange_connection_allowed_now",
    "private_account_data_allowed",
    "account_fetch_allowed",
    "balance_fetch_allowed",
    "positions_fetch_allowed",
    "orders_fetch_allowed",
    "fills_fetch_allowed",
    "order_generation_allowed",
    "order_submission_allowed",
    "fill_simulation_allowed",
    "lifecycle_mutation_allowed",
    "runtime_loop_allowed",
    "scheduler_allowed",
    "trading_controller_allowed",
    "decision_envelope_allowed",
    "strategy_execution_allowed",
    "ai_scoring_execution_allowed",
    "model_inference_execution_allowed",
    "live_mode_allowed",
    "testnet_mode_allowed_in_block_h",
    "live_credentials_allowed",
    "secrets_read_allowed",
    "secrets_export_allowed",
    "cloud_export_allowed",
    "external_export_allowed",
    "dynamic_action_dispatch_allowed",
    "new_qml_method_calls_allowed",
    "qml_changes_allowed",
    "bridge_api_changes_allowed",
    "exe_packaging_in_scope",
    "bat_productization_allowed",
}
EXPECTED_BLOCKED_CAPABILITIES = {
    "market data adapter implementation in Block H",
    "network I/O in Block H",
    "live market data fetch in Block H",
    "controlled refresh execution in Block H",
    "exchange API connection in Block H",
    "private account endpoint access",
    "account balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order generation",
    "order submission",
    "fill simulation",
    "lifecycle mutation",
    "runtime loop",
    "scheduler",
    "audit export",
    "bridge API changes",
    "TradingController / DecisionEnvelope",
    "live/testnet/account/secrets/export/cloud in Block H",
    "QML changes / new QML calls in closure audit",
    "EXE packaging",
}
EXPECTED_SOURCE_BOUNDARIES = {
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no scheduler import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no strategy/AI/scoring/recommendation import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import in Block H",
    "no market data runtime adapter import",
    "no account module import",
    "no secrets module import",
    "no filesystem I/O",
    "no network I/O",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
}
FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "qml",
    "runtime",
    "scheduler",
    "TradingController",
    "DecisionEnvelope",
    "order",
    "live",
    "testnet",
    "account",
    "secrets",
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "pathlib",
    "subprocess",
}
FORBIDDEN_CALLS = {
    "open",
    "read_text",
    "write_text",
    "requests",
    "subprocess",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "QQmlApplicationEngine",
    "TradingController",
    "DecisionEnvelope",
    "start_runtime",
    "start_loop",
    "submit_order",
    "place_order",
    "create_order",
    "send_order",
    "fill_order",
    "fetch_market_data",
    "fetch_balance",
    "fetch_account",
    "refresh_market_data",
    "export",
}


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def test_closure_audit_is_json_serializable_plain_dict_with_expected_top_level_fields() -> None:
    audit = build_preview_read_only_market_data_closure_audit()

    assert set(audit) == EXPECTED_TOP_LEVEL_FIELDS
    _assert_simple_types_only(audit)
    json.dumps(audit, sort_keys=True)


def test_closure_audit_records_status_decision_and_next_block() -> None:
    audit = build_preview_read_only_market_data_closure_audit()

    assert audit["market_data_closure_audit_status"] == READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_STATUS
    assert (
        audit["market_data_closure_audit_decision"] == READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_DECISION
    )
    assert audit["ready_for_next_block"] is True
    assert audit["next_block"] == NEXT_BLOCK
    assert audit["next_step"] == NEXT_STEP
    assert audit["next_step_title"] == NEXT_STEP_TITLE
    assert audit["status"] == STATUS


def test_handoff_reference_points_to_10_7_bridge_snapshot_ready_for_10_8() -> None:
    handoff = build_preview_read_only_market_data_closure_audit()["block_h_handoff_reference"]

    assert handoff["read_only_market_data_bridge_snapshot_ready_for_block_h_8"] is True
    assert handoff["read_only_market_data_bridge_snapshot_next_step"] == "FUNCTIONAL-PREVIEW-10.8"
    assert (
        handoff["read_only_market_data_bridge_snapshot_next_step_title"]
        == "READ-ONLY MARKET DATA CLOSURE AUDIT"
    )
    assert handoff["read_only_market_data_allowed_refresh_preview_count"] == 4
    assert (
        handoff["read_only_market_data_default_refresh_selection_id"] == "btc_usdt_static_fixture"
    )
    assert handoff["read_only_market_data_allowed_refresh_symbols"] == [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "ADA/USDT",
    ]


def test_completed_steps_and_completion_matrix_cover_10_0_through_10_8() -> None:
    audit = build_preview_read_only_market_data_closure_audit()

    assert audit["block_h_completed_steps"] == EXPECTED_COMPLETED_STEPS
    matrix = audit["block_h_completion_matrix"]
    assert [entry["step"] for entry in matrix] == [
        "10.0",
        "10.1",
        "10.2",
        "10.3",
        "10.4",
        "10.5",
        "10.6",
        "10.7",
        "10.8",
    ]
    assert [entry["artifact"] for entry in matrix] == [
        "contract",
        "read model",
        "static fixture",
        "audit envelope",
        "UI read-only surface",
        "selection gate",
        "controlled refresh preview",
        "bridge snapshot",
        "closure audit",
    ]
    for entry in matrix:
        assert entry["ready"] is True
        for flag in FALSE_COMPLETION_FLAGS:
            assert entry[flag] is False


def test_preview_path_summary_is_complete_and_non_executing() -> None:
    summary = build_preview_read_only_market_data_closure_audit()[
        "read_only_market_data_preview_path_summary"
    ]

    true_keys = {key for key, value in summary.items() if isinstance(value, bool)}
    assert true_keys == {
        "block_h_complete",
        "read_only_market_data_contract_ready",
        "read_only_market_data_read_model_ready",
        "static_fixture_ready",
        "audit_envelope_ready",
        "ui_read_only_surface_ready",
        "selection_gate_ready",
        "controlled_refresh_preview_ready",
        "bridge_snapshot_ready",
        "closure_audit_ready",
        "no_real_refresh_performed",
        "no_market_data_fetch_performed",
        "no_network_io_performed",
        "no_exchange_api_connection_opened",
        "no_account_data_accessed",
        "no_order_or_fill_action_performed",
        "no_runtime_loop_started",
        "no_scheduler_started",
        "no_qml_action_added",
        "no_bridge_api_changes",
        "no_live_or_testnet_connection",
        "no_credentials_or_secrets_accessed",
        "no_export_performed",
        "exe_direction_preserved",
        "ready_for_next_block",
    }
    assert all(summary[key] is True for key in true_keys)
    assert summary["allowed_market_symbols"] == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    assert summary["allowed_refresh_preview_count"] == 4
    assert summary["default_selection_id"] == "btc_usdt_static_fixture"


def test_bridge_snapshot_closure_summary_is_green() -> None:
    summary = build_preview_read_only_market_data_closure_audit()["bridge_snapshot_closure_summary"]

    assert summary == {
        "bridge_snapshot_data_only": True,
        "bridge_snapshot_qml_safe": True,
        "bridge_snapshot_ready_for_block_h_8": True,
        "controlled_refresh_preview_read": True,
        "allowed_refresh_preview_count": 4,
        "default_refresh_selection_id": "btc_usdt_static_fixture",
        "no_refresh_summary_green": True,
        "no_fetch_summary_green": True,
        "no_network_summary_green": True,
        "no_bridge_api_change_summary_green": True,
        "next_step_matched_closure_audit": True,
    }


def test_evidence_marks_only_audit_and_reference_as_evaluated() -> None:
    evidence = build_preview_read_only_market_data_closure_audit()[
        "no_live_no_fetch_no_runtime_evidence"
    ]

    assert evidence["closure_audit_evaluated"] is True
    assert evidence["bridge_snapshot_read"] is True
    assert evidence["block_h_artifacts_referenced"] is True
    for key, value in evidence.items():
        if key not in {
            "closure_audit_evaluated",
            "bridge_snapshot_read",
            "block_h_artifacts_referenced",
        }:
            assert value is False


def test_boundary_checks_block_execution_and_allow_next_contract_only() -> None:
    checks = build_preview_read_only_market_data_closure_audit()["boundary_checks"]

    for key in EXPECTED_BOUNDARY_TRUE:
        assert checks[key] is True
    for key in EXPECTED_BOUNDARY_FALSE:
        assert checks[key] is False


def test_blocked_capabilities_source_boundaries_and_next_requirements_are_complete() -> None:
    audit = build_preview_read_only_market_data_closure_audit()

    assert set(audit["blocked_capabilities"]) == EXPECTED_BLOCKED_CAPABILITIES
    assert set(audit["source_boundaries"]) == EXPECTED_SOURCE_BOUNDARIES
    assert audit["next_block_entry_requirements"] == {
        "next_block_contract_first": True,
        "testnet_or_sandbox_contract_required_before_any_adapter": True,
        "no_testnet_runtime_until_contract_and_guards": True,
        "no_live_mode_in_next_block_initial_step": True,
        "no_credentials_until_explicit_secrets_gate": True,
        "no_account_fetch_until_explicit_private_endpoint_gate": True,
        "no_order_submission_until_explicit_order_gate": True,
        "block_i_must_start_with_contract_only": True,
    }


def test_closure_decision_closes_block_h_with_exact_line() -> None:
    decision = build_preview_read_only_market_data_closure_audit()["closure_decision"]

    assert decision == {
        "close_block_h": True,
        "ready_for_next_block": True,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "closure_line": CLOSURE_LINE,
    }


def test_source_imports_only_stdlib_typing_and_bridge_snapshot_helper() -> None:
    source = Path("ui/pyside_app/preview_read_only_market_data_closure_audit.py").read_text()
    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)

    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_action_dispatch_bridge_snapshot",
    ]
    assert not any(import_name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS for import_name in imports)
    assert not (set(calls) & FORBIDDEN_CALLS)
