from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_selection_gate import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_SELECTION_GATE_DECISION,
    READ_ONLY_MARKET_DATA_SELECTION_GATE_STATUS,
    build_preview_read_only_market_data_selection_gate,
    build_preview_read_only_market_data_selection_result,
)

EXPECTED_ALLOWED = [
    (
        "btc_usdt_static_fixture",
        "BTC/USDT",
        "static_fixture",
        "ok_preview_only",
        "BTC/USDT static fixture",
    ),
    (
        "eth_usdt_static_fixture",
        "ETH/USDT",
        "static_fixture",
        "ok_preview_only",
        "ETH/USDT static fixture",
    ),
    (
        "sol_usdt_low_liquidity_preview",
        "SOL/USDT",
        "static_fixture_low_liquidity_preview",
        "low_liquidity_preview_only",
        "SOL/USDT low-liquidity preview",
    ),
    (
        "ada_usdt_stale_preview",
        "ADA/USDT",
        "static_fixture_stale_preview",
        "stale_preview_only",
        "ADA/USDT stale preview",
    ),
]
EXPECTED_SELECTION_FIELDS = {
    "selection_id",
    "symbol",
    "source",
    "quality_status",
    "selection_status",
    "selection_label",
    "network_io_allowed",
    "market_data_fetch_allowed",
    "controlled_refresh_allowed",
    "execution_allowed",
    "account_data_allowed",
    "order_or_fill_data_allowed",
    "credentials_or_secrets_allowed",
}
FALSE_SELECTION_FLAGS = {
    "network_io_allowed",
    "market_data_fetch_allowed",
    "controlled_refresh_allowed",
    "execution_allowed",
    "account_data_allowed",
    "order_or_fill_data_allowed",
    "credentials_or_secrets_allowed",
}
DENIED_FIELD_NAMES = {
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
}
EXPECTED_REJECTED_IDS = [
    "",
    "unknown_pair",
    "live_btc_usdt",
    "fetch_market_data",
    "account_balance",
]
EXPECTED_BLOCKED_CAPABILITIES = [
    "market data adapter implementation now",
    "network I/O",
    "live market data fetch now",
    "controlled refresh now",
    "exchange API connection",
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
    "TradingController / DecisionEnvelope",
    "live/testnet/account/secrets/export/cloud",
    "QML changes / new QML calls",
    "EXE packaging",
]
EXPECTED_SOURCE_BOUNDARIES = [
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
    "no market data runtime adapter import",
    "no account module import",
    "no secrets module import",
    "no filesystem I/O",
    "no network I/O",
    "no QML changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]


def test_selection_gate_is_plain_json_serializable_dict() -> None:
    gate = build_preview_read_only_market_data_selection_gate()

    assert isinstance(gate, dict)
    assert gate["schema_version"] == PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_SCHEMA_VERSION
    assert json.loads(json.dumps(gate)) == gate


def test_status_decision_next_step_and_audit_reference_are_correct() -> None:
    gate = build_preview_read_only_market_data_selection_gate()

    assert gate["block"] == "H"
    assert gate["step"] == "10.5"
    assert gate["market_data_selection_gate_status"] == READ_ONLY_MARKET_DATA_SELECTION_GATE_STATUS
    assert (
        gate["market_data_selection_gate_decision"] == READ_ONLY_MARKET_DATA_SELECTION_GATE_DECISION
    )
    assert gate["ready_for_block_h_6"] is True
    assert gate["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.6"
    assert gate["next_step_title"] == NEXT_STEP_TITLE
    assert gate["status"] == (
        "ready_for_functional_preview_10_6_read_only_market_data_controlled_refresh_preview"
    )
    assert gate["audit_envelope_reference"] == {
        "schema_version": "preview_read_only_market_data_audit_envelope.v1",
        "market_data_audit_envelope_kind": "functional_preview_block_h_read_only_market_data_audit_envelope",
        "market_data_audit_envelope_status": "read_only_market_data_audit_envelope_ready_no_export_no_network_io",
        "market_data_audit_envelope_decision": "BUILD_AUDIT_ENVELOPE_ONLY_NO_NETWORK_IO_NO_MARKET_DATA_FETCH_NO_EXPORT",
        "ready_for_block_h_4": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.4",
        "next_step_title": "READ-ONLY MARKET DATA UI READ-ONLY SURFACE",
        "status": "ready_for_functional_preview_10_4_read_only_market_data_ui_read_only_surface",
    }


def test_selection_gate_scope_is_selection_gate_only_no_refresh_no_network_no_qml() -> None:
    scope = build_preview_read_only_market_data_selection_gate()["selection_gate_scope"]
    true_keys = {
        "selection_gate_only",
        "audit_envelope_reference_required",
        "allowed_selections_static_only",
        "selection_result_preview_only",
    }

    assert scope["scope_name"] == "read_only_market_data_selection_gate"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_keys)


def test_selection_gate_metadata_is_static_only_non_executable_four_allowed() -> None:
    metadata = build_preview_read_only_market_data_selection_gate()["selection_gate"]

    assert metadata["selection_gate_available"] is True
    assert metadata["selection_gate_static_only"] is True
    assert metadata["selection_gate_executable"] is False
    assert metadata["selection_gate_name"] == "read_only_market_data_preview_selection_gate"
    assert metadata["selection_source_mode"] == "local_static_audit_selection_gate_no_network_io"
    assert metadata["allowed_selection_count"] == 4
    assert metadata["default_selection_id"] == "btc_usdt_static_fixture"
    assert metadata["unknown_selection_status"] == "rejected_unknown_selection_no_refresh"
    for key, value in metadata.items():
        if key in {
            "selection_gate_available",
            "selection_gate_static_only",
            "selection_gate_name",
            "selection_source_mode",
            "allowed_selection_count",
            "default_selection_id",
            "unknown_selection_status",
        }:
            continue
        assert value is False


def test_allowed_selections_are_exact_and_all_execution_flags_false() -> None:
    selections = build_preview_read_only_market_data_selection_gate()["allowed_selections"]

    assert len(selections) == 4
    for selection, (selection_id, symbol, source, quality, label) in zip(
        selections, EXPECTED_ALLOWED, strict=True
    ):
        assert set(selection) == EXPECTED_SELECTION_FIELDS
        assert DENIED_FIELD_NAMES.isdisjoint(selection)
        assert selection["selection_id"] == selection_id
        assert selection["symbol"] == symbol
        assert selection["source"] == source
        assert selection["quality_status"] == quality
        assert selection["selection_status"] == "allowed_preview_only_no_refresh"
        assert selection["selection_label"] == label
        for flag in FALSE_SELECTION_FLAGS:
            assert selection[flag] is False


def test_rejected_selection_examples_are_exact_fail_closed_and_all_flags_false() -> None:
    rejected = build_preview_read_only_market_data_selection_gate()["rejected_selection_examples"]

    assert len(rejected) == 5
    assert [item["selection_id"] for item in rejected] == EXPECTED_REJECTED_IDS
    for item in rejected:
        assert set(item) == {
            "selection_id",
            "selection_status",
            "reason",
            *FALSE_SELECTION_FLAGS,
        }
        assert item["selection_status"] == "rejected_fail_closed_no_refresh"
        assert isinstance(item["reason"], str) and item["reason"]
        for flag in FALSE_SELECTION_FLAGS:
            assert item[flag] is False


def test_selection_boundary_summary_is_complete() -> None:
    summary = build_preview_read_only_market_data_selection_gate()["selection_boundary_summary"]

    assert summary == {
        "allowed_selection_count": 4,
        "rejected_example_count": 5,
        "allowed_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        "normal_preview_symbols": ["BTC/USDT", "ETH/USDT"],
        "low_liquidity_preview_symbols": ["SOL/USDT"],
        "stale_preview_symbols": ["ADA/USDT"],
        "all_allowed_selections_preview_only": True,
        "all_allowed_selections_no_refresh": True,
        "all_allowed_selections_no_network": True,
        "all_allowed_selections_no_fetch": True,
        "all_allowed_selections_no_execution": True,
        "all_allowed_selections_no_account_data": True,
        "all_allowed_selections_no_order_or_fill_data": True,
        "all_allowed_selections_no_credentials_or_secrets": True,
        "unknown_selections_rejected_fail_closed": True,
        "live_or_private_selections_rejected_fail_closed": True,
    }


def test_evidence_has_only_gate_and_audit_static_build_flags_true() -> None:
    evidence = build_preview_read_only_market_data_selection_gate()[
        "no_refresh_no_fetch_no_execution_evidence"
    ]

    true_keys = {
        "selection_gate_evaluated",
        "audit_envelope_read",
        "allowed_selections_built_from_static_audit_events",
    }
    for key, value in evidence.items():
        assert value is (key in true_keys)


def test_boundary_checks_have_expected_true_false_values() -> None:
    checks = build_preview_read_only_market_data_selection_gate()["boundary_checks"]
    true_keys = {
        "local_only",
        "block_h_selection_gate_only",
        "audit_envelope_reference_valid",
        "read_only_market_data_scope_defined",
        "allowed_selections_static_only",
        "public_market_data_allowed_future_only",
        "recorded_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_6",
    }

    for key, value in checks.items():
        assert value is (key in true_keys)


def test_blocked_capabilities_source_boundaries_and_future_steps_are_complete() -> None:
    gate = build_preview_read_only_market_data_selection_gate()

    assert gate["blocked_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    assert gate["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert gate["future_steps"] == [
        "functional_preview_10_6_read_only_market_data_controlled_refresh_preview",
        "functional_preview_10_7_read_only_market_data_bridge_snapshot",
        "functional_preview_10_8_read_only_market_data_closure_audit",
    ]


def test_selection_result_accepts_only_allowlist_and_rejects_everything_else() -> None:
    for selection_id, symbol, source, quality, _label in EXPECTED_ALLOWED:
        result = build_preview_read_only_market_data_selection_result(selection_id)
        assert result == {
            "result_status": "accepted_preview_only_no_refresh",
            "selection_id": selection_id,
            "symbol": symbol,
            "source": source,
            "quality_status": quality,
            "selection_allowed": True,
            "controlled_refresh_allowed": False,
            "controlled_refresh_performed": False,
            "network_io_allowed": False,
            "network_io_performed": False,
            "market_data_fetch_allowed": False,
            "market_data_fetch_performed": False,
            "execution_allowed": False,
            "execution_performed": False,
            "account_data_allowed": False,
            "order_or_fill_data_allowed": False,
            "credentials_or_secrets_allowed": False,
        }

    for selection_id in [
        None,
        "",
        "unknown_pair",
        "live_btc_usdt",
        "fetch_market_data",
        "account_balance",
    ]:
        result = build_preview_read_only_market_data_selection_result(selection_id)
        assert result["result_status"] == "rejected_fail_closed_no_refresh"
        assert result["selection_allowed"] is False
        assert result["reason"] == "selection_id_not_allowed_by_static_preview_gate"
        for key, value in result.items():
            if key in {"result_status", "selection_id", "selection_allowed", "reason"}:
                continue
            assert value is False


def test_helper_imports_only_safe_stdlib_and_audit_envelope_and_has_no_forbidden_calls() -> None:
    module_path = Path("ui/pyside_app/preview_read_only_market_data_selection_gate.py")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_read_only_market_data_audit_envelope",
    ]

    forbidden_calls = {
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
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
    assert forbidden_calls.isdisjoint(call_names)


def test_qml_sources_remain_without_market_data_selection_actions() -> None:
    dashboard = Path("ui/pyside_app/qml/views/OperatorDashboard.qml").read_text(encoding="utf-8")
    main_window = Path("ui/pyside_app/qml/MainWindow.qml").read_text(encoding="utf-8")
    qml = dashboard + "\n" + main_window

    assert qml.count("previewSelectAction(") == 1
    assert (
        'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
        in qml
    )
    for forbidden in [
        ".previewSelectSourceControl(",
        ".resetPreviewSelection(",
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
    ]:
        assert forbidden not in qml
