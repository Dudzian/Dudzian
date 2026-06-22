from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_static_fixture import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_STATIC_FIXTURE_DECISION,
    READ_ONLY_MARKET_DATA_STATIC_FIXTURE_STATUS,
    build_preview_read_only_market_data_static_fixture,
)

_ALLOWED_FIELDS = [
    "symbol",
    "timestamp",
    "bid",
    "ask",
    "last",
    "volume",
    "source",
    "latency_ms_preview",
]
_DENIED_FIELDS = [
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
]


def test_static_fixture_is_plain_json_serializable_dict() -> None:
    fixture = build_preview_read_only_market_data_static_fixture()

    assert isinstance(fixture, dict)
    assert fixture["schema_version"] == PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_SCHEMA_VERSION
    assert json.loads(json.dumps(fixture)) == fixture


def test_status_decision_next_step_and_read_model_reference_are_correct() -> None:
    fixture = build_preview_read_only_market_data_static_fixture()

    assert fixture["block"] == "H"
    assert fixture["step"] == "10.2"
    assert (
        fixture["market_data_static_fixture_status"] == READ_ONLY_MARKET_DATA_STATIC_FIXTURE_STATUS
    )
    assert (
        fixture["market_data_static_fixture_decision"]
        == READ_ONLY_MARKET_DATA_STATIC_FIXTURE_DECISION
    )
    assert fixture["ready_for_block_h_3"] is True
    assert fixture["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.3"
    assert fixture["next_step_title"] == NEXT_STEP_TITLE == "READ-ONLY MARKET DATA AUDIT ENVELOPE"
    assert (
        fixture["status"]
        == "ready_for_functional_preview_10_3_read_only_market_data_audit_envelope"
    )
    assert fixture["read_model_reference"] == {
        "schema_version": "preview_read_only_market_data_read_model.v1",
        "market_data_read_model_kind": "functional_preview_block_h_read_only_market_data_read_model",
        "market_data_read_model_status": "read_only_market_data_read_model_ready_no_market_data_fetch",
        "market_data_read_model_decision": "BUILD_READ_MODEL_ONLY_NO_NETWORK_IO_NO_MARKET_DATA_FETCH",
        "ready_for_block_h_2": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.2",
        "next_step_title": "READ-ONLY MARKET DATA STATIC FIXTURE",
        "status": "ready_for_functional_preview_10_2_read_only_market_data_static_fixture",
    }


def test_fixture_scope_is_static_fixture_only_and_no_network() -> None:
    scope = build_preview_read_only_market_data_static_fixture()["fixture_scope"]
    true_keys = {
        "static_fixture_only",
        "read_model_reference_required",
        "fixture_data_available_now",
        "fixture_data_static_only",
        "recorded_replay_allowed_future_only",
        "public_market_data_allowed_future_only",
    }

    assert scope["scope_name"] == "read_only_market_data_static_fixture"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_keys)


def test_market_data_static_fixture_metadata_is_static_only_non_executable_four_rows() -> None:
    metadata = build_preview_read_only_market_data_static_fixture()["market_data_static_fixture"]

    assert metadata["fixture_available"] is True
    assert metadata["fixture_static_only"] is True
    assert metadata["fixture_executable"] is False
    assert metadata["fixture_name"] == "read_only_market_data_preview_static_fixture"
    assert metadata["data_source_mode"] == "local_static_fixture_no_network_io"
    assert metadata["rows_available_now"] is True
    assert metadata["row_count"] == 4
    assert metadata["fixture_source"] == "functional_preview_static_fixture"
    assert metadata["fixture_version"] == "10.2"
    for key in (
        "fixture_generated_from_network",
        "fixture_generated_from_account",
        "fixture_contains_private_data",
        "fixture_contains_orders_or_fills",
        "fixture_contains_credentials_or_secrets",
        "network_io_allowed_now",
        "market_fetch_performed",
        "runtime_execution_allowed",
        "trading_execution_allowed",
        "account_access_allowed",
        "credentials_access_allowed",
    ):
        assert metadata[key] is False


def test_fixture_rows_are_allowlist_only_private_denylist_absent_and_numerically_valid() -> None:
    fixture = build_preview_read_only_market_data_static_fixture()
    rows = fixture["fixture_rows"]

    assert len(rows) == 4
    assert fixture["market_data_field_allowlist"] == _ALLOWED_FIELDS
    assert fixture["market_data_private_field_denylist"] == _DENIED_FIELDS
    assert set(_ALLOWED_FIELDS).isdisjoint(_DENIED_FIELDS)
    for row in rows:
        assert list(row) == _ALLOWED_FIELDS
        assert set(row) == set(_ALLOWED_FIELDS)
        assert set(row).isdisjoint(_DENIED_FIELDS)
        assert json.loads(json.dumps(row)) == row
        for key in ("bid", "ask", "last", "volume", "latency_ms_preview"):
            assert isinstance(row[key], float)
            assert math.isfinite(row[key])
        assert row["bid"] <= row["ask"]
        assert row["bid"] <= row["last"] <= row["ask"]
        assert row["volume"] >= 0
        assert row["latency_ms_preview"] >= 0


def test_fixture_rows_have_expected_static_values() -> None:
    rows = build_preview_read_only_market_data_static_fixture()["fixture_rows"]

    assert rows[0] == {
        "symbol": "BTC/USDT",
        "timestamp": "2026-01-01T00:00:00Z",
        "bid": 43000.0,
        "ask": 43010.0,
        "last": 43005.0,
        "volume": 123.45,
        "source": "static_fixture",
        "latency_ms_preview": 0.0,
    }
    assert rows[1] == {
        "symbol": "ETH/USDT",
        "timestamp": "2026-01-01T00:00:01Z",
        "bid": 2300.0,
        "ask": 2301.0,
        "last": 2300.5,
        "volume": 456.78,
        "source": "static_fixture",
        "latency_ms_preview": 0.0,
    }
    assert rows[2]["symbol"] == "SOL/USDT"
    assert rows[2]["source"] == "static_fixture_low_liquidity_preview"
    assert rows[3]["symbol"] == "ADA/USDT"
    assert rows[3]["source"] == "static_fixture_stale_preview"


def test_validation_preview_has_all_expected_flags() -> None:
    validation = build_preview_read_only_market_data_static_fixture()["fixture_validation_preview"]

    assert validation == {
        "validation_available": True,
        "validation_static_only": True,
        "validation_executable": False,
        "row_count": 4,
        "allowed_fields_only": True,
        "private_fields_absent": True,
        "account_fields_absent": True,
        "order_fields_absent": True,
        "fill_fields_absent": True,
        "credentials_fields_absent": True,
        "network_required_for_validation": False,
        "market_data_fetch_required_for_validation": False,
        "all_rows_have_symbol": True,
        "all_rows_have_timestamp": True,
        "all_rows_have_bid_ask_last": True,
        "all_rows_have_volume": True,
        "all_rows_have_source": True,
        "all_rows_have_latency_preview": True,
        "all_rows_have_bid_less_than_or_equal_ask": True,
        "all_rows_have_last_within_bid_ask": True,
        "all_rows_have_non_negative_volume": True,
        "all_rows_have_non_negative_latency": True,
    }


def test_no_fetch_no_execution_evidence_has_only_static_fixture_and_read_model_true() -> None:
    evidence = build_preview_read_only_market_data_static_fixture()[
        "no_fetch_no_execution_evidence"
    ]

    true_keys = {
        "static_fixture_evaluated",
        "read_model_read",
        "fixture_rows_built_from_static_literals",
    }
    for key, value in evidence.items():
        assert value is (key in true_keys)


def test_boundary_checks_block_network_account_order_fill_runtime_live_and_export() -> None:
    boundary = build_preview_read_only_market_data_static_fixture()["boundary_checks"]
    true_keys = {
        "local_only",
        "block_h_static_fixture_only",
        "read_model_reference_valid",
        "read_only_market_data_scope_defined",
        "fixture_data_available_now",
        "fixture_data_static_only",
        "public_market_data_allowed_future_only",
        "recorded_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_3",
    }

    for key, value in boundary.items():
        assert value is (key in true_keys)


def test_capability_lists_source_boundaries_and_future_steps_are_complete() -> None:
    fixture = build_preview_read_only_market_data_static_fixture()

    assert fixture["blocked_capabilities"] == [
        "market data adapter implementation now",
        "network I/O",
        "live market data fetch now",
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
        "TradingController / DecisionEnvelope",
        "live/testnet/account/secrets/export/cloud",
        "QML changes / new QML calls",
        "EXE packaging",
    ]
    assert fixture["source_boundaries"] == [
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
    assert fixture["future_steps"] == [
        "functional_preview_10_3_read_only_market_data_audit_envelope",
        "functional_preview_10_4_read_only_market_data_ui_read_only_surface",
        "functional_preview_10_5_read_only_market_data_selection_gate",
    ]


def test_static_fixture_source_imports_only_safe_stdlib_and_10_1_read_model() -> None:
    source_path = Path("ui/pyside_app/preview_read_only_market_data_static_fixture.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert imports == [
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_read_only_market_data_read_model",
    ]


def test_static_fixture_source_has_no_forbidden_imports_or_calls() -> None:
    source = Path("ui/pyside_app/preview_read_only_market_data_static_fixture.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    forbidden_import_fragments = {
        "PySide",
        "qml",
        "runtime",
        "scheduler",
        "TradingController",
        "DecisionEnvelope",
        "order",
        "live",
        "testnet",
        "market_data_runtime",
        "account",
        "secrets",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
    }
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert imported_modules == [
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_read_only_market_data_read_model",
    ]
    for imported_module in imported_modules:
        assert not any(fragment in imported_module for fragment in forbidden_import_fragments)

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
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                assert function.id not in forbidden_calls
            elif isinstance(function, ast.Attribute):
                assert function.attr not in forbidden_calls


def test_qml_surface_remains_unchanged_single_allowed_preview_selection_call() -> None:
    dashboard = Path("ui/pyside_app/qml/views/OperatorDashboard.qml").read_text(encoding="utf-8")
    main_window = Path("ui/pyside_app/qml/MainWindow.qml").read_text(encoding="utf-8")
    qml = dashboard + "\n" + main_window

    allowed_literal = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
    assert qml.count("previewSelectAction(") == 1
    assert qml.count(allowed_literal) == 1
    assert "paperRuntimeActionDispatchBridge.previewSelectSourceControl" not in qml
    assert "paperRuntimeActionDispatchBridge.resetPreviewSelection" not in qml
    assert ".previewSelectSourceControl(" not in qml
    assert ".resetPreviewSelection(" not in qml
    for forbidden in (
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
    ):
        assert forbidden not in qml
