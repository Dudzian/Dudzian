from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_audit_envelope import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_DECISION,
    READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_STATUS,
    build_preview_read_only_market_data_audit_envelope,
)

EXPECTED_EVENTS = [
    (
        "market-data-audit-0001-btc-usdt-static-fixture",
        "BTC/USDT",
        "static_fixture",
        "ok_preview_only",
    ),
    (
        "market-data-audit-0002-eth-usdt-static-fixture",
        "ETH/USDT",
        "static_fixture",
        "ok_preview_only",
    ),
    (
        "market-data-audit-0003-sol-usdt-low-liquidity-preview",
        "SOL/USDT",
        "static_fixture_low_liquidity_preview",
        "low_liquidity_preview_only",
    ),
    (
        "market-data-audit-0004-ada-usdt-stale-preview",
        "ADA/USDT",
        "static_fixture_stale_preview",
        "stale_preview_only",
    ),
]
ALLOWLIST = ["symbol", "timestamp", "bid", "ask", "last", "volume", "source", "latency_ms_preview"]
DENYLIST = [
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
]


def test_audit_envelope_is_plain_json_serializable_dict() -> None:
    envelope = build_preview_read_only_market_data_audit_envelope()

    assert isinstance(envelope, dict)
    assert envelope["schema_version"] == PREVIEW_READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_SCHEMA_VERSION
    assert json.loads(json.dumps(envelope)) == envelope


def test_status_decision_next_step_and_static_fixture_reference_are_correct() -> None:
    envelope = build_preview_read_only_market_data_audit_envelope()

    assert envelope["block"] == "H"
    assert envelope["step"] == "10.3"
    assert (
        envelope["market_data_audit_envelope_status"] == READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_STATUS
    )
    assert (
        envelope["market_data_audit_envelope_decision"]
        == READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_DECISION
    )
    assert envelope["ready_for_block_h_4"] is True
    assert envelope["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.4"
    assert (
        envelope["next_step_title"]
        == NEXT_STEP_TITLE
        == "READ-ONLY MARKET DATA UI READ-ONLY SURFACE"
    )
    assert (
        envelope["status"]
        == "ready_for_functional_preview_10_4_read_only_market_data_ui_read_only_surface"
    )
    assert envelope["static_fixture_reference"] == {
        "schema_version": "preview_read_only_market_data_static_fixture.v1",
        "market_data_static_fixture_kind": "functional_preview_block_h_read_only_market_data_static_fixture",
        "market_data_static_fixture_status": "read_only_market_data_static_fixture_ready_no_market_data_fetch",
        "market_data_static_fixture_decision": "BUILD_STATIC_FIXTURE_ONLY_NO_NETWORK_IO_NO_MARKET_DATA_FETCH",
        "ready_for_block_h_3": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.3",
        "next_step_title": "READ-ONLY MARKET DATA AUDIT ENVELOPE",
        "status": "ready_for_functional_preview_10_3_read_only_market_data_audit_envelope",
    }


def test_audit_envelope_scope_is_audit_only_and_no_network_no_export() -> None:
    scope = build_preview_read_only_market_data_audit_envelope()["audit_envelope_scope"]
    true_keys = {
        "audit_envelope_only",
        "static_fixture_reference_required",
        "fixture_data_read_allowed_now",
        "fixture_row_audit_allowed_now",
        "field_boundary_audit_allowed_now",
        "data_quality_audit_preview_allowed_now",
        "ui_surface_allowed_next_step",
    }

    assert scope["scope_name"] == "read_only_market_data_audit_envelope"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_keys)


def test_market_data_audit_envelope_metadata_is_static_only_non_executable_four_events() -> None:
    metadata = build_preview_read_only_market_data_audit_envelope()["market_data_audit_envelope"]

    assert metadata["audit_envelope_available"] is True
    assert metadata["audit_envelope_static_only"] is True
    assert metadata["audit_envelope_executable"] is False
    assert metadata["audit_envelope_name"] == "read_only_market_data_preview_audit_envelope"
    assert metadata["audit_source_mode"] == "local_static_fixture_audit_no_network_io"
    assert metadata["audited_fixture_available"] is True
    assert metadata["audited_fixture_row_count"] == 4
    assert metadata["audit_event_count"] == 4
    assert metadata["audit_source"] == "functional_preview_static_fixture"
    assert metadata["audit_version"] == "10.3"
    false_keys = set(metadata) - {
        "audit_envelope_available",
        "audit_envelope_static_only",
        "audit_envelope_executable",
        "audit_envelope_name",
        "audit_source_mode",
        "audited_fixture_available",
        "audited_fixture_row_count",
        "audit_event_count",
        "audit_source",
        "audit_version",
    }
    for key in false_keys:
        assert metadata[key] is False


def test_audit_events_are_exact_and_have_all_execution_flags_false() -> None:
    events = build_preview_read_only_market_data_audit_envelope()["audit_events"]

    assert len(events) == 4
    for event, (event_id, symbol, source, quality_status) in zip(
        events, EXPECTED_EVENTS, strict=True
    ):
        assert event["event_id"] == event_id
        assert event["symbol"] == symbol
        assert event["event_type"] == "read_only_market_data_fixture_row_audited"
        assert event["source"] == source
        assert event["quality_status"] == quality_status
        assert event["field_boundary_status"] == "allowlist_only_private_fields_absent"
        for key in (
            "network_io_performed",
            "market_data_fetch_performed",
            "account_data_present",
            "order_or_fill_data_present",
            "credentials_or_secrets_present",
            "exported",
        ):
            assert event[key] is False


def test_fixture_row_summary_field_boundary_and_quality_audits_are_complete() -> None:
    envelope = build_preview_read_only_market_data_audit_envelope()
    summary = envelope["fixture_row_audit_summary"]
    field_boundary = envelope["field_boundary_audit"]
    quality = envelope["data_quality_audit_preview"]

    assert summary == {
        "row_audit_available": True,
        "row_audit_static_only": True,
        "row_audit_executable": False,
        "row_count": 4,
        "audited_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        "normal_preview_symbols": ["BTC/USDT", "ETH/USDT"],
        "low_liquidity_preview_symbols": ["SOL/USDT"],
        "stale_preview_symbols": ["ADA/USDT"],
        "all_rows_audited": True,
        "all_rows_allowlist_only": True,
        "private_fields_absent": True,
        "account_fields_absent": True,
        "order_fields_absent": True,
        "fill_fields_absent": True,
        "credentials_fields_absent": True,
        "network_required_for_audit": False,
        "market_data_fetch_required_for_audit": False,
        "export_required_for_audit": False,
    }
    assert field_boundary["allowlist_fields"] == ALLOWLIST
    assert field_boundary["denylist_fields"] == DENYLIST
    for key, value in field_boundary.items():
        if key in {"allowlist_fields", "denylist_fields"}:
            continue
        assert value is (
            key
            in {
                "field_boundary_audit_available",
                "field_boundary_static_only",
                "allowlist_denylist_disjoint",
                "fixture_rows_match_allowlist",
            }
        )
    assert quality == {
        "quality_audit_available": True,
        "quality_audit_static_only": True,
        "quality_audit_executable": False,
        "row_count": 4,
        "normal_preview_count": 2,
        "low_liquidity_preview_count": 1,
        "stale_preview_count": 1,
        "all_rows_have_bid_less_than_or_equal_ask": True,
        "all_rows_have_last_within_bid_ask": True,
        "all_rows_have_non_negative_volume": True,
        "all_rows_have_non_negative_latency": True,
        "market_data_fetch_required_for_quality": False,
        "network_required_now": False,
        "account_required_now": False,
    }


def test_evidence_and_boundary_checks_block_execution_network_account_order_fill_runtime_export() -> (
    None
):
    envelope = build_preview_read_only_market_data_audit_envelope()
    evidence = envelope["no_fetch_no_export_no_execution_evidence"]
    boundary = envelope["boundary_checks"]

    for key, value in evidence.items():
        assert value is (
            key
            in {
                "audit_envelope_evaluated",
                "static_fixture_read",
                "audit_events_built_from_static_literals",
            }
        )
    true_boundary = {
        "local_only",
        "block_h_audit_envelope_only",
        "static_fixture_reference_valid",
        "read_only_market_data_scope_defined",
        "audit_envelope_available_now",
        "audit_envelope_static_only",
        "public_market_data_allowed_future_only",
        "recorded_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_4",
    }
    for key, value in boundary.items():
        assert value is (key in true_boundary)


def test_capability_lists_source_boundaries_and_future_steps_are_complete() -> None:
    envelope = build_preview_read_only_market_data_audit_envelope()

    assert envelope["blocked_capabilities"] == [
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
        "audit export",
        "TradingController / DecisionEnvelope",
        "live/testnet/account/secrets/export/cloud",
        "QML changes / new QML calls",
        "EXE packaging",
    ]
    assert envelope["source_boundaries"] == [
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
    assert envelope["future_steps"] == [
        "functional_preview_10_4_read_only_market_data_ui_read_only_surface",
        "functional_preview_10_5_read_only_market_data_selection_gate",
        "functional_preview_10_6_read_only_market_data_controlled_refresh_preview",
    ]


def test_audit_envelope_source_imports_only_safe_stdlib_and_10_2_static_fixture() -> None:
    source_path = Path("ui/pyside_app/preview_read_only_market_data_audit_envelope.py")
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
        "ui.pyside_app.preview_read_only_market_data_static_fixture",
    ]


def test_audit_envelope_source_has_no_forbidden_imports_or_calls() -> None:
    source = Path("ui/pyside_app/preview_read_only_market_data_audit_envelope.py").read_text(
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
        "export",
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
