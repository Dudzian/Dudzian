from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_read_model import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_READ_MODEL_DECISION,
    READ_ONLY_MARKET_DATA_READ_MODEL_STATUS,
    build_preview_read_only_market_data_read_model,
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


def test_read_model_is_plain_json_serializable_dict() -> None:
    model = build_preview_read_only_market_data_read_model()

    assert isinstance(model, dict)
    assert model["schema_version"] == PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_SCHEMA_VERSION
    assert json.loads(json.dumps(model)) == model


def test_status_decision_next_step_and_contract_reference_are_correct() -> None:
    model = build_preview_read_only_market_data_read_model()

    assert model["block"] == "H"
    assert model["step"] == "10.1"
    assert model["market_data_read_model_status"] == READ_ONLY_MARKET_DATA_READ_MODEL_STATUS
    assert model["market_data_read_model_decision"] == READ_ONLY_MARKET_DATA_READ_MODEL_DECISION
    assert model["ready_for_block_h_2"] is True
    assert model["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.2"
    assert model["next_step_title"] == NEXT_STEP_TITLE == "READ-ONLY MARKET DATA STATIC FIXTURE"
    assert (
        model["status"] == "ready_for_functional_preview_10_2_read_only_market_data_static_fixture"
    )

    assert model["contract_reference"] == {
        "schema_version": "preview_read_only_market_data_adapter_contract.v1",
        "market_data_adapter_contract_kind": "functional_preview_block_h_read_only_market_data_adapter_contract",
        "market_data_adapter_contract_status": "read_only_market_data_adapter_contract_ready_no_network_io",
        "market_data_adapter_contract_decision": "START_BLOCK_H_WITH_CONTRACT_ONLY_NO_MARKET_DATA_FETCH",
        "ready_for_block_h_1": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.1",
        "next_step_title": "READ-ONLY MARKET DATA ADAPTER READ MODEL",
        "status": "ready_for_functional_preview_10_1_read_only_market_data_adapter_read_model",
    }


def test_read_model_scope_is_read_model_only_and_no_network() -> None:
    scope = build_preview_read_only_market_data_read_model()["read_model_scope"]
    true_keys = {
        "read_model_only",
        "contract_reference_required",
        "fixture_data_allowed_next_step",
        "recorded_replay_allowed_future_only",
        "public_market_data_allowed_future_only",
    }

    assert scope["scope_name"] == "read_only_market_data_read_model"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_keys)


def test_market_data_read_model_is_static_empty_and_non_executable() -> None:
    read_model = build_preview_read_only_market_data_read_model()["market_data_read_model"]

    assert read_model["read_model_available"] is True
    assert read_model["read_model_static_only"] is True
    assert read_model["read_model_executable"] is False
    assert read_model["read_model_name"] == "read_only_market_data_preview_read_model"
    assert read_model["data_source_mode"] == "not_connected_no_network_io"
    assert read_model["rows_available_now"] is False
    assert read_model["row_count"] == 0
    assert read_model["sample_rows"] == []
    assert (
        read_model["empty_state_reason"] == "no_market_data_fixture_until_functional_preview_10_2"
    )
    assert read_model["allowed_row_fields"] == _ALLOWED_FIELDS
    assert read_model["forbidden_row_fields"] == _DENIED_FIELDS
    for key in (
        "network_io_allowed_now",
        "market_fetch_performed",
        "runtime_execution_allowed",
        "trading_execution_allowed",
        "account_access_allowed",
        "credentials_access_allowed",
    ):
        assert read_model[key] is False


def test_allowlist_and_denylist_are_complete_public_private_and_disjoint() -> None:
    model = build_preview_read_only_market_data_read_model()

    assert model["market_data_field_allowlist"] == _ALLOWED_FIELDS
    assert model["market_data_private_field_denylist"] == _DENIED_FIELDS
    assert set(_ALLOWED_FIELDS).isdisjoint(_DENIED_FIELDS)


def test_quality_preview_is_static_only_and_not_evaluated_no_data() -> None:
    quality = build_preview_read_only_market_data_read_model()["read_model_quality_preview"]

    assert quality == {
        "quality_preview_available": True,
        "quality_static_only": True,
        "quality_executable": False,
        "freshness_status": "not_evaluated_no_data",
        "latency_status": "not_evaluated_no_data",
        "schema_status": "defined_by_contract_only",
        "source_status": "not_connected",
        "row_validation_allowed_next_step": True,
        "market_data_fetch_required_for_quality": False,
        "network_required_now": False,
        "account_required_now": False,
    }


def test_no_fetch_no_execution_evidence_has_only_read_model_and_contract_true() -> None:
    evidence = build_preview_read_only_market_data_read_model()["no_fetch_no_execution_evidence"]

    assert evidence["read_model_evaluated"] is True
    assert evidence["contract_read"] is True
    for key, value in evidence.items():
        if key not in {"read_model_evaluated", "contract_read"}:
            assert value is False


def test_boundary_checks_block_network_account_order_fill_runtime_live_and_export() -> None:
    boundary = build_preview_read_only_market_data_read_model()["boundary_checks"]
    true_keys = {
        "local_only",
        "block_h_read_model_only",
        "contract_reference_valid",
        "read_only_market_data_scope_defined",
        "fixture_data_allowed_next_step",
        "public_market_data_allowed_future_only",
        "recorded_fixture_allowed_next_step",
        "local_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_2",
    }

    for key, value in boundary.items():
        assert value is (key in true_keys)


def test_capability_lists_source_boundaries_and_future_steps_are_complete() -> None:
    model = build_preview_read_only_market_data_read_model()

    assert model["blocked_capabilities"] == [
        "market data adapter implementation now",
        "network I/O",
        "live market data fetch now",
        "exchange API connection",
        "private account endpoint access",
        "account balance fetch",
        "positions fetch",
        "orders fetch",
        "fills fetch",
        "read model population from market data now",
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
    assert model["source_boundaries"] == [
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
    assert model["future_steps"] == [
        "functional_preview_10_2_read_only_market_data_static_fixture",
        "functional_preview_10_3_read_only_market_data_audit_envelope",
        "functional_preview_10_4_read_only_market_data_ui_read_only_surface",
    ]


def test_read_model_source_imports_only_safe_stdlib_and_10_0_contract() -> None:
    source_path = Path("ui/pyside_app/preview_read_only_market_data_read_model.py")
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
        "ui.pyside_app.preview_read_only_market_data_adapter_contract",
    ]


def test_read_model_source_has_no_forbidden_imports_or_calls() -> None:
    source = Path("ui/pyside_app/preview_read_only_market_data_read_model.py").read_text(
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
        "ui.pyside_app.preview_read_only_market_data_adapter_contract",
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
