from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_adapter_contract import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_DECISION,
    READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_STATUS,
    build_preview_read_only_market_data_adapter_contract,
)


def test_contract_is_plain_json_serializable_dict() -> None:
    contract = build_preview_read_only_market_data_adapter_contract()

    assert isinstance(contract, dict)
    assert (
        contract["schema_version"] == PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_SCHEMA_VERSION
    )
    assert json.loads(json.dumps(contract)) == contract


def test_status_decision_next_step_and_block_g_handoff_are_correct() -> None:
    contract = build_preview_read_only_market_data_adapter_contract()

    assert contract["block"] == "H"
    assert contract["step"] == "10.0"
    assert (
        contract["market_data_adapter_contract_status"]
        == READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_STATUS
    )
    assert (
        contract["market_data_adapter_contract_decision"]
        == READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_DECISION
    )
    assert contract["ready_for_block_h_1"] is True
    assert contract["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.1"
    assert (
        contract["next_step_title"] == NEXT_STEP_TITLE == "READ-ONLY MARKET DATA ADAPTER READ MODEL"
    )
    assert (
        contract["status"]
        == "ready_for_functional_preview_10_1_read_only_market_data_adapter_read_model"
    )

    handoff = contract["block_g_handoff_reference"]
    assert handoff == {
        "schema_version": "preview_block_g_closure_audit.v1",
        "closure_audit_kind": "functional_preview_block_g_closure_audit",
        "block_g_closure_status": "block_g_closure_audit_complete_ready_for_block_h",
        "block_g_closure_decision": "CLOSE_BLOCK_G_PAPER_ONLY_DECISION_TO_ORDER_PATH_NO_RUNTIME_EXECUTION",
        "ready_for_block_h": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.0",
        "next_step_title": "BLOK H — READ-ONLY MARKET DATA ADAPTER CONTRACT",
        "status": "ready_for_functional_preview_10_0_block_h_read_only_market_data_contract",
    }


def test_read_only_market_data_scope_is_contract_only_and_no_network() -> None:
    scope = build_preview_read_only_market_data_adapter_contract()["read_only_market_data_scope"]

    true_keys = {
        "contract_only",
        "fixture_market_data_allowed_in_future",
        "recorded_market_data_replay_allowed_in_future",
        "read_only_exchange_market_data_allowed_in_future",
    }
    assert scope["scope_name"] == "read_only_market_data"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (key in true_keys)


def test_adapter_contract_is_static_only_non_executable_and_defines_field_boundaries() -> None:
    adapter = build_preview_read_only_market_data_adapter_contract()["adapter_contract"]

    assert adapter["contract_available"] is True
    assert adapter["contract_static_only"] is True
    assert adapter["contract_executable"] is False
    assert adapter["adapter_name"] == "read_only_market_data_adapter_preview_contract"
    assert adapter["adapter_mode"] == "contract_only_no_network_io"
    assert set(adapter["allowed_future_input_types"]) == {
        "recorded_fixture",
        "local_replay",
        "read_only_public_market_data",
    }
    assert set(adapter["forbidden_input_types"]) == {
        "live_account",
        "live_balance",
        "live_orders",
        "live_fills",
        "private_account_endpoint",
        "credential_backed_private_api",
    }
    private_fields = {
        "account_id",
        "balance",
        "position",
        "order_id",
        "fill_id",
        "api_key",
        "secret",
        "private_endpoint",
    }
    assert set(adapter["allowed_future_market_data_fields"]) == {
        "symbol",
        "timestamp",
        "bid",
        "ask",
        "last",
        "volume",
        "source",
        "latency_ms_preview",
    }
    assert private_fields.isdisjoint(adapter["allowed_future_market_data_fields"])
    assert set(adapter["forbidden_fields"]) == private_fields
    for key in (
        "network_io_allowed_now",
        "market_fetch_performed",
        "runtime_execution_allowed",
        "trading_execution_allowed",
        "account_access_allowed",
        "credentials_access_allowed",
    ):
        assert adapter[key] is False


def test_no_network_no_execution_evidence_has_only_contract_evaluated_true() -> None:
    evidence = build_preview_read_only_market_data_adapter_contract()[
        "no_network_no_execution_evidence"
    ]

    assert evidence["contract_evaluated"] is True
    for key, value in evidence.items():
        if key != "contract_evaluated":
            assert value is False


def test_boundary_checks_block_execution_network_account_order_fill_runtime_live_and_export() -> (
    None
):
    boundary = build_preview_read_only_market_data_adapter_contract()["boundary_checks"]
    true_keys = {
        "local_only",
        "paper_only_previous_block_preserved",
        "block_h_contract_only",
        "read_only_market_data_scope_defined",
        "public_market_data_allowed_future_only",
        "recorded_fixture_allowed_future_only",
        "local_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_1",
    }

    for key, value in boundary.items():
        assert value is (key in true_keys)


def test_capability_lists_source_boundaries_and_future_steps_are_complete() -> None:
    contract = build_preview_read_only_market_data_adapter_contract()

    assert set(contract["allowed_market_data_capabilities"]) == {
        "define read-only market data adapter contract",
        "define future recorded fixture input shape",
        "define future local replay input shape",
        "define future public read-only market data shape",
        "define market data field allowlist",
        "define market data private-field denylist",
        "prepare read-only market data read model next step",
    }
    assert set(contract["blocked_capabilities"]) == {
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
    }
    assert set(contract["source_boundaries"]) == {
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
    }
    assert contract["future_steps"] == [
        "functional_preview_10_1_read_only_market_data_adapter_read_model",
        "functional_preview_10_2_read_only_market_data_static_fixture",
        "functional_preview_10_3_read_only_market_data_audit_envelope",
    ]


def test_contract_source_imports_only_safe_stdlib_and_block_g_closure_audit() -> None:
    source_path = Path("ui/pyside_app/preview_read_only_market_data_adapter_contract.py")
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
        "ui.pyside_app.preview_block_g_closure_audit",
    ]


def test_contract_source_has_no_forbidden_imports_or_calls() -> None:
    source = Path("ui/pyside_app/preview_read_only_market_data_adapter_contract.py").read_text(
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
        "ui.pyside_app.preview_block_g_closure_audit",
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
