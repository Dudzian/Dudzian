"""Tests for FUNCTIONAL-PREVIEW-11.7 Block I private endpoint gate contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_private_endpoint_gate import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_KIND,
    PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_8,
    STATUS,
    TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_DECISION,
    TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_STATUS,
    build_preview_testnet_sandbox_private_endpoint_gate,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_private_endpoint_gate.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "testnet_sandbox_private_endpoint_gate_kind",
    "block",
    "step",
    "testnet_sandbox_private_endpoint_gate_status",
    "testnet_sandbox_private_endpoint_gate_decision",
    "ready_for_block_i_8",
    "next_step",
    "next_step_title",
    "public_market_data_probe_preview_reference",
    "private_endpoint_gate_scope",
    "private_endpoint_gate_entries",
    "default_private_endpoint_gate_selection",
    "private_endpoint_gate_summary",
    "private_endpoint_contract_requirements",
    "private_endpoint_gate_matrix",
    "blocked_private_endpoint_capabilities",
    "private_endpoint_gate_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
EXPECTED_ENTRY_FIELDS = [
    "private_endpoint_gate_id",
    "source_public_probe_preview_id",
    "source_capability",
    "display_name",
    "private_endpoint_gate_classification",
    "private_endpoint_surface_type",
    "planned_private_endpoint_categories",
    "required_prior_gate",
    "required_future_risk_gate",
    "required_future_observability_gate",
    "allowed_future_private_read_categories",
    "forbidden_private_endpoint_categories",
    "eligible_for_11_8_closure_audit",
    "private_endpoint_access_allowed_now",
    "private_endpoint_probe_allowed_now",
    "private_endpoint_validation_allowed_now",
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "order_submission_allowed_now",
    "order_generation_allowed_now",
    "fill_simulation_allowed_now",
    "real_market_data_fetch_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "runtime_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
    "gate_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]
READ_CATEGORIES = ["account_read", "balance_read", "positions_read", "orders_read", "fills_read"]
FORBIDDEN_CATEGORIES = [
    "order_submission",
    "order_cancel",
    "order_replace",
    "withdrawal",
    "transfer",
    "deposit_address_generation",
    "live_trading",
    "margin_or_leverage_mutation",
]
EXPECTED_BLOCKED = [
    "real private endpoint access",
    "private endpoint probe",
    "private endpoint validation",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order submission",
    "order generation",
    "order cancel",
    "order replace",
    "withdrawal",
    "transfer",
    "deposit address generation",
    "margin or leverage mutation",
    "live trading",
    "real market data fetch",
    "adapter instantiation",
    "adapter config application",
    "credential read",
    "secret read",
    "secure store read",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "runtime loop",
    "scheduler",
    "QML action dispatch",
    "bridge API changes",
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
    "no sandbox adapter import",
    "no exchange adapter runtime import",
    "no account module import",
    "no secrets module import",
    "no security store import",
    "no filesystem I/O",
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
    "no real market data fetch",
    "no private endpoint access",
    "no account fetch",
    "no balance fetch",
    "no positions fetch",
    "no orders fetch",
    "no fills fetch",
    "no order submission",
    "no order generation",
    "no order cancel",
    "no order replace",
    "no withdrawal",
    "no transfer",
    "no margin/leverage mutation",
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
    "sandbox",
    "exchange",
    "account",
    "secrets",
    "security",
    "network",
    "filesystem",
    "yaml",
    "json",
    "os",
    "pathlib",
    "subprocess",
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
}
FORBIDDEN_CALLS = {
    "open",
    "read_text",
    "write_text",
    "yaml",
    "json",
    "getenv",
    "environ",
    "requests",
    "subprocess",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "getaddrinfo",
    "create_connection",
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
    "cancel_order",
    "replace_order",
    "withdraw",
    "transfer",
    "fetch_market_data",
    "fetch_balance",
    "fetch_account",
    "fetch_positions",
    "fetch_orders",
    "fetch_fills",
    "refresh_market_data",
    "export",
}


def _assert_plain(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            _assert_plain(child)
    elif isinstance(value, list):
        for child in value:
            _assert_plain(child)


def test_private_endpoint_gate_shape_is_plain_and_identified() -> None:
    gate = build_preview_testnet_sandbox_private_endpoint_gate()
    _assert_plain(gate)
    json.dumps(gate, sort_keys=True)
    assert list(gate) == EXPECTED_TOP_LEVEL_FIELDS
    assert gate["schema_version"] == PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_SCHEMA_VERSION
    assert (
        gate["testnet_sandbox_private_endpoint_gate_kind"]
        == PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_KIND
    )
    assert gate["block"] == "I"
    assert gate["step"] == "11.7"
    assert (
        gate["testnet_sandbox_private_endpoint_gate_status"]
        == TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_STATUS
    )
    assert (
        gate["testnet_sandbox_private_endpoint_gate_decision"]
        == TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_DECISION
    )
    assert gate["ready_for_block_i_8"] is READY_FOR_BLOCK_I_8 is True
    assert gate["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.8"
    assert gate["next_step_title"] == NEXT_STEP_TITLE
    assert gate["status"] == STATUS


def test_public_market_data_probe_preview_reference_points_to_11_6() -> None:
    reference = build_preview_testnet_sandbox_private_endpoint_gate()[
        "public_market_data_probe_preview_reference"
    ]
    assert list(reference) == [
        "schema_version",
        "testnet_sandbox_public_market_data_probe_preview_kind",
        "testnet_sandbox_public_market_data_probe_preview_status",
        "testnet_sandbox_public_market_data_probe_preview_decision",
        "ready_for_block_i_7",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert reference["ready_for_block_i_7"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.7"
    assert reference["next_step_title"] == "TESTNET/SANDBOX PRIVATE ENDPOINT GATE"


def test_scope_contract_only_blocks_all_activation_paths() -> None:
    scope = build_preview_testnet_sandbox_private_endpoint_gate()["private_endpoint_gate_scope"]
    assert scope["scope_name"] == "testnet_sandbox_private_endpoint_gate"
    assert scope["private_endpoint_gate_contract_only"] is True
    assert scope["derived_from_public_probe_preview_11_6"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "private_endpoint_gate_contract_only",
            "derived_from_public_probe_preview_11_6",
            "exe_direction_preserved",
        }:
            assert value is False


def test_entry_is_only_exchange_adapter_layer_and_blocks_private_runtime_order_paths() -> None:
    entries = build_preview_testnet_sandbox_private_endpoint_gate()["private_endpoint_gate_entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert list(entry) == EXPECTED_ENTRY_FIELDS
    assert entry == {
        "private_endpoint_gate_id": "private_endpoint_gate_exchange_adapter_layer",
        "source_public_probe_preview_id": "public_probe_preview_exchange_adapter_layer",
        "source_capability": "exchange_adapter_layer",
        "display_name": "Exchange adapter layer private endpoint gate",
        "private_endpoint_gate_classification": "private_endpoint_contract_only",
        "private_endpoint_surface_type": "exchange_adapter_private_endpoint_gate_contract",
        "planned_private_endpoint_categories": READ_CATEGORIES,
        "required_prior_gate": "public_probe_preview_exchange_adapter_layer",
        "required_future_risk_gate": "BLOK J — RISK GOVERNOR / LIMITS / KILL SWITCH",
        "required_future_observability_gate": "BLOK K — OBSERVABILITY / AUDIT / SOAK",
        "allowed_future_private_read_categories": READ_CATEGORIES,
        "forbidden_private_endpoint_categories": FORBIDDEN_CATEGORIES,
        "eligible_for_11_8_closure_audit": True,
        "private_endpoint_access_allowed_now": False,
        "private_endpoint_probe_allowed_now": False,
        "private_endpoint_validation_allowed_now": False,
        "account_fetch_allowed_now": False,
        "balance_fetch_allowed_now": False,
        "positions_fetch_allowed_now": False,
        "orders_fetch_allowed_now": False,
        "fills_fetch_allowed_now": False,
        "order_submission_allowed_now": False,
        "order_generation_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "real_market_data_fetch_allowed_now": False,
        "network_io_allowed_now": False,
        "dns_lookup_allowed_now": False,
        "http_request_allowed_now": False,
        "websocket_allowed_now": False,
        "adapter_instantiation_allowed_now": False,
        "runtime_allowed_now": False,
        "credentials_allowed_now": False,
        "secrets_allowed_now": False,
        "gate_safe_for_offline_tests": True,
        "operator_visibility": "blocked_until_private_endpoint_gate",
        "notes": entry["notes"],
    }
    assert entry["notes"]


def test_selection_summary_requirements_matrix_boundaries_and_evidence() -> None:
    gate = build_preview_testnet_sandbox_private_endpoint_gate()
    assert gate["default_private_endpoint_gate_selection"] == {
        "private_endpoint_gate_id": "private_endpoint_gate_exchange_adapter_layer",
        "source_capability": "exchange_adapter_layer",
        "reason": "only 11.7-eligible public probe preview; private endpoint contract only, no account/balance/positions/orders/fills fetch, no order submission, no network I/O",
        "private_endpoint_access_allowed_now": False,
        "network_io_allowed_now": False,
        "order_submission_allowed_now": False,
    }
    summary = gate["private_endpoint_gate_summary"]
    assert summary["entry_count"] == 1
    assert summary["default_selection_id"] == "private_endpoint_gate_exchange_adapter_layer"
    assert summary["offline_safe_entry_count"] == 1
    assert summary["entries_eligible_for_11_8_closure_audit"] == 1
    assert summary["safe_to_render_in_future_ui_as_read_only"] is True
    assert summary["safe_for_runtime_execution_now"] is False
    assert summary["safe_for_order_execution_now"] is False
    assert summary["safe_for_private_endpoint_access_now"] is False
    for key, value in summary.items():
        if key.endswith("_enabled_entry_count"):
            assert value == 0
    requirements = gate["private_endpoint_contract_requirements"]
    assert list(requirements["planned_private_read_categories_by_source_capability"]) == [
        "exchange_adapter_layer"
    ]
    assert list(requirements["forbidden_private_endpoint_categories_by_source_capability"]) == [
        "exchange_adapter_layer"
    ]
    assert list(requirements["required_future_gates_by_source_capability"]) == [
        "exchange_adapter_layer"
    ]
    assert requirements["global_allowed_future_private_read_categories"] == READ_CATEGORIES
    assert requirements["global_forbidden_private_endpoint_categories"] == FORBIDDEN_CATEGORIES
    for key in (
        "private_endpoint_read_only_contract",
        "private_endpoint_mutation_forbidden",
        "order_submission_forbidden",
        "live_trading_forbidden",
        "risk_governor_required_before_any_order_flow",
        "observability_required_before_any_soak",
    ):
        assert requirements[key] is True
    gate_id_list = ["private_endpoint_gate_exchange_adapter_layer"]
    assert gate["private_endpoint_gate_matrix"] == {
        "private_endpoint_gate_ids": gate_id_list,
        "gates_eligible_for_11_8_closure_audit": gate_id_list,
        "gates_requiring_risk_governor_later": gate_id_list,
        "gates_requiring_observability_soak_later": gate_id_list,
        "gates_never_runtime_enabled_in_11_7": gate_id_list,
        "gates_never_order_enabled_in_11_7": gate_id_list,
        "gates_never_private_endpoint_enabled_in_11_7": gate_id_list,
    }
    assert gate["blocked_private_endpoint_capabilities"] == EXPECTED_BLOCKED
    assert all(gate["private_endpoint_gate_boundaries"].values())
    evidence = gate["non_activation_evidence"]
    assert evidence["public_market_data_probe_preview_11_6_read"] is True
    assert evidence["private_endpoint_gate_built"] is True
    for key, value in evidence.items():
        if key not in {"public_market_data_probe_preview_11_6_read", "private_endpoint_gate_built"}:
            assert value is False
    assert gate["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert gate["future_steps"] == ["functional_preview_11_8_testnet_sandbox_adapter_closure_audit"]


def test_source_imports_and_calls_are_safe() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_testnet_sandbox_public_market_data_probe_preview",
    ]
    imported_roots = {name.split(".")[0] for name in imports}
    assert imported_roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS)
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert calls.isdisjoint(FORBIDDEN_CALLS)


def test_qml_preview_selection_bridge_remains_unchanged() -> None:
    text = QML_SOURCE.read_text(encoding="utf-8")
    allowed = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
    assert text.count("previewSelectAction(") == 1
    assert allowed in text
    assert "paperRuntimeActionDispatchBridge.previewSelectSourceControl" not in text
    assert "paperRuntimeActionDispatchBridge.resetPreviewSelection" not in text
    assert ".previewSelectSourceControl(" not in text
    assert ".resetPreviewSelection(" not in text
    for forbidden in (
        "start_runtime",
        "start_loop",
        "stop_runtime",
        "pause_runtime",
        "resume_runtime",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "cancel_order",
        "replace_order",
        "fetch_market_data",
        "refresh_market_data",
    ):
        assert forbidden not in text
