"""Tests for FUNCTIONAL-PREVIEW-11.6 Block I public market data probe preview."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_public_market_data_probe_preview import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_KIND,
    PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_7,
    STATUS,
    TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_DECISION,
    TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_STATUS,
    build_preview_testnet_sandbox_public_market_data_probe_preview,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_public_market_data_probe_preview.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "testnet_sandbox_public_market_data_probe_preview_kind",
    "block",
    "step",
    "testnet_sandbox_public_market_data_probe_preview_status",
    "testnet_sandbox_public_market_data_probe_preview_decision",
    "ready_for_block_i_7",
    "next_step",
    "next_step_title",
    "credentials_gate_contract_reference",
    "public_probe_preview_scope",
    "public_probe_preview_entries",
    "default_public_probe_preview_selection",
    "public_probe_preview_summary",
    "public_probe_preview_matrix",
    "blocked_public_probe_capabilities",
    "public_probe_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
EXPECTED_ENTRY_FIELDS = [
    "public_probe_preview_id",
    "source_capability",
    "display_name",
    "probe_preview_classification",
    "probe_surface_type",
    "planned_probe_category",
    "planned_symbol_scope",
    "planned_timeframe_scope",
    "planned_rate_limit_profile",
    "required_prior_gate",
    "eligible_for_11_7_private_endpoint_gate",
    "real_probe_allowed_now",
    "real_market_data_fetch_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "runtime_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
    "private_endpoint_allowed_now",
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "order_submission_allowed_now",
    "probe_preview_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]
EXPECTED_BLOCKED = [
    "real public market data probe",
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
    "private endpoint access",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order submission",
    "fill simulation",
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
    "no network I/O",
    "no DNS lookup",
    "no HTTP request",
    "no WebSocket connection",
    "no private endpoint access",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]
EXPECTED_FUTURE_STEPS = [
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
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
    "fetch_market_data",
    "fetch_balance",
    "fetch_account",
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


def test_public_market_data_probe_preview_shape_is_plain_and_identified() -> None:
    preview = build_preview_testnet_sandbox_public_market_data_probe_preview()
    _assert_plain(preview)
    json.dumps(preview, sort_keys=True)
    assert list(preview) == EXPECTED_TOP_LEVEL_FIELDS
    assert (
        preview["schema_version"]
        == PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_SCHEMA_VERSION
    )
    assert (
        preview["testnet_sandbox_public_market_data_probe_preview_kind"]
        == PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_KIND
    )
    assert preview["block"] == "I"
    assert preview["step"] == "11.6"
    assert (
        preview["testnet_sandbox_public_market_data_probe_preview_status"]
        == TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_STATUS
    )
    assert (
        preview["testnet_sandbox_public_market_data_probe_preview_decision"]
        == TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_DECISION
    )
    assert preview["ready_for_block_i_7"] is READY_FOR_BLOCK_I_7 is True
    assert preview["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.7"
    assert preview["next_step_title"] == NEXT_STEP_TITLE == "TESTNET/SANDBOX PRIVATE ENDPOINT GATE"
    assert preview["status"] == STATUS


def test_credentials_gate_reference_points_to_11_5() -> None:
    reference = build_preview_testnet_sandbox_public_market_data_probe_preview()[
        "credentials_gate_contract_reference"
    ]
    assert set(reference) == {
        "schema_version",
        "testnet_sandbox_credentials_gate_contract_kind",
        "testnet_sandbox_credentials_gate_contract_status",
        "testnet_sandbox_credentials_gate_contract_decision",
        "ready_for_block_i_6",
        "next_step",
        "next_step_title",
        "status",
    }
    assert reference["ready_for_block_i_6"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.6"
    assert reference["next_step_title"] == "TESTNET/SANDBOX PUBLIC MARKET DATA PROBE PREVIEW"


def test_scope_preview_only_blocks_all_activation_paths() -> None:
    scope = build_preview_testnet_sandbox_public_market_data_probe_preview()[
        "public_probe_preview_scope"
    ]
    assert scope["scope_name"] == "testnet_sandbox_public_market_data_probe_preview"
    assert scope["probe_preview_only"] is True
    assert scope["derived_from_credentials_gate_11_5"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "probe_preview_only",
            "derived_from_credentials_gate_11_5",
            "exe_direction_preserved",
        }:
            assert value is False


def test_entries_order_fields_values_and_flags() -> None:
    entries = build_preview_testnet_sandbox_public_market_data_probe_preview()[
        "public_probe_preview_entries"
    ]
    assert [entry["source_capability"] for entry in entries] == [
        "read_only_market_data_provider",
        "exchange_adapter_layer",
    ]
    assert [list(entry) for entry in entries] == [EXPECTED_ENTRY_FIELDS, EXPECTED_ENTRY_FIELDS]
    assert entries[0] | {} == {
        **entries[0],
        "public_probe_preview_id": "public_probe_preview_read_only_market_data_provider",
        "display_name": "Read-only market data provider public probe preview",
        "probe_preview_classification": "lowest_risk_public_market_data_probe_preview",
        "probe_surface_type": "public_market_data_probe_preview",
        "planned_probe_category": "public_market_data",
        "planned_rate_limit_profile": "public_read_only_preview",
        "required_prior_gate": "adapter_config_gate_read_only_market_data_provider",
        "eligible_for_11_7_private_endpoint_gate": False,
        "operator_visibility": "read_only_future",
    }
    assert entries[1] | {} == {
        **entries[1],
        "public_probe_preview_id": "public_probe_preview_exchange_adapter_layer",
        "display_name": "Exchange adapter layer public probe preview",
        "probe_preview_classification": "exchange_adapter_public_market_data_probe_preview",
        "probe_surface_type": "exchange_adapter_public_probe_preview",
        "planned_probe_category": "exchange_public_market_data",
        "planned_rate_limit_profile": "exchange_adapter_public_preview_guarded",
        "required_prior_gate": "credentials_gate_exchange_adapter_layer",
        "eligible_for_11_7_private_endpoint_gate": True,
        "operator_visibility": "blocked_until_probe_gate",
    }
    for entry in entries:
        assert entry["planned_symbol_scope"] == ["BTC/USDT", "ETH/USDT"]
        assert entry["planned_timeframe_scope"] == ["1m", "5m"]
        assert entry["probe_preview_safe_for_offline_tests"] is True
        assert entry["notes"]
        for key in EXPECTED_ENTRY_FIELDS:
            if key.endswith("_allowed_now"):
                assert entry[key] is False


def test_selection_summary_matrix_boundaries_and_evidence() -> None:
    preview = build_preview_testnet_sandbox_public_market_data_probe_preview()
    assert preview["default_public_probe_preview_selection"] == {
        "public_probe_preview_id": "public_probe_preview_read_only_market_data_provider",
        "source_capability": "read_only_market_data_provider",
        "reason": "lowest-risk public market data probe preview; no real fetch, no network I/O, no credentials, no private endpoint, no runtime activation",
        "real_probe_allowed_now": False,
        "real_market_data_fetch_allowed_now": False,
        "network_io_allowed_now": False,
    }
    summary = preview["public_probe_preview_summary"]
    assert summary["entry_count"] == 2
    assert summary["offline_safe_entry_count"] == 2
    assert summary["entries_eligible_for_11_7_private_endpoint_gate"] == 1
    assert summary["safe_to_render_in_future_ui_as_read_only"] is True
    assert summary["safe_for_runtime_execution_now"] is False
    for key, value in summary.items():
        if key.endswith("_enabled_entry_count"):
            assert value == 0
    matrix = preview["public_probe_preview_matrix"]
    assert matrix == {
        "public_probe_preview_ids": [
            "public_probe_preview_read_only_market_data_provider",
            "public_probe_preview_exchange_adapter_layer",
        ],
        "public_market_data_only_preview_ids": [
            "public_probe_preview_read_only_market_data_provider"
        ],
        "exchange_adapter_public_probe_preview_ids": [
            "public_probe_preview_exchange_adapter_layer"
        ],
        "previews_eligible_for_private_endpoint_gate_later": [
            "public_probe_preview_exchange_adapter_layer"
        ],
        "previews_never_runtime_enabled_in_11_6": [
            "public_probe_preview_read_only_market_data_provider",
            "public_probe_preview_exchange_adapter_layer",
        ],
        "planned_symbols_by_preview_id": {
            "public_probe_preview_read_only_market_data_provider": ["BTC/USDT", "ETH/USDT"],
            "public_probe_preview_exchange_adapter_layer": ["BTC/USDT", "ETH/USDT"],
        },
        "planned_timeframes_by_preview_id": {
            "public_probe_preview_read_only_market_data_provider": ["1m", "5m"],
            "public_probe_preview_exchange_adapter_layer": ["1m", "5m"],
        },
    }
    assert preview["blocked_public_probe_capabilities"] == EXPECTED_BLOCKED
    assert all(preview["public_probe_boundaries"].values())
    evidence = preview["non_activation_evidence"]
    assert evidence["credentials_gate_contract_11_5_read"] is True
    assert evidence["public_market_data_probe_preview_built"] is True
    for key, value in evidence.items():
        if key not in {
            "credentials_gate_contract_11_5_read",
            "public_market_data_probe_preview_built",
        }:
            assert value is False
    assert preview["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert preview["future_steps"] == EXPECTED_FUTURE_STEPS


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
        "ui.pyside_app.preview_testnet_sandbox_credentials_gate_contract",
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
        "submit_order",
        "fill_order",
        "fetch_market_data",
    ):
        assert forbidden not in text
