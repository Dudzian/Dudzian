"""Tests for FUNCTIONAL-PREVIEW-11.5 Block I credentials gate contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_credentials_gate_contract import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_KIND,
    PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_6,
    STATUS,
    TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_DECISION,
    TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_STATUS,
    build_preview_testnet_sandbox_credentials_gate_contract,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_credentials_gate_contract.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_credentials_gate_contract_kind",
    "block",
    "step",
    "testnet_sandbox_credentials_gate_contract_status",
    "testnet_sandbox_credentials_gate_contract_decision",
    "ready_for_block_i_6",
    "next_step",
    "next_step_title",
    "adapter_config_gate_reference",
    "credentials_gate_scope",
    "credentials_gate_entries",
    "default_credentials_gate_selection",
    "credentials_gate_summary",
    "credential_contract_requirements",
    "credential_gate_matrix",
    "blocked_credentials_gate_capabilities",
    "credentials_gate_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_ENTRY_FIELDS = {
    "credentials_gate_id",
    "source_config_gate_id",
    "source_capability",
    "display_name",
    "credentials_gate_classification",
    "credential_surface_type",
    "required_credential_reference_shape",
    "optional_credential_reference_shape",
    "forbidden_credential_material",
    "allowed_credential_reference_material",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "credential_secret_read_allowed_now",
    "credential_discovery_allowed_now",
    "credential_validation_allowed_now",
    "credential_material_handling_allowed_now",
    "secret_material_handling_allowed_now",
    "environment_variable_read_allowed_now",
    "config_file_read_allowed_now",
    "secure_store_read_allowed_now",
    "secure_store_write_allowed_now",
    "api_key_value_allowed_in_payload",
    "api_secret_value_allowed_in_payload",
    "passphrase_value_allowed_in_payload",
    "raw_secret_allowed_in_payload",
    "account_identifier_allowed_in_payload",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "runtime_allowed_now",
    "gate_safe_for_offline_tests",
    "operator_visibility",
    "notes",
}
EXPECTED_FORBIDDEN_MATERIAL = [
    "api_key",
    "api_key_value",
    "api_secret",
    "api_secret_value",
    "passphrase",
    "passphrase_value",
    "raw_secret",
    "private_key",
    "mnemonic",
    "account_id",
    "account_number",
    "wallet_address",
    "live_endpoint_override",
    "order_submission_enabled",
]
EXPECTED_BLOCKED = [
    "real credential read",
    "credential discovery",
    "credential validation",
    "credential material handling",
    "secret material handling",
    "environment variable read",
    "real config file read",
    "secure store read",
    "secure store write",
    "API key value in payload",
    "API secret value in payload",
    "passphrase value in payload",
    "raw secret in payload",
    "account identifier in payload",
    "adapter instantiation",
    "adapter config application",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "private endpoint access",
    "public market data fetch",
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
EXPECTED_FUTURE_STEPS = [
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
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


def _assert_plain(value):
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_plain(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain(item)


def test_contract_is_plain_json_serializable_and_identity():
    contract = build_preview_testnet_sandbox_credentials_gate_contract()
    _assert_plain(contract)
    json.dumps(contract, sort_keys=True)
    assert set(contract) == EXPECTED_TOP_LEVEL_FIELDS
    assert (
        contract["schema_version"]
        == PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_SCHEMA_VERSION
    )
    assert (
        contract["testnet_sandbox_credentials_gate_contract_kind"]
        == PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_KIND
    )
    assert contract["block"] == "I"
    assert contract["step"] == "11.5"
    assert (
        contract["testnet_sandbox_credentials_gate_contract_status"]
        == TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_STATUS
    )
    assert (
        contract["testnet_sandbox_credentials_gate_contract_decision"]
        == TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_DECISION
    )
    assert contract["ready_for_block_i_6"] is READY_FOR_BLOCK_I_6
    assert contract["next_step"] == NEXT_STEP
    assert contract["next_step_title"] == NEXT_STEP_TITLE
    assert contract["status"] == STATUS


def test_adapter_config_gate_reference_points_to_11_4():
    reference = build_preview_testnet_sandbox_credentials_gate_contract()[
        "adapter_config_gate_reference"
    ]
    assert reference["ready_for_block_i_5"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.5"
    assert reference["next_step_title"] == "TESTNET/SANDBOX CREDENTIALS GATE CONTRACT"


def test_scope_entries_summary_and_default_selection_are_locked_down():
    contract = build_preview_testnet_sandbox_credentials_gate_contract()
    scope = contract["credentials_gate_scope"]
    assert scope["scope_name"] == "testnet_sandbox_credentials_gate_contract"
    assert scope["credentials_gate_contract_only"] is True
    assert scope["derived_from_adapter_config_gate_11_4"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "credentials_gate_contract_only",
            "derived_from_adapter_config_gate_11_4",
            "exe_direction_preserved",
        }:
            assert value is False
    assert scope["exe_direction_preserved"] is True

    entries = contract["credentials_gate_entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == EXPECTED_ENTRY_FIELDS
    assert entry["credentials_gate_id"] == "credentials_gate_exchange_adapter_layer"
    assert entry["source_config_gate_id"] == "adapter_config_gate_exchange_adapter_layer"
    assert entry["source_capability"] == "exchange_adapter_layer"
    assert {item["source_capability"] for item in entries} == {"exchange_adapter_layer"}
    assert "read_only_market_data_provider" not in {item["source_capability"] for item in entries}
    assert "exchange_network_guard" not in {item["source_capability"] for item in entries}
    for key, value in entry.items():
        if key.endswith("_allowed_now") or key.endswith("_allowed_in_payload"):
            assert value is False
    assert entry["private_endpoint_allowed_now"] is False
    assert entry["order_submission_allowed_now"] is False
    assert entry["runtime_allowed_now"] is False
    assert entry["gate_safe_for_offline_tests"] is True
    assert entry["operator_visibility"] == "blocked_until_credentials_gate"
    assert entry["notes"]

    assert contract["default_credentials_gate_selection"] == {
        "credentials_gate_id": "credentials_gate_exchange_adapter_layer",
        "source_capability": "exchange_adapter_layer",
        "reason": "only 11.5-eligible config gate; credential references only, no secret read, no env read, no network I/O",
        "credential_secret_read_allowed_now": False,
        "network_io_allowed_now": False,
    }
    summary = contract["credentials_gate_summary"]
    assert summary["entry_count"] == 1
    assert summary["offline_safe_entry_count"] == 1
    assert summary["entries_eligible_for_11_6_public_market_data_probe_preview"] == 1
    assert summary["entries_eligible_for_11_7_private_endpoint_gate"] == 1
    for key, value in summary.items():
        if key.endswith("_enabled_entry_count") or key.endswith("_payload_enabled_entry_count"):
            assert value == 0
    assert summary["safe_to_render_in_future_ui_as_read_only"] is True
    assert summary["safe_for_runtime_execution_now"] is False


def test_requirements_matrix_boundaries_and_evidence_are_exact():
    contract = build_preview_testnet_sandbox_credentials_gate_contract()
    requirements = contract["credential_contract_requirements"]
    assert set(requirements["required_reference_shapes_by_source_capability"]) == {
        "exchange_adapter_layer"
    }
    assert set(requirements["forbidden_material_by_source_capability"]) == {
        "exchange_adapter_layer"
    }
    assert set(requirements["allowed_reference_material_by_source_capability"]) == {
        "exchange_adapter_layer"
    }
    assert requirements["global_required_reference_fields"] == [
        "credential_profile_reference",
        "credential_scope",
        "environment",
        "exchange_id",
        "redaction_policy",
    ]
    assert requirements["global_forbidden_credential_material"] == EXPECTED_FORBIDDEN_MATERIAL
    assert requirements["credential_redaction_required"] is True
    assert requirements["credential_logging_forbidden"] is True
    assert requirements["credential_payload_secret_values_forbidden"] is True
    assert requirements["credential_reference_only"] is True

    gate_id = "credentials_gate_exchange_adapter_layer"
    assert contract["credential_gate_matrix"] == {
        "credential_gate_ids": [gate_id],
        "gates_requiring_secret_store_later": [gate_id],
        "gates_eligible_for_public_market_data_probe_preview_later": [gate_id],
        "gates_eligible_for_private_endpoint_gate_later": [gate_id],
        "gates_never_runtime_enabled_in_11_5": [gate_id],
    }
    assert contract["blocked_credentials_gate_capabilities"] == EXPECTED_BLOCKED
    assert all(value is True for value in contract["credentials_gate_boundaries"].values())

    evidence = contract["non_activation_evidence"]
    assert evidence["adapter_config_gate_11_4_read"] is True
    assert evidence["credentials_gate_contract_built"] is True
    for key, value in evidence.items():
        if key not in {"adapter_config_gate_11_4_read", "credentials_gate_contract_built"}:
            assert value is False
    assert contract["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert contract["future_steps"] == EXPECTED_FUTURE_STEPS


def test_source_imports_and_forbidden_calls_are_static_contract_only():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_testnet_sandbox_adapter_config_gate",
    ]
    for imported in imports:
        assert imported.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS

    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert calls.isdisjoint(FORBIDDEN_CALLS)


def test_qml_preview_selection_bridge_shape_unchanged():
    source = QML_SOURCE.read_text(encoding="utf-8")
    allowed = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
    assert source.count("previewSelectAction(") == 1
    assert allowed in source
    assert "previewSelectSourceControl(" not in source
    assert "resetPreviewSelection(" not in source
    forbidden_fragments = [
        "dynamicDispatch(",
        "startRuntime(",
        "stopRuntime(",
        "pauseRuntime(",
        "resumeRuntime(",
        "submitOrder(",
        "placeOrder(",
        "createOrder(",
        "sendOrder(",
        "fillOrder(",
        "lifecycle(",
        "fetchMarketData(",
        "refreshMarketData(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source
