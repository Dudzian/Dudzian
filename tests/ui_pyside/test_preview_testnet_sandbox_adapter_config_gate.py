"""Tests for FUNCTIONAL-PREVIEW-11.4 Block I adapter config gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_adapter_config_gate import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_KIND,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_5,
    STATUS,
    TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_DECISION,
    TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_STATUS,
    build_preview_testnet_sandbox_adapter_config_gate,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_adapter_config_gate.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_adapter_config_gate_kind",
    "block",
    "step",
    "testnet_sandbox_adapter_config_gate_status",
    "testnet_sandbox_adapter_config_gate_decision",
    "ready_for_block_i_5",
    "next_step",
    "next_step_title",
    "static_connectivity_fixture_reference",
    "config_gate_scope",
    "adapter_config_gate_entries",
    "default_adapter_config_gate_selection",
    "adapter_config_gate_summary",
    "config_shape_requirements",
    "config_gate_matrix",
    "blocked_config_gate_capabilities",
    "config_gate_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_ENTRY_FIELDS = {
    "config_gate_id",
    "source_connectivity_fixture_id",
    "source_capability",
    "display_name",
    "config_gate_classification",
    "config_surface_type",
    "required_config_shape",
    "optional_config_shape",
    "forbidden_config_material",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "config_file_read_allowed_now",
    "config_discovery_allowed_now",
    "yaml_parse_allowed_now",
    "json_parse_allowed_now",
    "environment_variable_read_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
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
EXPECTED_ORDER = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
]
EXPECTED_BLOCKED = [
    "real config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "credential material handling",
    "secret material handling",
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
    "no filesystem I/O",
    "no config file read",
    "no config discovery",
    "no YAML parse",
    "no JSON parse",
    "no environment variable read",
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
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
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
ALLOWED_IMPORTS = [
    "__future__",
    "typing",
    "ui.pyside_app.preview_testnet_sandbox_static_connectivity_fixture",
]


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    if isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def _gate() -> dict[str, object]:
    return build_preview_testnet_sandbox_adapter_config_gate()


def test_gate_is_plain_json_serializable_dict_with_exact_top_level_fields() -> None:
    gate = _gate()

    assert isinstance(gate, dict)
    _assert_simple_types_only(gate)
    json.dumps(gate, sort_keys=True)
    assert set(gate) == EXPECTED_TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    gate = _gate()

    assert gate["schema_version"] == PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_SCHEMA_VERSION
    assert (
        gate["testnet_sandbox_adapter_config_gate_kind"]
        == PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_KIND
    )
    assert gate["block"] == "I"
    assert gate["step"] == "11.4"
    assert (
        gate["testnet_sandbox_adapter_config_gate_status"]
        == TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_STATUS
    )
    assert (
        gate["testnet_sandbox_adapter_config_gate_decision"]
        == TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_DECISION
    )
    assert gate["ready_for_block_i_5"] is READY_FOR_BLOCK_I_5 is True
    assert gate["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.5"
    assert gate["next_step_title"] == NEXT_STEP_TITLE == "TESTNET/SANDBOX CREDENTIALS GATE CONTRACT"
    assert gate["status"] == STATUS


def test_static_connectivity_fixture_reference_points_to_11_3() -> None:
    reference = _gate()["static_connectivity_fixture_reference"]

    assert set(reference) == {
        "schema_version",
        "testnet_sandbox_static_connectivity_fixture_kind",
        "testnet_sandbox_static_connectivity_fixture_status",
        "testnet_sandbox_static_connectivity_fixture_decision",
        "ready_for_block_i_4",
        "next_step",
        "next_step_title",
        "status",
    }
    assert reference["ready_for_block_i_4"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.4"
    assert reference["next_step_title"] == "TESTNET/SANDBOX ADAPTER CONFIG GATE"


def test_config_gate_scope_is_shape_only_and_non_runtime() -> None:
    scope = _gate()["config_gate_scope"]

    assert scope["scope_name"] == "testnet_sandbox_adapter_config_gate"
    assert scope["config_gate_shape_only"] is True
    assert scope["derived_from_static_connectivity_fixture_11_3"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "config_gate_shape_only",
            "derived_from_static_connectivity_fixture_11_3",
            "exe_direction_preserved",
        }:
            assert value is False, key


def test_entries_are_exact_ordered_and_shape_only() -> None:
    entries = _gate()["adapter_config_gate_entries"]

    assert [entry["source_capability"] for entry in entries] == EXPECTED_ORDER
    assert "paper_execution_oracle" not in {entry["source_capability"] for entry in entries}
    for entry in entries:
        assert set(entry) == EXPECTED_ENTRY_FIELDS
        assert entry["notes"]
        assert entry["gate_safe_for_offline_tests"] is True
        for key in (
            "config_file_read_allowed_now",
            "config_discovery_allowed_now",
            "yaml_parse_allowed_now",
            "json_parse_allowed_now",
            "environment_variable_read_allowed_now",
            "credentials_allowed_now",
            "secrets_allowed_now",
            "network_io_allowed_now",
            "dns_lookup_allowed_now",
            "http_request_allowed_now",
            "websocket_allowed_now",
            "private_endpoint_allowed_now",
            "order_submission_allowed_now",
            "runtime_allowed_now",
        ):
            assert entry[key] is False, key


def test_entry_values_match_required_contract() -> None:
    entries = {
        entry["source_capability"]: entry for entry in _gate()["adapter_config_gate_entries"]
    }

    assert (
        entries["read_only_market_data_provider"] | {} == entries["read_only_market_data_provider"]
    )
    assert (
        entries["read_only_market_data_provider"]["config_gate_id"]
        == "adapter_config_gate_read_only_market_data_provider"
    )
    assert entries["read_only_market_data_provider"]["required_config_shape"] == [
        "mode",
        "provider_id",
        "symbols_allowlist",
        "timeframe",
        "rate_limit_profile",
    ]
    assert entries["read_only_market_data_provider"]["eligible_for_11_5_credentials_gate"] is False
    assert (
        entries["read_only_market_data_provider"][
            "eligible_for_11_6_public_market_data_probe_preview"
        ]
        is True
    )
    assert (
        entries["read_only_market_data_provider"]["eligible_for_11_7_private_endpoint_gate"]
        is False
    )
    assert (
        entries["exchange_adapter_layer"]["config_gate_id"]
        == "adapter_config_gate_exchange_adapter_layer"
    )
    assert entries["exchange_adapter_layer"]["required_config_shape"] == [
        "mode",
        "exchange_id",
        "adapter_family",
        "environment",
        "sandbox_or_testnet_flag",
        "symbols_allowlist",
        "rate_limit_profile",
        "network_guard_profile",
    ]
    assert entries["exchange_adapter_layer"]["eligible_for_11_5_credentials_gate"] is True
    assert (
        entries["exchange_adapter_layer"]["eligible_for_11_6_public_market_data_probe_preview"]
        is True
    )
    assert entries["exchange_adapter_layer"]["eligible_for_11_7_private_endpoint_gate"] is True
    assert (
        entries["exchange_network_guard"]["config_gate_id"]
        == "adapter_config_gate_exchange_network_guard"
    )
    assert entries["exchange_network_guard"]["required_config_shape"] == [
        "mode",
        "network_policy",
        "allowed_endpoint_categories",
        "blocked_endpoint_categories",
        "rate_limit_profile",
        "audit_profile",
    ]
    assert entries["exchange_network_guard"]["eligible_for_11_5_credentials_gate"] is False
    assert (
        entries["exchange_network_guard"]["eligible_for_11_6_public_market_data_probe_preview"]
        is False
    )
    assert entries["exchange_network_guard"]["eligible_for_11_7_private_endpoint_gate"] is True


def test_summary_default_requirements_matrix_and_lists() -> None:
    gate = _gate()
    entries = gate["adapter_config_gate_entries"]

    assert gate["default_adapter_config_gate_selection"] == {
        "config_gate_id": "adapter_config_gate_read_only_market_data_provider",
        "source_capability": "read_only_market_data_provider",
        "reason": "lowest-risk config gate; public-market-data shape only, no config read, no credentials, no network I/O",
        "config_file_read_allowed_now": False,
        "network_io_allowed_now": False,
    }
    assert gate["adapter_config_gate_summary"] == {
        "entry_count": 3,
        "default_selection_id": "adapter_config_gate_read_only_market_data_provider",
        "config_file_read_enabled_entry_count": 0,
        "config_discovery_enabled_entry_count": 0,
        "yaml_parse_enabled_entry_count": 0,
        "json_parse_enabled_entry_count": 0,
        "environment_variable_read_enabled_entry_count": 0,
        "credentials_enabled_entry_count": 0,
        "secrets_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "dns_lookup_enabled_entry_count": 0,
        "http_request_enabled_entry_count": 0,
        "websocket_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "order_submission_enabled_entry_count": 0,
        "runtime_enabled_entry_count": 0,
        "offline_safe_entry_count": 3,
        "entries_eligible_for_11_5_credentials_gate": 1,
        "entries_eligible_for_11_6_public_market_data_probe_preview": 2,
        "entries_eligible_for_11_7_private_endpoint_gate": 2,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
    }
    requirements = gate["config_shape_requirements"]
    assert list(requirements["required_shapes_by_source_capability"]) == EXPECTED_ORDER
    assert list(requirements["forbidden_material_by_source_capability"]) == EXPECTED_ORDER
    assert requirements["global_required_shape_fields"] == ["mode", "rate_limit_profile"]
    assert requirements["global_forbidden_material"] == [
        "api_key_value",
        "api_secret_value",
        "passphrase_value",
        "raw_secret",
        "live_endpoint_override",
        "order_submission_enabled",
        "account_id",
        "private_endpoint_url",
        "order_endpoint_url",
    ]
    assert gate["config_gate_matrix"] == {
        "public_market_data_config_gate_ids": [
            "adapter_config_gate_read_only_market_data_provider"
        ],
        "exchange_adapter_config_gate_ids": ["adapter_config_gate_exchange_adapter_layer"],
        "network_guard_config_gate_ids": ["adapter_config_gate_exchange_network_guard"],
        "gates_requiring_credentials_gate_later": ["adapter_config_gate_exchange_adapter_layer"],
        "gates_requiring_private_endpoint_gate_later": [
            "adapter_config_gate_exchange_adapter_layer",
            "adapter_config_gate_exchange_network_guard",
        ],
        "gates_eligible_for_public_market_data_probe_preview_later": [
            "adapter_config_gate_read_only_market_data_provider",
            "adapter_config_gate_exchange_adapter_layer",
        ],
        "gates_never_runtime_enabled_in_11_4": [entry["config_gate_id"] for entry in entries],
    }
    assert gate["blocked_config_gate_capabilities"] == EXPECTED_BLOCKED
    assert gate["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert gate["future_steps"] == EXPECTED_FUTURE_STEPS


def test_boundaries_and_non_activation_evidence_are_exact() -> None:
    gate = _gate()

    assert all(value is True for value in gate["config_gate_boundaries"].values())
    evidence = gate["non_activation_evidence"]
    assert evidence["static_connectivity_fixture_11_3_read"] is True
    assert evidence["adapter_config_gate_built"] is True
    for key, value in evidence.items():
        if key not in {"static_connectivity_fixture_11_3_read", "adapter_config_gate_built"}:
            assert value is False, key


def test_source_imports_only_safe_typing_and_static_fixture_helper() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert imports == ALLOWED_IMPORTS
    for imported in imports:
        root = imported.split(".")[0]
        assert root not in FORBIDDEN_IMPORT_ROOTS


def test_source_has_no_forbidden_runtime_calls_or_names() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Attribute):
            names.add(node.attr)

    assert not (FORBIDDEN_CALLS & names)


def test_qml_operator_dashboard_preview_selection_contract_unchanged() -> None:
    source = QML_SOURCE.read_text(encoding="utf-8")

    assert source.count("previewSelectAction(") == 1
    assert (
        'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
        in source
    )
    assert "paperRuntimeActionDispatchBridge.previewSelectSourceControl" not in source
    assert "paperRuntimeActionDispatchBridge.resetPreviewSelection" not in source
    assert ".previewSelectSourceControl(" not in source
    assert ".resetPreviewSelection(" not in source
    for forbidden in (
        "dynamicDispatch",
        "startRuntime",
        "stopRuntime",
        "pauseRuntime",
        "resumeRuntime",
        "submitOrder",
        "placeOrder",
        "createOrder",
        "sendOrder",
        "fillOrder",
        "fetchMarketData",
        "refreshMarketData",
    ):
        assert forbidden not in source
