"""Tests for FUNCTIONAL-PREVIEW-11.3 Block I static connectivity fixture."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_static_connectivity_fixture import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_KIND,
    PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_4,
    STATUS,
    TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_DECISION,
    TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_STATUS,
    build_preview_testnet_sandbox_static_connectivity_fixture,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_static_connectivity_fixture.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_static_connectivity_fixture_kind",
    "block",
    "step",
    "testnet_sandbox_static_connectivity_fixture_status",
    "testnet_sandbox_static_connectivity_fixture_decision",
    "ready_for_block_i_4",
    "next_step",
    "next_step_title",
    "adapter_read_model_reference",
    "fixture_scope",
    "static_connectivity_fixture_entries",
    "default_static_connectivity_fixture_selection",
    "static_connectivity_fixture_summary",
    "connectivity_fixture_matrix",
    "blocked_connectivity_capabilities",
    "fixture_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_ENTRY_FIELDS = {
    "connectivity_fixture_id",
    "source_adapter_read_model_id",
    "source_capability",
    "display_name",
    "fixture_classification",
    "connectivity_surface_type",
    "static_connectivity_state",
    "simulated_endpoint_category",
    "eligible_for_11_4_config_gate",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "real_probe_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "config_read_allowed_now",
    "credentials_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "runtime_allowed_now",
    "fixture_safe_for_offline_tests",
    "operator_visibility",
    "notes",
}
EXPECTED_ORDER = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
    "paper_execution_oracle",
]
EXPECTED_BLOCKED = [
    "real connectivity probe",
    "adapter instantiation",
    "adapter config read",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "credentials read",
    "secrets read",
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
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
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
    "order",
    "live",
    "testnet",
    "sandbox",
    "exchange",
    "account",
    "secrets",
    "network",
    "filesystem",
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
    "ui.pyside_app.preview_testnet_sandbox_adapter_read_model",
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


def _fixture() -> dict[str, object]:
    return build_preview_testnet_sandbox_static_connectivity_fixture()


def test_fixture_is_plain_json_serializable_dict_with_exact_top_level_fields() -> None:
    fixture = _fixture()

    assert isinstance(fixture, dict)
    _assert_simple_types_only(fixture)
    assert json.loads(json.dumps(fixture, sort_keys=True)) == fixture
    assert set(fixture) == EXPECTED_TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    fixture = _fixture()

    assert (
        fixture["schema_version"]
        == PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_SCHEMA_VERSION
    )
    assert (
        fixture["testnet_sandbox_static_connectivity_fixture_kind"]
        == PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_KIND
    )
    assert fixture["block"] == "I"
    assert fixture["step"] == "11.3"
    assert (
        fixture["testnet_sandbox_static_connectivity_fixture_status"]
        == TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_STATUS
    )
    assert (
        fixture["testnet_sandbox_static_connectivity_fixture_decision"]
        == TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_DECISION
    )
    assert fixture["ready_for_block_i_4"] is READY_FOR_BLOCK_I_4 is True
    assert fixture["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.4"
    assert fixture["next_step_title"] == NEXT_STEP_TITLE == "TESTNET/SANDBOX ADAPTER CONFIG GATE"
    assert fixture["status"] == STATUS


def test_adapter_read_model_reference_points_to_11_2() -> None:
    reference = _fixture()["adapter_read_model_reference"]

    assert set(reference) == {
        "schema_version",
        "testnet_sandbox_adapter_read_model_kind",
        "testnet_sandbox_adapter_read_model_status",
        "testnet_sandbox_adapter_read_model_decision",
        "ready_for_block_i_3",
        "next_step",
        "next_step_title",
        "status",
    }
    assert reference["ready_for_block_i_3"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.3"
    assert reference["next_step_title"] == "TESTNET/SANDBOX STATIC CONNECTIVITY FIXTURE"


def test_fixture_scope_is_static_only_with_all_activation_flags_blocked() -> None:
    scope = _fixture()["fixture_scope"]

    assert scope["scope_name"] == "testnet_sandbox_static_connectivity_fixture"
    assert scope["static_fixture_only"] is True
    assert scope["derived_from_adapter_read_model_11_2"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key.endswith("allowed_now") or key in {
            "qml_changes_allowed",
            "new_qml_method_calls_allowed",
            "bridge_api_changes_allowed",
            "exe_packaging_in_scope",
            "bat_productization_allowed",
        }:
            assert value is False


def test_static_entries_have_exact_fields_order_and_no_activation() -> None:
    entries = _fixture()["static_connectivity_fixture_entries"]

    assert len(entries) == 4
    assert [entry["source_capability"] for entry in entries] == EXPECTED_ORDER
    for entry in entries:
        assert set(entry) == EXPECTED_ENTRY_FIELDS
        assert entry["static_connectivity_state"] == "fixture_available_no_probe"
        assert entry["notes"]
        for key in [
            "real_probe_allowed_now",
            "network_io_allowed_now",
            "dns_lookup_allowed_now",
            "http_request_allowed_now",
            "websocket_allowed_now",
            "config_read_allowed_now",
            "credentials_allowed_now",
            "private_endpoint_allowed_now",
            "order_submission_allowed_now",
            "runtime_allowed_now",
        ]:
            assert entry[key] is False
        assert entry["fixture_safe_for_offline_tests"] is True


def test_static_entries_specific_values() -> None:
    entries = {
        entry["source_capability"]: entry
        for entry in _fixture()["static_connectivity_fixture_entries"]
    }

    assert (
        entries["read_only_market_data_provider"] | {} == entries["read_only_market_data_provider"]
    )
    assert (
        entries["read_only_market_data_provider"]["connectivity_fixture_id"]
        == "static_connectivity_fixture_read_only_market_data_provider"
    )
    assert (
        entries["read_only_market_data_provider"]["source_adapter_read_model_id"]
        == "adapter_read_model_read_only_market_data_provider"
    )
    assert (
        entries["read_only_market_data_provider"]["fixture_classification"]
        == "lowest_risk_public_market_data_fixture"
    )
    assert (
        entries["read_only_market_data_provider"]["connectivity_surface_type"]
        == "public_market_data_static_fixture"
    )
    assert (
        entries["read_only_market_data_provider"]["simulated_endpoint_category"]
        == "public_market_data"
    )
    assert entries["read_only_market_data_provider"]["eligible_for_11_4_config_gate"] is True
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
    assert entries["read_only_market_data_provider"]["operator_visibility"] == "read_only_future"

    assert (
        entries["exchange_adapter_layer"]["connectivity_fixture_id"]
        == "static_connectivity_fixture_exchange_adapter_layer"
    )
    assert entries["exchange_adapter_layer"]["eligible_for_11_5_credentials_gate"] is True
    assert entries["exchange_adapter_layer"]["eligible_for_11_7_private_endpoint_gate"] is True
    assert entries["exchange_adapter_layer"]["operator_visibility"] == "blocked_until_gated"

    assert (
        entries["exchange_network_guard"]["connectivity_fixture_id"]
        == "static_connectivity_fixture_exchange_network_guard"
    )
    assert (
        entries["exchange_network_guard"]["eligible_for_11_6_public_market_data_probe_preview"]
        is False
    )
    assert entries["exchange_network_guard"]["eligible_for_11_7_private_endpoint_gate"] is True
    assert entries["exchange_network_guard"]["operator_visibility"] == "safety_guard_future"

    assert (
        entries["paper_execution_oracle"]["connectivity_fixture_id"]
        == "static_connectivity_fixture_paper_execution_oracle"
    )
    assert entries["paper_execution_oracle"]["eligible_for_11_4_config_gate"] is False
    assert entries["paper_execution_oracle"]["operator_visibility"] == "comparison_only_future"


def test_summary_default_and_matrix_match_entries() -> None:
    fixture = _fixture()
    entries = fixture["static_connectivity_fixture_entries"]
    summary = fixture["static_connectivity_fixture_summary"]
    default = fixture["default_static_connectivity_fixture_selection"]

    assert default == {
        "connectivity_fixture_id": "static_connectivity_fixture_read_only_market_data_provider",
        "source_capability": "read_only_market_data_provider",
        "reason": "lowest-risk static connectivity fixture; no config read, no credentials, no network I/O, no probe, no runtime activation",
        "real_probe_allowed_now": False,
        "network_io_allowed_now": False,
    }
    assert summary == {
        "entry_count": 4,
        "default_selection_id": "static_connectivity_fixture_read_only_market_data_provider",
        "real_probe_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "dns_lookup_enabled_entry_count": 0,
        "http_request_enabled_entry_count": 0,
        "websocket_enabled_entry_count": 0,
        "config_read_enabled_entry_count": 0,
        "credentials_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "order_submission_enabled_entry_count": 0,
        "runtime_enabled_entry_count": 0,
        "offline_safe_entry_count": 4,
        "entries_eligible_for_11_4_config_gate": 3,
        "entries_eligible_for_11_5_credentials_gate": 1,
        "entries_eligible_for_11_6_public_market_data_probe_preview": 2,
        "entries_eligible_for_11_7_private_endpoint_gate": 2,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
    }
    assert summary["entry_count"] == len(entries)
    assert fixture["connectivity_fixture_matrix"] == {
        "public_market_data_fixture_ids": [
            "static_connectivity_fixture_read_only_market_data_provider"
        ],
        "exchange_adapter_fixture_ids": ["static_connectivity_fixture_exchange_adapter_layer"],
        "network_guard_fixture_ids": ["static_connectivity_fixture_exchange_network_guard"],
        "paper_oracle_fixture_ids": ["static_connectivity_fixture_paper_execution_oracle"],
        "fixtures_requiring_credentials_gate_later": [
            "static_connectivity_fixture_exchange_adapter_layer"
        ],
        "fixtures_requiring_private_endpoint_gate_later": [
            "static_connectivity_fixture_exchange_adapter_layer",
            "static_connectivity_fixture_exchange_network_guard",
        ],
        "fixtures_eligible_for_public_market_data_probe_preview_later": [
            "static_connectivity_fixture_read_only_market_data_provider",
            "static_connectivity_fixture_exchange_adapter_layer",
        ],
        "fixtures_never_runtime_enabled_in_11_3": [
            entry["connectivity_fixture_id"] for entry in entries
        ],
    }


def test_static_boundaries_evidence_and_future_steps_are_exact() -> None:
    fixture = _fixture()

    assert fixture["blocked_connectivity_capabilities"] == EXPECTED_BLOCKED
    assert all(fixture["fixture_boundaries"].values())
    evidence = fixture["non_activation_evidence"]
    assert evidence["adapter_read_model_11_2_read"] is True
    assert evidence["static_connectivity_fixture_built"] is True
    for key, value in evidence.items():
        if key not in {"adapter_read_model_11_2_read", "static_connectivity_fixture_built"}:
            assert value is False
    assert fixture["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert fixture["future_steps"] == EXPECTED_FUTURE_STEPS


def test_source_imports_only_safe_typing_and_adapter_read_model_helper() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imports.append(node.module)

    assert imports == ALLOWED_IMPORTS
    for imported in imports:
        assert imported.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS


def test_source_has_no_forbidden_runtime_network_filesystem_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            if isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)

    assert call_names.isdisjoint(FORBIDDEN_CALLS)


def test_qml_surface_remains_single_allowed_preview_selection_call() -> None:
    qml = QML_SOURCE.read_text(encoding="utf-8")
    allowed_literal = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

    assert qml.count("previewSelectAction(") == 1
    assert allowed_literal in qml
    assert "previewSelectSourceControl(" not in qml
    assert "resetPreviewSelection(" not in qml
    for forbidden in [
        "previewSelectAction(action",
        "previewSelectAction(model",
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
    ]:
        assert forbidden not in qml
