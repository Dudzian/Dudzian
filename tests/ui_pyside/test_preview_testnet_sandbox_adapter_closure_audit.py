"""Tests for FUNCTIONAL-PREVIEW-11.8 Block I closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_adapter_closure_audit import (
    BLOCKED_CAPABILITIES,
    CLOSURE_LINE,
    FUTURE_BLOCKS,
    NEXT_BLOCK,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_KIND,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_J,
    SOURCE_BOUNDARIES,
    STATUS,
    TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_DECISION,
    TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_STATUS,
    build_preview_testnet_sandbox_adapter_closure_audit,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_adapter_closure_audit.py")
QML_SOURCE = Path("ui/pyside_app/qml/views/OperatorDashboard.qml")
EXPECTED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "testnet_sandbox_adapter_closure_audit_kind",
    "block",
    "step",
    "testnet_sandbox_adapter_closure_audit_status",
    "testnet_sandbox_adapter_closure_audit_decision",
    "ready_for_block_j",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
    "private_endpoint_gate_reference",
    "block_i_closure_scope",
    "block_i_completed_steps",
    "block_i_completion_matrix",
    "testnet_sandbox_path_summary",
    "adapter_activation_evidence",
    "runtime_network_order_safety_evidence",
    "blocked_capabilities",
    "source_boundaries",
    "block_j_entry_requirements",
    "future_blocks",
    "status",
]
EXPECTED_COMPLETED = [
    (
        "FUNCTIONAL-PREVIEW-11.0",
        "TESTNET/SANDBOX ADAPTER CONTRACT",
        "preview_testnet_sandbox_adapter_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.1",
        "TESTNET/SANDBOX BACKEND CAPABILITY HANDOFF",
        "preview_testnet_sandbox_backend_capability_handoff",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.2",
        "TESTNET/SANDBOX ADAPTER READ MODEL",
        "preview_testnet_sandbox_adapter_read_model",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.3",
        "TESTNET/SANDBOX STATIC CONNECTIVITY FIXTURE",
        "preview_testnet_sandbox_static_connectivity_fixture",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.4",
        "TESTNET/SANDBOX ADAPTER CONFIG GATE",
        "preview_testnet_sandbox_adapter_config_gate",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.5",
        "TESTNET/SANDBOX CREDENTIALS GATE CONTRACT",
        "preview_testnet_sandbox_credentials_gate_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.6",
        "TESTNET/SANDBOX PUBLIC MARKET DATA PROBE PREVIEW",
        "preview_testnet_sandbox_public_market_data_probe_preview",
    ),
    (
        "FUNCTIONAL-PREVIEW-11.7",
        "TESTNET/SANDBOX PRIVATE ENDPOINT GATE",
        "preview_testnet_sandbox_private_endpoint_gate",
    ),
]
FALSE_SCOPE_FLAGS = [
    "adapter_implementation_allowed_now",
    "adapter_instantiation_allowed_now",
    "adapter_wiring_allowed_now",
    "runtime_execution_allowed_now",
    "scheduler_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "real_market_data_fetch_allowed_now",
    "real_public_probe_allowed_now",
    "private_endpoint_access_allowed_now",
    "private_endpoint_probe_allowed_now",
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "withdrawal_allowed_now",
    "transfer_allowed_now",
    "margin_or_leverage_mutation_allowed_now",
    "credential_secret_read_allowed_now",
    "credential_validation_allowed_now",
    "secure_store_read_allowed_now",
    "secure_store_write_allowed_now",
    "config_file_read_allowed_now",
    "config_discovery_allowed_now",
    "yaml_parse_allowed_now",
    "json_parse_allowed_now",
    "environment_variable_read_allowed_now",
    "qml_changes_allowed",
    "new_qml_method_calls_allowed",
    "bridge_api_changes_allowed",
    "exe_packaging_in_scope",
    "bat_productization_allowed",
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


def assert_plain(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            assert_plain(child)
    if isinstance(value, list):
        for child in value:
            assert_plain(child)


def test_closure_audit_is_plain_json_serializable_with_exact_top_level_fields() -> None:
    audit = build_preview_testnet_sandbox_adapter_closure_audit()

    assert list(audit) == EXPECTED_TOP_LEVEL_FIELDS
    assert_plain(audit)
    assert json.loads(json.dumps(audit)) == audit


def test_identity_status_decision_next_and_closure_line() -> None:
    audit = build_preview_testnet_sandbox_adapter_closure_audit()

    assert audit["schema_version"] == PREVIEW_TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_SCHEMA_VERSION
    assert (
        audit["testnet_sandbox_adapter_closure_audit_kind"]
        == PREVIEW_TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_KIND
    )
    assert audit["block"] == "I"
    assert audit["step"] == "11.8"
    assert (
        audit["testnet_sandbox_adapter_closure_audit_status"]
        == TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_STATUS
    )
    assert (
        audit["testnet_sandbox_adapter_closure_audit_decision"]
        == TESTNET_SANDBOX_ADAPTER_CLOSURE_AUDIT_DECISION
    )
    assert audit["ready_for_block_j"] is READY_FOR_BLOCK_J
    assert audit["next_block"] == NEXT_BLOCK
    assert audit["next_step"] == NEXT_STEP
    assert audit["next_step_title"] == NEXT_STEP_TITLE
    assert audit["closure_line"] == CLOSURE_LINE == "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"
    assert audit["status"] == STATUS


def test_private_endpoint_gate_reference_points_to_11_7_safe_subset() -> None:
    ref = build_preview_testnet_sandbox_adapter_closure_audit()["private_endpoint_gate_reference"]

    assert list(ref) == [
        "schema_version",
        "testnet_sandbox_private_endpoint_gate_kind",
        "testnet_sandbox_private_endpoint_gate_status",
        "testnet_sandbox_private_endpoint_gate_decision",
        "ready_for_block_i_8",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert ref["ready_for_block_i_8"] is True
    assert ref["next_step"] == "FUNCTIONAL-PREVIEW-11.8"
    assert ref["next_step_title"] == "BLOK I — TESTNET/SANDBOX ADAPTER CLOSURE AUDIT"


def test_block_i_closure_scope_is_closure_only_and_blocks_activation() -> None:
    scope = build_preview_testnet_sandbox_adapter_closure_audit()["block_i_closure_scope"]

    assert scope["scope_name"] == "testnet_sandbox_adapter_closure_audit"
    assert scope["closure_audit_only"] is True
    assert scope["derived_from_private_endpoint_gate_11_7"] is True
    assert scope["closes_block_i"] is True
    assert scope["ready_for_block_j"] is True
    assert scope["exe_direction_preserved"] is True
    for key in FALSE_SCOPE_FLAGS:
        assert scope[key] is False


def test_block_i_completed_steps_are_exact_and_non_activating() -> None:
    steps = build_preview_testnet_sandbox_adapter_closure_audit()["block_i_completed_steps"]

    assert len(steps) == 8
    for item, (step, title, artifact) in zip(steps, EXPECTED_COMPLETED, strict=True):
        assert list(item) == [
            "step",
            "title",
            "status",
            "artifact",
            "runtime_activation",
            "network_io",
            "order_flow",
            "private_endpoint_access",
            "ready_for_next_step",
        ]
        assert item == {
            "step": step,
            "title": title,
            "status": "accepted",
            "artifact": artifact,
            "runtime_activation": False,
            "network_io": False,
            "order_flow": False,
            "private_endpoint_access": False,
            "ready_for_next_step": True,
        }


def test_completion_matrix_summary_evidence_boundaries_and_future_blocks_are_exact() -> None:
    audit = build_preview_testnet_sandbox_adapter_closure_audit()

    assert audit["block_i_completion_matrix"] == {
        "completed_step_count": 8,
        "accepted_step_count": 8,
        "contract_steps_completed": True,
        "handoff_completed": True,
        "read_model_completed": True,
        "static_fixture_completed": True,
        "config_gate_completed": True,
        "credentials_gate_completed": True,
        "public_probe_preview_completed": True,
        "private_endpoint_gate_completed": True,
        "closure_audit_completed": True,
        "ready_for_block_j": True,
        "runtime_activation_count": 0,
        "network_io_count": 0,
        "order_flow_count": 0,
        "private_endpoint_access_count": 0,
    }
    assert audit["testnet_sandbox_path_summary"] == {
        "path_status": "closed_as_contract_ready_no_runtime_activation",
        "block_i_result": "testnet_sandbox_adapter_path_mapped_and_gated",
        "testnet_adapter_implemented": False,
        "sandbox_adapter_implemented": False,
        "adapter_instantiated": False,
        "adapter_wired_to_runtime": False,
        "network_probe_performed": False,
        "market_data_fetch_performed": False,
        "private_endpoint_access_performed": False,
        "account_fetch_performed": False,
        "balance_fetch_performed": False,
        "positions_fetch_performed": False,
        "orders_fetch_performed": False,
        "fills_fetch_performed": False,
        "order_generation_performed": False,
        "order_submission_performed": False,
        "runtime_started": False,
        "scheduler_started": False,
        "ready_for_risk_governor_block": True,
        "requires_block_j_before_any_order_flow": True,
        "requires_block_k_before_any_soak": True,
        "requires_block_l_before_any_live_canary": True,
    }
    assert audit["adapter_activation_evidence"] == {
        "adapter_contract_created": True,
        "backend_capability_handoff_created": True,
        "adapter_read_model_created": True,
        "static_connectivity_fixture_created": True,
        "adapter_config_gate_created": True,
        "credentials_gate_contract_created": True,
        "public_market_data_probe_preview_created": True,
        "private_endpoint_gate_created": True,
        "closure_audit_created": True,
        "testnet_adapter_runtime_created": False,
        "sandbox_adapter_runtime_created": False,
        "exchange_adapter_imported_for_runtime": False,
        "adapter_instantiated": False,
        "adapter_wired_to_bridge": False,
        "adapter_wired_to_runtime": False,
        "trading_controller_touched": False,
        "decision_envelope_touched": False,
    }
    assert all(value is False for value in audit["runtime_network_order_safety_evidence"].values())
    assert audit["blocked_capabilities"] == BLOCKED_CAPABILITIES
    assert audit["source_boundaries"] == SOURCE_BOUNDARIES
    assert audit["block_j_entry_requirements"] == {
        "block_j_name": NEXT_BLOCK,
        "entry_status": "ready",
        "requires_order_flow_to_remain_blocked_until_block_j_complete": True,
        "requires_private_endpoint_to_remain_blocked_until_explicit_gate": True,
        "requires_network_io_to_remain_blocked_until_explicit_probe_gate": True,
        "requires_live_trading_to_remain_blocked": True,
        "first_step": NEXT_STEP,
        "first_step_title": NEXT_STEP_TITLE,
    }
    assert audit["future_blocks"] == FUTURE_BLOCKS


def test_source_imports_only_safe_typing_and_private_endpoint_gate_helper() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", ",".join(alias.name for alias in node.names)))

    assert imports == [
        ("__future__", "annotations"),
        ("typing", "Any,Final"),
        (
            "ui.pyside_app.preview_testnet_sandbox_private_endpoint_gate",
            "build_preview_testnet_sandbox_private_endpoint_gate",
        ),
    ]
    for module, _names in imports:
        assert module.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS


def test_source_has_no_forbidden_runtime_network_filesystem_or_order_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            assert name not in FORBIDDEN_CALLS


def test_qml_is_unchanged_and_keeps_single_allowed_preview_selection_bridge_call() -> None:
    source = QML_SOURCE.read_text(encoding="utf-8")

    assert source.count("previewSelectAction(") == 1
    assert (
        'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
        in source
    )
