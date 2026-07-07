"""Tests for FUNCTIONAL-PREVIEW-13.0 Block K static contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_observability_audit_rollback_soak_contract import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_DECISION,
    OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_STATUS,
    PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_KIND,
    PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_1,
    STATUS,
    STEP_ID,
    build_preview_observability_audit_rollback_soak_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_observability_audit_rollback_soak_contract.py"
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "observability_audit_rollback_soak_contract_kind",
    "block",
    "step",
    "observability_audit_rollback_soak_contract_status",
    "observability_audit_rollback_soak_contract_decision",
    "ready_for_block_k_1",
    "next_step",
    "next_step_title",
    "block_j_closure_reference",
    "block_k_contract_scope",
    "block_k_contract_principles",
    "observability_contract",
    "audit_contract",
    "rollback_contract",
    "soak_contract",
    "block_k_dependency_matrix",
    "blocked_block_k_contract_capabilities",
    "block_k_contract_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]

FALSE_SCOPE_FLAGS = [
    "observability_runtime_allowed_now",
    "audit_writer_allowed_now",
    "audit_export_allowed_now",
    "rollback_execution_allowed_now",
    "soak_runtime_allowed_now",
    "metrics_export_allowed_now",
    "log_file_read_allowed_now",
    "log_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "runtime_enforcement_allowed_now",
    "risk_decision_runtime_allowed_now",
    "limit_enforcement_runtime_allowed_now",
    "kill_switch_runtime_allowed_now",
    "manual_trigger_allowed_now",
    "automatic_trigger_allowed_now",
    "kill_switch_state_mutation_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "position_mutation_allowed_now",
    "private_endpoint_access_allowed_now",
    "account_read_allowed_now",
    "balance_read_allowed_now",
    "positions_read_allowed_now",
    "orders_read_allowed_now",
    "fills_read_allowed_now",
    "market_data_read_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "adapter_wiring_allowed_now",
    "scheduler_allowed_now",
    "config_file_read_allowed_now",
    "config_discovery_allowed_now",
    "yaml_parse_allowed_now",
    "json_parse_allowed_now",
    "environment_variable_read_allowed_now",
    "credential_secret_read_allowed_now",
    "credential_validation_allowed_now",
    "secure_store_read_allowed_now",
    "secure_store_write_allowed_now",
    "qml_changes_allowed",
    "new_qml_method_calls_allowed",
    "bridge_api_changes_allowed",
    "exe_packaging_in_scope",
    "bat_productization_allowed",
]

EXPECTED_PRINCIPLES = [
    "observability_before_runtime",
    "audit_before_side_effects",
    "rollback_before_activation",
    "soak_before_runtime_expansion",
    "fail_closed_on_missing_audit",
    "no_runtime_without_explicit_gate",
    "no_export_without_explicit_gate",
    "no_filesystem_io_without_explicit_gate",
    "no_network_without_explicit_gate",
    "live_trading_stays_blocked",
]

EXPECTED_BLOCKED_CAPABILITIES = [
    "observability runtime collection",
    "metrics export",
    "log file read",
    "log file write",
    "audit writer",
    "audit export",
    "rollback execution",
    "runtime shutdown",
    "soak runtime",
    "soak scheduler",
    "filesystem I/O",
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "kill switch state mutation",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account read",
    "balance read",
    "positions read",
    "orders read",
    "fills read",
    "market data read",
    "adapter instantiation",
    "adapter runtime wiring",
    "runtime loop",
    "scheduler",
    "network I/O",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "credential read",
    "secret read",
    "secure store read",
    "secure store write",
    "config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "TradingController change",
    "DecisionEnvelope change",
    "QML action dispatch",
    "bridge API changes",
    "PyInstaller/EXE packaging",
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
    "no observability runtime import",
    "no logger/exporter runtime import",
    "no metrics exporter import",
    "no rollback runner import",
    "no soak runner import",
    "no filesystem I/O",
    "no log file read",
    "no log file write",
    "no audit write",
    "no audit export",
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
    "no real market data read",
    "no private endpoint access",
    "no account read",
    "no balance read",
    "no positions read",
    "no orders read",
    "no fills read",
    "no order generation",
    "no order submission",
    "no order cancel",
    "no order replace",
    "no position mutation",
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


def _payload() -> dict[str, Any]:
    return build_preview_observability_audit_rollback_soak_contract()


def test_payload_is_plain_json_serializable_and_has_exact_top_level_fields() -> None:
    payload = _payload()
    assert list(payload) == TOP_LEVEL_FIELDS
    assert json.loads(json.dumps(payload)) == payload


def test_identity_status_decision_and_next_step_are_exact() -> None:
    payload = _payload()
    assert (
        payload["schema_version"]
        == PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_SCHEMA_VERSION
    )
    assert (
        payload["observability_audit_rollback_soak_contract_kind"]
        == PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_KIND
    )
    assert payload["block"] == BLOCK_ID
    assert payload["step"] == STEP_ID
    assert (
        payload["observability_audit_rollback_soak_contract_status"]
        == OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_STATUS
    )
    assert (
        payload["observability_audit_rollback_soak_contract_decision"]
        == OBSERVABILITY_AUDIT_ROLLBACK_SOAK_CONTRACT_DECISION
    )
    assert payload["ready_for_block_k_1"] is READY_FOR_BLOCK_K_1
    assert payload["next_step"] == NEXT_STEP
    assert payload["next_step_title"] == NEXT_STEP_TITLE
    assert payload["status"] == STATUS


def test_block_j_closure_reference_is_safe_subset_and_points_to_12_5() -> None:
    reference = _payload()["block_j_closure_reference"]
    assert list(reference) == [
        "schema_version",
        "block_j_closure_audit_kind",
        "block_j_closure_audit_status",
        "block_j_closure_audit_decision",
        "ready_for_next_block",
        "next_block",
        "next_step",
        "next_step_title",
        "closure_line",
        "status",
    ]
    assert reference["ready_for_next_block"] is True
    assert reference["next_block"] == "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK"
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-13.0"
    assert (
        reference["next_step_title"] == "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK CONTRACT"
    )
    assert reference["closure_line"] == "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"


def test_scope_is_contract_only_and_all_runtime_flags_are_false() -> None:
    scope = _payload()["block_k_contract_scope"]
    assert scope["scope_name"] == "observability_audit_rollback_soak_contract"
    assert scope["contract_only"] is True
    assert scope["starts_block_k"] is True
    assert scope["derived_from_block_j_closure_12_5"] is True
    assert scope["exe_direction_preserved"] is True
    for flag in FALSE_SCOPE_FLAGS:
        assert scope[flag] is False


def test_principles_are_exact_ordered_required_disabled_and_offline_safe() -> None:
    principles = _payload()["block_k_contract_principles"]
    assert [item["principle_id"] for item in principles] == EXPECTED_PRINCIPLES
    for item in principles:
        assert list(item) == [
            "principle_id",
            "description",
            "required_for_next_step",
            "runtime_enabled_now",
            "order_flow_enabled_now",
            "safe_for_offline_tests",
        ]
        assert item["description"]
        assert item["required_for_next_step"] is True
        assert item["runtime_enabled_now"] is False
        assert item["order_flow_enabled_now"] is False
        assert item["safe_for_offline_tests"] is True


def test_area_contracts_are_exact() -> None:
    payload = _payload()
    assert payload["observability_contract"] == {
        "contract_id": "block_k_observability_contract",
        "read_model_required": True,
        "runtime_metrics_collection_allowed_now": False,
        "metrics_export_allowed_now": False,
        "log_file_read_allowed_now": False,
        "log_file_write_allowed_now": False,
        "filesystem_io_allowed_now": False,
        "network_io_allowed_now": False,
        "ui_surface_allowed_now": False,
        "next_step": "FUNCTIONAL-PREVIEW-13.1",
        "next_step_title": "OBSERVABILITY READ MODEL",
    }
    assert payload["audit_contract"] == {
        "contract_id": "block_k_audit_contract",
        "audit_envelope_required": True,
        "audit_writer_allowed_now": False,
        "audit_export_allowed_now": False,
        "filesystem_io_allowed_now": False,
        "network_io_allowed_now": False,
        "credential_secret_read_allowed_now": False,
        "next_step": "FUNCTIONAL-PREVIEW-13.2",
        "next_step_title": "AUDIT ENVELOPE READ MODEL",
    }
    assert payload["rollback_contract"] == {
        "contract_id": "block_k_rollback_contract",
        "rollback_plan_required": True,
        "rollback_execution_allowed_now": False,
        "state_mutation_allowed_now": False,
        "runtime_shutdown_allowed_now": False,
        "order_cancel_allowed_now": False,
        "private_endpoint_access_allowed_now": False,
        "next_step": "FUNCTIONAL-PREVIEW-13.3",
        "next_step_title": "ROLLBACK READ MODEL",
    }
    assert payload["soak_contract"] == {
        "contract_id": "block_k_soak_contract",
        "soak_plan_required": True,
        "soak_runtime_allowed_now": False,
        "scheduler_allowed_now": False,
        "runtime_loop_allowed_now": False,
        "network_io_allowed_now": False,
        "order_flow_allowed_now": False,
        "private_endpoint_access_allowed_now": False,
        "next_step": "FUNCTIONAL-PREVIEW-13.4",
        "next_step_title": "SOAK READ MODEL",
    }


def test_dependency_matrix_blocked_capabilities_boundaries_sources_and_future_steps_are_exact() -> (
    None
):
    payload = _payload()
    assert payload["block_k_dependency_matrix"] == {
        "source_block": "BLOK J",
        "source_step": "FUNCTIONAL-PREVIEW-12.5",
        "target_block": "BLOK K",
        "target_step": "FUNCTIONAL-PREVIEW-13.0",
        "requires_block_j_closure": True,
        "requires_observability_contract": True,
        "requires_audit_contract": True,
        "requires_rollback_contract": True,
        "requires_soak_contract": True,
        "requires_order_flow_blocked": True,
        "requires_private_endpoint_blocked": True,
        "requires_network_blocked_until_future_gate": True,
        "requires_live_trading_blocked": True,
        "ready_for_13_1_observability_read_model": True,
    }
    assert payload["blocked_block_k_contract_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    boundaries = payload["block_k_contract_boundaries"]
    assert "block_k_contract_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    assert payload["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert payload["future_steps"] == [
        "functional_preview_13_1_observability_read_model",
        "functional_preview_13_2_audit_envelope_read_model",
        "functional_preview_13_3_rollback_read_model",
        "functional_preview_13_4_soak_read_model",
        "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
        "functional_preview_13_6_block_k_closure_audit",
    ]


def test_non_activation_evidence_true_false_contract() -> None:
    evidence = _payload()["non_activation_evidence"]
    assert evidence["block_j_closure_12_5_read"] is True
    assert evidence["block_k_contract_built"] is True
    for key, value in evidence.items():
        if key not in {"block_j_closure_12_5_read", "block_k_contract_built"}:
            assert value is False


def test_source_imports_only_safe_modules_and_has_no_forbidden_runtime_calls() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        ("ui.pyside_app.preview_block_j_closure_audit", ["build_preview_block_j_closure_audit"]),
    ]
    forbidden = [
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
        "start_observability",
        "collect_metrics",
        "export_metrics",
        "write_log",
        "read_log",
        "write_audit",
        "export_audit",
        "execute_rollback",
        "start_soak",
        "run_soak",
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
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
    ]
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attrs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for token in forbidden:
        assert token not in called_names
        assert token not in called_attrs
