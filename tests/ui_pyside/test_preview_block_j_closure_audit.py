"""Tests for FUNCTIONAL-PREVIEW-12.5 Block J closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_j_closure_audit import (
    BLOCK_ID,
    BLOCK_J_CLOSURE_AUDIT_DECISION,
    BLOCK_J_CLOSURE_AUDIT_STATUS,
    CLOSURE_LINE,
    NEXT_BLOCK,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_J_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_J_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_NEXT_BLOCK,
    STATUS,
    STEP_ID,
    build_preview_block_j_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_j_closure_audit.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_j_closure_audit_kind",
    "block",
    "step",
    "block_j_closure_audit_status",
    "block_j_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
    "risk_contract_reference",
    "risk_limits_read_model_reference",
    "risk_limits_static_fixture_reference",
    "kill_switch_read_model_reference",
    "risk_governor_gate_matrix_reference",
    "block_j_closure_scope",
    "block_j_completed_steps",
    "block_j_completion_matrix",
    "block_j_safety_summary",
    "risk_governor_capability_closure",
    "runtime_order_private_network_closure",
    "blocked_block_j_capabilities",
    "block_j_source_boundaries",
    "non_activation_evidence",
    "next_block_entry_requirements",
    "future_blocks",
    "status",
]

FALSE_SCOPE_FLAGS = [
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

EXPECTED_STEPS = [
    (
        "FUNCTIONAL-PREVIEW-12.0",
        "Risk Governor Limits Kill Switch Contract",
        "preview_risk_governor_limits_kill_switch_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-12.1",
        "Risk Governor Limits Read Model",
        "preview_risk_governor_limits_read_model",
    ),
    ("FUNCTIONAL-PREVIEW-12.2", "Risk Limits Static Fixture", "preview_risk_limits_static_fixture"),
    ("FUNCTIONAL-PREVIEW-12.3", "Kill Switch Read Model", "preview_kill_switch_read_model"),
    ("FUNCTIONAL-PREVIEW-12.4", "Risk Governor Gate Matrix", "preview_risk_governor_gate_matrix"),
    ("FUNCTIONAL-PREVIEW-12.5", "Block J Closure Audit", "preview_block_j_closure_audit"),
]

EXPECTED_BLOCKED_CAPABILITIES = [
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


def _assert_plain(value: Any) -> None:
    assert value is None or isinstance(value, str | bool | int | float | list | dict)
    if isinstance(value, list):
        for item in value:
            _assert_plain(item)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_plain(item)


def test_block_j_closure_audit_identity_and_plain_payload() -> None:
    audit = build_preview_block_j_closure_audit()

    _assert_plain(audit)
    json.dumps(audit, sort_keys=True)
    assert list(audit) == TOP_LEVEL_FIELDS
    assert audit["schema_version"] == PREVIEW_BLOCK_J_CLOSURE_AUDIT_SCHEMA_VERSION
    assert audit["block_j_closure_audit_kind"] == PREVIEW_BLOCK_J_CLOSURE_AUDIT_KIND
    assert audit["block"] == BLOCK_ID
    assert audit["step"] == STEP_ID
    assert audit["block_j_closure_audit_status"] == BLOCK_J_CLOSURE_AUDIT_STATUS
    assert audit["block_j_closure_audit_decision"] == BLOCK_J_CLOSURE_AUDIT_DECISION
    assert audit["ready_for_next_block"] is READY_FOR_NEXT_BLOCK
    assert audit["next_block"] == NEXT_BLOCK
    assert audit["next_step"] == NEXT_STEP
    assert audit["next_step_title"] == NEXT_STEP_TITLE
    assert audit["closure_line"] == CLOSURE_LINE
    assert audit["status"] == STATUS


def test_references_are_safe_ready_subsets() -> None:
    audit = build_preview_block_j_closure_audit()

    assert audit["risk_contract_reference"]["ready_for_block_j_1"] is True
    assert audit["risk_contract_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.1"
    assert audit["risk_limits_read_model_reference"]["ready_for_block_j_2"] is True
    assert audit["risk_limits_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.2"
    assert audit["risk_limits_static_fixture_reference"]["ready_for_block_j_3"] is True
    assert audit["risk_limits_static_fixture_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.3"
    assert audit["kill_switch_read_model_reference"]["ready_for_block_j_4"] is True
    assert audit["kill_switch_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.4"
    assert audit["risk_governor_gate_matrix_reference"]["ready_for_block_j_5"] is True
    assert audit["risk_governor_gate_matrix_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.5"
    assert (
        audit["risk_governor_gate_matrix_reference"]["next_step_title"] == "BLOCK J CLOSURE AUDIT"
    )


def test_closure_scope_and_completed_steps() -> None:
    audit = build_preview_block_j_closure_audit()
    scope = audit["block_j_closure_scope"]

    assert scope["scope_name"] == "block_j_closure_audit"
    assert scope["closure_audit_only"] is True
    assert scope["closes_block_j"] is True
    assert scope["derived_from_risk_contract_12_0"] is True
    assert scope["derived_from_limits_read_model_12_1"] is True
    assert scope["derived_from_static_fixture_12_2"] is True
    assert scope["derived_from_kill_switch_read_model_12_3"] is True
    assert scope["derived_from_gate_matrix_12_4"] is True
    assert scope["exe_direction_preserved"] is True
    for key in FALSE_SCOPE_FLAGS:
        assert scope[key] is False

    assert len(audit["block_j_completed_steps"]) == 6
    for item, (step, title, artifact) in zip(
        audit["block_j_completed_steps"], EXPECTED_STEPS, strict=True
    ):
        assert item == {
            "step": step,
            "title": title,
            "artifact": artifact,
            "status": "complete",
            "runtime_enabled": False,
            "order_flow_enabled": False,
            "private_endpoint_enabled": False,
            "network_enabled": False,
            "ready_for_next_step": True,
        }


def test_exact_matrices_and_lists() -> None:
    audit = build_preview_block_j_closure_audit()

    assert audit["block_j_completion_matrix"] == {
        "completed_step_count": 6,
        "expected_step_count": 6,
        "all_steps_complete": True,
        "all_references_ready": True,
        "all_runtime_paths_blocked": True,
        "all_order_flow_paths_blocked": True,
        "all_private_endpoint_paths_blocked": True,
        "all_network_paths_blocked": True,
        "all_live_trading_paths_blocked": True,
        "ready_for_next_block": True,
        "closure_line": CLOSURE_LINE,
    }
    assert audit["block_j_safety_summary"] == {
        "risk_contract_complete": True,
        "limits_read_model_complete": True,
        "static_fixture_complete": True,
        "kill_switch_read_model_complete": True,
        "gate_matrix_complete": True,
        "closure_audit_complete": True,
        "runtime_enforcement_enabled": False,
        "limit_enforcement_enabled": False,
        "kill_switch_runtime_trigger_enabled": False,
        "manual_trigger_enabled": False,
        "automatic_trigger_enabled": False,
        "order_flow_enabled": False,
        "private_endpoint_enabled": False,
        "network_io_enabled": False,
        "live_trading_enabled": False,
        "safe_to_enter_next_block": True,
    }
    assert audit["risk_governor_capability_closure"] == {
        "risk_governor_contract_defined": True,
        "limits_read_model_defined": True,
        "static_limit_fixture_defined": True,
        "kill_switch_read_model_defined": True,
        "gate_matrix_defined": True,
        "closure_audit_defined": True,
        "risk_governor_runtime_enabled": False,
        "limit_runtime_enforcement_enabled": False,
        "kill_switch_runtime_enabled": False,
        "gate_matrix_runtime_enabled": False,
        "runtime_activation_requires_future_block": True,
        "order_flow_activation_requires_future_block": True,
        "private_endpoint_activation_requires_future_block": True,
        "network_activation_requires_future_block": True,
        "live_trading_activation_requires_later_live_canary": True,
    }
    assert all(value is False for value in audit["runtime_order_private_network_closure"].values())
    assert audit["blocked_block_j_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    assert audit["block_j_source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert audit["next_block_entry_requirements"] == {
        "requires_observability_contract": True,
        "requires_audit_envelope_contract": True,
        "requires_rollback_contract": True,
        "requires_soak_contract": True,
        "requires_no_runtime_activation_at_entry": True,
        "requires_order_flow_to_remain_blocked": True,
        "requires_private_endpoint_to_remain_blocked": True,
        "requires_network_io_to_remain_blocked_until_explicit_gate": True,
        "requires_live_trading_to_remain_blocked_until_live_canary": True,
        "requires_exe_direction_to_remain_preserved": True,
    }
    assert audit["future_blocks"] == [
        "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK",
        "BLOK L — LIVE CANARY / LIVE TRANSITION GATES",
        "RELEASE — EXE PACKAGING / INSTALLER / SIGNING",
    ]


def test_non_activation_evidence_true_and_false_values() -> None:
    evidence = build_preview_block_j_closure_audit()["non_activation_evidence"]

    for key in [
        "risk_contract_12_0_read",
        "risk_limits_read_model_12_1_read",
        "risk_limits_static_fixture_12_2_read",
        "kill_switch_read_model_12_3_read",
        "risk_governor_gate_matrix_12_4_read",
        "block_j_closure_audit_built",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if not key.endswith("read") and key != "block_j_closure_audit_built":
            assert value is False


def test_source_imports_and_forbidden_tokens_are_blocked() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    imported_modules = [node.module for node in imports if isinstance(node, ast.ImportFrom)] + [
        alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names
    ]

    assert imported_modules == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_kill_switch_read_model",
        "ui.pyside_app.preview_risk_governor_gate_matrix",
        "ui.pyside_app.preview_risk_governor_limits_kill_switch_contract",
        "ui.pyside_app.preview_risk_governor_limits_read_model",
        "ui.pyside_app.preview_risk_limits_static_fixture",
    ]
    forbidden = [
        "open(",
        "read_text",
        "write_text",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "getaddrinfo",
        "create_connection",
        "QQmlApplicationEngine",
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
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
        "export",
        "cc" + "xt",
    ]
    for token in forbidden:
        assert token not in source
