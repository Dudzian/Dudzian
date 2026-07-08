"""Tests for FUNCTIONAL-PREVIEW-13.5 Block K gate matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_observability_audit_rollback_soak_gate_matrix import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_DECISION,
    OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_STATUS,
    PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_KIND,
    PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_6,
    STATUS,
    STEP_ID,
    build_preview_observability_audit_rollback_soak_gate_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_observability_audit_rollback_soak_gate_matrix.py"
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "observability_audit_rollback_soak_gate_matrix_kind",
    "block",
    "step",
    "observability_audit_rollback_soak_gate_matrix_status",
    "observability_audit_rollback_soak_gate_matrix_decision",
    "ready_for_block_k_6",
    "next_step",
    "next_step_title",
    "block_k_contract_reference",
    "observability_read_model_reference",
    "audit_envelope_read_model_reference",
    "rollback_read_model_reference",
    "soak_read_model_reference",
    "gate_matrix_scope",
    "gate_matrix_entries",
    "default_gate_matrix_selection",
    "gate_matrix_summary",
    "gate_matrix_dependency_map",
    "gate_matrix_contract",
    "blocked_gate_matrix_capabilities",
    "gate_matrix_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
ENTRY_FIELDS = [
    "gate_matrix_entry_id",
    "source_gate_id",
    "display_name",
    "gate_domain",
    "gate_type",
    "source_step",
    "required_source_status",
    "gate_matrix_state",
    "required_for_block_k_closure",
    "eligible_for_13_6_closure_audit",
    "runtime_gate_execution_allowed_now",
    "gate_state_mutation_allowed_now",
    "runtime_activation_allowed_now",
    "audit_writer_allowed_now",
    "rollback_execution_allowed_now",
    "soak_runtime_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
    "safe_for_offline_tests",
    "notes",
]
ENTRY_DEFINITIONS = [
    (
        "block_k_contract_present_gate",
        "Block K contract present gate",
        "contract",
        "source_presence",
        "FUNCTIONAL-PREVIEW-13.0",
        "observability_audit_rollback_soak_contract_ready_no_runtime",
        "passed_static_source_gate",
    ),
    (
        "observability_read_model_present_gate",
        "Observability read model present gate",
        "observability",
        "source_presence",
        "FUNCTIONAL-PREVIEW-13.1",
        "observability_read_model_ready_no_runtime_collection",
        "passed_static_source_gate",
    ),
    (
        "audit_envelope_read_model_present_gate",
        "Audit envelope read model present gate",
        "audit",
        "source_presence",
        "FUNCTIONAL-PREVIEW-13.2",
        "audit_envelope_read_model_ready_no_audit_writer_no_exports",
        "passed_static_source_gate",
    ),
    (
        "rollback_read_model_present_gate",
        "Rollback read model present gate",
        "rollback",
        "source_presence",
        "FUNCTIONAL-PREVIEW-13.3",
        "rollback_read_model_ready_no_execution_no_runtime_shutdown",
        "passed_static_source_gate",
    ),
    (
        "soak_read_model_present_gate",
        "Soak read model present gate",
        "soak",
        "source_presence",
        "FUNCTIONAL-PREVIEW-13.4",
        "soak_read_model_ready_no_runtime_no_scheduler",
        "passed_static_source_gate",
    ),
    (
        "runtime_activation_blocked_gate",
        "Runtime activation blocked gate",
        "runtime",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-13.5",
        "runtime_activation_blocked_until_future_gate",
        "blocked_until_future_gate",
    ),
    (
        "audit_writer_export_blocked_gate",
        "Audit writer export blocked gate",
        "audit",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-13.5",
        "audit_writer_export_blocked_until_future_gate",
        "blocked_until_future_gate",
    ),
    (
        "rollback_execution_blocked_gate",
        "Rollback execution blocked gate",
        "rollback",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-13.5",
        "rollback_execution_blocked_until_future_gate",
        "blocked_until_future_gate",
    ),
    (
        "soak_runtime_blocked_gate",
        "Soak runtime blocked gate",
        "soak",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-13.5",
        "soak_runtime_blocked_until_future_gate",
        "blocked_until_future_gate",
    ),
    (
        "order_private_network_blocked_gate",
        "Order private network blocked gate",
        "execution_safety",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-13.5",
        "order_private_network_blocked_until_future_gate",
        "blocked_until_future_gate",
    ),
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
    "no audit writer import",
    "no audit exporter import",
    "no rollback runner import",
    "no rollback executor import",
    "no soak runner import",
    "no soak scheduler import",
    "no filesystem I/O",
    "no audit file read",
    "no audit file write",
    "no log file read",
    "no log file write",
    "no audit write",
    "no audit export",
    "no runtime shutdown",
    "no state mutation",
    "no wall-clock runtime measurement",
    "no stability probe",
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
    "no gate execution",
    "no gate state mutation",
    "no runtime activation",
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


def _model() -> dict[str, Any]:
    return build_preview_observability_audit_rollback_soak_gate_matrix()


def test_identity_references_and_plain_data_are_exact() -> None:
    model = _model()
    json.dumps(model)
    assert list(model) == TOP_LEVEL_FIELDS
    assert (
        model["schema_version"]
        == PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_SCHEMA_VERSION
    )
    assert (
        model["observability_audit_rollback_soak_gate_matrix_kind"]
        == PREVIEW_OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_KIND
    )
    assert model["block"] == BLOCK_ID
    assert model["step"] == STEP_ID
    assert (
        model["observability_audit_rollback_soak_gate_matrix_status"]
        == OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_STATUS
    )
    assert (
        model["observability_audit_rollback_soak_gate_matrix_decision"]
        == OBSERVABILITY_AUDIT_ROLLBACK_SOAK_GATE_MATRIX_DECISION
    )
    assert model["ready_for_block_k_6"] is READY_FOR_BLOCK_K_6
    assert model["next_step"] == NEXT_STEP
    assert model["next_step_title"] == NEXT_STEP_TITLE
    assert model["status"] == STATUS
    assert model["block_k_contract_reference"]["ready_for_block_k_1"] is True
    assert model["block_k_contract_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-13.1"
    assert model["block_k_contract_reference"]["next_step_title"] == "OBSERVABILITY READ MODEL"
    assert model["observability_read_model_reference"]["ready_for_block_k_2"] is True
    assert model["observability_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-13.2"
    assert (
        model["observability_read_model_reference"]["next_step_title"]
        == "AUDIT ENVELOPE READ MODEL"
    )
    assert model["audit_envelope_read_model_reference"]["ready_for_block_k_3"] is True
    assert model["audit_envelope_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-13.3"
    assert model["audit_envelope_read_model_reference"]["next_step_title"] == "ROLLBACK READ MODEL"
    assert model["rollback_read_model_reference"]["ready_for_block_k_4"] is True
    assert model["rollback_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-13.4"
    assert model["rollback_read_model_reference"]["next_step_title"] == "SOAK READ MODEL"
    assert model["soak_read_model_reference"]["ready_for_block_k_5"] is True
    assert model["soak_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-13.5"
    assert (
        model["soak_read_model_reference"]["next_step_title"]
        == "OBSERVABILITY AUDIT ROLLBACK SOAK GATE MATRIX"
    )


def test_scope_entries_default_summary_dependency_and_contract_are_exact() -> None:
    model = _model()
    scope = model["gate_matrix_scope"]
    assert scope["scope_name"] == "observability_audit_rollback_soak_gate_matrix"
    assert scope["gate_matrix_only"] is True
    for key in [
        "derived_from_block_k_contract_13_0",
        "derived_from_observability_read_model_13_1",
        "derived_from_audit_envelope_read_model_13_2",
        "derived_from_rollback_read_model_13_3",
        "derived_from_soak_read_model_13_4",
        "exe_direction_preserved",
    ]:
        assert scope[key] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key.endswith("_allowed_now") or key.endswith("_in_scope")
    )
    assert scope["qml_changes_allowed"] is False
    assert scope["new_qml_method_calls_allowed"] is False
    assert scope["bridge_api_changes_allowed"] is False
    assert scope["bat_productization_allowed"] is False

    entries = model["gate_matrix_entries"]
    assert [entry["source_gate_id"] for entry in entries] == [item[0] for item in ENTRY_DEFINITIONS]
    for entry, definition in zip(entries, ENTRY_DEFINITIONS, strict=True):
        source_id, display, domain, gate_type, source_step, status, state = definition
        assert list(entry) == ENTRY_FIELDS
        assert entry["gate_matrix_entry_id"] == f"gate_matrix_{source_id}"
        assert entry["display_name"] == display
        assert entry["gate_domain"] == domain
        assert entry["gate_type"] == gate_type
        assert entry["source_step"] == source_step
        assert entry["required_source_status"] == status
        assert entry["gate_matrix_state"] == state
        assert entry["required_for_block_k_closure"] is True
        assert entry["eligible_for_13_6_closure_audit"] is True
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
        assert all(entry[key] is False for key in ENTRY_FIELDS if key.endswith("_allowed_now"))

    assert model["default_gate_matrix_selection"] == {
        "gate_matrix_entry_id": "gate_matrix_block_k_contract_present_gate",
        "source_gate_id": "block_k_contract_present_gate",
        "reason": "first Block K gate matrix entry; static source gate only, no runtime execution",
        "runtime_gate_execution_allowed_now": False,
        "gate_state_mutation_allowed_now": False,
    }
    assert model["gate_matrix_summary"] == {
        "entry_count": 10,
        "static_source_gate_count": 5,
        "blocked_capability_gate_count": 5,
        "passed_static_source_gate_count": 5,
        "blocked_until_future_gate_count": 5,
        "runtime_gate_execution_enabled_entry_count": 0,
        "gate_state_mutation_enabled_entry_count": 0,
        "runtime_activation_enabled_entry_count": 0,
        "audit_writer_enabled_entry_count": 0,
        "rollback_execution_enabled_entry_count": 0,
        "soak_runtime_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_io_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "offline_safe_entry_count": 10,
        "entries_required_for_block_k_closure": 10,
        "entries_eligible_for_13_6_closure_audit": 10,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_gate_execution_now": False,
        "safe_for_export_now": False,
    }
    assert model["gate_matrix_dependency_map"] == {
        "source_steps_in_order": [
            "FUNCTIONAL-PREVIEW-13.0",
            "FUNCTIONAL-PREVIEW-13.1",
            "FUNCTIONAL-PREVIEW-13.2",
            "FUNCTIONAL-PREVIEW-13.3",
            "FUNCTIONAL-PREVIEW-13.4",
        ],
        "target_step": "FUNCTIONAL-PREVIEW-13.5",
        "next_step": "FUNCTIONAL-PREVIEW-13.6",
        "required_ready_flags": [
            "ready_for_block_k_1",
            "ready_for_block_k_2",
            "ready_for_block_k_3",
            "ready_for_block_k_4",
            "ready_for_block_k_5",
        ],
        "required_static_source_statuses": [item[5] for item in ENTRY_DEFINITIONS[:5]],
        "requires_runtime_blocked": True,
        "requires_audit_writer_export_blocked": True,
        "requires_rollback_execution_blocked": True,
        "requires_soak_runtime_blocked": True,
        "requires_order_flow_blocked": True,
        "requires_private_endpoint_blocked": True,
        "requires_network_blocked": True,
        "requires_filesystem_io_blocked": True,
        "ready_for_13_6_block_k_closure_audit": True,
    }
    assert model["gate_matrix_contract"] == {
        "contract_id": "block_k_observability_audit_rollback_soak_gate_matrix_contract",
        "gate_matrix_is_static": True,
        "all_source_models_present": True,
        "all_source_models_are_read_only": True,
        "all_source_models_safe_for_offline_tests": True,
        "runtime_activation_blocked_until_future_gate": True,
        "audit_writer_export_blocked_until_future_gate": True,
        "rollback_execution_blocked_until_future_gate": True,
        "soak_runtime_blocked_until_future_gate": True,
        "order_flow_blocked_until_future_gate": True,
        "private_endpoint_blocked_until_future_gate": True,
        "network_io_blocked_until_future_gate": True,
        "filesystem_io_blocked_until_future_gate": True,
        "closure_audit_required_next": True,
    }


def test_capabilities_boundaries_evidence_source_boundaries_and_future_steps_are_exact() -> None:
    model = _model()
    assert model["blocked_gate_matrix_capabilities"] == [
        "runtime gate execution",
        "gate state mutation",
        "runtime activation",
        "observability runtime collection",
        "metrics collection",
        "metrics export",
        "audit writer",
        "audit export",
        "audit file read",
        "audit file write",
        "log file read",
        "log file write",
        "rollback execution",
        "runtime shutdown",
        "soak runtime",
        "soak scheduler",
        "runtime loop",
        "wall-clock runtime measurement",
        "stability probe",
        "state mutation",
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
        "scheduler",
        "filesystem I/O",
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
    boundaries = model["gate_matrix_boundaries"]
    assert "gate_matrix_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = model["non_activation_evidence"]
    true_keys = {
        "block_k_contract_13_0_read",
        "observability_read_model_13_1_read",
        "audit_envelope_read_model_13_2_read",
        "rollback_read_model_13_3_read",
        "soak_read_model_13_4_read",
        "gate_matrix_built",
    }
    assert {key for key, value in evidence.items() if value is True} == true_keys
    assert all(value is False for key, value in evidence.items() if key not in true_keys)
    assert model["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert model["future_steps"] == ["functional_preview_13_6_block_k_closure_audit"]


def test_source_imports_and_forbidden_runtime_calls_remain_blocked() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    imported_modules = [node.module for node in imports if isinstance(node, ast.ImportFrom)]
    assert imported_modules == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_audit_envelope_read_model",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
        "ui.pyside_app.preview_observability_read_model",
        "ui.pyside_app.preview_rollback_read_model",
        "ui.pyside_app.preview_soak_read_model",
    ]
    assert not [node for node in imports if isinstance(node, ast.Import)]
    forbidden_modules = {
        "yaml",
        "json",
        "os",
        "pathlib",
        "subprocess",
        "requests",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "time",
        "datetime",
        "threading",
        "asyncio",
    }
    assert not (set(imported_modules) & forbidden_modules)
    forbidden_calls = {
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
        "shutdown_runtime",
        "mutate_state",
        "start_soak",
        "run_soak",
        "start_scheduler",
        "schedule",
        "sleep",
        "time",
        "execute_gate",
        "activate_runtime",
        "mutate_gate",
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
    }
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attr_call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not ((call_names | attr_call_names) & forbidden_calls)
    assert "fetch_" + "balance" not in source
    assert "cc" + "xt" not in source
