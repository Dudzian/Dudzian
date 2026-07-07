"""Tests for FUNCTIONAL-PREVIEW-13.1 Block K observability read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_observability_read_model import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    OBSERVABILITY_READ_MODEL_DECISION,
    OBSERVABILITY_READ_MODEL_STATUS,
    PREVIEW_OBSERVABILITY_READ_MODEL_KIND,
    PREVIEW_OBSERVABILITY_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_2,
    STATUS,
    STEP_ID,
    build_preview_observability_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_observability_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "observability_read_model_kind",
    "block",
    "step",
    "observability_read_model_status",
    "observability_read_model_decision",
    "ready_for_block_k_2",
    "next_step",
    "next_step_title",
    "block_k_contract_reference",
    "observability_read_model_scope",
    "observability_read_model_entries",
    "default_observability_read_model_selection",
    "observability_read_model_summary",
    "observability_read_model_matrix",
    "observability_surface_contract",
    "blocked_observability_read_model_capabilities",
    "observability_read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]

FALSE_SCOPE_FLAGS = [
    "observability_runtime_allowed_now",
    "runtime_metrics_collection_allowed_now",
    "metrics_export_allowed_now",
    "log_file_read_allowed_now",
    "log_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "audit_writer_allowed_now",
    "audit_export_allowed_now",
    "rollback_execution_allowed_now",
    "soak_runtime_allowed_now",
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

ENTRY_FIELDS = [
    "observability_read_model_id",
    "source_observability_id",
    "display_name",
    "read_model_classification",
    "observation_domain",
    "planned_signal_source",
    "planned_signal_type",
    "planned_display_group",
    "operator_visibility",
    "eligible_for_13_5_gate_matrix",
    "runtime_collection_allowed_now",
    "metrics_export_allowed_now",
    "log_file_read_allowed_now",
    "log_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "network_io_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "safe_for_offline_tests",
    "notes",
]

ENTRY_DEFINITIONS = [
    (
        "runtime_liveness",
        "Runtime liveness",
        "runtime",
        "future_runtime_liveness_probe",
        "status",
        "runtime_health",
    ),
    (
        "runtime_safety_gate",
        "Runtime safety gate",
        "safety",
        "future_safety_gate_probe",
        "gate_state",
        "safety",
    ),
    (
        "audit_envelope_health",
        "Audit envelope health",
        "audit",
        "future_audit_envelope_probe",
        "health",
        "audit",
    ),
    (
        "rollback_readiness",
        "Rollback readiness",
        "rollback",
        "future_rollback_readiness_probe",
        "readiness",
        "rollback",
    ),
    (
        "soak_readiness",
        "Soak readiness",
        "soak",
        "future_soak_readiness_probe",
        "readiness",
        "soak",
    ),
    (
        "order_flow_block_state",
        "Order flow block state",
        "order_flow",
        "future_order_flow_gate_probe",
        "block_state",
        "execution_safety",
    ),
    (
        "private_endpoint_block_state",
        "Private endpoint block state",
        "private_endpoint",
        "future_private_endpoint_gate_probe",
        "block_state",
        "execution_safety",
    ),
    (
        "network_block_state",
        "Network block state",
        "network",
        "future_network_gate_probe",
        "block_state",
        "execution_safety",
    ),
]
EXPECTED_IDS = [f"observability_read_model_{entry[0]}" for entry in ENTRY_DEFINITIONS]

EXPECTED_BLOCKED_CAPABILITIES = [
    "observability runtime collection",
    "metrics collection",
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
    return build_preview_observability_read_model()


def test_payload_is_plain_json_serializable_dict() -> None:
    payload = _payload()
    assert isinstance(payload, dict)
    assert json.loads(json.dumps(payload)) == payload


def test_top_level_identity_status_decision_and_next_step() -> None:
    payload = _payload()
    assert list(payload) == TOP_LEVEL_FIELDS
    assert payload["schema_version"] == PREVIEW_OBSERVABILITY_READ_MODEL_SCHEMA_VERSION
    assert payload["observability_read_model_kind"] == PREVIEW_OBSERVABILITY_READ_MODEL_KIND
    assert payload["block"] == BLOCK_ID == "K"
    assert payload["step"] == STEP_ID == "13.1"
    assert payload["observability_read_model_status"] == OBSERVABILITY_READ_MODEL_STATUS
    assert payload["observability_read_model_decision"] == OBSERVABILITY_READ_MODEL_DECISION
    assert payload["ready_for_block_k_2"] is READY_FOR_BLOCK_K_2 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-13.2"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "AUDIT ENVELOPE READ MODEL"
    assert payload["status"] == STATUS


def test_block_k_contract_reference_points_to_13_0() -> None:
    reference = _payload()["block_k_contract_reference"]
    assert list(reference) == [
        "schema_version",
        "observability_audit_rollback_soak_contract_kind",
        "observability_audit_rollback_soak_contract_status",
        "observability_audit_rollback_soak_contract_decision",
        "ready_for_block_k_1",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert reference["ready_for_block_k_1"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-13.1"
    assert reference["next_step_title"] == "OBSERVABILITY READ MODEL"


def test_scope_is_read_model_only_and_all_runtime_io_flags_are_false() -> None:
    scope = _payload()["observability_read_model_scope"]
    assert scope["scope_name"] == "observability_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_block_k_contract_13_0"] is True
    assert scope["exe_direction_preserved"] is True
    for key in FALSE_SCOPE_FLAGS:
        assert scope[key] is False


def test_entries_have_exact_order_fields_values_and_safe_flags() -> None:
    entries = _payload()["observability_read_model_entries"]
    assert [entry["source_observability_id"] for entry in entries] == [
        entry[0] for entry in ENTRY_DEFINITIONS
    ]
    assert len(entries) == 8
    for entry, definition in zip(entries, ENTRY_DEFINITIONS, strict=True):
        source_id, display_name, domain, signal_source, signal_type, display_group = definition
        assert list(entry) == ENTRY_FIELDS
        assert entry["observability_read_model_id"] == f"observability_read_model_{source_id}"
        assert entry["source_observability_id"] == source_id
        assert entry["display_name"] == display_name
        assert entry["read_model_classification"] == "static_observability_read_model_only"
        assert entry["observation_domain"] == domain
        assert entry["planned_signal_source"] == signal_source
        assert entry["planned_signal_type"] == signal_type
        assert entry["planned_display_group"] == display_group
        assert entry["operator_visibility"] == "future_read_only_observability"
        assert entry["eligible_for_13_5_gate_matrix"] is True
        assert entry["runtime_collection_allowed_now"] is False
        assert entry["metrics_export_allowed_now"] is False
        assert entry["log_file_read_allowed_now"] is False
        assert entry["log_file_write_allowed_now"] is False
        assert entry["filesystem_io_allowed_now"] is False
        assert entry["network_io_allowed_now"] is False
        assert entry["order_flow_allowed_now"] is False
        assert entry["private_endpoint_access_allowed_now"] is False
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]


def test_default_selection_summary_matrix_and_surface_contract_are_exact() -> None:
    payload = _payload()
    assert payload["default_observability_read_model_selection"] == {
        "observability_read_model_id": "observability_read_model_runtime_liveness",
        "source_observability_id": "runtime_liveness",
        "reason": "first observability read-model signal; static only, no runtime collection, no exports",
        "runtime_collection_allowed_now": False,
        "metrics_export_allowed_now": False,
    }
    assert payload["observability_read_model_summary"] == {
        "entry_count": 8,
        "default_selection_id": "observability_read_model_runtime_liveness",
        "runtime_collection_enabled_entry_count": 0,
        "metrics_export_enabled_entry_count": 0,
        "log_file_read_enabled_entry_count": 0,
        "log_file_write_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "offline_safe_entry_count": 8,
        "entries_eligible_for_13_5_gate_matrix": 8,
        "runtime_domain_entry_count": 1,
        "safety_domain_entry_count": 1,
        "audit_domain_entry_count": 1,
        "rollback_domain_entry_count": 1,
        "soak_domain_entry_count": 1,
        "execution_safety_entry_count": 3,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_metrics_collection_now": False,
        "safe_for_export_now": False,
    }
    assert payload["observability_read_model_matrix"] == {
        "observability_read_model_ids": EXPECTED_IDS,
        "runtime_domain_ids": ["observability_read_model_runtime_liveness"],
        "safety_domain_ids": ["observability_read_model_runtime_safety_gate"],
        "audit_domain_ids": ["observability_read_model_audit_envelope_health"],
        "rollback_domain_ids": ["observability_read_model_rollback_readiness"],
        "soak_domain_ids": ["observability_read_model_soak_readiness"],
        "execution_safety_domain_ids": [
            "observability_read_model_order_flow_block_state",
            "observability_read_model_private_endpoint_block_state",
            "observability_read_model_network_block_state",
        ],
        "entries_requiring_13_5_gate_matrix": EXPECTED_IDS,
        "entries_never_runtime_collected_in_13_1": EXPECTED_IDS,
        "planned_signal_sources_by_id": {
            f"observability_read_model_{entry[0]}": entry[3] for entry in ENTRY_DEFINITIONS
        },
        "planned_display_groups_by_id": {
            f"observability_read_model_{entry[0]}": entry[5] for entry in ENTRY_DEFINITIONS
        },
    }
    assert payload["observability_surface_contract"] == {
        "surface_contract_id": "block_k_observability_read_model_surface_contract",
        "read_model_is_static": True,
        "signals_are_planned_only": True,
        "signals_are_not_collected_now": True,
        "metrics_are_not_exported_now": True,
        "logs_are_not_read_now": True,
        "logs_are_not_written_now": True,
        "filesystem_io_forbidden_now": True,
        "network_io_forbidden_now": True,
        "runtime_collection_requires_future_gate": True,
        "ui_surface_requires_future_qml_gate": True,
    }


def test_blocked_capabilities_boundaries_non_activation_and_future_steps_are_exact() -> None:
    payload = _payload()
    assert payload["blocked_observability_read_model_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    boundaries = payload["observability_read_model_boundaries"]
    assert "observability_read_model_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = payload["non_activation_evidence"]
    assert evidence["block_k_contract_13_0_read"] is True
    assert evidence["observability_read_model_built"] is True
    for key, value in evidence.items():
        if key not in {"block_k_contract_13_0_read", "observability_read_model_built"}:
            assert value is False
    assert payload["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert payload["future_steps"] == [
        "functional_preview_13_2_audit_envelope_read_model",
        "functional_preview_13_3_rollback_read_model",
        "functional_preview_13_4_soak_read_model",
        "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
        "functional_preview_13_6_block_k_closure_audit",
    ]


def test_source_imports_only_safe_typing_and_13_0_contract_helper() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [
        (node.module, [alias.name for alias in node.names])
        for node in imports
        if isinstance(node, ast.ImportFrom)
    ] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        (
            "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
            ["build_preview_observability_audit_rollback_soak_contract"],
        ),
    ]
    assert not [node for node in imports if isinstance(node, ast.Import)]


def test_source_has_no_forbidden_imports_or_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_import_parts = [
        "PySide",
        "qml",
        "runtime",
        "scheduler",
        "trading_controller",
        "decision_envelope",
        "strategy",
        "scoring",
        "recommendation",
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
        "observability",
        "logging",
        "exporter",
        "metrics",
        "rollback",
        "soak",
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
        "cc" + "xt",
    ]
    allowed_modules = {
        "__future__",
        "typing",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed_modules
        elif isinstance(node, ast.Import):
            raise AssertionError("plain imports are not allowed")
    imported_modules = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    for module in imported_modules:
        if module in allowed_modules:
            continue
        for part in forbidden_import_parts:
            assert part not in module

    forbidden_calls = [
        "open",
        "read_text",
        "write_text",
        "getenv",
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
        "export",
    ]
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = {func.id for func in calls if isinstance(func, ast.Name)}
    call_names.update(func.attr for func in calls if isinstance(func, ast.Attribute))
    for forbidden_call in forbidden_calls:
        assert forbidden_call not in call_names
