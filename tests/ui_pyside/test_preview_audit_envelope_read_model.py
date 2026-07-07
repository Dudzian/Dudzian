"""Tests for FUNCTIONAL-PREVIEW-13.2 Block K audit envelope read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_audit_envelope_read_model import (
    AUDIT_ENVELOPE_READ_MODEL_DECISION,
    AUDIT_ENVELOPE_READ_MODEL_STATUS,
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_AUDIT_ENVELOPE_READ_MODEL_KIND,
    PREVIEW_AUDIT_ENVELOPE_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_3,
    STATUS,
    STEP_ID,
    build_preview_audit_envelope_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_audit_envelope_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "audit_envelope_read_model_kind",
    "block",
    "step",
    "audit_envelope_read_model_status",
    "audit_envelope_read_model_decision",
    "ready_for_block_k_3",
    "next_step",
    "next_step_title",
    "block_k_contract_reference",
    "observability_read_model_reference",
    "audit_envelope_read_model_scope",
    "audit_envelope_read_model_entries",
    "default_audit_envelope_read_model_selection",
    "audit_envelope_required_fields",
    "audit_envelope_read_model_summary",
    "audit_envelope_read_model_matrix",
    "audit_envelope_surface_contract",
    "blocked_audit_envelope_read_model_capabilities",
    "audit_envelope_read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]

ENTRY_FIELDS = [
    "audit_envelope_read_model_id",
    "source_audit_event_id",
    "display_name",
    "read_model_classification",
    "audit_domain",
    "planned_event_source",
    "planned_event_type",
    "required_envelope_profile",
    "planned_retention_class",
    "operator_visibility",
    "eligible_for_13_5_gate_matrix",
    "audit_writer_allowed_now",
    "audit_export_allowed_now",
    "audit_file_read_allowed_now",
    "audit_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "network_io_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "safe_for_offline_tests",
    "notes",
]

ENTRY_DEFINITIONS = [
    (
        "runtime_lifecycle_event",
        "Runtime lifecycle event",
        "runtime",
        "future_runtime_lifecycle_audit",
        "lifecycle",
        "runtime_lifecycle_profile",
        "local_short_retention_planned_only",
    ),
    (
        "decision_event",
        "Decision event",
        "decision",
        "future_decision_audit",
        "decision",
        "decision_profile",
        "local_standard_retention_planned_only",
    ),
    (
        "risk_gate_event",
        "Risk gate event",
        "risk",
        "future_risk_gate_audit",
        "gate_state",
        "risk_gate_profile",
        "local_standard_retention_planned_only",
    ),
    (
        "order_flow_event",
        "Order flow event",
        "order_flow",
        "future_order_flow_audit",
        "order_flow",
        "order_flow_profile",
        "local_extended_retention_planned_only",
    ),
    (
        "private_endpoint_gate_event",
        "Private endpoint gate event",
        "private_endpoint",
        "future_private_endpoint_gate_audit",
        "gate_state",
        "private_endpoint_gate_profile",
        "local_standard_retention_planned_only",
    ),
    (
        "network_gate_event",
        "Network gate event",
        "network",
        "future_network_gate_audit",
        "gate_state",
        "network_gate_profile",
        "local_standard_retention_planned_only",
    ),
    (
        "rollback_event",
        "Rollback event",
        "rollback",
        "future_rollback_audit",
        "rollback",
        "rollback_profile",
        "local_extended_retention_planned_only",
    ),
    (
        "soak_event",
        "Soak event",
        "soak",
        "future_soak_audit",
        "soak",
        "soak_profile",
        "local_short_retention_planned_only",
    ),
]
EXPECTED_IDS = [f"audit_envelope_read_model_{entry[0]}" for entry in ENTRY_DEFINITIONS]
REQUIRED_FIELDS = [
    "event_id",
    "event_type",
    "event_time",
    "source_component",
    "block_id",
    "step_id",
    "mode",
    "safety_state",
    "risk_state",
    "operator_action_id",
    "correlation_id",
    "payload_summary",
]
FALSE_SCOPE_FLAGS = [
    "audit_writer_allowed_now",
    "audit_export_allowed_now",
    "audit_file_read_allowed_now",
    "audit_file_write_allowed_now",
    "log_file_read_allowed_now",
    "log_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "observability_runtime_allowed_now",
    "runtime_metrics_collection_allowed_now",
    "metrics_export_allowed_now",
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
EXPECTED_BLOCKED = [
    "audit writer",
    "audit export",
    "audit file read",
    "audit file write",
    "log file read",
    "log file write",
    "filesystem I/O",
    "observability runtime collection",
    "metrics collection",
    "metrics export",
    "rollback execution",
    "runtime shutdown",
    "soak runtime",
    "soak scheduler",
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
    "no audit writer import",
    "no audit exporter import",
    "no rollback runner import",
    "no soak runner import",
    "no filesystem I/O",
    "no audit file read",
    "no audit file write",
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


def _model() -> dict[str, Any]:
    return build_preview_audit_envelope_read_model()


def test_plain_serializable_identity_and_references() -> None:
    model = _model()
    assert list(model) == TOP_LEVEL_FIELDS
    assert json.loads(json.dumps(model)) == model
    assert model["schema_version"] == PREVIEW_AUDIT_ENVELOPE_READ_MODEL_SCHEMA_VERSION
    assert model["audit_envelope_read_model_kind"] == PREVIEW_AUDIT_ENVELOPE_READ_MODEL_KIND
    assert model["block"] == BLOCK_ID
    assert model["step"] == STEP_ID
    assert model["audit_envelope_read_model_status"] == AUDIT_ENVELOPE_READ_MODEL_STATUS
    assert model["audit_envelope_read_model_decision"] == AUDIT_ENVELOPE_READ_MODEL_DECISION
    assert model["ready_for_block_k_3"] is READY_FOR_BLOCK_K_3
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


def test_scope_entries_required_fields_summary_and_matrix() -> None:
    model = _model()
    scope = model["audit_envelope_read_model_scope"]
    assert scope["scope_name"] == "audit_envelope_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_block_k_contract_13_0"] is True
    assert scope["derived_from_observability_read_model_13_1"] is True
    for flag in FALSE_SCOPE_FLAGS:
        assert scope[flag] is False
    assert scope["exe_direction_preserved"] is True

    entries = model["audit_envelope_read_model_entries"]
    assert [entry["source_audit_event_id"] for entry in entries] == [
        entry[0] for entry in ENTRY_DEFINITIONS
    ]
    for entry, expected in zip(entries, ENTRY_DEFINITIONS, strict=True):
        source_id, display, domain, source, event_type, profile, retention = expected
        assert list(entry) == ENTRY_FIELDS
        assert entry["audit_envelope_read_model_id"] == f"audit_envelope_read_model_{source_id}"
        assert entry["display_name"] == display
        assert entry["read_model_classification"] == "static_audit_envelope_read_model_only"
        assert entry["audit_domain"] == domain
        assert entry["planned_event_source"] == source
        assert entry["planned_event_type"] == event_type
        assert entry["required_envelope_profile"] == profile
        assert entry["planned_retention_class"] == retention
        assert entry["operator_visibility"] == "future_read_only_audit_envelope"
        assert entry["eligible_for_13_5_gate_matrix"] is True
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
        for flag in [
            "audit_writer_allowed_now",
            "audit_export_allowed_now",
            "audit_file_read_allowed_now",
            "audit_file_write_allowed_now",
            "filesystem_io_allowed_now",
            "network_io_allowed_now",
            "order_flow_allowed_now",
            "private_endpoint_access_allowed_now",
        ]:
            assert entry[flag] is False

    assert model["default_audit_envelope_read_model_selection"] == {
        "audit_envelope_read_model_id": "audit_envelope_read_model_runtime_lifecycle_event",
        "source_audit_event_id": "runtime_lifecycle_event",
        "reason": "first audit envelope read-model event; static only, no writer, no exports, no filesystem I/O",
        "audit_writer_allowed_now": False,
        "audit_export_allowed_now": False,
    }
    required = model["audit_envelope_required_fields"]
    assert [field["field_name"] for field in required] == REQUIRED_FIELDS
    for field in required:
        assert field["field_classification"] == "planned_audit_envelope_field_only"
        assert field["required_for_future_writer"] is True
        assert field["contains_secret_material"] is False
        assert field["contains_private_key_material"] is False
        assert field["allowed_to_collect_now"] is False
        assert field["notes"]

    assert model["audit_envelope_read_model_summary"] == {
        "entry_count": 8,
        "required_field_count": 12,
        "default_selection_id": "audit_envelope_read_model_runtime_lifecycle_event",
        "audit_writer_enabled_entry_count": 0,
        "audit_export_enabled_entry_count": 0,
        "audit_file_read_enabled_entry_count": 0,
        "audit_file_write_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "offline_safe_entry_count": 8,
        "entries_eligible_for_13_5_gate_matrix": 8,
        "runtime_domain_entry_count": 1,
        "decision_domain_entry_count": 1,
        "risk_domain_entry_count": 1,
        "order_flow_domain_entry_count": 1,
        "private_endpoint_domain_entry_count": 1,
        "network_domain_entry_count": 1,
        "rollback_domain_entry_count": 1,
        "soak_domain_entry_count": 1,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_audit_writer_now": False,
        "safe_for_export_now": False,
        "safe_for_filesystem_io_now": False,
    }
    matrix = model["audit_envelope_read_model_matrix"]
    assert matrix["audit_envelope_read_model_ids"] == EXPECTED_IDS
    assert matrix["entries_requiring_13_5_gate_matrix"] == EXPECTED_IDS
    assert matrix["entries_never_written_in_13_2"] == EXPECTED_IDS
    assert matrix["planned_event_sources_by_id"] == {
        f"audit_envelope_read_model_{e[0]}": e[3] for e in ENTRY_DEFINITIONS
    }
    assert matrix["required_envelope_profiles_by_id"] == {
        f"audit_envelope_read_model_{e[0]}": e[5] for e in ENTRY_DEFINITIONS
    }


def test_surface_boundaries_evidence_source_future_and_static_import_guard() -> None:
    model = _model()
    assert model["audit_envelope_surface_contract"] == {
        "surface_contract_id": "block_k_audit_envelope_read_model_surface_contract",
        "read_model_is_static": True,
        "events_are_planned_only": True,
        "events_are_not_written_now": True,
        "events_are_not_exported_now": True,
        "audit_files_are_not_read_now": True,
        "audit_files_are_not_written_now": True,
        "filesystem_io_forbidden_now": True,
        "network_io_forbidden_now": True,
        "writer_requires_future_gate": True,
        "export_requires_future_gate": True,
        "ui_surface_requires_future_qml_gate": True,
    }
    assert model["blocked_audit_envelope_read_model_capabilities"] == EXPECTED_BLOCKED
    boundaries = model["audit_envelope_read_model_boundaries"]
    assert "audit_envelope_read_model_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = model["non_activation_evidence"]
    for true_key in [
        "block_k_contract_13_0_read",
        "observability_read_model_13_1_read",
        "audit_envelope_read_model_built",
    ]:
        assert evidence[true_key] is True
    for key, value in evidence.items():
        if key not in {
            "block_k_contract_13_0_read",
            "observability_read_model_13_1_read",
            "audit_envelope_read_model_built",
        }:
            assert value is False
    assert model["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert model["future_steps"] == [
        "functional_preview_13_3_rollback_read_model",
        "functional_preview_13_4_soak_read_model",
        "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
        "functional_preview_13_6_block_k_closure_audit",
    ]

    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 4
    assert all(not isinstance(node, ast.Import) for node in imports)
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom)}
    assert modules == {
        "__future__",
        "typing",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
        "ui.pyside_app.preview_observability_read_model",
    }
    forbidden_call_names = {
        "open",
        "read_text",
        "write_text",
        "getenv",
        "getaddrinfo",
        "create_connection",
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
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
    }
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    call_names.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    assert call_names.isdisjoint(forbidden_call_names)
