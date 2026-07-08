"""Tests for FUNCTIONAL-PREVIEW-13.3 Block K rollback read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_rollback_read_model import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_ROLLBACK_READ_MODEL_KIND,
    PREVIEW_ROLLBACK_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_4,
    ROLLBACK_READ_MODEL_DECISION,
    ROLLBACK_READ_MODEL_STATUS,
    STATUS,
    STEP_ID,
    build_preview_rollback_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_rollback_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "rollback_read_model_kind",
    "block",
    "step",
    "rollback_read_model_status",
    "rollback_read_model_decision",
    "ready_for_block_k_4",
    "next_step",
    "next_step_title",
    "block_k_contract_reference",
    "observability_read_model_reference",
    "audit_envelope_read_model_reference",
    "rollback_read_model_scope",
    "rollback_read_model_entries",
    "default_rollback_read_model_selection",
    "rollback_readiness_required_fields",
    "rollback_read_model_summary",
    "rollback_read_model_matrix",
    "rollback_surface_contract",
    "blocked_rollback_read_model_capabilities",
    "rollback_read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
ENTRY_FIELDS = [
    "rollback_read_model_id",
    "source_rollback_id",
    "display_name",
    "read_model_classification",
    "rollback_domain",
    "planned_rollback_source",
    "planned_rollback_type",
    "required_readiness_profile",
    "planned_operator_action",
    "planned_audit_event_id",
    "operator_visibility",
    "eligible_for_13_5_gate_matrix",
    "rollback_execution_allowed_now",
    "runtime_shutdown_allowed_now",
    "state_mutation_allowed_now",
    "order_cancel_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "filesystem_io_allowed_now",
    "safe_for_offline_tests",
    "notes",
]
ENTRY_DEFINITIONS = [
    (
        "operator_requested_rollback",
        "Operator requested rollback",
        "operator",
        "future_operator_rollback_request",
        "manual_request",
        "operator_rollback_readiness_profile",
        "future_operator_confirmed_rollback",
        "runtime_lifecycle_event",
    ),
    (
        "runtime_health_rollback",
        "Runtime health rollback",
        "runtime",
        "future_runtime_health_failure",
        "runtime_health",
        "runtime_health_rollback_readiness_profile",
        "future_operator_review_required",
        "runtime_lifecycle_event",
    ),
    (
        "risk_gate_rollback",
        "Risk gate rollback",
        "risk",
        "future_risk_gate_failure",
        "risk_gate",
        "risk_gate_rollback_readiness_profile",
        "future_operator_review_required",
        "risk_gate_event",
    ),
    (
        "audit_failure_rollback",
        "Audit failure rollback",
        "audit",
        "future_audit_envelope_failure",
        "audit_failure",
        "audit_failure_rollback_readiness_profile",
        "future_operator_review_required",
        "rollback_event",
    ),
    (
        "order_flow_rollback",
        "Order flow rollback",
        "order_flow",
        "future_order_flow_failure",
        "order_flow",
        "order_flow_rollback_readiness_profile",
        "future_operator_review_required",
        "order_flow_event",
    ),
    (
        "private_endpoint_rollback",
        "Private endpoint rollback",
        "private_endpoint",
        "future_private_endpoint_failure",
        "private_endpoint",
        "private_endpoint_rollback_readiness_profile",
        "future_operator_review_required",
        "private_endpoint_gate_event",
    ),
    (
        "network_failure_rollback",
        "Network failure rollback",
        "network",
        "future_network_failure",
        "network",
        "network_rollback_readiness_profile",
        "future_operator_review_required",
        "network_gate_event",
    ),
    (
        "soak_failure_rollback",
        "Soak failure rollback",
        "soak",
        "future_soak_failure",
        "soak",
        "soak_rollback_readiness_profile",
        "future_operator_review_required",
        "soak_event",
    ),
]
EXPECTED_IDS = [f"rollback_read_model_{entry[0]}" for entry in ENTRY_DEFINITIONS]
REQUIRED_FIELDS = [
    "rollback_id",
    "rollback_type",
    "rollback_reason",
    "source_component",
    "required_operator_action",
    "pre_rollback_state_snapshot",
    "rollback_target_state",
    "audit_correlation_id",
    "safety_gate_state",
    "rollback_blocking_reason",
]


def _model() -> dict[str, Any]:
    return build_preview_rollback_read_model()


def test_preview_rollback_read_model_identity_and_plain_payload() -> None:
    model = _model()
    json.dumps(model, sort_keys=True)
    assert list(model) == TOP_LEVEL_FIELDS
    assert model["schema_version"] == PREVIEW_ROLLBACK_READ_MODEL_SCHEMA_VERSION
    assert model["rollback_read_model_kind"] == PREVIEW_ROLLBACK_READ_MODEL_KIND
    assert model["block"] == BLOCK_ID
    assert model["step"] == STEP_ID
    assert model["rollback_read_model_status"] == ROLLBACK_READ_MODEL_STATUS
    assert model["rollback_read_model_decision"] == ROLLBACK_READ_MODEL_DECISION
    assert model["ready_for_block_k_4"] is READY_FOR_BLOCK_K_4
    assert model["next_step"] == NEXT_STEP
    assert model["next_step_title"] == NEXT_STEP_TITLE
    assert model["status"] == STATUS


def test_preview_rollback_read_model_references_prior_block_k_steps() -> None:
    model = _model()
    block_k = model["block_k_contract_reference"]
    assert list(block_k) == [
        "schema_version",
        "observability_audit_rollback_soak_contract_kind",
        "observability_audit_rollback_soak_contract_status",
        "observability_audit_rollback_soak_contract_decision",
        "ready_for_block_k_1",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert block_k["ready_for_block_k_1"] is True
    assert block_k["next_step"] == "FUNCTIONAL-PREVIEW-13.1"
    assert block_k["next_step_title"] == "OBSERVABILITY READ MODEL"
    observability = model["observability_read_model_reference"]
    assert observability["ready_for_block_k_2"] is True
    assert observability["next_step"] == "FUNCTIONAL-PREVIEW-13.2"
    assert observability["next_step_title"] == "AUDIT ENVELOPE READ MODEL"
    audit = model["audit_envelope_read_model_reference"]
    assert audit["ready_for_block_k_3"] is True
    assert audit["next_step"] == "FUNCTIONAL-PREVIEW-13.3"
    assert audit["next_step_title"] == "ROLLBACK READ MODEL"


def test_preview_rollback_read_model_scope_is_static_and_non_runtime() -> None:
    scope = _model()["rollback_read_model_scope"]
    assert scope["scope_name"] == "rollback_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_block_k_contract_13_0"] is True
    assert scope["derived_from_observability_read_model_13_1"] is True
    assert scope["derived_from_audit_envelope_read_model_13_2"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key.endswith("_allowed_now") or key in {
            "qml_changes_allowed",
            "new_qml_method_calls_allowed",
            "bridge_api_changes_allowed",
            "exe_packaging_in_scope",
            "bat_productization_allowed",
        }:
            assert value is False, key


def test_preview_rollback_read_model_entries_defaults_and_required_fields() -> None:
    model = _model()
    entries = model["rollback_read_model_entries"]
    assert [entry["source_rollback_id"] for entry in entries] == [
        entry[0] for entry in ENTRY_DEFINITIONS
    ]
    assert [entry["rollback_read_model_id"] for entry in entries] == EXPECTED_IDS
    for entry, expected in zip(entries, ENTRY_DEFINITIONS, strict=True):
        assert list(entry) == ENTRY_FIELDS
        source_id, display, domain, source, rollback_type, profile, action, audit_id = expected
        assert entry["rollback_read_model_id"] == f"rollback_read_model_{source_id}"
        assert entry["display_name"] == display
        assert entry["rollback_domain"] == domain
        assert entry["planned_rollback_source"] == source
        assert entry["planned_rollback_type"] == rollback_type
        assert entry["required_readiness_profile"] == profile
        assert entry["planned_operator_action"] == action
        assert entry["planned_audit_event_id"] == audit_id
        assert entry["read_model_classification"] == "static_rollback_read_model_only"
        assert entry["operator_visibility"] == "future_read_only_rollback"
        assert entry["eligible_for_13_5_gate_matrix"] is True
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
        for flag in [
            "rollback_execution_allowed_now",
            "runtime_shutdown_allowed_now",
            "state_mutation_allowed_now",
            "order_cancel_allowed_now",
            "private_endpoint_access_allowed_now",
            "network_io_allowed_now",
            "filesystem_io_allowed_now",
        ]:
            assert entry[flag] is False
    assert model["default_rollback_read_model_selection"] == {
        "rollback_read_model_id": EXPECTED_IDS[0],
        "source_rollback_id": "operator_requested_rollback",
        "reason": "first rollback read-model scenario; static only, no rollback execution, no runtime shutdown",
        "rollback_execution_allowed_now": False,
        "runtime_shutdown_allowed_now": False,
    }
    required = model["rollback_readiness_required_fields"]
    assert [field["field_name"] for field in required] == REQUIRED_FIELDS
    for field in required:
        assert field["field_classification"] == "planned_rollback_readiness_field_only"
        assert field["required_for_future_executor"] is True
        assert field["contains_secret_material"] is False
        assert field["contains_private_key_material"] is False
        assert field["allowed_to_collect_now"] is False
        assert field["allowed_to_mutate_now"] is False
        assert field["notes"]


def test_preview_rollback_read_model_summary_matrix_surface_and_boundaries() -> None:
    model = _model()
    assert model["rollback_read_model_summary"] == {
        "entry_count": 8,
        "required_field_count": 10,
        "default_selection_id": EXPECTED_IDS[0],
        "rollback_execution_enabled_entry_count": 0,
        "runtime_shutdown_enabled_entry_count": 0,
        "state_mutation_enabled_entry_count": 0,
        "order_cancel_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "offline_safe_entry_count": 8,
        "entries_eligible_for_13_5_gate_matrix": 8,
        "operator_domain_entry_count": 1,
        "runtime_domain_entry_count": 1,
        "risk_domain_entry_count": 1,
        "audit_domain_entry_count": 1,
        "order_flow_domain_entry_count": 1,
        "private_endpoint_domain_entry_count": 1,
        "network_domain_entry_count": 1,
        "soak_domain_entry_count": 1,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_rollback_execution_now": False,
        "safe_for_runtime_shutdown_now": False,
        "safe_for_state_mutation_now": False,
    }
    matrix = model["rollback_read_model_matrix"]
    assert matrix["rollback_read_model_ids"] == EXPECTED_IDS
    assert matrix["entries_requiring_13_5_gate_matrix"] == EXPECTED_IDS
    assert matrix["entries_never_executed_in_13_3"] == EXPECTED_IDS
    for expected_id, expected in zip(EXPECTED_IDS, ENTRY_DEFINITIONS, strict=True):
        assert matrix["planned_rollback_sources_by_id"][expected_id] == expected[3]
        assert matrix["planned_audit_event_ids_by_id"][expected_id] == expected[7]
        assert matrix["required_readiness_profiles_by_id"][expected_id] == expected[5]
    assert model["rollback_surface_contract"] == {
        "surface_contract_id": "block_k_rollback_read_model_surface_contract",
        "read_model_is_static": True,
        "rollback_scenarios_are_planned_only": True,
        "rollback_is_not_executed_now": True,
        "runtime_shutdown_is_not_executed_now": True,
        "state_is_not_mutated_now": True,
        "orders_are_not_cancelled_now": True,
        "private_endpoints_are_not_accessed_now": True,
        "filesystem_io_forbidden_now": True,
        "network_io_forbidden_now": True,
        "executor_requires_future_gate": True,
        "ui_surface_requires_future_qml_gate": True,
    }
    boundaries = model["rollback_read_model_boundaries"]
    assert "rollback_read_model_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())


def test_preview_rollback_read_model_blocked_capabilities_evidence_boundaries_future_steps() -> (
    None
):
    model = _model()
    assert model["blocked_rollback_read_model_capabilities"][:5] == [
        "rollback execution",
        "runtime shutdown",
        "state mutation",
        "order cancel",
        "order replace",
    ]
    assert len(model["blocked_rollback_read_model_capabilities"]) == 55
    evidence = model["non_activation_evidence"]
    for key in [
        "block_k_contract_13_0_read",
        "observability_read_model_13_1_read",
        "audit_envelope_read_model_13_2_read",
        "rollback_read_model_built",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if key not in {
            "block_k_contract_13_0_read",
            "observability_read_model_13_1_read",
            "audit_envelope_read_model_13_2_read",
            "rollback_read_model_built",
        }:
            assert value is False, key
    assert model["source_boundaries"][0] == "no PySide import"
    assert model["source_boundaries"][-1] == "no workflow changes"
    assert model["future_steps"] == [
        "functional_preview_13_4_soak_read_model",
        "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
        "functional_preview_13_6_block_k_closure_audit",
    ]


def test_preview_rollback_read_model_source_imports_and_forbidden_calls() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_audit_envelope_read_model",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
        "ui.pyside_app.preview_observability_read_model",
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
        "shutdown_runtime",
        "mutate_state",
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
    called_tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_tokens.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_tokens.add(node.func.attr)
    for token in forbidden:
        assert token not in called_tokens
    assert "fetch_" + "balance" not in source
    assert "c" + "cxt" not in source
