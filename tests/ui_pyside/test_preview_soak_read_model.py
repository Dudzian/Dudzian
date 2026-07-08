"""Tests for FUNCTIONAL-PREVIEW-13.4 Block K soak read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_soak_read_model import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_SOAK_READ_MODEL_KIND,
    PREVIEW_SOAK_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_K_5,
    SOAK_READ_MODEL_DECISION,
    SOAK_READ_MODEL_STATUS,
    STATUS,
    STEP_ID,
    build_preview_soak_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_soak_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "soak_read_model_kind",
    "block",
    "step",
    "soak_read_model_status",
    "soak_read_model_decision",
    "ready_for_block_k_5",
    "next_step",
    "next_step_title",
    "block_k_contract_reference",
    "observability_read_model_reference",
    "audit_envelope_read_model_reference",
    "rollback_read_model_reference",
    "soak_read_model_scope",
    "soak_read_model_entries",
    "default_soak_read_model_selection",
    "soak_readiness_required_fields",
    "soak_read_model_summary",
    "soak_read_model_matrix",
    "soak_surface_contract",
    "blocked_soak_read_model_capabilities",
    "soak_read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
ENTRY_FIELDS = [
    "soak_read_model_id",
    "source_soak_id",
    "display_name",
    "read_model_classification",
    "soak_domain",
    "planned_soak_source",
    "planned_soak_type",
    "required_soak_profile",
    "planned_window_class",
    "planned_exit_criteria_profile",
    "planned_failure_audit_event_id",
    "operator_visibility",
    "eligible_for_13_5_gate_matrix",
    "soak_runtime_allowed_now",
    "soak_scheduler_allowed_now",
    "runtime_loop_allowed_now",
    "metrics_collection_allowed_now",
    "metrics_export_allowed_now",
    "filesystem_io_allowed_now",
    "network_io_allowed_now",
    "order_flow_allowed_now",
    "private_endpoint_access_allowed_now",
    "safe_for_offline_tests",
    "notes",
]
ENTRY_DEFINITIONS = [
    (
        "paper_runtime_soak",
        "Paper runtime soak",
        "paper_runtime",
        "future_paper_runtime_soak_plan",
        "paper_runtime_stability",
        "paper_runtime_soak_profile",
        "future_short_window_planned_only",
        "paper_runtime_exit_criteria_profile",
        "soak_event",
    ),
    (
        "testnet_runtime_soak",
        "Testnet runtime soak",
        "testnet_runtime",
        "future_testnet_runtime_soak_plan",
        "testnet_runtime_stability",
        "testnet_runtime_soak_profile",
        "future_standard_window_planned_only",
        "testnet_runtime_exit_criteria_profile",
        "soak_event",
    ),
    (
        "audit_pipeline_soak",
        "Audit pipeline soak",
        "audit",
        "future_audit_pipeline_soak_plan",
        "audit_pipeline_stability",
        "audit_pipeline_soak_profile",
        "future_short_window_planned_only",
        "audit_pipeline_exit_criteria_profile",
        "audit_failure_rollback",
    ),
    (
        "rollback_readiness_soak",
        "Rollback readiness soak",
        "rollback",
        "future_rollback_readiness_soak_plan",
        "rollback_readiness_stability",
        "rollback_readiness_soak_profile",
        "future_short_window_planned_only",
        "rollback_readiness_exit_criteria_profile",
        "rollback_event",
    ),
    (
        "risk_gate_soak",
        "Risk gate soak",
        "risk",
        "future_risk_gate_soak_plan",
        "risk_gate_stability",
        "risk_gate_soak_profile",
        "future_standard_window_planned_only",
        "risk_gate_exit_criteria_profile",
        "risk_gate_event",
    ),
    (
        "order_flow_block_soak",
        "Order flow block soak",
        "order_flow",
        "future_order_flow_block_soak_plan",
        "order_flow_block_stability",
        "order_flow_block_soak_profile",
        "future_standard_window_planned_only",
        "order_flow_block_exit_criteria_profile",
        "order_flow_event",
    ),
    (
        "private_endpoint_block_soak",
        "Private endpoint block soak",
        "private_endpoint",
        "future_private_endpoint_block_soak_plan",
        "private_endpoint_block_stability",
        "private_endpoint_block_soak_profile",
        "future_standard_window_planned_only",
        "private_endpoint_block_exit_criteria_profile",
        "private_endpoint_gate_event",
    ),
    (
        "network_block_soak",
        "Network block soak",
        "network",
        "future_network_block_soak_plan",
        "network_block_stability",
        "network_block_soak_profile",
        "future_standard_window_planned_only",
        "network_block_exit_criteria_profile",
        "network_gate_event",
    ),
]
REQUIRED_FIELDS = [
    "soak_id",
    "soak_type",
    "soak_window_class",
    "source_component",
    "required_exit_criteria",
    "failure_audit_event_id",
    "rollback_readiness_state",
    "safety_gate_state",
    "runtime_liveness_state",
    "soak_blocking_reason",
]


def _model() -> dict[str, Any]:
    return build_preview_soak_read_model()


def test_soak_read_model_identity_and_references_are_static_plain_data() -> None:
    model = _model()
    json.dumps(model)
    assert list(model) == TOP_LEVEL_FIELDS
    assert model["schema_version"] == PREVIEW_SOAK_READ_MODEL_SCHEMA_VERSION
    assert model["soak_read_model_kind"] == PREVIEW_SOAK_READ_MODEL_KIND
    assert model["block"] == BLOCK_ID
    assert model["step"] == STEP_ID
    assert model["soak_read_model_status"] == SOAK_READ_MODEL_STATUS
    assert model["soak_read_model_decision"] == SOAK_READ_MODEL_DECISION
    assert model["ready_for_block_k_5"] is READY_FOR_BLOCK_K_5
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


def test_scope_entries_required_fields_summary_and_matrix_are_exact() -> None:
    model = _model()
    scope = model["soak_read_model_scope"]
    assert scope["scope_name"] == "soak_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_block_k_contract_13_0"] is True
    assert scope["derived_from_observability_read_model_13_1"] is True
    assert scope["derived_from_audit_envelope_read_model_13_2"] is True
    assert scope["derived_from_rollback_read_model_13_3"] is True
    assert scope["exe_direction_preserved"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key.endswith("_allowed_now") or key.endswith("_in_scope")
    )
    assert scope["qml_changes_allowed"] is False
    assert scope["new_qml_method_calls_allowed"] is False
    assert scope["bridge_api_changes_allowed"] is False
    assert scope["bat_productization_allowed"] is False

    entries = model["soak_read_model_entries"]
    assert len(entries) == 8
    ids = [f"soak_read_model_{definition[0]}" for definition in ENTRY_DEFINITIONS]
    for entry, definition in zip(entries, ENTRY_DEFINITIONS, strict=True):
        source_id, display, domain, source, soak_type, profile, window, exit_profile, event = (
            definition
        )
        assert list(entry) == ENTRY_FIELDS
        assert entry["soak_read_model_id"] == f"soak_read_model_{source_id}"
        assert entry["source_soak_id"] == source_id
        assert entry["display_name"] == display
        assert entry["soak_domain"] == domain
        assert entry["planned_soak_source"] == source
        assert entry["planned_soak_type"] == soak_type
        assert entry["required_soak_profile"] == profile
        assert entry["planned_window_class"] == window
        assert entry["planned_exit_criteria_profile"] == exit_profile
        assert entry["planned_failure_audit_event_id"] == event
        assert entry["read_model_classification"] == "static_soak_read_model_only"
        assert entry["operator_visibility"] == "future_read_only_soak"
        assert entry["eligible_for_13_5_gate_matrix"] is True
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
        assert all(entry[key] is False for key in ENTRY_FIELDS if key.endswith("_allowed_now"))

    assert model["default_soak_read_model_selection"] == {
        "soak_read_model_id": "soak_read_model_paper_runtime_soak",
        "source_soak_id": "paper_runtime_soak",
        "reason": "first soak read-model scenario; static only, no soak runtime, no scheduler, no runtime loop",
        "soak_runtime_allowed_now": False,
        "soak_scheduler_allowed_now": False,
        "runtime_loop_allowed_now": False,
    }
    required = model["soak_readiness_required_fields"]
    assert [item["field_name"] for item in required] == REQUIRED_FIELDS
    for item in required:
        assert item["field_classification"] == "planned_soak_readiness_field_only"
        assert item["required_for_future_soak_runner"] is True
        assert item["contains_secret_material"] is False
        assert item["contains_private_key_material"] is False
        assert item["allowed_to_collect_now"] is False
        assert item["allowed_to_mutate_now"] is False
        assert item["notes"]

    assert model["soak_read_model_summary"] == {
        "entry_count": 8,
        "required_field_count": 10,
        "default_selection_id": "soak_read_model_paper_runtime_soak",
        "soak_runtime_enabled_entry_count": 0,
        "soak_scheduler_enabled_entry_count": 0,
        "runtime_loop_enabled_entry_count": 0,
        "metrics_collection_enabled_entry_count": 0,
        "metrics_export_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "offline_safe_entry_count": 8,
        "entries_eligible_for_13_5_gate_matrix": 8,
        "paper_runtime_domain_entry_count": 1,
        "testnet_runtime_domain_entry_count": 1,
        "audit_domain_entry_count": 1,
        "rollback_domain_entry_count": 1,
        "risk_domain_entry_count": 1,
        "order_flow_domain_entry_count": 1,
        "private_endpoint_domain_entry_count": 1,
        "network_domain_entry_count": 1,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_soak_runtime_now": False,
        "safe_for_scheduler_now": False,
        "safe_for_runtime_loop_now": False,
    }
    matrix = model["soak_read_model_matrix"]
    assert matrix["soak_read_model_ids"] == ids
    assert matrix["entries_requiring_13_5_gate_matrix"] == ids
    assert matrix["entries_never_run_in_13_4"] == ids
    assert matrix["planned_soak_sources_by_id"] == {
        f"soak_read_model_{d[0]}": d[3] for d in ENTRY_DEFINITIONS
    }
    assert matrix["planned_failure_audit_event_ids_by_id"] == {
        f"soak_read_model_{d[0]}": d[8] for d in ENTRY_DEFINITIONS
    }
    assert matrix["required_soak_profiles_by_id"] == {
        f"soak_read_model_{d[0]}": d[5] for d in ENTRY_DEFINITIONS
    }
    assert matrix["planned_exit_criteria_profiles_by_id"] == {
        f"soak_read_model_{d[0]}": d[7] for d in ENTRY_DEFINITIONS
    }


def test_surface_capabilities_boundaries_evidence_and_future_steps_are_exact() -> None:
    model = _model()
    assert model["soak_surface_contract"] == {
        "surface_contract_id": "block_k_soak_read_model_surface_contract",
        "read_model_is_static": True,
        "soak_scenarios_are_planned_only": True,
        "soak_is_not_run_now": True,
        "soak_scheduler_is_not_started_now": True,
        "runtime_loop_is_not_started_now": True,
        "metrics_are_not_collected_now": True,
        "metrics_are_not_exported_now": True,
        "logs_are_not_read_now": True,
        "logs_are_not_written_now": True,
        "filesystem_io_forbidden_now": True,
        "network_io_forbidden_now": True,
        "runner_requires_future_gate": True,
        "ui_surface_requires_future_qml_gate": True,
    }
    assert len(model["blocked_soak_read_model_capabilities"]) == 57
    assert model["blocked_soak_read_model_capabilities"][0] == "soak runtime"
    assert model["blocked_soak_read_model_capabilities"][-1] == "PyInstaller/EXE packaging"
    boundaries = model["soak_read_model_boundaries"]
    assert boundaries["soak_read_model_balance_read_blocked"] is True
    assert all(value is True for value in boundaries.values())
    evidence = model["non_activation_evidence"]
    for key in [
        "block_k_contract_13_0_read",
        "observability_read_model_13_1_read",
        "audit_envelope_read_model_13_2_read",
        "rollback_read_model_13_3_read",
        "soak_read_model_built",
    ]:
        assert evidence[key] is True
    assert all(
        value is False
        for key, value in evidence.items()
        if key
        not in {
            "block_k_contract_13_0_read",
            "observability_read_model_13_1_read",
            "audit_envelope_read_model_13_2_read",
            "rollback_read_model_13_3_read",
            "soak_read_model_built",
        }
    )
    assert model["source_boundaries"][0] == "no PySide import"
    assert model["source_boundaries"][-1] == "no workflow changes"
    assert model["future_steps"] == [
        "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
        "functional_preview_13_6_block_k_closure_audit",
    ]


def test_source_imports_remain_in_safe_read_model_subset() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    imported_modules = [node.module for node in imports if isinstance(node, ast.ImportFrom)]
    assert imported_modules == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_audit_envelope_read_model",
        "ui.pyside_app.preview_observability_audit_rollback_soak_contract",
        "ui.pyside_app.preview_observability_read_model",
        "ui.pyside_app.preview_rollback_read_model",
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
