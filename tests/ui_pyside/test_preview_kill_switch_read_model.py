"""Tests for FUNCTIONAL-PREVIEW-12.3 Block J kill switch read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_kill_switch_read_model import (
    BLOCK_ID,
    KILL_SWITCH_READ_MODEL_DECISION,
    KILL_SWITCH_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_KILL_SWITCH_READ_MODEL_KIND,
    PREVIEW_KILL_SWITCH_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_J_4,
    STATUS,
    STEP_ID,
    build_preview_kill_switch_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_kill_switch_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "kill_switch_read_model_kind",
    "block",
    "step",
    "kill_switch_read_model_status",
    "kill_switch_read_model_decision",
    "ready_for_block_j_4",
    "next_step",
    "next_step_title",
    "risk_limits_static_fixture_reference",
    "kill_switch_read_model_scope",
    "kill_switch_read_model_entries",
    "default_kill_switch_read_model_selection",
    "kill_switch_read_model_summary",
    "kill_switch_read_model_matrix",
    "kill_switch_trigger_contract",
    "blocked_kill_switch_read_model_capabilities",
    "kill_switch_read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
TRIGGER_IDS = [
    "operator_manual_stop",
    "loss_limit_breach",
    "drawdown_limit_breach",
    "order_rate_limit_breach",
    "private_endpoint_error_spike",
    "runtime_health_failure",
]
MODEL_IDS = [f"kill_switch_read_model_{trigger_id}" for trigger_id in TRIGGER_IDS]
ENTRY_FIELDS = [
    "kill_switch_read_model_id",
    "source_trigger_id",
    "display_name",
    "read_model_classification",
    "trigger_source_type",
    "planned_input_source",
    "required_prior_fixture_id",
    "planned_severity",
    "operator_visibility",
    "eligible_for_12_4_gate_matrix",
    "runtime_trigger_allowed_now",
    "manual_trigger_allowed_now",
    "automatic_trigger_allowed_now",
    "kill_switch_state_mutation_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "runtime_loop_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "config_file_read_allowed_now",
    "credential_secret_read_allowed_now",
    "safe_for_offline_tests",
    "notes",
]
EXPECTED_TRIGGER_VALUES = {
    "operator_manual_stop": (
        "Operator manual stop",
        "operator_manual",
        "future_operator_control",
        "risk_limit_static_fixture_max_order_notional",
        "critical",
    ),
    "loss_limit_breach": (
        "Loss limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_daily_loss",
        "risk_limit_static_fixture_max_daily_loss",
        "critical",
    ),
    "drawdown_limit_breach": (
        "Drawdown limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_drawdown",
        "risk_limit_static_fixture_max_drawdown",
        "critical",
    ),
    "order_rate_limit_breach": (
        "Order rate limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_order_rate",
        "risk_limit_static_fixture_max_order_rate",
        "high",
    ),
    "private_endpoint_error_spike": (
        "Private endpoint error spike",
        "private_endpoint_health",
        "future_private_endpoint_health_monitor",
        "risk_limit_static_fixture_allowed_modes",
        "high",
    ),
    "runtime_health_failure": (
        "Runtime health failure",
        "runtime_health",
        "future_runtime_health_monitor",
        "risk_limit_static_fixture_allowed_modes",
        "critical",
    ),
}
BLOCKED_CAPABILITIES = [
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "kill switch state mutation",
    "risk runtime enforcement",
    "limit runtime enforcement",
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
SOURCE_BOUNDARIES = [
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


def _model() -> dict[str, Any]:
    return build_preview_kill_switch_read_model()


def test_model_is_plain_serializable_dict_with_exact_top_level_fields() -> None:
    model = _model()
    assert isinstance(model, dict)
    json.dumps(model, sort_keys=True)
    assert list(model) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    model = _model()
    assert model["schema_version"] == PREVIEW_KILL_SWITCH_READ_MODEL_SCHEMA_VERSION
    assert model["kill_switch_read_model_kind"] == PREVIEW_KILL_SWITCH_READ_MODEL_KIND
    assert model["block"] == BLOCK_ID == "J"
    assert model["step"] == STEP_ID == "12.3"
    assert model["kill_switch_read_model_status"] == KILL_SWITCH_READ_MODEL_STATUS
    assert model["kill_switch_read_model_decision"] == KILL_SWITCH_READ_MODEL_DECISION
    assert model["ready_for_block_j_4"] == READY_FOR_BLOCK_J_4 is True
    assert model["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-12.4"
    assert model["next_step_title"] == NEXT_STEP_TITLE == "RISK GOVERNOR GATE MATRIX"
    assert model["status"] == STATUS


def test_risk_limits_static_fixture_reference_points_to_12_2() -> None:
    reference = _model()["risk_limits_static_fixture_reference"]
    assert list(reference) == [
        "schema_version",
        "risk_limits_static_fixture_kind",
        "risk_limits_static_fixture_status",
        "risk_limits_static_fixture_decision",
        "ready_for_block_j_3",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert reference["ready_for_block_j_3"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-12.3"
    assert reference["next_step_title"] == "KILL SWITCH READ MODEL"


def test_scope_is_read_model_only_and_blocks_runtime_network_private_config_credentials_ui() -> (
    None
):
    scope = _model()["kill_switch_read_model_scope"]
    assert scope["scope_name"] == "kill_switch_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_risk_limits_static_fixture_12_2"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "read_model_only",
            "derived_from_risk_limits_static_fixture_12_2",
            "exe_direction_preserved",
        }:
            assert value is False, key


def test_entries_have_exact_fields_order_values_and_safe_flags() -> None:
    entries = _model()["kill_switch_read_model_entries"]
    assert [entry["source_trigger_id"] for entry in entries] == TRIGGER_IDS
    assert [entry["kill_switch_read_model_id"] for entry in entries] == MODEL_IDS
    for entry in entries:
        assert list(entry) == ENTRY_FIELDS
        trigger_id = entry["source_trigger_id"]
        display, trigger_type, input_source, fixture_id, severity = EXPECTED_TRIGGER_VALUES[
            trigger_id
        ]
        assert entry["kill_switch_read_model_id"] == f"kill_switch_read_model_{trigger_id}"
        assert entry["display_name"] == display
        assert entry["read_model_classification"] == "static_kill_switch_read_model_only"
        assert entry["trigger_source_type"] == trigger_type
        assert entry["planned_input_source"] == input_source
        assert entry["required_prior_fixture_id"] == fixture_id
        assert entry["planned_severity"] == severity
        assert entry["operator_visibility"] == "future_read_only_kill_switch"
        assert entry["eligible_for_12_4_gate_matrix"] is True
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
        for key in [
            "runtime_trigger_allowed_now",
            "manual_trigger_allowed_now",
            "automatic_trigger_allowed_now",
            "kill_switch_state_mutation_allowed_now",
            "order_generation_allowed_now",
            "order_submission_allowed_now",
            "runtime_loop_allowed_now",
            "private_endpoint_access_allowed_now",
            "network_io_allowed_now",
            "config_file_read_allowed_now",
            "credential_secret_read_allowed_now",
        ]:
            assert entry[key] is False, key


def test_default_selection_summary_matrix_and_contract_are_exact() -> None:
    model = _model()
    assert model["default_kill_switch_read_model_selection"] == {
        "kill_switch_read_model_id": "kill_switch_read_model_operator_manual_stop",
        "source_trigger_id": "operator_manual_stop",
        "reason": "lowest-risk first read-model trigger; static only, no runtime trigger, no order flow",
        "runtime_trigger_allowed_now": False,
        "order_submission_allowed_now": False,
    }
    assert model["kill_switch_read_model_summary"] == {
        "entry_count": 6,
        "default_selection_id": "kill_switch_read_model_operator_manual_stop",
        "runtime_trigger_enabled_entry_count": 0,
        "manual_trigger_enabled_entry_count": 0,
        "automatic_trigger_enabled_entry_count": 0,
        "kill_switch_state_mutation_enabled_entry_count": 0,
        "order_generation_enabled_entry_count": 0,
        "order_submission_enabled_entry_count": 0,
        "runtime_loop_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "config_file_read_enabled_entry_count": 0,
        "credential_secret_read_enabled_entry_count": 0,
        "offline_safe_entry_count": 6,
        "entries_eligible_for_12_4_gate_matrix": 6,
        "critical_severity_entry_count": 4,
        "high_severity_entry_count": 2,
        "operator_manual_entry_count": 1,
        "risk_limit_entry_count": 3,
        "health_entry_count": 2,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_order_execution_now": False,
    }
    assert model["kill_switch_read_model_matrix"] == {
        "kill_switch_read_model_ids": MODEL_IDS,
        "critical_trigger_ids": [
            "kill_switch_read_model_operator_manual_stop",
            "kill_switch_read_model_loss_limit_breach",
            "kill_switch_read_model_drawdown_limit_breach",
            "kill_switch_read_model_runtime_health_failure",
        ],
        "high_trigger_ids": [
            "kill_switch_read_model_order_rate_limit_breach",
            "kill_switch_read_model_private_endpoint_error_spike",
        ],
        "operator_manual_trigger_ids": ["kill_switch_read_model_operator_manual_stop"],
        "risk_limit_trigger_ids": [
            "kill_switch_read_model_loss_limit_breach",
            "kill_switch_read_model_drawdown_limit_breach",
            "kill_switch_read_model_order_rate_limit_breach",
        ],
        "health_trigger_ids": [
            "kill_switch_read_model_private_endpoint_error_spike",
            "kill_switch_read_model_runtime_health_failure",
        ],
        "entries_requiring_12_4_gate_matrix": MODEL_IDS,
        "entries_never_runtime_triggered_in_12_3": MODEL_IDS,
        "required_fixture_ids_by_trigger_id": {
            model_id: EXPECTED_TRIGGER_VALUES[trigger_id][3]
            for model_id, trigger_id in zip(MODEL_IDS, TRIGGER_IDS, strict=True)
        },
        "planned_input_sources_by_trigger_id": {
            model_id: EXPECTED_TRIGGER_VALUES[trigger_id][2]
            for model_id, trigger_id in zip(MODEL_IDS, TRIGGER_IDS, strict=True)
        },
    }
    assert model["kill_switch_trigger_contract"] == {
        "trigger_contract_id": "hard_kill_switch_read_model_contract",
        "trigger_values_are_static": True,
        "trigger_values_are_examples_only": True,
        "trigger_values_are_not_runtime_triggers": True,
        "manual_trigger_cannot_fire_now": True,
        "automatic_trigger_cannot_fire_now": True,
        "kill_switch_state_cannot_mutate_now": True,
        "order_flow_remains_blocked": True,
        "runtime_loop_remains_blocked": True,
        "private_endpoint_access_remains_blocked": True,
        "trigger_values_require_12_4_gate_before_any_runtime_use": True,
        "live_production_trading_forbidden": True,
    }


def test_boundaries_blocked_capabilities_non_activation_source_boundaries_and_future_steps() -> (
    None
):
    model = _model()
    assert model["blocked_kill_switch_read_model_capabilities"] == BLOCKED_CAPABILITIES
    boundaries = model["kill_switch_read_model_boundaries"]
    assert "kill_switch_read_model_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = model["non_activation_evidence"]
    assert evidence["risk_limits_static_fixture_12_2_read"] is True
    assert evidence["kill_switch_read_model_built"] is True
    for key, value in evidence.items():
        if key not in {"risk_limits_static_fixture_12_2_read", "kill_switch_read_model_built"}:
            assert value is False, key
    assert model["source_boundaries"] == SOURCE_BOUNDARIES
    assert model["future_steps"] == [
        "functional_preview_12_4_risk_governor_gate_matrix",
        "functional_preview_12_5_block_j_closure_audit",
    ]


def test_source_imports_only_safe_typing_and_static_fixture_helper() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert [alias.name for alias in imports[1].names] == ["Any", "Final"]
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_risk_limits_static_fixture"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_risk_limits_static_fixture"
    ]


def test_source_has_no_forbidden_imports_or_calls() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden_tokens = [
        "QQmlApplicationEngine",
        "start_runtime",
        "start_loop",
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "getaddrinfo",
        "create_connection",
        "read_text",
        "write_text",
        "getenv",
        "export",
    ]
    for token in forbidden_tokens:
        assert token not in source, token
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules == {
        "__future__",
        "typing",
        "ui.pyside_app.preview_risk_limits_static_fixture",
    }
    assert all(not isinstance(node, ast.Import) for node in ast.walk(tree))
    forbidden_call_names = {"open"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_call_names
