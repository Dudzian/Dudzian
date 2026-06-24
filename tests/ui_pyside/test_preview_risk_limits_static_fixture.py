"""Tests for FUNCTIONAL-PREVIEW-12.2 Block J risk limits static fixture."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_risk_limits_static_fixture import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_RISK_LIMITS_STATIC_FIXTURE_KIND,
    PREVIEW_RISK_LIMITS_STATIC_FIXTURE_SCHEMA_VERSION,
    READY_FOR_BLOCK_J_3,
    RISK_LIMITS_STATIC_FIXTURE_DECISION,
    RISK_LIMITS_STATIC_FIXTURE_STATUS,
    STATUS,
    STEP_ID,
    build_preview_risk_limits_static_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_risk_limits_static_fixture.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "risk_limits_static_fixture_kind",
    "block",
    "step",
    "risk_limits_static_fixture_status",
    "risk_limits_static_fixture_decision",
    "ready_for_block_j_3",
    "next_step",
    "next_step_title",
    "risk_limits_read_model_reference",
    "risk_limits_static_fixture_scope",
    "risk_limits_static_fixture_entries",
    "default_risk_limits_static_fixture_selection",
    "risk_limits_static_fixture_summary",
    "risk_limits_static_fixture_matrix",
    "risk_limits_fixture_value_contract",
    "blocked_risk_limits_static_fixture_capabilities",
    "risk_limits_static_fixture_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
LIMIT_IDS = [
    "max_order_notional",
    "max_daily_notional",
    "max_position_notional",
    "max_open_positions",
    "max_daily_loss",
    "max_drawdown",
    "max_order_rate",
    "allowed_symbols",
    "allowed_modes",
]
FIXTURE_IDS = [f"risk_limit_static_fixture_{limit_id}" for limit_id in LIMIT_IDS]
ENTRY_FIELDS = [
    "risk_limit_static_fixture_id",
    "source_risk_limit_read_model_id",
    "source_limit_category_id",
    "display_name",
    "fixture_classification",
    "limit_scope",
    "planned_measurement",
    "planned_value_type",
    "fixture_value",
    "fixture_unit",
    "fixture_profile",
    "required_before_order_flow",
    "runtime_enforced_now",
    "operator_visibility",
    "eligible_for_12_3_kill_switch_read_model",
    "eligible_for_12_4_gate_matrix",
    "limit_enforcement_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "private_endpoint_access_allowed_now",
    "network_io_allowed_now",
    "config_file_read_allowed_now",
    "credential_secret_read_allowed_now",
    "safe_for_offline_tests",
    "notes",
]
EXPECTED_VALUES = {
    "max_order_notional": ("money", 100.0, "USDT"),
    "max_daily_notional": ("money", 500.0, "USDT"),
    "max_position_notional": ("money", 250.0, "USDT"),
    "max_open_positions": ("integer", 3, "positions"),
    "max_daily_loss": ("money", 50.0, "USDT"),
    "max_drawdown": ("percent", 5.0, "percent"),
    "max_order_rate": ("rate", 5, "orders_per_minute"),
    "allowed_symbols": ("symbol_allowlist", ["BTC/USDT", "ETH/USDT"], "symbols"),
    "allowed_modes": (
        "mode_allowlist",
        [
            "local_mock",
            "recorded_fixture_replay",
            "paper",
            "testnet_contract_only",
            "sandbox_contract_only",
        ],
        "modes",
    ),
}
BLOCKED_CAPABILITIES = [
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
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


def _fixture() -> dict[str, Any]:
    return build_preview_risk_limits_static_fixture()


def test_fixture_is_plain_serializable_dict_with_exact_top_level_fields() -> None:
    fixture = _fixture()
    assert isinstance(fixture, dict)
    json.dumps(fixture, sort_keys=True)
    assert list(fixture) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    fixture = _fixture()
    assert fixture["schema_version"] == PREVIEW_RISK_LIMITS_STATIC_FIXTURE_SCHEMA_VERSION
    assert fixture["risk_limits_static_fixture_kind"] == PREVIEW_RISK_LIMITS_STATIC_FIXTURE_KIND
    assert fixture["block"] == BLOCK_ID == "J"
    assert fixture["step"] == STEP_ID == "12.2"
    assert fixture["risk_limits_static_fixture_status"] == RISK_LIMITS_STATIC_FIXTURE_STATUS
    assert fixture["risk_limits_static_fixture_decision"] == RISK_LIMITS_STATIC_FIXTURE_DECISION
    assert fixture["ready_for_block_j_3"] == READY_FOR_BLOCK_J_3 is True
    assert fixture["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-12.3"
    assert fixture["next_step_title"] == NEXT_STEP_TITLE == "KILL SWITCH READ MODEL"
    assert fixture["status"] == STATUS


def test_risk_limits_read_model_reference_is_safe_subset_for_12_1() -> None:
    reference = _fixture()["risk_limits_read_model_reference"]
    assert list(reference) == [
        "schema_version",
        "risk_governor_limits_read_model_kind",
        "risk_governor_limits_read_model_status",
        "risk_governor_limits_read_model_decision",
        "ready_for_block_j_2",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert reference["ready_for_block_j_2"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-12.2"
    assert reference["next_step_title"] == "RISK LIMITS STATIC FIXTURE"


def test_scope_is_fixture_only_and_blocks_runtime_and_io() -> None:
    scope = _fixture()["risk_limits_static_fixture_scope"]
    assert scope["scope_name"] == "risk_limits_static_fixture"
    assert scope["fixture_only"] is True
    assert scope["derived_from_risk_limits_read_model_12_1"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "fixture_only",
            "derived_from_risk_limits_read_model_12_1",
            "exe_direction_preserved",
        }:
            assert value is False, key


def test_entries_have_exact_values_order_and_safe_flags() -> None:
    entries = _fixture()["risk_limits_static_fixture_entries"]
    assert [entry["source_limit_category_id"] for entry in entries] == LIMIT_IDS
    assert [entry["risk_limit_static_fixture_id"] for entry in entries] == FIXTURE_IDS
    for entry in entries:
        category_id = entry["source_limit_category_id"]
        value_type, value, unit = EXPECTED_VALUES[category_id]
        assert list(entry) == ENTRY_FIELDS
        assert entry["risk_limit_static_fixture_id"] == f"risk_limit_static_fixture_{category_id}"
        assert entry["source_risk_limit_read_model_id"] == f"risk_limit_read_model_{category_id}"
        assert entry["fixture_classification"] == "static_limit_fixture_only"
        assert entry["planned_value_type"] == value_type
        assert entry["fixture_value"] == value
        assert entry["fixture_unit"] == unit
        assert entry["fixture_profile"] == "conservative_preview"
        assert entry["required_before_order_flow"] is True
        assert entry["runtime_enforced_now"] is False
        assert entry["operator_visibility"] == "future_read_only_fixture"
        assert entry["eligible_for_12_3_kill_switch_read_model"] is True
        assert entry["eligible_for_12_4_gate_matrix"] is True
        assert entry["limit_enforcement_allowed_now"] is False
        assert entry["order_generation_allowed_now"] is False
        assert entry["order_submission_allowed_now"] is False
        assert entry["private_endpoint_access_allowed_now"] is False
        assert entry["network_io_allowed_now"] is False
        assert entry["config_file_read_allowed_now"] is False
        assert entry["credential_secret_read_allowed_now"] is False
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
    modes = entries[-1]
    assert "live_production" not in modes["fixture_value"]
    assert "Live production remains blocked" in modes["notes"]


def test_default_selection_summary_matrix_and_value_contract_are_exact() -> None:
    fixture = _fixture()
    assert fixture["default_risk_limits_static_fixture_selection"] == {
        "risk_limit_static_fixture_id": "risk_limit_static_fixture_max_order_notional",
        "source_limit_category_id": "max_order_notional",
        "reason": "lowest-risk first static fixture entry; no runtime enforcement, no order flow",
        "runtime_enforced_now": False,
        "order_submission_allowed_now": False,
    }
    assert fixture["risk_limits_static_fixture_summary"] == {
        "entry_count": 9,
        "default_selection_id": "risk_limit_static_fixture_max_order_notional",
        "runtime_enforced_entry_count": 0,
        "limit_enforcement_enabled_entry_count": 0,
        "order_generation_enabled_entry_count": 0,
        "order_submission_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "config_file_read_enabled_entry_count": 0,
        "credential_secret_read_enabled_entry_count": 0,
        "offline_safe_entry_count": 9,
        "entries_eligible_for_12_3_kill_switch_read_model": 9,
        "entries_eligible_for_12_4_gate_matrix": 9,
        "money_fixture_entry_count": 4,
        "integer_fixture_entry_count": 1,
        "percent_fixture_entry_count": 1,
        "rate_fixture_entry_count": 1,
        "allowlist_fixture_entry_count": 2,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_order_execution_now": False,
    }
    assert fixture["risk_limits_static_fixture_matrix"] == {
        "risk_limit_static_fixture_ids": FIXTURE_IDS,
        "money_fixture_ids": [
            "risk_limit_static_fixture_max_order_notional",
            "risk_limit_static_fixture_max_daily_notional",
            "risk_limit_static_fixture_max_position_notional",
            "risk_limit_static_fixture_max_daily_loss",
        ],
        "count_or_rate_fixture_ids": [
            "risk_limit_static_fixture_max_open_positions",
            "risk_limit_static_fixture_max_order_rate",
        ],
        "percent_fixture_ids": ["risk_limit_static_fixture_max_drawdown"],
        "allowlist_fixture_ids": [
            "risk_limit_static_fixture_allowed_symbols",
            "risk_limit_static_fixture_allowed_modes",
        ],
        "entries_requiring_12_3_kill_switch_read_model": FIXTURE_IDS,
        "entries_requiring_12_4_gate_matrix": FIXTURE_IDS,
        "entries_never_runtime_enabled_in_12_2": FIXTURE_IDS,
        "fixture_values_by_id": {
            f"risk_limit_static_fixture_{key}": value[1] for key, value in EXPECTED_VALUES.items()
        },
        "fixture_units_by_id": {
            f"risk_limit_static_fixture_{key}": value[2] for key, value in EXPECTED_VALUES.items()
        },
    }
    assert fixture["risk_limits_fixture_value_contract"] == {
        "fixture_profile": "conservative_preview",
        "fixture_values_are_static": True,
        "fixture_values_are_examples_only": True,
        "fixture_values_are_not_runtime_limits": True,
        "fixture_values_cannot_be_loaded_from_config": True,
        "fixture_values_cannot_be_loaded_from_env": True,
        "fixture_values_cannot_be_loaded_from_credentials": True,
        "fixture_values_cannot_be_overridden_now": True,
        "fixture_values_require_12_4_gate_before_any_enforcement": True,
        "live_production_mode_forbidden": True,
        "allowed_modes_fixture_excludes_live_production": True,
    }


def test_blocked_capabilities_boundaries_evidence_source_boundaries_and_future_steps() -> None:
    fixture = _fixture()
    assert fixture["blocked_risk_limits_static_fixture_capabilities"] == BLOCKED_CAPABILITIES
    boundaries = fixture["risk_limits_static_fixture_boundaries"]
    assert "risk_limits_static_fixture_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = fixture["non_activation_evidence"]
    assert evidence["risk_limits_read_model_12_1_read"] is True
    assert evidence["risk_limits_static_fixture_built"] is True
    for key, value in evidence.items():
        if key not in {"risk_limits_read_model_12_1_read", "risk_limits_static_fixture_built"}:
            assert value is False, key
    assert fixture["source_boundaries"] == SOURCE_BOUNDARIES
    assert fixture["future_steps"] == [
        "functional_preview_12_3_kill_switch_read_model",
        "functional_preview_12_4_risk_governor_gate_matrix",
        "functional_preview_12_5_block_j_closure_audit",
    ]


def test_source_imports_only_safe_typing_and_12_1_helper() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        (
            "ui.pyside_app.preview_risk_governor_limits_read_model",
            ["build_preview_risk_governor_limits_read_model"],
        ),
    ]


def test_source_does_not_use_forbidden_runtime_or_io_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_names = {
        "open",
        "read_text",
        "write_text",
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
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
        "export",
    }
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_names.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    assert not (forbidden_names & imported_modules)
    assert not (forbidden_names & called_names)
