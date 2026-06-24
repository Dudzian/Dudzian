"""Tests for FUNCTIONAL-PREVIEW-12.1 Block J risk limits read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_risk_governor_limits_read_model import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_KIND,
    PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_J_2,
    RISK_GOVERNOR_LIMITS_READ_MODEL_DECISION,
    RISK_GOVERNOR_LIMITS_READ_MODEL_STATUS,
    STATUS,
    STEP_ID,
    build_preview_risk_governor_limits_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_risk_governor_limits_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "risk_governor_limits_read_model_kind",
    "block",
    "step",
    "risk_governor_limits_read_model_status",
    "risk_governor_limits_read_model_decision",
    "ready_for_block_j_2",
    "next_step",
    "next_step_title",
    "risk_contract_reference",
    "risk_limits_read_model_scope",
    "risk_limits_read_model_entries",
    "default_risk_limit_read_model_selection",
    "risk_limits_read_model_summary",
    "risk_limits_read_model_matrix",
    "blocked_risk_limits_read_model_capabilities",
    "risk_limits_read_model_boundaries",
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
READ_MODEL_IDS = [f"risk_limit_read_model_{limit_id}" for limit_id in LIMIT_IDS]
ENTRY_FIELDS = [
    "risk_limit_read_model_id",
    "source_limit_category_id",
    "display_name",
    "read_model_classification",
    "limit_scope",
    "planned_measurement",
    "planned_value_type",
    "planned_fixture_key",
    "required_before_order_flow",
    "runtime_enforced_now",
    "operator_visibility",
    "eligible_for_12_2_static_fixture",
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
VALUE_TYPES = {
    "max_order_notional": "money",
    "max_daily_notional": "money",
    "max_position_notional": "money",
    "max_open_positions": "integer",
    "max_daily_loss": "money",
    "max_drawdown": "percent",
    "max_order_rate": "rate",
    "allowed_symbols": "symbol_allowlist",
    "allowed_modes": "mode_allowlist",
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


def _model() -> dict[str, Any]:
    return build_preview_risk_governor_limits_read_model()


def test_read_model_is_plain_serializable_dict_with_exact_top_level_fields() -> None:
    model = _model()
    assert isinstance(model, dict)
    json.dumps(model, sort_keys=True)
    assert list(model) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    model = _model()
    assert model["schema_version"] == PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_SCHEMA_VERSION
    assert (
        model["risk_governor_limits_read_model_kind"]
        == PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_KIND
    )
    assert model["block"] == BLOCK_ID == "J"
    assert model["step"] == STEP_ID == "12.1"
    assert model["risk_governor_limits_read_model_status"] == RISK_GOVERNOR_LIMITS_READ_MODEL_STATUS
    assert (
        model["risk_governor_limits_read_model_decision"]
        == RISK_GOVERNOR_LIMITS_READ_MODEL_DECISION
    )
    assert model["ready_for_block_j_2"] == READY_FOR_BLOCK_J_2 is True
    assert model["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-12.2"
    assert model["next_step_title"] == NEXT_STEP_TITLE == "RISK LIMITS STATIC FIXTURE"
    assert model["status"] == STATUS


def test_risk_contract_reference_is_safe_subset_for_12_0() -> None:
    ref = _model()["risk_contract_reference"]
    assert list(ref) == [
        "schema_version",
        "risk_governor_limits_kill_switch_contract_kind",
        "risk_governor_limits_kill_switch_contract_status",
        "risk_governor_limits_kill_switch_contract_decision",
        "ready_for_block_j_1",
        "next_step",
        "next_step_title",
        "status",
    ]
    assert ref["ready_for_block_j_1"] is True
    assert ref["next_step"] == "FUNCTIONAL-PREVIEW-12.1"
    assert ref["next_step_title"] == "RISK GOVERNOR LIMITS READ MODEL"


def test_scope_is_read_model_only_and_non_activating() -> None:
    scope = _model()["risk_limits_read_model_scope"]
    assert scope["scope_name"] == "risk_governor_limits_read_model"
    assert scope["read_model_only"] is True
    assert scope["derived_from_risk_contract_12_0"] is True
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


def test_entries_are_exact_order_safe_and_typed() -> None:
    entries = _model()["risk_limits_read_model_entries"]
    assert [entry["source_limit_category_id"] for entry in entries] == LIMIT_IDS
    for entry in entries:
        category_id = entry["source_limit_category_id"]
        assert list(entry) == ENTRY_FIELDS
        assert entry["risk_limit_read_model_id"] == f"risk_limit_read_model_{category_id}"
        assert entry["read_model_classification"] == "static_limit_read_model_only"
        assert entry["planned_fixture_key"] == f"fixture_{category_id}"
        assert entry["planned_value_type"] == VALUE_TYPES[category_id]
        assert entry["required_before_order_flow"] is True
        assert entry["runtime_enforced_now"] is False
        assert entry["operator_visibility"] == "future_read_only_contract"
        assert entry["eligible_for_12_2_static_fixture"] is True
        assert entry["eligible_for_12_4_gate_matrix"] is True
        for key in (
            "limit_enforcement_allowed_now",
            "order_generation_allowed_now",
            "order_submission_allowed_now",
            "private_endpoint_access_allowed_now",
            "network_io_allowed_now",
            "config_file_read_allowed_now",
            "credential_secret_read_allowed_now",
        ):
            assert entry[key] is False, key
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
    assert "Live production remains blocked" in entries[-1]["notes"]


def test_default_summary_matrix_blocked_boundaries_evidence_and_future_steps() -> None:
    model = _model()
    assert model["default_risk_limit_read_model_selection"] == {
        "risk_limit_read_model_id": "risk_limit_read_model_max_order_notional",
        "source_limit_category_id": "max_order_notional",
        "reason": "lowest-risk first read-model entry; static only, no runtime enforcement, no order flow",
        "runtime_enforced_now": False,
        "order_submission_allowed_now": False,
    }
    assert model["risk_limits_read_model_summary"] == {
        "entry_count": 9,
        "default_selection_id": "risk_limit_read_model_max_order_notional",
        "runtime_enforced_entry_count": 0,
        "limit_enforcement_enabled_entry_count": 0,
        "order_generation_enabled_entry_count": 0,
        "order_submission_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "config_file_read_enabled_entry_count": 0,
        "credential_secret_read_enabled_entry_count": 0,
        "offline_safe_entry_count": 9,
        "entries_eligible_for_12_2_static_fixture": 9,
        "entries_eligible_for_12_4_gate_matrix": 9,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_order_execution_now": False,
    }
    assert model["risk_limits_read_model_matrix"] == {
        "risk_limit_read_model_ids": READ_MODEL_IDS,
        "money_limit_ids": [
            "risk_limit_read_model_max_order_notional",
            "risk_limit_read_model_max_daily_notional",
            "risk_limit_read_model_max_position_notional",
            "risk_limit_read_model_max_daily_loss",
        ],
        "count_or_rate_limit_ids": [
            "risk_limit_read_model_max_open_positions",
            "risk_limit_read_model_max_order_rate",
        ],
        "percent_limit_ids": ["risk_limit_read_model_max_drawdown"],
        "allowlist_limit_ids": [
            "risk_limit_read_model_allowed_symbols",
            "risk_limit_read_model_allowed_modes",
        ],
        "entries_requiring_12_2_static_fixture": READ_MODEL_IDS,
        "entries_requiring_12_4_gate_matrix": READ_MODEL_IDS,
        "entries_never_runtime_enabled_in_12_1": READ_MODEL_IDS,
    }
    assert model["blocked_risk_limits_read_model_capabilities"] == BLOCKED_CAPABILITIES
    assert (
        "risk_limits_read_model_balance_read_blocked" in model["risk_limits_read_model_boundaries"]
    )
    assert all(model["risk_limits_read_model_boundaries"].values())
    assert model["non_activation_evidence"]["risk_contract_12_0_read"] is True
    assert model["non_activation_evidence"]["risk_limits_read_model_built"] is True
    for key, value in model["non_activation_evidence"].items():
        if key not in {"risk_contract_12_0_read", "risk_limits_read_model_built"}:
            assert value is False, key
    assert model["source_boundaries"] == SOURCE_BOUNDARIES
    assert model["future_steps"] == [
        "functional_preview_12_2_risk_limits_static_fixture",
        "functional_preview_12_3_kill_switch_read_model",
        "functional_preview_12_4_risk_governor_gate_matrix",
        "functional_preview_12_5_block_j_closure_audit",
    ]


def test_source_imports_only_safe_typing_and_12_0_helper() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert [
        (node.module, [alias.name for alias in node.names])
        for node in imports
        if isinstance(node, ast.ImportFrom)
    ] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        (
            "ui.pyside_app.preview_risk_governor_limits_kill_switch_contract",
            ["build_preview_risk_governor_limits_kill_switch_contract"],
        ),
    ]
    assert not [node for node in imports if isinstance(node, ast.Import)]


def test_source_has_no_forbidden_runtime_or_io_tokens() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "open(",
        "read_text",
        "write_text",
        "getenv",
        "environ[",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "socket.",
        "websocket.",
        "getaddrinfo",
        "create_connection",
        "QQmlApplicationEngine",
        "from TradingController",
        "from DecisionEnvelope",
        "start_runtime(",
        "start_loop(",
        "submit_order(",
        "place_order(",
        "create_order(",
        "send_order(",
        "fill_order(",
        "cancel_order(",
        "replace_order(",
        "withdraw(",
        "transfer(",
        "fetch_market_data(",
        "fetch_" + "balance(",
        "fetch_account(",
        "fetch_positions(",
        "fetch_orders(",
        "fetch_fills(",
        "refresh_market_data(",
        "export(",
        "cc" + "xt",
    ]
    for token in forbidden:
        assert token not in source, token
