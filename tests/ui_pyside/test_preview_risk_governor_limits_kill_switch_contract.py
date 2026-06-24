"""Tests for FUNCTIONAL-PREVIEW-12.0 Block J risk contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_risk_governor_limits_kill_switch_contract import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_KIND,
    PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_J_1,
    RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_DECISION,
    RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_STATUS,
    STATUS,
    STEP_ID,
    build_preview_risk_governor_limits_kill_switch_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_risk_governor_limits_kill_switch_contract.py"
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "risk_governor_limits_kill_switch_contract_kind",
    "block",
    "step",
    "risk_governor_limits_kill_switch_contract_status",
    "risk_governor_limits_kill_switch_contract_decision",
    "ready_for_block_j_1",
    "next_step",
    "next_step_title",
    "block_i_closure_reference",
    "risk_contract_scope",
    "risk_governor_contract_principles",
    "risk_limit_categories",
    "kill_switch_contract",
    "risk_governor_dependency_matrix",
    "blocked_risk_contract_capabilities",
    "risk_contract_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]

EXPECTED_BLOCKED_CAPABILITIES = [
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "market data fetch",
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
    "no real market data fetch",
    "no private endpoint access",
    "no account fetch",
    "no balance read",
    "no positions fetch",
    "no orders fetch",
    "no fills fetch",
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


def _contract() -> dict[str, Any]:
    return build_preview_risk_governor_limits_kill_switch_contract()


def test_contract_is_plain_serializable_dict_with_exact_top_level_fields() -> None:
    contract = _contract()
    assert isinstance(contract, dict)
    json.dumps(contract, sort_keys=True)
    assert list(contract) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    contract = _contract()
    assert (
        contract["schema_version"]
        == PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_SCHEMA_VERSION
    )
    assert (
        contract["risk_governor_limits_kill_switch_contract_kind"]
        == PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_KIND
    )
    assert contract["block"] == BLOCK_ID == "J"
    assert contract["step"] == STEP_ID == "12.0"
    assert (
        contract["risk_governor_limits_kill_switch_contract_status"]
        == RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_STATUS
    )
    assert (
        contract["risk_governor_limits_kill_switch_contract_decision"]
        == RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_DECISION
    )
    assert contract["ready_for_block_j_1"] == READY_FOR_BLOCK_J_1 is True
    assert contract["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-12.1"
    assert contract["next_step_title"] == NEXT_STEP_TITLE == "RISK GOVERNOR LIMITS READ MODEL"
    assert contract["status"] == STATUS


def test_block_i_closure_reference_is_safe_subset_for_11_8() -> None:
    ref = _contract()["block_i_closure_reference"]
    assert list(ref) == [
        "schema_version",
        "testnet_sandbox_adapter_closure_audit_kind",
        "testnet_sandbox_adapter_closure_audit_status",
        "testnet_sandbox_adapter_closure_audit_decision",
        "ready_for_block_j",
        "next_block",
        "next_step",
        "next_step_title",
        "closure_line",
        "status",
    ]
    assert ref["ready_for_block_j"] is True
    assert ref["next_step"] == "FUNCTIONAL-PREVIEW-12.0"
    assert ref["next_step_title"] == "BLOK J — RISK GOVERNOR / LIMITS / KILL SWITCH CONTRACT"
    assert ref["closure_line"] == "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"


def test_risk_contract_scope_flags_are_contract_only_and_non_activating() -> None:
    scope = _contract()["risk_contract_scope"]
    assert scope["scope_name"] == "risk_governor_limits_kill_switch_contract"
    assert scope["contract_only"] is True
    assert scope["derived_from_block_i_closure_11_8"] is True
    assert scope["starts_block_j"] is True
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


def test_principles_are_exact_order_and_non_runtime() -> None:
    principles = _contract()["risk_governor_contract_principles"]
    assert [item["principle_id"] for item in principles] == [
        "fail_closed_by_default",
        "explicit_allowlist_required",
        "hard_kill_switch_overrides_all",
        "position_and_notional_limits_required",
        "loss_limits_required",
        "private_endpoint_requires_read_only_gate",
        "order_flow_requires_block_j_completion",
        "live_trading_requires_later_live_canary_gate",
    ]
    for item in principles:
        assert list(item) == [
            "principle_id",
            "display_name",
            "description",
            "runtime_enforced_now",
            "required_before_order_flow",
        ]
        assert item["description"]
        assert item["runtime_enforced_now"] is False
        assert item["required_before_order_flow"] is True


def test_limit_categories_are_exact_order_and_non_runtime() -> None:
    limits = _contract()["risk_limit_categories"]
    assert [item["limit_category_id"] for item in limits] == [
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
    for item in limits:
        assert list(item) == [
            "limit_category_id",
            "display_name",
            "limit_scope",
            "planned_measurement",
            "required_before_order_flow",
            "runtime_enforced_now",
            "operator_visibility",
            "notes",
        ]
        assert item["required_before_order_flow"] is True
        assert item["runtime_enforced_now"] is False
        assert item["operator_visibility"] == "future_read_only_contract"
        assert item["notes"]
    assert "Live production remains blocked" in limits[-1]["notes"]


def test_kill_switch_contract_exact_fields_and_values() -> None:
    kill_switch = _contract()["kill_switch_contract"]
    assert list(kill_switch) == [
        "kill_switch_contract_id",
        "display_name",
        "contract_only",
        "runtime_trigger_allowed_now",
        "manual_operator_trigger_allowed_now",
        "automatic_trigger_allowed_now",
        "must_override_order_generation",
        "must_override_order_submission",
        "must_override_runtime_loop",
        "must_override_private_endpoint_access",
        "required_before_any_order_flow",
        "required_before_any_runtime_soak",
        "required_before_any_live_canary",
        "blocked_actions_now",
        "future_trigger_sources",
        "notes",
    ]
    assert kill_switch == {
        "kill_switch_contract_id": "hard_kill_switch_contract",
        "display_name": "Hard kill switch contract",
        "contract_only": True,
        "runtime_trigger_allowed_now": False,
        "manual_operator_trigger_allowed_now": False,
        "automatic_trigger_allowed_now": False,
        "must_override_order_generation": True,
        "must_override_order_submission": True,
        "must_override_runtime_loop": True,
        "must_override_private_endpoint_access": True,
        "required_before_any_order_flow": True,
        "required_before_any_runtime_soak": True,
        "required_before_any_live_canary": True,
        "blocked_actions_now": [
            "order_generation",
            "order_submission",
            "order_cancel",
            "order_replace",
            "runtime_loop",
            "private_endpoint_access",
            "live_trading",
        ],
        "future_trigger_sources": [
            "operator_manual_stop",
            "loss_limit_breach",
            "drawdown_limit_breach",
            "order_rate_limit_breach",
            "private_endpoint_error_spike",
            "runtime_health_failure",
        ],
        "notes": kill_switch["notes"],
    }
    assert kill_switch["notes"]


def test_dependency_matrix_blocked_capabilities_boundaries_and_future_steps() -> None:
    contract = _contract()
    assert contract["risk_governor_dependency_matrix"] == {
        "requires_block_i_closure": True,
        "block_i_closure_ready": True,
        "requires_limits_read_model_next": True,
        "requires_static_limit_fixture_later": True,
        "requires_kill_switch_read_model_later": True,
        "requires_order_flow_to_remain_blocked": True,
        "requires_private_endpoint_to_remain_blocked": True,
        "requires_network_io_to_remain_blocked": True,
        "requires_live_trading_to_remain_blocked": True,
        "risk_contract_ready_for_12_1": True,
        "runtime_enforcement_enabled_now": False,
        "order_flow_enabled_now": False,
        "private_endpoint_enabled_now": False,
        "network_io_enabled_now": False,
        "live_trading_enabled_now": False,
    }
    assert contract["blocked_risk_contract_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    assert all(contract["risk_contract_boundaries"].values())
    assert "risk_contract_balance_read_blocked" in contract["risk_contract_boundaries"]
    assert contract["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert contract["future_steps"] == [
        "functional_preview_12_1_risk_governor_limits_read_model",
        "functional_preview_12_2_risk_limits_static_fixture",
        "functional_preview_12_3_kill_switch_read_model",
        "functional_preview_12_4_risk_governor_gate_matrix",
        "functional_preview_12_5_block_j_closure_audit",
    ]


def test_non_activation_evidence_true_false_contract() -> None:
    evidence = _contract()["non_activation_evidence"]
    assert evidence["block_i_closure_11_8_read"] is True
    assert evidence["risk_governor_contract_built"] is True
    for key, value in evidence.items():
        if key not in {"block_i_closure_11_8_read", "risk_governor_contract_built"}:
            assert value is False, key


def test_source_imports_only_safe_dependencies_and_has_no_unsafe_calls() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_testnet_sandbox_adapter_closure_audit",
    ]
    forbidden_import_parts = (
        "PySide",
        "QML",
        "runtime",
        "scheduler",
        "TradingController",
        "DecisionEnvelope",
        "order",
        "live",
        "testnet_sandbox_adapter_contract",
        "exchange",
        "account",
        "secrets",
        "security",
        "network",
        "filesystem",
        "yaml",
        "json",
        "os",
        "pathlib",
        "subprocess",
    )
    unsafe_imports = [
        module
        for module in imports
        if module != "ui.pyside_app.preview_testnet_sandbox_adapter_closure_audit"
        for part in forbidden_import_parts
        if part in module
    ]
    assert unsafe_imports == []
    forbidden_calls = {
        "open",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "getaddrinfo",
        "create_connection",
        "QQmlApplicationEngine",
        "start_runtime",
        "start_loop",
        "submit_order",
        "place_order",
        "send_order",
        "fill_order",
        "cancel_order",
        "replace_order",
        "withdraw",
        "transfer",
        "refresh_market_data",
        "export",
    }
    assert not forbidden_calls.intersection(calls)
    assert "fetch" + "_balance" not in source
    assert "c" + "cxt" not in source
