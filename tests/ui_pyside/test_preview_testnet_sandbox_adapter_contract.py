"""Tests for FUNCTIONAL-PREVIEW-11.0 Block I testnet/sandbox adapter contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_testnet_sandbox_adapter_contract import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_KIND,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_1,
    STATUS,
    TESTNET_SANDBOX_ADAPTER_CONTRACT_DECISION,
    TESTNET_SANDBOX_ADAPTER_CONTRACT_STATUS,
    build_preview_testnet_sandbox_adapter_contract,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_adapter_contract_kind",
    "block",
    "step",
    "testnet_sandbox_adapter_contract_status",
    "testnet_sandbox_adapter_contract_decision",
    "ready_for_block_i_1",
    "next_step",
    "next_step_title",
    "contract_scope",
    "allowed_modes",
    "blocked_modes",
    "adapter_lifecycle_gates",
    "credential_and_secret_gates",
    "private_endpoint_gates",
    "market_data_gates",
    "order_execution_gates",
    "risk_governor_dependency",
    "observability_dependency",
    "backend_capability_handoff_requirements",
    "safety_invariants",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_ALLOWED_MODES = [
    "local_mock",
    "recorded_fixture_replay",
    "paper",
    "testnet_contract_only",
    "sandbox_contract_only",
]
EXPECTED_BLOCKED_MODES = [
    "live_production",
    "live_credentials",
    "live_private_account",
    "live_order_submission",
    "unbounded_testnet_runtime",
    "unbounded_sandbox_runtime",
    "testnet_without_risk_gate",
    "sandbox_without_risk_gate",
    "testnet_without_observability",
    "sandbox_without_observability",
]
EXPECTED_CLASSIFICATIONS = [
    "implemented",
    "implemented_not_wired",
    "contract_only",
    "fixture_only",
    "mock_only",
    "exists_but_blocked",
    "high_risk_requires_gate",
    "ready_for_contract_gate",
    "missing",
]
EXPECTED_BLOCKED_CAPABILITIES = [
    "testnet adapter implementation in 11.0",
    "sandbox adapter implementation in 11.0",
    "network I/O in 11.0",
    "exchange API connection in 11.0",
    "credentials/secrets access in 11.0",
    "private account endpoint access",
    "account balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "testnet order submission",
    "sandbox order submission",
    "live order submission",
    "runtime loop",
    "scheduler",
    "unattended execution",
    "risk governor implementation",
    "observability/soak implementation",
    "bridge API changes",
    "QML changes / new QML calls",
    "EXE packaging",
    "live production trading",
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
    "no filesystem I/O",
    "no network I/O",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]
EXPECTED_FUTURE_STEPS = [
    "functional_preview_11_1_testnet_sandbox_backend_capability_handoff",
    "functional_preview_11_2_testnet_sandbox_adapter_read_model",
    "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]
FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "qml",
    "runtime",
    "scheduler",
    "TradingController",
    "DecisionEnvelope",
    "order",
    "live",
    "testnet",
    "sandbox",
    "exchange",
    "account",
    "secrets",
    "network",
    "filesystem",
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "pathlib",
    "subprocess",
}
FORBIDDEN_CALLS = {
    "open",
    "read_text",
    "write_text",
    "requests",
    "subprocess",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
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
    "fetch_market_data",
    "fetch_balance",
    "fetch_account",
    "refresh_market_data",
    "export",
}


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    if isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def test_contract_is_plain_json_serializable_dict() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()

    assert isinstance(contract, dict)
    _assert_simple_types_only(contract)
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract
    assert set(contract) == EXPECTED_TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()

    assert contract["schema_version"] == PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_SCHEMA_VERSION
    assert (
        contract["testnet_sandbox_adapter_contract_kind"]
        == PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_KIND
    )
    assert contract["block"] == "I"
    assert contract["step"] == "11.0"
    assert (
        contract["testnet_sandbox_adapter_contract_status"]
        == TESTNET_SANDBOX_ADAPTER_CONTRACT_STATUS
    )
    assert (
        contract["testnet_sandbox_adapter_contract_decision"]
        == TESTNET_SANDBOX_ADAPTER_CONTRACT_DECISION
    )
    assert contract["ready_for_block_i_1"] is READY_FOR_BLOCK_I_1 is True
    assert contract["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.1"
    assert contract["next_step_title"] == NEXT_STEP_TITLE
    assert contract["status"] == STATUS


def test_contract_scope_blocks_execution_network_credentials_qml_and_bridge() -> None:
    scope = build_preview_testnet_sandbox_adapter_contract()["contract_scope"]

    assert scope["scope_name"] == "testnet_sandbox_adapter_contract"
    assert scope["contract_only"] is True
    for key, value in scope.items():
        if key in {"scope_name", "contract_only", "exe_direction_preserved"}:
            continue
        assert value is False, key
    assert scope["exe_direction_preserved"] is True


def test_allowed_and_blocked_modes_are_exact_and_contract_only() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()
    allowed = contract["allowed_modes"]
    blocked = contract["blocked_modes"]

    assert [entry["mode"] for entry in allowed] == EXPECTED_ALLOWED_MODES
    assert len(allowed) == 5
    for entry in allowed:
        assert set(entry) == {
            "mode",
            "allowed_now",
            "runtime_execution_allowed_now",
            "network_io_allowed_now",
            "credentials_allowed_now",
            "order_submission_allowed_now",
            "notes",
        }
        assert entry["allowed_now"] is True
        assert entry["runtime_execution_allowed_now"] is False
        assert entry["network_io_allowed_now"] is False
        assert entry["credentials_allowed_now"] is False
        assert entry["order_submission_allowed_now"] is False
        assert "Contract eligibility only" in entry["notes"]

    assert [entry["mode"] for entry in blocked] == EXPECTED_BLOCKED_MODES
    assert len(blocked) == 10
    assert all(entry["blocked_now"] is True and entry["reason"] for entry in blocked)


def test_gate_sections_are_complete_and_block_execution_paths() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()

    assert contract["adapter_lifecycle_gates"] == {
        "contract_required_before_adapter": True,
        "backend_capability_handoff_required_before_wiring": True,
        "adapter_read_model_required_before_runtime": True,
        "credential_gate_required_before_private_endpoint": True,
        "risk_governor_required_before_order_submission": True,
        "kill_switch_required_before_order_submission": True,
        "observability_required_before_soak": True,
        "soak_required_before_live_transition": True,
        "live_transition_gate_required_before_live": True,
        "adapter_implementation_allowed_in_11_0": False,
        "runtime_execution_allowed_in_11_0": False,
    }
    credentials = contract["credential_and_secret_gates"]
    assert credentials["credential_discovery_allowed_now"] is False
    assert credentials["credential_read_allowed_now"] is False
    assert credentials["secret_read_allowed_now"] is False
    assert credentials["api_key_material_allowed_in_report"] is False
    assert credentials["credentials_must_not_be_logged"] is True
    private = contract["private_endpoint_gates"]
    for key in [
        "private_endpoint_access_allowed_now",
        "account_fetch_allowed_now",
        "balance_fetch_allowed_now",
        "positions_fetch_allowed_now",
        "orders_fetch_allowed_now",
        "fills_fetch_allowed_now",
        "read_only_private_account_allowed_without_gate",
        "order_endpoint_allowed_without_gate",
    ]:
        assert private[key] is False
    market = contract["market_data_gates"]
    assert market["public_market_data_contract_allowed_now"] is True
    assert market["recorded_replay_allowed_now"] is True
    assert market["static_fixture_allowed_now"] is True
    assert market["public_market_data_fetch_allowed_now"] is False
    assert market["network_market_data_allowed_in_11_0"] is False
    orders = contract["order_execution_gates"]
    assert orders["testnet_order_submission_allowed_now"] is False
    assert orders["sandbox_order_submission_allowed_now"] is False
    assert orders["live_order_submission_allowed_now"] is False
    assert orders["risk_governor_required"] is True
    assert orders["kill_switch_required"] is True


def test_risk_observability_and_backend_handoff_requirements() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()

    risk = contract["risk_governor_dependency"]
    assert risk["block_j_required_before_unattended_order_execution"] is True
    assert risk["risk_governor_not_implemented_in_11_0"] is True
    observability = contract["observability_dependency"]
    assert observability["block_k_required_before_soak"] is True
    assert observability["observability_not_implemented_in_11_0"] is True
    handoff = contract["backend_capability_handoff_requirements"]
    assert handoff["backend_capability_inventory_required"] is True
    assert handoff["existing_backend_modules_must_not_be_reimplemented_blindly"] is True
    assert handoff["existing_backend_modules_must_be_classified_before_use"] is True
    assert handoff["classification_required"] == EXPECTED_CLASSIFICATIONS
    assert handoff["next_step_expected_output"] == "testnet_sandbox_backend_capability_handoff"
    assert handoff["no_backend_module_activation_in_11_0"] is True


def test_safety_invariants_blocked_capabilities_boundaries_future_steps_and_status() -> None:
    contract = build_preview_testnet_sandbox_adapter_contract()

    assert all(value is True for value in contract["safety_invariants"].values())
    assert contract["blocked_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    assert contract["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert contract["future_steps"] == EXPECTED_FUTURE_STEPS
    assert (
        contract["status"]
        == "ready_for_functional_preview_11_1_testnet_sandbox_backend_capability_handoff"
    )


def test_source_imports_only_safe_typing_and_has_no_runtime_calls() -> None:
    source_path = Path("ui/pyside_app/preview_testnet_sandbox_adapter_contract.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module.split(".")[0])
            if module == "typing":
                assert {alias.name for alias in node.names} <= {"Any", "Final"}
            else:
                assert module == "__future__"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)

    assert imports == ["__future__", "typing"]
    assert not (set(imports) & FORBIDDEN_IMPORT_ROOTS)
    assert not (set(calls) & FORBIDDEN_CALLS)
