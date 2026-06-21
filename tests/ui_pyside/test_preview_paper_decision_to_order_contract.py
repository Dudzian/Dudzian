"""Source-only tests for BLOK G paper decision-to-order contract."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_decision_to_order_contract import (
    BLOCK_STATUS,
    CONTRACT_DECISION,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND,
    PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_1,
    build_preview_paper_decision_to_order_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_decision_to_order_contract.py"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "contract_kind",
    "block",
    "block_status",
    "contract_decision",
    "ready_for_block_g_1",
    "next_step",
    "next_step_title",
    "block_f_closure_reference",
    "paper_path_scope",
    "allowed_future_inputs",
    "allowed_future_outputs",
    "required_gates_before_order_path",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}

EXPECTED_INPUTS = {
    "dry_run_decision_preview",
    "decision_reason_summary",
    "risk_check_preview",
    "audit_event_preview",
    "operator_selected_pair",
    "operator_selected_candidate",
    "paper_order_intent_context_id",
    "paper_order_intent_size_preview",
    "paper_order_intent_side_preview",
    "paper_order_intent_type_preview",
}

EXPECTED_OUTPUTS = {
    "paper_order_intent_preview",
    "paper_order_preview",
    "paper_order_validation_preview",
    "paper_order_refusal_preview",
    "paper_order_audit_event_preview",
    "paper_order_gate_status",
    "paper_order_risk_gate_preview",
    "paper_order_no_execution_status",
}

EXPECTED_BOUNDARY_CHECKS = {
    "local_only": True,
    "paper_only": True,
    "contract_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_allowed_now": False,
    "paper_order_generation_allowed_now": False,
    "paper_order_submission_allowed_now": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_runtime_execution_allowed_now": False,
    "risk_governor_execution_allowed_now": False,
    "manual_operator_confirmation_required_before_order": True,
    "kill_switch_required_before_order": True,
    "trading_controller_allowed": False,
    "decision_envelope_allowed": False,
    "strategy_execution_allowed": False,
    "ai_scoring_execution_allowed": False,
    "model_inference_execution_allowed": False,
    "runtime_loop_allowed": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed_initially": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "real_account_balance_allowed": False,
    "live_credentials_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "new_qml_method_calls_allowed": False,
    "qml_changes_allowed": False,
    "exe_packaging_in_scope": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_plain(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    assert not is_dataclass(value)
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_plain(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_plain(nested)


def test_contract_returns_json_serializable_plain_dict() -> None:
    contract = build_preview_paper_decision_to_order_contract()

    assert isinstance(contract, dict)
    assert set(contract) == EXPECTED_TOP_LEVEL
    _assert_plain(contract)
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract


def test_contract_identity_status_and_next_step_are_fixed() -> None:
    contract = build_preview_paper_decision_to_order_contract()

    assert contract["schema_version"] == PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_SCHEMA_VERSION
    assert contract["contract_kind"] == PREVIEW_PAPER_DECISION_TO_ORDER_CONTRACT_KIND
    assert contract["block"] == "G"
    assert contract["block_status"] == BLOCK_STATUS
    assert contract["contract_decision"] == CONTRACT_DECISION
    assert contract["ready_for_block_g_1"] is READY_FOR_BLOCK_G_1 is True
    assert contract["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.1"
    assert contract["next_step_title"] == NEXT_STEP_TITLE == "PAPER ORDER INTENT READ MODEL"
    assert contract["status"] == "ready_for_functional_preview_9_1_no_order_execution"


def test_block_f_closure_reference_confirms_closed_block_f() -> None:
    reference = build_preview_paper_decision_to_order_contract()["block_f_closure_reference"]

    assert reference == {
        "schema_version": "preview_block_f_closure_audit.v1",
        "audit_kind": "functional_preview_block_f_closure_audit",
        "block_status": "decision_engine_dry_run_read_only_complete_no_execution",
        "closure_decision": "CLOSE_BLOCK_F_AS_DRY_RUN_READ_ONLY_INTEGRATION_READY",
        "ready_for_block_g": True,
        "next_block": "G",
        "next_block_title": "PAPER-ONLY DECISION-TO-ORDER PATH",
    }


def test_paper_path_scope_is_local_paper_only_and_blocks_execution_now() -> None:
    scope = build_preview_paper_decision_to_order_contract()["paper_path_scope"]

    assert scope == {
        "paper_only": True,
        "local_only": True,
        "dry_run_source_allowed": True,
        "paper_order_intent_allowed_future_step": True,
        "paper_order_preview_allowed_future_step": True,
        "paper_order_submission_allowed_now": False,
        "paper_fill_simulation_allowed_now": False,
        "paper_runtime_execution_allowed_now": False,
        "live_trading_allowed": False,
        "testnet_trading_allowed_initially": False,
        "real_account_balance_allowed": False,
        "live_credentials_allowed": False,
    }


def test_future_inputs_outputs_and_gates_are_complete() -> None:
    contract = build_preview_paper_decision_to_order_contract()

    assert set(contract["allowed_future_inputs"]) == EXPECTED_INPUTS
    assert set(contract["allowed_future_outputs"]) == EXPECTED_OUTPUTS
    assert set(contract["required_gates_before_order_path"]) >= {
        "risk_governor_required_before_any_order_execution",
        "manual_operator_confirmation_required_before_any_order_submission",
        "kill_switch_required_before_any_order_submission",
        "paper_only_adapter_required_before_any_order_submission",
        "live_credentials_refusal_required",
    }


def test_boundaries_blocked_capabilities_and_source_boundaries_are_safe() -> None:
    contract = build_preview_paper_decision_to_order_contract()

    assert contract["boundary_checks"] == EXPECTED_BOUNDARY_CHECKS
    assert set(contract["blocked_capabilities"]) >= {
        "order generation now",
        "order submission now",
        "paper fills now",
        "live trading",
        "testnet/sandbox trading initially",
        "real account balance",
        "live credentials",
        "TradingController integration",
        "DecisionEnvelope integration",
        "strategy execution",
        "AI/scoring execution",
        "model inference",
        "runtime loop execution",
        "command dispatch execution",
        "lifecycle command execution",
        "dynamic action dispatch",
        "new QML action calls",
        "QML runtime behavior changes",
        "secrets read/export",
        "cloud/external export",
        "EXE packaging",
    }
    assert set(contract["source_boundaries"]) >= {
        "no PySide import",
        "no QML import",
        "no runtime loop import",
        "no TradingController import",
        "no DecisionEnvelope import",
        "no strategy/AI/scoring/recommendation import",
        "no order module import",
        "no live adapter import",
        "no testnet adapter import",
        "no account module import",
        "no secrets module import",
        "no filesystem I/O",
        "no network I/O",
        "no QML changes",
        "no .bat changes",
        "no app.py changes",
        "no dependency declarations changes",
        "no workflow changes",
    }


def test_future_steps_prepare_9_1_through_9_9() -> None:
    assert build_preview_paper_decision_to_order_contract()["future_steps"] == [
        "functional_preview_9_1_paper_order_intent_read_model",
        "functional_preview_9_2_paper_order_static_fixture",
        "functional_preview_9_3_paper_order_audit_envelope",
        "functional_preview_9_4_ui_read_only_paper_order_surface",
        "functional_preview_9_5_controlled_paper_order_intent_selection_gate",
        "functional_preview_9_6_controlled_paper_order_intent_no_submission",
        "functional_preview_9_7_paper_fill_simulator_contract_static_only",
        "functional_preview_9_8_paper_order_lifecycle_audit",
        "functional_preview_9_9_block_g_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_and_block_f_closure_audit() -> None:
    tree = ast.parse(_source(HELPER))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert imports == [
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_block_f_closure_audit",
    ]
    forbidden = (
        "PySide",
        "QML",
        "runtime",
        "TradingController",
        "DecisionEnvelope",
        "strategy",
        "scoring",
        "recommendation",
        "order",
        "live",
        "testnet",
        "account",
        "secrets",
        "network",
        "filesystem",
    )
    assert not any(any(term in module for term in forbidden) for module in imports)


def test_helper_has_no_forbidden_calls() -> None:
    tree = ast.parse(_source(HELPER))
    forbidden = {
        "open",
        "read_text",
        "write_text",
        "requests",
        "subprocess",
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
    }
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
    assert call_names.isdisjoint(forbidden)


def test_qml_keeps_single_allowed_preview_select_action_and_no_forbidden_calls() -> None:
    qml = _source(MAIN_WINDOW) + "\n" + _source(OPERATOR_DASHBOARD)

    assert qml.count("previewSelectAction(") == 1
    assert qml.count(ALLOWED_CALL) == 1
    assert "previewSelectSourceControl(" not in qml
    assert "resetPreviewSelection(" not in qml
    for forbidden in (
        "startRuntime",
        "start_runtime",
        "startLoop",
        "start_loop",
        "stopRuntime",
        "pauseRuntime",
        "resumeRuntime",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
    ):
        assert forbidden not in qml
