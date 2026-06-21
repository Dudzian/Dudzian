"""Source-only tests for BLOK G paper order intent read model."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_order_intent_read_model import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND,
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION,
    READ_MODEL_DECISION,
    READ_MODEL_STATUS,
    READY_FOR_BLOCK_G_2,
    build_preview_paper_order_intent_read_model_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_order_intent_read_model.py"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "read_model_kind",
    "block",
    "step",
    "read_model_status",
    "read_model_decision",
    "ready_for_block_g_2",
    "next_step",
    "next_step_title",
    "contract_reference",
    "input_snapshot",
    "input_snapshot_echo",
    "paper_order_intent_read_model",
    "paper_order_intent_preview",
    "paper_order_validation_preview",
    "paper_order_refusal_preview",
    "paper_order_gate_status",
    "paper_order_risk_gate_preview",
    "paper_order_no_execution_status",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}

EXPECTED_BOUNDARY_CHECKS = {
    "local_only": True,
    "paper_only": True,
    "read_model_only": True,
    "contract_only": False,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_allowed_now": False,
    "paper_order_intent_generated": False,
    "paper_order_generation_allowed_now": False,
    "paper_order_submission_allowed_now": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_runtime_execution_allowed_now": False,
    "risk_governor_execution_allowed_now": False,
    "risk_governor_execution_performed": False,
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


def test_read_model_returns_json_serializable_plain_dict() -> None:
    snapshot = build_preview_paper_order_intent_read_model_snapshot()

    assert isinstance(snapshot, dict)
    assert set(snapshot) == EXPECTED_TOP_LEVEL
    _assert_plain(snapshot)
    assert json.loads(json.dumps(snapshot, sort_keys=True)) == snapshot


def test_status_decision_next_step_and_contract_reference_are_fixed() -> None:
    snapshot = build_preview_paper_order_intent_read_model_snapshot()

    assert snapshot["schema_version"] == PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION
    assert snapshot["read_model_kind"] == PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND
    assert snapshot["block"] == "G"
    assert snapshot["step"] == "9.1"
    assert snapshot["read_model_status"] == READ_MODEL_STATUS
    assert snapshot["read_model_decision"] == READ_MODEL_DECISION
    assert snapshot["ready_for_block_g_2"] is READY_FOR_BLOCK_G_2 is True
    assert snapshot["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.2"
    assert snapshot["next_step_title"] == NEXT_STEP_TITLE == "PAPER ORDER STATIC FIXTURE"
    assert snapshot["contract_reference"] == {
        "schema_version": "preview_paper_decision_to_order_contract.v1",
        "contract_kind": "functional_preview_block_g_paper_decision_to_order_contract",
        "block_status": "paper_decision_to_order_contract_ready_no_order_execution",
        "contract_decision": "START_BLOCK_G_WITH_CONTRACT_ONLY_NO_ORDER_EXECUTION",
        "ready_for_block_g_1": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.1",
    }


def test_default_input_snapshot_is_normalized_and_echoed() -> None:
    snapshot = build_preview_paper_order_intent_read_model_snapshot()
    input_snapshot = snapshot["input_snapshot"]

    assert (
        input_snapshot["paper_order_intent_context_id"]
        == "local-preview-paper-order-intent-context"
    )
    assert input_snapshot["dry_run_decision_preview"] == {
        "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
        "decision_status": "not_executed",
        "confidence_preview": 0.0,
    }
    assert input_snapshot["operator_selected_pair"] == "BTC/USDT"
    assert snapshot["input_snapshot_echo"] == {**input_snapshot, "unknown_input_keys": []}


def test_custom_input_snapshot_is_defensively_copied_not_mutated_and_unknown_keys_reported() -> (
    None
):
    custom = {
        "paper_order_intent_context_id": "ctx-1",
        "dry_run_decision_preview": {
            "decision_action": "HOLD",
            "decision_status": "preview",
            "confidence_preview": 0.2,
        },
        "operator_selected_pair": "ETH/USDT",
        "operator_selected_candidate": {"pair": "ETH/USDT", "source": "test", "confidence": 0.2},
        "unknown_execution_request": {"submit_order": True},
    }
    original = json.loads(json.dumps(custom))

    snapshot = build_preview_paper_order_intent_read_model_snapshot(custom)
    custom["operator_selected_candidate"]["pair"] = "MUTATED/USDT"

    assert original["operator_selected_candidate"]["pair"] == "ETH/USDT"
    assert snapshot["input_snapshot"]["operator_selected_candidate"]["pair"] == "ETH/USDT"
    assert (
        snapshot["input_snapshot"]["decision_reason_summary"]
        == "read-model-only placeholder; no decision execution"
    )
    assert snapshot["input_snapshot_echo"]["unknown_input_keys"] == ["unknown_execution_request"]
    assert "unknown_execution_request" not in snapshot["input_snapshot"]


def test_intent_preview_validation_refusal_gate_risk_and_no_execution_are_safe() -> None:
    snapshot = build_preview_paper_order_intent_read_model_snapshot()

    assert (
        snapshot["paper_order_intent_read_model"]
        | {
            "read_model_only": True,
            "order_intent_generated": False,
            "order_generation_allowed": False,
            "order_submission_allowed": False,
            "fill_simulation_allowed": False,
            "runtime_execution_allowed": False,
        }
        == snapshot["paper_order_intent_read_model"]
    )
    assert (
        snapshot["paper_order_intent_preview"]["intent_action"] == "NO_PAPER_ORDER_INTENT_GENERATED"
    )
    assert snapshot["paper_order_intent_preview"]["order_intent_generated"] is False
    assert snapshot["paper_order_validation_preview"] == {
        "validation_status": "not_evaluated_read_model_only",
        "validation_performed": False,
        "validation_passed": False,
        "risk_governor_evaluated": False,
        "manual_confirmation_present": False,
        "kill_switch_checked": False,
    }
    assert snapshot["paper_order_refusal_preview"]["blocked_by_contract"] is True
    assert all(
        snapshot["paper_order_gate_status"][key] is False
        for key in (
            "paper_order_generation_gate_open",
            "paper_order_submission_gate_open",
            "paper_fill_simulation_gate_open",
            "runtime_execution_gate_open",
        )
    )
    assert snapshot["paper_order_risk_gate_preview"] == {
        "risk_gate_status": "required_not_evaluated",
        "risk_governor_execution_allowed_now": False,
        "risk_governor_execution_performed": False,
        "risk_governor_required_before_any_order_execution": True,
    }
    assert all(value is False for value in snapshot["paper_order_no_execution_status"].values())


def test_boundary_checks_blocked_capabilities_and_source_boundaries_are_complete() -> None:
    snapshot = build_preview_paper_order_intent_read_model_snapshot()

    assert snapshot["boundary_checks"] == EXPECTED_BOUNDARY_CHECKS
    assert set(snapshot["blocked_capabilities"]) >= {
        "paper order intent generation now",
        "paper order validation now",
        "risk governor evaluation now",
        "paper order submission now",
        "paper fills now",
        "live/testnet/account/secrets/export/cloud",
        "TradingController / DecisionEnvelope",
        "QML changes / new QML calls",
        "EXE packaging",
    }
    assert set(snapshot["source_boundaries"]) >= {
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


def test_helper_imports_only_safe_stdlib_and_9_0_contract() -> None:
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
        "ui.pyside_app.preview_paper_decision_to_order_contract",
    ]
    forbidden = (
        "PySide",
        "QML",
        "runtime_loop",
        "TradingController",
        "DecisionEnvelope",
        "strategy",
        "scoring",
        "recommendation",
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
