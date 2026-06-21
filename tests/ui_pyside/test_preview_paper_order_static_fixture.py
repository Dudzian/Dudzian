"""Source-only tests for BLOK G paper order static fixture."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_order_static_fixture import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_3,
    STATIC_FIXTURE_DECISION,
    STATIC_FIXTURE_STATUS,
    build_preview_paper_order_static_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_order_static_fixture.py"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "fixture_kind",
    "block",
    "step",
    "static_fixture_status",
    "static_fixture_decision",
    "ready_for_block_g_3",
    "next_step",
    "next_step_title",
    "read_model_reference",
    "fixture_cases",
    "fixture_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}

EXPECTED_CASE_IDS = [
    "baseline_btc_no_intent_no_order",
    "eth_size_preview_no_intent_no_order",
    "sol_risk_blocked_preview_no_intent_no_order",
    "unknown_input_keys_reported_no_execution",
]

EXPECTED_CASE_KEYS = {
    "case_id",
    "case_description",
    "input_snapshot",
    "read_model_snapshot",
    "paper_order_fixture_preview",
    "fixture_no_execution_evidence",
    "boundary_snapshot",
    "case_status",
}

EXPECTED_PREVIEW = {
    "fixture_status": "static_fixture_only_no_order_generation",
    "paper_only": True,
    "local_only": True,
    "read_model_only": True,
    "static_fixture_only": True,
    "order_intent_generated": False,
    "order_generated": False,
    "order_submission_allowed": False,
    "fill_simulation_allowed": False,
    "runtime_execution_allowed": False,
    "live_execution_allowed": False,
    "testnet_execution_allowed": False,
}

EXPECTED_NO_EXECUTION = {
    "decision_engine_execution_performed": False,
    "paper_order_intent_generated": False,
    "paper_order_generated": False,
    "paper_order_submitted": False,
    "paper_fill_simulated": False,
    "paper_runtime_execution_performed": False,
    "risk_governor_execution_performed": False,
    "trading_controller_touched": False,
    "decision_envelope_touched": False,
    "live_execution_performed": False,
    "testnet_execution_performed": False,
    "account_fetch_performed": False,
    "secrets_read_performed": False,
    "export_performed": False,
}

EXPECTED_BOUNDARY_CHECKS = {
    "local_only": True,
    "paper_only": True,
    "static_fixture_only": True,
    "read_model_only": True,
    "contract_only": False,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_allowed_now": False,
    "paper_order_intent_generated": False,
    "paper_order_generation_allowed_now": False,
    "paper_order_generated": False,
    "paper_order_submission_allowed_now": False,
    "paper_order_submitted": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_fill_simulated": False,
    "paper_runtime_execution_allowed_now": False,
    "paper_runtime_execution_performed": False,
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


def _cases_by_id() -> dict[str, dict[str, object]]:
    fixture = build_preview_paper_order_static_fixture()
    return {case["case_id"]: case for case in fixture["fixture_cases"]}


def test_static_fixture_returns_json_serializable_plain_dict() -> None:
    fixture = build_preview_paper_order_static_fixture()

    assert isinstance(fixture, dict)
    assert set(fixture) == EXPECTED_TOP_LEVEL
    _assert_plain(fixture)
    assert json.loads(json.dumps(fixture, sort_keys=True)) == fixture


def test_status_decision_next_step_and_read_model_reference_are_fixed() -> None:
    fixture = build_preview_paper_order_static_fixture()

    assert fixture["schema_version"] == PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION
    assert fixture["fixture_kind"] == PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND
    assert fixture["block"] == "G"
    assert fixture["step"] == "9.2"
    assert fixture["static_fixture_status"] == STATIC_FIXTURE_STATUS
    assert fixture["static_fixture_decision"] == STATIC_FIXTURE_DECISION
    assert fixture["ready_for_block_g_3"] is READY_FOR_BLOCK_G_3 is True
    assert fixture["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.3"
    assert fixture["next_step_title"] == NEXT_STEP_TITLE == "PAPER ORDER AUDIT ENVELOPE"
    assert fixture["read_model_reference"] == {
        "schema_version": "preview_paper_order_intent_read_model.v1",
        "read_model_kind": "functional_preview_block_g_paper_order_intent_read_model",
        "read_model_status": "paper_order_intent_read_model_ready_no_order_generation",
        "read_model_decision": "BUILD_PAPER_ORDER_INTENT_READ_MODEL_ONLY_NO_ORDER_GENERATION",
        "ready_for_block_g_2": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.2",
    }


def test_fixture_cases_are_exact_and_each_uses_9_1_read_model_snapshot() -> None:
    fixture = build_preview_paper_order_static_fixture()
    cases = fixture["fixture_cases"]

    assert len(cases) == 4
    assert [case["case_id"] for case in cases] == EXPECTED_CASE_IDS
    for case in cases:
        assert set(case) == EXPECTED_CASE_KEYS
        read_model = case["read_model_snapshot"]
        assert read_model["schema_version"] == "preview_paper_order_intent_read_model.v1"
        assert (
            read_model["read_model_kind"]
            == "functional_preview_block_g_paper_order_intent_read_model"
        )
        assert read_model["step"] == "9.1"
        assert read_model["next_step"] == "FUNCTIONAL-PREVIEW-9.2"
        assert case["input_snapshot"] == read_model["input_snapshot"]
        assert case["paper_order_fixture_preview"] == EXPECTED_PREVIEW
        assert case["fixture_no_execution_evidence"] == EXPECTED_NO_EXECUTION
        assert case["case_status"] == "static_fixture_case_ready_no_intent_no_order_no_execution"


def test_case_2_echoes_eth_size_preview_but_no_intent_or_order() -> None:
    case = _cases_by_id()["eth_size_preview_no_intent_no_order"]
    read_model = case["read_model_snapshot"]

    assert case["input_snapshot"]["operator_selected_pair"] == "ETH/USDT"
    assert case["input_snapshot"]["operator_selected_candidate"]["pair"] == "ETH/USDT"
    assert case["input_snapshot"]["operator_selected_candidate"]["confidence"] == 0.15
    assert case["input_snapshot"]["paper_order_intent_size_preview"]["value"] == 25.0
    assert case["input_snapshot"]["paper_order_intent_side_preview"] == "buy_preview_only"
    assert case["input_snapshot"]["paper_order_intent_type_preview"] == "market_preview_only"
    assert read_model["input_snapshot_echo"]["unknown_input_keys"] == []
    assert read_model["paper_order_intent_preview"]["order_intent_generated"] is False
    assert read_model["paper_order_no_execution_status"]["order_generated"] is False
    assert case["fixture_no_execution_evidence"]["paper_order_submitted"] is False
    assert case["fixture_no_execution_evidence"]["paper_fill_simulated"] is False


def test_case_3_shows_risk_blocked_preview_without_risk_execution_or_order() -> None:
    case = _cases_by_id()["sol_risk_blocked_preview_no_intent_no_order"]
    read_model = case["read_model_snapshot"]

    assert case["input_snapshot"]["operator_selected_pair"] == "SOL/USDT"
    assert case["input_snapshot"]["operator_selected_candidate"]["pair"] == "SOL/USDT"
    assert case["input_snapshot"]["operator_selected_candidate"]["confidence"] == 0.05
    assert case["input_snapshot"]["risk_check_preview"] == {
        "risk_status": "blocked_preview_only",
        "risk_engine_execution_performed": False,
    }
    assert case["input_snapshot"]["paper_order_intent_size_preview"]["value"] == 0.0
    assert case["input_snapshot"]["paper_order_intent_side_preview"] == "none"
    assert case["input_snapshot"]["paper_order_intent_type_preview"] == "none"
    assert read_model["paper_order_risk_gate_preview"]["risk_governor_execution_performed"] is False
    assert case["fixture_no_execution_evidence"]["risk_governor_execution_performed"] is False
    assert read_model["paper_order_intent_preview"]["order_intent_generated"] is False
    assert case["fixture_no_execution_evidence"]["paper_order_generated"] is False


def test_case_4_reports_unknown_keys_and_does_not_execute_or_echo_them_to_input_snapshot() -> None:
    case = _cases_by_id()["unknown_input_keys_reported_no_execution"]
    read_model = case["read_model_snapshot"]

    assert read_model["input_snapshot_echo"]["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert "unsafe_submit_order_request" not in case["input_snapshot"]
    assert "live_credentials_reference" not in case["input_snapshot"]
    assert case["fixture_no_execution_evidence"] == EXPECTED_NO_EXECUTION
    assert read_model["paper_order_no_execution_status"]["secrets_read_performed"] is False
    assert read_model["paper_order_no_execution_status"]["live_execution_performed"] is False


def test_fixture_summary_boundaries_blocked_capabilities_and_source_boundaries_are_complete() -> (
    None
):
    fixture = build_preview_paper_order_static_fixture()

    assert fixture["fixture_summary"] == {
        "case_count": 4,
        "all_cases_static_only": True,
        "all_cases_no_intent_generated": True,
        "all_cases_no_order_generated": True,
        "all_cases_no_submission": True,
        "all_cases_no_fills": True,
        "all_cases_no_runtime_execution": True,
        "all_cases_no_live_or_testnet": True,
        "all_cases_no_account_or_secrets": True,
        "all_cases_no_export": True,
        "ready_for_audit_envelope_step": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.3",
    }
    assert fixture["boundary_checks"] == EXPECTED_BOUNDARY_CHECKS
    assert set(fixture["blocked_capabilities"]) >= {
        "static fixture execution",
        "paper order intent generation now",
        "paper order generation now",
        "paper order submission now",
        "paper fill simulation now",
        "paper runtime execution now",
        "risk governor execution now",
        "live/testnet/account/secrets/export/cloud",
        "TradingController / DecisionEnvelope",
        "QML changes / new QML calls",
        "EXE packaging",
    }
    assert set(fixture["source_boundaries"]) >= {
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


def test_helper_imports_only_safe_stdlib_and_9_1_read_model() -> None:
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
        "ui.pyside_app.preview_paper_order_intent_read_model",
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
        "order.live",
        "order.testnet",
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
