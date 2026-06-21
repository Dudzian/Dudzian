"""Source-only tests for BLOK G paper order audit envelope."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_order_audit_envelope import (
    AUDIT_ENVELOPE_DECISION,
    AUDIT_ENVELOPE_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_4,
    build_preview_paper_order_audit_envelope,
)
from ui.pyside_app.preview_paper_order_static_fixture import (
    build_preview_paper_order_static_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_order_audit_envelope.py"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "envelope_kind",
    "block",
    "step",
    "audit_envelope_status",
    "audit_envelope_decision",
    "ready_for_block_g_4",
    "next_step",
    "next_step_title",
    "static_fixture_reference",
    "audit_events",
    "audit_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_EVENT_IDS = [
    "paper-order-audit-0001-baseline-btc-no-intent-no-order",
    "paper-order-audit-0002-eth-size-preview-no-intent-no-order",
    "paper-order-audit-0003-sol-risk-blocked-preview-no-intent-no-order",
    "paper-order-audit-0004-unknown-input-keys-reported-no-execution",
]
EXPECTED_CASE_IDS = [
    "baseline_btc_no_intent_no_order",
    "eth_size_preview_no_intent_no_order",
    "sol_risk_blocked_preview_no_intent_no_order",
    "unknown_input_keys_reported_no_execution",
]
EXPECTED_EVENT_KEYS = {
    "event_id",
    "event_type",
    "event_status",
    "case_id",
    "case_description",
    "input_snapshot",
    "read_model_reference",
    "paper_order_fixture_preview",
    "paper_order_audit_preview",
    "no_execution_evidence",
    "boundary_snapshot",
    "unknown_input_keys",
    "audit_export_status",
}
EXPECTED_AUDIT_PREVIEW = {
    "audit_preview_status": "static_audit_only_no_export",
    "paper_only": True,
    "local_only": True,
    "static_fixture_only": True,
    "audit_event_generated": True,
    "audit_export_allowed": False,
    "audit_export_performed": False,
    "order_intent_generated": False,
    "order_generated": False,
    "order_submitted": False,
    "fill_simulated": False,
    "runtime_execution_performed": False,
    "live_execution_performed": False,
    "testnet_execution_performed": False,
}
EXPECTED_EXPORT_STATUS = {
    "export_allowed": False,
    "export_performed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "secrets_export_allowed": False,
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
    "audit_envelope_only": True,
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
    "audit_event_generation_allowed": True,
    "audit_export_allowed": False,
    "audit_export_performed": False,
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


def _events_by_case_id() -> dict[str, dict[str, object]]:
    envelope = build_preview_paper_order_audit_envelope()
    return {event["case_id"]: event for event in envelope["audit_events"]}


def test_audit_envelope_returns_json_serializable_plain_dict() -> None:
    envelope = build_preview_paper_order_audit_envelope()

    assert isinstance(envelope, dict)
    assert set(envelope) == EXPECTED_TOP_LEVEL
    _assert_plain(envelope)
    assert json.loads(json.dumps(envelope, sort_keys=True)) == envelope


def test_status_decision_next_step_and_static_fixture_reference_are_fixed() -> None:
    envelope = build_preview_paper_order_audit_envelope()

    assert envelope["schema_version"] == PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION
    assert envelope["envelope_kind"] == PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND
    assert envelope["block"] == "G"
    assert envelope["step"] == "9.3"
    assert envelope["audit_envelope_status"] == AUDIT_ENVELOPE_STATUS
    assert envelope["audit_envelope_decision"] == AUDIT_ENVELOPE_DECISION
    assert envelope["ready_for_block_g_4"] is READY_FOR_BLOCK_G_4 is True
    assert envelope["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.4"
    assert envelope["next_step_title"] == NEXT_STEP_TITLE == "PAPER ORDER UI READ-ONLY SURFACE"
    assert envelope["static_fixture_reference"] == {
        "schema_version": "preview_paper_order_static_fixture.v1",
        "fixture_kind": "functional_preview_block_g_paper_order_static_fixture",
        "static_fixture_status": "paper_order_static_fixture_ready_no_order_generation",
        "static_fixture_decision": "BUILD_PAPER_ORDER_STATIC_FIXTURE_ONLY_NO_ORDER_GENERATION",
        "ready_for_block_g_3": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.3",
    }


def test_audit_events_are_deterministic_and_match_static_fixture_cases() -> None:
    envelope = build_preview_paper_order_audit_envelope()
    fixture = build_preview_paper_order_static_fixture()
    events = envelope["audit_events"]

    assert len(events) == 4
    assert [event["event_id"] for event in events] == EXPECTED_EVENT_IDS
    assert [event["case_id"] for event in events] == EXPECTED_CASE_IDS
    assert [case["case_id"] for case in fixture["fixture_cases"]] == EXPECTED_CASE_IDS
    for event, case in zip(events, fixture["fixture_cases"], strict=True):
        assert set(event) == EXPECTED_EVENT_KEYS
        assert event["event_type"] == "paper_order_static_fixture_case_audit"
        assert event["event_status"] == "ready_no_intent_no_order_no_execution"
        assert event["case_id"] == case["case_id"]
        assert event["case_description"] == case["case_description"]
        assert event["input_snapshot"] == case["input_snapshot"]
        assert event["paper_order_fixture_preview"] == case["paper_order_fixture_preview"]
        assert event["paper_order_audit_preview"] == EXPECTED_AUDIT_PREVIEW
        assert event["no_execution_evidence"] == EXPECTED_NO_EXECUTION
        assert event["no_execution_evidence"] == case["fixture_no_execution_evidence"]
        assert event["audit_export_status"] == EXPECTED_EXPORT_STATUS


def test_event_read_model_reference_is_limited_to_9_1_reference_fields() -> None:
    envelope = build_preview_paper_order_audit_envelope()

    for event in envelope["audit_events"]:
        assert event["read_model_reference"] == {
            "schema_version": "preview_paper_order_intent_read_model.v1",
            "read_model_kind": "functional_preview_block_g_paper_order_intent_read_model",
            "read_model_status": "paper_order_intent_read_model_ready_no_order_generation",
            "read_model_decision": "BUILD_PAPER_ORDER_INTENT_READ_MODEL_ONLY_NO_ORDER_GENERATION",
            "ready_for_block_g_2": True,
            "next_step": "FUNCTIONAL-PREVIEW-9.2",
        }


def test_case_2_audits_eth_size_preview_without_intent_or_order() -> None:
    event = _events_by_case_id()["eth_size_preview_no_intent_no_order"]

    assert event["input_snapshot"]["operator_selected_pair"] == "ETH/USDT"
    assert event["input_snapshot"]["operator_selected_candidate"]["pair"] == "ETH/USDT"
    assert event["input_snapshot"]["paper_order_intent_size_preview"]["value"] == 25.0
    assert event["paper_order_audit_preview"]["order_intent_generated"] is False
    assert event["paper_order_audit_preview"]["order_generated"] is False
    assert event["no_execution_evidence"]["paper_order_intent_generated"] is False
    assert event["no_execution_evidence"]["paper_order_generated"] is False


def test_case_3_audits_risk_blocked_preview_without_risk_execution_or_order() -> None:
    event = _events_by_case_id()["sol_risk_blocked_preview_no_intent_no_order"]

    assert event["input_snapshot"]["operator_selected_pair"] == "SOL/USDT"
    assert event["input_snapshot"]["risk_check_preview"] == {
        "risk_status": "blocked_preview_only",
        "risk_engine_execution_performed": False,
    }
    assert event["no_execution_evidence"]["risk_governor_execution_performed"] is False
    assert event["no_execution_evidence"]["paper_order_generated"] is False


def test_case_4_reports_unknown_keys_without_executing_or_normalizing_them() -> None:
    event = _events_by_case_id()["unknown_input_keys_reported_no_execution"]

    assert event["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert "live_credentials_reference" not in event["input_snapshot"]
    assert "unsafe_submit_order_request" not in event["input_snapshot"]
    assert event["no_execution_evidence"]["secrets_read_performed"] is False
    assert event["no_execution_evidence"]["live_execution_performed"] is False
    assert event["audit_export_status"] == EXPECTED_EXPORT_STATUS


def test_audit_summary_boundaries_blocked_capabilities_and_source_boundaries_are_complete() -> None:
    envelope = build_preview_paper_order_audit_envelope()

    assert envelope["audit_summary"] == {
        "event_count": 4,
        "all_events_static_audit_only": True,
        "all_events_no_intent_generated": True,
        "all_events_no_order_generated": True,
        "all_events_no_submission": True,
        "all_events_no_fills": True,
        "all_events_no_runtime_execution": True,
        "all_events_no_live_or_testnet": True,
        "all_events_no_account_or_secrets": True,
        "all_events_no_export": True,
        "unknown_input_key_events": 1,
        "ready_for_ui_read_only_surface_step": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.4",
    }
    assert envelope["boundary_checks"] == EXPECTED_BOUNDARY_CHECKS
    assert set(envelope["blocked_capabilities"]) >= {
        "paper order audit export",
        "paper order audit runtime dispatch",
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
    assert set(envelope["source_boundaries"]) >= {
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


def test_helper_imports_only_safe_stdlib_and_9_2_static_fixture() -> None:
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
        "ui.pyside_app.preview_paper_order_static_fixture",
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
