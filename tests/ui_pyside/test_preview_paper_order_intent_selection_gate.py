"""Source-only tests for BLOK G paper order intent selection gate."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_order_audit_envelope import (
    build_preview_paper_order_audit_envelope,
)
from ui.pyside_app.preview_paper_order_intent_selection_gate import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_6,
    SELECTION_GATE_DECISION,
    SELECTION_GATE_STATUS,
    build_preview_paper_order_intent_selection_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_order_intent_selection_gate.py"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'

EXPECTED_CASE_IDS = [
    "baseline_btc_no_intent_no_order",
    "eth_size_preview_no_intent_no_order",
    "sol_risk_blocked_preview_no_intent_no_order",
    "unknown_input_keys_reported_no_execution",
]
EXPECTED_TOP_LEVEL = {
    "schema_version",
    "selection_gate_kind",
    "block",
    "step",
    "selection_gate_status",
    "selection_gate_decision",
    "ready_for_block_g_6",
    "next_step",
    "next_step_title",
    "audit_envelope_reference",
    "available_selections",
    "selected_case_id",
    "selection_result",
    "selected_audit_event_preview",
    "selection_no_execution_evidence",
    "selection_gate_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_AVAILABLE_SELECTION_FLAGS = {
    "selection_status": "available_for_controlled_preview_selection_only",
    "paper_only": True,
    "local_only": True,
    "intent_generation_allowed": False,
    "order_generation_allowed": False,
    "submission_allowed": False,
    "fills_allowed": False,
    "runtime_execution_allowed": False,
    "audit_export_allowed": False,
}
EXPECTED_NO_EXECUTION = {
    "selection_gate_evaluated": True,
    "paper_order_intent_generated": False,
    "paper_order_generated": False,
    "paper_order_submitted": False,
    "paper_fill_simulated": False,
    "paper_runtime_execution_performed": False,
    "risk_governor_execution_performed": False,
    "audit_export_performed": False,
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
    "selection_gate_only": True,
    "ui_surface_read_only": True,
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
    "selection_gate_evaluated": True,
    "selection_acceptance_can_enable_intent_generation": False,
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
REQUIRED_BLOCKED = {
    "paper order intent selection enabling execution",
    "selection-driven order intent generation",
    "selection-driven paper order generation",
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
REQUIRED_SOURCE_BOUNDARIES = {
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
FORBIDDEN_IMPORT_PARTS = {
    "PySide",
    "QtQml",
    "QQmlApplicationEngine",
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
    "requests",
    "subprocess",
    "pathlib",
}
FORBIDDEN_CALLS = {
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


def test_selection_gate_returns_json_serializable_plain_dict() -> None:
    gate = build_preview_paper_order_intent_selection_gate()

    assert isinstance(gate, dict)
    assert set(gate) == EXPECTED_TOP_LEVEL
    _assert_plain(gate)
    assert json.loads(json.dumps(gate, sort_keys=True)) == gate


def test_status_decision_next_step_and_audit_reference_are_fixed() -> None:
    gate = build_preview_paper_order_intent_selection_gate()

    assert gate["schema_version"] == PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION
    assert gate["selection_gate_kind"] == PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND
    assert gate["block"] == "G"
    assert gate["step"] == "9.5"
    assert gate["selection_gate_status"] == SELECTION_GATE_STATUS
    assert gate["selection_gate_decision"] == SELECTION_GATE_DECISION
    assert gate["ready_for_block_g_6"] is READY_FOR_BLOCK_G_6 is True
    assert gate["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.6"
    assert (
        gate["next_step_title"] == NEXT_STEP_TITLE == "CONTROLLED PAPER ORDER INTENT NO SUBMISSION"
    )
    assert gate["audit_envelope_reference"] == {
        "schema_version": "preview_paper_order_audit_envelope.v1",
        "envelope_kind": "functional_preview_block_g_paper_order_audit_envelope",
        "audit_envelope_status": "paper_order_audit_envelope_ready_no_order_generation",
        "audit_envelope_decision": "BUILD_PAPER_ORDER_AUDIT_ENVELOPE_ONLY_NO_ORDER_GENERATION",
        "ready_for_block_g_4": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.4",
    }


def test_available_selections_match_four_audit_events_and_are_selection_only() -> None:
    gate = build_preview_paper_order_intent_selection_gate()
    envelope = build_preview_paper_order_audit_envelope()
    selections = gate["available_selections"]

    assert len(selections) == 4
    assert [selection["case_id"] for selection in selections] == EXPECTED_CASE_IDS
    assert [selection["case_id"] for selection in selections] == [
        event["case_id"] for event in envelope["audit_events"]
    ]
    assert [selection["event_id"] for selection in selections] == [
        event["event_id"] for event in envelope["audit_events"]
    ]
    for selection in selections:
        assert set(selection) == {
            "case_id",
            "event_id",
            "label",
            "selection_status",
            "paper_only",
            "local_only",
            "intent_generation_allowed",
            "order_generation_allowed",
            "submission_allowed",
            "fills_allowed",
            "runtime_execution_allowed",
            "audit_export_allowed",
        }
        for key, expected in EXPECTED_AVAILABLE_SELECTION_FLAGS.items():
            assert (
                selection[key] is expected
                if isinstance(expected, bool)
                else selection[key] == expected
            )


def test_no_selection_default_is_fail_safe_no_execution() -> None:
    gate = build_preview_paper_order_intent_selection_gate()

    assert gate["selected_case_id"] is None
    assert gate["selection_result"] == {
        "selection_status": "no_selection_preview_only",
        "selection_accepted": False,
        "selection_rejected": False,
        "rejection_reason": None,
        "selected_case_id": None,
        "selected_event_id": None,
        "intent_generation_allowed": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
    }
    assert gate["selected_audit_event_preview"] == {
        "selected": False,
        "case_id": None,
        "event_id": None,
        "input_snapshot": None,
        "unknown_input_keys": [],
        "paper_order_fixture_preview": None,
        "paper_order_audit_preview": None,
    }
    assert gate["selection_no_execution_evidence"] == EXPECTED_NO_EXECUTION


def test_known_selection_for_each_case_is_preview_only_and_copy_safe() -> None:
    envelope_before = build_preview_paper_order_audit_envelope()
    events_by_case = {event["case_id"]: event for event in envelope_before["audit_events"]}

    for case_id in EXPECTED_CASE_IDS:
        gate = build_preview_paper_order_intent_selection_gate(case_id)
        event = events_by_case[case_id]
        assert gate["selected_case_id"] == case_id
        assert (
            gate["selection_result"]["selection_status"]
            == "accepted_for_preview_only_no_intent_generation"
        )
        assert gate["selection_result"]["selection_accepted"] is True
        assert gate["selection_result"]["selection_rejected"] is False
        assert gate["selection_result"]["selected_event_id"] == event["event_id"]
        assert gate["selection_result"]["intent_generation_allowed"] is False
        assert gate["selection_result"]["order_generation_allowed"] is False
        preview = gate["selected_audit_event_preview"]
        assert preview == {
            "selected": True,
            "case_id": event["case_id"],
            "event_id": event["event_id"],
            "input_snapshot": event["input_snapshot"],
            "unknown_input_keys": event["unknown_input_keys"],
            "paper_order_fixture_preview": event["paper_order_fixture_preview"],
            "paper_order_audit_preview": event["paper_order_audit_preview"],
        }
        preview["input_snapshot"]["mutated"] = True
        assert build_preview_paper_order_audit_envelope() == envelope_before


def test_unknown_empty_and_non_string_selection_fail_closed_without_exception() -> None:
    for value in ["unknown", "", "  ", 123, 1.5, True, [], {}]:
        gate = build_preview_paper_order_intent_selection_gate(value)  # type: ignore[arg-type]
        assert gate["selected_case_id"] is None
        assert (
            gate["selection_result"]["selection_status"]
            == "rejected_fail_closed_unknown_or_invalid_selection"
        )
        assert gate["selection_result"]["selection_accepted"] is False
        assert gate["selection_result"]["selection_rejected"] is True
        assert gate["selection_result"]["rejection_reason"] == "unknown_or_invalid_selection"
        assert gate["selected_audit_event_preview"]["selected"] is False
        assert gate["selection_no_execution_evidence"] == EXPECTED_NO_EXECUTION


def test_case_2_eth_selection_preserves_pair_and_size_without_intent_or_order() -> None:
    gate = build_preview_paper_order_intent_selection_gate("eth_size_preview_no_intent_no_order")
    snapshot = gate["selected_audit_event_preview"]["input_snapshot"]

    assert snapshot["operator_selected_pair"] == "ETH/USDT"
    assert snapshot["operator_selected_candidate"]["pair"] == "ETH/USDT"
    assert snapshot["paper_order_intent_size_preview"]["value"] == 25.0
    assert gate["selection_result"]["intent_generation_allowed"] is False
    assert gate["selection_result"]["order_generation_allowed"] is False
    assert gate["selection_no_execution_evidence"]["paper_order_intent_generated"] is False
    assert gate["selection_no_execution_evidence"]["paper_order_generated"] is False


def test_case_3_sol_risk_blocked_selection_preserves_preview_without_risk_execution() -> None:
    gate = build_preview_paper_order_intent_selection_gate(
        "sol_risk_blocked_preview_no_intent_no_order"
    )
    snapshot = gate["selected_audit_event_preview"]["input_snapshot"]

    assert snapshot["operator_selected_pair"] == "SOL/USDT"
    assert snapshot["risk_check_preview"]["risk_status"] == "blocked_preview_only"
    assert snapshot["risk_check_preview"]["risk_engine_execution_performed"] is False
    assert gate["selection_no_execution_evidence"]["risk_governor_execution_performed"] is False
    assert gate["selection_no_execution_evidence"]["paper_order_generated"] is False


def test_case_4_unknown_keys_are_reported_not_executed() -> None:
    gate = build_preview_paper_order_intent_selection_gate(
        "unknown_input_keys_reported_no_execution"
    )
    preview = gate["selected_audit_event_preview"]

    assert preview["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert "unsafe_submit_order_request" not in preview["input_snapshot"]
    assert "live_credentials_reference" not in preview["input_snapshot"]
    assert gate["selection_no_execution_evidence"]["secrets_read_performed"] is False
    assert gate["selection_no_execution_evidence"]["export_performed"] is False
    assert gate["selection_result"]["submission_allowed"] is False


def test_selection_summary_boundary_blocked_and_source_boundaries_are_complete() -> None:
    gate = build_preview_paper_order_intent_selection_gate("baseline_btc_no_intent_no_order")

    assert gate["selection_gate_summary"] == {
        "available_selection_count": 4,
        "selection_requested": True,
        "selection_accepted": True,
        "selection_rejected": False,
        "known_selection": True,
        "ready_for_controlled_intent_preview_step": True,
        "all_available_selections_no_intent_generation": True,
        "all_available_selections_no_order_generation": True,
        "all_available_selections_no_submission": True,
        "all_available_selections_no_fills": True,
        "all_available_selections_no_runtime_execution": True,
        "all_available_selections_no_live_or_testnet": True,
        "all_available_selections_no_account_or_secrets": True,
        "all_available_selections_no_export": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.6",
    }
    for key, expected in EXPECTED_BOUNDARY_CHECKS.items():
        assert gate["boundary_checks"][key] is expected
    assert REQUIRED_BLOCKED.issubset(set(gate["blocked_capabilities"]))
    assert REQUIRED_SOURCE_BOUNDARIES.issubset(set(gate["source_boundaries"]))
    assert gate["future_steps"] == [
        "functional_preview_9_6_controlled_paper_order_intent_no_submission",
        "functional_preview_9_7_paper_fill_simulator_contract_static_only",
        "functional_preview_9_8_paper_order_lifecycle_audit",
        "functional_preview_9_9_block_g_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_and_audit_envelope_and_has_no_forbidden_calls() -> None:
    source = _source(HELPER)
    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    assert imports == [
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_paper_order_audit_envelope",
    ]
    assert all(
        imported == "ui.pyside_app.preview_paper_order_audit_envelope"
        or not any(forbidden in imported for forbidden in FORBIDDEN_IMPORT_PARTS)
        for imported in imports
    )
    assert FORBIDDEN_CALLS.isdisjoint(calls)


def test_qml_files_are_unchanged_source_guarded_and_no_new_calls_added() -> None:
    dashboard = _source(OPERATOR_DASHBOARD)
    main_window = _source(MAIN_WINDOW)
    qml = dashboard + "\n" + main_window

    assert dashboard.count("previewSelectAction(") == 1
    assert dashboard.count(ALLOWED_CALL) == 1
    assert qml.count("previewSelectSourceControl(") == 0
    assert qml.count("resetPreviewSelection(") == 0
    for forbidden in [
        "start_runtime",
        "start_loop",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
    ]:
        assert forbidden not in qml
