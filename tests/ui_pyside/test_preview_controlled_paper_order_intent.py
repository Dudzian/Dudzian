"""Source-only tests for BLOK G controlled paper order intent preview."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_controlled_paper_order_intent import (
    CONTROLLED_INTENT_DECISION,
    CONTROLLED_INTENT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_7,
    build_preview_controlled_paper_order_intent,
)
from ui.pyside_app.preview_paper_order_intent_selection_gate import (
    build_preview_paper_order_intent_selection_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_controlled_paper_order_intent.py"
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
    "controlled_intent_kind",
    "block",
    "step",
    "controlled_intent_status",
    "controlled_intent_decision",
    "ready_for_block_g_7",
    "next_step",
    "next_step_title",
    "selection_gate_reference",
    "selected_case_id",
    "selection_result",
    "controlled_intent_result",
    "paper_order_intent_preview",
    "intent_preview_validation",
    "intent_preview_refusal",
    "intent_no_execution_evidence",
    "controlled_intent_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
EXECUTION_FALSE = {
    "order_generation_allowed": False,
    "submission_allowed": False,
    "fills_allowed": False,
    "runtime_execution_allowed": False,
}
REQUIRED_BOUNDARY = {
    "local_only": True,
    "paper_only": True,
    "controlled_intent_preview_only": True,
    "selection_gate_only": True,
    "ui_surface_read_only": True,
    "audit_envelope_only": True,
    "static_fixture_only": True,
    "read_model_only": True,
    "contract_only": False,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_preview_allowed_now": True,
    "executable_intent_generation_allowed": False,
    "executable_intent_generated": False,
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
    "future_risk_governor_required_before_order": True,
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
    "executable paper order intent generation",
    "preview intent conversion to order",
    "preview intent submission",
    "paper order generation now",
    "paper order submission now",
    "paper fill simulation now",
    "paper runtime execution now",
    "risk governor execution now",
    "audit export",
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


def _assert_all_execution_false(payload: dict[str, object]) -> None:
    for key, expected in EXECUTION_FALSE.items():
        assert payload[key] is expected


def test_controlled_intent_returns_json_serializable_plain_dict() -> None:
    preview = build_preview_controlled_paper_order_intent()

    assert isinstance(preview, dict)
    assert set(preview) == EXPECTED_TOP_LEVEL
    _assert_plain(preview)
    assert json.loads(json.dumps(preview, sort_keys=True)) == preview


def test_status_decision_next_step_and_selection_gate_reference_are_fixed() -> None:
    preview = build_preview_controlled_paper_order_intent()

    assert preview["schema_version"] == PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION
    assert preview["controlled_intent_kind"] == PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND
    assert preview["block"] == "G"
    assert preview["step"] == "9.6"
    assert preview["controlled_intent_status"] == CONTROLLED_INTENT_STATUS
    assert preview["controlled_intent_decision"] == CONTROLLED_INTENT_DECISION
    assert preview["ready_for_block_g_7"] is READY_FOR_BLOCK_G_7 is True
    assert preview["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.7"
    assert preview["next_step_title"] == NEXT_STEP_TITLE
    assert preview["selection_gate_reference"] == {
        "schema_version": "preview_paper_order_intent_selection_gate.v1",
        "selection_gate_kind": "functional_preview_block_g_paper_order_intent_selection_gate",
        "selection_gate_status": "paper_order_intent_selection_gate_ready_no_intent_generation",
        "selection_gate_decision": "BUILD_SELECTION_GATE_ONLY_NO_INTENT_GENERATION",
        "ready_for_block_g_6": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.6",
    }


def test_no_selection_default_does_not_build_preview_intent_and_is_safe() -> None:
    preview = build_preview_controlled_paper_order_intent()

    assert preview["selected_case_id"] is None
    assert preview["controlled_intent_result"] == {
        "intent_preview_status": "not_built_no_selection",
        "intent_preview_built": False,
        "intent_preview_rejected": False,
        "rejection_reason": None,
        "selected_case_id": None,
        "selected_event_id": None,
        "intent_preview_executable": False,
        **EXECUTION_FALSE,
    }
    assert preview["paper_order_intent_preview"] == {
        "preview_available": False,
        "preview_only": True,
        "executable": False,
        "case_id": None,
        "event_id": None,
        "pair": None,
        "side": None,
        "order_type": None,
        "size_value": 0.0,
        "size_unit": "preview_only",
        "source": "no_selection_or_rejected_selection",
        "confidence_preview": 0.0,
        "risk_status_preview": "not_evaluated_no_selection",
        "unknown_input_keys": [],
        **EXECUTION_FALSE,
    }
    assert preview["intent_preview_validation"]["validation_status"] == "not_evaluated_no_selection"
    assert preview["intent_preview_refusal"]["refusal_status"] == "refused_no_selection"


def test_known_selection_for_each_case_builds_preview_only_intent_and_copies_selection_gate() -> (
    None
):
    gate_before = build_preview_paper_order_intent_selection_gate(
        "eth_size_preview_no_intent_no_order"
    )

    for case_id in EXPECTED_CASE_IDS:
        preview = build_preview_controlled_paper_order_intent(case_id)
        gate = build_preview_paper_order_intent_selection_gate(case_id)
        selected = gate["selected_audit_event_preview"]
        intent = preview["paper_order_intent_preview"]
        assert preview["selected_case_id"] == case_id
        assert preview["selection_result"] == {
            key: gate["selection_result"][key] for key in preview["selection_result"]
        }
        assert preview["controlled_intent_result"]["intent_preview_status"] == (
            "built_for_preview_only_no_order_generation"
        )
        assert preview["controlled_intent_result"]["intent_preview_built"] is True
        assert preview["controlled_intent_result"]["intent_preview_executable"] is False
        assert intent["preview_available"] is True
        assert intent["preview_only"] is True
        assert intent["executable"] is False
        assert intent["case_id"] == selected["case_id"]
        assert intent["event_id"] == selected["event_id"]
        _assert_all_execution_false(intent)
        intent["unknown_input_keys"].append("mutated")
    assert (
        build_preview_paper_order_intent_selection_gate("eth_size_preview_no_intent_no_order")
        == gate_before
    )


def test_unknown_empty_and_non_string_selection_fail_closed_without_exception() -> None:
    for value in ["unknown", "", "  ", 123, 1.5, True, [], {}]:
        preview = build_preview_controlled_paper_order_intent(value)  # type: ignore[arg-type]
        assert preview["selected_case_id"] is None
        assert preview["controlled_intent_result"] == {
            "intent_preview_status": "rejected_fail_closed_unknown_or_invalid_selection",
            "intent_preview_built": False,
            "intent_preview_rejected": True,
            "rejection_reason": "unknown_or_invalid_selection",
            "selected_case_id": None,
            "selected_event_id": None,
            "intent_preview_executable": False,
            **EXECUTION_FALSE,
        }
        assert preview["paper_order_intent_preview"]["preview_available"] is False
        assert preview["intent_preview_validation"]["validation_status"] == (
            "rejected_fail_closed_unknown_or_invalid_selection"
        )
        assert preview["intent_preview_refusal"]["refusal_status"] == (
            "refused_unknown_or_invalid_selection"
        )


def test_case_previews_are_expected_and_remain_non_executable() -> None:
    baseline = build_preview_controlled_paper_order_intent("baseline_btc_no_intent_no_order")[
        "paper_order_intent_preview"
    ]
    eth = build_preview_controlled_paper_order_intent("eth_size_preview_no_intent_no_order")[
        "paper_order_intent_preview"
    ]
    sol = build_preview_controlled_paper_order_intent(
        "sol_risk_blocked_preview_no_intent_no_order"
    )["paper_order_intent_preview"]
    unknown = build_preview_controlled_paper_order_intent(
        "unknown_input_keys_reported_no_execution"
    )

    assert baseline["pair"] == "BTC/USDT"
    assert baseline["side"] == "none"
    assert baseline["order_type"] == "none"
    assert baseline["size_value"] == 0.0
    assert baseline["executable"] is False
    assert eth["pair"] == "ETH/USDT"
    assert eth["side"] == "buy_preview_only"
    assert eth["order_type"] == "market_preview_only"
    assert eth["size_value"] == 25.0
    assert eth["preview_available"] is True
    assert eth["executable"] is False
    assert sol["risk_status_preview"] == "blocked_preview_only"
    assert sol["size_value"] == 0.0
    assert unknown["paper_order_intent_preview"]["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert unknown["intent_no_execution_evidence"]["secrets_read_performed"] is False
    assert unknown["intent_no_execution_evidence"]["export_performed"] is False
    assert unknown["controlled_intent_summary"]["live_or_testnet_allowed"] is False


def test_validation_refusal_evidence_summary_boundary_blocked_and_source_boundaries_are_complete() -> (
    None
):
    preview = build_preview_controlled_paper_order_intent("eth_size_preview_no_intent_no_order")

    assert preview["intent_preview_validation"] == {
        "validation_status": "validated_for_preview_only_no_order_generation",
        "validated_for_preview_only": True,
        "executable_intent_allowed": False,
        **EXECUTION_FALSE,
        "requires_manual_confirmation_before_any_future_order": True,
        "requires_kill_switch_before_any_future_order": True,
        "requires_risk_governor_before_any_future_order": True,
    }
    assert preview["intent_preview_refusal"] == {
        "refusal_status": "not_refused_preview_only",
        "refused": False,
        "refusal_reason": None,
        "order_generation_refused": True,
        "submission_refused": True,
        "fills_refused": True,
        "runtime_execution_refused": True,
        "audit_export_refused": True,
        "live_execution_refused": True,
        "testnet_execution_refused": True,
        "account_fetch_refused": True,
        "secrets_read_refused": True,
    }
    evidence = preview["intent_no_execution_evidence"]
    assert evidence["controlled_intent_preview_evaluated"] is True
    assert evidence["paper_order_intent_preview_built"] is True
    for key, value in evidence.items():
        if key not in {"controlled_intent_preview_evaluated", "paper_order_intent_preview_built"}:
            assert value is False
    assert preview["controlled_intent_summary"] == {
        "selection_requested": True,
        "selection_accepted": True,
        "intent_preview_built": True,
        "intent_preview_executable": False,
        "ready_for_static_fill_contract_step": True,
        **EXECUTION_FALSE,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_risk_governor_before_order": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.7",
    }
    for key, expected in REQUIRED_BOUNDARY.items():
        assert preview["boundary_checks"][key] is expected
    assert preview["boundary_checks"]["paper_order_intent_preview_built"] is True
    assert REQUIRED_BLOCKED.issubset(set(preview["blocked_capabilities"]))
    assert REQUIRED_SOURCE_BOUNDARIES.issubset(set(preview["source_boundaries"]))
    assert preview["future_steps"] == [
        "functional_preview_9_7_paper_fill_simulator_contract_static_only",
        "functional_preview_9_8_paper_order_lifecycle_audit",
        "functional_preview_9_9_block_g_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_and_selection_gate_and_has_no_forbidden_calls() -> None:
    tree = ast.parse(_source(HELPER))
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
        "ui.pyside_app.preview_paper_order_intent_selection_gate",
    ]
    assert all(
        imported == "ui.pyside_app.preview_paper_order_intent_selection_gate"
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
