"""Source-only tests for BLOK G paper order lifecycle static audit."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_order_lifecycle_audit import (
    LIFECYCLE_AUDIT_DECISION,
    LIFECYCLE_AUDIT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND,
    PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_9,
    build_preview_paper_order_lifecycle_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_order_lifecycle_audit.py"
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
    "lifecycle_audit_kind",
    "block",
    "step",
    "lifecycle_audit_status",
    "lifecycle_audit_decision",
    "ready_for_block_g_9",
    "next_step",
    "next_step_title",
    "fill_simulator_contract_reference",
    "selected_case_id",
    "controlled_intent_result",
    "paper_order_intent_preview",
    "fill_simulator_contract",
    "lifecycle_audit",
    "lifecycle_transition_preview",
    "lifecycle_mutation_refusal",
    "lifecycle_no_execution_evidence",
    "lifecycle_audit_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
REQUIRED_BOUNDARY = {
    "local_only": True,
    "paper_only": True,
    "paper_order_lifecycle_audit_static_only": True,
    "paper_fill_simulator_contract_static_only": True,
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
    "paper_fill_simulation_contract_allowed_now": True,
    "paper_fill_simulation_allowed_now": False,
    "paper_fill_simulated": False,
    "fill_event_generation_allowed_now": False,
    "fill_event_generated": False,
    "order_lifecycle_audit_allowed_now": True,
    "order_lifecycle_mutation_allowed_now": False,
    "order_lifecycle_mutated": False,
    "lifecycle_transition_allowed_now": False,
    "lifecycle_transition_performed": False,
    "paper_runtime_execution_allowed_now": False,
    "paper_runtime_execution_performed": False,
    "risk_governor_execution_allowed_now": False,
    "risk_governor_execution_performed": False,
    "market_data_fetch_allowed": False,
    "market_data_fetch_performed": False,
    "selection_gate_evaluated": True,
    "selection_acceptance_can_enable_intent_generation": False,
    "audit_event_generation_allowed": True,
    "audit_export_allowed": False,
    "audit_export_performed": False,
    "manual_operator_confirmation_required_before_order": True,
    "kill_switch_required_before_order": True,
    "future_risk_governor_required_before_order": True,
    "future_market_data_required_before_fill": True,
    "future_order_lifecycle_contract_required_before_mutation": True,
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
    "paper order lifecycle mutation",
    "paper order lifecycle transition execution",
    "paper order lifecycle runtime dispatch",
    "paper fill simulation execution",
    "paper fill event generation",
    "market data fetch for fills",
    "executable paper order intent generation",
    "preview intent conversion to order",
    "preview intent submission",
    "paper order generation now",
    "paper order submission now",
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


def test_lifecycle_audit_returns_json_serializable_plain_dict() -> None:
    audit = build_preview_paper_order_lifecycle_audit()
    assert isinstance(audit, dict)
    assert set(audit) == EXPECTED_TOP_LEVEL
    _assert_plain(audit)
    assert json.loads(json.dumps(audit, sort_keys=True)) == audit


def test_status_decision_next_step_and_fill_contract_reference_are_fixed() -> None:
    audit = build_preview_paper_order_lifecycle_audit()
    assert audit["schema_version"] == PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION
    assert audit["lifecycle_audit_kind"] == PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND
    assert audit["block"] == "G"
    assert audit["step"] == "9.8"
    assert audit["lifecycle_audit_status"] == LIFECYCLE_AUDIT_STATUS
    assert audit["lifecycle_audit_decision"] == LIFECYCLE_AUDIT_DECISION
    assert audit["ready_for_block_g_9"] is READY_FOR_BLOCK_G_9 is True
    assert audit["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.9"
    assert audit["next_step_title"] == NEXT_STEP_TITLE == "BLOCK G CLOSURE AUDIT"
    assert audit["fill_simulator_contract_reference"] == {
        "schema_version": "preview_paper_fill_simulator_contract.v1",
        "fill_simulator_contract_kind": "functional_preview_block_g_paper_fill_simulator_contract",
        "fill_simulator_contract_status": "paper_fill_simulator_contract_static_ready_no_fill_simulation",
        "fill_simulator_contract_decision": "BUILD_FILL_SIMULATOR_CONTRACT_STATIC_ONLY_NO_FILL_SIMULATION",
        "ready_for_block_g_8": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.8",
    }


def test_no_selection_default_does_not_activate_lifecycle_audit_and_is_safe() -> None:
    audit = build_preview_paper_order_lifecycle_audit()
    assert audit["selected_case_id"] is None
    assert audit["lifecycle_audit"] == {
        "audit_available": False,
        "audit_static_only": True,
        "audit_executable": False,
        "case_id": None,
        "event_id": None,
        "pair": None,
        "side": None,
        "order_type": None,
        "size_value": 0.0,
        "size_unit": "preview_only",
        "initial_lifecycle_state": "not_available_no_selection_or_rejected_selection",
        "terminal_lifecycle_state": "not_available_no_selection_or_rejected_selection",
        "lifecycle_policy": "not_available_no_selection_or_rejected_selection",
        "lifecycle_mutation_allowed_now": False,
        "lifecycle_transition_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "order_generation_allowed_now": False,
        "submission_allowed_now": False,
        "runtime_execution_allowed": False,
        "audit_export_allowed": False,
    }
    assert audit["lifecycle_transition_preview"]["transition_preview_available"] is False
    assert audit["lifecycle_transition_preview"]["transition_rejected"] is False
    assert audit["lifecycle_mutation_refusal"]["refusal_status"] == (
        "lifecycle_mutation_refused_no_selection"
    )
    assert audit["lifecycle_no_execution_evidence"]["static_lifecycle_audit_built"] is False


def test_known_selection_for_each_case_builds_static_non_executable_lifecycle_audit() -> None:
    for case_id in EXPECTED_CASE_IDS:
        audit = build_preview_paper_order_lifecycle_audit(case_id)
        lifecycle = audit["lifecycle_audit"]
        transition = audit["lifecycle_transition_preview"]
        fill_contract = audit["fill_simulator_contract"]
        assert audit["selected_case_id"] == case_id
        assert lifecycle["audit_available"] is True
        assert lifecycle["audit_static_only"] is True
        assert lifecycle["audit_executable"] is False
        assert lifecycle["case_id"] == fill_contract["case_id"] == case_id
        assert lifecycle["lifecycle_policy"] == "static_lifecycle_audit_only_no_mutation"
        assert lifecycle["initial_lifecycle_state"] == "preview_intent_static_only"
        assert lifecycle["terminal_lifecycle_state"] == "awaiting_future_order_lifecycle_contract"
        assert transition["transition_preview_available"] is True
        assert transition["transition_static_only"] is True
        assert transition["transition_executable"] is False
        assert transition["transition_rejected"] is False
        assert audit["lifecycle_no_execution_evidence"]["static_lifecycle_audit_built"] is True


def test_unknown_empty_and_non_string_selection_fail_closed_without_exception() -> None:
    for value in ["unknown", "", "  ", 123, 1.5, True, [], {}]:
        audit = build_preview_paper_order_lifecycle_audit(value)  # type: ignore[arg-type]
        assert audit["selected_case_id"] is None
        assert audit["controlled_intent_result"]["intent_preview_rejected"] is True
        assert audit["lifecycle_audit"]["audit_available"] is False
        assert audit["lifecycle_transition_preview"]["transition_rejected"] is True
        assert audit["lifecycle_transition_preview"]["rejection_reason"] == (
            "unknown_or_invalid_selection"
        )
        assert audit["lifecycle_mutation_refusal"]["refusal_status"] == (
            "lifecycle_mutation_refused_unknown_or_invalid_selection"
        )
        assert audit["lifecycle_no_execution_evidence"]["static_lifecycle_audit_built"] is False


def test_case_lifecycle_audits_preserve_preview_context_without_execution() -> None:
    baseline = build_preview_paper_order_lifecycle_audit("baseline_btc_no_intent_no_order")
    eth = build_preview_paper_order_lifecycle_audit("eth_size_preview_no_intent_no_order")
    sol = build_preview_paper_order_lifecycle_audit("sol_risk_blocked_preview_no_intent_no_order")
    unknown = build_preview_paper_order_lifecycle_audit("unknown_input_keys_reported_no_execution")

    assert baseline["lifecycle_audit"]["pair"] == "BTC/USDT"
    assert baseline["lifecycle_audit"]["size_value"] == 0.0
    assert baseline["lifecycle_audit"]["lifecycle_mutation_allowed_now"] is False
    assert eth["lifecycle_audit"]["pair"] == "ETH/USDT"
    assert eth["lifecycle_audit"]["size_value"] == 25.0
    assert eth["lifecycle_no_execution_evidence"]["lifecycle_mutation_performed"] is False
    assert eth["lifecycle_no_execution_evidence"]["fill_simulation_performed"] is False
    assert eth["lifecycle_no_execution_evidence"]["paper_order_generated"] is False
    assert eth["lifecycle_no_execution_evidence"]["paper_order_submitted"] is False
    assert eth["lifecycle_no_execution_evidence"]["paper_runtime_execution_performed"] is False
    assert sol["paper_order_intent_preview"]["risk_status_preview"] == "blocked_preview_only"
    assert sol["lifecycle_no_execution_evidence"]["risk_governor_execution_performed"] is False
    assert sol["lifecycle_audit"]["lifecycle_mutation_allowed_now"] is False
    assert unknown["paper_order_intent_preview"]["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert unknown["lifecycle_no_execution_evidence"]["secrets_read_performed"] is False
    assert unknown["lifecycle_audit_summary"]["audit_export_allowed"] is False
    assert unknown["lifecycle_audit_summary"]["live_or_testnet_allowed"] is False


def test_refusal_evidence_summary_boundary_blocked_and_source_boundaries_are_complete() -> None:
    audit = build_preview_paper_order_lifecycle_audit("eth_size_preview_no_intent_no_order")
    refusal = audit["lifecycle_mutation_refusal"]
    for key in (
        "lifecycle_mutation_refused",
        "lifecycle_transition_refused",
        "fill_simulation_refused",
        "fill_event_generation_refused",
        "order_generation_refused",
        "submission_refused",
        "runtime_execution_refused",
        "market_data_fetch_refused",
        "audit_export_refused",
        "live_execution_refused",
        "testnet_execution_refused",
        "account_fetch_refused",
        "secrets_read_refused",
    ):
        assert refusal[key] is True
    evidence = audit["lifecycle_no_execution_evidence"]
    assert evidence["lifecycle_audit_evaluated"] is True
    assert evidence["static_lifecycle_audit_built"] is True
    for key, value in evidence.items():
        if key not in {"lifecycle_audit_evaluated", "static_lifecycle_audit_built"}:
            assert value is False
    assert audit["lifecycle_audit_summary"] == {
        "selection_requested": True,
        "controlled_intent_preview_built": True,
        "fill_contract_available": True,
        "lifecycle_audit_available": True,
        "lifecycle_audit_executable": False,
        "lifecycle_mutation_allowed_now": False,
        "lifecycle_transition_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
        "market_data_fetch_allowed": False,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_order_lifecycle_contract_before_mutation": True,
        "requires_future_fill_event_contract_before_fill": True,
        "requires_future_market_data_before_fill": True,
        "requires_future_risk_governor_before_order": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "ready_for_block_g_closure_audit": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.9",
    }
    for key, expected in REQUIRED_BOUNDARY.items():
        assert audit["boundary_checks"][key] is expected
    assert audit["boundary_checks"]["paper_order_intent_preview_built"] is True
    assert REQUIRED_BLOCKED.issubset(set(audit["blocked_capabilities"]))
    assert REQUIRED_SOURCE_BOUNDARIES.issubset(set(audit["source_boundaries"]))
    assert audit["future_steps"] == ["functional_preview_9_9_block_g_closure_audit"]


def test_selected_lifecycle_audit_copy_is_safe() -> None:
    audit = build_preview_paper_order_lifecycle_audit("unknown_input_keys_reported_no_execution")
    audit["paper_order_intent_preview"]["unknown_input_keys"].append("mutated")
    fresh = build_preview_paper_order_lifecycle_audit("unknown_input_keys_reported_no_execution")
    assert fresh["paper_order_intent_preview"]["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]


def test_helper_imports_only_safe_stdlib_and_fill_contract_and_has_no_forbidden_calls() -> None:
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
        "ui.pyside_app.preview_paper_fill_simulator_contract",
    ]
    assert all(
        imported == "ui.pyside_app.preview_paper_fill_simulator_contract"
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
        "lifecycle_transition",
        "lifecycle_mutation",
    ]:
        assert forbidden not in qml
