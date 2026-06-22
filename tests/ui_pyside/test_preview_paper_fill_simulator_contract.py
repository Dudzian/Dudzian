"""Source-only tests for BLOK G paper fill simulator static contract."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_paper_fill_simulator_contract import (
    FILL_SIMULATOR_CONTRACT_DECISION,
    FILL_SIMULATOR_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_8,
    build_preview_paper_fill_simulator_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_paper_fill_simulator_contract.py"
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
    "fill_simulator_contract_kind",
    "block",
    "step",
    "fill_simulator_contract_status",
    "fill_simulator_contract_decision",
    "ready_for_block_g_8",
    "next_step",
    "next_step_title",
    "controlled_intent_reference",
    "selected_case_id",
    "controlled_intent_result",
    "paper_order_intent_preview",
    "fill_simulator_contract",
    "fill_simulation_request_preview",
    "fill_simulation_refusal",
    "fill_simulation_no_execution_evidence",
    "fill_contract_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
REQUIRED_BOUNDARY = {
    "local_only": True,
    "paper_only": True,
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
    "order_lifecycle_mutation_allowed_now": False,
    "order_lifecycle_mutated": False,
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
    "paper fill simulation execution",
    "paper fill event generation",
    "paper order lifecycle mutation",
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


def test_contract_returns_json_serializable_plain_dict() -> None:
    contract = build_preview_paper_fill_simulator_contract()

    assert isinstance(contract, dict)
    assert set(contract) == EXPECTED_TOP_LEVEL
    _assert_plain(contract)
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract


def test_status_decision_next_step_and_controlled_intent_reference_are_fixed() -> None:
    contract = build_preview_paper_fill_simulator_contract()

    assert contract["schema_version"] == PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION
    assert contract["fill_simulator_contract_kind"] == PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND
    assert contract["block"] == "G"
    assert contract["step"] == "9.7"
    assert contract["fill_simulator_contract_status"] == FILL_SIMULATOR_CONTRACT_STATUS
    assert contract["fill_simulator_contract_decision"] == FILL_SIMULATOR_CONTRACT_DECISION
    assert contract["ready_for_block_g_8"] is READY_FOR_BLOCK_G_8 is True
    assert contract["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-9.8"
    assert contract["next_step_title"] == NEXT_STEP_TITLE
    assert contract["controlled_intent_reference"] == {
        "schema_version": "preview_controlled_paper_order_intent.v1",
        "controlled_intent_kind": "functional_preview_block_g_controlled_paper_order_intent",
        "controlled_intent_status": "controlled_paper_order_intent_preview_ready_no_order_generation",
        "controlled_intent_decision": "BUILD_CONTROLLED_INTENT_PREVIEW_ONLY_NO_ORDER_GENERATION",
        "ready_for_block_g_7": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.7",
    }


def test_no_selection_default_does_not_activate_fill_contract_and_is_safe() -> None:
    contract = build_preview_paper_fill_simulator_contract()

    assert contract["selected_case_id"] is None
    assert contract["fill_simulator_contract"] == {
        "contract_available": False,
        "contract_static_only": True,
        "contract_executable": False,
        "case_id": None,
        "event_id": None,
        "pair": None,
        "side": None,
        "order_type": None,
        "size_value": 0.0,
        "size_unit": "preview_only",
        "fill_policy": "not_available_no_selection_or_rejected_selection",
        "fill_price_source": "none",
        "market_data_required_for_future_execution": True,
        "risk_governor_required_for_future_execution": True,
        "manual_confirmation_required_for_future_execution": True,
        "kill_switch_required_for_future_execution": True,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_lifecycle_mutation_allowed_now": False,
        "runtime_execution_allowed": False,
    }
    assert contract["fill_simulation_request_preview"]["request_available"] is False
    assert contract["fill_simulation_request_preview"]["request_rejected"] is False
    assert (
        contract["fill_simulation_refusal"]["refusal_status"]
        == "fill_simulation_refused_no_selection"
    )
    assert contract["fill_simulation_no_execution_evidence"]["static_contract_built"] is False


def test_known_selection_for_each_case_builds_static_non_executable_contract() -> None:
    for case_id in EXPECTED_CASE_IDS:
        contract = build_preview_paper_fill_simulator_contract(case_id)
        fill_contract = contract["fill_simulator_contract"]
        request = contract["fill_simulation_request_preview"]
        preview = contract["paper_order_intent_preview"]
        assert contract["selected_case_id"] == case_id
        assert fill_contract["contract_available"] is True
        assert fill_contract["contract_static_only"] is True
        assert fill_contract["contract_executable"] is False
        assert fill_contract["case_id"] == preview["case_id"] == case_id
        assert fill_contract["fill_policy"] == "static_preview_contract_only_no_fill_simulation"
        assert fill_contract["fill_price_source"] == "not_evaluated_no_market_data"
        assert request["request_available"] is True
        assert request["request_static_only"] is True
        assert request["request_executable"] is False
        assert request["request_rejected"] is False
        assert contract["fill_simulation_no_execution_evidence"]["static_contract_built"] is True


def test_unknown_empty_and_non_string_selection_fail_closed_without_exception() -> None:
    for value in ["unknown", "", "  ", 123, 1.5, True, [], {}]:
        contract = build_preview_paper_fill_simulator_contract(value)  # type: ignore[arg-type]
        assert contract["selected_case_id"] is None
        assert contract["controlled_intent_result"]["intent_preview_rejected"] is True
        assert contract["fill_simulator_contract"]["contract_available"] is False
        assert contract["fill_simulation_request_preview"]["request_rejected"] is True
        assert contract["fill_simulation_request_preview"]["rejection_reason"] == (
            "unknown_or_invalid_selection"
        )
        assert contract["fill_simulation_refusal"]["refusal_status"] == (
            "fill_simulation_refused_unknown_or_invalid_selection"
        )
        assert contract["fill_simulation_no_execution_evidence"]["static_contract_built"] is False


def test_case_contracts_preserve_preview_context_without_execution() -> None:
    baseline = build_preview_paper_fill_simulator_contract("baseline_btc_no_intent_no_order")
    eth = build_preview_paper_fill_simulator_contract("eth_size_preview_no_intent_no_order")
    sol = build_preview_paper_fill_simulator_contract("sol_risk_blocked_preview_no_intent_no_order")
    unknown = build_preview_paper_fill_simulator_contract(
        "unknown_input_keys_reported_no_execution"
    )

    assert baseline["fill_simulator_contract"]["pair"] == "BTC/USDT"
    assert baseline["fill_simulator_contract"]["size_value"] == 0.0
    assert eth["fill_simulator_contract"]["pair"] == "ETH/USDT"
    assert eth["fill_simulator_contract"]["size_value"] == 25.0
    assert eth["fill_simulation_no_execution_evidence"]["fill_simulation_performed"] is False
    assert eth["fill_simulation_no_execution_evidence"]["order_lifecycle_mutated"] is False
    assert (
        eth["fill_simulation_no_execution_evidence"]["paper_runtime_execution_performed"] is False
    )
    assert sol["paper_order_intent_preview"]["risk_status_preview"] == "blocked_preview_only"
    assert (
        sol["fill_simulation_no_execution_evidence"]["risk_governor_execution_performed"] is False
    )
    assert unknown["paper_order_intent_preview"]["unknown_input_keys"] == [
        "live_credentials_reference",
        "unsafe_submit_order_request",
    ]
    assert unknown["fill_simulation_no_execution_evidence"]["secrets_read_performed"] is False
    assert unknown["fill_contract_summary"]["audit_export_allowed"] is False
    assert unknown["fill_contract_summary"]["live_or_testnet_allowed"] is False


def test_refusal_evidence_summary_boundary_blocked_and_source_boundaries_are_complete() -> None:
    contract = build_preview_paper_fill_simulator_contract("eth_size_preview_no_intent_no_order")

    refusal = contract["fill_simulation_refusal"]
    for key in (
        "fill_simulation_refused",
        "fill_event_generation_refused",
        "order_lifecycle_mutation_refused",
        "order_generation_refused",
        "submission_refused",
        "runtime_execution_refused",
        "audit_export_refused",
        "live_execution_refused",
        "testnet_execution_refused",
        "market_data_fetch_refused",
        "account_fetch_refused",
        "secrets_read_refused",
    ):
        assert refusal[key] is True
    evidence = contract["fill_simulation_no_execution_evidence"]
    assert evidence["fill_simulator_contract_evaluated"] is True
    assert evidence["static_contract_built"] is True
    for key, value in evidence.items():
        if key not in {"fill_simulator_contract_evaluated", "static_contract_built"}:
            assert value is False
    assert contract["fill_contract_summary"] == {
        "selection_requested": True,
        "controlled_intent_preview_built": True,
        "fill_contract_available": True,
        "fill_contract_executable": False,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_lifecycle_mutation_allowed_now": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
        "market_data_fetch_allowed": False,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_market_data_before_fill": True,
        "requires_future_risk_governor_before_fill": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "ready_for_order_lifecycle_audit_step": True,
        "next_step": "FUNCTIONAL-PREVIEW-9.8",
    }
    for key, expected in REQUIRED_BOUNDARY.items():
        assert contract["boundary_checks"][key] is expected
    assert contract["boundary_checks"]["paper_order_intent_preview_built"] is True
    assert REQUIRED_BLOCKED.issubset(set(contract["blocked_capabilities"]))
    assert REQUIRED_SOURCE_BOUNDARIES.issubset(set(contract["source_boundaries"]))
    assert contract["future_steps"] == [
        "functional_preview_9_8_paper_order_lifecycle_audit",
        "functional_preview_9_9_block_g_closure_audit",
    ]


def test_selected_contract_copy_is_safe() -> None:
    contract = build_preview_paper_fill_simulator_contract("eth_size_preview_no_intent_no_order")
    contract["paper_order_intent_preview"]["unknown_input_keys"].append("mutated")
    fresh = build_preview_paper_fill_simulator_contract("eth_size_preview_no_intent_no_order")
    assert fresh["paper_order_intent_preview"]["unknown_input_keys"] == []


def test_helper_imports_only_safe_stdlib_and_controlled_intent_and_has_no_forbidden_calls() -> None:
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
        "ui.pyside_app.preview_controlled_paper_order_intent",
    ]
    assert all(
        imported == "ui.pyside_app.preview_controlled_paper_order_intent"
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
