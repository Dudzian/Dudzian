from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_decision_engine_dry_run_contract import (
    build_preview_decision_engine_dry_run_contract,
)

HELPER_PATH = Path("ui/pyside_app/preview_decision_engine_dry_run_contract.py")
QML_PATHS = (
    Path("ui/pyside_app/qml/MainWindow.qml"),
    Path("ui/pyside_app/qml/views/OperatorDashboard.qml"),
)
BAT_PATHS = tuple(Path(".").glob("*.bat"))
APP_PATH = Path("ui/pyside_app/app.py")

EXPECTED_INPUTS = [
    "local_preview_state_snapshot",
    "paper_runtime_snapshot",
    "scanner_candidate_snapshot",
    "risk_preview_snapshot",
    "portfolio_preview_snapshot",
    "operator_selected_pair",
    "operator_selected_candidate",
    "dry_run_context_id",
]
EXPECTED_OUTPUTS = [
    "dry_run_decision_preview",
    "decision_reason_summary",
    "input_snapshot_echo",
    "risk_check_preview",
    "confidence_preview",
    "no_order_decision_preview",
    "audit_event_preview",
    "blocked_reason_preview",
]
EXPECTED_FUTURE_STEPS = [
    "functional_preview_8_1_decision_engine_dry_run_read_model_snapshot",
    "functional_preview_8_2_decision_engine_dry_run_static_fixture",
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
    "functional_preview_8_5_block_f_closure_audit",
]
EXPECTED_BOUNDARIES = {
    "local_only": True,
    "paper_only": True,
    "dry_run_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "trading_controller_allowed": False,
    "decision_envelope_allowed": False,
    "strategy_execution_allowed": False,
    "ai_scoring_execution_allowed": False,
    "runtime_loop_allowed": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "order_generation_allowed": False,
    "order_submission_allowed": False,
    "fills_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "new_qml_method_calls_allowed": False,
    "exe_packaging_in_scope": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}
EXPECTED_BLOCKED_CAPABILITIES = {
    "live trading",
    "testnet/sandbox trading",
    "decision engine execution",
    "TradingController integration",
    "DecisionEnvelope integration",
    "strategy execution",
    "AI/scoring execution",
    "runtime loop execution",
    "lifecycle command execution",
    "dynamic action dispatch",
    "order generation",
    "order submission",
    "fills",
    "account/balance fetch",
    "secrets read/export",
    "cloud/external export",
    "new QML action calls",
    "EXE packaging",
}
EXPECTED_SOURCE_BOUNDARIES = {
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no filesystem I/O",
    "no network I/O",
    "no secrets access",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
}


def test_contract_is_plain_json_serializable_dict() -> None:
    contract = build_preview_decision_engine_dry_run_contract()

    assert type(contract) is dict
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract
    assert set(contract) == {
        "schema_version",
        "contract_kind",
        "block",
        "block_status",
        "dry_run_mode",
        "contract_decision",
        "ready_for_block_f_1",
        "next_step",
        "next_step_title",
        "summary",
        "allowed_dry_run_inputs",
        "allowed_dry_run_outputs",
        "required_boundaries",
        "blocked_capabilities",
        "source_boundaries",
        "future_steps",
        "status",
    }


def test_status_decision_and_next_step_are_fixed() -> None:
    contract = build_preview_decision_engine_dry_run_contract()

    assert contract["schema_version"] == "preview_decision_engine_dry_run_contract.v1"
    assert (
        contract["contract_kind"] == "functional_preview_block_f_decision_engine_dry_run_contract"
    )
    assert contract["block"] == "F"
    assert contract["block_status"] == "decision_engine_dry_run_contract_ready_no_execution"
    assert contract["dry_run_mode"] == "local_paper_dry_run"
    assert contract["contract_decision"] == "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION"
    assert contract["ready_for_block_f_1"] is True
    assert contract["next_step"] == "FUNCTIONAL-PREVIEW-8.1"
    assert contract["next_step_title"] == "DECISION ENGINE DRY-RUN READ MODEL SNAPSHOT"
    assert contract["status"] == "contract_ready_no_execution"


def test_allowed_inputs_outputs_are_exact_contract_only() -> None:
    contract = build_preview_decision_engine_dry_run_contract()

    assert contract["allowed_dry_run_inputs"] == EXPECTED_INPUTS
    assert contract["allowed_dry_run_outputs"] == EXPECTED_OUTPUTS
    forbidden_runtime_fields = {
        "order",
        "orders",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "live_adapter",
        "testnet_adapter",
        "TradingController",
        "DecisionEnvelope",
    }
    serialized = json.dumps(
        {
            "inputs": contract["allowed_dry_run_inputs"],
            "outputs": contract["allowed_dry_run_outputs"],
        }
    )
    assert forbidden_runtime_fields.isdisjoint(serialized.split())


def test_required_boundaries_are_exact_true_false_contract() -> None:
    contract = build_preview_decision_engine_dry_run_contract()

    assert contract["required_boundaries"] == EXPECTED_BOUNDARIES


def test_blocked_capabilities_source_boundaries_and_future_steps_are_explicit() -> None:
    contract = build_preview_decision_engine_dry_run_contract()

    assert set(contract["blocked_capabilities"]) == EXPECTED_BLOCKED_CAPABILITIES
    assert set(contract["source_boundaries"]) == EXPECTED_SOURCE_BOUNDARIES
    assert contract["future_steps"] == EXPECTED_FUTURE_STEPS


def test_helper_source_imports_only_safe_stdlib_modules() -> None:
    tree = ast.parse(HELPER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert imported <= {"__future__", "copy", "typing"}


def test_helper_source_has_no_forbidden_calls_or_runtime_tokens() -> None:
    source = HELPER_PATH.read_text(encoding="utf-8")

    forbidden_fragments = (
        "open(",
        ".read_text(",
        ".write_text(",
        "requests",
        "subprocess",
        "QQmlApplicationEngine(",
        "TradingController(",
        "DecisionEnvelope(",
        "start_runtime(",
        "start_loop(",
        "submit_order(",
        "place_order(",
        "create_order(",
        "send_order(",
        "fill_order(",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source


def test_qml_app_and_bat_files_are_not_part_of_contract_patch() -> None:
    changed = {line[3:] for line in _git_name_status() if line.startswith(("M  ", "A  ", "D  "))}

    assert not {str(path) for path in QML_PATHS} & changed
    assert str(APP_PATH) not in changed
    assert not {str(path) for path in BAT_PATHS} & changed


def test_no_new_qml_method_calls_were_added_to_operator_dashboard() -> None:
    source = Path("ui/pyside_app/qml/views/OperatorDashboard.qml").read_text(encoding="utf-8")

    expected_call = (
        "paperRuntimeActionDispatchBridge.previewSelectAction("
        '"paper_runtime_snapshot_refresh_requested")'
    )
    assert source.count("previewSelectAction(") == 1
    assert expected_call in source
    assert "previewSelectSourceControl(" not in source
    assert "resetPreviewSelection(" not in source


def _git_name_status() -> list[str]:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        check=True,
        capture_output=True,
        text=True,
    )
    unstaged = subprocess.run(
        ["git", "diff", "--name-status"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in (result.stdout + unstaged.stdout).splitlines() if line]
