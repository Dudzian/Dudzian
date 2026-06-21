from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_decision_engine_dry_run_contract import REQUIRED_BOUNDARIES
from ui.pyside_app.preview_decision_engine_dry_run_read_model import (
    build_preview_decision_engine_dry_run_read_model_snapshot,
)
from ui.pyside_app.preview_decision_engine_dry_run_static_fixture import (
    build_preview_decision_engine_dry_run_static_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_decision_engine_dry_run_static_fixture.py"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
ALLOWED_CALL = (
    "paperRuntimeActionDispatchBridge.previewSelectAction("
    '"paper_runtime_snapshot_refresh_requested")'
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "fixture_kind",
    "block",
    "step",
    "static_fixture_status",
    "dry_run_mode",
    "static_fixture_decision",
    "ready_for_block_f_3",
    "next_step",
    "next_step_title",
    "contract_reference",
    "read_model_reference",
    "fixture_cases",
    "fixture_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
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


def test_static_fixture_is_plain_json_serializable_dict() -> None:
    fixture = build_preview_decision_engine_dry_run_static_fixture()

    assert type(fixture) is dict
    _assert_plain(fixture)
    assert json.loads(json.dumps(fixture, sort_keys=True)) == fixture
    assert set(fixture) == EXPECTED_TOP_LEVEL_KEYS


def test_static_fixture_references_contract_and_read_model() -> None:
    fixture = build_preview_decision_engine_dry_run_static_fixture()

    assert fixture["schema_version"] == "preview_decision_engine_dry_run_static_fixture.v1"
    assert (
        fixture["fixture_kind"]
        == "functional_preview_block_f_decision_engine_dry_run_static_fixture"
    )
    assert fixture["block"] == "F"
    assert fixture["step"] == "8.2"
    assert fixture["static_fixture_status"] == "static_fixture_ready_no_engine_execution"
    assert fixture["dry_run_mode"] == "local_paper_dry_run"
    assert fixture["static_fixture_decision"] == "BUILD_STATIC_FIXTURE_ONLY_NO_ENGINE_EXECUTION"
    assert fixture["ready_for_block_f_3"] is True
    assert fixture["next_step"] == "FUNCTIONAL-PREVIEW-8.3"
    assert fixture["next_step_title"] == "DECISION ENGINE DRY-RUN AUDIT ENVELOPE"
    assert fixture["contract_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_contract.v1",
        "contract_kind": "functional_preview_block_f_decision_engine_dry_run_contract",
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION",
    }
    assert fixture["read_model_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_read_model.v1",
        "read_model_kind": "functional_preview_block_f_decision_engine_dry_run_read_model",
        "read_model_status": "read_model_snapshot_ready_no_engine_execution",
        "read_model_decision": "BUILD_READ_MODEL_ONLY_NO_ENGINE_EXECUTION",
    }


def test_static_fixture_has_required_cases_and_each_uses_read_model_snapshot() -> None:
    fixture = build_preview_decision_engine_dry_run_static_fixture()
    cases = fixture["fixture_cases"]

    assert len(cases) >= 3
    by_id = {case["case_id"]: case for case in cases}
    assert {
        "baseline_btc_no_order",
        "scanner_eth_candidate_no_order",
        "risk_blocked_sol_no_order",
    } <= set(by_id)
    assert by_id["baseline_btc_no_order"]["input_snapshot"]["operator_selected_pair"] == "BTC/USDT"
    assert (
        by_id["scanner_eth_candidate_no_order"]["input_snapshot"]["operator_selected_pair"]
        == "ETH/USDT"
    )
    assert (
        by_id["risk_blocked_sol_no_order"]["input_snapshot"]["operator_selected_pair"] == "SOL/USDT"
    )
    assert (
        by_id["risk_blocked_sol_no_order"]["input_snapshot"]["risk_preview_snapshot"]["risk_status"]
        == "blocked_not_evaluated_contract_only"
    )

    for case in cases:
        assert case[
            "read_model_snapshot"
        ] == build_preview_decision_engine_dry_run_read_model_snapshot(case["input_snapshot"])


def test_each_case_is_no_order_no_execution_no_risk_engine_no_export() -> None:
    fixture = build_preview_decision_engine_dry_run_static_fixture()

    for case in fixture["fixture_cases"]:
        assert case["expected_decision_action"] == "NO_ORDER_DRY_RUN_PREVIEW"
        assert case["expected_decision_status"] == "not_executed"
        assert case["expected_order_generation_allowed"] is False
        assert case["expected_order_submission_allowed"] is False
        assert case["expected_execution_performed"] is False
        assert case["expected_risk_engine_execution_performed"] is False
        assert case["expected_audit_export_allowed"] is False
        assert case["case_status"] == "ready_no_engine_execution_no_orders"


def test_fixture_summary_boundaries_blocked_capabilities_and_source_boundaries_are_safe() -> None:
    fixture = build_preview_decision_engine_dry_run_static_fixture()
    expected_boundaries = dict(REQUIRED_BOUNDARIES)
    expected_boundaries["model_inference_execution_allowed"] = False

    assert fixture["fixture_summary"] == {
        "fixture_case_count": len(fixture["fixture_cases"]),
        "all_cases_no_order": True,
        "all_cases_no_execution": True,
        "all_cases_no_risk_engine_execution": True,
        "all_cases_no_export": True,
        "all_cases_json_serializable": True,
        "engine_execution_performed": False,
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "live_mode_allowed": False,
        "testnet_mode_allowed": False,
    }
    assert fixture["boundary_checks"] == expected_boundaries
    assert {
        "decision engine execution",
        "order generation",
        "order submission",
        "live trading",
        "testnet/sandbox trading",
        "secrets read/export",
        "cloud/external export",
        "model inference",
        "risk engine evaluation",
        "EXE packaging",
    } <= set(fixture["blocked_capabilities"])
    assert {
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
    } <= set(fixture["source_boundaries"])
    assert fixture["future_steps"] == [
        "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
        "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
        "functional_preview_8_5_block_f_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_contract_read_model_and_has_no_forbidden_calls() -> None:
    tree = ast.parse(_source(HELPER))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            calls.add(getattr(node.func, "attr", None) or getattr(node.func, "id", ""))

    assert imports <= {
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_decision_engine_dry_run_contract",
        "ui.pyside_app.preview_decision_engine_dry_run_read_model",
    }
    forbidden_import_roots = {
        "PySide6",
        "pathlib",
        "requests",
        "subprocess",
        "ui.pyside_app.app",
        "ui.pyside_app.qml_bridge",
    }
    forbidden_import_fragments = (
        "TradingController",
        "DecisionEnvelope",
        "strategy",
        "scoring",
        "recommendation",
        "live_adapter",
        "testnet_adapter",
        "account_adapter",
        "secrets",
        "network",
    )
    assert not any(name.split(".")[0] in forbidden_import_roots for name in imports)
    assert not any(fragment in name for name in imports for fragment in forbidden_import_fragments)
    assert (
        not {
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
        & calls
    )


def test_qml_bat_app_and_preview_bridge_calls_are_not_changed_or_expanded() -> None:
    joined_qml = "\n".join((_source(MAIN_WINDOW), _source(OPERATOR_DASHBOARD)))
    changed = {line[3:] for line in _git_name_status() if line.startswith(("M  ", "A  ", "D  "))}
    bat_paths = {str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.glob("*.bat")}

    assert joined_qml.count(ALLOWED_CALL) == 1
    assert joined_qml.count("previewSelectAction(") == 1
    assert "previewSelectSourceControl(" not in joined_qml
    assert "resetPreviewSelection(" not in joined_qml
    assert (
        not {"ui/pyside_app/qml/MainWindow.qml", "ui/pyside_app/qml/views/OperatorDashboard.qml"}
        & changed
    )
    assert "ui/pyside_app/app.py" not in changed
    assert not bat_paths & changed


def _git_name_status() -> list[str]:
    import subprocess

    return subprocess.check_output(
        ["git", "diff", "--cached", "--name-status"],
        text=True,
    ).splitlines()
