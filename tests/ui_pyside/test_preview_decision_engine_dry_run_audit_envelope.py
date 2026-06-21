from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_decision_engine_dry_run_contract import REQUIRED_BOUNDARIES
from ui.pyside_app.preview_decision_engine_dry_run_audit_envelope import (
    build_preview_decision_engine_dry_run_audit_envelope,
)
from ui.pyside_app.preview_decision_engine_dry_run_static_fixture import (
    build_preview_decision_engine_dry_run_static_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_decision_engine_dry_run_audit_envelope.py"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
ALLOWED_CALL = (
    "paperRuntimeActionDispatchBridge.previewSelectAction("
    '"paper_runtime_snapshot_refresh_requested")'
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "envelope_kind",
    "block",
    "step",
    "audit_envelope_status",
    "dry_run_mode",
    "audit_envelope_decision",
    "ready_for_block_f_4",
    "next_step",
    "next_step_title",
    "contract_reference",
    "read_model_reference",
    "static_fixture_reference",
    "audit_events",
    "audit_summary",
    "boundary_checks",
    "blocked_capabilities",
    "source_boundaries",
    "future_steps",
    "status",
}
SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SAFE_FALSE_EVIDENCE = {
    "decision_engine_execution_performed",
    "risk_engine_execution_performed",
    "model_inference_execution_allowed",
    "order_generation_allowed",
    "order_submission_allowed",
    "fills_allowed",
    "live_mode_allowed",
    "testnet_mode_allowed",
    "audit_export_allowed",
    "cloud_export_allowed",
    "external_export_allowed",
}


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


def test_audit_envelope_is_plain_json_serializable_dict() -> None:
    envelope = build_preview_decision_engine_dry_run_audit_envelope()

    assert type(envelope) is dict
    _assert_plain(envelope)
    assert json.loads(json.dumps(envelope, sort_keys=True)) == envelope
    assert set(envelope) == EXPECTED_TOP_LEVEL_KEYS


def test_audit_envelope_references_contract_read_model_and_static_fixture() -> None:
    envelope = build_preview_decision_engine_dry_run_audit_envelope()

    assert envelope["schema_version"] == "preview_decision_engine_dry_run_audit_envelope.v1"
    assert (
        envelope["envelope_kind"]
        == "functional_preview_block_f_decision_engine_dry_run_audit_envelope"
    )
    assert envelope["block"] == "F"
    assert envelope["step"] == "8.3"
    assert envelope["audit_envelope_status"] == "audit_envelope_ready_no_engine_execution"
    assert envelope["dry_run_mode"] == "local_paper_dry_run"
    assert envelope["audit_envelope_decision"] == "BUILD_AUDIT_ENVELOPE_ONLY_NO_ENGINE_EXECUTION"
    assert envelope["ready_for_block_f_4"] is True
    assert envelope["next_step"] == "FUNCTIONAL-PREVIEW-8.4"
    assert envelope["next_step_title"] == "DECISION ENGINE DRY-RUN UI READ-ONLY SURFACE"
    assert envelope["contract_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_contract.v1",
        "contract_kind": "functional_preview_block_f_decision_engine_dry_run_contract",
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION",
    }
    assert envelope["read_model_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_read_model.v1",
        "read_model_kind": "functional_preview_block_f_decision_engine_dry_run_read_model",
        "read_model_status": "read_model_snapshot_ready_no_engine_execution",
        "read_model_decision": "BUILD_READ_MODEL_ONLY_NO_ENGINE_EXECUTION",
    }
    assert envelope["static_fixture_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_static_fixture.v1",
        "fixture_kind": "functional_preview_block_f_decision_engine_dry_run_static_fixture",
        "static_fixture_status": "static_fixture_ready_no_engine_execution",
        "static_fixture_decision": "BUILD_STATIC_FIXTURE_ONLY_NO_ENGINE_EXECUTION",
    }


def test_audit_events_match_fixture_cases_and_echo_preview_data() -> None:
    envelope = build_preview_decision_engine_dry_run_audit_envelope()
    fixture = build_preview_decision_engine_dry_run_static_fixture()

    assert len(envelope["audit_events"]) == len(fixture["fixture_cases"])
    for index, (event, fixture_case) in enumerate(
        zip(envelope["audit_events"], fixture["fixture_cases"], strict=True), start=1
    ):
        read_model_snapshot = fixture_case["read_model_snapshot"]
        input_snapshot = fixture_case["input_snapshot"]
        assert event["audit_event_id"] == (
            f"dry-run-audit-{index:04d}-{fixture_case['case_id'].replace('_', '-')}"
        )
        assert event["audit_event_type"] == "decision_engine_dry_run_fixture_case_audit"
        assert event["case_id"] == fixture_case["case_id"]
        assert event["case_description"] == fixture_case["description"]
        assert event["dry_run_context_id"] == input_snapshot["dry_run_context_id"]
        assert event["operator_selected_pair"] == input_snapshot["operator_selected_pair"]
        assert event["operator_selected_candidate"] == input_snapshot["operator_selected_candidate"]
        assert event["decision_preview"] == read_model_snapshot["decision_preview"]
        assert event["risk_check_preview"] == read_model_snapshot["risk_check_preview"]
        assert event["audit_preview"] == read_model_snapshot["audit_preview"]
        assert event["boundary_snapshot"] == read_model_snapshot["boundary_checks"]
        assert event["event_status"] == "ready_no_engine_execution_no_orders"


def test_each_event_has_safe_false_no_execution_evidence() -> None:
    envelope = build_preview_decision_engine_dry_run_audit_envelope()

    for event in envelope["audit_events"]:
        assert set(event["no_execution_evidence"]) == SAFE_FALSE_EVIDENCE
        assert all(value is False for value in event["no_execution_evidence"].values())


def test_audit_summary_boundaries_blocked_capabilities_and_source_boundaries_are_safe() -> None:
    envelope = build_preview_decision_engine_dry_run_audit_envelope()
    expected_boundaries = dict(REQUIRED_BOUNDARIES)
    expected_boundaries["model_inference_execution_allowed"] = False

    assert envelope["audit_summary"] == {
        "audit_event_count": len(envelope["audit_events"]),
        "fixture_case_count": len(
            build_preview_decision_engine_dry_run_static_fixture()["fixture_cases"]
        ),
        "all_events_have_deterministic_ids": True,
        "all_events_no_engine_execution": True,
        "all_events_no_risk_engine_execution": True,
        "all_events_no_model_inference": True,
        "all_events_no_order_generation": True,
        "all_events_no_order_submission": True,
        "all_events_no_fills": True,
        "all_events_no_live": True,
        "all_events_no_testnet": True,
        "all_events_no_export": True,
        "all_events_json_serializable": True,
        "engine_execution_performed": False,
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "live_mode_allowed": False,
        "testnet_mode_allowed": False,
    }
    assert envelope["boundary_checks"] == expected_boundaries
    assert {
        "decision engine execution",
        "real decision recommendation",
        "model inference",
        "risk engine evaluation",
        "order generation",
        "order submission",
        "fills",
        "live trading",
        "testnet/sandbox trading",
        "account/balance fetch",
        "secrets read/export",
        "cloud/external export",
        "new QML action calls",
        "EXE packaging",
    } <= set(envelope["blocked_capabilities"])
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
    } <= set(envelope["source_boundaries"])
    assert envelope["future_steps"] == [
        "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
        "functional_preview_8_5_block_f_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_and_dry_run_modules_with_no_forbidden_calls() -> None:
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
        "ui.pyside_app.preview_decision_engine_dry_run_static_fixture",
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
        "order",
        "live",
        "testnet",
        "account",
        "secrets",
        "network",
        "filesystem",
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
