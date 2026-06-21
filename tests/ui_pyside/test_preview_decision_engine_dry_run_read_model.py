from __future__ import annotations

import ast
import copy
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_decision_engine_dry_run_contract import (
    ALLOWED_DRY_RUN_INPUTS,
    REQUIRED_BOUNDARIES,
)
from ui.pyside_app.preview_decision_engine_dry_run_read_model import (
    build_preview_decision_engine_dry_run_read_model_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_decision_engine_dry_run_read_model.py"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
APP = REPO_ROOT / "ui" / "pyside_app" / "app.py"
ALLOWED_CALL = (
    "paperRuntimeActionDispatchBridge.previewSelectAction("
    '"paper_runtime_snapshot_refresh_requested")'
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "read_model_kind",
    "block",
    "step",
    "read_model_status",
    "dry_run_mode",
    "read_model_decision",
    "ready_for_block_f_2",
    "next_step",
    "next_step_title",
    "contract_reference",
    "input_snapshot",
    "input_snapshot_echo",
    "read_model",
    "decision_preview",
    "risk_check_preview",
    "audit_preview",
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


def test_read_model_snapshot_is_plain_json_serializable_dict() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()

    assert type(snapshot) is dict
    _assert_plain(snapshot)
    assert json.loads(json.dumps(snapshot, sort_keys=True)) == snapshot
    assert set(snapshot) == EXPECTED_TOP_LEVEL_KEYS


def test_default_snapshot_has_fixed_status_contract_reference_and_all_required_fields() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()

    assert snapshot["schema_version"] == "preview_decision_engine_dry_run_read_model.v1"
    assert (
        snapshot["read_model_kind"]
        == "functional_preview_block_f_decision_engine_dry_run_read_model"
    )
    assert snapshot["block"] == "F"
    assert snapshot["step"] == "8.1"
    assert snapshot["read_model_status"] == "read_model_snapshot_ready_no_engine_execution"
    assert snapshot["dry_run_mode"] == "local_paper_dry_run"
    assert snapshot["read_model_decision"] == "BUILD_READ_MODEL_ONLY_NO_ENGINE_EXECUTION"
    assert snapshot["ready_for_block_f_2"] is True
    assert snapshot["next_step"] == "FUNCTIONAL-PREVIEW-8.2"
    assert snapshot["next_step_title"] == "DECISION ENGINE DRY-RUN STATIC FIXTURE"
    assert snapshot["contract_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_contract.v1",
        "contract_kind": "functional_preview_block_f_decision_engine_dry_run_contract",
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION",
    }
    assert snapshot["status"] == "ready_no_engine_execution_no_orders"


def test_default_input_snapshot_is_normalized_and_echoed() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()
    input_snapshot = snapshot["input_snapshot"]

    assert input_snapshot == snapshot["input_snapshot_echo"]
    assert set(input_snapshot) == set(ALLOWED_DRY_RUN_INPUTS)
    assert input_snapshot["dry_run_context_id"] == "local-preview-dry-run-context"
    assert input_snapshot["operator_selected_pair"] == "BTC/USDT"
    assert input_snapshot["operator_selected_candidate"] == {
        "pair": "BTC/USDT",
        "source": "local_preview_default",
        "confidence": 0.0,
    }
    for key in (
        "local_preview_state_snapshot",
        "paper_runtime_snapshot",
        "scanner_candidate_snapshot",
        "risk_preview_snapshot",
        "portfolio_preview_snapshot",
    ):
        assert input_snapshot[key] == {"source": key, "available": False}


def test_custom_plain_input_is_copied_normalized_and_unknown_keys_reported() -> None:
    custom = {
        "dry_run_context_id": "custom-context",
        "operator_selected_pair": "ETH/USDT",
        "operator_selected_candidate": {"pair": "ETH/USDT", "confidence": 0.25},
        "unknown_engine_trigger": {"execute": True},
        "submit_order": "must_be_ignored",
    }
    original = copy.deepcopy(custom)

    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot(custom)

    assert custom == original
    assert snapshot["input_snapshot"]["dry_run_context_id"] == "custom-context"
    assert snapshot["input_snapshot"]["operator_selected_pair"] == "ETH/USDT"
    assert snapshot["input_snapshot"]["operator_selected_candidate"] == {
        "pair": "ETH/USDT",
        "confidence": 0.25,
    }
    assert "unknown_engine_trigger" not in snapshot["input_snapshot"]
    assert "submit_order" not in snapshot["input_snapshot"]
    assert snapshot["read_model"]["ignored_input_keys"] == [
        "submit_order",
        "unknown_engine_trigger",
    ]
    assert snapshot["input_snapshot_echo"] == snapshot["input_snapshot"]


def test_read_model_decision_risk_and_audit_previews_are_no_execution_placeholders() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()

    assert snapshot["read_model"]["read_model_ready"] is True
    assert snapshot["read_model"]["engine_execution_required"] is False
    assert snapshot["read_model"]["engine_execution_performed"] is False
    assert snapshot["read_model"]["model_source"] == "contract_only_static_read_model"
    assert snapshot["read_model"]["input_keys_present"] == list(ALLOWED_DRY_RUN_INPUTS)
    assert snapshot["read_model"]["input_keys_missing"] == []
    assert snapshot["read_model"]["allowed_input_keys"] == list(ALLOWED_DRY_RUN_INPUTS)

    assert snapshot["decision_preview"] == {
        "decision_preview_ready": True,
        "decision_source": "dry_run_read_model_no_engine",
        "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
        "decision_status": "not_executed",
        "confidence_preview": 0.0,
        "reason_summary": "Decision engine execution is disabled in FUNCTIONAL-PREVIEW-8.1.",
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "execution_performed": False,
    }
    assert snapshot["risk_check_preview"] == {
        "risk_check_preview_ready": True,
        "risk_engine_execution_performed": False,
        "risk_status": "not_evaluated_contract_only",
        "blocked_reason_preview": "Risk engine execution is disabled in FUNCTIONAL-PREVIEW-8.1.",
    }
    assert snapshot["audit_preview"] == {
        "audit_event_preview_ready": True,
        "audit_event_type": "decision_engine_dry_run_read_model_snapshot",
        "audit_export_allowed": False,
        "cloud_export_allowed": False,
        "external_export_allowed": False,
    }


def test_boundaries_match_contract_and_add_model_inference_block() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()
    expected = dict(REQUIRED_BOUNDARIES)
    expected["model_inference_execution_allowed"] = False

    assert snapshot["boundary_checks"] == expected


def test_blocked_capabilities_source_boundaries_and_future_steps_are_explicit() -> None:
    snapshot = build_preview_decision_engine_dry_run_read_model_snapshot()
    blocked = set(snapshot["blocked_capabilities"])

    assert {
        "decision engine execution",
        "order generation",
        "order submission",
        "live trading",
        "testnet/sandbox trading",
        "secrets read/export",
        "cloud/external export",
        "real decision recommendation",
        "model inference",
        "risk engine evaluation",
    } <= blocked
    assert "no PySide import" in snapshot["source_boundaries"]
    assert "no filesystem I/O" in snapshot["source_boundaries"]
    assert snapshot["future_steps"] == [
        "functional_preview_8_2_decision_engine_dry_run_static_fixture",
        "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
        "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
        "functional_preview_8_5_block_f_closure_audit",
    ]


def test_helper_imports_only_safe_stdlib_plus_contract_and_has_no_forbidden_calls() -> None:
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
        "secrets",
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


def test_qml_bat_and_app_sources_are_not_changed_or_expanded_by_read_model() -> None:
    joined_qml = "\n".join((_source(MAIN_WINDOW), _source(OPERATOR_DASHBOARD)))
    changed = {line[3:] for line in _git_name_status() if line.startswith(("M  ", "A  ", "D  "))}
    bat_paths = {str(path) for path in REPO_ROOT.glob("*.bat")}

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
