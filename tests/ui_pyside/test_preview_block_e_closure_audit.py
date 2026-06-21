"""Source-only tests for BLOK E closure audit."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_block_e_closure_audit import (
    BLOCK_STATUS,
    CLOSURE_DECISION,
    NEXT_BLOCK,
    NEXT_BLOCK_TITLE,
    PREVIEW_BLOCK_E_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_E_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_F,
    build_preview_block_e_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_block_e_closure_audit.py"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
PAPER_TERMINAL = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "PaperTerminal.qml"
SMOKE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
APP = REPO_ROOT / "ui" / "pyside_app" / "app.py"
ALLOWED_CALL = (
    "paperRuntimeActionDispatchBridge.previewSelectAction("
    '"paper_runtime_snapshot_refresh_requested")'
)
EXPECTED_COMPLETED = {
    "block_d_bridge_registered_in_real_qml_context",
    "operator_dashboard_reads_action_dispatch_snapshot",
    "disabled_action_catalog_visible",
    "selection_preview_gate_reconciled_after_first_call",
    "exactly_one_preview_select_action_qml_call",
    "snapshot_refresh_preview_only_call_runtime_proven",
    "no_shadowing_of_paper_runtime_action_dispatch_bridge",
    "no_smoke_ad_hoc_bridge_creation",
    "operator_workflow_pair_propagation_green",
    "selected_terminal_pair_direct_writer_guarded",
    "windows_qml_green_on_current_head",
}
EXPECTED_BOUNDARIES = {
    "paper_only": True,
    "local_only": True,
    "execution_allowed": False,
    "execution_performed": False,
    "runtime_loop_started": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "order_generation_allowed": False,
    "order_submission_allowed": False,
    "fills_allowed": False,
    "trading_controller_touched": False,
    "decision_envelope_touched": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "preview_select_source_control_allowed": False,
    "reset_preview_selection_allowed": False,
    "start_stop_pause_resume_qml_calls_allowed": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}
EXPECTED_BLOCKED = {
    "live trading",
    "testnet/sandbox trading",
    "order generation",
    "order submission",
    "fills",
    "runtime loop execution",
    "lifecycle command execution",
    "dynamic action dispatch",
    "previewSelectSourceControl",
    "resetPreviewSelection",
    "start/stop/pause/resume QML calls",
    "TradingController integration",
    "DecisionEnvelope integration",
    "account/balance fetch",
    "secrets read/export",
    "cloud/external export",
    "EXE packaging",
}
EXPECTED_FUTURE_BLOCKS = [
    "block_f_decision_engine_dry_run_integration",
    "block_g_paper_only_decision_to_order_path",
    "block_h_read_only_real_market_adapter",
    "block_i_testnet_sandbox_adapter",
    "block_j_risk_governor_limits_kill_switch",
    "block_k_observability_audit_rollback_soak",
    "block_l_live_canary_live_transition_gates",
    "future_exe_packaging_block",
]
SIMPLE_TYPES = (dict, list, str, bool, int, type(None))


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


def test_closure_audit_returns_json_serializable_plain_dict() -> None:
    audit = build_preview_block_e_closure_audit()

    assert isinstance(audit, dict)
    _assert_plain(audit)
    assert json.loads(json.dumps(audit, sort_keys=True)) == audit


def test_status_closure_decision_and_block_f_readiness_are_fixed() -> None:
    audit = build_preview_block_e_closure_audit()

    assert audit["schema_version"] == PREVIEW_BLOCK_E_CLOSURE_AUDIT_SCHEMA_VERSION
    assert audit["audit_kind"] == PREVIEW_BLOCK_E_CLOSURE_AUDIT_KIND
    assert audit["block"] == "E"
    assert audit["block_status"] == BLOCK_STATUS
    assert audit["closure_decision"] == CLOSURE_DECISION
    assert audit["ready_for_block_f"] is READY_FOR_BLOCK_F
    assert audit["next_block"] == NEXT_BLOCK
    assert audit["next_block_title"] == NEXT_BLOCK_TITLE
    assert audit["status"] == "closed_ready_for_block_f_no_execution"


def test_completed_capabilities_are_explicit() -> None:
    audit = build_preview_block_e_closure_audit()

    assert set(audit["completed_capabilities"]) == EXPECTED_COMPLETED


def test_all_boundary_flags_match_no_execution_contract() -> None:
    audit = build_preview_block_e_closure_audit()

    assert audit["boundary_checks"] == EXPECTED_BOUNDARIES


def test_exactly_one_action_call_contract_is_snapshot_refresh_only() -> None:
    contract = build_preview_block_e_closure_audit()["action_dispatch_contract"]

    assert contract == {
        "allowed_qml_method_call_count": 1,
        "allowed_qml_method": "previewSelectAction",
        "allowed_qml_action": "paper_runtime_snapshot_refresh_requested",
        "allowed_qml_call_literal": ALLOWED_CALL,
        "allowed_qml_call_status": "accepted_intent_not_executed",
        "runtime_proof_required": True,
        "runtime_proof_present": True,
        "execution_allowed": False,
        "execution_performed": False,
    }


def test_bridge_runtime_operator_writer_and_exe_contracts_are_closed() -> None:
    audit = build_preview_block_e_closure_audit()

    assert audit["qml_context_bridge_contract"] == {
        "registered_context_property": "paperRuntimeActionDispatchBridge",
        "central_registration_required": True,
        "operator_dashboard_shadowing_allowed": False,
        "operator_dashboard_shadowing_present": False,
        "smoke_ad_hoc_bridge_creation_allowed": False,
        "smoke_ad_hoc_bridge_creation_present": False,
        "second_qqmlapplicationengine_allowed": False,
        "app_py_ad_hoc_registration_allowed": False,
    }
    assert audit["runtime_smoke_contract"]["preview_call_accepted"] is True
    assert audit["runtime_smoke_contract"]["preview_call_executed"] is False
    assert audit["operator_workflow_contract"] == {
        "select_scanner_pair_propagates_to_terminal_pair": True,
        "terminal_open_preserves_selected_pair": True,
        "terminal_active_pair_uses_shared_state_before_default": True,
        "link_usdt_not_terminal_default": True,
        "selected_terminal_pair_direct_writers_guarded": True,
    }
    assert audit["selected_terminal_pair_writer_contract"]["direct_writers_guarded"] is True
    assert audit["exe_direction_contract"] == {
        "final_artifact_direction": "windows_exe",
        "bat_files_are_dev_preview_only": True,
        "bat_productization_allowed": False,
        "pyinstaller_packaging_in_scope": False,
    }


def test_blocked_capabilities_and_future_blocks_are_explicit() -> None:
    audit = build_preview_block_e_closure_audit()

    assert set(audit["blocked_capabilities"]) == EXPECTED_BLOCKED
    assert audit["open_items_for_future_blocks"] == EXPECTED_FUTURE_BLOCKS


def test_helper_imports_no_pyside_and_uses_no_io_or_runtime_modules() -> None:
    tree = ast.parse(_source(HELPER))
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
            calls.append(name)

    forbidden_import_roots = {
        "PySide6",
        "os",
        "pathlib",
        "requests",
        "urllib",
        "subprocess",
        "ui.pyside_app.app",
        "ui.pyside_app.qml_bridge",
    }
    assert not any(name.split(".")[0] in forbidden_import_roots for name in imports)
    assert not any("PySide" in name or "Qt" in name or "Qml" in name for name in imports)
    assert not {"open", "read_text", "write_text", "QQmlApplicationEngine"} & set(calls)


def test_qml_source_guard_keeps_exactly_one_allowed_call_and_no_forbidden_calls() -> None:
    sources = {
        "MainWindow.qml": _source(MAIN_WINDOW),
        "OperatorDashboard.qml": _source(OPERATOR_DASHBOARD),
        "PaperTerminal.qml": _source(PAPER_TERMINAL),
    }
    joined = "\n".join(sources.values())

    assert joined.count(ALLOWED_CALL) == 1
    assert joined.count("previewSelectAction(") == 1
    assert "previewSelectSourceControl(" not in joined
    assert "resetPreviewSelection(" not in joined
    assert "startRuntime" not in joined
    assert "stopRuntime" not in joined
    assert "pauseRuntime" not in joined
    assert "resumeRuntime" not in joined


def test_no_shadowing_no_injection_and_no_app_py_ad_hoc_registration_sources_hold() -> None:
    dashboard_source = _source(OPERATOR_DASHBOARD)
    smoke_source = _source(SMOKE)
    app_source = _source(APP)

    assert "property var paperRuntimeActionDispatchBridge" not in dashboard_source
    assert "id: paperRuntimeActionDispatchBridge" not in dashboard_source
    assert "paperRuntimeActionDispatchBridge:" not in dashboard_source
    assert 'setProperty("paperRuntimeActionDispatchBridge"' not in smoke_source
    assert "preview_action_dispatch_qt_bridge_registration" not in app_source
    assert "register_paper_runtime_action_dispatch_qt_bridge" not in app_source


def test_selected_terminal_pair_writer_guard_source_still_uses_helper() -> None:
    main_source = _source(MAIN_WINDOW)

    assert "function setTerminalPairFromSource(pair, writer)" in main_source
    assert "selectedTerminalPairLastWriter" in main_source
    assert (
        'function setTerminalPair(pair) { setTerminalPairFromSource(pair, "setTerminalPair") }'
        in main_source
    )
    assert 'setTerminalPairFromSource(scannerSelectedPair, "selectScannerPair")' in main_source
    assert 'setTerminalPairFromSource(preferredPair, "ensureSelectedTerminalPair")' in main_source
