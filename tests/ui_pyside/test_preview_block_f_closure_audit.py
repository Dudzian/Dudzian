"""Source-only tests for BLOK F closure audit."""

from __future__ import annotations

import ast
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType

from ui.pyside_app.preview_block_f_closure_audit import (
    BLOCK_STATUS,
    CLOSURE_DECISION,
    NEXT_BLOCK,
    NEXT_BLOCK_TITLE,
    PREVIEW_BLOCK_F_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_F_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G,
    build_preview_block_f_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_block_f_closure_audit.py"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
PAPER_TERMINAL = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "PaperTerminal.qml"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
EXPECTED_COMPLETED_STEPS = [
    "functional_preview_8_0_decision_engine_dry_run_contract",
    "functional_preview_8_1_decision_engine_dry_run_read_model_snapshot",
    "functional_preview_8_2_decision_engine_dry_run_static_fixture",
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
]
EXPECTED_BOUNDARIES = {
    "local_only": True,
    "paper_only": True,
    "dry_run_only": True,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "model_inference_execution_allowed": False,
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
EXPECTED_BLOCKED = {
    "decision engine execution",
    "real decision recommendation",
    "model inference",
    "risk engine evaluation",
    "TradingController integration",
    "DecisionEnvelope integration",
    "strategy execution",
    "AI/scoring execution",
    "dynamic action dispatch",
    "runtime loop execution",
    "lifecycle command execution",
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
}
EXPECTED_SOURCE_BOUNDARIES = {
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
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
}
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
    audit = build_preview_block_f_closure_audit()

    assert isinstance(audit, dict)
    _assert_plain(audit)
    assert json.loads(json.dumps(audit, sort_keys=True)) == audit


def test_status_closure_decision_and_block_g_readiness_are_fixed() -> None:
    audit = build_preview_block_f_closure_audit()

    assert audit["schema_version"] == PREVIEW_BLOCK_F_CLOSURE_AUDIT_SCHEMA_VERSION
    assert audit["audit_kind"] == PREVIEW_BLOCK_F_CLOSURE_AUDIT_KIND
    assert audit["block"] == "F"
    assert audit["block_status"] == BLOCK_STATUS
    assert audit["closure_decision"] == CLOSURE_DECISION
    assert audit["ready_for_block_g"] is READY_FOR_BLOCK_G
    assert audit["next_block"] == NEXT_BLOCK
    assert audit["next_block_title"] == NEXT_BLOCK_TITLE
    assert audit["status"] == "closed_ready_for_block_g_paper_only_no_execution"


def test_completed_steps_and_required_evidence_cover_8_0_through_8_4() -> None:
    audit = build_preview_block_f_closure_audit()

    assert audit["completed_steps"] == EXPECTED_COMPLETED_STEPS
    assert set(audit["required_evidence"]) >= {
        "contract_ready_no_execution",
        "read_model_snapshot_ready_no_engine_execution",
        "static_fixture_ready_no_engine_execution",
        "audit_envelope_ready_no_engine_execution",
        "ui_read_only_surface_ready_no_engine_execution",
        "exactly_one_qml_preview_select_action_call",
        "no_decision_engine_execution",
        "no_order_generation",
        "no_live_or_testnet",
        "exe_direction_preserved",
    }


def test_references_for_8_0_through_8_4_are_fixed() -> None:
    audit = build_preview_block_f_closure_audit()

    assert audit["contract_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_contract.v1",
        "contract_kind": "functional_preview_block_f_decision_engine_dry_run_contract",
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": "START_BLOCK_F_WITH_CONTRACT_ONLY_NO_ENGINE_EXECUTION",
    }
    assert audit["read_model_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_read_model.v1",
        "read_model_kind": "functional_preview_block_f_decision_engine_dry_run_read_model",
        "read_model_status": "read_model_snapshot_ready_no_engine_execution",
        "read_model_decision": "BUILD_READ_MODEL_ONLY_NO_ENGINE_EXECUTION",
    }
    assert audit["static_fixture_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_static_fixture.v1",
        "fixture_kind": "functional_preview_block_f_decision_engine_dry_run_static_fixture",
        "static_fixture_status": "static_fixture_ready_no_engine_execution",
        "static_fixture_decision": "BUILD_STATIC_FIXTURE_ONLY_NO_ENGINE_EXECUTION",
    }
    assert audit["audit_envelope_reference"] == {
        "schema_version": "preview_decision_engine_dry_run_audit_envelope.v1",
        "envelope_kind": "functional_preview_block_f_decision_engine_dry_run_audit_envelope",
        "audit_envelope_status": "audit_envelope_ready_no_engine_execution",
        "audit_envelope_decision": "BUILD_AUDIT_ENVELOPE_ONLY_NO_ENGINE_EXECUTION",
    }
    assert audit["ui_surface_reference"] == {
        "ui_surface_status": "read_only_surface_ready_no_engine_execution",
        "ready_for_block_f_5": True,
        "next_step_after_ui_surface": "FUNCTIONAL-PREVIEW-8.5",
        "surface_kind": "decision_engine_dry_run_ui_read_only_surface",
    }


def test_boundary_checks_blocked_capabilities_and_source_boundaries_are_safe() -> None:
    audit = build_preview_block_f_closure_audit()

    assert audit["boundary_checks"] == EXPECTED_BOUNDARIES
    assert set(audit["blocked_capabilities"]) == EXPECTED_BLOCKED
    assert set(audit["source_boundaries"]) == EXPECTED_SOURCE_BOUNDARIES


def test_qml_surface_contract_is_read_only_with_exactly_one_allowed_call() -> None:
    contract = build_preview_block_f_closure_audit()["qml_surface_contract"]

    assert contract == {
        "surface_present": True,
        "surface_object_name": "operatorDashboardDecisionEngineDryRunAuditCard",
        "surface_status_object_name": "operatorDashboardDecisionEngineDryRunAuditStatus",
        "surface_summary_object_name": "operatorDashboardDecisionEngineDryRunAuditSummary",
        "surface_events_object_name": "operatorDashboardDecisionEngineDryRunAuditEvents",
        "surface_read_only": True,
        "new_qml_method_calls_added": False,
        "execution_buttons_added": False,
        "allowed_qml_method_call_count": 1,
        "allowed_qml_method_call_literal": ALLOWED_CALL,
        "preview_select_source_control_present": False,
        "reset_preview_selection_present": False,
        "start_stop_pause_resume_calls_present": False,
        "dynamic_action_dispatch_present": False,
    }


def test_next_block_contract_keeps_block_g_paper_only_and_gated() -> None:
    contract = build_preview_block_f_closure_audit()["next_block_contract"]

    assert contract == {
        "next_block": "G",
        "next_block_title": "PAPER-ONLY DECISION-TO-ORDER PATH",
        "block_g_start_allowed": True,
        "block_g_must_remain_paper_only": True,
        "block_g_live_trading_allowed": False,
        "block_g_testnet_allowed_initially": False,
        "block_g_requires_new_gates_before_orders": True,
        "block_g_requires_risk_governor_before_any_order_path": True,
        "block_g_must_not_enable_live_credentials": True,
        "block_g_must_not_use_real_account_balance": True,
    }


def test_open_items_for_future_blocks_are_explicit() -> None:
    assert build_preview_block_f_closure_audit()["open_items_for_future_blocks"] == [
        "block_g_paper_only_decision_to_order_path",
        "block_h_read_only_real_market_adapter",
        "block_i_testnet_sandbox_adapter",
        "block_j_risk_governor_limits_kill_switch",
        "block_k_observability_audit_rollback_soak",
        "block_l_live_canary_live_transition_gates",
        "future_exe_packaging_block",
    ]


def test_helper_imports_only_safe_stdlib_and_accepted_8_0_to_8_4_helpers() -> None:
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

    allowed_imports = {
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_decision_engine_dry_run_contract",
        "ui.pyside_app.preview_decision_engine_dry_run_read_model",
        "ui.pyside_app.preview_decision_engine_dry_run_static_fixture",
        "ui.pyside_app.preview_decision_engine_dry_run_audit_envelope",
    }
    forbidden_import_fragments = (
        "PySide",
        "Qt",
        "Qml",
        "qml",
        "runtime",
        "TradingController",
        "DecisionEnvelope",
        "strategy",
        "scoring",
        "recommendation",
        "order",
        "live",
        "testnet",
        "account",
        "secret",
        "network",
        "filesystem",
        "requests",
        "subprocess",
        "pathlib",
        "os",
    )
    forbidden_calls = {
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

    assert set(imports) <= allowed_imports
    assert not any(fragment in name for name in imports for fragment in forbidden_import_fragments)
    assert not forbidden_calls & set(calls)


def test_qml_still_has_8_4_read_only_card_and_no_forbidden_action_surface() -> None:
    operator_source = _source(OPERATOR_DASHBOARD)
    joined = "\n".join([_source(MAIN_WINDOW), operator_source, _source(PAPER_TERMINAL)])

    assert "operatorDashboardDecisionEngineDryRunAuditCard" in operator_source
    assert "operatorDashboardDecisionEngineDryRunAuditStatus" in operator_source
    assert "operatorDashboardDecisionEngineDryRunAuditSummary" in operator_source
    assert "operatorDashboardDecisionEngineDryRunAuditEvents" in operator_source
    assert joined.count(ALLOWED_CALL) == 1
    assert joined.count("previewSelectAction(") == 1
    assert "previewSelectSourceControl(" not in joined
    assert "resetPreviewSelection(" not in joined
    assert "dynamicActionDispatch" not in joined
    assert "startRuntime" not in joined
    assert "stopRuntime" not in joined
    assert "pauseRuntime" not in joined
    assert "resumeRuntime" not in joined
    assert "submitOrder" not in joined
    assert "placeOrder" not in joined
    assert "createOrder" not in joined
    assert "sendOrder" not in joined
    assert "fillOrder" not in joined
