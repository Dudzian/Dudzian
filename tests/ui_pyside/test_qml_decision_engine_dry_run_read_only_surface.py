from __future__ import annotations

import ast
from pathlib import Path

from ui.pyside_app.preview_action_dispatch_bridge_provider import (
    build_paper_runtime_action_dispatch_bridge_provider_snapshot,
)
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_decision_engine_dry_run_audit_envelope import (
    build_preview_decision_engine_dry_run_audit_envelope,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
SNAPSHOT_HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_action_dispatch_bridge_snapshot.py"
PROVIDER_HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_action_dispatch_bridge_provider.py"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
BOUNDARY_FALSE_FLAGS = {
    "decision_engine_execution_allowed",
    "decision_engine_execution_performed",
    "model_inference_execution_allowed",
    "trading_controller_allowed",
    "decision_envelope_allowed",
    "strategy_execution_allowed",
    "ai_scoring_execution_allowed",
    "runtime_loop_allowed",
    "command_dispatch_execution_allowed",
    "lifecycle_execution_allowed",
    "order_generation_allowed",
    "order_submission_allowed",
    "fills_allowed",
    "live_mode_allowed",
    "testnet_mode_allowed",
    "account_fetch_allowed",
    "market_account_fetch_allowed",
    "secrets_read_allowed",
    "secrets_export_allowed",
    "cloud_export_allowed",
    "external_export_allowed",
    "dynamic_action_dispatch_allowed",
    "new_qml_method_calls_allowed",
    "exe_packaging_in_scope",
    "bat_productization_allowed",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_snapshot_and_provider_expose_8_3_audit_envelope_plain_dict() -> None:
    expected = build_preview_decision_engine_dry_run_audit_envelope()
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    provider_snapshot = build_paper_runtime_action_dispatch_bridge_provider_snapshot()

    for payload in (snapshot, provider_snapshot):
        envelope = payload["decision_engine_dry_run_audit_envelope"]
        assert type(envelope) is dict
        assert envelope == expected
        assert envelope["schema_version"] == "preview_decision_engine_dry_run_audit_envelope.v1"
        assert envelope["step"] == "8.3"
        assert envelope["audit_envelope_status"] == "audit_envelope_ready_no_engine_execution"
        assert envelope["ready_for_block_f_4"] is True
        assert envelope["next_step"] == "FUNCTIONAL-PREVIEW-8.4"
        assert payload["decision_engine_dry_run_ui_surface_status"] == (
            "read_only_surface_ready_no_engine_execution"
        )
        assert payload["ready_for_block_f_5"] is True
        assert payload["next_step_after_ui_surface"] == "FUNCTIONAL-PREVIEW-8.5"


def test_audit_envelope_boundary_flags_remain_no_execution_no_export() -> None:
    envelope = build_paper_runtime_action_dispatch_bridge_provider_snapshot()[
        "decision_engine_dry_run_audit_envelope"
    ]
    boundaries = envelope["boundary_checks"]

    assert boundaries["local_only"] is True
    assert boundaries["paper_only"] is True
    assert boundaries["dry_run_only"] is True
    assert boundaries["exe_direction_preserved"] is True
    for flag in BOUNDARY_FALSE_FLAGS:
        assert boundaries[flag] is False


def test_operator_dashboard_has_read_only_audit_card_and_required_text() -> None:
    source = _source(OPERATOR_DASHBOARD)

    for token in (
        "decisionEngineDryRunAuditEnvelope",
        "decisionEngineDryRunAuditSummary",
        "decisionEngineDryRunAuditEvents",
        "decision_engine_dry_run_audit_envelope",
        "operatorDashboardDecisionEngineDryRunAuditCard",
        "operatorDashboardDecisionEngineDryRunAuditStatus",
        "operatorDashboardDecisionEngineDryRunAuditSummary",
        "operatorDashboardDecisionEngineDryRunAuditEvents",
        "Dry-run read-only UI surface from 8.3 audit envelope",
        "no engine execution",
        "no orders",
        "event_count",
        "all_events_no_engine_execution",
        "all_events_no_order_generation",
        "all_events_no_export",
        "FUNCTIONAL-PREVIEW-8.5",
    ):
        assert token in source


def test_operator_dashboard_did_not_add_new_qml_bridge_method_calls() -> None:
    joined_qml = "\n".join((_source(MAIN_WINDOW), _source(OPERATOR_DASHBOARD)))

    assert joined_qml.count("previewSelectAction(") == 1
    assert joined_qml.count(ALLOWED_CALL) == 1
    assert "previewSelectSourceControl(" not in joined_qml
    assert "resetPreviewSelection(" not in joined_qml
    assert "start_runtime(" not in joined_qml
    assert "start_loop(" not in joined_qml
    assert "runtime_loop(" not in joined_qml
    assert "submit_order(" not in joined_qml
    assert "place_order(" not in joined_qml
    assert "create_order(" not in joined_qml
    assert "send_order(" not in joined_qml
    assert "fill_order(" not in joined_qml


def test_audit_card_source_slice_has_no_execution_handlers_or_buttons() -> None:
    source = _source(OPERATOR_DASHBOARD)
    start = source.index('objectName: "operatorDashboardDecisionEngineDryRunAuditCard"')
    end = source.index('objectName: "operatorDashboardBlockCReadOnlyBindingSummary"')
    card_source = source[start:end]

    for forbidden in (
        "onClicked",
        "MouseArea",
        "TapHandler",
        "Button",
        "IconButton",
        "previewSelectAction",
        "previewSelectSourceControl",
        "resetPreviewSelection",
    ):
        assert forbidden not in card_source


def test_snapshot_helper_imports_only_existing_audit_helper_no_forbidden_runtime_modules() -> None:
    tree = ast.parse(_source(SNAPSHOT_HELPER))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            calls.add(getattr(node.func, "attr", None) or getattr(node.func, "id", ""))

    assert "ui.pyside_app.preview_decision_engine_dry_run_audit_envelope" in imports
    for forbidden in (
        "TradingController",
        "DecisionEnvelope",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "QQmlApplicationEngine",
    ):
        assert forbidden not in imports
        assert forbidden not in calls


def test_provider_source_remains_without_direct_audit_or_qml_runtime_imports() -> None:
    source = _source(PROVIDER_HELPER)

    assert "PySide6" not in source
    assert "QQmlApplicationEngine" not in source
    assert "TradingController" not in source
    assert "DecisionEnvelope" not in source
    assert "build_preview_decision_engine_dry_run_audit_envelope" not in source
