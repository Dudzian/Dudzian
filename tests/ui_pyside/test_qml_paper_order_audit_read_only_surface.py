from __future__ import annotations

import ast
from pathlib import Path

from ui.pyside_app.preview_action_dispatch_bridge_provider import (
    build_paper_runtime_action_dispatch_bridge_provider_snapshot,
)
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_paper_order_audit_envelope import (
    build_preview_paper_order_audit_envelope,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
MAIN_WINDOW = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
SNAPSHOT_HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_action_dispatch_bridge_snapshot.py"
PROVIDER_HELPER = REPO_ROOT / "ui" / "pyside_app" / "preview_action_dispatch_bridge_provider.py"
ALLOWED_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
NO_EXECUTION_TRUE_FLAGS = {
    "all_events_no_intent_generated",
    "all_events_no_order_generated",
    "all_events_no_submission",
    "all_events_no_fills",
    "all_events_no_runtime_execution",
    "all_events_no_live_or_testnet",
    "all_events_no_account_or_secrets",
    "all_events_no_export",
}
FORBIDDEN_IMPORT_OR_CALL_TOKENS = {
    "TradingController",
    "DecisionEnvelope",
    "submit_order",
    "place_order",
    "create_order",
    "send_order",
    "fill_order",
    "runtime_loop",
    "start_runtime",
    "start_loop",
    "live_adapter",
    "testnet_adapter",
    "secrets",
    "network",
    "QQmlApplicationEngine",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_snapshot_exposes_9_3_paper_order_audit_envelope_plain_dict() -> None:
    expected = build_preview_paper_order_audit_envelope()
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert type(snapshot["paper_order_audit_envelope"]) is dict
    assert snapshot["paper_order_audit_envelope"] == expected
    assert snapshot["paper_order_audit_status"] == (
        "paper_order_audit_envelope_ready_no_order_generation"
    )
    assert snapshot["paper_order_audit_next_step"] == "FUNCTIONAL-PREVIEW-9.4"
    assert snapshot["paper_order_audit_ready_for_next_step"] is True
    assert snapshot["paper_order_audit_ready_for_ui_surface"] is True
    assert snapshot["paper_order_audit_ready_for_block_g_4"] is True
    assert snapshot["paper_order_audit_event_count"] == 4
    assert snapshot["paper_order_audit_unknown_input_key_events"] == 1


def test_snapshot_paper_order_audit_no_execution_summary_is_safe() -> None:
    summary = build_paper_runtime_action_dispatch_bridge_snapshot()[
        "paper_order_audit_no_execution_summary"
    ]

    assert type(summary) is dict
    for flag in NO_EXECUTION_TRUE_FLAGS:
        assert summary[flag] is True
    assert summary["audit_export_allowed"] is False
    assert summary["audit_export_performed"] is False


def test_provider_returns_paper_order_audit_fields_without_mutation_or_execution() -> None:
    provider_snapshot = build_paper_runtime_action_dispatch_bridge_provider_snapshot()
    bridge_snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    for key in (
        "paper_order_audit_envelope",
        "paper_order_audit_status",
        "paper_order_audit_ready_for_next_step",
        "paper_order_audit_next_step",
        "paper_order_audit_ready_for_ui_surface",
        "paper_order_audit_ready_for_block_g_4",
        "paper_order_audit_event_count",
        "paper_order_audit_unknown_input_key_events",
        "paper_order_audit_no_execution_summary",
    ):
        assert provider_snapshot[key] == bridge_snapshot[key]
    assert provider_snapshot["provider_execution_allowed"] is False
    assert provider_snapshot["provider_execution_performed"] is False
    assert provider_snapshot["execution_allowed"] is False
    assert provider_snapshot["execution_performed"] is False

    provider_snapshot["paper_order_audit_envelope"]["audit_events"].clear()
    assert (
        len(
            build_paper_runtime_action_dispatch_bridge_provider_snapshot()[
                "paper_order_audit_envelope"
            ]["audit_events"]
        )
        == 4
    )


def test_operator_dashboard_has_paper_order_audit_read_only_card_and_texts() -> None:
    source = _source(OPERATOR_DASHBOARD)

    for token in (
        "paperOrderAuditEnvelope",
        "paperOrderAuditNoExecutionSummary",
        "paper_order_audit_envelope",
        "paper_order_audit_no_execution_summary",
        "operatorDashboardPaperOrderAuditReadOnlyCard",
        "operatorDashboardPaperOrderAuditStatusLabel",
        "operatorDashboardPaperOrderAuditEventCountLabel",
        "operatorDashboardPaperOrderAuditNoExecutionLabel",
        "operatorDashboardPaperOrderAuditExportBlockedLabel",
        "operatorDashboardPaperOrderAuditNextStepLabel",
        "Paper order audit — read-only",
        "No order intent generated",
        "No paper order generated",
        "No submission",
        "No fills",
        "No runtime execution",
        "Audit export blocked",
        "Live/testnet blocked",
        "Next: %1",
        "FUNCTIONAL-PREVIEW-9.4",
        "event_count",
        "unknown_input_key_events",
        "account/secrets/export blocked",
    ):
        assert token in source


def test_operator_dashboard_does_not_add_new_qml_method_calls_or_order_calls() -> None:
    joined_qml = "\n".join((_source(MAIN_WINDOW), _source(OPERATOR_DASHBOARD)))

    assert joined_qml.count("previewSelectAction(") == 1
    assert joined_qml.count(ALLOWED_CALL) == 1
    assert "previewSelectSourceControl(" not in joined_qml
    assert "resetPreviewSelection(" not in joined_qml
    for forbidden in (
        "submit_order(",
        "place_order(",
        "create_order(",
        "send_order(",
        "fill_order(",
        "start_runtime(",
        "start_loop(",
        "runtime_loop(",
    ):
        assert forbidden not in joined_qml


def test_paper_order_audit_card_source_slice_has_no_handlers_buttons_or_bridge_calls() -> None:
    source = _source(OPERATOR_DASHBOARD)
    start = source.index('objectName: "operatorDashboardPaperOrderAuditReadOnlyCard"')
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
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
    ):
        assert forbidden not in card_source


def test_bridge_snapshot_imports_audit_envelope_only_without_runtime_order_live_modules() -> None:
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

    assert "ui.pyside_app.preview_paper_order_audit_envelope" in imports
    assert "build_preview_paper_order_audit_envelope" in calls
    for forbidden in FORBIDDEN_IMPORT_OR_CALL_TOKENS:
        assert forbidden not in imports
        assert forbidden not in calls


def test_provider_remains_indirect_snapshot_consumer_without_qml_runtime_imports() -> None:
    source = _source(PROVIDER_HELPER)

    assert "build_preview_paper_order_audit_envelope" not in source
    assert "PySide6" not in source
    assert "QQmlApplicationEngine" not in source
    assert "TradingController" not in source
    assert "DecisionEnvelope" not in source
