"""Non-PySide source proof for BLOK C read-only QML value consumption."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_DASHBOARD_QML = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"


def _block_c_panel_source(source: str) -> str:
    panel_marker = 'objectName: "operatorDashboardBlockCReadOnlyBindingSummary"'
    panel_start = source.find(panel_marker)
    assert panel_start >= 0
    next_card_start = source.find("Components.PreviewCard", panel_start + len(panel_marker))
    assert next_card_start > panel_start
    return source[panel_start:next_card_start]


def test_operator_dashboard_block_c_controlled_value_consumption_source_without_pyside() -> None:
    source = OPERATOR_DASHBOARD_QML.read_text(encoding="utf-8")
    panel_source = _block_c_panel_source(source)

    assert "function blockCReadOnlyBindingValue(key, fallback)" in source
    assert '"blockCReadOnlyBindingState"' in source
    for consumed_key in (
        "bindingKind",
        "blockStatus",
        "integrationGateStatus",
        "readyForUiRuntimeIntegration",
        "runtimeLoopStarted",
        "runtimeBacked",
        "uiBound",
        "generatedOrderCount",
        "generatedDecisionCount",
        "exportSink",
        "cloudSink",
        "externalExport",
    ):
        assert f'blockCReadOnlyBindingValue("{consumed_key}",' in panel_source

    for safe_fallback in (
        '"blocked"',
        "false",
        '"none"',
        "0",
    ):
        assert safe_fallback in panel_source

    assert "integration gate: blocked" not in panel_source
    assert 'blockCReadOnlyBindingValue("integrationGateStatus", "blocked")' in panel_source
    assert 'blockCReadOnlyBindingValue("runtimeLoopStarted", false)' in panel_source
    assert 'blockCReadOnlyBindingValue("runtimeBacked", false)' in panel_source
    assert 'blockCReadOnlyBindingValue("readyForUiRuntimeIntegration", false)' in panel_source

    assert "decision/export/live readiness: false" in panel_source
    assert "integration gate: blocked" not in panel_source
    assert 'blockCReadOnlyBindingValue("integrationGateStatus", "blocked")' in panel_source
    assert 'blockCReadOnlyBindingValue("runtimeLoopStarted", false)' in panel_source
    assert 'blockCReadOnlyBindingValue("runtimeBacked", false)' in panel_source
    assert 'blockCReadOnlyBindingValue("readyForUiRuntimeIntegration", false)' in panel_source
    for forbidden_action_token in (
        "onClicked",
        "exportButton",
        "exportHandler",
        "submit",
        "execute",
        "Button",
        "MouseArea",
        "startRuntime",
        "stopRuntime",
        "runRuntime",
        "command",
        "lifecycle",
    ):
        assert forbidden_action_token not in panel_source
