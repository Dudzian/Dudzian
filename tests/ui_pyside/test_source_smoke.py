"""Source-level smoke checks for the existing PySide6/QML UI."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ui.pyside_app.app import AppOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
PLAN_SOURCE = REPO_ROOT / "scripts" / "ui_preview_launch_plan.py"
APP_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "app.py"
QML_SOURCE_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
PALETTE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "theme" / "palette.json"
FORBIDDEN_SOURCE_TOKENS = (
    "create" + "_" + "order",
    "fetch" + "_" + "balance",
    "load" + "_" + "markets",
    "key" + "ring",
    "dot" + "env",
    "shell" + "=True",
    "subprocess" + "." + "run",
    "os" + "." + "environ",
)

PANEL_AUDIT_IDS = (
    "sidePanel",
    "telemetryPanel",
    "aiDecisionsPanel",
    "diagnosticsPanel",
    "chartView",
    "strategyWorkbench",
    "strategiesPanel",
    "riskControlsPanel",
    "modeWizardPanel",
    "strategyManagerPanel",
)


def _run_ui_smoke(*extra_args: str) -> subprocess.CompletedProcess[str]:
    run_process = getattr(subprocess, "run")
    return run_process(
        [
            sys.executable,
            "-m",
            "ui.pyside_app",
            "--config",
            "ui/config/preview_local.yaml",
            "--smoke",
            "--offscreen",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _smoke_payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    result.stdout.encode("cp1252")
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def test_smoke_flags_are_available_in_parser() -> None:
    options = AppOptions.parse(
        ["--config", "ui/config/preview_local.yaml", "--smoke", "--offscreen"]
    )

    assert options.smoke is True
    assert options.offscreen is True
    assert options.enable_cloud_runtime is False


def test_source_smoke_finishes_and_reports_safety_contract() -> None:
    result = _run_ui_smoke()
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["ui_loaded"] is True
    assert payload["qml_loaded"] is True
    assert payload["runtime_loop_started"] is False
    assert payload["production_runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["operator_dashboard_present"] is True
    assert payload["operator_dashboard_default"] is True
    assert payload["operator_dashboard_visible"] is True
    assert payload["active_panel_id"] in {"sidePanel", "operatorDashboard"}
    assert payload["central_content_empty"] is False
    panel_load_results = payload["panel_load_results"]
    assert isinstance(panel_load_results, dict)
    for panel_id in PANEL_AUDIT_IDS:
        result = panel_load_results[panel_id]
        assert result["loaded"] is True
        assert result["empty"] is False
        assert result["visible"] is True
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False
    assert payload["issues"] == []


def test_smoke_blocks_live_runtime_flag_without_qt_bootstrap() -> None:
    result = _run_ui_smoke("--enable-cloud-runtime")
    payload = _smoke_payload(result)

    assert result.returncode == 2
    assert payload["status"] == "blocked"
    assert payload["ui_loaded"] is False
    assert payload["qml_loaded"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["live_mode_allowed"] is False
    assert payload["operator_dashboard_present"] is False
    assert payload["operator_dashboard_default"] is False
    assert payload["operator_dashboard_visible"] is False
    assert payload["active_panel_id"] == ""
    assert payload["central_content_empty"] is True
    assert payload["issues"] == ["smoke_mode_blocks_enable_cloud_runtime"]


def test_smoke_and_plan_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (SMOKE_SOURCE, PLAN_SOURCE, APP_SOURCE)
    )

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def _qml_sources() -> list[Path]:
    return sorted(QML_SOURCE_ROOT.rglob("*.qml"))


def _qml_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _qml_sources())


def test_qml_design_system_color_tokens_are_registered() -> None:
    # Source safety guard: QML falls back poorly when a token is misspelled or removed,
    # so every literal designSystem.color("TOKEN") use in preview sources must exist in
    # the real dark palette shipped with the PySide preview.
    palette = json.loads(PALETTE_SOURCE.read_text(encoding="utf-8"))
    allowed_tokens = set(palette["palettes"]["dark"])
    token_pattern = re.compile(r"(?:\bdesignSystem|\b\w+\.designSystem)\.color\(\"([^\"]+)\"\)")

    used_tokens: set[str] = set()
    for qml_source in _qml_sources():
        used_tokens.update(token_pattern.findall(qml_source.read_text(encoding="utf-8")))

    assert used_tokens
    assert used_tokens <= allowed_tokens
    assert "success" not in used_tokens


def test_visible_preview_lists_use_dark_scrollbar_sources() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    ai_decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")
    strategy_manager = (QML_SOURCE_ROOT / "views" / "StrategyManager.qml").read_text(
        encoding="utf-8"
    )

    for source in (main_window, ai_decisions, strategy_manager):
        assert "ScrollBar.vertical: ScrollBar" in source
        assert 'designSystem.color("surfaceElevated")' in source
        assert 'designSystem.color("border")' in source


def test_operator_dashboard_uses_stable_contrast_tokens_for_visible_statuses() -> None:
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    assert 'accent: "accent"' in dashboard
    assert 'designSystem.color("accent")' in dashboard
    assert 'designSystem.color("success")' not in dashboard
    assert 'designSystem.color("positive")' not in dashboard
    assert "root.designSystem.color(modelData.accent)" not in dashboard


def test_qml_operator_dashboard_is_default_selected_panel() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert 'property string defaultPanelId: "sidePanel"' in main_window
    assert "property string currentPanelId: defaultPanelId" in main_window
    assert "showOperatorDashboard()" in main_window
    assert "layoutController.registerPanels(panelMetadata)" in main_window
    assert 'panelId: "sidePanel", title: qsTr("Dashboard operatora")' in main_window
    assert 'defaultPanelId: "modeWizardPanel"' not in main_window
    assert 'currentPanelId: "modeWizardPanel"' not in main_window
    assert 'defaultPanelId: ""' not in main_window


def test_qml_operator_dashboard_default_content_and_labels() -> None:
    source = _qml_text()
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    assert 'objectName: "operatorDashboardRoot"' in dashboard
    assert 'objectName: "operatorDashboardTitle"' in dashboard
    assert 'objectName: "operatorDashboardSafetySummary"' in dashboard
    assert 'objectName: "operatorDashboardFeed"' in dashboard
    assert 'objectName: "operatorDashboardRiskControls"' in dashboard
    assert "Dashboard operatora" in dashboard
    assert "Tryb: Demo / Paper" in dashboard
    assert "Kontrola ryzyka" in dashboard
    assert "Strumień decyzji" in source
    assert "Menedżer strategii" in source
    assert "Warsztat strategii" in source
    assert ("Decyzje governor" in source) or ("Decyzje strategii" in source)


def test_qml_operator_preview_removes_raw_labels_and_refresh_garbled_glyph() -> None:
    source = _qml_text()

    forbidden_labels = (
        "Chart_Decision Stream",
        "Chart & Decision Stream",
        "Decyzje AI",
        "Strategy Workbench",
        "Risk Controls",
        "Strategy Manager",
        "ǎ Odśwież dane",
        "ǎ",
    )
    for label in forbidden_labels:
        assert label not in source
    assert 'text: qsTr("Odśwież dane")' in source


def test_qml_operator_preview_demo_offline_safety_copy() -> None:
    source = _qml_text()

    required_copy = (
        "Exchange I/O disabled",
        "Order submission disabled",
        "Runtime loop not started",
        "API keys required: false",
        "Active strategy: Demo Momentum Guard",
        "Last decision: HOLD / NO ORDER",
        "Live trading: blocked / disabled",
        "Live disabled",
        "BTC/USDT demo row | HOLD | confidence 0.62 | no order",
        "ETH/USDT demo row | WAIT | confidence 0.55 | no order",
        "SOL/USDT demo row | BLOCKED LIVE | reason: demo mode",
        "Max drawdown guard: demo only",
        "Kill switch: armed / preview",
        "podłączony lokalny preview bridge",
    )
    for text in required_copy:
        assert text in source


def test_qml_operator_dashboard_has_real_visible_central_component_and_restorable_menu_action() -> (
    None
):
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    assert (
        '"sidePanel": { title: qsTr("Dashboard operatora"), icon: "fingerprint", component: sidePanelComponent }'
        in main_window
    )
    assert "id: sidePanelComponent" in main_window
    assert "Views.OperatorDashboard" in main_window
    assert 'objectName: "centralContentRoot"' in main_window
    assert 'objectName: "centralContentLoader"' in main_window
    assert "sourceComponent: root.selectedPanelComponent()" in main_window
    assert "return sidePanelComponent" in main_window
    assert "visible: false" in main_window
    assert "root.showOperatorDashboard()" in main_window
    assert "layoutController.setPanelVisibility(panelId, true)" in main_window
    assert "implicitWidth:" in dashboard
    assert "implicitHeight:" in dashboard
    assert "anchors.fill: parent" in dashboard
    assert "Layout.fillWidth: true" in dashboard


def test_qml_work_modes_has_demo_offline_placeholder() -> None:
    mode_wizard = (QML_SOURCE_ROOT / "views" / "ModeWizard.qml").read_text(encoding="utf-8")

    assert "Brak danych o profilach cloud" in mode_wizard
    assert "Tryb demo/offline" in mode_wizard
    assert "Live trading pozostaje wyłączony" in mode_wizard


def test_changed_ui_qml_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = _qml_text()

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def test_all_preview_menu_panels_have_visible_content_markers() -> None:
    source = _qml_text()

    required_markers = {
        "Dashboard operatora": 'objectName: "operatorDashboardTitle"',
        "Telemetria feedu": 'objectName: "telemetryFeedPreviewTitle"',
        "Decyzje governor": 'objectName: "aiDecisionsView"',
        "Diagnostyka": 'objectName: "diagnosticsPreviewTitle"',
        "Strumień decyzji": 'objectName: "decisionStreamPreviewTitle"',
        "Warsztat strategii": 'objectName: "strategyWorkbenchPreviewTitle"',
        "Strategie": 'objectName: "strategiesPreviewTitle"',
        "Kontrola ryzyka": 'objectName: "riskControlsPreviewTitle"',
        "Tryby pracy": 'objectName: "modeWizardPreviewPanel"',
        "Menedżer strategii": 'objectName: "strategyManagerPreviewTitle"',
    }
    for panel_name, marker in required_markers.items():
        assert panel_name in source
        assert marker in source


def test_strategy_and_risk_panels_use_styled_controls_for_dark_theme_contrast() -> None:
    strategies = (QML_SOURCE_ROOT / "views" / "Strategies.qml").read_text(encoding="utf-8")
    risk = (QML_SOURCE_ROOT / "views" / "RiskControls.qml").read_text(encoding="utf-8")

    assert "Components.StyledTextField" in strategies
    assert 'designSystem.color("textPrimary")' in strategies
    assert 'designSystem.color("textSecondary")' in strategies
    assert "Components.StyledSpinBox" in risk
    assert "Components.StyledTextField" in risk
    assert "Components.StyledSwitch" in risk
    assert "contentItem: Text" in risk
    assert 'designSystem.color("textPrimary")' in risk
    assert 'designSystem.color("textSecondary")' in risk


def test_preview_empty_states_have_demo_copy_and_styled_actions() -> None:
    manager = (QML_SOURCE_ROOT / "views" / "StrategyManager.qml").read_text(encoding="utf-8")
    workbench = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    mode_wizard = (QML_SOURCE_ROOT / "views" / "ModeWizard.qml").read_text(encoding="utf-8")

    for source in (manager, workbench, mode_wizard):
        assert "demo/offline" in source
        assert "Components.IconButton" in source

    assert "Marketplace unavailable" in manager
    assert "4 zdarzenia BTC/USDT" in manager
    assert "strategyWorkbenchPreviewPanel" in workbench
    assert "Brak live danych" in workbench
    assert "modeWizardPreviewPanel" in mode_wizard
    assert "Otwórz kreator" in mode_wizard


def test_preview_card_has_default_layout_content_container() -> None:
    preview_card = (QML_SOURCE_ROOT / "components" / "PreviewCard.qml").read_text(encoding="utf-8")

    assert "default property alias content: extraContent.data" in preview_card
    assert "id: extraContent" in preview_card
    assert 'objectName: "previewCardExtraContent"' in preview_card
    assert "Layout.fillWidth: true" in preview_card
    assert "implicitHeight: cardContent.implicitHeight + cardPadding * 2" in preview_card
    assert "property alias contentItem: cardContent" not in preview_card


def test_offscreen_smoke_audits_every_menu_panel_loads_non_empty() -> None:
    result = _run_ui_smoke()
    payload = _smoke_payload(result)

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["ui_loaded"] is True
    assert payload["qml_loaded"] is True
    assert payload["central_content_empty"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False

    panel_load_results = payload["panel_load_results"]
    assert isinstance(panel_load_results, dict)
    assert set(panel_load_results) == set(PANEL_AUDIT_IDS)
    for panel_id in PANEL_AUDIT_IDS:
        panel_result = panel_load_results[panel_id]
        assert panel_result["loaded"] is True
        assert panel_result["empty"] is False
        assert panel_result["visible"] is True
        assert panel_result["width"] > 0 or panel_result["implicitWidth"] > 0
        assert panel_result["height"] > 0 or panel_result["implicitHeight"] > 0
