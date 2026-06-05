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
    "get" + "env",
    "c" + "cxt",
)

PANEL_AUDIT_IDS = (
    "sidePanel",
    "aiCenterPanel",
    "tradingUniversePanel",
    "portfolioPerformancePanel",
    "terminalPanel",
    "strategiesPanel",
    "riskControlsPanel",
    "aiDecisionsPanel",
    "telemetryPanel",
    "diagnosticsPanel",
    "helpGlossaryPanel",
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

    exercise_options = AppOptions.parse(
        [
            "--config",
            "ui/config/preview_local.yaml",
            "--smoke",
            "--offscreen",
            "--exercise-preview-state",
        ]
    )
    assert exercise_options.exercise_preview_state is True


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
    assert payload["i18n_language_selector_present"] is True
    assert payload["i18n_pl_en_available"] is True
    assert payload["i18n_language_switch_local_only"] is True
    assert payload["help_glossary_present"] is True
    assert payload["glossary_required_terms_present"] is True
    assert payload["tooltips_present"] is True
    assert payload["safety_boundary_ok"] is True
    assert payload["portfolio_filters_do_not_mutate_paper_state"] is True
    assert payload["issues"] == []


def test_exercise_preview_state_smoke_mutates_local_state_only() -> None:
    result = _run_ui_smoke("--exercise-preview-state")
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["preview_state_exercised"] is True
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False

    audit = payload["preview_state_audit"]
    assert isinstance(audit, dict)
    assert audit["passed"] is True
    assert audit["smoke_only"] is True
    assert audit["network_api_calls"] == "disabled"
    assert audit["runtime_loop_started"] is False
    assert audit["live_trading_disabled"] is True
    assert audit["exchange_io_disabled"] is True
    assert audit["order_submission_disabled"] is True
    assert audit["api_keys_required"] is False
    assert audit["safety_boundary_ok"] is True
    assert payload["safety_boundary_ok"] is True
    assert payload["portfolio_filters_do_not_mutate_paper_state"] is True

    assert audit["start_sets_running"] is True
    assert audit["start_tick_delta"] >= 1
    assert audit["generate_tick_delta"] == 1
    assert audit["generate_tick_appended_order"] is True
    assert audit["generate_tick_appended_decision"] is True
    assert audit["generate_tick_appended_telemetry"] is True
    assert audit["run_ten_tick_delta"] == 10
    assert audit["pause_sets_paused"] is True
    assert audit["stop_sets_stopped"] is True
    assert audit["reset_sets_stopped"] is True
    assert audit["reset_ticks_zero"] is True
    assert audit["reset_clears_orders"] is True
    assert audit["governor_updates_decision"] is True
    assert audit["ping_appends_telemetry"] is True

    assert audit["select_top20_count"] == 20
    assert audit["select_top20_propagates_terminal_pair"] is True
    assert audit["select_all_visible_at_least_top20"] is True
    assert audit["clear_selected_pairs_zero"] is True
    assert audit["toggle_pair_selects_pair"] is True
    assert audit["toggle_pair_updates_terminal_pair"] is True
    assert audit["pair_selection_updates_decision_summary"] is True
    assert audit["risk_profile_updates"] is True
    assert audit["risk_summary_updates"] is True


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


def test_ui_preview_8_0d_live_like_paper_simulation_contract() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")
    terminal = (QML_SOURCE_ROOT / "views" / "PaperTerminal.qml").read_text(encoding="utf-8")
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")

    for token in (
        "property bool simulationRunning",
        "property bool simulationPaused",
        "property int simulationSpeed",
        "property int simulationTickIntervalMs",
        "property string simulationScenario",
        "property int simulationTickCount",
        "property string simulationLastTickAt",
        "property string simulationMarketMode",
        "property string simulationStatusLabel",
        "property var simulationEvents",
        "function startLiveLikePaperSimulation()",
        "function pauseLiveLikePaperSimulation()",
        "function stopLiveLikePaperSimulation()",
        "function resetLiveLikePaperSimulation()",
        "function runSimulationTick()",
        "function runSimulationBurst(count)",
        "id: simulationTimer",
    ):
        assert token in main_window

    for scenario in (
        "Balanced preview",
        "Bull trend",
        "Bear trend",
        "High volatility",
        "Sideways/range",
    ):
        assert scenario in main_window

    for control_contract in (
        "onClicked: previewState.startPaperPreview()",
        "onClicked: previewState.pausePaperPreview()",
        "onClicked: previewState.stopPaperPreview()",
        "onClicked: previewState.resetPaperPreview()",
        "onClicked: previewState.generatePaperTick()",
        "onClicked: previewState.runTenMockTicks()",
        "function startPaperPreview() { startLiveLikePaperSimulation() }",
        "function pausePaperPreview() { pauseLiveLikePaperSimulation() }",
        "function stopPaperPreview() { stopLiveLikePaperSimulation() }",
        "function resetPaperPreview() { resetLiveLikePaperSimulation() }",
    ):
        assert control_contract in main_window or control_contract in dashboard

    for safety_copy in (
        "Local paper loop only",
        "no exchange API",
        "no real orders",
        "no secrets",
        "production runtime loop not started",
        "no exchange I/O",
        "order submission disabled",
        "API keys not required",
    ):
        assert safety_copy in source

    for ui_marker in (
        "Simulation status",
        "Simulation speed / tick count",
        "Last simulated scan",
        "Safety boundary",
        "Market scenario",
        "Live-like paper simulation",
        "Paper loop local-only",
    ):
        assert ui_marker in dashboard

    for terminal_marker in (
        "previewState.mockTerminalCandles",
        "simulationMarketMode",
        "simulationTickCount",
        "live-like paper simulation",
        "Live-like paper simulation",
    ):
        assert terminal_marker in terminal

    for decision_marker in (
        "confidence, strategy/governor, reason, safety state",
        "paper order event:",
        "Strategy source / governor",
        "Safety state",
    ):
        assert decision_marker in decisions

    for tooltip in (
        "Live-like paper simulation",
        "Simulation speed",
        "Market scenario",
        "Paper loop",
        "No real orders",
    ):
        assert tooltip in main_window

    forbidden_tokens = (
        "create" + "_" + "order",
        "fetch" + "_" + "balance",
        "load" + "_" + "markets",
        "key" + "ring",
        "dot" + "env",
        "shell" + "=True",
        "subprocess" + "." + "run",
        "os" + "." + "environ",
        "get" + "env",
        "c" + "cxt",
    )
    scoped_source = "\n".join(
        path.read_text(encoding="utf-8") for path in [*_qml_sources(), SMOKE_SOURCE]
    )
    for token in forbidden_tokens:
        assert token not in scoped_source


def test_ai_decisions_view_preserves_timeline_count_contract() -> None:
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")

    assert "property int timelineCount" in decisions


def test_ui_preview_7_7_2_qml_standalone_guards_and_terminal_parse_patterns() -> None:
    terminal = (QML_SOURCE_ROOT / "views" / "PaperTerminal.qml").read_text(encoding="utf-8")
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")
    icon_button = (QML_SOURCE_ROOT / "components" / "IconButton.qml").read_text(encoding="utf-8")

    # Guard against the compact inline child-object syntax that broke QML parsing
    # with `Unexpected token ;` in PaperTerminal chart/order-book rows.
    assert re.search(r"ColumnLayout\s*\{[^}]*Label\s*\{[^}]*\}\s*;\s*Label", terminal) is None
    assert re.search(r"RowLayout\s*\{[^}]*Label\s*\{[^}]*\}\s*;\s*Label", terminal) is None

    for guard in (
        "function safeColor",
        "function previewValue",
        "function decisionRows",
        "function decisionFilterValue",
        "function decisionPairFilterValue",
        "function paperSessionStatusValue",
        "function lastGovernorDecisionValue",
        'root.previewValue("decisionPreviewRows", [])',
        'root.previewValue("paperSessionStatus", "stopped")',
        'root.previewValue("lastGovernorDecision", qsTr("No paper decision yet"))',
    ):
        assert guard in decisions

    for unguarded_pattern in (
        r"description:\s*previewState\.lastGovernorDecision",
        r"arg\(previewState\.paperSessionStatus\)",
        r"subtle:\s*previewState\.decisionFilter",
        r"subtle:\s*previewState\.decisionPairFilter",
        r"color:\s*designSystem\.color\(",
    ):
        assert re.search(unguarded_pattern, decisions) is None

    assert "function safeIconSource" in icon_button
    assert 'typeof control.designSystem.iconSource === "function"' in icon_button
    assert "designSystem.iconSource(control.iconName)" not in icon_button


def test_i18n_language_selector_help_glossary_and_tooltips_are_declared() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    icon_button = (QML_SOURCE_ROOT / "components" / "IconButton.qml").read_text(encoding="utf-8")

    for token in (
        "property string currentLanguage",
        "property var languageOptions",
        "function setLanguage(lang)",
        "function trText(key)",
        "function previewT(key)",
        "translationDictionary",
        'code: "PL"',
        'code: "EN"',
        'objectName: "languageSelector"',
        "🇵🇱 PL",
        "🌐 EN",
    ):
        assert token in main_window

    for token in (
        'panelId: "helpGlossaryPanel"',
        "Pomoc / Słownik",
        'objectName: "helpGlossaryRoot"',
        "glossaryCategories",
        "Trading",
        "Ryzyko",
        "AI / Governor",
        "Strategie",
        "Paper / Live",
        "Giełdy / API",
        "Diagnostyka",
    ):
        assert token in source

    for term in (
        "PnL",
        "ROI",
        "drawdown",
        "slippage",
        "spread",
        "order book",
        "paper trading",
        "sandbox/testnet",
        "API key",
        "governor",
        "confidence",
        "strategy",
        "risk guard",
        "kill-switch",
        "TP",
        "SL",
        "fee/prowizja",
        "equity",
        "available balance",
        "in positions",
        "blacklist",
        "whitelist",
    ):
        assert term in source

    assert "property string helpText" in icon_button
    assert "ToolTip.delay: 800" in icon_button
    assert "hovered || activeFocus" in icon_button
    for tooltip_key in (
        "Start Paper Preview",
        "Pause",
        "Stop",
        "Reset",
        "Generate Next Tick",
        "Run 10 paper ticks",
        "Generate governor recommendation",
        "Import markets preview",
        "Select top 20",
        "Blacklist selected",
        "Whitelist selected",
        "Simulate buy/sell order",
        "Risk profile Conservative",
        "Risk profile Balanced",
        "Risk profile Aggressive",
        "Custom risk",
        "AI recommended risk",
        "Custom range",
        "Zastosuj zakres",
    ):
        assert tooltip_key in source


def test_ui_preview_sources_have_no_forbidden_runtime_or_secret_tokens() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in _qml_sources())
    source += SMOKE_SOURCE.read_text(encoding="utf-8")

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


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
    strategy_manager = (QML_SOURCE_ROOT / "views" / "StrategyManager.qml").read_text(
        encoding="utf-8"
    )

    for source in (main_window, strategy_manager):
        assert "ScrollBar.vertical: ScrollBar" in source
        assert 'designSystem.color("surfaceElevated")' in source
        assert 'designSystem.color("border")' in source


def test_operator_dashboard_uses_stable_contrast_tokens_for_visible_statuses() -> None:
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    assert 'designSystem.color("textPrimary")' in dashboard
    assert 'designSystem.color("textSecondary")' in dashboard
    assert 'designSystem.color("success")' not in dashboard
    assert 'designSystem.color("positive")' not in dashboard
    assert "root.designSystem.color(modelData.accent)" not in dashboard


def test_qml_operator_dashboard_is_default_selected_panel() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert 'property string defaultPanelId: "sidePanel"' in main_window
    assert "property string currentPanelId: defaultPanelId" in main_window
    assert "showOperatorDashboard()" in main_window
    assert "layoutController.registerPanels(panelMetadata)" in main_window
    assert 'panelId: "sidePanel", title: qsTr("Dashboard")' in main_window
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
    assert "Dashboard" in dashboard
    assert "AI / Governor mode" in dashboard
    assert "Trading Universe" in source
    assert "Model readiness" in source
    assert "Decyzje" in source


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
    assert (
        'text: qsTr("Odśwież preview")' in source
        or 'text: root.trText("refresh.preview")' in source
    )


def test_qml_operator_preview_demo_offline_safety_copy() -> None:
    source = _qml_text()

    required_copy = (
        "Live trading disabled",
        "Exchange route disabled",
        "Order submission disabled",
        "Runtime loop not started",
        "API keys not required",
        "Preview only",
        "BTC/USDT HOLD",
        "ETH/USDT",
        "SOL/USDT",
        "Model readiness",
        "Safety kill-switch",
        "NO ORDER — preview only",
    )
    for text in required_copy:
        assert text in source


def test_qml_operator_dashboard_has_real_visible_central_component_and_tab_navigation() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    assert (
        '"sidePanel": { title: qsTr("Dashboard"), icon: "fingerprint", component: sidePanelComponent }'
        in main_window
    )
    assert "id: sidePanelComponent" in main_window
    assert "Views.OperatorDashboard" in main_window
    assert 'objectName: "centralContentRoot"' in main_window
    assert 'objectName: "centralContentLoader"' in main_window
    assert "sourceComponent: root.selectedPanelComponent()" in main_window
    assert "return sidePanelComponent" in main_window
    assert "visible: false" in main_window
    assert "showOperatorDashboard()" in main_window
    assert "layoutController.setPanelVisibility(panelId, true)" in main_window
    assert "implicitWidth:" in dashboard
    assert "implicitHeight:" in dashboard
    assert "Layout.fillWidth: true" in dashboard


def test_qml_work_modes_has_demo_offline_placeholder() -> None:
    mode_wizard = (QML_SOURCE_ROOT / "views" / "ModeWizard.qml").read_text(encoding="utf-8")

    assert "Brak danych o profilach cloud" in mode_wizard
    assert "Tryb demo/offline" in mode_wizard
    assert "Live trading pozostaje wyłączony" in mode_wizard


def test_product_preview_shell_has_tabs_and_required_preview_labels() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert 'objectName: "productPreviewTabBar"' in main_window
    assert "Menu {" not in main_window
    assert "MenuItem {" not in main_window
    for label in (
        "Dashboard",
        "AI Center",
        "Centrum autonomii",
        "Trading Universe",
        "Strategie",
        "Ryzyko",
        "Decyzje",
        "Telemetria",
        "Diagnostyka",
        "Binance",
        "Bybit",
        "OKX",
        "KuCoin",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "Model readiness",
        "Training/coverage",
        "Live trading disabled",
        "Exchange route disabled",
        "Order submission disabled",
        "API keys not required",
        "Runtime loop not started",
    ):
        assert label in source


def test_final_product_dashboard_ai_universe_risk_decision_telemetry_copy() -> None:
    source = _qml_text()

    for label in (
        "Bot status: Demo/Paper Preview",
        "AI/Governor status",
        "Active AI model / governor engine",
        "Model readiness %",
        "Training/coverage",
        "Autonomy level",
        "Selected exchanges",
        "Selected coins/pairs",
        "Active strategies",
        "Last AI/governor decision",
        "Risk state",
        "Decision Governor Preview Core",
        "Model family/type",
        "Model version/build",
        "Training/readiness percent",
        "Data coverage percent",
        "Current autonomy mode",
        "Market scanner",
        "Strategy governor",
        "Risk governor",
        "Execution guard",
        "Recovery monitor",
        "Telemetry monitor",
        "Kill-switch",
        "Market data status",
        "API key status",
        "Live trading status / Order route",
        "Max position",
        "Max open positions",
        "Stop loss",
        "Take profit",
        "Max slippage",
        "Max drawdown",
        "Opportunity governor mode",
        "Risk reason",
        "Strategy source",
        "Timestamp",
        "Safety block state",
        "Feed status",
        "Reconnects",
        "Downtime",
        "Last heartbeat",
        "Data freshness",
        "Telemetry heartbeat feed",
        "Preview diagnostics readiness",
    ):
        assert label in source


def test_product_preview_qml_has_no_native_primary_menu_or_unsafe_visual_tokens() -> None:
    source = _qml_text()

    forbidden_visual_tokens = (
        'designSystem.color("success")',
        'designSystem.color("positive")',
        'color: "black"',
        'color: "#000',
    )
    for token in forbidden_visual_tokens:
        assert token not in source


def test_final_product_panels_avoid_raw_native_form_controls() -> None:
    final_panel_sources = [
        QML_SOURCE_ROOT / "views" / "Strategies.qml",
        QML_SOURCE_ROOT / "views" / "RiskControls.qml",
        QML_SOURCE_ROOT / "views" / "TradingUniverse.qml",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in final_panel_sources)

    assert re.search(r"(^|\n)\s*TextField\s*\{", source) is None
    assert re.search(r"(^|\n)\s*SpinBox\s*\{", source) is None
    assert re.search(r"(^|\n)\s*ComboBox\s*\{", source) is None
    assert "Components.StyledTextField" in source
    assert "Components.StyledSpinBox" in source
    assert "Components.StyledSwitch" in source


def test_changed_ui_qml_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = _qml_text()

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def test_all_preview_menu_panels_have_visible_content_markers() -> None:
    source = _qml_text()

    required_markers = {
        "Dashboard": 'objectName: "operatorDashboardTitle"',
        "AI Center": 'objectName: "aiControlCenterTitle"',
        "Trading Universe": 'objectName: "tradingUniverseTitle"',
        "Strategie": 'objectName: "strategiesPreviewTitle"',
        "Ryzyko": 'objectName: "riskControlsTitle"',
        "Decyzje": 'objectName: "aiDecisionsTitle"',
        "Telemetria": 'objectName: "telemetryFeedPreviewTitle"',
        "Diagnostyka": 'objectName: "diagnosticsPreviewTitle"',
    }
    for panel_name, marker in required_markers.items():
        assert panel_name in source
        assert marker in source


def test_strategy_and_risk_panels_use_styled_controls_for_dark_theme_contrast() -> None:
    strategies = (QML_SOURCE_ROOT / "views" / "Strategies.qml").read_text(encoding="utf-8")
    risk = (QML_SOURCE_ROOT / "views" / "RiskControls.qml").read_text(encoding="utf-8")

    assert "Components.StyledSwitch" in strategies
    assert 'designSystem.color("textPrimary")' in strategies
    assert 'designSystem.color("textSecondary")' in strategies
    assert "riskProfileSegmentedControl" in risk
    assert "Components.StyledSpinBox" not in risk
    assert "Components.StyledTextField" not in risk
    assert "ComboBox" not in risk
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

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

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


def test_ui_preview_7_2_interactive_shell_source_contract() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert 'objectName: "productPreviewTabBar"' in main_window
    assert "Flickable" in main_window
    assert "flickableDirection: Flickable.HorizontalFlick" in main_window
    assert "Menu {" not in main_window
    assert "MenuItem {" not in main_window
    for tab_label in (
        "Dashboard",
        "AI Center",
        "Trading Universe",
        "Strategie",
        "Ryzyko",
        "Decyzje",
        "Telemetria",
        "Diagnostyka",
    ):
        assert tab_label in source

    for state_token in (
        "property var selectedExchanges",
        "property var selectedPairs",
        "property var activeStrategies",
        "property string paperSessionState",
        "property int paperTicks",
        "property var paperOrdersPreview",
        "property string lastGovernorDecision",
        "property string autonomyMode",
        "property int modelReadiness",
        "property int trainingCoverage",
        "property bool riskLocked",
        "property bool liveTradingDisabled: true",
        "property bool exchangeIoDisabled: true",
        "property bool orderSubmissionDisabled: true",
        "property bool apiKeysRequired: false",
        "property bool runtimeLoopStarted: false",
    ):
        assert state_token in main_window


def test_ui_preview_7_2_dashboard_paper_cockpit_controls() -> None:
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")

    for label in (
        "Paper session status",
        "Start Paper Preview",
        "Pause",
        "Stop",
        "Reset",
        "Generate Next Tick",
        "Run 10 paper ticks",
        "Paper session PnL / equity",
        "Paper order blotter",
        "Time",
        "Pair",
        "Action",
        "Status",
        "Confidence",
        "Reason",
        "order submission disabled",
    ):
        assert label in dashboard


def test_ui_preview_7_2_trading_universe_market_selector_contract() -> None:
    source = (QML_SOURCE_ROOT / "views" / "TradingUniverse.qml").read_text(encoding="utf-8")
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    for label in (
        "Import markets preview",
        "Search pair",
        "Select all visible",
        "Clear selected",
        "All",
        "USDT",
        "USDC",
        "BTC",
        "ETH",
        "Major",
        "AI candidates",
        "Excluded / blacklist",
        "Select top 20",
        "Whitelist selected",
        "Blacklist selected",
    ):
        assert label in source

    preview_pairs = re.findall(r'"[A-Z0-9]+/(?:USDT|USDC|BTC|ETH)"', main_window)
    assert len(set(preview_pairs)) >= 100

    for exchange in (
        "Binance",
        "Bybit",
        "OKX",
        "KuCoin",
        "Coinbase",
        "Kraken",
        "Bitget",
        "Gate.io",
        "MEXC",
        "Paper Preview Catalog",
    ):
        assert exchange in main_window


def test_ui_preview_7_2_ai_strategies_risk_decisions_telemetry_diagnostics_contract() -> None:
    source = _qml_text()
    strategies = (QML_SOURCE_ROOT / "views" / "Strategies.qml").read_text(encoding="utf-8")

    for label in (
        "Autonomy and policy controls",
        "Model readiness",
        "Training/coverage",
        "Data coverage percent",
        "Generate governor recommendation",
        "confidence threshold",
        "Market scanner",
        "Strategy governor",
        "Risk governor",
        "Execution guard",
        "Recovery monitor",
        "Telemetry monitor",
        "Kill-switch",
    ):
        assert label in source

    for strategy_name in (
        "Momentum Guard",
        "Range Guard",
        "Volatility Breakout Preview",
        "Mean Reversion Preview",
        "Trend Follow Preview",
        "Liquidity Sweep Preview",
    ):
        assert strategy_name in strategies
    assert strategies.count("Components.StyledSwitch") >= 1

    for label in (
        "Daily loss limit",
        "Risk profile segmented control",
        "Conservative",
        "Balanced",
        "Aggressive",
        "Generate next decision",
        "action",
        "confidence",
        "reason:",
        "Risk reason",
        "Ping feed",
        "Last heartbeat",
        "Generate diagnostic bundle",
        "Included",
        "Excluded",
        "secrets • env files • keychain • real environment values • exchange state",
    ):
        assert label in source


def test_ui_preview_7_2_smoke_contract_fields_are_reported() -> None:
    result = _run_ui_smoke()
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["final_preview_tabs_loaded"] is True
    assert payload["paper_session_controls_present"] is True
    assert payload["market_universe_controls_present"] is True
    assert payload["ai_governor_controls_present"] is True


def test_ui_preview_7_6_paper_terminal_source_contract() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    terminal_path = QML_SOURCE_ROOT / "views" / "PaperTerminal.qml"
    terminal = terminal_path.read_text(encoding="utf-8")

    assert terminal_path.exists()
    assert 'objectName: "paperTerminalRoot"' in terminal
    assert "Paper Terminal" in terminal or "Terminal" in terminal
    assert 'panelId: "terminalPanel", title: qsTr("Paper Terminal")' in main_window
    assert '"terminalPanel": { title: qsTr("Paper Terminal")' in main_window
    assert "terminalPanelComponent" in main_window

    for token in (
        "Order Form",
        "BUY",
        "SELL",
        "LIMIT",
        "MARKET",
        "Order Book",
        "Positions",
        "Orders",
        "History",
        "Reserved",
        "Strategy",
        "Log",
        "Messages",
    ):
        assert token in terminal

    for token in (
        "Paper Preview only",
        "Live trading disabled",
        "Exchange I/O disabled",
        "Order submission disabled",
        "API keys not required",
        "Runtime loop not started",
        "No real orders",
    ):
        assert token in terminal

    for token in (
        "Selektor pary",
        "paperTerminalPairSelector",
        "paperTerminalPairSearchInput",
        "Search pair",
        "aktywna para",
        "fallbackiem do previewMarketPairs",
        "Aktywny timeframe jest stanem lokalnym",
        "terminalTimeframe",
        "setTerminalTimeframe",
        "available balance preview",
        "fee estimate preview",
        "order value preview",
        "TP preview",
        "SL preview",
        "post-only local",
        "reduce-only local",
        "time-in-force GTC",
    ):
        assert token in terminal or token in main_window

    assert main_window.count('action: "Use ask"') >= 10
    assert main_window.count('action: "Use bid"') >= 10

    for token in (
        "selectedTerminalPair",
        "terminalSide",
        "terminalOrderType",
        "terminalPrice",
        "terminalAmount",
        "terminalTotal",
        "terminalAutoConfirm",
        "terminalSelectedBottomTab",
        "mockOrderBookAsks",
        "mockOrderBookBids",
        "mockTerminalPositions",
        "mockTerminalOrders",
        "mockTerminalHistory",
        "terminalLogRows",
        "mockTerminalReservedBalances",
        "terminalPairSearch",
        "terminalTimeframe",
        "function terminalPairCandidates",
        "function setTerminalTimeframe",
        "function setTerminalPair",
        "function setTerminalSide",
        "function setTerminalOrderType",
        "function setTerminalPrice",
        "function setTerminalAmount",
        "function applyTerminalPercent",
        "function simulateTerminalOrder",
        "function selectTerminalBottomTab",
        "function useOrderBookPrice",
    ):
        assert token in main_window

    assert "Lokalny wykres preview" in terminal
    assert "BTC/USDT" in main_window
    assert (
        'selectedPairs && selectedPairs.length > 0 ? selectedPairs[0] : "BTC/USDT"' in main_window
    )
    assert "preview-only local catalog" in source
    assert "Paper Preview only" in terminal

    forbidden_terminal_tokens = (
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
        "os.environ",
        "getenv",
        "keyring",
        "dotenv",
        "subprocess",
        "shell=True",
    )
    for token in forbidden_terminal_tokens:
        assert token not in terminal


def test_ui_preview_7_7_local_paper_bridge_state_contract() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    terminal = (QML_SOURCE_ROOT / "views" / "PaperTerminal.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")

    required_state_tokens = (
        "paperSessionStatus",
        "paperEquity",
        "paperPnl",
        "paperSessionTicks",
        "paperOrdersCount",
        "paperBlockedCount",
        "paperNoOrderCount",
        "paperSimulatedCount",
        "paperOrderRows",
        "paperTelemetryRows",
        "paperOpenPositions",
        "paperClosedTrades",
        "validatePaperOrderPreview",
        "appendPaperTelemetry",
        "appendPaperDecision",
        "submitLocalPaperOrder",
        "updatePaperPositionPreview",
        "updatePaperEquityPreview",
    )
    for token in required_state_tokens:
        assert token in main_window

    for token in (
        "local-only paper bridge/state",
        "Paper Preview only",
        "Live trading disabled",
        "Exchange I/O disabled",
        "Order submission disabled",
        "API keys not required",
        "Runtime loop not started",
        "No real orders",
    ):
        assert token in main_window or token in terminal or token in dashboard or token in decisions

    for token in (
        "previewState.paperSessionStatus",
        "previewState.paperEquity",
        "previewState.paperPnl",
        "previewState.paperSessionTicks",
        "previewState.paperOrderRows",
    ):
        assert token in dashboard

    for token in (
        "previewState.decisionPreviewRows",
        "Simulate terminal order",
        "Generate next decision",
        "Generate governor recommendation",
        "previewState.paperSessionStatus",
    ):
        assert token in decisions or token in main_window

    for token in (
        "root.paperTelemetryRows",
        "root.paperSessionTicks",
        "last paper event",
        "bounded list 8–12 rows",
        "exchange I/O disabled",
        "runtime loop not started",
    ):
        assert token in main_window

    for token in (
        "previewState.paperOpenPositions",
        "previewState.paperOrderRows",
        "previewState.paperClosedTrades",
        "previewState.terminalLogRows",
        "Safety/system messages",
    ):
        assert token in terminal

    guarded_sources = "\n".join((main_window, terminal, dashboard, decisions))
    forbidden_tokens = (
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
        "os.environ",
        "getenv",
        "keyring",
        "dotenv",
        "subprocess",
        "shell=True",
    )
    for token in forbidden_tokens:
        assert token not in guarded_sources


def test_ui_preview_7_4_product_ux_source_contract() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")
    universe = (QML_SOURCE_ROOT / "views" / "TradingUniverse.qml").read_text(encoding="utf-8")
    ai_center = (QML_SOURCE_ROOT / "views" / "AiControlCenter.qml").read_text(encoding="utf-8")
    risk = (QML_SOURCE_ROOT / "views" / "RiskControls.qml").read_text(encoding="utf-8")
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")

    assert "Menu {" not in main_window
    assert "MenuItem {" not in main_window
    assert 'objectName: "productPreviewTabBar"' in main_window

    raw_marker_patterns = (
        r'text:\s*qsTr\("[|▌▍❘]\\s*(Dashboard|Trading Universe|Strategie|Decyzje|Telemetria)',
        r'text:\s*"[|▌▍❘]\\s*(Dashboard|Trading Universe|Strategie|Decyzje|Telemetria)',
        r'color:\s*"white"[\s\S]{0,120}objectName:\s*".*TitleAccentBar"',
    )
    for pattern in raw_marker_patterns:
        assert re.search(pattern, source) is None
    for accent in (
        "operatorDashboardTitleAccentBar",
        "tradingUniverseTitleAccentBar",
        "strategiesTitleAccentBar",
        "aiDecisionsTitleAccentBar",
        "telemetryTitleAccentBar",
    ):
        assert accent in source

    preview_pairs = re.findall(r'"[A-Z0-9]+/(?:USDT|USDC|BTC|ETH)"', main_window)
    assert len(set(preview_pairs)) >= 100

    for token in (
        "marketQuoteFilter",
        "marketCategoryFilter",
        "quote filter: USDT, USDC, BTC, ETH",
        "category filter: Major, AI, Meme, DeFi, Layer1, Layer2, High volume",
        "Select all visible",
        "Select top 20",
        "Whitelist selected",
        "Blacklist selected",
        "Exchange market import flow",
        "preview-only local catalog",
        "paper/testserver trading",
    ):
        assert token in universe or token in main_window

    for token in (
        "Time",
        "Pair",
        "Action",
        "Status",
        "Confidence",
        "Reason",
        "status chips",
        "action chips",
    ):
        assert token in dashboard

    for token in (
        "ProgressBar",
        "Readiness badge",
        "Autonomy level",
        "policy selector",
        "confidence threshold",
    ):
        assert token in ai_center

    assert "riskProfileSegmentedControl" in risk
    for token in ("Conservative", "Balanced", "Aggressive"):
        assert token in risk

    for token in ("Decision filters", "all", "paper", "blocked", "no-order", "selected pair"):
        assert token in decisions

    for token in ("property int telemetryTick", "heartbeat/tick source state", "freshness status"):
        assert token in source

    for token in ("Preview diagnostics readiness", "included", "excluded", "Safety boundary"):
        assert token in source

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def test_ui_preview_7_5_final_polish_clickable_preview_state_contract() -> None:
    source = _qml_text()
    ai_center = (QML_SOURCE_ROOT / "views" / "AiControlCenter.qml").read_text(encoding="utf-8")
    universe = (QML_SOURCE_ROOT / "views" / "TradingUniverse.qml").read_text(encoding="utf-8")
    strategies = (QML_SOURCE_ROOT / "views" / "Strategies.qml").read_text(encoding="utf-8")
    risk = (QML_SOURCE_ROOT / "views" / "RiskControls.qml").read_text(encoding="utf-8")
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert "Components.StyledProgressBar" in ai_center
    assert re.search(r"(^|\n)\s*ProgressBar\s*\{", ai_center) is None
    for token in (
        "activeGovernorEngine",
        "modelVersionBuild",
        "decision policy",
        "confidence threshold",
        "cyan fill",
        "rounded corners",
    ):
        assert token in source

    for token in (
        "choose exchange -> sandbox/testnet/API planned/disabled -> import markets preview -> AI scans eligible pairs -> paper/testserver route planned/disabled",
        "total pairs",
        "visible pairs",
        "selected pairs",
        "whitelisted",
        "blacklisted",
        "AI candidates",
        "no real API calls",
    ):
        assert token in universe

    for token in (
        "local Save Preview action",
        "confidence floor",
        "cooldown",
        "timeframe",
        "max allocation",
        "allowed pairs count",
        "risk profile",
    ):
        assert token in strategies

    for token in (
        "Per-symbol exposure",
        "paper bridge not connected/planned",
        "live disabled",
        "exchange route disabled",
        "order submission disabled",
    ):
        assert token in risk

    for token in (
        "included: UI state, telemetry snapshot, governor rows, config preview metadata",
        "excluded: secrets, env files, keychain, real environment values, exchange state",
        "zero real network/API calls",
        "property string lastStrategySaveStatus",
        "function visiblePairsCount",
    ):
        assert token in main_window


def test_ui_preview_7_8_shared_state_interactivity_and_responsive_terminal_contract() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    terminal = (QML_SOURCE_ROOT / "views" / "PaperTerminal.qml").read_text(encoding="utf-8")
    dashboard = (QML_SOURCE_ROOT / "views" / "OperatorDashboard.qml").read_text(encoding="utf-8")
    ai_center = (QML_SOURCE_ROOT / "views" / "AiControlCenter.qml").read_text(encoding="utf-8")
    universe = (QML_SOURCE_ROOT / "views" / "TradingUniverse.qml").read_text(encoding="utf-8")
    strategies = (QML_SOURCE_ROOT / "views" / "Strategies.qml").read_text(encoding="utf-8")
    risk = (QML_SOURCE_ROOT / "views" / "RiskControls.qml").read_text(encoding="utf-8")
    decisions = (QML_SOURCE_ROOT / "views" / "AiDecisionsView.qml").read_text(encoding="utf-8")
    source = "\n".join(
        (main_window, terminal, dashboard, ai_center, universe, strategies, risk, decisions)
    )

    for component_token in (
        "Views.OperatorDashboard {\n            previewState: root",
        "Views.AiControlCenter {\n            previewState: root",
        "Views.TradingUniverse {\n            previewState: root",
        "Views.PaperTerminal {\n            previewState: root",
        "Views.AiDecisionsView {\n            previewState: root",
        "root.paperTelemetryRows",
        "root.paperSessionTicks",
        "previewState.paperSessionStatus",
        "previewState.decisionPreviewRows",
        "previewState.paperOrderRows",
        "previewState.activeStrategies",
        "previewState.riskProfile",
    ):
        assert component_token in source

    handlers = {
        "Start Paper Preview": "previewState.startPaperPreview()",
        "Pause": "previewState.pausePaperPreview()",
        "Stop": "previewState.stopPaperPreview()",
        "Reset": "previewState.resetPaperPreview()",
        "Generate Next Tick": "previewState.generatePaperTick()",
        "Run 10 paper ticks": "previewState.runTenMockTicks()",
        "Generate governor recommendation": "generateGovernorRecommendation",
        "Generate next decision": "generateNextDecision",
        "Ping feed": "pingTelemetryFeed()",
        "Generate diagnostic bundle": "generateDiagnosticBundle()",
        "Select all visible": "selectAllVisiblePairs()",
        "Clear selected": "clearSelectedPairs()",
        "Select top 20": "selectTop20Pairs()",
        "Blacklist selected": "blacklistSelectedPairs()",
        "Whitelist selected": "whitelistSelectedPairs()",
        "Save Preview": "saveStrategyPreview",
        "riskProfileSegmentedControl": "setRiskProfile",
    }
    for label, handler in handlers.items():
        assert label in source
        assert handler in source

    for token in (
        "function syncUniverseSelectionState",
        "function ensureSelectedTerminalPair",
        "selected pairs update Dashboard, Decisions and Paper Terminal",
        "Import markets preview complete; AI scans eligible local pairs only",
        "Exchange toggle preview",
        "sandbox/testnet/API planned/disabled",
        "marketAiCandidatesOnly = true",
    ):
        assert token in main_window

    preview_pairs = re.findall(r'"[A-Z0-9]+/(?:USDT|USDC|BTC|ETH)"', main_window)
    assert len(set(preview_pairs)) >= 100

    for safety_token in (
        "Live trading disabled",
        "Exchange I/O disabled",
        "Order submission disabled",
        "API keys not required",
        "no network/API call",
        "No real orders",
        "no real order / paper simulation only",
    ):
        assert safety_token in source

    for layout_token in (
        'objectName: "paperTerminalOrderForm"',
        'objectName: "paperTerminalChartArea"',
        'objectName: "paperTerminalOrderBook"',
        'objectName: "paperTerminalOrderBookScroll"',
        "ScrollBar.horizontal.policy: ScrollBar.AlwaysOff",
        'objectName: "paperTerminalResponsiveCockpitGrid"',
        "readonly property int cockpitColumns",
        "columns: root.cockpitColumns",
        "orderFormPreferredWidth",
        "chartPreferredWidth",
        "orderBookPreferredWidth",
        "orderBookScrollHeight",
        "Positions / Orders / History / Reserved / Strategy / Log / Messages",
    ):
        assert layout_token in terminal

    assert "Menu {" not in main_window
    assert "MenuItem {" not in main_window
    assert 'objectName: "productPreviewTabBar"' in main_window
    assert "Flickable.HorizontalFlick" in main_window
    assert "root.showPanel(modelData.panelId)" in main_window
    assert "active ? 2 : 1" in main_window


def test_ui_preview_8_0b_portfolio_performance_source_contract() -> None:
    source = _qml_text()
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    portfolio = (QML_SOURCE_ROOT / "views" / "PortfolioPerformance.qml").read_text(encoding="utf-8")

    assert 'panelId: "portfolioPerformancePanel", title: qsTr("Portfel / Wyniki")' in main_window
    assert '"portfolioPerformancePanel": { title: qsTr("Portfel / Wyniki")' in main_window
    assert "Views.PortfolioPerformance" in main_window
    assert 'objectName: "portfolioPerformanceRoot"' in portfolio
    assert 'objectName: "portfolioPerformanceTiles"' in portfolio

    for label in (
        "Portfel / Wyniki",
        "Stan konta",
        "Fiat balance / equity",
        "Trading balance / equity",
        "Available balance",
        "In positions",
        "Reserved / margin preview",
        "Ostatni cykl transakcyjny",
        "Cycle id / timestamp",
        "Cycle PnL",
        "Cycle trades count",
        "Winners / losers",
        "Net result",
        "Bieżąca sesja Paper",
        "Paper session equity",
        "Paper session PnL",
        "Paper session ticks",
        "Simulated orders",
        "Blocked / no-order counts",
        "Wynik całkowity",
        "All-time PnL",
        "Realized PnL",
        "Unrealized PnL",
        "Fees total",
        "Net PnL",
        "ROI % preview",
        "Filtry czasu",
        "Custom range",
        "Zastosuj zakres",
        "Tabela wyników / cykle",
        "Time",
        "Pair/Cycle",
        "Trades",
        "Gross PnL",
        "Fees",
        "Result",
        "TP",
        "SL",
        "AI exit",
        "manual preview",
        "risk guard",
    ):
        assert label in source

    for time_filter in ('"1h"', '"1d"', '"7d"', '"1m"', '"1y"', '"all"', '"custom"'):
        assert time_filter in main_window

    for state_token in (
        "property string portfolioBaseCurrency",
        "property real portfolioStartingEquityUsd",
        "property real portfolioTotalEquityUsd",
        "property real portfolioAvailableBalanceUsd",
        "property real portfolioInPositionsUsd",
        "property real portfolioReservedMarginUsd",
        "property real portfolioTradingEquityUsdt",
        "property real portfolioRealizedPnlUsd",
        "property real portfolioUnrealizedPnlUsd",
        "property real portfolioSessionPnlUsd",
        "property real portfolioLastCyclePnlUsd",
        "property string portfolioLastCycleId",
        "property string portfolioLastCycleTimestamp",
        "property int portfolioLastCycleTradesCount",
        "property int portfolioLastCycleWinners",
        "property int portfolioLastCycleLosers",
        "property real portfolioLastCycleFeesUsd",
        "property real portfolioLastCycleNetUsd",
        "property real portfolioAllTimePnlUsd",
        "property real portfolioRoiPercent",
        "property real portfolioFeesUsd",
        "property real portfolioFundingOtherCostsUsd",
        "property real portfolioNetPnlUsd",
        "property int portfolioTradeCount",
        "property string portfolioWinRate",
        "property string portfolioMaxDrawdown",
        "property string portfolioBestPair",
        "property string portfolioWorstPair",
        "property var portfolioCycleRows",
        "property var portfolioRangeSnapshots",
        "function setPortfolioTimeRange",
        "function applyPortfolioCustomRange",
        "function syncPortfolioPerformanceState",
        "function recomputePortfolioTotals",
        "function applyPortfolioSnapshot",
    ):
        assert state_token in main_window

    assert "previewState.setPortfolioTimeRange(modelData)" in portfolio
    assert (
        "previewState.applyPortfolioCustomRange(customFromField.text, customToField.text)"
        in portfolio
    )
    assert "Portfolio/Wyniki to preview/report state" in portfolio
    assert "Live trading disabled" in portfolio
    assert "Time filters do not alter active Paper session" in portfolio
    assert "runtime loop not started" in portfolio
    assert "exchange I/O disabled" in portfolio
    assert "order submission disabled" in portfolio
    assert "API keys not required" in portfolio
    assert "no secrets/env/keychain reads" in portfolio


def test_ui_preview_8_0b_smoke_audits_portfolio_state_and_safety() -> None:
    result = _run_ui_smoke("--exercise-preview-state")
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False

    panel_load_results = payload["panel_load_results"]
    assert panel_load_results["portfolioPerformancePanel"]["loaded"] is True
    assert panel_load_results["portfolioPerformancePanel"]["empty"] is False

    audit = payload["preview_state_audit"]
    assert audit["portfolio_fields_present"] is True
    assert audit["portfolio_filters_count"] == 7
    assert audit["portfolio_cycles_count"] >= 4
    assert audit["portfolio_cards_count"] >= 13
    assert audit["portfolio_custom_filter_updates_label"] is True
    assert audit["portfolio_custom_range_updates_report_state"] is True
    assert audit["portfolio_equity_formula_ok"] is True
    assert audit["portfolio_net_pnl_formula_ok"] is True
    assert audit["portfolio_no_double_count_ok"] is True
    assert audit["portfolio_range_snapshot_changes_values"] is True
    assert audit["portfolio_time_filter_does_not_mutate_paper_state"] is True
    assert audit["portfolio_time_filter_updates_report_state"] is True
    assert audit["portfolio_custom_filter_does_not_mutate_paper_state"] is True
    assert audit["paper_tick_updates_operational_state"] is True
    assert audit["paper_tick_can_update_financial_state_when_unblocked"] is True
    assert audit["risk_blocked_tick_does_not_mutate_paper_pnl"] is True
    assert audit["risk_blocked_tick_does_not_mutate_paper_equity"] is True
    assert audit["risk_blocked_tick_increments_blocked_count"] is True
    assert audit["risk_blocked_tick_appends_decision"] is True
    assert audit["risk_blocked_tick_appends_telemetry"] is True
    assert audit["risk_blocked_tick_creates_no_filled_order"] is True
    assert audit["risk_unlocked_tick_can_update_financial_state"] is True
    assert audit["dashboard_separates_paper_and_portfolio_report"] is True
    assert audit["portfolio_money_formatting_ok"] is True
    assert audit["safety_boundary_ok"] is True
    assert payload["safety_boundary_ok"] is True
    assert payload["portfolio_filters_do_not_mutate_paper_state"] is True
