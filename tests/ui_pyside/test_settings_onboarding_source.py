"""Source audits for UI-PREVIEW-8.0I settings and onboarding polish."""

from __future__ import annotations

import re
from pathlib import Path

QML_ROOT = Path("ui/pyside_app/qml")
MAIN_WINDOW = QML_ROOT / "MainWindow.qml"
DASHBOARD = QML_ROOT / "views" / "OperatorDashboard.qml"
SMOKE = Path("ui/pyside_app/smoke.py")


def _source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(QML_ROOT.rglob("*.qml")))


def test_settings_tab_state_functions_and_panel_exist() -> None:
    main_window = MAIN_WINDOW.read_text(encoding="utf-8")
    for token in (
        'panelId: "settingsPanel", title: qsTr("Ustawienia")',
        '"nav.settings": "Ustawienia"',
        '"nav.settings": "Settings"',
        "id: settingsPanelComponent",
        'objectName: "settingsPreviewPanel"',
        "settingsPanelOpen",
        "appModePreview",
        "baseCurrency",
        "uiDensity",
        "themeModePreview",
        "defaultPreviewExchange",
        "defaultTerminalPair",
        "defaultRiskProfile",
        "settingsDirty",
        "settingsLastUpdatedAt",
        "settingsSafetySummary",
        "function setAppModePreview(mode)",
        "function setBaseCurrency(currency)",
        "function setUiDensity(density)",
        "function setThemeModePreview(mode)",
        "function setDefaultPreviewExchange(exchange)",
        "function setDefaultTerminalPair(pair)",
        "function setDefaultRiskProfile(profile)",
        "function applyPreviewSettings()",
        "function resetPreviewSettings()",
        "function resetLocalPreviewState()",
    ):
        assert token in main_window


def test_onboarding_state_steps_functions_and_safety_copy_exist() -> None:
    main_window = MAIN_WINDOW.read_text(encoding="utf-8")
    for token in (
        "firstRunWizardVisible",
        "onboardingStep",
        "onboardingCompletedPreview",
        "function startOnboardingPreview()",
        "function nextOnboardingStep()",
        "function previousOnboardingStep()",
        "function completeOnboardingPreview()",
        "function skipOnboardingPreview()",
        "Wybierz język",
        "Wybierz walutę bazową",
        "Wybierz tryb",
        "Wybierz giełdę preview",
        "Wybierz profil ryzyka",
        "Uruchom Paper Preview / przejdź do Dashboard",
        "Settings are local preview only",
        "No runtime config is written",
        "No secrets are read",
        "No exchange/API calls",
        "No order submission",
        "Live trading remains disabled",
        "Ustawienia działają lokalnie w preview",
        "Konfiguracja runtime nie jest zapisywana",
        "Sekrety nie są odczytywane",
        "Brak połączeń giełda/API",
        "Brak składania zleceń",
        "Live trading pozostaje wyłączony",
    ):
        assert token in main_window


def test_global_status_bar_quick_actions_tooltips_and_glossary_exist() -> None:
    source = _source()
    dashboard = DASHBOARD.read_text(encoding="utf-8")
    for token in (
        'objectName: "globalAppStatusBar"',
        'objectName: "globalSafetyBadges"',
        "Mode: ",
        "Live trading: disabled",
        "Exchange I/O: disabled",
        "Order submission: disabled",
        "API keys: not required",
        "Runtime loop: not started",
        "Safety: safe preview",
        "Alerts: ",
        "Lang: ",
        "Base: ",
        "Risk: ",
        "Simulation: ",
    ):
        assert token in source
    for action in (
        "Szybkie akcje / Quick actions",
        "Start Paper Preview",
        "Pause",
        "Stop",
        "Run 10 ticks",
        "Start Scanner",
        "AI Recommended Risk",
        "Open Alerts",
        "Open Settings",
        "Open Help",
        "Generate Diagnostic Bundle",
    ):
        assert action in dashboard
    for term in (
        "Settings",
        "Onboarding",
        "Demo Preview",
        "Paper Preview",
        "Sandbox planned",
        "Live disabled",
        "Base currency",
        "UI density",
        "App mode",
        "Local preview state",
        "Reset local preview state",
        "App mode selector",
        "Base currency selector",
        "UI density selector",
        "Theme preview",
        "Apply preview settings",
        "Start onboarding",
        "Complete onboarding",
        "Open Settings",
        "Open Alerts",
        "Open Help",
    ):
        assert term in source


def test_top_navigation_order_unique_with_settings_and_scrollable() -> None:
    main_window = MAIN_WINDOW.read_text(encoding="utf-8")
    expected_order = {
        "sidePanel": 0,
        "aiCenterPanel": 1,
        "tradingUniversePanel": 2,
        "marketScannerPanel": 3,
        "portfolioPerformancePanel": 4,
        "terminalPanel": 5,
        "strategiesPanel": 6,
        "riskControlsPanel": 7,
        "aiDecisionsPanel": 8,
        "telemetryPanel": 9,
        "alertsPanel": 10,
        "diagnosticsPanel": 11,
        "settingsPanel": 12,
        "helpGlossaryPanel": 13,
    }
    matches = re.findall(r'panelId: "([^"]+)",[^\n]+defaultOrder: (\d+)', main_window)
    panel_orders = {panel_id: int(order) for panel_id, order in matches}

    assert panel_orders == expected_order
    assert len(set(panel_orders.values())) == len(panel_orders)
    assert 'objectName: "productPreviewTabBar"' in main_window
    assert "Flickable" in main_window
    assert "HorizontalFlick" in main_window


def test_smoke_contract_has_settings_onboarding_fields() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")
    for field in (
        "settings_tab_present",
        "settings_state_present",
        "settings_apply_local_only",
        "settings_reset_local_only",
        "settings_no_runtime_config_write",
        "settings_no_secret_reads",
        "app_status_bar_present",
        "app_mode_preview_present",
        "onboarding_state_present",
        "onboarding_steps_present",
        "onboarding_next_previous_works",
        "onboarding_complete_local_only",
        "top_navigation_order_unique_with_settings",
        "top_navigation_scroll_or_compact_present",
        "dashboard_quick_actions_present",
        "global_safety_badges_present",
        "settings_safety_boundary_ok",
    ):
        assert field in smoke
