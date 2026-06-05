"""Source-level checks for UI Preview Alert Center timeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"


def _source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(QML_ROOT.rglob("*.qml")))


def test_alert_center_state_and_functions_exist() -> None:
    source = _source()
    for token in (
        "alertCenterOpen",
        "alertRows",
        "alertUnreadCount",
        "alertCriticalCount",
        "alertWarningCount",
        "alertInfoCount",
        "alertSelectedSeverity",
        "alertSelectedCategory",
        "alertLastEventAt",
        "alertMutedPreview",
        "alertSoundEnabledPreview",
        "alertDesktopNotificationsPreview",
        "alertSelectedEvent",
        "alertEventExplanation",
        "function appendPreviewAlert(severity, category, title, message, source, pair, action)",
        "function markAlertRead(index)",
        "function markAllAlertsRead()",
        "function clearPreviewAlerts()",
        "function setAlertSeverityFilter(severity)",
        "function setAlertCategoryFilter(category)",
        "function selectAlertEvent(index)",
        "function explainAlertEvent(index)",
        "function toggleAlertMutePreview()",
        "function toggleAlertSoundPreview()",
        "function toggleDesktopNotificationsPreview()",
    ):
        assert token in source


def test_alert_center_tab_filters_categories_and_detail_exist() -> None:
    source = _source()
    for token in (
        "alertsPanel",
        "Alerty",
        "Alerts",
        "alertCenterRoot",
        "alertSeverityFilters",
        "All",
        "Critical",
        "Warning",
        "Info",
        "alertCategoryFilters",
        "Trading",
        "Risk",
        "AI",
        "Scanner",
        "Paper",
        "Portfolio",
        "Telemetry",
        "Diagnostics",
        "Safety",
        "alertCenterEventList",
        "time • severity • category • source • pair • title • message • action • status/read",
        "alertCenterDetailPanel",
        "Wyjaśnij zdarzenie",
        "Explain event",
    ):
        assert token in source


def test_required_alert_types_and_dashboard_summary_exist() -> None:
    source = _source()
    for token in (
        "AI decision generated",
        "Scanner candidate found",
        "Scanner rejected setup",
        "Risk blocked action",
        "Kill-switch active",
        "Paper order simulated",
        "Paper order blocked",
        "PnL changed",
        "Drawdown warning",
        "Risk profile changed",
        "AI Recommended risk applied",
        "Portfolio range changed",
        "Telemetry heartbeat stale",
        "Telemetry heartbeat fresh",
        "Diagnostic bundle generated",
        "Safety boundary reminder",
        "operatorDashboardAlertSummary",
        "unread alerts",
        "critical count",
        "last alert",
        "Otwórz Alerty",
    ):
        assert token in source


def test_alert_center_glossary_tooltips_and_safety_copy_exist() -> None:
    source = _source()
    for token in (
        "Alert Center",
        "Critical alert",
        "Warning alert",
        "Info alert",
        "unread",
        "event timeline",
        "muted alerts",
        "desktop notification preview",
        "stale heartbeat",
        "drawdown warning",
        "Mark all read",
        "Clear alerts",
        "Mute alerts",
        "Sound preview",
        "Desktop notification preview",
        "Explain event",
        "Severity filter",
        "Category filter",
        "Alerts are local preview only",
        "No OS notifications sent",
        "No backend calls",
        "No exchange/API calls",
        "No order submission",
        "No secrets read",
        "Alerty działają lokalnie w preview",
        "Brak systemowych powiadomień OS",
        "Brak wywołań backendu",
        "Brak połączeń giełda/API",
        "Brak składania zleceń",
        "Brak odczytu sekretów",
    ):
        assert token in source


def test_alert_smoke_fields_present() -> None:
    smoke = SMOKE_SOURCE.read_text(encoding="utf-8")
    for token in (
        "alerts_state_present",
        "alerts_tab_present",
        "alerts_append_increments_unread",
        "alerts_mark_read_works",
        "alerts_mark_all_read_works",
        "alerts_clear_works",
        "alerts_filters_present",
        "alerts_categories_present",
        "alerts_detail_present",
        "alerts_explain_event_local_only",
        "alerts_dashboard_summary_present",
        "alerts_simulation_tick_appends_event",
        "alerts_scanner_tick_appends_event",
        "alerts_risk_block_appends_event",
        "alerts_no_os_notifications",
        "alerts_no_backend_calls",
        "alerts_no_exchange_api_calls",
        "alerts_no_order_submission",
        "alerts_no_secret_reads",
        "alert_center_safety_boundary_ok",
    ):
        assert token in smoke


def test_alert_sources_avoid_forbidden_tokens() -> None:
    source = _source()
    for token in (
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
    ):
        assert token not in source
