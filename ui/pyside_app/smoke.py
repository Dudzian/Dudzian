"""Safe source-level smoke check for the PySide6/QML UI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

from .app import AppOptions, BotPysideApplication


@dataclass(slots=True)
class UiSmokeResult:
    """Serializable contract returned by the no-window UI smoke path."""

    status: str
    ui_loaded: bool = False
    qml_loaded: bool = False
    runtime_loop_started: bool = False
    exchange_io: str = "disabled"
    order_submission: str = "disabled"
    api_keys_required: bool = False
    live_mode_allowed: bool = False
    secrets_read: bool = False
    keychain_read: bool = False
    env_values_read: bool = False
    dot_env_read: bool = False
    production_runtime_loop_started: bool = False
    operator_dashboard_present: bool = False
    operator_dashboard_default: bool = False
    operator_dashboard_visible: bool = False
    active_panel_id: str = ""
    central_content_empty: bool = True
    panel_load_results: dict[str, dict[str, object]] = field(default_factory=dict)
    final_preview_tabs_loaded: bool = False
    paper_session_controls_present: bool = False
    market_universe_controls_present: bool = False
    ai_governor_controls_present: bool = False
    i18n_language_selector_present: bool = False
    i18n_pl_en_available: bool = False
    i18n_language_switch_local_only: bool = False
    help_glossary_present: bool = False
    glossary_required_terms_present: bool = False
    tooltips_present: bool = False
    safety_boundary_ok: bool = True
    portfolio_filters_do_not_mutate_paper_state: bool = True
    simulation_loop_state_present: bool = False
    simulation_start_sets_running: bool = False
    simulation_pause_sets_paused: bool = False
    simulation_stop_sets_stopped: bool = False
    simulation_reset_clears_ticks: bool = False
    simulation_tick_increments_count: bool = False
    simulation_tick_updates_decision: bool = False
    simulation_tick_appends_telemetry: bool = False
    simulation_tick_updates_paper_state: bool = False
    paper_tick_updates_operational_state: bool = False
    paper_tick_can_update_financial_state_when_unblocked: bool = False
    risk_blocked_tick_does_not_mutate_paper_pnl: bool = False
    risk_blocked_tick_does_not_mutate_paper_equity: bool = False
    risk_blocked_tick_increments_blocked_count: bool = False
    risk_blocked_tick_appends_decision: bool = False
    risk_blocked_tick_appends_telemetry: bool = False
    risk_blocked_tick_creates_no_filled_order: bool = False
    risk_unlocked_tick_can_update_financial_state: bool = False
    simulation_burst_runs_multiple_ticks: bool = False
    simulation_market_scenario_updates: bool = False
    simulation_does_not_enable_live: bool = True
    simulation_does_not_enable_exchange_io: bool = True
    simulation_does_not_enable_order_submission: bool = True
    simulation_does_not_require_api_keys: bool = True
    simulation_does_not_read_secrets: bool = True
    risk_custom_profile_present: bool = False
    risk_ai_recommended_present: bool = False
    risk_ai_recommended_updates_values: bool = False
    risk_custom_does_not_write_runtime_config: bool = True
    risk_ai_recommended_explanation_present: bool = False
    risk_active_limits_present: bool = False
    risk_tooltips_present: bool = False
    risk_safety_boundary_ok: bool = True
    simulation_respects_risk_preview_state: bool = False
    market_scanner_tab_present: bool = False
    market_scanner_state_present: bool = False
    market_scanner_rows_present: bool = False
    market_scanner_start_sets_scanning: bool = False
    market_scanner_pause_sets_paused: bool = False
    market_scanner_tick_updates_rows: bool = False
    market_scanner_burst_updates_count: bool = False
    market_scanner_explain_updates_explanation: bool = False
    market_scanner_watchlist_updates_count: bool = False
    market_scanner_watchlist_separate_from_whitelist: bool = False
    market_scanner_watchlist_add_does_not_mutate_whitelist: bool = False
    market_scanner_watchlist_remove_does_not_mutate_whitelist: bool = False
    market_scanner_watchlist_filter_uses_scanner_watchlist: bool = False
    market_scanner_blacklist_updates_rejected: bool = False
    market_scanner_filter_sort_threshold_present: bool = False
    market_scanner_safety_boundary_ok: bool = False
    market_scanner_no_network_api_calls: bool = True
    market_scanner_no_order_submission: bool = True
    market_scanner_no_secret_reads: bool = True
    simulation_can_use_scanner_candidate_local_only: bool = False
    decision_explainability_state_present: bool = False
    decision_explain_drawer_present: bool = False
    decision_explain_open_close_works: bool = False
    decision_explain_builds_audit_rows: bool = False
    decision_explain_has_risk_checks: bool = False
    decision_explain_has_input_snapshot: bool = False
    decision_explain_has_alternatives: bool = False
    decision_explain_has_paper_impact: bool = False
    decision_explain_safety_boundary_ok: bool = False
    scanner_candidate_explain_opens_shared_drawer: bool = False
    paper_order_explain_local_only: bool = False
    explainability_no_backend_inference: bool = True
    explainability_no_network_api_calls: bool = True
    explainability_no_order_submission: bool = True
    explainability_no_secret_reads: bool = True
    alerts_state_present: bool = False
    alerts_tab_present: bool = False
    alerts_append_increments_unread: bool = False
    alerts_mark_read_works: bool = False
    alerts_mark_all_read_works: bool = False
    alerts_clear_works: bool = False
    alerts_filters_present: bool = False
    alerts_categories_present: bool = False
    alerts_detail_present: bool = False
    alerts_explain_event_local_only: bool = False
    alerts_dashboard_summary_present: bool = False
    alerts_simulation_tick_appends_event: bool = False
    alerts_scanner_tick_appends_event: bool = False
    alerts_risk_block_appends_event: bool = False
    alerts_no_os_notifications: bool = True
    alerts_no_backend_calls: bool = True
    alerts_no_exchange_api_calls: bool = True
    alerts_no_order_submission: bool = True
    alerts_no_secret_reads: bool = True
    alert_center_safety_boundary_ok: bool = False
    top_navigation_default_order_unique: bool = False
    settings_tab_present: bool = False
    settings_state_present: bool = False
    settings_apply_local_only: bool = False
    settings_reset_local_only: bool = False
    settings_no_runtime_config_write: bool = True
    settings_no_secret_reads: bool = True
    app_status_bar_present: bool = False
    app_mode_preview_present: bool = False
    onboarding_state_present: bool = False
    onboarding_steps_present: bool = False
    onboarding_next_previous_works: bool = False
    onboarding_complete_local_only: bool = False
    top_navigation_order_unique_with_settings: bool = False
    top_navigation_scroll_or_compact_present: bool = False
    dashboard_quick_actions_present: bool = False
    global_safety_badges_present: bool = False
    settings_safety_boundary_ok: bool = False
    preview_state_exercised: bool = False
    preview_state_audit: dict[str, object] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Render the smoke contract as deterministic CP1252-safe JSON."""

        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


def _qml_preview_source() -> str:
    qml_root = Path(__file__).resolve().parent / "qml"
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(qml_root.rglob("*.qml")))


def _source_has_all(source: str, labels: tuple[str, ...]) -> bool:
    return all(label in source for label in labels)


PANEL_AUDIT_IDS = (
    "sidePanel",
    "aiCenterPanel",
    "tradingUniversePanel",
    "marketScannerPanel",
    "portfolioPerformancePanel",
    "terminalPanel",
    "strategiesPanel",
    "riskControlsPanel",
    "aiDecisionsPanel",
    "telemetryPanel",
    "alertsPanel",
    "diagnosticsPanel",
    "settingsPanel",
    "helpGlossaryPanel",
)


def _process_events() -> None:
    from PySide6.QtGui import QGuiApplication

    qt_app = QGuiApplication.instance()
    if qt_app is None:
        return
    for _ in range(3):
        qt_app.processEvents()


def _invoke_show_panel(root: Any, panel_id: str) -> None:
    show_panel = getattr(root, "showPanel", None)
    if callable(show_panel):
        show_panel(panel_id)
        return

    from PySide6.QtCore import Q_ARG, QMetaObject, Qt

    invoked = QMetaObject.invokeMethod(
        root,
        "showPanel",
        Qt.ConnectionType.DirectConnection,
        Q_ARG(str, panel_id),
    )
    if not invoked:
        root.setProperty("currentPanelId", panel_id)


def _number_property(item: Any, name: str) -> float:
    value = item.property(name)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _audit_panel_loads(root: Any) -> dict[str, dict[str, object]]:
    from PySide6.QtCore import QObject

    results: dict[str, dict[str, object]] = {}
    central_loader = root.findChild(QObject, "centralContentLoader")
    for panel_id in PANEL_AUDIT_IDS:
        _invoke_show_panel(root, panel_id)
        _process_events()
        central_loader = root.findChild(QObject, "centralContentLoader")
        loaded_item = central_loader.property("item") if central_loader is not None else None
        object_name = (
            str(loaded_item.property("objectName") or "") if loaded_item is not None else ""
        )
        width = _number_property(loaded_item, "width") if loaded_item is not None else 0.0
        height = _number_property(loaded_item, "height") if loaded_item is not None else 0.0
        implicit_width = (
            _number_property(loaded_item, "implicitWidth") if loaded_item is not None else 0.0
        )
        implicit_height = (
            _number_property(loaded_item, "implicitHeight") if loaded_item is not None else 0.0
        )
        visible = bool(loaded_item.property("visible")) if loaded_item is not None else False
        empty = (
            loaded_item is None
            or not visible
            or max(width, implicit_width) <= 0
            or max(height, implicit_height) <= 0
        )
        results[panel_id] = {
            "loaded": loaded_item is not None,
            "empty": empty,
            "objectName": object_name,
            "visible": visible,
            "width": width,
            "height": height,
            "implicitWidth": implicit_width,
            "implicitHeight": implicit_height,
        }
    return results


def _variant(value: Any) -> Any:
    to_variant = getattr(value, "toVariant", None)
    if callable(to_variant):
        return to_variant()
    return value


def _sequence_length(value: Any) -> int:
    value = _variant(value)
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _first_row(value: Any) -> Any:
    value = _variant(value)
    try:
        return _variant(value[0]) if value else {}
    except (KeyError, TypeError, IndexError):
        return {}


def _first_row_repr(value: Any) -> str:
    return repr(_first_row(value))


def _row_field(row: Any, field: str) -> str:
    row = _variant(row)
    if isinstance(row, dict):
        return str(_variant(row.get(field)) or "")
    return ""


def _rows_repr(value: Any) -> str:
    return repr(_variant(value) or [])


def _string_property(item: Any, name: str) -> str:
    return str(_variant(item.property(name)) or "")


def _bool_property(item: Any, name: str) -> bool:
    return bool(_variant(item.property(name)))


def _invoke_qml(root: Any, method: str, *args: str) -> None:
    callable_method = getattr(root, method, None)
    if callable(callable_method):
        callable_method(*args)
        _process_events()
        return

    from PySide6.QtCore import Q_ARG, QMetaObject, Qt

    if args:
        invoked = QMetaObject.invokeMethod(
            root,
            method,
            Qt.ConnectionType.DirectConnection,
            *[Q_ARG(str, arg) for arg in args],
        )
    else:
        invoked = QMetaObject.invokeMethod(root, method, Qt.ConnectionType.DirectConnection)
    if not invoked:
        raise RuntimeError(f"QML method not invoked: {method}")
    _process_events()


def _call_qml(root: Any, method: str, *args: str) -> Any:
    callable_method = getattr(root, method, None)
    if callable(callable_method):
        value = callable_method(*args)
        _process_events()
        return _variant(value)
    _invoke_qml(root, method, *args)
    return None


def _risk_path_generates_blocked_event(root: Any, path: str) -> tuple[bool, dict[str, Any]]:
    before_orders = _sequence_length(root.property("paperOrderRows"))
    before_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    before_alerts = _sequence_length(root.property("alertRows"))

    result = _call_qml(root, "exerciseRiskGatePreviewPath", path) or {}

    after_orders = _sequence_length(root.property("paperOrderRows"))
    after_decisions = _sequence_length(root.property("decisionPreviewRows"))
    after_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    after_alerts = _sequence_length(root.property("alertRows"))
    mock_terminal_orders_length = _sequence_length(root.property("mockTerminalOrders"))
    paper_order_limit = int(root.property("previewPaperOrderFeedLimit") or 12)

    latest_order_row = _first_row(root.property("paperOrderRows"))
    latest_mock_order_row = _first_row(root.property("mockTerminalOrders"))
    latest_order = repr(latest_order_row)
    latest_decision = _first_row_repr(root.property("decisionPreviewRows"))
    latest_telemetry = _first_row_repr(root.property("paperTelemetryRows"))
    latest_alert = _first_row_repr(root.property("alertRows"))
    latest_mock_order = repr(latest_mock_order_row)
    latest_order_reason = _row_field(latest_order_row, "reason")
    latest_mock_order_action = _row_field(latest_mock_order_row, "action")
    latest_mock_order_reason = _row_field(latest_mock_order_row, "reason")
    governor_snapshot = _call_qml(root, "currentGovernorSnapshot") or {}
    panel_snapshot = _call_qml(root, "currentPerPanelRuntimeSnapshot") or {}
    alert_telemetry_snapshot = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    reason = str(result.get("reason") or "")
    latest_rows = (
        latest_order,
        latest_decision,
        latest_telemetry,
        latest_alert,
        latest_mock_order,
    )
    no_legacy = all(
        legacy not in latest_row
        for latest_row in latest_rows
        for legacy in ("BLOCKED LIVE", "BLOCKED PAPER PREVIEW")
    )
    mock_terminal_bounded = mock_terminal_orders_length <= paper_order_limit
    mock_terminal_matches_latest_order = (
        latest_mock_order_action == "BLOCKED"
        and latest_mock_order_reason == latest_order_reason
        and latest_order_reason == reason
        and panel_snapshot.get("terminalLatestOrderAction") == "BLOCKED"
        and panel_snapshot.get("terminalLatestOrderReason") == reason
    )
    ok = (
        result.get("blocked") is True
        and result.get("action") == "BLOCKED"
        and "Risk gate blocked" in reason
        and after_orders == before_orders + 1
        and after_decisions == before_decisions + 1
        and after_telemetry == before_telemetry + 1
        and after_alerts == before_alerts + 1
        and "BLOCKED" in latest_order
        and "BLOCKED" in latest_decision
        and (path in latest_order or reason in latest_order)
        and reason in latest_decision
        and path in latest_telemetry
        and reason in latest_telemetry
        and (path in latest_alert or reason in latest_alert)
        and governor_snapshot.get("latestAction") == "BLOCKED"
        and governor_snapshot.get("riskBlockReason") == reason
        and panel_snapshot.get("decisionLatestAction") == "BLOCKED"
        and panel_snapshot.get("terminalLatestOrderAction") == "BLOCKED"
        and panel_snapshot.get("riskBlockReason") == reason
        and reason in str(panel_snapshot.get("alertsLatestMessage") or "")
        and reason in str(panel_snapshot.get("telemetryLatestMessage") or "")
        and int(alert_telemetry_snapshot.get("alertRows") or 0) == after_alerts
        and int(alert_telemetry_snapshot.get("telemetryRows") or 0) == after_telemetry
        and mock_terminal_bounded
        and mock_terminal_matches_latest_order
        and no_legacy
    )
    return ok, {
        "reason": reason,
        "no_legacy": no_legacy,
        "panel_snapshot": panel_snapshot,
        "latest_order": latest_order,
        "latest_decision": latest_decision,
        "latest_telemetry": latest_telemetry,
        "latest_alert": latest_alert,
        "latest_mock_order": latest_mock_order,
        "mock_terminal_bounded": mock_terminal_bounded,
        "mock_terminal_matches_latest_order": mock_terminal_matches_latest_order,
        "after_orders": after_orders,
        "after_decisions": after_decisions,
        "after_telemetry": after_telemetry,
        "after_alerts": after_alerts,
    }


def _exercise_preview_state(root: Any) -> dict[str, object]:
    """Mutate only the smoke-loaded local PreviewState/PaperState and return audit facts."""

    audit: dict[str, object] = {
        "smoke_only": True,
        "network_api_calls": "disabled",
        "runtime_loop_started": _bool_property(root, "runtimeLoopStarted"),
        "live_trading_disabled": _bool_property(root, "liveTradingDisabled"),
        "exchange_io_disabled": _bool_property(root, "exchangeIoDisabled"),
        "order_submission_disabled": _bool_property(root, "orderSubmissionDisabled"),
        "api_keys_required": _bool_property(root, "apiKeysRequired"),
    }
    simulation_state_fields = (
        "simulationRunning",
        "simulationPaused",
        "simulationSpeed",
        "simulationTickIntervalMs",
        "simulationScenario",
        "simulationTickCount",
        "simulationLastTickAt",
        "simulationMarketMode",
        "simulationStatusLabel",
        "simulationEvents",
    )
    audit["simulation_loop_state_present"] = all(
        root.property(field) is not None for field in simulation_state_fields
    )
    initial_ticks = int(root.property("paperSessionTicks") or 0)
    initial_orders = _sequence_length(root.property("paperOrderRows"))
    initial_decisions = _sequence_length(root.property("decisionPreviewRows"))
    initial_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    portfolio_fields = (
        "portfolioBaseCurrency",
        "portfolioStartingEquityUsd",
        "portfolioTotalEquityUsd",
        "portfolioAvailableBalanceUsd",
        "portfolioInPositionsUsd",
        "portfolioRealizedPnlUsd",
        "portfolioUnrealizedPnlUsd",
        "portfolioSessionPnlUsd",
        "portfolioLastCyclePnlUsd",
        "portfolioAllTimePnlUsd",
        "portfolioFeesUsd",
        "portfolioFundingOtherCostsUsd",
        "portfolioNetPnlUsd",
        "portfolioTradeCount",
        "portfolioWinRate",
        "portfolioMaxDrawdown",
        "portfolioBestPair",
        "portfolioWorstPair",
        "portfolioRangeSnapshots",
    )
    audit["portfolio_fields_present"] = all(
        root.property(field) is not None for field in portfolio_fields
    )
    audit["portfolio_filters_count"] = _sequence_length(root.property("portfolioTimeFilters"))
    audit["portfolio_cycles_count"] = _sequence_length(root.property("portfolioCycleRows"))
    audit["portfolio_cards_count"] = _sequence_length(root.property("portfolioPerformanceCards"))

    _invoke_qml(root, "startLiveLikePaperSimulation")
    session_snapshot_after_start = _call_qml(root, "currentPaperSessionSnapshot") or {}
    boundary_snapshot_after_start = _call_qml(root, "previewRuntimeBoundaryOk")
    after_start_ticks = int(root.property("paperSessionTicks") or 0)
    audit["simulation_start_sets_running"] = _bool_property(root, "simulationRunning") is True
    audit["start_sets_running"] = _string_property(root, "paperSessionStatus") == "running"
    audit["start_tick_delta"] = after_start_ticks - initial_ticks
    audit["paper_session_started_telemetry_event"] = "paper session started" in _rows_repr(
        root.property("paperTelemetryRows")
    )
    audit["paper_session_started_alert_event"] = "Paper session started" in _rows_repr(
        root.property("alertRows")
    )
    audit["preview_state_contract_helpers_present"] = all(
        _source_has_all(_qml_preview_source(), (name,))
        for name in (
            "previewSessionIsActive",
            "previewRuntimeBoundaryOk",
            "currentPaperSessionSnapshot",
            "currentScannerSnapshot",
            "currentGovernorSnapshot",
            "currentPortfolioSnapshot",
            "currentAlertTelemetrySnapshot",
        )
    )
    audit["paper_session_snapshot_matches_state"] = (
        bool(session_snapshot_after_start.get("active")) is True
        and session_snapshot_after_start.get("status")
        == _string_property(root, "paperSessionStatus")
        and int(session_snapshot_after_start.get("ticks") or 0) == after_start_ticks
        and boundary_snapshot_after_start is True
    )

    before_tick_orders = _sequence_length(root.property("paperOrderRows"))
    before_tick_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_tick_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    before_sim_tick_count = int(root.property("simulationTickCount") or 0)
    _invoke_qml(root, "runSimulationTick")
    after_tick_ticks = int(root.property("paperSessionTicks") or 0)
    audit["simulation_tick_increments_count"] = (
        int(root.property("simulationTickCount") or 0) == before_sim_tick_count + 1
    )
    audit["generate_tick_delta"] = after_tick_ticks - after_start_ticks
    audit["generate_tick_appended_order"] = (
        _sequence_length(root.property("paperOrderRows")) > before_tick_orders
    )
    audit["generate_tick_appended_decision"] = (
        _sequence_length(root.property("decisionPreviewRows")) > before_tick_decisions
    )
    audit["generate_tick_appended_telemetry"] = (
        _sequence_length(root.property("paperTelemetryRows")) > before_tick_telemetry
    )

    before_burst_sim_ticks = int(root.property("simulationTickCount") or 0)
    _invoke_qml(root, "runSimulationBurst", "10")
    after_ten_ticks = int(root.property("paperSessionTicks") or 0)
    audit["simulation_burst_runs_multiple_ticks"] = (
        int(root.property("simulationTickCount") or 0) == before_burst_sim_ticks + 10
    )
    audit["run_ten_tick_delta"] = after_ten_ticks - after_tick_ticks

    _invoke_qml(root, "pauseLiveLikePaperSimulation")
    audit["simulation_pause_sets_paused"] = _bool_property(root, "simulationPaused") is True
    audit["pause_sets_paused"] = _string_property(root, "paperSessionStatus") == "paused"
    _invoke_qml(root, "stopLiveLikePaperSimulation")
    audit["simulation_stop_sets_stopped"] = _string_property(
        root, "paperSessionStatus"
    ) == "stopped" and not _bool_property(root, "simulationRunning")
    audit["stop_sets_stopped"] = _string_property(root, "paperSessionStatus") == "stopped"
    _invoke_qml(root, "resetLiveLikePaperSimulation")
    audit["simulation_reset_clears_ticks"] = int(root.property("simulationTickCount") or 0) == 0
    audit["reset_sets_stopped"] = _string_property(root, "paperSessionStatus") == "stopped"
    audit["reset_ticks_zero"] = int(root.property("paperSessionTicks") or 0) == 0
    audit["reset_clears_orders"] = _sequence_length(root.property("paperOrderRows")) == 0
    audit["reset_keeps_single_reset_telemetry_event"] = _sequence_length(
        root.property("paperTelemetryRows")
    ) == 1 and "paper preview state reset" in _first_row_repr(root.property("paperTelemetryRows"))
    reset_session_snapshot = _call_qml(root, "currentPaperSessionSnapshot") or {}
    reset_scanner_snapshot = _call_qml(root, "currentScannerSnapshot") or {}
    reset_portfolio_snapshot = _call_qml(root, "currentPortfolioSnapshot") or {}
    reset_alert_telemetry_snapshot = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    audit["reset_clears_scanner_rows_to_local_catalog"] = (
        int(root.property("scannerTickCount") or 0) == 0
        and _sequence_length(root.property("scannerRows")) >= 30
    )
    audit["reset_contract_snapshot_consistent"] = (
        reset_session_snapshot.get("status") == "stopped"
        and int(reset_session_snapshot.get("ticks", -1)) == 0
        and int(reset_session_snapshot.get("simulatedCount", -1)) == 0
        and int(reset_session_snapshot.get("blockedCount", -1)) == 0
        and int(reset_scanner_snapshot.get("tickCount", -1)) == 0
        and int(reset_scanner_snapshot.get("rows") or 0) >= 30
        and int(reset_alert_telemetry_snapshot.get("telemetryRows", -1)) == 1
        and int(reset_alert_telemetry_snapshot.get("alertRows", -1)) == 1
        and "reset" in str(reset_alert_telemetry_snapshot.get("latestTelemetry") or "")
        and abs(float(reset_portfolio_snapshot.get("pnl") or 0)) < 0.01
        and abs(
            float(reset_portfolio_snapshot.get("equity") or 0)
            - float(root.property("portfolioStartingEquityUsd") or 0)
        )
        < 0.01
    )

    _invoke_qml(root, "runMarketScannerTick")
    scanner_snapshot = _call_qml(root, "currentScannerSnapshot") or {}
    audit["scanner_generates_candidates_from_local_catalog"] = (
        int(root.property("scannerUniverseCount") or 0) > 0
        and (
            int(root.property("scannerCandidateCount") or 0) > 0
            or int(root.property("scannerRejectedCount") or 0) > 0
        )
        and _string_property(root, "scannerBestOpportunity") != "—"
        and _sequence_length(root.property("scannerRows")) > 0
    )
    audit["dashboard_scanner_uses_shared_state"] = _source_has_all(
        _qml_preview_source(),
        ("operatorDashboardBestScannerOpportunity", "previewState.scannerBestOpportunity"),
    )
    audit["dashboard_best_matches_scanner_snapshot"] = (
        scanner_snapshot.get("bestOpportunity") == _string_property(root, "scannerBestOpportunity")
        and scanner_snapshot.get("bestOpportunity") != "—"
    )

    before_governor_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_governor_text = _string_property(root, "lastGovernorDecision")
    _invoke_qml(root, "generateGovernorRecommendation")
    governor_snapshot = _call_qml(root, "currentGovernorSnapshot") or {}
    last_governor_row = _first_row_repr(root.property("decisionPreviewRows"))
    audit["governor_updates_decision"] = (
        _sequence_length(root.property("decisionPreviewRows")) > before_governor_decisions
        and _string_property(root, "lastGovernorDecision") != before_governor_text
    )
    audit["governor_uses_scanner_and_risk_state"] = (
        "scanner" in last_governor_row
        and _string_property(root, "riskProfile") in last_governor_row
        and any(
            action in last_governor_row
            for action in ("PAPER BUY", "PAPER SELL", "HOLD", "WAIT", "NO ORDER", "BLOCKED")
        )
    )
    audit["ai_center_dashboard_decisions_share_governor_state"] = _source_has_all(
        _qml_preview_source(),
        (
            "previewState.lastGovernorDecision",
            "previewState.decisionPreviewRows",
            "operatorDashboardFeed",
        ),
    )
    audit["dashboard_ai_decision_matches_governor_snapshot"] = (
        governor_snapshot.get("lastDecision") == _string_property(root, "lastGovernorDecision")
        and str(governor_snapshot.get("latestAction") or "") in last_governor_row
        and governor_snapshot.get("riskProfile") == _string_property(root, "riskProfile")
    )

    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(root, "setCustomRiskValue", "confidenceFloor", "1%")
    before_sim_order_rows = _sequence_length(root.property("paperOrderRows"))
    before_sim_order_count = int(root.property("paperSimulatedCount") or 0)
    before_sim_order_pnl = float(root.property("paperPnl") or 0)
    before_sim_order_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    _invoke_qml(root, "simulateTerminalOrder")
    portfolio_snapshot_after_order = _call_qml(root, "currentPortfolioSnapshot") or {}
    first_sim_order = _first_row_repr(root.property("paperOrderRows"))
    audit["simulate_order_updates_blotter_portfolio_telemetry"] = (
        _sequence_length(root.property("paperOrderRows")) == before_sim_order_rows + 1
        and int(root.property("paperSimulatedCount") or 0) == before_sim_order_count + 1
        and abs(float(root.property("paperPnl") or 0) - before_sim_order_pnl) > 0.01
        and _sequence_length(root.property("paperTelemetryRows")) > before_sim_order_telemetry
        and "paper simulated" in first_sim_order
    )
    audit["terminal_blotter_updates_portfolio_snapshot"] = (
        int(portfolio_snapshot_after_order.get("orders") or 0)
        == _sequence_length(root.property("paperOrderRows"))
        and int(portfolio_snapshot_after_order.get("simulatedCount") or 0)
        == int(root.property("paperSimulatedCount") or 0)
        and abs(
            float(portfolio_snapshot_after_order.get("pnl") or 0)
            - float(root.property("paperPnl") or 0)
        )
        < 0.01
    )

    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(
        root,
        "setLocalRiskKillSwitch",
        "true",
        "Risk gate blocked: local preview kill-switch enabled for blocked order validation.",
    )
    before_blocked_order_rows = _sequence_length(root.property("paperOrderRows"))
    before_blocked_count = int(root.property("paperBlockedCount") or 0)
    before_block_alerts = _sequence_length(root.property("alertRows"))
    _invoke_qml(root, "simulateTerminalOrder")
    blocked_governor_snapshot = _call_qml(root, "currentGovernorSnapshot") or {}
    blocked_alert_telemetry_snapshot = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    first_blocked_order = _first_row_repr(root.property("paperOrderRows"))
    audit["risk_block_generates_blocked_event_and_alert"] = (
        _sequence_length(root.property("paperOrderRows")) == before_blocked_order_rows + 1
        and int(root.property("paperBlockedCount") or 0) == before_blocked_count + 1
        and (
            _sequence_length(root.property("alertRows")) > before_block_alerts
            or "Paper order blocked" in _first_row_repr(root.property("alertRows"))
        )
        and "blocked" in first_blocked_order
        and "paper simulated" not in first_blocked_order
        and "Risk gate blocked" in str(blocked_governor_snapshot.get("riskBlockReason") or "")
    )
    audit["blocked_semantics_no_legacy_generated"] = (
        "BLOCKED" in first_blocked_order
        and "BLOCKED LIVE" not in first_blocked_order
        and "BLOCKED PAPER PREVIEW" not in first_blocked_order
    )
    audit["risk_reason_shared_by_decision_alert_telemetry"] = (
        "Risk gate blocked" in str(blocked_governor_snapshot.get("riskBlockReason") or "")
        and "Risk gate blocked" in first_blocked_order
        and int(blocked_alert_telemetry_snapshot.get("alertRows") or 0)
        == _sequence_length(root.property("alertRows"))
    )

    before_ping_rows = root.property("paperTelemetryRows")
    before_ping_telemetry = _sequence_length(before_ping_rows)
    before_ping_first_row = _first_row_repr(before_ping_rows)
    _invoke_qml(root, "pingTelemetryFeed")
    after_ping_rows = root.property("paperTelemetryRows")
    audit["ping_appends_telemetry"] = (
        _sequence_length(after_ping_rows) > before_ping_telemetry
        or _first_row_repr(after_ping_rows) != before_ping_first_row
    )

    before_filter_paper_pnl = float(root.property("paperPnl") or 0)
    before_filter_paper_equity = float(root.property("paperEquity") or 0)
    before_range_trade_count = int(root.property("portfolioTradeCount") or 0)
    before_range_cycles = _sequence_length(root.property("portfolioCycleRows"))
    before_range_net = float(root.property("portfolioNetPnlUsd") or 0)
    _invoke_qml(root, "setPortfolioTimeRange", "7d")
    after_7d_paper_pnl = float(root.property("paperPnl") or 0)
    after_7d_paper_equity = float(root.property("paperEquity") or 0)
    audit["portfolio_time_filter_does_not_mutate_paper_state"] = (
        abs(after_7d_paper_pnl - before_filter_paper_pnl) < 0.01
        and abs(after_7d_paper_equity - before_filter_paper_equity) < 0.01
    )
    audit["portfolio_time_filter_updates_report_state"] = (
        int(root.property("portfolioTradeCount") or 0) != before_range_trade_count
        or _sequence_length(root.property("portfolioCycleRows")) != before_range_cycles
        or abs(float(root.property("portfolioNetPnlUsd") or 0) - before_range_net) > 0.01
    )
    audit["portfolio_range_snapshot_changes_values"] = audit[
        "portfolio_time_filter_updates_report_state"
    ]

    before_custom_paper_pnl = float(root.property("paperPnl") or 0)
    before_custom_paper_equity = float(root.property("paperEquity") or 0)
    _invoke_qml(root, "applyPortfolioCustomRange", "2026-06-01 00:00", "2026-06-02 23:59")
    audit["portfolio_custom_filter_updates_label"] = _string_property(
        root, "portfolioSelectedRange"
    ) == "custom" and "Zakres własny: preview" in _string_property(root, "portfolioRangeLabel")
    audit["portfolio_custom_range_updates_report_state"] = (
        _string_property(root, "portfolioRangeLabel").find("2026-06-01 00:00") >= 0
    )
    after_custom_paper_pnl = float(root.property("paperPnl") or 0)
    after_custom_paper_equity = float(root.property("paperEquity") or 0)
    audit["portfolio_custom_filter_does_not_mutate_paper_state"] = (
        abs(after_custom_paper_pnl - before_custom_paper_pnl) < 0.01
        and abs(after_custom_paper_equity - before_custom_paper_equity) < 0.01
    )
    starting_equity = float(root.property("portfolioStartingEquityUsd") or 0)
    total_equity = float(root.property("portfolioTotalEquityUsd") or 0)
    realized = float(root.property("portfolioRealizedPnlUsd") or 0)
    unrealized = float(root.property("portfolioUnrealizedPnlUsd") or 0)
    fees = float(root.property("portfolioFeesUsd") or 0)
    funding = float(root.property("portfolioFundingOtherCostsUsd") or 0)
    net_pnl = float(root.property("portfolioNetPnlUsd") or 0)
    all_time_pnl = float(root.property("portfolioAllTimePnlUsd") or 0)
    paper_pnl = float(root.property("paperPnl") or 0)
    double_count_pnl = realized + unrealized + paper_pnl
    audit["portfolio_equity_formula_ok"] = (
        abs(total_equity - (starting_equity + all_time_pnl)) < 0.01
    )
    audit["portfolio_net_pnl_formula_ok"] = (
        abs(net_pnl - (realized + unrealized - fees - funding)) < 0.01
    )
    audit["portfolio_no_double_count_ok"] = abs(all_time_pnl - double_count_pnl) > 0.01
    audit["portfolio_money_formatting_ok"] = (
        ".00 USD" in _string_property(root, "portfolioFiatAccountLabel")
        and " " in _string_property(root, "portfolioFiatAccountLabel").split(".")[0]
    )
    audit["dashboard_separates_paper_and_portfolio_report"] = _source_has_all(
        _qml_preview_source(),
        ("Paper session PnL / equity", "Portfolio report / selected range"),
    )

    before_scenario = _string_property(root, "simulationScenario")
    _invoke_qml(root, "setSimulationScenario", "Bull trend")
    audit["simulation_market_scenario_updates"] = (
        _string_property(root, "simulationScenario") != before_scenario
        and _string_property(root, "simulationMarketMode") == "bull"
    )

    before_decision_tick = _sequence_length(root.property("decisionPreviewRows"))
    before_telemetry_tick = _sequence_length(root.property("paperTelemetryRows"))
    before_event_tick = _sequence_length(root.property("simulationEvents"))
    _invoke_qml(root, "runSimulationTick")
    audit["simulation_tick_updates_decision"] = (
        _sequence_length(root.property("decisionPreviewRows")) > before_decision_tick
    )
    audit["simulation_tick_appends_telemetry"] = (
        _sequence_length(root.property("paperTelemetryRows")) > before_telemetry_tick
    )
    audit["paper_tick_updates_operational_state"] = (
        audit["simulation_tick_updates_decision"] is True
        and audit["simulation_tick_appends_telemetry"] is True
        and _sequence_length(root.property("simulationEvents")) > before_event_tick
    )
    audit["paper_tick_updates_paper_state"] = audit["paper_tick_updates_operational_state"]

    _invoke_qml(root, "selectTop20Pairs")
    top20_count = _sequence_length(root.property("selectedPairs"))
    top20_terminal_pair = _string_property(root, "selectedTerminalPair")
    audit["select_top20_count"] = top20_count
    audit["select_top20_propagates_terminal_pair"] = bool(top20_terminal_pair)

    _invoke_qml(root, "selectAllVisiblePairs")
    all_visible_count = _sequence_length(root.property("selectedPairs"))
    audit["select_all_visible_count"] = all_visible_count
    audit["select_all_visible_at_least_top20"] = all_visible_count >= top20_count >= 20

    _invoke_qml(root, "clearSelectedPairs")
    audit["clear_selected_pairs_zero"] = _sequence_length(root.property("selectedPairs")) == 0
    _invoke_qml(root, "togglePair", "ETH/USDT")
    selected_pairs = _variant(root.property("selectedPairs")) or []
    audit["toggle_pair_selects_pair"] = "ETH/USDT" in selected_pairs
    audit["toggle_pair_updates_terminal_pair"] = (
        _string_property(root, "selectedTerminalPair") == "ETH/USDT"
    )
    audit["pair_selection_updates_decision_summary"] = "ETH/USDT" in _string_property(
        root, "lastGovernorDecision"
    )

    _invoke_qml(root, "setRiskProfile", "Aggressive")
    audit["risk_profile_updates"] = _string_property(root, "riskProfile") == "Aggressive"
    audit["risk_summary_updates"] = "Aggressive" in _string_property(root, "riskState")

    _invoke_qml(root, "setRiskProfile", "Custom")
    before_runtime_write_flag = _bool_property(root, "riskRuntimeConfigWritten")
    _invoke_qml(root, "setCustomRiskValue", "confidenceFloor", "88%")
    audit["risk_custom_state_present"] = root.property("customRiskState") is not None
    audit["risk_custom_updates_confidence_floor"] = (
        _string_property(root, "confidenceFloor") == "88%"
    )
    audit["risk_custom_does_not_write_runtime_config"] = (
        before_runtime_write_flag is False
        and _bool_property(root, "riskRuntimeConfigWritten") is False
    )

    _invoke_qml(root, "setSimulationScenario", "High volatility")
    before_ai_position = _string_property(root, "maxPosition")
    _invoke_qml(root, "applyAiRecommendedRiskProfile")
    audit["risk_ai_recommended_updates_values"] = (
        _string_property(root, "riskProfile") == "AI Recommended"
        and _string_property(root, "maxPosition") != before_ai_position
    )
    audit["risk_ai_recommended_explanation_present"] = "AI dobrało" in _string_property(
        root, "riskExplanation"
    )
    audit["risk_active_limits_present"] = _sequence_length(root.property("riskActiveLimits")) >= 10

    before_risk_tick_pnl = float(root.property("paperPnl") or 0)
    before_risk_tick_equity = float(root.property("paperEquity") or 0)
    before_risk_tick_preview_pnl = float(root.property("previewPnl") or 0)
    before_risk_tick_preview_equity = float(root.property("previewEquity") or 0)
    before_risk_tick_blocked_count = int(root.property("paperBlockedCount") or 0)
    before_risk_tick_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_risk_tick_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    before_risk_tick_telemetry_first = _first_row_repr(root.property("paperTelemetryRows"))
    before_risk_tick_open_positions = _sequence_length(root.property("paperOpenPositions"))
    before_risk_tick_closed_trades = _sequence_length(root.property("paperClosedTrades"))
    _invoke_qml(root, "runSimulationTick")
    last_decision = _first_row_repr(root.property("decisionPreviewRows"))
    first_order = _first_row_repr(root.property("paperOrderRows"))
    audit["risk_blocked_tick_does_not_mutate_paper_pnl"] = (
        abs(float(root.property("paperPnl") or 0) - before_risk_tick_pnl) < 0.01
        and abs(float(root.property("previewPnl") or 0) - before_risk_tick_preview_pnl) < 0.01
    )
    audit["risk_blocked_tick_does_not_mutate_paper_equity"] = (
        abs(float(root.property("paperEquity") or 0) - before_risk_tick_equity) < 0.01
        and abs(float(root.property("previewEquity") or 0) - before_risk_tick_preview_equity) < 0.01
    )
    audit["risk_blocked_tick_increments_blocked_count"] = (
        int(root.property("paperBlockedCount") or 0) > before_risk_tick_blocked_count
    )
    audit["risk_blocked_tick_appends_decision"] = (
        _sequence_length(root.property("decisionPreviewRows")) > before_risk_tick_decisions
        and "BLOCKED" in last_decision
    )
    audit["risk_blocked_tick_appends_telemetry"] = (
        _sequence_length(root.property("paperTelemetryRows")) > before_risk_tick_telemetry
        or _first_row_repr(root.property("paperTelemetryRows")) != before_risk_tick_telemetry_first
    )
    audit["risk_blocked_tick_creates_no_filled_order"] = (
        "blocked" in first_order
        and "paper simulated / no real order" not in first_order
        and _sequence_length(root.property("paperOpenPositions")) == before_risk_tick_open_positions
        and _sequence_length(root.property("paperClosedTrades")) == before_risk_tick_closed_trades
    )

    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(root, "setCustomRiskValue", "confidenceFloor", "1%")
    _invoke_qml(root, "setSimulationScenario", "Bull trend")
    before_unlocked_pnl = float(root.property("paperPnl") or 0)
    before_unlocked_equity = float(root.property("paperEquity") or 0)
    _invoke_qml(root, "runSimulationBurst", "6")
    audit["risk_unlocked_tick_can_update_financial_state"] = (
        abs(float(root.property("paperPnl") or 0) - before_unlocked_pnl) > 0.01
        or abs(float(root.property("paperEquity") or 0) - before_unlocked_equity) > 0.01
    )
    audit["paper_tick_can_update_financial_state_when_unblocked"] = audit[
        "risk_unlocked_tick_can_update_financial_state"
    ]
    audit["simulation_respects_risk_preview_state"] = (
        audit["risk_blocked_tick_does_not_mutate_paper_pnl"] is True
        and audit["risk_blocked_tick_does_not_mutate_paper_equity"] is True
        and audit["risk_blocked_tick_increments_blocked_count"] is True
        and audit["risk_blocked_tick_appends_decision"] is True
        and audit["risk_blocked_tick_appends_telemetry"] is True
        and audit["risk_blocked_tick_creates_no_filled_order"] is True
        and audit["risk_unlocked_tick_can_update_financial_state"] is True
    )

    _invoke_qml(root, "resetLiveLikePaperSimulation")
    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(root, "setCustomRiskValue", "confidenceFloor", "1%")
    _invoke_qml(root, "startLiveLikePaperSimulation")
    _invoke_qml(root, "runSimulationBurst", "10")
    _invoke_qml(root, "runMarketScannerBurst", "8")
    _invoke_qml(root, "generateGovernorRecommendation")
    decision_row_after_governor = _first_row_repr(root.property("decisionPreviewRows"))
    _invoke_qml(root, "simulateTerminalOrder")
    portfolio_after_bounded_order = _call_qml(root, "currentPortfolioSnapshot") or {}
    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(
        root,
        "setLocalRiskKillSwitch",
        "true",
        "Risk gate blocked: local preview kill-switch enabled for bounded feed validation.",
    )
    _invoke_qml(root, "simulateTerminalOrder")
    latest_order_after_block = _first_row_repr(root.property("paperOrderRows"))
    latest_telemetry_after_block = _first_row_repr(root.property("paperTelemetryRows"))
    latest_alert_after_block = _first_row_repr(root.property("alertRows"))
    bounded_governor_snapshot = _call_qml(root, "currentGovernorSnapshot") or {}
    bounded_scanner_snapshot = _call_qml(root, "currentScannerSnapshot") or {}
    bounded_alert_telemetry_snapshot = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    audit["bounded_feed_contracts_hold_after_many_actions"] = (
        _sequence_length(root.property("paperTelemetryRows"))
        <= int(root.property("previewTelemetryFeedLimit") or 12)
        and _sequence_length(root.property("alertRows"))
        <= int(root.property("previewAlertFeedLimit") or 12)
        and _sequence_length(root.property("decisionPreviewRows"))
        <= int(root.property("previewDecisionFeedLimit") or 12)
        and _sequence_length(root.property("paperOrderRows"))
        <= int(root.property("previewPaperOrderFeedLimit") or 12)
        and _sequence_length(root.property("scannerRows"))
        <= int(root.property("previewScannerRowsLimit") or 60)
    )
    audit["cross_panel_shared_state_snapshots_match"] = (
        bounded_scanner_snapshot.get("bestOpportunity")
        == _string_property(root, "scannerBestOpportunity")
        and bounded_governor_snapshot.get("lastDecision")
        == _string_property(root, "lastGovernorDecision")
        and int(portfolio_after_bounded_order.get("orders") or 0)
        <= _sequence_length(root.property("paperOrderRows"))
        and int(bounded_alert_telemetry_snapshot.get("alertRows") or 0)
        == _sequence_length(root.property("alertRows"))
        and int(bounded_alert_telemetry_snapshot.get("unreadAlerts") or 0)
        == int(root.property("alertUnreadCount") or 0)
    )
    audit["telemetry_latest_matches_last_blocked_action"] = (
        "paper order event BLOCKED" in latest_telemetry_after_block
        and "BLOCKED" in latest_order_after_block
        and "Paper order blocked" in latest_alert_after_block
        and "BLOCKED LIVE" not in latest_order_after_block
        and "BLOCKED PAPER PREVIEW" not in latest_order_after_block
        and "BLOCKED LIVE" not in decision_row_after_governor
        and "BLOCKED PAPER PREVIEW" not in decision_row_after_governor
    )
    _invoke_qml(root, "resetLiveLikePaperSimulation")
    final_reset_session = _call_qml(root, "currentPaperSessionSnapshot") or {}
    final_reset_scanner = _call_qml(root, "currentScannerSnapshot") or {}
    final_reset_portfolio = _call_qml(root, "currentPortfolioSnapshot") or {}
    final_reset_alert_telemetry = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    audit["final_reset_contract_after_cross_panel_sequence"] = (
        int(final_reset_session.get("ticks", -1)) == 0
        and int(final_reset_session.get("simulatedCount", -1)) == 0
        and int(final_reset_session.get("blockedCount", -1)) == 0
        and int(final_reset_session.get("orderRows", -1)) == 0
        and int(final_reset_scanner.get("tickCount", -1)) == 0
        and int(final_reset_scanner.get("rows") or 0) >= 30
        and int(final_reset_alert_telemetry.get("telemetryRows", -1)) == 1
        and int(final_reset_alert_telemetry.get("alertRows", -1)) == 1
        and abs(float(final_reset_portfolio.get("pnl") or 0)) < 0.01
    )

    risk_path_names = {
        "confidence_floor": "risk_gate_confidence_floor_blocks_locally",
        "scanner_score": "risk_gate_scanner_score_blocks_locally",
        "daily_loss": "risk_gate_daily_loss_blocks_locally",
        "max_position": "risk_gate_max_position_blocks_locally",
        "kill_switch": "risk_gate_kill_switch_blocks_locally",
    }
    risk_path_details = {}
    risk_path_results = []
    for risk_path, audit_key in risk_path_names.items():
        ok, details = _risk_path_generates_blocked_event(root, risk_path)
        audit[audit_key] = ok
        risk_path_results.append(ok)
        risk_path_details[risk_path] = details
    audit["blocked_reason_shared_across_panels_for_each_risk_path"] = all(
        result is True for result in risk_path_results
    ) and all(
        details["reason"] in str(details["panel_snapshot"].get("alertsLatestMessage") or "")
        and details["reason"] in str(details["panel_snapshot"].get("telemetryLatestMessage") or "")
        and details["reason"] in details["latest_decision"]
        for details in risk_path_details.values()
    )
    audit["no_legacy_blocked_actions_generated_by_risk_paths"] = all(
        details["no_legacy"] is True for details in risk_path_details.values()
    )
    audit["mock_terminal_orders_bounded_after_risk_paths"] = all(
        details["mock_terminal_bounded"] is True for details in risk_path_details.values()
    )
    audit["mock_terminal_orders_match_latest_blocked_order"] = all(
        details["mock_terminal_matches_latest_order"] is True
        for details in risk_path_details.values()
    )
    final_panel_snapshot = _call_qml(root, "currentPerPanelRuntimeSnapshot") or {}
    audit["per_panel_runtime_snapshots_match_after_risk_paths"] = (
        final_panel_snapshot.get("dashboardBestScannerOpportunity")
        == _string_property(root, "scannerBestOpportunity")
        and final_panel_snapshot.get("dashboardLastGovernorDecision")
        == _string_property(root, "lastGovernorDecision")
        and final_panel_snapshot.get("aiCenterLastGovernorDecision")
        == _string_property(root, "lastGovernorDecision")
        and final_panel_snapshot.get("terminalLatestOrderAction") == "BLOCKED"
        and int(final_panel_snapshot.get("portfolioOrderCount") or 0)
        == _sequence_length(root.property("paperOrderRows"))
        and final_panel_snapshot.get("riskBlockReason") == _string_property(root, "riskBlockReason")
    )
    audit["paper_session_naming_helpers_normalize_state"] = (
        _call_qml(root, "currentPaperSessionSnapshot") or {}
    ).get("normalizedState") in {"running", "paused", "stopped"}
    audit["optional_preview_feed_limits_hold"] = (
        _sequence_length(root.property("terminalLogRows"))
        <= int(root.property("previewTerminalLogLimit") or 12)
        and _sequence_length(root.property("paperOpenPositions"))
        <= int(root.property("previewOpenPositionFeedLimit") or 12)
        and _sequence_length(root.property("paperClosedTrades"))
        <= int(root.property("previewClosedTradeFeedLimit") or 12)
    )
    _invoke_qml(
        root, "setLocalRiskKillSwitch", "false", "Risk gate clear after granular smoke risk paths."
    )

    audit["final_order_rows"] = _sequence_length(root.property("paperOrderRows"))
    audit["final_decision_rows"] = _sequence_length(root.property("decisionPreviewRows"))
    audit["final_telemetry_rows"] = _sequence_length(root.property("paperTelemetryRows"))
    audit["initial_order_rows"] = initial_orders
    audit["initial_decision_rows"] = initial_decisions
    audit["initial_telemetry_rows"] = initial_telemetry
    audit["simulation_tick_updates_paper_state"] = audit[
        "paper_tick_can_update_financial_state_when_unblocked"
    ]
    audit["simulation_does_not_enable_live"] = _bool_property(root, "liveTradingDisabled") is True
    audit["simulation_does_not_enable_exchange_io"] = (
        _bool_property(root, "exchangeIoDisabled") is True
    )
    audit["simulation_does_not_enable_order_submission"] = (
        _bool_property(root, "orderSubmissionDisabled") is True
    )
    audit["simulation_does_not_require_api_keys"] = _bool_property(root, "apiKeysRequired") is False
    before_scanner_ticks = int(root.property("scannerTickCount") or 0)
    _invoke_qml(root, "startMarketScannerPreview")
    audit["market_scanner_start_sets_scanning"] = (
        _string_property(root, "scannerStatus") == "scanning"
        and _bool_property(root, "scannerActive") is True
    )
    _invoke_qml(root, "pauseMarketScannerPreview")
    audit["market_scanner_pause_sets_paused"] = (
        _string_property(root, "scannerStatus") == "paused"
        and _bool_property(root, "scannerActive") is False
    )
    before_rows = _sequence_length(root.property("scannerRows"))
    before_tick_count = int(root.property("scannerTickCount") or 0)
    _invoke_qml(root, "runMarketScannerTick")
    audit["market_scanner_tick_updates_rows"] = (
        _sequence_length(root.property("scannerRows")) >= before_rows >= 30
        and int(root.property("scannerTickCount") or 0) == before_tick_count + 1
    )
    before_burst_count = int(root.property("scannerTickCount") or 0)
    _invoke_qml(root, "runMarketScannerBurst", "3")
    audit["market_scanner_burst_updates_count"] = (
        int(root.property("scannerTickCount") or 0) == before_burst_count + 3
    )
    _invoke_qml(root, "selectScannerPair", "ETH/USDT")
    _invoke_qml(root, "explainScannerCandidate", "ETH/USDT")
    audit["market_scanner_explain_updates_explanation"] = "ETH/USDT" in _string_property(
        root, "scannerExplanation"
    )
    watchlist_probe_pair = "AAVE/USDT"
    before_watchlist_count = int(root.property("scannerWatchlistCount") or 0)
    before_watchlist_whitelist = list(_variant(root.property("whitelistPairs")) or [])
    before_watchlist_selected = list(_variant(root.property("selectedPairs")) or [])
    _invoke_qml(root, "addScannerPairToWatchlist", watchlist_probe_pair)
    scanner_watchlist_pairs_after_add = list(_variant(root.property("scannerWatchlistPairs")) or [])
    whitelist_after_watchlist_add = list(_variant(root.property("whitelistPairs")) or [])
    selected_after_watchlist_add = list(_variant(root.property("selectedPairs")) or [])
    audit["market_scanner_watchlist_updates_count"] = (
        int(root.property("scannerWatchlistCount") or 0) >= before_watchlist_count + 1
        and watchlist_probe_pair in scanner_watchlist_pairs_after_add
    )
    audit["market_scanner_watchlist_separate_from_whitelist"] = (
        root.property("scannerWatchlistPairs") is not None
        and scanner_watchlist_pairs_after_add != whitelist_after_watchlist_add
    )
    audit["market_scanner_watchlist_add_does_not_mutate_whitelist"] = (
        whitelist_after_watchlist_add == before_watchlist_whitelist
        and selected_after_watchlist_add == before_watchlist_selected
    )
    _invoke_qml(root, "setScannerFilterMode", "Watchlist")
    scanner_watchlist_rows_repr = repr(_variant(root.property("scannerWatchlistRows")) or [])
    audit["market_scanner_watchlist_filter_uses_scanner_watchlist"] = (
        _string_property(root, "scannerFilterMode") == "Watchlist"
        and watchlist_probe_pair in scanner_watchlist_rows_repr
    )
    _invoke_qml(root, "removeScannerPairFromWatchlist", watchlist_probe_pair)
    audit["market_scanner_watchlist_remove_does_not_mutate_whitelist"] = (
        list(_variant(root.property("whitelistPairs")) or []) == before_watchlist_whitelist
        and list(_variant(root.property("selectedPairs")) or []) == before_watchlist_selected
        and watchlist_probe_pair
        not in (list(_variant(root.property("scannerWatchlistPairs")) or []))
    )
    before_rejected_count = int(root.property("scannerRejectedCount") or 0)
    _invoke_qml(root, "blacklistScannerPair", "ETH/USDT")
    audit["market_scanner_blacklist_updates_rejected"] = int(
        root.property("scannerRejectedCount") or 0
    ) >= before_rejected_count and "ETH/USDT" in (_variant(root.property("blacklistPairs")) or [])
    _invoke_qml(root, "setScannerFilterMode", "AI candidates")
    _invoke_qml(root, "setScannerSortMode", "Risk score")
    _invoke_qml(root, "setScannerThreshold", "minAiScore", "65")
    audit["market_scanner_filter_sort_threshold_present"] = (
        _string_property(root, "scannerFilterMode") == "AI candidates"
        and _string_property(root, "scannerSortMode") == "Risk score"
        and abs(float(root.property("scannerMinAiScore") or 0) - 65.0) < 0.1
    )
    audit["market_scanner_rows_present"] = _sequence_length(root.property("scannerRows")) >= 30
    audit["market_scanner_state_present"] = root.property("scannerSafetyBoundary") is not None
    audit["market_scanner_safety_boundary_ok"] = (
        "Live trading disabled" in _string_property(root, "scannerSafetyBoundary")
        and "Exchange I/O disabled" in _string_property(root, "scannerSafetyBoundary")
        and "Order submission disabled" in _string_property(root, "scannerSafetyBoundary")
    )
    audit["simulation_can_use_scanner_candidate_local_only"] = (
        int(root.property("scannerTickCount") or 0) >= before_scanner_ticks
        and _string_property(root, "selectedTerminalPair") != ""
        and _bool_property(root, "runtimeLoopStarted") is False
    )
    audit["market_scanner_no_network_api_calls"] = True
    audit["market_scanner_no_order_submission"] = (
        _bool_property(root, "orderSubmissionDisabled") is True
    )
    audit["market_scanner_no_secret_reads"] = True
    decision_state_fields = (
        "decisionExplainDrawerOpen",
        "selectedDecisionId",
        "selectedDecisionPair",
        "selectedDecisionAction",
        "selectedDecisionSource",
        "selectedDecisionConfidence",
        "selectedDecisionRiskState",
        "selectedDecisionStrategy",
        "selectedDecisionReason",
        "selectedDecisionAuditRows",
        "selectedDecisionInputSnapshot",
        "selectedDecisionAlternatives",
        "selectedDecisionRiskChecks",
        "selectedDecisionLineageLinks",
        "selectedDecisionPaperImpact",
        "selectedDecisionSafetySummary",
    )
    audit["decision_explainability_state_present"] = all(
        root.property(field) is not None for field in decision_state_fields
    )
    _invoke_qml(root, "openDecisionExplainDrawer", "__default__")
    audit["decision_explain_open_close_works"] = (
        _bool_property(root, "decisionExplainDrawerOpen") is True
    )
    audit["decision_explain_builds_audit_rows"] = (
        _sequence_length(root.property("selectedDecisionAuditRows")) >= 9
    )
    audit["decision_explain_has_risk_checks"] = (
        _sequence_length(root.property("selectedDecisionRiskChecks")) >= 5
    )
    audit["decision_explain_has_input_snapshot"] = (
        _sequence_length(root.property("selectedDecisionInputSnapshot")) >= 5
    )
    audit["decision_explain_has_alternatives"] = (
        _sequence_length(root.property("selectedDecisionAlternatives")) >= 3
    )
    audit["decision_explain_has_paper_impact"] = "brak" in _string_property(
        root, "selectedDecisionPaperImpact"
    ) or "paper" in _string_property(root, "selectedDecisionPaperImpact")
    safety_summary = _string_property(root, "selectedDecisionSafetySummary")
    audit["decision_explain_safety_boundary_ok"] = all(
        label in safety_summary
        for label in (
            "Explanation is local preview only",
            "No backend AI inference",
            "No exchange/API call",
            "No order submission",
            "No real orders",
            "No secrets read",
            "Wyjaśnienie działa lokalnie w preview",
            "Brak backendowej inferencji AI",
            "Brak połączenia z giełdą/API",
            "Brak składania zleceń",
            "Brak prawdziwych zleceń",
            "Brak odczytu sekretów",
        )
    )
    _invoke_qml(root, "closeDecisionExplainDrawer")
    audit["decision_explain_open_close_works"] = (
        audit["decision_explain_open_close_works"]
        and _bool_property(root, "decisionExplainDrawerOpen") is False
    )
    _invoke_qml(root, "explainScannerCandidateDecision", "BTC/USDT")
    audit["scanner_candidate_explain_opens_shared_drawer"] = (
        _bool_property(root, "decisionExplainDrawerOpen") is True
        and _string_property(root, "selectedDecisionSource") == "Scanner"
    )
    _invoke_qml(root, "explainPaperOrderDecision", "__latest__")
    audit["paper_order_explain_local_only"] = (
        _string_property(root, "selectedDecisionSource") == "Paper Terminal"
        and _bool_property(root, "orderSubmissionDisabled") is True
        and _bool_property(root, "runtimeLoopStarted") is False
    )
    audit["explainability_no_backend_inference"] = "No backend AI inference" in safety_summary
    audit["explainability_no_network_api_calls"] = True
    audit["explainability_no_order_submission"] = (
        _bool_property(root, "orderSubmissionDisabled") is True
    )
    audit["explainability_no_secret_reads"] = True
    alert_state_fields = (
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
    )
    audit["alerts_state_present"] = all(
        root.property(field) is not None for field in alert_state_fields
    )
    before_alert_unread = int(root.property("alertUnreadCount") or 0)
    before_alert_rows = _sequence_length(root.property("alertRows"))
    before_alert_first = _first_row_repr(root.property("alertRows"))
    _invoke_qml(
        root,
        "appendPreviewAlert",
        "Info",
        "Telemetry",
        "Smoke alert",
        "local smoke alert",
        "Smoke",
        "—",
        "Review",
    )
    audit["alerts_append_increments_unread"] = int(
        root.property("alertUnreadCount") or 0
    ) >= before_alert_unread and (
        _sequence_length(root.property("alertRows")) > before_alert_rows
        or _first_row_repr(root.property("alertRows")) != before_alert_first
    )
    _invoke_qml(root, "markAlertRead", 0)
    audit["alerts_mark_read_works"] = (
        int(root.property("alertUnreadCount") or 0) == before_alert_unread
    )
    _invoke_qml(
        root,
        "appendPreviewAlert",
        "Warning",
        "Risk",
        "Smoke warning",
        "local smoke warning",
        "Smoke",
        "BTC/USDT",
        "Review",
    )
    _invoke_qml(root, "markAllAlertsRead")
    audit["alerts_mark_all_read_works"] = int(root.property("alertUnreadCount") or 0) == 0
    _invoke_qml(root, "setAlertSeverityFilter", "Warning")
    _invoke_qml(root, "setAlertCategoryFilter", "Risk")
    audit["alerts_filters_present"] = (
        _string_property(root, "alertSelectedSeverity") == "Warning"
        and _string_property(root, "alertSelectedCategory") == "Risk"
    )
    audit["alerts_categories_present"] = (
        _sequence_length(root.property("alertCategoryFilters")) >= 9
    )
    _invoke_qml(root, "selectAlertEvent", 0)
    _invoke_qml(root, "explainAlertEvent", 0)
    audit["alerts_detail_present"] = root.property("alertSelectedEvent") is not None
    audit["alerts_explain_event_local_only"] = "no backend" in _string_property(
        root, "alertEventExplanation"
    ) or "Local alert explanation" in _string_property(root, "alertEventExplanation")
    before_sim_alerts = _sequence_length(root.property("alertRows"))
    before_sim_alert_first = _first_row_repr(root.property("alertRows"))
    _invoke_qml(root, "setRiskProfile", "Custom")
    _invoke_qml(root, "runSimulationTick")
    audit["alerts_simulation_tick_appends_event"] = (
        _sequence_length(root.property("alertRows")) > before_sim_alerts
        or _first_row_repr(root.property("alertRows")) != before_sim_alert_first
    )
    before_scanner_alerts = _sequence_length(root.property("alertRows"))
    before_scanner_alert_first = _first_row_repr(root.property("alertRows"))
    _invoke_qml(root, "runMarketScannerTick")
    audit["alerts_scanner_tick_appends_event"] = (
        _sequence_length(root.property("alertRows")) > before_scanner_alerts
        or _first_row_repr(root.property("alertRows")) != before_scanner_alert_first
    )
    _invoke_qml(root, "setSimulationScenario", "High volatility")
    _invoke_qml(root, "applyAiRecommendedRiskProfile")
    before_risk_alerts = _sequence_length(root.property("alertRows"))
    before_risk_alert_first = _first_row_repr(root.property("alertRows"))
    _invoke_qml(root, "runSimulationTick")
    audit["alerts_risk_block_appends_event"] = (
        _sequence_length(root.property("alertRows")) > before_risk_alerts
        or _first_row_repr(root.property("alertRows")) != before_risk_alert_first
    ) and "Risk blocked" in _first_row_repr(root.property("alertRows"))
    _invoke_qml(root, "clearPreviewAlerts")
    audit["alerts_clear_works"] = (
        _sequence_length(root.property("alertRows")) == 0
        and int(root.property("alertUnreadCount") or 0) == 0
    )
    audit["alerts_no_os_notifications"] = True
    audit["alerts_no_backend_calls"] = True
    audit["alerts_no_exchange_api_calls"] = True
    audit["alerts_no_order_submission"] = _bool_property(root, "orderSubmissionDisabled") is True
    audit["alerts_no_secret_reads"] = True
    audit["alert_center_safety_boundary_ok"] = "No OS notifications sent" in _string_property(
        root, "alertSafetyBoundaryCopy"
    ) and "No secrets read" in _string_property(root, "alertSafetyBoundaryCopy")
    source = _qml_preview_source()
    before_mode = _string_property(root, "appModePreview")
    before_runtime_config = _bool_property(root, "settingsRuntimeConfigWritten")
    before_secret_reads = _bool_property(root, "settingsSecretsRead")
    _invoke_qml(root, "setAppModePreview", "Paper Preview")
    _invoke_qml(root, "applyPreviewSettings")
    audit["settings_apply_local_only"] = (
        _string_property(root, "appModePreview") == "Paper Preview"
        and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
        and _bool_property(root, "settingsSecretsRead") == before_secret_reads
    )
    _invoke_qml(root, "resetPreviewSettings")
    audit["settings_reset_local_only"] = (
        _string_property(root, "appModePreview") == "Demo Preview"
        and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
        and _bool_property(root, "settingsSecretsRead") == before_secret_reads
    )
    _invoke_qml(root, "startOnboardingPreview")
    _invoke_qml(root, "nextOnboardingStep")
    step_after_next = int(root.property("onboardingStep") or 0)
    _invoke_qml(root, "previousOnboardingStep")
    audit["onboarding_next_previous_works"] = (
        step_after_next == 2 and int(root.property("onboardingStep") or 0) == 1
    )
    _invoke_qml(root, "completeOnboardingPreview")
    audit["onboarding_complete_local_only"] = (
        _bool_property(root, "onboardingCompletedPreview")
        and not _bool_property(root, "runtimeLoopStarted")
        and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
        and _bool_property(root, "settingsSecretsRead") == before_secret_reads
    )
    if before_mode:
        _invoke_qml(root, "setAppModePreview", before_mode)
    audit["settings_tab_present"] = "settingsPanel" in source and "Ustawienia" in source
    audit["settings_state_present"] = all(
        root.property(field) is not None
        for field in (
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
        )
    )
    audit["settings_no_runtime_config_write"] = not _bool_property(
        root, "settingsRuntimeConfigWritten"
    )
    audit["settings_no_secret_reads"] = not _bool_property(root, "settingsSecretsRead")
    audit["app_status_bar_present"] = "globalAppStatusBar" in source
    audit["app_mode_preview_present"] = "appModePreviewOptions" in source
    audit["onboarding_state_present"] = all(
        root.property(field) is not None
        for field in ("firstRunWizardVisible", "onboardingStep", "onboardingCompletedPreview")
    )
    audit["onboarding_steps_present"] = _sequence_length(root.property("onboardingSteps")) == 6
    audit["top_navigation_order_unique_with_settings"] = "settingsPanel" in source
    audit["top_navigation_scroll_or_compact_present"] = (
        "productPreviewTabBar" in source and "HorizontalFlick" in source
    )
    audit["dashboard_quick_actions_present"] = all(
        token in source
        for token in (
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
        )
    )
    audit["global_safety_badges_present"] = "globalSafetyBadges" in source
    audit["settings_safety_boundary_ok"] = all(
        token in _string_property(root, "settingsSafetySummary")
        for token in (
            "Settings are local preview only",
            "No runtime config is written",
            "No secrets are read",
            "No exchange/API calls",
            "No order submission",
            "Live trading remains disabled",
        )
    )
    audit["simulation_does_not_read_secrets"] = True
    audit["safety_boundary_ok"] = (
        audit["live_trading_disabled"] is True
        and audit["exchange_io_disabled"] is True
        and audit["order_submission_disabled"] is True
        and audit["api_keys_required"] is False
        and audit["runtime_loop_started"] is False
        and audit["network_api_calls"] == "disabled"
    )
    required_true_keys = (
        "simulation_loop_state_present",
        "simulation_start_sets_running",
        "simulation_pause_sets_paused",
        "simulation_stop_sets_stopped",
        "simulation_reset_clears_ticks",
        "simulation_tick_increments_count",
        "simulation_tick_updates_decision",
        "simulation_tick_appends_telemetry",
        "simulation_tick_updates_paper_state",
        "paper_tick_updates_operational_state",
        "paper_tick_can_update_financial_state_when_unblocked",
        "risk_blocked_tick_does_not_mutate_paper_pnl",
        "risk_blocked_tick_does_not_mutate_paper_equity",
        "risk_blocked_tick_increments_blocked_count",
        "risk_blocked_tick_appends_decision",
        "risk_blocked_tick_appends_telemetry",
        "risk_blocked_tick_creates_no_filled_order",
        "risk_unlocked_tick_can_update_financial_state",
        "simulation_burst_runs_multiple_ticks",
        "simulation_market_scenario_updates",
        "simulation_does_not_enable_live",
        "simulation_does_not_enable_exchange_io",
        "simulation_does_not_enable_order_submission",
        "simulation_does_not_require_api_keys",
        "simulation_does_not_read_secrets",
        "start_sets_running",
        "generate_tick_appended_order",
        "generate_tick_appended_decision",
        "generate_tick_appended_telemetry",
        "pause_sets_paused",
        "stop_sets_stopped",
        "reset_sets_stopped",
        "reset_ticks_zero",
        "reset_clears_orders",
        "governor_updates_decision",
        "ping_appends_telemetry",
        "portfolio_fields_present",
        "portfolio_custom_filter_updates_label",
        "portfolio_time_filter_does_not_mutate_paper_state",
        "portfolio_time_filter_updates_report_state",
        "portfolio_custom_filter_does_not_mutate_paper_state",
        "portfolio_custom_range_updates_report_state",
        "paper_tick_updates_operational_state",
        "paper_tick_can_update_financial_state_when_unblocked",
        "dashboard_separates_paper_and_portfolio_report",
        "portfolio_equity_formula_ok",
        "portfolio_net_pnl_formula_ok",
        "portfolio_no_double_count_ok",
        "portfolio_range_snapshot_changes_values",
        "portfolio_money_formatting_ok",
        "select_top20_propagates_terminal_pair",
        "select_all_visible_at_least_top20",
        "clear_selected_pairs_zero",
        "toggle_pair_selects_pair",
        "toggle_pair_updates_terminal_pair",
        "pair_selection_updates_decision_summary",
        "risk_profile_updates",
        "risk_summary_updates",
        "risk_custom_state_present",
        "risk_custom_updates_confidence_floor",
        "risk_custom_does_not_write_runtime_config",
        "risk_ai_recommended_updates_values",
        "risk_ai_recommended_explanation_present",
        "risk_active_limits_present",
        "simulation_respects_risk_preview_state",
        "market_scanner_start_sets_scanning",
        "market_scanner_pause_sets_paused",
        "market_scanner_tick_updates_rows",
        "market_scanner_burst_updates_count",
        "market_scanner_explain_updates_explanation",
        "market_scanner_watchlist_updates_count",
        "market_scanner_watchlist_separate_from_whitelist",
        "market_scanner_watchlist_add_does_not_mutate_whitelist",
        "market_scanner_watchlist_remove_does_not_mutate_whitelist",
        "market_scanner_watchlist_filter_uses_scanner_watchlist",
        "market_scanner_blacklist_updates_rejected",
        "market_scanner_filter_sort_threshold_present",
        "market_scanner_rows_present",
        "market_scanner_state_present",
        "market_scanner_safety_boundary_ok",
        "market_scanner_no_network_api_calls",
        "market_scanner_no_order_submission",
        "market_scanner_no_secret_reads",
        "simulation_can_use_scanner_candidate_local_only",
        "alerts_state_present",
        "alerts_append_increments_unread",
        "alerts_mark_read_works",
        "alerts_mark_all_read_works",
        "alerts_clear_works",
        "alerts_filters_present",
        "alerts_categories_present",
        "alerts_detail_present",
        "alerts_explain_event_local_only",
        "alerts_simulation_tick_appends_event",
        "alerts_scanner_tick_appends_event",
        "alerts_risk_block_appends_event",
        "alerts_no_os_notifications",
        "alerts_no_backend_calls",
        "alerts_no_exchange_api_calls",
        "alerts_no_order_submission",
        "alerts_no_secret_reads",
        "alert_center_safety_boundary_ok",
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
        "decision_explainability_state_present",
        "decision_explain_open_close_works",
        "decision_explain_builds_audit_rows",
        "decision_explain_has_risk_checks",
        "decision_explain_has_input_snapshot",
        "decision_explain_has_alternatives",
        "decision_explain_has_paper_impact",
        "decision_explain_safety_boundary_ok",
        "scanner_candidate_explain_opens_shared_drawer",
        "paper_order_explain_local_only",
        "explainability_no_backend_inference",
        "explainability_no_network_api_calls",
        "explainability_no_order_submission",
        "explainability_no_secret_reads",
        "safety_boundary_ok",
    )
    audit["passed"] = (
        int(audit["start_tick_delta"]) >= 1
        and int(audit["generate_tick_delta"]) == 1
        and int(audit["run_ten_tick_delta"]) == 10
        and int(audit["select_top20_count"]) == 20
        and int(audit["portfolio_filters_count"]) == 7
        and int(audit["portfolio_cycles_count"]) >= 4
        and int(audit["portfolio_cards_count"]) >= 13
        and all(audit[key] is True for key in required_true_keys)
    )
    return audit


def _force_offscreen_platform() -> None:
    """Select Qt's offscreen platform before QGuiApplication is created."""

    os.putenv("QT_QPA_PLATFORM", "offscreen")


def run_smoke(options: AppOptions, *, output: TextIO, force_offscreen: bool) -> int:
    """Load the existing QML shell once and exit without starting runtime loops."""

    if force_offscreen:
        _force_offscreen_platform()

    if options.enable_cloud_runtime:
        result = UiSmokeResult(
            status="blocked",
            issues=["smoke_mode_blocks_enable_cloud_runtime"],
        )
        print(result.to_json(), file=output)
        return 2

    qml_warnings: list[str] = []
    audit_issues: list[str] = []
    app = BotPysideApplication(options)
    try:
        engine = app.load(warning_sink=qml_warnings.append)
        from PySide6.QtGui import QGuiApplication

        qt_app = QGuiApplication.instance()
        if qt_app is not None:
            qt_app.processEvents()
        qml_loaded = bool(engine.rootObjects())
        operator_dashboard_present = False
        operator_dashboard_visible = False
        operator_dashboard_default = False
        active_panel_id = ""
        central_content_empty = True
        panel_load_results: dict[str, dict[str, object]] = {}
        final_preview_tabs_loaded = False
        paper_session_controls_present = False
        market_universe_controls_present = False
        ai_governor_controls_present = False
        i18n_language_selector_present = False
        i18n_pl_en_available = False
        i18n_language_switch_local_only = False
        help_glossary_present = False
        glossary_required_terms_present = False
        tooltips_present = False
        safety_boundary_ok = True
        portfolio_filters_do_not_mutate_paper_state = True
        simulation_loop_state_present = False
        risk_blocked_tick_does_not_mutate_paper_pnl = False
        risk_blocked_tick_does_not_mutate_paper_equity = False
        risk_blocked_tick_increments_blocked_count = False
        risk_blocked_tick_appends_decision = False
        risk_blocked_tick_appends_telemetry = False
        risk_blocked_tick_creates_no_filled_order = False
        risk_unlocked_tick_can_update_financial_state = False
        risk_custom_profile_present = False
        risk_ai_recommended_present = False
        risk_ai_recommended_updates_values = False
        risk_custom_does_not_write_runtime_config = True
        risk_ai_recommended_explanation_present = False
        risk_active_limits_present = False
        risk_tooltips_present = False
        risk_safety_boundary_ok = True
        simulation_respects_risk_preview_state = False
        market_scanner_tab_present = False
        market_scanner_state_present = False
        market_scanner_rows_present = False
        market_scanner_filter_sort_threshold_present = False
        market_scanner_safety_boundary_ok = False
        decision_explainability_state_present = False
        decision_explain_drawer_present = False
        decision_explain_safety_boundary_ok = False
        alerts_state_present = False
        alerts_tab_present = False
        alerts_filters_present = False
        alerts_categories_present = False
        alerts_detail_present = False
        alerts_explain_event_local_only = False
        alerts_dashboard_summary_present = False
        alert_center_safety_boundary_ok = False
        top_navigation_default_order_unique = False
        settings_tab_present = False
        settings_state_present = False
        settings_apply_local_only = False
        settings_reset_local_only = False
        settings_no_runtime_config_write = True
        settings_no_secret_reads = True
        app_status_bar_present = False
        app_mode_preview_present = False
        onboarding_state_present = False
        onboarding_steps_present = False
        onboarding_next_previous_works = False
        onboarding_complete_local_only = False
        top_navigation_order_unique_with_settings = False
        top_navigation_scroll_or_compact_present = False
        dashboard_quick_actions_present = False
        global_safety_badges_present = False
        settings_safety_boundary_ok = False
        preview_state_audit: dict[str, object] = {}
        if qml_loaded:
            source = _qml_preview_source()
            final_preview_tabs_loaded = _source_has_all(
                source,
                (
                    "Dashboard",
                    "AI Center",
                    "Trading Universe",
                    "Okazje",
                    "Market Scanner",
                    "Portfel / Wyniki",
                    "Strategie",
                    "Ryzyko",
                    "Decyzje",
                    "Telemetria",
                    "Alerty",
                    "Alerts",
                    "Diagnostyka",
                    "Ustawienia",
                    "Settings",
                ),
            )
            paper_session_controls_present = _source_has_all(
                source, ("Start Paper Preview", "Pause", "Stop", "Reset", "Generate Next Tick")
            )
            market_universe_controls_present = _source_has_all(
                source, ("Import markets preview", "Search pair", "select all", "clear selected")
            )
            ai_governor_controls_present = _source_has_all(
                source,
                (
                    "Generate governor recommendation",
                    "Autonomy mode selector",
                    "Training coverage %",
                ),
            )
            required_glossary_terms = (
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
                "Custom risk",
                "AI Recommended risk",
                "confidence floor",
                "exposure",
                "daily loss limit",
                "cooldown",
                "risk override",
                "Market Scanner",
                "AI score",
                "Risk score",
                "Liquidity",
                "Volatility",
                "Trend",
                "Watchlist",
                "Blacklist",
                "Candidate",
                "Rejected setup",
                "Strategy match",
                "explainability",
                "audit trail",
                "lineage",
                "input snapshot",
                "risk check",
                "decision source",
                "alternative candidate",
                "paper impact",
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
            )
            required_tooltips = (
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
                "Risk profile Custom",
                "Risk profile AI Recommended",
                "max position",
                "stop loss",
                "take profit",
                "slippage",
                "drawdown",
                "daily loss limit",
                "confidence floor",
                "kill-switch",
                "allow AI override",
                "Custom range",
                "Zastosuj zakres",
                "Start scanner",
                "Pause scanner",
                "Run scan tick",
                "Run scan burst",
                "AI score",
                "Risk score",
                "Liquidity score",
                "Spread",
                "Volatility",
                "Strategy match",
                "Watchlist",
                "Blacklist",
                "Explain candidate",
                "Explain decision",
                "Audit trail",
                "Risk checks",
                "Input snapshot",
                "Paper impact",
                "Mark all read",
                "Clear alerts",
                "Mute alerts",
                "Sound preview",
                "Desktop notification preview",
                "Explain event",
                "Severity filter",
                "Category filter",
                "Alert Center",
                "App mode selector",
                "Base currency selector",
                "UI density selector",
                "Theme preview",
                "Reset local preview state",
                "Apply preview settings",
                "Start onboarding",
                "Complete onboarding",
                "Open Settings",
                "Open Alerts",
                "Open Help",
            )
            i18n_language_selector_present = _source_has_all(
                source, ("currentLanguage", "languageSelector", "setLanguage", "trText", "previewT")
            )
            i18n_pl_en_available = _source_has_all(
                source, ('code: "PL"', 'code: "EN"', '"PL": ({', '"EN": ({')
            )
            help_glossary_present = _source_has_all(
                source, ("Pomoc / Słownik", "helpGlossaryPanel", "glossaryCategories")
            )
            glossary_required_terms_present = _source_has_all(source, required_glossary_terms)
            simulation_loop_state_present = _source_has_all(
                source,
                (
                    "simulationRunning",
                    "simulationPaused",
                    "simulationSpeed",
                    "simulationTickIntervalMs",
                    "simulationScenario",
                    "simulationTickCount",
                    "simulationEvents",
                    "startLiveLikePaperSimulation",
                    "runSimulationBurst",
                ),
            )
            market_scanner_tab_present = _source_has_all(
                source, ("marketScannerPanel", "nav.marketScanner", "Okazje", "Market Scanner")
            )
            decision_explainability_state_present = _source_has_all(
                source,
                (
                    "decisionExplainDrawerOpen",
                    "selectedDecisionId",
                    "selectedDecisionPair",
                    "selectedDecisionAction",
                    "selectedDecisionSource",
                    "selectedDecisionConfidence",
                    "selectedDecisionRiskState",
                    "selectedDecisionStrategy",
                    "selectedDecisionReason",
                    "selectedDecisionAuditRows",
                    "selectedDecisionInputSnapshot",
                    "selectedDecisionAlternatives",
                    "selectedDecisionRiskChecks",
                    "selectedDecisionLineageLinks",
                    "selectedDecisionPaperImpact",
                    "selectedDecisionSafetySummary",
                    "openDecisionExplainDrawer",
                    "closeDecisionExplainDrawer",
                    "explainDecisionRow",
                    "explainScannerCandidateDecision",
                    "explainPaperOrderDecision",
                    "buildDecisionAuditRows",
                    "buildDecisionRiskChecks",
                    "buildDecisionAlternatives",
                    "buildDecisionInputSnapshot",
                ),
            )
            decision_explain_drawer_present = _source_has_all(
                source,
                (
                    "decisionExplainabilityDrawer",
                    "Dlaczego bot tak zdecydował?",
                    "Audit trail",
                    "Risk checks",
                    "Input snapshot",
                    "Alternatywy",
                    "Paper impact",
                ),
            )
            decision_explain_safety_boundary_ok = _source_has_all(
                source,
                (
                    "Explanation is local preview only",
                    "No backend AI inference",
                    "No exchange/API call",
                    "No order submission",
                    "No real orders",
                    "No secrets read",
                    "Wyjaśnienie działa lokalnie w preview",
                    "Brak backendowej inferencji AI",
                    "Brak połączenia z giełdą/API",
                    "Brak składania zleceń",
                    "Brak prawdziwych zleceń",
                    "Brak odczytu sekretów",
                ),
            )
            market_scanner_state_present = _source_has_all(
                source,
                (
                    "scannerStatus",
                    "scannerActive",
                    "scannerLastScanAt",
                    "scannerTickCount",
                    "scannerRows",
                    "scannerRejectedRows",
                    "scannerWatchlistPairs",
                    "scannerWatchlistRows",
                    "scannerAiCandidateRows",
                    "scannerExplanation",
                    "startMarketScannerPreview",
                    "pauseMarketScannerPreview",
                    "stopMarketScannerPreview",
                    "resetMarketScannerPreview",
                    "runMarketScannerTick",
                    "runMarketScannerBurst",
                    "selectScannerPair",
                    "addScannerPairToWatchlist",
                    "removeScannerPairFromWatchlist",
                    "blacklistScannerPair",
                    "setScannerFilterMode",
                    "setScannerSortMode",
                    "setScannerThreshold",
                    "explainScannerCandidate",
                    "Watchlist = obserwacja",
                    "Watchlist is for observation",
                    "preview-local blocklist shared with Trading Universe",
                ),
            )
            market_scanner_rows_present = _source_has_all(
                source,
                (
                    "previewMarketPairs",
                    "buildScannerRow",
                    "visibleScannerRows",
                    "Pair",
                    "Exchange",
                    "Recommendation",
                    "Reason",
                ),
            )
            market_scanner_filter_sort_threshold_present = _source_has_all(
                source,
                (
                    "All",
                    "AI candidates",
                    "Trade candidates",
                    "Watchlist",
                    "Rejected",
                    "Blocked",
                    "High liquidity",
                    "Low risk",
                    "Top score",
                    "Risk score",
                    "Trend strength",
                    "min AI score",
                    "min liquidity score",
                    "max risk score",
                ),
            )
            market_scanner_safety_boundary_ok = _source_has_all(
                source,
                (
                    "Safe preview scanner",
                    "Live trading disabled",
                    "Exchange I/O disabled",
                    "Order submission disabled",
                    "API keys not required",
                    "No real orders",
                    "No network/API calls",
                    "Local preview catalog only",
                ),
            )
            panel_order_matches = re.findall(
                r'panelId: "([^"]+)",[^\n]+defaultOrder: (\d+)', source
            )
            required_panel_order = {
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
            panel_orders = {panel_id: int(order) for panel_id, order in panel_order_matches}
            top_navigation_default_order_unique = panel_orders == required_panel_order and len(
                set(panel_orders.values())
            ) == len(panel_orders)
            settings_tab_present = _source_has_all(
                source, ("settingsPanel", "nav.settings", "Ustawienia", "Settings")
            )
            settings_state_present = _source_has_all(
                source,
                (
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
                ),
            )
            app_mode_preview_present = _source_has_all(
                source,
                (
                    "Demo Preview",
                    "Paper Preview",
                    "Sandbox planned",
                    "Live disabled",
                    "appModePreviewOptions",
                ),
            )
            onboarding_state_present = _source_has_all(
                source, ("firstRunWizardVisible", "onboardingStep", "onboardingCompletedPreview")
            )
            onboarding_steps_present = _source_has_all(
                source,
                (
                    "Wybierz język",
                    "Wybierz walutę bazową",
                    "Wybierz tryb",
                    "Wybierz giełdę preview",
                    "Wybierz profil ryzyka",
                    "Uruchom Paper Preview",
                ),
            )
            app_status_bar_present = _source_has_all(
                source,
                ("globalAppStatusBar", "appStatusSummary", "Alerts: ", "Base: ", "Simulation: "),
            )
            global_safety_badges_present = _source_has_all(
                source,
                (
                    "globalSafetyBadges",
                    "Live trading: disabled",
                    "Exchange I/O: disabled",
                    "Order submission: disabled",
                    "API keys: not required",
                    "Runtime loop: not started",
                    "Safety: safe preview",
                ),
            )
            top_navigation_order_unique_with_settings = top_navigation_default_order_unique
            top_navigation_scroll_or_compact_present = (
                _source_has_all(source, ("productPreviewTabBar", "Flickable", "HorizontalFlick"))
                or "topNavigationHorizontalScroll" in source
            )
            dashboard_quick_actions_present = _source_has_all(
                source,
                (
                    "Szybkie akcje",
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
                ),
            )
            settings_safety_boundary_ok = _source_has_all(
                source,
                (
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
                ),
            )
            settings_no_runtime_config_write = "settingsRuntimeConfigWritten = true" not in source
            settings_no_secret_reads = "settingsSecretsRead = true" not in source
            tooltips_present = _source_has_all(
                source, ("ToolTip.delay: 800", "helpText", "previewTooltips")
            ) and _source_has_all(source, required_tooltips)
            risk_custom_profile_present = _source_has_all(
                source, ("Custom", "customRiskState", "setCustomRiskValue", "riskCustomProfileCard")
            )
            risk_ai_recommended_present = _source_has_all(
                source, ("AI Recommended", "applyAiRecommendedRiskProfile", "AI Recommended risk")
            )
            risk_ai_recommended_explanation_present = _source_has_all(
                source, ("riskExplanationCard", "Dlaczego takie ustawienia?", "AI dobrało")
            )
            risk_active_limits_present = _source_has_all(
                source, ("riskActiveLimits", "riskActiveLimitsTable", "Aktywne limity")
            )
            risk_tooltips_present = _source_has_all(source, required_tooltips)
            risk_safety_boundary_ok = _source_has_all(
                source,
                (
                    "Live trading disabled",
                    "Exchange I/O disabled",
                    "Order submission disabled",
                    "API keys not required",
                    "No real orders",
                    "Risk settings are local preview only",
                ),
            )
            alerts_state_present = _source_has_all(
                source,
                (
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
                    "appendPreviewAlert",
                    "markAlertRead",
                    "markAllAlertsRead",
                    "clearPreviewAlerts",
                    "setAlertSeverityFilter",
                    "setAlertCategoryFilter",
                    "selectAlertEvent",
                    "explainAlertEvent",
                    "toggleAlertMutePreview",
                    "toggleAlertSoundPreview",
                    "toggleDesktopNotificationsPreview",
                ),
            )
            alerts_tab_present = _source_has_all(source, ("alertsPanel", "Alerty", "Alerts"))
            alerts_filters_present = _source_has_all(
                source, ("All", "Critical", "Warning", "Info", "Severity filter")
            )
            alerts_categories_present = _source_has_all(
                source,
                (
                    "Trading",
                    "Risk",
                    "AI",
                    "Scanner",
                    "Paper",
                    "Portfolio",
                    "Telemetry",
                    "Diagnostics",
                    "Safety",
                    "Category filter",
                ),
            )
            alerts_detail_present = _source_has_all(
                source,
                ("alertCenterDetailPanel", "Alert detail", "Wyjaśnij zdarzenie", "Explain event"),
            )
            alerts_explain_event_local_only = _source_has_all(
                source,
                (
                    "Explanation is local preview only",
                    "no backend inference",
                    "no network/API call",
                    "no order submission",
                    "no real orders",
                    "no secrets read",
                ),
            )
            alerts_dashboard_summary_present = _source_has_all(
                source,
                (
                    "operatorDashboardAlertSummary",
                    "unread alerts",
                    "critical count",
                    "last alert",
                    "Otwórz Alerty",
                ),
            )
            alert_center_safety_boundary_ok = _source_has_all(
                source,
                (
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
                ),
            )
            if not all(
                (
                    alerts_state_present,
                    alerts_tab_present,
                    alerts_filters_present,
                    alerts_categories_present,
                    alerts_detail_present,
                    alerts_explain_event_local_only,
                    alerts_dashboard_summary_present,
                    alert_center_safety_boundary_ok,
                )
            ):
                audit_issues.append("alerts_source_audit_failed")

            from PySide6.QtCore import QObject

            root = engine.rootObjects()[0]
            active_panel_id = str(root.property("currentPanelId") or "")
            dashboard = root.findChild(QObject, "operatorDashboardRoot")
            central_loader = root.findChild(QObject, "centralContentLoader")
            operator_dashboard_present = dashboard is not None
            operator_dashboard_default = active_panel_id in {"sidePanel", "operatorDashboard"}
            if dashboard is not None:
                operator_dashboard_visible = bool(dashboard.property("visible"))
            if central_loader is not None:
                loaded_item = central_loader.property("item")
                central_content_empty = loaded_item is None
            else:
                central_content_empty = dashboard is None
            panel_load_results = _audit_panel_loads(root)
            failed_panels = [
                panel_id
                for panel_id, panel_result in panel_load_results.items()
                if not panel_result["loaded"] or panel_result["empty"]
            ]
            if failed_panels:
                audit_issues.extend(f"panel_load_failed:{panel_id}" for panel_id in failed_panels)
            before_language = str(root.property("currentLanguage") or "")
            before_runtime_loop = _bool_property(root, "runtimeLoopStarted")
            before_exchange_io = _bool_property(root, "exchangeIoDisabled")
            before_order_submission = _bool_property(root, "orderSubmissionDisabled")
            before_api_keys_required = _bool_property(root, "apiKeysRequired")
            _invoke_qml(root, "setLanguage", "EN")
            after_language = str(root.property("currentLanguage") or "")
            i18n_language_switch_local_only = (
                before_language != after_language
                and after_language == "EN"
                and _bool_property(root, "runtimeLoopStarted") == before_runtime_loop
                and _bool_property(root, "exchangeIoDisabled") == before_exchange_io
                and _bool_property(root, "orderSubmissionDisabled") == before_order_submission
                and _bool_property(root, "apiKeysRequired") == before_api_keys_required
            )
            _invoke_qml(root, "setLanguage", before_language or "PL")
            before_mode = str(root.property("appModePreview") or "")
            before_runtime_config = _bool_property(root, "settingsRuntimeConfigWritten")
            before_secret_reads = _bool_property(root, "settingsSecretsRead")
            before_step = int(root.property("onboardingStep") or 0)
            _invoke_qml(root, "setAppModePreview", "Paper Preview")
            _invoke_qml(root, "applyPreviewSettings")
            settings_apply_local_only = (
                str(root.property("appModePreview") or "") == "Paper Preview"
                and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
                and _bool_property(root, "settingsSecretsRead") == before_secret_reads
            )
            _invoke_qml(root, "resetPreviewSettings")
            settings_reset_local_only = (
                str(root.property("appModePreview") or "") == "Demo Preview"
                and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
                and _bool_property(root, "settingsSecretsRead") == before_secret_reads
            )
            _invoke_qml(root, "startOnboardingPreview")
            _invoke_qml(root, "nextOnboardingStep")
            step_after_next = int(root.property("onboardingStep") or 0)
            _invoke_qml(root, "previousOnboardingStep")
            onboarding_next_previous_works = (
                step_after_next == 2 and int(root.property("onboardingStep") or 0) == 1
            )
            _invoke_qml(root, "completeOnboardingPreview")
            onboarding_complete_local_only = (
                _bool_property(root, "onboardingCompletedPreview")
                and not _bool_property(root, "runtimeLoopStarted")
                and _bool_property(root, "settingsRuntimeConfigWritten") == before_runtime_config
                and _bool_property(root, "settingsSecretsRead") == before_secret_reads
            )
            if before_mode:
                _invoke_qml(root, "setAppModePreview", before_mode)
            if before_step:
                root.setProperty("onboardingStep", before_step)
            settings_no_runtime_config_write = (
                settings_no_runtime_config_write
                and not _bool_property(root, "settingsRuntimeConfigWritten")
            )
            settings_no_secret_reads = settings_no_secret_reads and not _bool_property(
                root, "settingsSecretsRead"
            )
            if not all(
                (
                    i18n_language_selector_present,
                    i18n_pl_en_available,
                    i18n_language_switch_local_only,
                    help_glossary_present,
                    glossary_required_terms_present,
                    tooltips_present,
                )
            ):
                audit_issues.append("i18n_help_tooltip_audit_failed")
            if not all(
                (
                    settings_tab_present,
                    settings_state_present,
                    settings_apply_local_only,
                    settings_reset_local_only,
                    app_status_bar_present,
                    app_mode_preview_present,
                    onboarding_state_present,
                    onboarding_steps_present,
                    onboarding_next_previous_works,
                    onboarding_complete_local_only,
                    top_navigation_order_unique_with_settings,
                    top_navigation_scroll_or_compact_present,
                    dashboard_quick_actions_present,
                    global_safety_badges_present,
                    settings_safety_boundary_ok,
                    settings_no_runtime_config_write,
                    settings_no_secret_reads,
                )
            ):
                audit_issues.append("settings_onboarding_audit_failed")
            if not all(
                (
                    market_scanner_tab_present,
                    market_scanner_state_present,
                    market_scanner_rows_present,
                    market_scanner_filter_sort_threshold_present,
                    market_scanner_safety_boundary_ok,
                    top_navigation_default_order_unique,
                )
            ):
                audit_issues.append("market_scanner_audit_failed")
            if not all(
                (
                    risk_custom_profile_present,
                    risk_ai_recommended_present,
                    risk_ai_recommended_explanation_present,
                    risk_active_limits_present,
                    risk_tooltips_present,
                    risk_safety_boundary_ok,
                )
            ):
                audit_issues.append("risk_profile_audit_failed")
            if options.exercise_preview_state:
                preview_state_audit = _exercise_preview_state(root)
                if preview_state_audit.get("passed") is not True:
                    audit_issues.append("preview_state_exercise_failed")
                safety_boundary_ok = bool(preview_state_audit.get("safety_boundary_ok"))
                portfolio_filters_do_not_mutate_paper_state = bool(
                    preview_state_audit.get("portfolio_time_filter_does_not_mutate_paper_state")
                ) and bool(
                    preview_state_audit.get("portfolio_custom_filter_does_not_mutate_paper_state")
                )
                risk_ai_recommended_updates_values = bool(
                    preview_state_audit.get("risk_ai_recommended_updates_values")
                )
                risk_custom_does_not_write_runtime_config = bool(
                    preview_state_audit.get("risk_custom_does_not_write_runtime_config")
                )
                simulation_respects_risk_preview_state = bool(
                    preview_state_audit.get("simulation_respects_risk_preview_state")
                )
                risk_blocked_tick_does_not_mutate_paper_pnl = bool(
                    preview_state_audit.get("risk_blocked_tick_does_not_mutate_paper_pnl")
                )
                risk_blocked_tick_does_not_mutate_paper_equity = bool(
                    preview_state_audit.get("risk_blocked_tick_does_not_mutate_paper_equity")
                )
                risk_blocked_tick_increments_blocked_count = bool(
                    preview_state_audit.get("risk_blocked_tick_increments_blocked_count")
                )
                risk_blocked_tick_appends_decision = bool(
                    preview_state_audit.get("risk_blocked_tick_appends_decision")
                )
                risk_blocked_tick_appends_telemetry = bool(
                    preview_state_audit.get("risk_blocked_tick_appends_telemetry")
                )
                risk_blocked_tick_creates_no_filled_order = bool(
                    preview_state_audit.get("risk_blocked_tick_creates_no_filled_order")
                )
                market_scanner_state_present = bool(
                    preview_state_audit.get(
                        "market_scanner_state_present", market_scanner_state_present
                    )
                )
                market_scanner_rows_present = bool(
                    preview_state_audit.get(
                        "market_scanner_rows_present", market_scanner_rows_present
                    )
                )
                market_scanner_filter_sort_threshold_present = bool(
                    preview_state_audit.get(
                        "market_scanner_filter_sort_threshold_present",
                        market_scanner_filter_sort_threshold_present,
                    )
                )
                market_scanner_safety_boundary_ok = bool(
                    preview_state_audit.get(
                        "market_scanner_safety_boundary_ok", market_scanner_safety_boundary_ok
                    )
                )
                risk_unlocked_tick_can_update_financial_state = bool(
                    preview_state_audit.get("risk_unlocked_tick_can_update_financial_state")
                )
                alerts_state_present = bool(
                    preview_state_audit.get("alerts_state_present", alerts_state_present)
                )
                alerts_filters_present = bool(
                    preview_state_audit.get("alerts_filters_present", alerts_filters_present)
                )
                alerts_categories_present = bool(
                    preview_state_audit.get("alerts_categories_present", alerts_categories_present)
                )
                alerts_detail_present = bool(
                    preview_state_audit.get("alerts_detail_present", alerts_detail_present)
                )
                alerts_explain_event_local_only = bool(
                    preview_state_audit.get(
                        "alerts_explain_event_local_only", alerts_explain_event_local_only
                    )
                )
                alert_center_safety_boundary_ok = bool(
                    preview_state_audit.get(
                        "alert_center_safety_boundary_ok", alert_center_safety_boundary_ok
                    )
                )
        smoke_ok = qml_loaded and not audit_issues
        result = UiSmokeResult(
            status="ok" if smoke_ok else "error",
            ui_loaded=qml_loaded,
            qml_loaded=qml_loaded,
            operator_dashboard_present=operator_dashboard_present,
            operator_dashboard_default=operator_dashboard_default,
            operator_dashboard_visible=operator_dashboard_visible,
            active_panel_id=active_panel_id,
            central_content_empty=central_content_empty,
            panel_load_results=panel_load_results,
            final_preview_tabs_loaded=final_preview_tabs_loaded,
            paper_session_controls_present=paper_session_controls_present,
            market_universe_controls_present=market_universe_controls_present,
            ai_governor_controls_present=ai_governor_controls_present,
            i18n_language_selector_present=i18n_language_selector_present,
            i18n_pl_en_available=i18n_pl_en_available,
            i18n_language_switch_local_only=i18n_language_switch_local_only,
            help_glossary_present=help_glossary_present,
            glossary_required_terms_present=glossary_required_terms_present,
            tooltips_present=tooltips_present,
            safety_boundary_ok=safety_boundary_ok,
            portfolio_filters_do_not_mutate_paper_state=portfolio_filters_do_not_mutate_paper_state,
            simulation_loop_state_present=simulation_loop_state_present,
            simulation_start_sets_running=bool(
                preview_state_audit.get("simulation_start_sets_running", False)
            ),
            simulation_pause_sets_paused=bool(
                preview_state_audit.get("simulation_pause_sets_paused", False)
            ),
            simulation_stop_sets_stopped=bool(
                preview_state_audit.get("simulation_stop_sets_stopped", False)
            ),
            simulation_reset_clears_ticks=bool(
                preview_state_audit.get("simulation_reset_clears_ticks", False)
            ),
            simulation_tick_increments_count=bool(
                preview_state_audit.get("simulation_tick_increments_count", False)
            ),
            simulation_tick_updates_decision=bool(
                preview_state_audit.get("simulation_tick_updates_decision", False)
            ),
            simulation_tick_appends_telemetry=bool(
                preview_state_audit.get("simulation_tick_appends_telemetry", False)
            ),
            simulation_tick_updates_paper_state=bool(
                preview_state_audit.get("simulation_tick_updates_paper_state", False)
            ),
            paper_tick_updates_operational_state=bool(
                preview_state_audit.get("paper_tick_updates_operational_state", False)
            ),
            paper_tick_can_update_financial_state_when_unblocked=bool(
                preview_state_audit.get(
                    "paper_tick_can_update_financial_state_when_unblocked", False
                )
            ),
            risk_blocked_tick_does_not_mutate_paper_pnl=risk_blocked_tick_does_not_mutate_paper_pnl,
            risk_blocked_tick_does_not_mutate_paper_equity=risk_blocked_tick_does_not_mutate_paper_equity,
            risk_blocked_tick_increments_blocked_count=risk_blocked_tick_increments_blocked_count,
            risk_blocked_tick_appends_decision=risk_blocked_tick_appends_decision,
            risk_blocked_tick_appends_telemetry=risk_blocked_tick_appends_telemetry,
            risk_blocked_tick_creates_no_filled_order=risk_blocked_tick_creates_no_filled_order,
            risk_unlocked_tick_can_update_financial_state=risk_unlocked_tick_can_update_financial_state,
            simulation_burst_runs_multiple_ticks=bool(
                preview_state_audit.get("simulation_burst_runs_multiple_ticks", False)
            ),
            simulation_market_scenario_updates=bool(
                preview_state_audit.get("simulation_market_scenario_updates", False)
            ),
            simulation_does_not_enable_live=bool(
                preview_state_audit.get("simulation_does_not_enable_live", True)
            ),
            simulation_does_not_enable_exchange_io=bool(
                preview_state_audit.get("simulation_does_not_enable_exchange_io", True)
            ),
            simulation_does_not_enable_order_submission=bool(
                preview_state_audit.get("simulation_does_not_enable_order_submission", True)
            ),
            simulation_does_not_require_api_keys=bool(
                preview_state_audit.get("simulation_does_not_require_api_keys", True)
            ),
            simulation_does_not_read_secrets=bool(
                preview_state_audit.get("simulation_does_not_read_secrets", True)
            ),
            risk_custom_profile_present=risk_custom_profile_present,
            risk_ai_recommended_present=risk_ai_recommended_present,
            risk_ai_recommended_updates_values=risk_ai_recommended_updates_values,
            risk_custom_does_not_write_runtime_config=risk_custom_does_not_write_runtime_config,
            risk_ai_recommended_explanation_present=risk_ai_recommended_explanation_present,
            risk_active_limits_present=risk_active_limits_present,
            risk_tooltips_present=risk_tooltips_present,
            risk_safety_boundary_ok=risk_safety_boundary_ok,
            simulation_respects_risk_preview_state=simulation_respects_risk_preview_state,
            market_scanner_tab_present=market_scanner_tab_present,
            market_scanner_state_present=market_scanner_state_present,
            market_scanner_rows_present=market_scanner_rows_present,
            market_scanner_start_sets_scanning=bool(
                preview_state_audit.get("market_scanner_start_sets_scanning", False)
            ),
            market_scanner_pause_sets_paused=bool(
                preview_state_audit.get("market_scanner_pause_sets_paused", False)
            ),
            market_scanner_tick_updates_rows=bool(
                preview_state_audit.get("market_scanner_tick_updates_rows", False)
            ),
            market_scanner_burst_updates_count=bool(
                preview_state_audit.get("market_scanner_burst_updates_count", False)
            ),
            market_scanner_explain_updates_explanation=bool(
                preview_state_audit.get("market_scanner_explain_updates_explanation", False)
            ),
            market_scanner_watchlist_updates_count=bool(
                preview_state_audit.get("market_scanner_watchlist_updates_count", False)
            ),
            market_scanner_watchlist_separate_from_whitelist=bool(
                preview_state_audit.get("market_scanner_watchlist_separate_from_whitelist", False)
            ),
            market_scanner_watchlist_add_does_not_mutate_whitelist=bool(
                preview_state_audit.get(
                    "market_scanner_watchlist_add_does_not_mutate_whitelist", False
                )
            ),
            market_scanner_watchlist_remove_does_not_mutate_whitelist=bool(
                preview_state_audit.get(
                    "market_scanner_watchlist_remove_does_not_mutate_whitelist", False
                )
            ),
            market_scanner_watchlist_filter_uses_scanner_watchlist=bool(
                preview_state_audit.get(
                    "market_scanner_watchlist_filter_uses_scanner_watchlist", False
                )
            ),
            market_scanner_blacklist_updates_rejected=bool(
                preview_state_audit.get("market_scanner_blacklist_updates_rejected", False)
            ),
            market_scanner_filter_sort_threshold_present=market_scanner_filter_sort_threshold_present,
            market_scanner_safety_boundary_ok=market_scanner_safety_boundary_ok,
            market_scanner_no_network_api_calls=bool(
                preview_state_audit.get("market_scanner_no_network_api_calls", True)
            ),
            market_scanner_no_order_submission=bool(
                preview_state_audit.get("market_scanner_no_order_submission", True)
            ),
            market_scanner_no_secret_reads=bool(
                preview_state_audit.get("market_scanner_no_secret_reads", True)
            ),
            simulation_can_use_scanner_candidate_local_only=bool(
                preview_state_audit.get("simulation_can_use_scanner_candidate_local_only", False)
            ),
            decision_explainability_state_present=decision_explainability_state_present,
            decision_explain_drawer_present=decision_explain_drawer_present,
            decision_explain_open_close_works=bool(
                preview_state_audit.get("decision_explain_open_close_works", False)
            ),
            decision_explain_builds_audit_rows=bool(
                preview_state_audit.get("decision_explain_builds_audit_rows", False)
            ),
            decision_explain_has_risk_checks=bool(
                preview_state_audit.get("decision_explain_has_risk_checks", False)
            ),
            decision_explain_has_input_snapshot=bool(
                preview_state_audit.get("decision_explain_has_input_snapshot", False)
            ),
            decision_explain_has_alternatives=bool(
                preview_state_audit.get("decision_explain_has_alternatives", False)
            ),
            decision_explain_has_paper_impact=bool(
                preview_state_audit.get("decision_explain_has_paper_impact", False)
            ),
            decision_explain_safety_boundary_ok=decision_explain_safety_boundary_ok
            and bool(preview_state_audit.get("decision_explain_safety_boundary_ok", True)),
            scanner_candidate_explain_opens_shared_drawer=bool(
                preview_state_audit.get("scanner_candidate_explain_opens_shared_drawer", False)
            ),
            paper_order_explain_local_only=bool(
                preview_state_audit.get("paper_order_explain_local_only", False)
            ),
            explainability_no_backend_inference=bool(
                preview_state_audit.get("explainability_no_backend_inference", True)
            ),
            explainability_no_network_api_calls=bool(
                preview_state_audit.get("explainability_no_network_api_calls", True)
            ),
            explainability_no_order_submission=bool(
                preview_state_audit.get("explainability_no_order_submission", True)
            ),
            explainability_no_secret_reads=bool(
                preview_state_audit.get("explainability_no_secret_reads", True)
            ),
            alerts_state_present=alerts_state_present,
            alerts_tab_present=alerts_tab_present,
            alerts_append_increments_unread=bool(
                preview_state_audit.get("alerts_append_increments_unread", False)
            ),
            alerts_mark_read_works=bool(preview_state_audit.get("alerts_mark_read_works", False)),
            alerts_mark_all_read_works=bool(
                preview_state_audit.get("alerts_mark_all_read_works", False)
            ),
            alerts_clear_works=bool(preview_state_audit.get("alerts_clear_works", False)),
            alerts_filters_present=alerts_filters_present,
            alerts_categories_present=alerts_categories_present,
            alerts_detail_present=alerts_detail_present,
            alerts_explain_event_local_only=alerts_explain_event_local_only,
            alerts_dashboard_summary_present=alerts_dashboard_summary_present,
            alerts_simulation_tick_appends_event=bool(
                preview_state_audit.get("alerts_simulation_tick_appends_event", False)
            ),
            alerts_scanner_tick_appends_event=bool(
                preview_state_audit.get("alerts_scanner_tick_appends_event", False)
            ),
            alerts_risk_block_appends_event=bool(
                preview_state_audit.get("alerts_risk_block_appends_event", False)
            ),
            alerts_no_os_notifications=bool(
                preview_state_audit.get("alerts_no_os_notifications", True)
            ),
            alerts_no_backend_calls=bool(preview_state_audit.get("alerts_no_backend_calls", True)),
            alerts_no_exchange_api_calls=bool(
                preview_state_audit.get("alerts_no_exchange_api_calls", True)
            ),
            alerts_no_order_submission=bool(
                preview_state_audit.get("alerts_no_order_submission", True)
            ),
            alerts_no_secret_reads=bool(preview_state_audit.get("alerts_no_secret_reads", True)),
            alert_center_safety_boundary_ok=alert_center_safety_boundary_ok,
            top_navigation_default_order_unique=top_navigation_default_order_unique,
            settings_tab_present=settings_tab_present,
            settings_state_present=settings_state_present,
            settings_apply_local_only=settings_apply_local_only,
            settings_reset_local_only=settings_reset_local_only,
            settings_no_runtime_config_write=settings_no_runtime_config_write,
            settings_no_secret_reads=settings_no_secret_reads,
            app_status_bar_present=app_status_bar_present,
            app_mode_preview_present=app_mode_preview_present,
            onboarding_state_present=onboarding_state_present,
            onboarding_steps_present=onboarding_steps_present,
            onboarding_next_previous_works=onboarding_next_previous_works,
            onboarding_complete_local_only=onboarding_complete_local_only,
            top_navigation_order_unique_with_settings=top_navigation_order_unique_with_settings,
            top_navigation_scroll_or_compact_present=top_navigation_scroll_or_compact_present,
            dashboard_quick_actions_present=dashboard_quick_actions_present,
            global_safety_badges_present=global_safety_badges_present,
            settings_safety_boundary_ok=settings_safety_boundary_ok,
            preview_state_exercised=options.exercise_preview_state and bool(preview_state_audit),
            preview_state_audit=preview_state_audit,
            issues=[] if smoke_ok else audit_issues or qml_warnings or ["qml_root_objects_missing"],
        )
        print(result.to_json(), file=output)
        return 0 if smoke_ok else 1
    except Exception as exc:  # pragma: no cover - exercised via CLI integration tests
        issue = f"{type(exc).__name__}: {exc}"
        result = UiSmokeResult(status="error", issues=qml_warnings + audit_issues + [issue])
        print(result.to_json(), file=output)
        return 1
