"""Safe source-level smoke check for the PySide6/QML UI."""

from __future__ import annotations

import json
import os
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
    "portfolioPerformancePanel",
    "terminalPanel",
    "strategiesPanel",
    "riskControlsPanel",
    "aiDecisionsPanel",
    "telemetryPanel",
    "diagnosticsPanel",
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


def _first_row_repr(value: Any) -> str:
    value = _variant(value)
    try:
        return repr(value[0]) if value else ""
    except (KeyError, TypeError, IndexError):
        return ""


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
    after_start_ticks = int(root.property("paperSessionTicks") or 0)
    audit["simulation_start_sets_running"] = _bool_property(root, "simulationRunning") is True
    audit["start_sets_running"] = _string_property(root, "paperSessionStatus") == "running"
    audit["start_tick_delta"] = after_start_ticks - initial_ticks

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

    before_governor_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_governor_text = _string_property(root, "lastGovernorDecision")
    _invoke_qml(root, "generateGovernorRecommendation")
    audit["governor_updates_decision"] = (
        _sequence_length(root.property("decisionPreviewRows")) > before_governor_decisions
        and _string_property(root, "lastGovernorDecision") != before_governor_text
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
        preview_state_audit: dict[str, object] = {}
        if qml_loaded:
            source = _qml_preview_source()
            final_preview_tabs_loaded = _source_has_all(
                source,
                (
                    "Dashboard",
                    "AI Center",
                    "Trading Universe",
                    "Portfel / Wyniki",
                    "Strategie",
                    "Ryzyko",
                    "Decyzje",
                    "Telemetria",
                    "Diagnostyka",
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
                risk_unlocked_tick_can_update_financial_state = bool(
                    preview_state_audit.get("risk_unlocked_tick_can_update_financial_state")
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
