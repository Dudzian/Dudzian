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
    "terminalPanel",
    "strategiesPanel",
    "riskControlsPanel",
    "aiDecisionsPanel",
    "telemetryPanel",
    "diagnosticsPanel",
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
    initial_ticks = int(root.property("paperSessionTicks") or 0)
    initial_orders = _sequence_length(root.property("paperOrderRows"))
    initial_decisions = _sequence_length(root.property("decisionPreviewRows"))
    initial_telemetry = _sequence_length(root.property("paperTelemetryRows"))

    _invoke_qml(root, "startPaperPreview")
    after_start_ticks = int(root.property("paperSessionTicks") or 0)
    audit["start_sets_running"] = _string_property(root, "paperSessionStatus") == "running"
    audit["start_tick_delta"] = after_start_ticks - initial_ticks

    before_tick_orders = _sequence_length(root.property("paperOrderRows"))
    before_tick_decisions = _sequence_length(root.property("decisionPreviewRows"))
    before_tick_telemetry = _sequence_length(root.property("paperTelemetryRows"))
    _invoke_qml(root, "generatePaperTick")
    after_tick_ticks = int(root.property("paperSessionTicks") or 0)
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

    _invoke_qml(root, "runTenMockTicks")
    after_ten_ticks = int(root.property("paperSessionTicks") or 0)
    audit["run_ten_tick_delta"] = after_ten_ticks - after_tick_ticks

    _invoke_qml(root, "pausePaperPreview")
    audit["pause_sets_paused"] = _string_property(root, "paperSessionStatus") == "paused"
    _invoke_qml(root, "stopPaperPreview")
    audit["stop_sets_stopped"] = _string_property(root, "paperSessionStatus") == "stopped"
    _invoke_qml(root, "resetPaperPreview")
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

    audit["final_order_rows"] = _sequence_length(root.property("paperOrderRows"))
    audit["final_decision_rows"] = _sequence_length(root.property("decisionPreviewRows"))
    audit["final_telemetry_rows"] = _sequence_length(root.property("paperTelemetryRows"))
    audit["initial_order_rows"] = initial_orders
    audit["initial_decision_rows"] = initial_decisions
    audit["initial_telemetry_rows"] = initial_telemetry
    audit["safety_boundary_ok"] = (
        audit["live_trading_disabled"] is True
        and audit["exchange_io_disabled"] is True
        and audit["order_submission_disabled"] is True
        and audit["api_keys_required"] is False
        and audit["runtime_loop_started"] is False
        and audit["network_api_calls"] == "disabled"
    )
    required_true_keys = (
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
        "select_top20_propagates_terminal_pair",
        "select_all_visible_at_least_top20",
        "clear_selected_pairs_zero",
        "toggle_pair_selects_pair",
        "toggle_pair_updates_terminal_pair",
        "pair_selection_updates_decision_summary",
        "risk_profile_updates",
        "risk_summary_updates",
        "safety_boundary_ok",
    )
    audit["passed"] = (
        int(audit["start_tick_delta"]) >= 1
        and int(audit["generate_tick_delta"]) == 1
        and int(audit["run_ten_tick_delta"]) == 10
        and int(audit["select_top20_count"]) == 20
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
        preview_state_audit: dict[str, object] = {}
        if qml_loaded:
            source = _qml_preview_source()
            final_preview_tabs_loaded = _source_has_all(
                source,
                (
                    "Dashboard",
                    "AI Center",
                    "Trading Universe",
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
            if options.exercise_preview_state:
                preview_state_audit = _exercise_preview_state(root)
                if preview_state_audit.get("passed") is not True:
                    audit_issues.append("preview_state_exercise_failed")
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
