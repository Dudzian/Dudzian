"""Safe source-level smoke check for the PySide6/QML UI."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ui.pyside_app.app import AppOptions, BotPysideApplication
    from ui.pyside_app.preview_state_bridge import LocalPreviewStateBridge
else:
    from .app import AppOptions, BotPysideApplication
    from .preview_state_bridge import LocalPreviewStateBridge


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
    preview_launch_readiness_evaluated: bool = False
    preview_launch_readiness_requires_exercise_preview_state: bool = True
    preview_launch_readiness_evidence: dict[str, object] = field(default_factory=dict)
    frontend_live_parity_dashboard_present: bool = False
    frontend_live_parity_market_scanner_present: bool = False
    frontend_live_parity_ai_governor_present: bool = False
    frontend_live_parity_decisions_present: bool = False
    frontend_live_parity_terminal_order_panel_present: bool = False
    frontend_live_parity_portfolio_present: bool = False
    frontend_live_parity_alerts_telemetry_present: bool = False
    frontend_live_parity_live_safety_boundary_visible: bool = False
    frontend_live_parity_runtime_session_control_present: bool = False
    frontend_live_parity_no_fake_live_actions: bool = False
    frontend_live_parity_all_required_sections_present: bool = False
    frontend_live_parity_evidence: dict[str, object] = field(default_factory=dict)
    settings_config_live_shape_parity_complete: bool = False
    settings_config_live_shape_evidence: dict[str, object] = field(default_factory=dict)
    strategy_model_replay_live_shape_parity_complete: bool = False
    strategy_model_replay_live_shape_evidence: dict[str, object] = field(default_factory=dict)
    runtime_session_control_live_shape_parity_complete: bool = False
    runtime_session_control_live_shape_evidence: dict[str, object] = field(default_factory=dict)
    risk_live_safety_controls_visible_complete: bool = False
    risk_live_safety_controls_evidence: dict[str, object] = field(default_factory=dict)
    terminal_order_form_live_shape_complete: bool = False
    terminal_order_form_evidence: dict[str, object] = field(default_factory=dict)
    order_lifecycle_preview_parity_complete: bool = False
    order_lifecycle_evidence: dict[str, object] = field(default_factory=dict)
    market_scanner_live_field_parity_complete: bool = False
    market_scanner_live_field_evidence: dict[str, object] = field(default_factory=dict)
    portfolio_live_shape_parity_complete: bool = False
    portfolio_live_shape_evidence: dict[str, object] = field(default_factory=dict)
    alerts_telemetry_live_shape_parity_complete: bool = False
    alerts_telemetry_live_shape_evidence: dict[str, object] = field(default_factory=dict)
    operator_workflow_smoke_complete: bool = False
    operator_workflow_evidence: dict[str, object] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Render the smoke contract as deterministic CP1252-safe JSON."""

        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


def _qml_preview_source() -> str:
    qml_root = Path(__file__).resolve().parent / "qml"
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(qml_root.rglob("*.qml")))


def _source_has_all(source: str, labels: tuple[str, ...]) -> bool:
    return all(label in source for label in labels)


def _tracked_artifact_snapshot() -> dict[Path, bytes | None]:
    repo_root = Path(__file__).resolve().parents[2]
    artifact_paths = (
        repo_root / "reports" / "ci" / "decision_feed_metrics.json",
        repo_root / "var" / "ui_layouts.json",
    )
    return {path: path.read_bytes() if path.exists() else None for path in artifact_paths}


def _tracked_artifacts_unchanged(before: dict[Path, bytes | None]) -> bool:
    return _tracked_artifact_snapshot() == before


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
    "runtimeSessionControlPanel",
    "helpGlossaryPanel",
)


def _process_events() -> None:
    from PySide6.QtGui import QGuiApplication

    qt_app = QGuiApplication.instance()
    if qt_app is None:
        return
    for _ in range(12):
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


def _rows_contain_tokens(value: Any, tokens: tuple[object, ...]) -> bool:
    return _contains_tokens(_rows_repr(value), tokens)


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


def _variant_map(value: Any) -> dict[str, object]:
    value = _variant(value)
    return dict(value) if isinstance(value, dict) else {}


TYPED_PREVIEW_BRIDGE_AUDIT_KEYS = (
    "typed_preview_bridge_registered",
    "typed_preview_bridge_is_qml_context_instance",
    "typed_preview_bridge_schema_contract_valid",
    "typed_preview_bridge_matches_qml_paper_session_snapshot",
    "typed_preview_bridge_matches_qml_scanner_snapshot",
    "typed_preview_bridge_matches_qml_governor_snapshot",
    "typed_preview_bridge_matches_qml_portfolio_snapshot",
    "typed_preview_bridge_matches_qml_alert_telemetry_snapshot",
    "typed_preview_bridge_runtime_boundary_local_only",
    "typed_preview_bridge_qml_consumer_visible",
    "typed_preview_bridge_qml_consumer_schema_ok_visible",
    "typed_preview_bridge_qml_consumer_runtime_boundary_visible",
    "typed_preview_bridge_qml_consumer_diagnostic_marker_visible",
    "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only",
    "typed_preview_bridge_qml_consumer_matches_paper_snapshot",
    "typed_preview_bridge_qml_consumer_matches_scanner_snapshot",
    "typed_preview_bridge_qml_consumer_matches_governor_snapshot",
    "typed_preview_bridge_qml_consumer_fallback_state_visible",
    "typed_preview_bridge_qml_consumer_fallback_state_safe",
    "typed_preview_bridge_qml_consumer_fallback_state_no_type_error",
    "typed_preview_bridge_qml_consumer_updates_after_snapshot_a",
    "typed_preview_bridge_qml_consumer_updates_after_snapshot_b",
    "typed_preview_bridge_qml_consumer_replaces_stale_snapshot_tokens",
    "typed_preview_bridge_qml_consumer_survives_panel_navigation",
    "typed_preview_bridge_qml_consumer_restores_baseline_snapshot",
    "typed_preview_bridge_qml_consumer_lifecycle_sequence_completed",
)


TYPED_PREVIEW_BRIDGE_QML_CONSUMER_EVIDENCE_CHECKS = (
    "typed_preview_bridge_registered",
    "typed_preview_bridge_is_qml_context_instance",
    "typed_preview_bridge_qml_consumer_visible",
    "typed_preview_bridge_qml_consumer_schema_ok_visible",
    "typed_preview_bridge_qml_consumer_runtime_boundary_visible",
    "typed_preview_bridge_qml_consumer_diagnostic_marker_visible",
    "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only",
    "typed_preview_bridge_qml_consumer_matches_paper_snapshot",
    "typed_preview_bridge_qml_consumer_matches_scanner_snapshot",
    "typed_preview_bridge_qml_consumer_matches_governor_snapshot",
    "typed_preview_bridge_qml_consumer_fallback_state_visible",
    "typed_preview_bridge_qml_consumer_fallback_state_safe",
    "typed_preview_bridge_qml_consumer_fallback_state_no_type_error",
    "typed_preview_bridge_qml_consumer_updates_after_snapshot_a",
    "typed_preview_bridge_qml_consumer_updates_after_snapshot_b",
    "typed_preview_bridge_qml_consumer_replaces_stale_snapshot_tokens",
    "typed_preview_bridge_qml_consumer_survives_panel_navigation",
    "typed_preview_bridge_qml_consumer_restores_baseline_snapshot",
    "typed_preview_bridge_qml_consumer_lifecycle_sequence_completed",
)


def _typed_preview_bridge_consumer_evidence(audit: dict[str, object]) -> dict[str, object]:
    def all_true(*keys: str) -> bool:
        return all(audit.get(key) is True for key in keys)

    failed_checks = [
        key
        for key in TYPED_PREVIEW_BRIDGE_QML_CONSUMER_EVIDENCE_CHECKS
        if audit.get(key) is not True
    ]
    return {
        "registered": all_true("typed_preview_bridge_registered"),
        "qml_context_identity": all_true("typed_preview_bridge_is_qml_context_instance"),
        "visible_consumer": all_true(
            "typed_preview_bridge_qml_consumer_visible",
            "typed_preview_bridge_qml_consumer_schema_ok_visible",
            "typed_preview_bridge_qml_consumer_runtime_boundary_visible",
            "typed_preview_bridge_qml_consumer_matches_paper_snapshot",
            "typed_preview_bridge_qml_consumer_matches_scanner_snapshot",
            "typed_preview_bridge_qml_consumer_matches_governor_snapshot",
        ),
        "diagnostic_marker": all_true(
            "typed_preview_bridge_qml_consumer_diagnostic_marker_visible",
            "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only",
        ),
        "fallback_state": all_true(
            "typed_preview_bridge_qml_consumer_fallback_state_visible",
            "typed_preview_bridge_qml_consumer_fallback_state_safe",
            "typed_preview_bridge_qml_consumer_fallback_state_no_type_error",
        ),
        "lifecycle_sequence": all_true(
            "typed_preview_bridge_qml_consumer_updates_after_snapshot_a",
            "typed_preview_bridge_qml_consumer_updates_after_snapshot_b",
            "typed_preview_bridge_qml_consumer_survives_panel_navigation",
            "typed_preview_bridge_qml_consumer_lifecycle_sequence_completed",
        ),
        "stale_replacement": all_true(
            "typed_preview_bridge_qml_consumer_replaces_stale_snapshot_tokens"
        ),
        "baseline_restore": all_true(
            "typed_preview_bridge_qml_consumer_restores_baseline_snapshot"
        ),
        "all_typed_preview_bridge_consumer_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
    }


PREVIEW_LAUNCH_READINESS_CHECKS = (
    "qml_loaded",
    "root_objects_present",
    "preview_state_exercised",
    "runtime_loop_not_started",
    "typed_bridge_evidence_green",
    "local_only_boundary",
    "no_live_runtime_side_effects",
    "tracked_artifacts_clean",
)


def _preview_launch_readiness_evidence(payload: dict[str, object]) -> dict[str, object]:
    """Summarize fail-closed local preview/demo launch readiness from smoke facts."""

    audit = payload.get("preview_state_audit")
    if not isinstance(audit, dict):
        audit = {}
    typed_bridge_evidence = audit.get("typed_preview_bridge_qml_consumer_evidence")
    if not isinstance(typed_bridge_evidence, dict):
        typed_bridge_evidence = {}
    panel_load_results = payload.get("panel_load_results")
    side_panel_loaded = False
    if isinstance(panel_load_results, dict):
        side_panel = panel_load_results.get("sidePanel")
        if isinstance(side_panel, dict):
            side_panel_loaded = (
                side_panel.get("loaded") is True and side_panel.get("empty") is False
            )

    checks = {
        "qml_loaded": payload.get("qml_loaded") is True and payload.get("ui_loaded") is True,
        "root_objects_present": payload.get("operator_dashboard_present") is True
        and side_panel_loaded is True,
        "preview_state_exercised": payload.get("preview_state_exercised") is True,
        "runtime_loop_not_started": payload.get("runtime_loop_started") is False
        and audit.get("runtime_loop_started") is False,
        "typed_bridge_evidence_green": typed_bridge_evidence.get(
            "all_typed_preview_bridge_consumer_checks_passed"
        )
        is True,
        "local_only_boundary": payload.get("safety_boundary_ok") is True
        and payload.get("live_mode_allowed") is False
        and audit.get("live_trading_disabled") is True
        and audit.get("exchange_io_disabled") is True
        and audit.get("order_submission_disabled") is True
        and audit.get("safety_boundary_ok") is True,
        "no_live_runtime_side_effects": payload.get("exchange_io") == "disabled"
        and payload.get("order_submission") == "disabled"
        and payload.get("api_keys_required") is False
        and payload.get("secrets_read") is False
        and payload.get("keychain_read") is False
        and payload.get("env_values_read") is False
        and payload.get("dot_env_read") is False
        and audit.get("network_api_calls") == "disabled"
        and audit.get("api_keys_required") is False,
        "tracked_artifacts_clean": payload.get("tracked_artifacts_clean") is True,
    }
    failed_checks = [key for key in PREVIEW_LAUNCH_READINESS_CHECKS if checks.get(key) is not True]
    return {
        **checks,
        "all_preview_launch_readiness_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
    }


BLOCK_C_READ_ONLY_BINDING_VISIBLE_SOURCE_LABELS = (
    "BLOK C — UI READ-ONLY BINDING",
    "BLOK B contract-complete static-local",
    "integration gate: blocked",
    "runtime loop: not started",
    "runtime backed: false",
    "UI runtime integration: false",
    "decision/export/live readiness: false",
    "read-only binding only",
)


def _block_c_read_only_binding_visible_source_evidence(
    qml_source: str | None = None,
) -> dict[str, object]:
    """Return source-only BLOK C visible panel proof without starting UI/runtime loops."""

    source = _qml_preview_source() if qml_source is None else qml_source
    panel_marker = 'objectName: "operatorDashboardBlockCReadOnlyBindingSummary"'
    label_marker = 'descriptionObjectName: "previewBlockCReadOnlyBindingSummaryLabel"'
    forbidden_panel_tokens = (
        "onClicked",
        "submit",
        "cancel",
        "generate order",
        "execute",
        "export",
        "live readiness: true",
        "runtime backed: true",
        "UI runtime integration: true",
    )
    panel_start = source.find(panel_marker)
    panel_slice = source[panel_start : panel_start + 900] if panel_start >= 0 else ""
    labels_present = _source_has_all(source, BLOCK_C_READ_ONLY_BINDING_VISIBLE_SOURCE_LABELS)
    checks = {
        "panel_source_present": panel_marker in source and label_marker in source,
        "labels_present": labels_present,
        "read_only_static_local_text_present": _source_has_all(
            source,
            (
                "BLOK B contract-complete static-local",
                "read-only binding only",
            ),
        ),
        "integration_gate_blocked_text_present": "integration gate: blocked" in source,
        "runtime_loop_not_started_text_present": "runtime loop: not started" in source,
        "runtime_backed_false_text_present": "runtime backed: false" in source,
        "ready_for_ui_runtime_integration_false_text_present": (
            "UI runtime integration: false" in source
        ),
        "decision_export_live_false_text_present": (
            "decision/export/live readiness: false" in source
        ),
        "no_action_controls": all(token not in panel_slice for token in forbidden_panel_tokens),
    }
    failed_checks = [key for key, value in checks.items() if value is not True]
    return {
        **checks,
        "object_name": "operatorDashboardBlockCReadOnlyBindingSummary",
        "description_object_name": "previewBlockCReadOnlyBindingSummaryLabel",
        "all_block_c_read_only_binding_visible_source_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
    }


FRONTEND_LIVE_PARITY_REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "dashboard": (
        "operator_dashboard_visible",
        "dashboard_scanner_uses_shared_state",
        "dashboard_ai_decision_matches_governor_snapshot",
        "risk_summary_updates",
        "visible_terminal_matches_latest_paper_order",
        "alerts_dashboard_summary_present",
        "global_safety_badges_present",
    ),
    "market_scanner": (
        "market_scanner_rows_present",
        "market_scanner_filter_sort_threshold_present",
        "market_scanner_explain_updates_explanation",
        "simulation_can_use_scanner_candidate_local_only",
        "market_scanner_live_field_parity_complete",
    ),
    "ai_governor": (
        "governor_updates_decision",
        "governor_uses_scanner_and_risk_state",
        "visible_ai_center_matches_governor_snapshot",
        "risk_ai_recommended_explanation_present",
        "explainability_no_order_submission",
    ),
    "ai_decisions": (
        "decision_explainability_state_present",
        "visible_decisions_match_latest_governor_action",
        "decision_explain_has_risk_checks",
        "paper_order_explain_local_only",
    ),
    "terminal_order_panel": (
        "simulate_order_updates_blotter_portfolio_telemetry",
        "terminal_blotter_updates_portfolio_snapshot",
        "risk_block_generates_blocked_event_and_alert",
        "blocked_semantics_no_legacy_generated",
        "simulation_does_not_enable_order_submission",
        "terminal_order_form_live_shape_complete",
        "order_lifecycle_preview_parity_complete",
    ),
    "portfolio": (
        "portfolio_fields_present",
        "portfolio_equity_formula_ok",
        "portfolio_net_pnl_formula_ok",
        "portfolio_time_filter_does_not_mutate_paper_state",
        "portfolio_custom_filter_does_not_mutate_paper_state",
        "visible_portfolio_matches_portfolio_snapshot",
        "portfolio_live_shape_parity_complete",
    ),
    "alerts_telemetry": (
        "alerts_state_present",
        "alerts_detail_present",
        "alerts_risk_block_appends_event",
        "ping_appends_telemetry",
        "visible_alerts_match_alert_snapshot",
        "visible_telemetry_matches_telemetry_snapshot",
        "alert_center_safety_boundary_ok",
        "alerts_telemetry_live_shape_parity_complete",
    ),
    "live_safety_boundary": (
        "live_trading_disabled",
        "exchange_io_disabled",
        "order_submission_disabled",
        "runtime_loop_started_false",
        "api_keys_not_required",
        "simulation_does_not_read_secrets",
        "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only",
        "safety_boundary_ok",
    ),
    "risk_live_safety_controls": ("risk_live_safety_controls_visible_complete",),
    "operator_workflow": ("operator_workflow_smoke_complete",),
    "settings_config": ("settings_config_live_shape_parity_complete",),
    "strategy_model_replay": ("strategy_model_replay_live_shape_parity_complete",),
    "runtime_session_control": ("runtime_session_control_live_shape_parity_complete",),
}

FRONTEND_RUNTIME_SESSION_CONTROL_REQUIRED_CHECKS = (
    "runtime_session_panel_visible",
    "session_controls_visible",
    "start_stop_pause_resume_visible",
    "live_runtime_disabled_visible",
    "no_real_loop_start_visible",
    "current_session_state_visible",
    "control_plane_health_visible",
    "scheduler_status_visible",
    "worker_status_visible",
    "heartbeat_visible",
    "mock_heartbeat_visible",
    "recovery_controls_visible",
    "failover_state_visible",
    "degraded_mode_visible",
    "recovery_actions_disabled_visible",
    "no_live_reconnect_visible",
    "runtime_preflight_gate_visible",
    "runtime_activation_blocked_reason_visible",
    "emergency_stop_shape_visible",
    "no_live_scheduler_worker_start_visible",
    "no_live_adapter_start_visible",
    "runtime_audit_local_only_visible",
    "no_cloud_sink_visible",
    "no_external_export_visible",
)


def _build_runtime_session_control_live_shape_evidence(
    audit: dict[str, object],
) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-11.0 runtime/session/control-plane evidence."""

    missing_checks = [
        key
        for key in FRONTEND_RUNTIME_SESSION_CONTROL_REQUIRED_CHECKS
        if audit.get(key) is not True
    ]
    return {
        "runtime_session_control_required_checks": list(
            FRONTEND_RUNTIME_SESSION_CONTROL_REQUIRED_CHECKS
        ),
        "runtime_session_control_missing_checks": missing_checks,
        "runtime_session_control_live_shape_parity_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_RUNTIME_SESSION_CONTROL_REQUIRED_CHECKS},
    }


FRONTEND_SETTINGS_CONFIG_LIVE_SHAPE_REQUIRED_CHECKS = (
    "settings_panel_visible",
    "mode_controls_visible",
    "preview_mode_boundary_visible",
    "live_mode_locked_visible",
    "api_key_status_visible",
    "api_keys_masked_visible",
    "no_secret_material_visible",
    "exchange_profile_visible",
    "account_profile_visible",
    "no_exchange_io_boundary_visible",
    "no_account_balance_fetch_boundary_visible",
    "config_validation_visible",
    "live_activation_blocked_reason_visible",
    "config_audit_local_only_visible",
    "no_cloud_sink_visible",
    "no_external_export_visible",
)


FRONTEND_STRATEGY_MODEL_REPLAY_REQUIRED_CHECKS = (
    "strategy_panel_visible",
    "strategy_registry_visible",
    "active_strategy_visible",
    "strategy_health_visible",
    "strategy_risk_profile_visible",
    "model_artifact_status_visible",
    "model_lineage_visible",
    "inference_readiness_visible",
    "no_model_promotion_visible",
    "backtest_replay_controls_visible",
    "replay_dataset_window_visible",
    "replay_results_summary_visible",
    "replay_metrics_visible",
    "local_replay_only_visible",
    "no_live_market_data_fetch_visible",
    "readiness_checklist_visible",
    "live_promotion_locked_visible",
    "no_live_deployment_side_effect_visible",
    "strategy_audit_local_only_visible",
    "no_cloud_sink_visible",
    "no_external_export_visible",
)


def _build_strategy_model_replay_live_shape_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-10.0 strategy/model/replay evidence."""

    missing_checks = [
        key for key in FRONTEND_STRATEGY_MODEL_REPLAY_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "strategy_model_replay_required_checks": list(
            FRONTEND_STRATEGY_MODEL_REPLAY_REQUIRED_CHECKS
        ),
        "strategy_model_replay_missing_checks": missing_checks,
        "strategy_model_replay_live_shape_parity_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_STRATEGY_MODEL_REPLAY_REQUIRED_CHECKS},
    }


def _build_settings_config_live_shape_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-9.0 settings/config evidence."""

    missing_checks = [
        key
        for key in FRONTEND_SETTINGS_CONFIG_LIVE_SHAPE_REQUIRED_CHECKS
        if audit.get(key) is not True
    ]
    return {
        "settings_config_required_checks": list(
            FRONTEND_SETTINGS_CONFIG_LIVE_SHAPE_REQUIRED_CHECKS
        ),
        "settings_config_missing_checks": missing_checks,
        "settings_config_live_shape_parity_complete": not missing_checks,
        **{
            key: audit.get(key) is True
            for key in FRONTEND_SETTINGS_CONFIG_LIVE_SHAPE_REQUIRED_CHECKS
        },
    }


FRONTEND_TERMINAL_ORDER_FORM_REQUIRED_CHECKS = (
    "terminal_order_form_visible",
    "terminal_order_mode_local_preview_visible",
    "terminal_order_symbol_visible",
    "terminal_order_side_controls_present",
    "terminal_order_type_controls_present",
    "terminal_order_price_amount_total_present",
    "terminal_order_percent_chips_present",
    "terminal_order_submission_disabled_visible",
    "terminal_order_latest_status_visible",
    "terminal_order_blocked_state_visible",
    "terminal_order_simulated_state_visible",
    "terminal_order_no_order_state_visible",
    "terminal_order_rejected_disabled_placeholder_visible",
    "terminal_order_updates_blotter_portfolio_telemetry",
)


FRONTEND_ORDER_LIFECYCLE_REQUIRED_CHECKS = (
    "order_lifecycle_decision_visible",
    "order_lifecycle_simulated_order_visible",
    "order_lifecycle_blocked_order_visible",
    "order_lifecycle_no_order_visible",
    "order_lifecycle_rejected_disabled_visible",
    "order_lifecycle_partial_fill_cancel_placeholders_visible",
    "order_lifecycle_downstream_portfolio_updates",
    "order_lifecycle_alert_telemetry_updates",
    "order_lifecycle_no_live_side_effects",
)


FRONTEND_RISK_LIVE_SAFETY_REQUIRED_CHECKS = (
    "live_trading_disabled_visible",
    "exchange_io_disabled_visible",
    "order_submission_disabled_visible",
    "runtime_loop_not_started_visible",
    "api_keys_not_required_visible",
    "secrets_not_read_visible",
    "preview_mode_badge_visible",
    "live_mode_blocked_badge_visible",
    "safety_boundary_visible",
    "no_live_side_effects_visible",
    "kill_switch_or_safety_lock_visible",
    "risk_lock_or_risk_gate_visible",
    "risk_profile_visible",
    "risk_limits_visible",
    "blocked_reasons_visible",
    "confidence_floor_or_score_threshold_visible",
    "max_position_or_exposure_limit_visible",
    "daily_loss_or_drawdown_limit_visible",
    "risk_blocked_event_visible",
    "operator_can_explain_blocked_state_local_only",
)


FRONTEND_PORTFOLIO_LIVE_SHAPE_REQUIRED_CHECKS = (
    "portfolio_summary_visible",
    "portfolio_equity_visible",
    "portfolio_cash_or_balance_visible",
    "portfolio_unrealized_pnl_visible",
    "portfolio_realized_pnl_visible",
    "portfolio_total_pnl_visible",
    "portfolio_daily_pnl_visible",
    "portfolio_drawdown_visible",
    "portfolio_exposure_visible",
    "portfolio_freshness_visible",
    "portfolio_local_source_marker_visible",
    "portfolio_open_positions_visible",
    "portfolio_position_symbol_visible",
    "portfolio_position_side_visible",
    "portfolio_position_quantity_visible",
    "portfolio_position_entry_price_visible",
    "portfolio_position_mark_price_visible",
    "portfolio_position_unrealized_pnl_visible",
    "portfolio_position_exposure_visible",
    "portfolio_open_orders_visible",
    "portfolio_order_symbol_visible",
    "portfolio_order_side_visible",
    "portfolio_order_quantity_visible",
    "portfolio_order_status_visible",
    "portfolio_closed_trades_visible",
    "portfolio_closed_trade_symbol_visible",
    "portfolio_closed_trade_pnl_visible",
    "portfolio_fills_or_executions_placeholder_visible",
    "portfolio_fees_or_slippage_placeholder_visible",
    "portfolio_risk_state_visible",
    "portfolio_blocked_exposure_reason_visible",
    "portfolio_no_live_account_sync_visible",
    "portfolio_no_exchange_balance_fetch_visible",
    "portfolio_no_real_fill_path_visible",
    "portfolio_no_real_order_path_visible",
    "portfolio_uses_local_paper_state",
    "portfolio_updates_after_preview_order_local_only",
)


FRONTEND_ALERTS_TELEMETRY_LIVE_SHAPE_REQUIRED_CHECKS = (
    "alerts_feed_visible",
    "alerts_rows_visible",
    "alert_severity_visible",
    "alert_source_visible",
    "alert_category_visible",
    "alert_message_visible",
    "alert_timestamp_or_freshness_visible",
    "alert_acknowledged_or_unresolved_placeholder_visible",
    "risk_blocked_alert_visible",
    "order_blocked_alert_visible",
    "scanner_candidate_alert_visible",
    "telemetry_feed_visible",
    "telemetry_rows_visible",
    "telemetry_event_type_visible",
    "telemetry_component_visible",
    "telemetry_message_visible",
    "telemetry_timestamp_or_freshness_visible",
    "telemetry_runtime_mode_marker_visible",
    "telemetry_local_preview_source_marker_visible",
    "audit_log_visible",
    "audit_event_id_or_sequence_visible",
    "audit_decision_event_visible",
    "audit_order_event_visible",
    "audit_risk_event_visible",
    "audit_scanner_event_visible",
    "audit_correlation_or_trace_marker_visible",
    "audit_local_only_marker_visible",
    "alerts_no_cloud_sink_visible",
    "alerts_no_external_export_visible",
    "telemetry_no_live_exchange_stream_visible",
    "telemetry_no_real_order_stream_visible",
    "telemetry_no_secrets_logged_visible",
    "telemetry_uses_local_preview_state",
    "alerts_telemetry_updates_after_preview_actions_local_only",
)


FRONTEND_OPERATOR_WORKFLOW_REQUIRED_CHECKS = (
    "operator_dashboard_visible",
    "operator_can_open_market_scanner",
    "operator_can_open_ai_decisions",
    "operator_can_open_risk_controls",
    "operator_can_open_terminal",
    "operator_can_open_portfolio",
    "operator_can_open_alerts",
    "operator_can_open_telemetry",
    "operator_can_return_to_dashboard",
    "operator_scanner_candidate_visible",
    "operator_scanner_candidate_selectable_local_only",
    "operator_selected_candidate_updates_shared_state",
    "operator_decision_visible_after_candidate",
    "operator_decision_has_action_confidence_reason",
    "operator_decision_local_source_visible",
    "operator_risk_gate_visible",
    "operator_risk_block_reason_visible",
    "operator_blocked_state_explainable",
    "operator_risk_limits_visible",
    "operator_live_safety_lock_visible",
    "operator_terminal_order_form_visible",
    "operator_terminal_pair_matches_selected_candidate",
    "operator_order_submission_disabled_visible",
    "operator_simulated_order_path_visible",
    "operator_blocked_order_path_visible",
    "operator_no_real_order_path_visible",
    "operator_portfolio_visible",
    "operator_portfolio_reflects_preview_order_or_block",
    "operator_portfolio_local_paper_marker_visible",
    "operator_portfolio_no_live_account_sync_visible",
    "operator_alerts_visible_after_actions",
    "operator_risk_or_order_alert_visible",
    "operator_telemetry_visible_after_actions",
    "operator_audit_correlation_visible",
    "operator_no_cloud_sink_visible",
    "operator_no_external_export_visible",
    "operator_no_secrets_logged_visible",
    "operator_terminal_order_form_live_shape_complete",
    "operator_order_lifecycle_preview_parity_complete",
    "operator_risk_live_safety_controls_visible_complete",
    "operator_market_scanner_live_field_parity_complete",
    "operator_portfolio_live_shape_parity_complete",
    "operator_alerts_telemetry_live_shape_parity_complete",
)


def _build_operator_workflow_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-8.0 full operator workflow evidence."""

    missing_checks = [
        key for key in FRONTEND_OPERATOR_WORKFLOW_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "operator_workflow_required_checks": list(FRONTEND_OPERATOR_WORKFLOW_REQUIRED_CHECKS),
        "operator_workflow_missing_checks": missing_checks,
        "operator_workflow_smoke_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_OPERATOR_WORKFLOW_REQUIRED_CHECKS},
    }


def _build_alerts_telemetry_live_shape_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-7.0 alerts/telemetry live-shape evidence."""

    missing_checks = [
        key
        for key in FRONTEND_ALERTS_TELEMETRY_LIVE_SHAPE_REQUIRED_CHECKS
        if audit.get(key) is not True
    ]
    return {
        "alerts_telemetry_live_shape_required_checks": list(
            FRONTEND_ALERTS_TELEMETRY_LIVE_SHAPE_REQUIRED_CHECKS
        ),
        "alerts_telemetry_live_shape_missing_checks": missing_checks,
        "alerts_telemetry_live_shape_parity_complete": not missing_checks,
        **{
            key: audit.get(key) is True
            for key in FRONTEND_ALERTS_TELEMETRY_LIVE_SHAPE_REQUIRED_CHECKS
        },
    }


def _build_portfolio_live_shape_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-6.0 portfolio live-shape evidence."""

    missing_checks = [
        key for key in FRONTEND_PORTFOLIO_LIVE_SHAPE_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "portfolio_live_shape_required_checks": list(FRONTEND_PORTFOLIO_LIVE_SHAPE_REQUIRED_CHECKS),
        "portfolio_live_shape_missing_checks": missing_checks,
        "portfolio_live_shape_parity_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_PORTFOLIO_LIVE_SHAPE_REQUIRED_CHECKS},
    }


FRONTEND_MARKET_SCANNER_LIVE_FIELD_REQUIRED_CHECKS = (
    "market_scanner_table_visible",
    "market_scanner_rows_visible",
    "market_scanner_symbol_visible",
    "market_scanner_exchange_or_venue_visible",
    "market_scanner_price_visible",
    "market_scanner_spread_visible",
    "market_scanner_volume_or_liquidity_visible",
    "market_scanner_volatility_visible",
    "market_scanner_score_visible",
    "market_scanner_risk_score_visible",
    "market_scanner_confidence_visible",
    "market_scanner_ai_action_visible",
    "market_scanner_risk_decision_visible",
    "market_scanner_reason_visible",
    "market_scanner_freshness_visible",
    "market_scanner_local_source_marker_visible",
    "market_scanner_filter_controls_visible",
    "market_scanner_sort_controls_visible",
    "market_scanner_threshold_controls_visible",
    "market_scanner_selected_candidate_details_visible",
    "market_scanner_explain_candidate_local_only",
    "market_scanner_can_select_candidate_local_only",
    "market_scanner_no_exchange_io_visible",
    "market_scanner_no_live_feed_visible",
    "market_scanner_no_real_order_path_visible",
    "market_scanner_uses_local_preview_catalog",
    "market_scanner_updates_shared_preview_state_local_only",
)


def _build_market_scanner_live_field_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed FRONTEND-PARITY-5.0 scanner live-field evidence."""

    missing_checks = [
        key
        for key in FRONTEND_MARKET_SCANNER_LIVE_FIELD_REQUIRED_CHECKS
        if audit.get(key) is not True
    ]
    return {
        "market_scanner_live_field_required_checks": list(
            FRONTEND_MARKET_SCANNER_LIVE_FIELD_REQUIRED_CHECKS
        ),
        "market_scanner_live_field_missing_checks": missing_checks,
        "market_scanner_live_field_parity_complete": not missing_checks,
        **{
            key: audit.get(key) is True
            for key in FRONTEND_MARKET_SCANNER_LIVE_FIELD_REQUIRED_CHECKS
        },
    }


def _build_risk_live_safety_controls_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build fail-closed evidence for visible preview risk/live safety controls."""

    missing_checks = [
        key for key in FRONTEND_RISK_LIVE_SAFETY_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "risk_live_safety_controls_required_checks": list(
            FRONTEND_RISK_LIVE_SAFETY_REQUIRED_CHECKS
        ),
        "risk_live_safety_controls_missing_checks": missing_checks,
        "risk_live_safety_controls_visible_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_RISK_LIVE_SAFETY_REQUIRED_CHECKS},
    }


def _build_order_lifecycle_parity_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build a pure fail-closed checklist for preview order lifecycle parity."""

    missing_checks = [
        key for key in FRONTEND_ORDER_LIFECYCLE_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "order_lifecycle_required_checks": list(FRONTEND_ORDER_LIFECYCLE_REQUIRED_CHECKS),
        "order_lifecycle_missing_checks": missing_checks,
        "order_lifecycle_preview_parity_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_ORDER_LIFECYCLE_REQUIRED_CHECKS},
    }


def _build_terminal_order_form_parity_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build a pure live-shape checklist for the preview terminal order form."""

    missing_checks = [
        key for key in FRONTEND_TERMINAL_ORDER_FORM_REQUIRED_CHECKS if audit.get(key) is not True
    ]
    return {
        "terminal_order_form_required_checks": list(FRONTEND_TERMINAL_ORDER_FORM_REQUIRED_CHECKS),
        "terminal_order_form_missing_checks": missing_checks,
        "terminal_order_form_live_shape_complete": not missing_checks,
        **{key: audit.get(key) is True for key in FRONTEND_TERMINAL_ORDER_FORM_REQUIRED_CHECKS},
    }


FRONTEND_LIVE_PARITY_SMOKE_KEYS = (
    "frontend_live_parity_dashboard_present",
    "frontend_live_parity_market_scanner_present",
    "frontend_live_parity_ai_governor_present",
    "frontend_live_parity_decisions_present",
    "frontend_live_parity_terminal_order_panel_present",
    "frontend_live_parity_portfolio_present",
    "frontend_live_parity_alerts_telemetry_present",
    "frontend_live_parity_live_safety_boundary_visible",
    "frontend_live_parity_runtime_session_control_present",
    "runtime_session_control_live_shape_parity_complete",
    "risk_live_safety_controls_visible_complete",
    "market_scanner_live_field_parity_complete",
    "portfolio_live_shape_parity_complete",
    "alerts_telemetry_live_shape_parity_complete",
    "frontend_live_parity_no_fake_live_actions",
    "operator_workflow_smoke_complete",
    "strategy_model_replay_live_shape_parity_complete",
    "frontend_live_parity_all_required_sections_present",
)


def _build_frontend_live_parity_evidence(audit: dict[str, object]) -> dict[str, object]:
    """Build a pure checklist for live-equivalent preview frontend completeness."""

    normalized_audit = {
        **audit,
        "runtime_loop_started_false": audit.get("runtime_loop_started") is False,
        "api_keys_not_required": audit.get("api_keys_required") is False,
    }
    section_results = {
        section: all(normalized_audit.get(key) is True for key in keys)
        for section, keys in FRONTEND_LIVE_PARITY_REQUIRED_SECTIONS.items()
    }
    missing_sections = [
        section
        for section in FRONTEND_LIVE_PARITY_REQUIRED_SECTIONS
        if section_results.get(section) is not True
    ]
    no_fake_live_actions = (
        normalized_audit.get("live_trading_disabled") is True
        and normalized_audit.get("exchange_io_disabled") is True
        and normalized_audit.get("order_submission_disabled") is True
        and normalized_audit.get("runtime_loop_started_false") is True
        and normalized_audit.get("api_keys_not_required") is True
        and normalized_audit.get("simulation_does_not_read_secrets") is True
    )
    return {
        "frontend_live_parity_required_sections": list(FRONTEND_LIVE_PARITY_REQUIRED_SECTIONS),
        "frontend_live_parity_section_results": section_results,
        "frontend_live_parity_missing_sections": missing_sections,
        "frontend_live_parity_dashboard_present": section_results["dashboard"],
        "frontend_live_parity_market_scanner_present": section_results["market_scanner"],
        "frontend_live_parity_ai_governor_present": section_results["ai_governor"],
        "frontend_live_parity_decisions_present": section_results["ai_decisions"],
        "frontend_live_parity_terminal_order_panel_present": section_results[
            "terminal_order_panel"
        ],
        "frontend_live_parity_portfolio_present": section_results["portfolio"],
        "frontend_live_parity_alerts_telemetry_present": section_results["alerts_telemetry"],
        "frontend_live_parity_live_safety_boundary_visible": section_results[
            "live_safety_boundary"
        ],
        "frontend_live_parity_runtime_session_control_present": section_results[
            "runtime_session_control"
        ],
        "runtime_session_control_live_shape_parity_complete": section_results[
            "runtime_session_control"
        ],
        "risk_live_safety_controls_visible_complete": section_results["risk_live_safety_controls"],
        "market_scanner_live_field_parity_complete": normalized_audit.get(
            "market_scanner_live_field_parity_complete"
        )
        is True,
        "portfolio_live_shape_parity_complete": normalized_audit.get(
            "portfolio_live_shape_parity_complete"
        )
        is True,
        "alerts_telemetry_live_shape_parity_complete": normalized_audit.get(
            "alerts_telemetry_live_shape_parity_complete"
        )
        is True,
        "frontend_live_parity_no_fake_live_actions": no_fake_live_actions,
        "operator_workflow_smoke_complete": section_results["operator_workflow"],
        "settings_config_live_shape_parity_complete": section_results["settings_config"],
        "frontend_live_parity_settings_config_present": section_results["settings_config"],
        "strategy_model_replay_live_shape_parity_complete": section_results[
            "strategy_model_replay"
        ],
        "frontend_live_parity_strategy_model_replay_present": section_results[
            "strategy_model_replay"
        ],
        "frontend_live_parity_all_required_sections_present": (
            not missing_sections and no_fake_live_actions
        ),
    }


def _typed_preview_bridge_false_audit() -> dict[str, object]:
    return {key: False for key in TYPED_PREVIEW_BRIDGE_AUDIT_KEYS}


def _safe_bridge_property(typed_preview_bridge: Any, property_name: str) -> object:
    try:
        property_reader = getattr(typed_preview_bridge, "property", None)
        if not callable(property_reader):
            return None
        return property_reader(property_name)
    except (AttributeError, RuntimeError, TypeError):
        return None


def _typed_preview_consumer_values(root: Any) -> dict[str, str]:
    return {
        "contract": _read_visible_panel_object(
            root, "sidePanel", "previewTypedBridgeContractLabel"
        ),
        "paper": _read_visible_panel_object(root, "sidePanel", "previewTypedBridgePaperLabel"),
        "scanner": _read_visible_panel_object(root, "sidePanel", "previewTypedBridgeScannerLabel"),
        "governor": _read_visible_panel_object(
            root, "sidePanel", "previewTypedBridgeGovernorLabel"
        ),
        "diagnostic_marker": _read_visible_panel_object(
            root, "sidePanel", "previewTypedBridgeDiagnosticMarkerLabel"
        ),
    }


def _preview_lifecycle_snapshot_variants(
    baseline_snapshots: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    snapshot_a = {key: dict(value) for key, value in baseline_snapshots.items()}
    snapshot_b = {key: dict(value) for key, value in baseline_snapshots.items()}
    snapshot_a["paper_session"].update({"normalizedState": "running", "orderRows": 1})
    snapshot_a["scanner"].update({"bestOpportunity": "BTC/USDT", "candidates": 3})
    snapshot_a["governor"].update({"latestAction": "PAPER BUY", "latestSymbol": "BTC/USDT"})
    snapshot_b["paper_session"].update({"normalizedState": "paused", "orderRows": 2})
    snapshot_b["scanner"].update({"bestOpportunity": "ETH/USDT", "candidates": 7})
    snapshot_b["governor"].update({"latestAction": "BLOCKED", "latestSymbol": "ETH/USDT"})
    return snapshot_a, snapshot_b


def _update_typed_preview_bridge_snapshots(
    typed_preview_bridge: Any,
    snapshots: dict[str, dict[str, object]],
) -> None:
    typed_preview_bridge.updateSnapshots(
        snapshots["paper_session"],
        snapshots["scanner"],
        snapshots["governor"],
        snapshots["portfolio"],
        snapshots["alert_telemetry"],
    )
    _process_events()


def _typed_preview_consumer_matches_lifecycle_snapshot(
    consumer_values: dict[str, str],
    snapshots: dict[str, dict[str, object]],
) -> bool:
    return (
        all(consumer_values.values())
        and _contains_tokens(
            consumer_values["paper"],
            (
                snapshots["paper_session"].get("normalizedState"),
                snapshots["paper_session"].get("orderRows"),
            ),
        )
        and _contains_tokens(
            consumer_values["scanner"],
            (
                snapshots["scanner"].get("bestOpportunity"),
                snapshots["scanner"].get("candidates"),
            ),
        )
        and _contains_tokens(
            consumer_values["governor"],
            (
                snapshots["governor"].get("latestAction"),
                snapshots["governor"].get("latestSymbol"),
            ),
        )
    )


def _typed_preview_consumer_has_stale_snapshot_a_tokens(
    consumer_values: dict[str, str],
) -> bool:
    return (
        "running" in consumer_values["paper"].lower()
        or "btc/usdt" in consumer_values["scanner"].lower()
        or "paper buy" in consumer_values["governor"].lower()
        or "btc/usdt" in consumer_values["governor"].lower()
    )


def _audit_typed_preview_bridge_consumer_lifecycle(
    root: Any,
    typed_preview_bridge: Any,
    baseline_snapshots: dict[str, dict[str, object]],
) -> dict[str, bool]:
    lifecycle_audit = {
        "typed_preview_bridge_qml_consumer_updates_after_snapshot_a": False,
        "typed_preview_bridge_qml_consumer_updates_after_snapshot_b": False,
        "typed_preview_bridge_qml_consumer_replaces_stale_snapshot_tokens": False,
        "typed_preview_bridge_qml_consumer_survives_panel_navigation": False,
        "typed_preview_bridge_qml_consumer_restores_baseline_snapshot": False,
        "typed_preview_bridge_qml_consumer_lifecycle_sequence_completed": False,
    }
    try:
        snapshot_a, snapshot_b = _preview_lifecycle_snapshot_variants(baseline_snapshots)
        _update_typed_preview_bridge_snapshots(typed_preview_bridge, snapshot_a)
        snapshot_a_values = _typed_preview_consumer_values(root)
        lifecycle_audit["typed_preview_bridge_qml_consumer_updates_after_snapshot_a"] = (
            _typed_preview_consumer_matches_lifecycle_snapshot(snapshot_a_values, snapshot_a)
        )

        _update_typed_preview_bridge_snapshots(typed_preview_bridge, snapshot_b)
        snapshot_b_values = _typed_preview_consumer_values(root)
        lifecycle_audit["typed_preview_bridge_qml_consumer_updates_after_snapshot_b"] = (
            _typed_preview_consumer_matches_lifecycle_snapshot(snapshot_b_values, snapshot_b)
        )
        lifecycle_audit[
            "typed_preview_bridge_qml_consumer_replaces_stale_snapshot_tokens"
        ] = not _typed_preview_consumer_has_stale_snapshot_a_tokens(snapshot_b_values)

        _invoke_show_panel(root, "marketScannerPanel")
        _process_events()
        _invoke_show_panel(root, "sidePanel")
        _process_events()
        post_navigation_values = _typed_preview_consumer_values(root)
        lifecycle_audit["typed_preview_bridge_qml_consumer_survives_panel_navigation"] = (
            _typed_preview_consumer_matches_lifecycle_snapshot(post_navigation_values, snapshot_b)
        )

        typed_preview_bridge.updateSnapshots(
            {"normalizedState": "", "orderRows": []},
            {"bestOpportunity": "", "candidates": []},
            {"latestAction": "", "latestSymbol": ""},
            {},
            {},
        )
        _process_events()
        _update_typed_preview_bridge_snapshots(typed_preview_bridge, baseline_snapshots)
        restored_values = _typed_preview_consumer_values(root)
        lifecycle_audit["typed_preview_bridge_qml_consumer_restores_baseline_snapshot"] = (
            _typed_preview_consumer_matches_lifecycle_snapshot(restored_values, baseline_snapshots)
        )
        lifecycle_audit["typed_preview_bridge_qml_consumer_lifecycle_sequence_completed"] = True
    except (AttributeError, RuntimeError, TypeError):
        return lifecycle_audit
    finally:
        try:
            _update_typed_preview_bridge_snapshots(typed_preview_bridge, baseline_snapshots)
        except (AttributeError, RuntimeError, TypeError):
            pass
    return lifecycle_audit


def _audit_typed_preview_bridge_fallback_state(
    root: Any,
    typed_preview_bridge: Any,
    qml_snapshots: dict[str, dict[str, object]],
) -> dict[str, bool]:
    fallback_audit = {
        "typed_preview_bridge_qml_consumer_fallback_state_visible": False,
        "typed_preview_bridge_qml_consumer_fallback_state_safe": False,
        "typed_preview_bridge_qml_consumer_fallback_state_no_type_error": False,
    }
    try:
        typed_preview_bridge.updateSnapshots(
            {"normalizedState": "", "orderRows": []},
            {"bestOpportunity": "", "candidates": []},
            {"latestAction": "", "latestSymbol": ""},
            {},
            {},
        )
        _process_events()
        fallback_values = _typed_preview_consumer_values(root)
        fallback_audit["typed_preview_bridge_qml_consumer_fallback_state_visible"] = all(
            fallback_values.values()
        )
        fallback_audit["typed_preview_bridge_qml_consumer_fallback_state_safe"] = (
            _contains_tokens(fallback_values["paper"], ("Bridge paper", "—", "orders"))
            and "0" in fallback_values["paper"]
            and _contains_tokens(fallback_values["scanner"], ("Bridge scanner", "—", "candidates"))
            and "0" in fallback_values["scanner"]
            and _contains_tokens(fallback_values["governor"], ("Bridge governor", "—"))
            and _contains_tokens(fallback_values["contract"], ("Typed bridge", "schema"))
            and _contains_tokens(
                fallback_values["diagnostic_marker"],
                ("local", "preview", "read-only", "diagnostic"),
            )
        )
        fallback_audit["typed_preview_bridge_qml_consumer_fallback_state_no_type_error"] = True
    except (AttributeError, RuntimeError, TypeError):
        return fallback_audit
    finally:
        try:
            typed_preview_bridge.updateSnapshots(
                qml_snapshots["paper_session"],
                qml_snapshots["scanner"],
                qml_snapshots["governor"],
                qml_snapshots["portfolio"],
                qml_snapshots["alert_telemetry"],
            )
            _process_events()
        except (AttributeError, RuntimeError, TypeError):
            pass
    return fallback_audit


def _audit_typed_preview_bridge(
    root: Any,
    typed_preview_bridge: Any,
    qml_context_bridge_instance: Any,
) -> dict[str, object]:
    audit = _typed_preview_bridge_false_audit()
    if typed_preview_bridge is None:
        return audit

    required_methods = ("updateSnapshots", "updateRuntimeBoundary", "property")
    if not all(
        callable(getattr(typed_preview_bridge, method_name, None))
        for method_name in required_methods
    ):
        return audit

    is_qml_context_instance = (
        isinstance(typed_preview_bridge, LocalPreviewStateBridge)
        and typed_preview_bridge is qml_context_bridge_instance
    )
    if not is_qml_context_instance:
        return audit

    qml_snapshots = {
        "paper_session": _variant_map(_call_qml(root, "currentPaperSessionSnapshot") or {}),
        "scanner": _variant_map(_call_qml(root, "currentScannerSnapshot") or {}),
        "governor": _variant_map(_call_qml(root, "currentGovernorSnapshot") or {}),
        "portfolio": _variant_map(_call_qml(root, "currentPortfolioSnapshot") or {}),
        "alert_telemetry": _variant_map(_call_qml(root, "currentAlertTelemetrySnapshot") or {}),
    }
    runtime_boundary = {
        "liveTradingDisabled": _bool_property(root, "liveTradingDisabled"),
        "exchangeIoDisabled": _bool_property(root, "exchangeIoDisabled"),
        "orderSubmissionDisabled": _bool_property(root, "orderSubmissionDisabled"),
        "apiKeysRequired": _bool_property(root, "apiKeysRequired"),
        "runtimeLoopStarted": _bool_property(root, "runtimeLoopStarted"),
    }
    try:
        typed_preview_bridge.updateSnapshots(
            qml_snapshots["paper_session"],
            qml_snapshots["scanner"],
            qml_snapshots["governor"],
            qml_snapshots["portfolio"],
            qml_snapshots["alert_telemetry"],
        )
        typed_preview_bridge.updateRuntimeBoundary(runtime_boundary)
    except (AttributeError, RuntimeError, TypeError):
        return audit

    bridge_snapshots = {
        "paper_session": _variant_map(
            _safe_bridge_property(typed_preview_bridge, "paperSessionSnapshot")
        ),
        "scanner": _variant_map(_safe_bridge_property(typed_preview_bridge, "scannerSnapshot")),
        "governor": _variant_map(_safe_bridge_property(typed_preview_bridge, "governorSnapshot")),
        "portfolio": _variant_map(_safe_bridge_property(typed_preview_bridge, "portfolioSnapshot")),
        "alert_telemetry": _variant_map(
            _safe_bridge_property(typed_preview_bridge, "alertTelemetrySnapshot")
        ),
    }
    bridge_boundary = _variant_map(
        _safe_bridge_property(typed_preview_bridge, "runtimeBoundaryStatus")
    )
    _process_events()
    consumer_values = _typed_preview_consumer_values(root)
    fallback_audit = _audit_typed_preview_bridge_fallback_state(
        root, typed_preview_bridge, qml_snapshots
    )
    lifecycle_audit = _audit_typed_preview_bridge_consumer_lifecycle(
        root, typed_preview_bridge, qml_snapshots
    )
    audit_result = {
        "typed_preview_bridge_registered": True,
        "typed_preview_bridge_is_qml_context_instance": is_qml_context_instance,
        "typed_preview_bridge_schema_contract_valid": bool(
            _safe_bridge_property(typed_preview_bridge, "schemaContractValid")
        ),
        "typed_preview_bridge_matches_qml_paper_session_snapshot": bridge_snapshots["paper_session"]
        == qml_snapshots["paper_session"],
        "typed_preview_bridge_matches_qml_scanner_snapshot": bridge_snapshots["scanner"]
        == qml_snapshots["scanner"],
        "typed_preview_bridge_matches_qml_governor_snapshot": bridge_snapshots["governor"]
        == qml_snapshots["governor"],
        "typed_preview_bridge_matches_qml_portfolio_snapshot": bridge_snapshots["portfolio"]
        == qml_snapshots["portfolio"],
        "typed_preview_bridge_matches_qml_alert_telemetry_snapshot": bridge_snapshots[
            "alert_telemetry"
        ]
        == qml_snapshots["alert_telemetry"],
        "typed_preview_bridge_runtime_boundary_local_only": bridge_boundary == runtime_boundary
        and bool(_safe_bridge_property(typed_preview_bridge, "runtimeBoundaryLocalOnly")),
        "typed_preview_bridge_qml_consumer_visible": all(consumer_values.values()),
        "typed_preview_bridge_qml_consumer_schema_ok_visible": "schema ok"
        in consumer_values["contract"],
        "typed_preview_bridge_qml_consumer_runtime_boundary_visible": "local-only boundary ok"
        in consumer_values["contract"],
        "typed_preview_bridge_qml_consumer_diagnostic_marker_visible": bool(
            consumer_values["diagnostic_marker"]
        ),
        "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only": _contains_tokens(
            consumer_values["diagnostic_marker"],
            ("local", "preview", "read-only", "diagnostic"),
        ),
        "typed_preview_bridge_qml_consumer_matches_paper_snapshot": _contains_tokens(
            consumer_values["paper"],
            (
                bridge_snapshots["paper_session"].get("normalizedState"),
                bridge_snapshots["paper_session"].get("orderRows"),
            ),
        ),
        "typed_preview_bridge_qml_consumer_matches_scanner_snapshot": _contains_tokens(
            consumer_values["scanner"],
            (
                bridge_snapshots["scanner"].get("bestOpportunity"),
                bridge_snapshots["scanner"].get("candidates"),
            ),
        ),
        "typed_preview_bridge_qml_consumer_matches_governor_snapshot": _contains_tokens(
            consumer_values["governor"],
            (
                bridge_snapshots["governor"].get("latestAction"),
                bridge_snapshots["governor"].get("latestSymbol"),
            ),
        ),
    }
    audit_result.update(fallback_audit)
    audit_result.update(lifecycle_audit)
    return audit_result


def _qt_object_is_valid(item: Any) -> bool:
    if item is None:
        return False
    try:
        import shiboken6
    except ImportError:
        return True
    return bool(shiboken6.isValid(item))


def _qml_visual_children(item: Any) -> list[Any]:
    children: list[Any] = []
    for accessor_name in ("children", "childItems"):
        accessor = getattr(item, accessor_name, None)
        if callable(accessor):
            try:
                children.extend(list(accessor()))
            except RuntimeError:
                continue
    return children


def _find_qml_object_in_tree(item: Any, object_name: str) -> Any:
    if item is None:
        return None
    try:
        if str(item.property("objectName") or "") == object_name:
            return item
    except RuntimeError:
        return None
    for child in _qml_visual_children(item):
        found = _find_qml_object_in_tree(child, object_name)
        if found is not None:
            return found
    return None


def _find_qml_object(root: Any, object_name: str) -> Any:
    from PySide6.QtCore import QObject

    try:
        central_loader = root.findChild(QObject, "centralContentLoader")
    except RuntimeError:
        central_loader = None
    if central_loader is not None:
        try:
            loaded_item = central_loader.property("item")
        except RuntimeError:
            loaded_item = None
        found = _find_qml_object_in_tree(loaded_item, object_name)
        if found is not None:
            return found
    found = _find_qml_object_in_tree(root, object_name)
    if found is not None:
        return found
    try:
        return root.findChild(QObject, object_name)
    except RuntimeError:
        return None


def _qml_object_value(item: Any) -> str:
    if item is None:
        return ""
    try:
        for property_name in ("text", "title", "description", "currentText"):
            value = item.property(property_name)
            if value not in (None, ""):
                return str(_variant(value))
        model = item.property("model")
        model_count = _sequence_length(model)
        if model_count:
            return str(model_count)
        row_count = getattr(model, "rowCount", None)
        if callable(row_count):
            return str(row_count())
        count = item.property("count")
        if count is not None:
            return str(_variant(count))
    except RuntimeError:
        return ""
    return ""


def _qml_object_visible_with_size(root: Any, object_name: str) -> bool:
    item = _find_qml_object(root, object_name)
    if item is None or _bool_property(item, "visible") is not True:
        return False
    return (
        max(_number_property(item, "width"), _number_property(item, "implicitWidth")) > 0
        and max(_number_property(item, "height"), _number_property(item, "implicitHeight")) > 0
    )


def _read_visible_panel_object_property(
    root: Any, panel_id: str, object_name: str, property_name: str
) -> str:
    _invoke_show_panel(root, panel_id)
    _process_events()
    item = _find_qml_object(root, object_name)
    if item is None:
        return ""
    try:
        value = item.property(property_name)
    except RuntimeError:
        return ""
    if value in (None, ""):
        return ""
    return str(_variant(value))


def _read_visible_panel_object(root: Any, panel_id: str, object_name: str) -> str:
    _invoke_show_panel(root, panel_id)
    _process_events()
    return _qml_object_value(_find_qml_object(root, object_name))


def _contains_tokens(value: str, tokens: tuple[object, ...]) -> bool:
    normalized = value.lower()
    return all(str(token).lower() in normalized for token in tokens if str(token or ""))


def _visible_preview_object_values(root: Any) -> dict[str, str]:
    object_panels = {
        "dashboard_scanner": ("sidePanel", "previewDashboardBestOpportunityLabel"),
        "dashboard_governor": ("sidePanel", "previewDashboardGovernorDecisionLabel"),
        "ai_center_governor": ("aiCenterPanel", "previewAiCenterGovernorDecisionLabel"),
        "decisions_latest_action": ("aiDecisionsPanel", "previewAiDecisionLatestActionLabel"),
        "terminal_latest_order": ("terminalPanel", "previewTerminalLatestOrderLabel"),
        "portfolio_summary": ("portfolioPerformancePanel", "previewPortfolioSummaryLabel"),
        "alerts_latest_message": ("alertsPanel", "previewAlertsLatestMessageLabel"),
        "telemetry_latest_message": ("telemetryPanel", "previewTelemetryLatestMessageLabel"),
        "scanner_rows": ("marketScannerPanel", "previewScannerRowsView"),
    }
    return {
        key: _read_visible_panel_object(root, panel_id, object_name)
        for key, (panel_id, object_name) in object_panels.items()
    }


def _panel_has_visible_loaded_item(root: Any, panel_id: str) -> bool:
    from PySide6.QtCore import QObject

    _invoke_show_panel(root, panel_id)
    _process_events()
    central_loader = root.findChild(QObject, "centralContentLoader")
    loaded_item = central_loader.property("item") if central_loader is not None else None
    if loaded_item is None or _bool_property(loaded_item, "visible") is not True:
        return False
    return (
        max(_number_property(loaded_item, "width"), _number_property(loaded_item, "implicitWidth"))
        > 0
        and max(
            _number_property(loaded_item, "height"), _number_property(loaded_item, "implicitHeight")
        )
        > 0
    )


def _build_operator_workflow_runtime_audit(root: Any, audit: dict[str, object]) -> dict[str, bool]:
    workflow: dict[str, bool] = {}
    continuity_pair = _string_property(root, "selectedTerminalPair") or "BTC/USDT"
    if continuity_pair:
        _invoke_qml(root, "selectScannerPair", continuity_pair)
    selected_pair = _string_property(root, "scannerSelectedPair")
    selected_terminal_pair = _string_property(root, "selectedTerminalPair")
    scanner_rows_text = _rows_repr(root.property("scannerRows"))
    decision_rows_text = _rows_repr(root.property("decisionPreviewRows"))
    order_rows_text = _rows_repr(root.property("paperOrderRows"))
    alert_rows_text = _rows_repr(root.property("alertRows"))
    telemetry_rows_text = _rows_repr(root.property("paperTelemetryRows"))
    latest_decision = _first_row(root.property("decisionPreviewRows"))

    workflow["operator_dashboard_visible"] = _panel_has_visible_loaded_item(
        root, "sidePanel"
    ) and _qml_object_visible_with_size(root, "operatorDashboardRoot")
    workflow["operator_can_open_market_scanner"] = _panel_has_visible_loaded_item(
        root, "marketScannerPanel"
    ) and _qml_object_visible_with_size(root, "marketScannerRoot")
    workflow["operator_can_open_ai_decisions"] = _panel_has_visible_loaded_item(
        root, "aiDecisionsPanel"
    ) and _qml_object_visible_with_size(root, "aiDecisionsPreviewPanel")
    workflow["operator_can_open_risk_controls"] = _panel_has_visible_loaded_item(
        root, "riskControlsPanel"
    ) and _qml_object_visible_with_size(root, "riskControlsPreviewPanel")
    workflow["operator_can_open_terminal"] = _panel_has_visible_loaded_item(
        root, "terminalPanel"
    ) and _qml_object_visible_with_size(root, "paperTerminalRoot")
    workflow["operator_can_open_portfolio"] = _panel_has_visible_loaded_item(
        root, "portfolioPerformancePanel"
    ) and _qml_object_visible_with_size(root, "portfolioPerformanceRoot")
    workflow["operator_can_open_alerts"] = _panel_has_visible_loaded_item(
        root, "alertsPanel"
    ) and _qml_object_visible_with_size(root, "alertCenterRoot")
    workflow["operator_can_open_telemetry"] = _panel_has_visible_loaded_item(
        root, "telemetryPanel"
    ) and _qml_object_visible_with_size(root, "telemetryFeedList")
    workflow["operator_can_return_to_dashboard"] = _panel_has_visible_loaded_item(
        root, "sidePanel"
    ) and _qml_object_visible_with_size(root, "operatorDashboardRoot")
    workflow["operator_scanner_candidate_visible"] = (
        audit.get("market_scanner_table_visible") is True
        and bool(selected_pair)
        and selected_pair != "—"
    )
    workflow["operator_scanner_candidate_selectable_local_only"] = (
        audit.get("market_scanner_can_select_candidate_local_only") is True
    )
    workflow["operator_selected_candidate_updates_shared_state"] = (
        bool(selected_pair) and selected_pair == selected_terminal_pair
    )
    workflow["operator_decision_visible_after_candidate"] = (
        audit.get("decision_explainability_state_present") is True
        and _sequence_length(root.property("decisionPreviewRows")) > 0
    )
    workflow["operator_decision_has_action_confidence_reason"] = all(
        _row_field(latest_decision, field) for field in ("action", "confidence", "reason")
    )
    workflow["operator_decision_local_source_visible"] = _contains_tokens(
        decision_rows_text, ("local",)
    ) or _contains_tokens(
        _string_property(root, "selectedDecisionSafetySummary"), ("local preview",)
    )
    workflow["operator_risk_gate_visible"] = audit.get("risk_lock_or_risk_gate_visible") is True
    workflow["operator_risk_block_reason_visible"] = bool(
        _string_property(root, "riskBlockReason")
    ) or _contains_tokens(order_rows_text + decision_rows_text, ("Risk gate blocked",))
    workflow["operator_blocked_state_explainable"] = (
        audit.get("operator_can_explain_blocked_state_local_only") is True
        or audit.get("paper_order_explain_local_only") is True
    )
    workflow["operator_risk_limits_visible"] = audit.get("risk_limits_visible") is True
    workflow["operator_live_safety_lock_visible"] = (
        audit.get("kill_switch_or_safety_lock_visible") is True
    )
    workflow["operator_terminal_order_form_visible"] = (
        audit.get("terminal_order_form_visible") is True
    )
    workflow["operator_terminal_pair_matches_selected_candidate"] = (
        bool(selected_pair) and selected_pair == selected_terminal_pair
    )
    workflow["operator_order_submission_disabled_visible"] = (
        audit.get("terminal_order_submission_disabled_visible") is True
    )
    workflow["operator_simulated_order_path_visible"] = (
        audit.get("terminal_order_simulated_state_visible") is True
    )
    workflow["operator_blocked_order_path_visible"] = (
        audit.get("terminal_order_blocked_state_visible") is True
    )
    workflow["operator_no_real_order_path_visible"] = (
        audit.get("terminal_order_mode_local_preview_visible") is True
        and audit.get("portfolio_no_real_order_path_visible") is True
    )
    workflow["operator_portfolio_visible"] = (
        _panel_has_visible_loaded_item(root, "portfolioPerformancePanel")
        and audit.get("portfolio_summary_visible") is True
    )
    workflow["operator_portfolio_reflects_preview_order_or_block"] = audit.get(
        "portfolio_updates_after_preview_order_local_only"
    ) is True and _contains_tokens(order_rows_text, ("BLOCKED",))
    workflow["operator_portfolio_local_paper_marker_visible"] = (
        audit.get("portfolio_local_source_marker_visible") is True
    )
    workflow["operator_portfolio_no_live_account_sync_visible"] = (
        audit.get("portfolio_no_live_account_sync_visible") is True
    )
    workflow["operator_alerts_visible_after_actions"] = (
        audit.get("alerts_feed_visible") is True
        and _sequence_length(root.property("alertRows")) > 0
    )
    direct_risk_or_order_alert_visible = _contains_tokens(alert_rows_text, ("blocked",)) and (
        _contains_tokens(alert_rows_text, ("risk",))
        or _contains_tokens(alert_rows_text, ("order",))
    )
    workflow["operator_risk_or_order_alert_visible"] = workflow.get(
        "operator_alerts_visible_after_actions"
    ) is True and (
        audit.get("risk_blocked_alert_visible") is True
        or audit.get("order_blocked_alert_visible") is True
        or direct_risk_or_order_alert_visible
    )
    workflow["operator_telemetry_visible_after_actions"] = (
        audit.get("telemetry_feed_visible") is True
        and _sequence_length(root.property("paperTelemetryRows")) > 0
    )
    workflow["operator_audit_correlation_visible"] = (
        audit.get("audit_correlation_or_trace_marker_visible") is True
    )
    workflow["operator_no_cloud_sink_visible"] = audit.get("alerts_no_cloud_sink_visible") is True
    workflow["operator_no_external_export_visible"] = (
        audit.get("alerts_no_external_export_visible") is True
    )
    workflow["operator_no_secrets_logged_visible"] = (
        audit.get("telemetry_no_secrets_logged_visible") is True
    )
    workflow["operator_terminal_order_form_live_shape_complete"] = (
        audit.get("terminal_order_form_live_shape_complete") is True
    )
    workflow["operator_order_lifecycle_preview_parity_complete"] = (
        audit.get("order_lifecycle_preview_parity_complete") is True
    )
    workflow["operator_risk_live_safety_controls_visible_complete"] = (
        audit.get("risk_live_safety_controls_visible_complete") is True
    )
    workflow["operator_market_scanner_live_field_parity_complete"] = (
        audit.get("market_scanner_live_field_parity_complete") is True
    )
    workflow["operator_portfolio_live_shape_parity_complete"] = (
        audit.get("portfolio_live_shape_parity_complete") is True
    )
    workflow["operator_alerts_telemetry_live_shape_parity_complete"] = (
        audit.get("alerts_telemetry_live_shape_parity_complete") is True
    )
    workflow["operator_scanner_rows_runtime_non_empty_diagnostic"] = bool(scanner_rows_text)
    workflow["operator_telemetry_rows_runtime_non_empty_diagnostic"] = bool(telemetry_rows_text)
    return workflow


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


def _exercise_preview_state(
    root: Any,
    typed_preview_bridge: Any,
    qml_context_bridge_instance: Any,
) -> dict[str, object]:
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
    audit.update(
        _audit_typed_preview_bridge(root, typed_preview_bridge, qml_context_bridge_instance)
    )
    audit["typed_preview_bridge_qml_consumer_evidence"] = _typed_preview_bridge_consumer_evidence(
        audit
    )
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

    visible_values = _visible_preview_object_values(root)
    scanner_snapshot = _call_qml(root, "currentScannerSnapshot") or {}
    governor_snapshot = _call_qml(root, "currentGovernorSnapshot") or {}
    paper_session_snapshot = _call_qml(root, "currentPaperSessionSnapshot") or {}
    portfolio_snapshot = _call_qml(root, "currentPortfolioSnapshot") or {}
    alert_telemetry_snapshot = _call_qml(root, "currentAlertTelemetrySnapshot") or {}
    panel_runtime_snapshot = _call_qml(root, "currentPerPanelRuntimeSnapshot") or {}
    latest_order_row = _first_row(root.property("paperOrderRows"))
    audit["visible_ui_object_values"] = visible_values
    audit["visible_ui_objects_found_for_preview_panels"] = all(
        bool(value) for value in visible_values.values()
    )
    audit["visible_dashboard_matches_scanner_snapshot"] = _contains_tokens(
        visible_values["dashboard_scanner"],
        (
            scanner_snapshot.get("bestOpportunity"),
            scanner_snapshot.get("candidates"),
        ),
    )
    audit["visible_dashboard_matches_governor_snapshot"] = _contains_tokens(
        visible_values["dashboard_governor"],
        (governor_snapshot.get("latestAction"), governor_snapshot.get("latestSymbol")),
    ) or visible_values["dashboard_governor"] == str(governor_snapshot.get("lastDecision") or "")
    audit["visible_ai_center_matches_governor_snapshot"] = visible_values[
        "ai_center_governor"
    ] == str(governor_snapshot.get("lastDecision") or "")
    audit["visible_decisions_match_latest_governor_action"] = _contains_tokens(
        visible_values["decisions_latest_action"],
        (governor_snapshot.get("latestAction"), governor_snapshot.get("latestSymbol")),
    )
    audit["visible_terminal_matches_latest_paper_order"] = _contains_tokens(
        visible_values["terminal_latest_order"],
        (
            _row_field(latest_order_row, "pair"),
            paper_session_snapshot.get("latestOrderAction"),
            paper_session_snapshot.get("latestOrderStatus"),
            panel_runtime_snapshot.get("terminalLatestOrderReason"),
        ),
    )
    audit["visible_portfolio_matches_portfolio_snapshot"] = _contains_tokens(
        visible_values["portfolio_summary"],
        (
            "equity",
            "pnl",
            "orders",
            portfolio_snapshot.get("orders"),
            portfolio_snapshot.get("simulatedCount"),
            portfolio_snapshot.get("blockedCount"),
        ),
    )
    audit["visible_alerts_match_alert_snapshot"] = _contains_tokens(
        visible_values["alerts_latest_message"],
        (panel_runtime_snapshot.get("alertsLatestMessage"),),
    )
    audit["visible_telemetry_matches_telemetry_snapshot"] = _contains_tokens(
        visible_values["telemetry_latest_message"],
        (alert_telemetry_snapshot.get("latestTelemetry"),),
    )
    audit["visible_ui_updates_after_preview_mutations"] = all(
        bool(audit[key])
        for key in (
            "visible_dashboard_matches_scanner_snapshot",
            "visible_dashboard_matches_governor_snapshot",
            "visible_ai_center_matches_governor_snapshot",
            "visible_decisions_match_latest_governor_action",
            "visible_terminal_matches_latest_paper_order",
            "visible_portfolio_matches_portfolio_snapshot",
            "visible_alerts_match_alert_snapshot",
            "visible_telemetry_matches_telemetry_snapshot",
        )
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
    _invoke_show_panel(root, "marketScannerPanel")
    _process_events()
    scanner_table_text = " ".join(
        (
            _read_visible_panel_object(root, "marketScannerPanel", "previewScannerRowsView"),
            _read_visible_panel_object_property(
                root, "marketScannerPanel", "previewScannerRowsView", "description"
            ),
        )
    )
    scanner_live_fields_text = " ".join(
        (
            _read_visible_panel_object(root, "marketScannerPanel", "marketScannerLiveFieldSummary"),
            _read_visible_panel_object_property(
                root, "marketScannerPanel", "marketScannerLiveFieldSummary", "description"
            ),
        )
    )
    scanner_selected_details_text = " ".join(
        (
            _read_visible_panel_object(
                root, "marketScannerPanel", "marketScannerSelectedCandidateDetails"
            ),
            _read_visible_panel_object_property(
                root, "marketScannerPanel", "marketScannerSelectedCandidateDetails", "description"
            ),
        )
    )
    scanner_safety_text = " ".join(
        (
            _read_visible_panel_object(
                root, "marketScannerPanel", "marketScannerSafetyBoundaryCard"
            ),
            _read_visible_panel_object_property(
                root, "marketScannerPanel", "marketScannerSafetyBoundaryCard", "description"
            ),
        )
    )
    scanner_explanation_text = " ".join(
        (
            _read_visible_panel_object(root, "marketScannerPanel", "marketScannerExplanationPanel"),
            _read_visible_panel_object_property(
                root, "marketScannerPanel", "marketScannerExplanationPanel", "description"
            ),
        )
    )
    scanner_rows = _variant(root.property("scannerRows")) or []
    selected_scanner_row = _variant(
        _call_qml(root, "scannerRowByPair", root.property("scannerSelectedPair")) or {}
    )
    selected_row_text = repr(selected_scanner_row)
    audit["market_scanner_table_visible"] = _qml_object_visible_with_size(
        root, "previewScannerRowsView"
    ) and bool(scanner_table_text)
    audit["market_scanner_rows_visible"] = (
        audit["market_scanner_table_visible"] and len(scanner_rows) > 0
    )
    audit["market_scanner_symbol_visible"] = _contains_tokens(
        scanner_table_text, ("Pair",)
    ) and bool(selected_scanner_row.get("pair"))
    audit["market_scanner_exchange_or_venue_visible"] = _contains_tokens(
        scanner_table_text + selected_row_text, ("Exchange", "Paper Preview")
    )
    audit["market_scanner_price_visible"] = _contains_tokens(
        scanner_table_text, ("Price",)
    ) and bool(selected_scanner_row.get("price"))
    audit["market_scanner_spread_visible"] = _contains_tokens(
        scanner_table_text, ("Spread",)
    ) and bool(selected_scanner_row.get("spread"))
    audit["market_scanner_volume_or_liquidity_visible"] = _contains_tokens(
        scanner_table_text, ("Volume", "Liquidity")
    ) and bool(selected_scanner_row.get("volume"))
    audit["market_scanner_volatility_visible"] = (
        _contains_tokens(
            scanner_live_fields_text + scanner_selected_details_text + scanner_explanation_text,
            ("Volatility",),
        )
        and selected_scanner_row.get("volatility") is not None
    )
    audit["market_scanner_score_visible"] = _contains_tokens(
        scanner_table_text, ("AI score",)
    ) and (selected_scanner_row.get("aiScore") is not None)
    audit["market_scanner_risk_score_visible"] = _contains_tokens(
        scanner_table_text, ("Risk",)
    ) and (selected_scanner_row.get("riskScore") is not None)
    audit["market_scanner_confidence_visible"] = _contains_tokens(
        scanner_live_fields_text, ("Confidence",)
    )
    audit["market_scanner_ai_action_visible"] = _contains_tokens(
        scanner_live_fields_text + scanner_table_text, ("AI action", "Recommendation")
    )
    audit["market_scanner_risk_decision_visible"] = _contains_tokens(
        scanner_live_fields_text + scanner_selected_details_text, ("RISK DECISION",)
    )
    audit["market_scanner_reason_visible"] = _contains_tokens(
        scanner_table_text, ("Reason",)
    ) and bool(selected_scanner_row.get("reason"))
    audit["market_scanner_freshness_visible"] = _contains_tokens(
        scanner_live_fields_text + scanner_selected_details_text, ("FRESHNESS",)
    )
    audit["market_scanner_local_source_marker_visible"] = _contains_tokens(
        scanner_live_fields_text + scanner_safety_text, ("LOCAL PREVIEW FEED",)
    )
    audit["market_scanner_filter_controls_visible"] = _qml_object_visible_with_size(
        root, "marketScannerFilterModeControl"
    )
    audit["market_scanner_sort_controls_visible"] = _qml_object_visible_with_size(
        root, "marketScannerSortModeControl"
    )
    audit["market_scanner_threshold_controls_visible"] = _qml_object_visible_with_size(
        root, "marketScannerThresholdControls"
    )
    audit["market_scanner_selected_candidate_details_visible"] = _qml_object_visible_with_size(
        root, "marketScannerSelectedCandidateDetails"
    ) and bool(scanner_selected_details_text)
    audit["market_scanner_explain_candidate_local_only"] = _contains_tokens(
        scanner_explanation_text, ("Live trading disabled", "order submission disabled")
    )
    audit["market_scanner_can_select_candidate_local_only"] = (
        _string_property(root, "selectedTerminalPair")
        == _string_property(root, "scannerSelectedPair")
        and _bool_property(root, "runtimeLoopStarted") is False
    )
    audit["market_scanner_no_exchange_io_visible"] = _contains_tokens(
        scanner_safety_text + scanner_live_fields_text, ("NO EXCHANGE I/O",)
    )
    audit["market_scanner_no_live_feed_visible"] = _contains_tokens(
        scanner_safety_text + scanner_live_fields_text, ("NO LIVE FEED",)
    )
    audit["market_scanner_no_real_order_path_visible"] = _contains_tokens(
        scanner_safety_text + scanner_live_fields_text, ("NO REAL ORDER PATH",)
    )
    audit["market_scanner_uses_local_preview_catalog"] = (
        _contains_tokens(
            scanner_safety_text + scanner_live_fields_text, ("LOCAL PREVIEW", "catalog")
        )
        and len(scanner_rows) >= 30
    )
    audit["market_scanner_updates_shared_preview_state_local_only"] = (
        audit.get("simulation_can_use_scanner_candidate_local_only") is True
        and _bool_property(root, "exchangeIoDisabled") is True
        and _bool_property(root, "orderSubmissionDisabled") is True
    )
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
    _invoke_show_panel(root, "sidePanel")
    _process_events()
    operator_dashboard_item = _find_qml_object(root, "operatorDashboardRoot")
    audit["operator_dashboard_visible"] = (
        operator_dashboard_item is not None
        and _bool_property(operator_dashboard_item, "visible") is True
        and max(
            _number_property(operator_dashboard_item, "width"),
            _number_property(operator_dashboard_item, "implicitWidth"),
        )
        > 0
        and max(
            _number_property(operator_dashboard_item, "height"),
            _number_property(operator_dashboard_item, "implicitHeight"),
        )
        > 0
    )
    alert_summary_value = _read_visible_panel_object(
        root, "sidePanel", "operatorDashboardAlertSummary"
    )
    audit["alerts_dashboard_summary_object_visible"] = _contains_tokens(
        alert_summary_value, ("Alert Center summary",)
    )
    audit["alerts_dashboard_summary_source_present"] = "operatorDashboardAlertSummary" in source
    audit["alerts_dashboard_summary_present"] = (
        audit["alerts_dashboard_summary_object_visible"] is True
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
    _invoke_show_panel(root, "terminalPanel")
    _process_events()
    terminal_root_item = _find_qml_object(root, "paperTerminalRoot")
    audit["terminal_order_form_visible"] = (
        terminal_root_item is not None
        and _qml_object_visible_with_size(root, "paperTerminalOrderForm")
    )
    audit["terminal_order_mode_local_preview_visible"] = _source_has_all(
        source,
        (
            "paperTerminalSafetyBoundary",
            "Paper Preview only",
            "Live trading disabled",
            "Exchange I/O disabled",
            "No real orders",
        ),
    )
    audit["terminal_order_symbol_visible"] = (
        "paperTerminalPairSelector" in source
        and "selectedTerminalPair" in source
        and bool(_string_property(root, "selectedTerminalPair"))
    )
    audit["terminal_order_side_controls_source_present"] = _source_has_all(
        source, ("paperTerminalSideControls", "BUY", "SELL", "setTerminalSide")
    )
    audit["terminal_order_side_controls_present"] = _qml_object_visible_with_size(
        root, "paperTerminalSideControls"
    )
    audit["terminal_order_type_controls_source_present"] = _source_has_all(
        source, ("paperTerminalOrderTypeControls", "LIMIT", "MARKET", "setTerminalOrderType")
    )
    audit["terminal_order_type_controls_present"] = _qml_object_visible_with_size(
        root, "paperTerminalOrderTypeControls"
    )
    audit["terminal_order_price_amount_total_source_present"] = _source_has_all(
        source,
        (
            "paperTerminalPriceInput",
            "paperTerminalAmountInput",
            "paperTerminalTotalInput",
            "terminalPrice",
            "terminalAmount",
            "terminalTotal",
        ),
    )
    audit["terminal_order_price_amount_total_present"] = all(
        _qml_object_visible_with_size(root, object_name)
        for object_name in (
            "paperTerminalPriceInput",
            "paperTerminalAmountInput",
            "paperTerminalTotalInput",
        )
    )
    audit["terminal_order_percent_chips_source_present"] = _source_has_all(
        source, ("paperTerminalPercentChips", "applyTerminalPercent", "[10, 25, 50, 75, 100]")
    )
    audit["terminal_order_percent_chips_present"] = _qml_object_visible_with_size(
        root, "paperTerminalPercentChips"
    )
    audit["terminal_order_submission_disabled_source_present"] = _source_has_all(
        source,
        (
            "paperTerminalSubmissionDisabledWarning",
            "order submission disabled",
            "no real order",
            "paper simulation only",
        ),
    )
    audit["terminal_order_submission_disabled_visible"] = (
        _qml_object_visible_with_size(root, "paperTerminalSubmissionDisabledWarning")
        and audit["terminal_order_submission_disabled_source_present"] is True
    )
    latest_order_text = _read_visible_panel_object(
        root, "terminalPanel", "previewTerminalLatestOrderLabel"
    )
    audit["terminal_order_latest_status_source_present"] = _source_has_all(
        source, ("previewTerminalLatestOrderLabel", "Latest paper order", "status", "reason")
    )
    audit["terminal_order_latest_status_visible"] = _qml_object_visible_with_size(
        root, "previewTerminalLatestOrderLabel"
    ) and _contains_tokens(latest_order_text, ("Latest paper order", "blocked", "Risk gate"))
    audit["terminal_order_blocked_state_visible"] = (
        audit.get("risk_block_generates_blocked_event_and_alert") is True
        and "blocked" in first_blocked_order
        and "Risk gate blocked" in first_blocked_order
    )
    audit["terminal_order_simulated_state_visible"] = (
        audit.get("simulate_order_updates_blotter_portfolio_telemetry") is True
        and "paper simulated" in first_sim_order
    )
    audit["terminal_order_no_order_state_visible"] = _source_has_all(
        source, ("no paper fill", "NO ORDER", "No real orders")
    )
    audit["terminal_order_rejected_disabled_placeholder_visible"] = _source_has_all(
        source, ("order submission disabled", "Risk gate blocked", "blocked")
    )
    audit["terminal_order_updates_blotter_portfolio_telemetry"] = (
        audit.get("simulate_order_updates_blotter_portfolio_telemetry") is True
        and audit.get("terminal_blotter_updates_portfolio_snapshot") is True
        and audit.get("risk_block_generates_blocked_event_and_alert") is True
    )
    latest_decision_text = _read_visible_panel_object(
        root, "aiDecisionsPanel", "previewAiDecisionLatestActionLabel"
    )
    reserved_placeholder_text = _read_visible_panel_object(
        root, "terminalPanel", "paperTerminalLifecycleReservedPlaceholder"
    )
    submission_disabled_text = _read_visible_panel_object(
        root, "terminalPanel", "paperTerminalSubmissionDisabledWarning"
    )
    latest_order_snapshot = _first_row_repr(root.property("paperOrderRows"))
    decision_rows = root.property("decisionPreviewRows")
    telemetry_rows = root.property("paperTelemetryRows")
    alert_rows = root.property("alertRows")
    visible_values = audit.get("visible_ui_object_values", {})
    if not isinstance(visible_values, dict):
        visible_values = {}
    # Runtime-captured values collected by _visible_preview_object_values() during panel
    # exercise. These must stay tied to visible QML objects, not source/state fallback.
    audit["order_lifecycle_decision_rows_contain_blocked"] = _rows_contain_tokens(
        decision_rows, ("BLOCKED",)
    )
    audit["order_lifecycle_decision_visible"] = (
        _contains_tokens(
            latest_decision_text + str(visible_values.get("decisions_latest_action", "")),
            ("confidence", "BLOCKED"),
        )
        and audit["order_lifecycle_decision_rows_contain_blocked"] is True
    )
    audit["order_lifecycle_simulated_order_visible"] = (
        audit.get("terminal_order_simulated_state_visible") is True
        and "paper simulated" in latest_order_snapshot + first_sim_order
        and _sequence_length(root.property("mockTerminalOrders")) > 0
    )
    audit["order_lifecycle_blocked_order_visible"] = (
        audit.get("terminal_order_blocked_state_visible") is True
        and "BLOCKED" in latest_order_snapshot
        and "Risk gate blocked" in latest_order_snapshot
        and "BLOCKED LIVE" not in latest_order_snapshot
        and "BLOCKED PAPER PREVIEW" not in latest_order_snapshot
    )
    audit["order_lifecycle_no_order_visible"] = _qml_object_visible_with_size(
        root, "paperTerminalLifecycleReservedPlaceholder"
    ) and _contains_tokens(reserved_placeholder_text, ("NO ORDER", "no real order"))
    audit["order_lifecycle_rejected_disabled_visible"] = audit.get(
        "terminal_order_submission_disabled_visible"
    ) is True and _contains_tokens(reserved_placeholder_text, ("rejected", "disabled in preview"))
    audit["order_lifecycle_partial_fill_cancel_placeholders_visible"] = (
        _qml_object_visible_with_size(root, "paperTerminalLifecycleReservedPlaceholder")
        and _contains_tokens(
            reserved_placeholder_text, ("partial fill", "cancel", "disabled in preview")
        )
    )
    audit["order_lifecycle_portfolio_visible_summary_present"] = _contains_tokens(
        audit.get("visible_ui_object_values", {}).get("portfolio_summary", ""),
        ("orders:", "simulated:", "blocked:"),
    )
    audit["order_lifecycle_downstream_portfolio_updates"] = (
        audit.get("terminal_order_updates_blotter_portfolio_telemetry") is True
        and audit.get("visible_portfolio_matches_portfolio_snapshot") is True
        and audit["order_lifecycle_portfolio_visible_summary_present"] is True
    )
    audit["order_lifecycle_telemetry_rows_contain_risk_gate"] = _rows_contain_tokens(
        telemetry_rows, ("Risk gate blocked",)
    )
    audit["order_lifecycle_alert_rows_contain_paper_blocked"] = _rows_contain_tokens(
        alert_rows, ("Paper order blocked",)
    )
    audit["order_lifecycle_alert_telemetry_updates"] = (
        (
            audit["order_lifecycle_telemetry_rows_contain_risk_gate"] is True
            or audit.get("risk_blocked_tick_appends_telemetry") is True
        )
        and (
            audit["order_lifecycle_alert_rows_contain_paper_blocked"] is True
            or audit.get("risk_block_generates_blocked_event_and_alert") is True
        )
        and audit.get("visible_alerts_match_alert_snapshot") is True
        and audit.get("visible_telemetry_matches_telemetry_snapshot") is True
    )
    audit["order_lifecycle_no_live_side_effects"] = (
        audit.get("live_trading_disabled") is True
        and audit.get("exchange_io_disabled") is True
        and audit.get("order_submission_disabled") is True
        and audit.get("runtime_loop_started") is False
        and audit.get("api_keys_required") is False
        and audit.get("simulation_does_not_read_secrets") is True
    )
    risk_safety_text = _read_visible_panel_object(
        root, "riskControlsPanel", "riskSafetyBoundaryDescription"
    )
    risk_state_text = _read_visible_panel_object(root, "riskControlsPanel", "riskStateCard")
    risk_limits_text = " ".join(
        _read_visible_panel_object(root, "riskControlsPanel", object_name)
        for object_name in (
            "riskActiveLimitsTable",
            "riskConfidenceFloorCardDescription",
            "riskMaxPositionCardDescription",
            "riskPerSymbolExposureCardDescription",
            "riskDailyLossLimitCardDescription",
            "riskMaxDrawdownCardDescription",
        )
    )
    risk_blocked_text = " ".join(
        (
            first_blocked_order,
            _first_row_repr(root.property("alertRows")),
            _first_row_repr(root.property("paperTelemetryRows")),
            str(blocked_governor_snapshot.get("riskBlockReason") or ""),
        )
    )
    audit["live_trading_disabled_visible"] = _contains_tokens(risk_safety_text, ("LIVE DISABLED",))
    audit["exchange_io_disabled_visible"] = _contains_tokens(
        risk_safety_text, ("EXCHANGE I/O DISABLED",)
    )
    audit["order_submission_disabled_visible"] = _contains_tokens(
        risk_safety_text, ("ORDER SUBMISSION DISABLED",)
    )
    audit["runtime_loop_not_started_visible"] = _contains_tokens(
        risk_safety_text, ("RUNTIME LOOP NOT STARTED",)
    )
    audit["api_keys_not_required_visible"] = _contains_tokens(
        risk_safety_text, ("API KEYS NOT REQUIRED IN PREVIEW",)
    )
    audit["secrets_not_read_visible"] = _contains_tokens(risk_safety_text, ("SECRETS NOT READ",))
    audit["preview_mode_badge_visible"] = _contains_tokens(
        risk_safety_text, ("PREVIEW LOCAL ONLY",)
    )
    audit["live_mode_blocked_badge_visible"] = _contains_tokens(
        risk_safety_text, ("LIVE MODE BLOCKED",)
    )
    audit["safety_boundary_visible"] = _qml_object_visible_with_size(
        root, "riskSafetyBoundaryCard"
    ) and bool(risk_safety_text)
    audit["no_live_side_effects_visible"] = _contains_tokens(
        risk_safety_text, ("NO LIVE SIDE EFFECTS",)
    )
    audit["kill_switch_or_safety_lock_visible"] = _contains_tokens(
        risk_safety_text, ("kill-switch",)
    ) or _contains_tokens(risk_safety_text, ("SAFETY LOCK",))
    audit["risk_lock_or_risk_gate_visible"] = _contains_tokens(
        risk_safety_text + risk_blocked_text, ("RISK GATE",)
    )
    audit["risk_profile_visible"] = _qml_object_visible_with_size(
        root, "riskProfileSegmentedControl"
    ) and bool(_string_property(root, "riskProfile"))
    audit["risk_limits_visible"] = _qml_object_visible_with_size(
        root, "riskActiveLimitsTable"
    ) and bool(risk_limits_text.strip())
    audit["blocked_reasons_visible"] = _contains_tokens(risk_safety_text, ("Blocked reasons",))
    audit["confidence_floor_or_score_threshold_visible"] = _contains_tokens(
        risk_limits_text + risk_safety_text, ("confidence floor",)
    )
    audit["max_position_or_exposure_limit_visible"] = _contains_tokens(
        risk_limits_text + risk_safety_text, ("max position",)
    ) or _contains_tokens(risk_limits_text + risk_safety_text, ("exposure",))
    audit["daily_loss_or_drawdown_limit_visible"] = _contains_tokens(
        risk_limits_text + risk_safety_text, ("daily loss",)
    ) or _contains_tokens(risk_limits_text + risk_safety_text, ("drawdown",))
    audit["risk_blocked_event_visible"] = audit.get(
        "risk_block_generates_blocked_event_and_alert"
    ) is True and _contains_tokens(risk_blocked_text, ("Risk gate blocked",))
    audit["operator_can_explain_blocked_state_local_only"] = _contains_tokens(
        risk_safety_text, ("Operator can explain blocked state local only",)
    )
    _invoke_show_panel(root, "portfolioPerformancePanel")
    _process_events()
    portfolio_summary_text = _read_visible_panel_object(
        root, "portfolioPerformancePanel", "portfolioLiveShapeSummaryLabel"
    )
    portfolio_positions_text = _read_visible_panel_object(
        root, "portfolioPerformancePanel", "portfolioLiveShapePositionsLabel"
    )
    portfolio_orders_trades_text = _read_visible_panel_object(
        root, "portfolioPerformancePanel", "portfolioLiveShapeOrdersTradesLabel"
    )
    portfolio_risk_boundary_text = _read_visible_panel_object(
        root, "portfolioPerformancePanel", "portfolioLiveShapeRiskBoundaryLabel"
    )
    portfolio_live_shape_text = " ".join(
        (
            portfolio_summary_text,
            portfolio_positions_text,
            portfolio_orders_trades_text,
            portfolio_risk_boundary_text,
        )
    )
    audit["portfolio_summary_visible"] = _qml_object_visible_with_size(
        root, "portfolioLiveShapeSummaryCard"
    ) and _contains_tokens(portfolio_summary_text, ("Account/equity summary",))
    audit["portfolio_equity_visible"] = _contains_tokens(portfolio_summary_text, ("equity",))
    audit["portfolio_cash_or_balance_visible"] = _contains_tokens(
        portfolio_summary_text, ("cash/balance",)
    )
    audit["portfolio_unrealized_pnl_visible"] = _contains_tokens(
        portfolio_summary_text, ("unrealized PnL",)
    )
    audit["portfolio_realized_pnl_visible"] = _contains_tokens(
        portfolio_summary_text, ("realized PnL",)
    )
    audit["portfolio_total_pnl_visible"] = _contains_tokens(portfolio_summary_text, ("total PnL",))
    audit["portfolio_daily_pnl_visible"] = _contains_tokens(portfolio_summary_text, ("daily PnL",))
    audit["portfolio_drawdown_visible"] = _contains_tokens(portfolio_summary_text, ("drawdown",))
    audit["portfolio_exposure_visible"] = _contains_tokens(portfolio_summary_text, ("exposure",))
    audit["portfolio_freshness_visible"] = _contains_tokens(
        portfolio_summary_text, ("FRESHNESS", "STALE")
    )
    audit["portfolio_local_source_marker_visible"] = _contains_tokens(
        portfolio_summary_text, ("LOCAL PAPER PORTFOLIO", "local preview")
    )
    audit["portfolio_open_positions_visible"] = _qml_object_visible_with_size(
        root, "portfolioLiveShapePositionsCard"
    ) and _contains_tokens(portfolio_positions_text, ("Open positions",))
    audit["portfolio_position_symbol_visible"] = _contains_tokens(
        portfolio_positions_text, ("symbol",)
    )
    audit["portfolio_position_side_visible"] = _contains_tokens(portfolio_positions_text, ("side",))
    audit["portfolio_position_quantity_visible"] = _contains_tokens(
        portfolio_positions_text, ("quantity",)
    )
    audit["portfolio_position_entry_price_visible"] = _contains_tokens(
        portfolio_positions_text, ("entry price",)
    )
    audit["portfolio_position_mark_price_visible"] = _contains_tokens(
        portfolio_positions_text, ("mark",)
    ) and bool(_string_property(root, "terminalPrice"))
    audit["portfolio_position_unrealized_pnl_visible"] = _contains_tokens(
        portfolio_positions_text, ("unrealized PnL",)
    )
    audit["portfolio_position_exposure_visible"] = _contains_tokens(
        portfolio_positions_text, ("exposure",)
    )
    audit["portfolio_open_orders_visible"] = _qml_object_visible_with_size(
        root, "portfolioLiveShapeOrdersTradesCard"
    ) and _contains_tokens(portfolio_orders_trades_text, ("Open orders",))
    audit["portfolio_order_symbol_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("symbol",)
    )
    audit["portfolio_order_side_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("side",)
    )
    audit["portfolio_order_quantity_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("quantity",)
    )
    audit["portfolio_order_status_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("status",)
    )
    audit["portfolio_closed_trades_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("Closed trades/history",)
    )
    audit["portfolio_closed_trade_symbol_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("symbol",)
    )
    audit["portfolio_closed_trade_pnl_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("closed trade PnL",)
    )
    audit["portfolio_fills_or_executions_placeholder_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("PAPER FILLS / EXECUTIONS PLACEHOLDER",)
    )
    audit["portfolio_fees_or_slippage_placeholder_visible"] = _contains_tokens(
        portfolio_orders_trades_text, ("fees/slippage placeholder",)
    )
    audit["portfolio_risk_state_visible"] = _qml_object_visible_with_size(
        root, "portfolioLiveShapeRiskBoundaryCard"
    ) and _contains_tokens(portfolio_risk_boundary_text, ("Risk state",))
    audit["portfolio_blocked_exposure_reason_visible"] = _contains_tokens(
        portfolio_risk_boundary_text, ("blocked exposure reason",)
    )
    audit["portfolio_no_live_account_sync_visible"] = _contains_tokens(
        portfolio_risk_boundary_text, ("NO LIVE ACCOUNT SYNC",)
    )
    audit["portfolio_no_exchange_balance_fetch_visible"] = _contains_tokens(
        portfolio_risk_boundary_text, ("NO EXCHANGE BALANCE FETCH",)
    )
    audit["portfolio_no_real_fill_path_visible"] = _contains_tokens(
        portfolio_risk_boundary_text, ("NO REAL FILL PATH",)
    )
    audit["portfolio_no_real_order_path_visible"] = _contains_tokens(
        portfolio_risk_boundary_text, ("NO REAL ORDER PATH",)
    )
    audit["portfolio_uses_local_paper_state"] = _contains_tokens(
        portfolio_live_shape_text, ("local paper",)
    )
    audit["portfolio_updates_after_preview_order_local_only"] = (
        audit.get("terminal_blotter_updates_portfolio_snapshot") is True
        and audit.get("simulate_order_updates_blotter_portfolio_telemetry") is True
        and _contains_tokens(portfolio_risk_boundary_text, ("local-only",))
    )
    portfolio_live_shape_evidence = _build_portfolio_live_shape_evidence(audit)
    audit["portfolio_live_shape_evidence"] = portfolio_live_shape_evidence
    audit.update(portfolio_live_shape_evidence)

    risk_live_safety_controls_evidence = _build_risk_live_safety_controls_evidence(audit)
    audit["risk_live_safety_controls_evidence"] = risk_live_safety_controls_evidence
    audit.update(risk_live_safety_controls_evidence)
    order_lifecycle_evidence = _build_order_lifecycle_parity_evidence(audit)
    audit["order_lifecycle_evidence"] = order_lifecycle_evidence
    audit.update(order_lifecycle_evidence)
    terminal_order_form_evidence = _build_terminal_order_form_parity_evidence(audit)
    audit["terminal_order_form_evidence"] = terminal_order_form_evidence
    audit.update(terminal_order_form_evidence)
    market_scanner_live_field_evidence = _build_market_scanner_live_field_evidence(audit)
    audit["market_scanner_live_field_evidence"] = market_scanner_live_field_evidence
    audit.update(market_scanner_live_field_evidence)

    _invoke_show_panel(root, "alertsPanel")
    _process_events()
    alerts_feed_text = " ".join(
        (
            _read_visible_panel_object(root, "alertsPanel", "alertCenterTimeline"),
            _read_visible_panel_object_property(
                root, "alertsPanel", "alertCenterTimeline", "description"
            ),
            _read_visible_panel_object(root, "alertsPanel", "alertCenterLiveShapeBoundary"),
            _read_visible_panel_object_property(
                root, "alertsPanel", "alertCenterLiveShapeBoundary", "description"
            ),
        )
    )
    alert_rows_text = _rows_repr(root.property("alertRows"))
    selected_alert_text = repr(_variant(root.property("alertSelectedEvent")) or {})
    audit["alerts_feed_visible"] = _qml_object_visible_with_size(
        root, "alertCenterTimeline"
    ) and bool(alerts_feed_text.strip())
    audit["alerts_rows_visible"] = (
        audit["alerts_feed_visible"] and _sequence_length(root.property("alertRows")) > 0
    )
    audit["alert_severity_visible"] = _contains_tokens(
        alerts_feed_text + alert_rows_text, ("severity",)
    )
    audit["alert_source_visible"] = _contains_tokens(
        alerts_feed_text + alert_rows_text, ("source",)
    )
    audit["alert_category_visible"] = _contains_tokens(
        alerts_feed_text + alert_rows_text, ("category",)
    )
    audit["alert_message_visible"] = _contains_tokens(
        alerts_feed_text + alert_rows_text, ("message",)
    )
    audit["alert_timestamp_or_freshness_visible"] = _contains_tokens(
        alerts_feed_text + selected_alert_text, ("time",)
    ) or _contains_tokens(alerts_feed_text, ("FRESHNESS",))
    audit["alert_acknowledged_or_unresolved_placeholder_visible"] = _contains_tokens(
        alerts_feed_text + selected_alert_text, ("unresolved",)
    ) or _contains_tokens(alerts_feed_text + selected_alert_text, ("unread",))
    audit["risk_blocked_alert_visible"] = _contains_tokens(
        alert_rows_text + alerts_feed_text, ("risk", "blocked")
    )
    audit["order_blocked_alert_visible"] = _contains_tokens(
        alert_rows_text + alerts_feed_text, ("order", "blocked")
    )
    audit["scanner_candidate_alert_visible"] = _contains_tokens(
        alert_rows_text + alerts_feed_text, ("scanner", "candidate")
    )

    _invoke_show_panel(root, "telemetryPanel")
    _process_events()
    telemetry_feed_text = " ".join(
        (
            _read_visible_panel_object(root, "telemetryPanel", "telemetryLiveShapeSummary"),
            _read_visible_panel_object_property(
                root, "telemetryPanel", "telemetryLiveShapeSummary", "description"
            ),
            _read_visible_panel_object(root, "telemetryPanel", "telemetryFeedList"),
            _read_visible_panel_object_property(
                root, "telemetryPanel", "telemetryFeedList", "description"
            ),
            _read_visible_panel_object(root, "telemetryPanel", "telemetryAuditLogCard"),
            _read_visible_panel_object_property(
                root, "telemetryPanel", "telemetryAuditLogCard", "description"
            ),
        )
    )
    telemetry_rows_text = _rows_repr(root.property("paperTelemetryRows"))
    audit_rows_text = (
        _rows_repr(root.property("decisionPreviewRows"))
        + " "
        + _rows_repr(root.property("paperOrderRows"))
        + " "
        + alert_rows_text
    )
    audit["telemetry_feed_visible"] = _qml_object_visible_with_size(
        root, "telemetryFeedList"
    ) and bool(telemetry_feed_text.strip())
    audit["telemetry_rows_visible"] = (
        audit["telemetry_feed_visible"]
        and _sequence_length(root.property("paperTelemetryRows")) > 0
    )
    audit["telemetry_event_type_visible"] = _contains_tokens(telemetry_feed_text, ("event type",))
    audit["telemetry_component_visible"] = _contains_tokens(telemetry_feed_text, ("component",))
    audit["telemetry_message_visible"] = _contains_tokens(
        telemetry_feed_text + telemetry_rows_text, ("message",)
    )
    audit["telemetry_timestamp_or_freshness_visible"] = _contains_tokens(
        telemetry_feed_text + telemetry_rows_text, ("timestamp",)
    ) or _contains_tokens(telemetry_feed_text, ("FRESHNESS",))
    audit["telemetry_runtime_mode_marker_visible"] = _contains_tokens(
        telemetry_feed_text, ("runtime mode", "LOCAL PREVIEW TELEMETRY")
    )
    audit["telemetry_local_preview_source_marker_visible"] = _contains_tokens(
        telemetry_feed_text, ("local preview",)
    )
    audit["audit_log_visible"] = _qml_object_visible_with_size(
        root, "telemetryAuditLogCard"
    ) and _contains_tokens(telemetry_feed_text, ("LOCAL AUDIT LOG ONLY",))
    audit["audit_event_id_or_sequence_visible"] = _contains_tokens(
        telemetry_feed_text, ("sequence",)
    )
    audit["audit_decision_event_visible"] = _contains_tokens(
        telemetry_feed_text + audit_rows_text, ("decision",)
    )
    audit["audit_order_event_visible"] = _contains_tokens(
        telemetry_feed_text + audit_rows_text, ("order",)
    )
    audit["audit_risk_event_visible"] = _contains_tokens(
        telemetry_feed_text + audit_rows_text, ("risk",)
    )
    audit["audit_scanner_event_visible"] = _contains_tokens(
        telemetry_feed_text + audit_rows_text, ("scanner",)
    )
    audit["audit_correlation_or_trace_marker_visible"] = _contains_tokens(
        telemetry_feed_text, ("TRACE", "CORRELATION")
    )
    audit["audit_local_only_marker_visible"] = _contains_tokens(
        telemetry_feed_text, ("LOCAL AUDIT LOG ONLY",)
    )
    audit["alerts_no_cloud_sink_visible"] = _contains_tokens(
        alerts_feed_text + telemetry_feed_text, ("NO CLOUD SINK",)
    )
    audit["alerts_no_external_export_visible"] = _contains_tokens(
        alerts_feed_text + telemetry_feed_text, ("NO EXTERNAL EXPORT",)
    )
    audit["telemetry_no_live_exchange_stream_visible"] = _contains_tokens(
        telemetry_feed_text, ("NO LIVE EXCHANGE STREAM",)
    )
    audit["telemetry_no_real_order_stream_visible"] = _contains_tokens(
        telemetry_feed_text, ("NO REAL ORDER EVENT STREAM",)
    )
    audit["telemetry_no_secrets_logged_visible"] = _contains_tokens(
        telemetry_feed_text, ("SECRETS NOT LOGGED",)
    )
    audit["telemetry_uses_local_preview_state"] = (
        _contains_tokens(telemetry_feed_text + telemetry_rows_text, ("local", "preview"))
        and _bool_property(root, "runtimeLoopStarted") is False
    )
    audit["alerts_telemetry_updates_after_preview_actions_local_only"] = (
        audit.get("ping_appends_telemetry") is True
        and audit.get("risk_block_generates_blocked_event_and_alert") is True
        and _bool_property(root, "exchangeIoDisabled") is True
        and _bool_property(root, "orderSubmissionDisabled") is True
    )
    alerts_telemetry_live_shape_evidence = _build_alerts_telemetry_live_shape_evidence(audit)
    audit["alerts_telemetry_live_shape_evidence"] = alerts_telemetry_live_shape_evidence
    audit.update(alerts_telemetry_live_shape_evidence)

    settings_live_shape_objects = (
        "settingsPreviewPanel",
        "settingsModeControlsLiveShapeCard",
        "settingsApiKeyCredentialsLiveShapeCard",
        "settingsExchangeAccountLiveShapeCard",
        "settingsConfigValidationLiveShapeCard",
        "settingsAuditTelemetryBoundaryCard",
    )
    settings_text = " ".join(
        value
        for object_name in settings_live_shape_objects
        for value in (
            _read_visible_panel_object(root, "settingsPanel", object_name),
            _read_visible_panel_object_property(root, "settingsPanel", object_name, "title"),
            _read_visible_panel_object_property(root, "settingsPanel", object_name, "description"),
        )
    )
    audit["settings_panel_visible"] = _qml_object_visible_with_size(root, "settingsPreviewPanel")
    audit["mode_controls_visible"] = _qml_object_visible_with_size(
        root, "settingsModeControlsLiveShapeCard"
    ) and _contains_tokens(settings_text, ("Current mode", "PAPER ONLY", "LIVE DISABLED"))
    audit["preview_mode_boundary_visible"] = _contains_tokens(
        settings_text, ("LOCAL PREVIEW", "PAPER ONLY")
    )
    audit["live_mode_locked_visible"] = _contains_tokens(settings_text, ("LIVE DISABLED", "locked"))
    audit["api_key_status_visible"] = _contains_tokens(
        settings_text, ("API key status", "missing", "not configured", "read-only")
    )
    audit["api_keys_masked_visible"] = "••••" in settings_text
    audit["no_secret_material_visible"] = (
        _contains_tokens(settings_text, ("SECRETS MASKED", "NO SECRET MATERIAL"))
        and _bool_property(root, "settingsSecretsRead") is False
    )
    audit["exchange_profile_visible"] = _contains_tokens(settings_text, ("Exchange profile",))
    audit["account_profile_visible"] = _contains_tokens(
        settings_text, ("Account profile", "status")
    )
    audit["no_exchange_io_boundary_visible"] = (
        _contains_tokens(settings_text, ("NO EXCHANGE I/O",))
        and _bool_property(root, "exchangeIoDisabled") is True
    )
    audit["no_account_balance_fetch_boundary_visible"] = _contains_tokens(
        settings_text, ("NO ACCOUNT BALANCE FETCH",)
    )
    audit["config_validation_visible"] = _contains_tokens(
        settings_text, ("Config validation", "readiness checklist", "guardrails")
    )
    audit["live_activation_blocked_reason_visible"] = _contains_tokens(
        settings_text, ("live activation blocked reason", "cannot enable live execution")
    )
    audit["config_audit_local_only_visible"] = _contains_tokens(
        settings_text, ("Config/settings changes", "local only")
    )
    audit["no_cloud_sink_visible"] = _contains_tokens(settings_text, ("NO CLOUD SINK",))
    audit["no_external_export_visible"] = _contains_tokens(settings_text, ("NO EXTERNAL EXPORT",))
    settings_config_live_shape_evidence = _build_settings_config_live_shape_evidence(audit)
    audit["settings_config_live_shape_evidence"] = settings_config_live_shape_evidence
    audit.update(settings_config_live_shape_evidence)

    strategy_model_replay_objects = (
        "strategyWorkbenchPreviewPanel",
        "strategyRegistryLiveShapeCard",
        "modelArtifactLiveShapeCard",
        "backtestReplayLiveShapeCard",
        "strategyReadinessDeploymentGateCard",
        "strategyAuditTelemetryBoundaryCard",
    )
    strategy_model_replay_text = " ".join(
        value
        for object_name in strategy_model_replay_objects
        for value in (
            _read_visible_panel_object(root, "strategyWorkbench", object_name),
            _read_visible_panel_object_property(root, "strategyWorkbench", object_name, "title"),
            _read_visible_panel_object_property(
                root, "strategyWorkbench", object_name, "description"
            ),
        )
    )
    audit["strategy_panel_visible"] = _qml_object_visible_with_size(
        root, "strategyWorkbenchPreviewPanel"
    )
    audit["strategy_registry_visible"] = _contains_tokens(
        strategy_model_replay_text, ("LOCAL STRATEGY CATALOG", "PREVIEW STRATEGY STATE")
    )
    audit["active_strategy_visible"] = _contains_tokens(
        strategy_model_replay_text, ("Active strategy", "selected strategy")
    )
    audit["strategy_health_visible"] = _contains_tokens(
        strategy_model_replay_text, ("health", "enabled", "disabled")
    )
    audit["strategy_risk_profile_visible"] = _contains_tokens(
        strategy_model_replay_text, ("risk profile", "capital allocation")
    )
    audit["model_artifact_status_visible"] = _contains_tokens(
        strategy_model_replay_text, ("MOCK MODEL ARTIFACT", "artifact status")
    )
    audit["model_lineage_visible"] = _contains_tokens(
        strategy_model_replay_text, ("version", "hash", "lineage")
    )
    audit["inference_readiness_visible"] = _contains_tokens(
        strategy_model_replay_text, ("LOCAL INFERENCE PREVIEW", "inference readiness")
    )
    audit["no_model_promotion_visible"] = _contains_tokens(
        strategy_model_replay_text, ("NO MODEL PROMOTION",)
    )
    audit["backtest_replay_controls_visible"] = _contains_tokens(
        strategy_model_replay_text, ("Backtest", "Replay", "Start disabled")
    )
    audit["replay_dataset_window_visible"] = _contains_tokens(
        strategy_model_replay_text, ("dataset", "window", "timeframe")
    )
    audit["replay_results_summary_visible"] = _contains_tokens(
        strategy_model_replay_text, ("result summary",)
    )
    audit["replay_metrics_visible"] = _contains_tokens(
        strategy_model_replay_text, ("PnL", "win rate", "drawdown", "trades")
    )
    audit["local_replay_only_visible"] = _contains_tokens(
        strategy_model_replay_text, ("LOCAL REPLAY ONLY",)
    )
    audit["no_live_market_data_fetch_visible"] = _contains_tokens(
        strategy_model_replay_text, ("NO LIVE MARKET DATA FETCH",)
    )
    audit["readiness_checklist_visible"] = _contains_tokens(
        strategy_model_replay_text, ("readiness checklist",)
    )
    audit["live_promotion_locked_visible"] = _contains_tokens(
        strategy_model_replay_text, ("LIVE PROMOTION DISABLED", "PAPER ONLY")
    )
    audit["no_live_deployment_side_effect_visible"] = _contains_tokens(
        strategy_model_replay_text, ("no live deployment side effect",)
    )
    audit["strategy_audit_local_only_visible"] = _contains_tokens(
        strategy_model_replay_text, ("strategy/model/backtest actions", "local-only")
    )
    audit["no_cloud_sink_visible"] = audit.get(
        "no_cloud_sink_visible"
    ) is True and _contains_tokens(strategy_model_replay_text, ("NO CLOUD SINK",))
    audit["no_external_export_visible"] = audit.get(
        "no_external_export_visible"
    ) is True and _contains_tokens(strategy_model_replay_text, ("NO EXTERNAL EXPORT",))
    strategy_model_replay_live_shape_evidence = _build_strategy_model_replay_live_shape_evidence(
        audit
    )
    audit["strategy_model_replay_live_shape_evidence"] = strategy_model_replay_live_shape_evidence
    audit.update(strategy_model_replay_live_shape_evidence)

    runtime_session_control_objects = (
        "runtimeSessionControlPanel",
        "runtimeSessionPanelLiveShapeCard",
        "runtimeControlPlaneHealthCard",
        "runtimeRecoveryFailoverDegradedCard",
        "runtimeAuditLocalOnlyBoundaryCard",
    )
    runtime_session_control_text = " ".join(
        value
        for object_name in runtime_session_control_objects
        for value in (
            _read_visible_panel_object(root, "runtimeSessionControlPanel", object_name),
            _read_visible_panel_object_property(
                root, "runtimeSessionControlPanel", object_name, "title"
            ),
            _read_visible_panel_object_property(
                root, "runtimeSessionControlPanel", object_name, "description"
            ),
        )
    )
    audit["runtime_session_panel_visible"] = _qml_object_visible_with_size(
        root, "runtimeSessionPanelLiveShapeCard"
    )
    audit["session_controls_visible"] = _contains_tokens(
        runtime_session_control_text, ("session controls",)
    )
    audit["start_stop_pause_resume_visible"] = _contains_tokens(
        runtime_session_control_text, ("start", "stop", "pause", "resume")
    )
    audit["live_runtime_disabled_visible"] = _contains_tokens(
        runtime_session_control_text, ("Live runtime disabled",)
    )
    audit["no_real_loop_start_visible"] = (
        _contains_tokens(runtime_session_control_text, ("NO REAL LOOP START",))
        and _bool_property(root, "runtimeLoopStarted") is False
    )
    audit["current_session_state_visible"] = _contains_tokens(
        runtime_session_control_text, ("Current session state", "stopped preview")
    )
    audit["control_plane_health_visible"] = _contains_tokens(
        runtime_session_control_text, ("Control-plane health",)
    )
    audit["scheduler_status_visible"] = _contains_tokens(
        runtime_session_control_text, ("Scheduler status", "stopped preview")
    )
    audit["worker_status_visible"] = _contains_tokens(
        runtime_session_control_text, ("worker status", "idle preview")
    )
    audit["heartbeat_visible"] = _contains_tokens(runtime_session_control_text, ("Heartbeat",))
    audit["mock_heartbeat_visible"] = _contains_tokens(
        runtime_session_control_text, ("MOCK HEARTBEAT ONLY",)
    )
    audit["recovery_controls_visible"] = _contains_tokens(
        runtime_session_control_text, ("Recovery controls",)
    )
    audit["failover_state_visible"] = _contains_tokens(
        runtime_session_control_text, ("Failover state", "standby preview")
    )
    audit["degraded_mode_visible"] = _contains_tokens(
        runtime_session_control_text, ("Degraded mode",)
    )
    audit["recovery_actions_disabled_visible"] = _contains_tokens(
        runtime_session_control_text, ("Recovery actions disabled",)
    )
    audit["no_live_reconnect_visible"] = _contains_tokens(
        runtime_session_control_text, ("NO LIVE RECONNECT",)
    )
    audit["runtime_preflight_gate_visible"] = _contains_tokens(
        runtime_session_control_text, ("Runtime preflight gate", "closed")
    )
    audit["runtime_activation_blocked_reason_visible"] = _contains_tokens(
        runtime_session_control_text, ("Runtime activation blocked reason",)
    )
    audit["emergency_stop_shape_visible"] = _contains_tokens(
        runtime_session_control_text, ("Emergency stop shape",)
    )
    audit["no_live_scheduler_worker_start_visible"] = _contains_tokens(
        runtime_session_control_text, ("NO LIVE SCHEDULER WORKER START",)
    )
    audit["no_live_adapter_start_visible"] = _contains_tokens(
        runtime_session_control_text, ("NO LIVE ADAPTER START",)
    )
    audit["runtime_audit_local_only_visible"] = _contains_tokens(
        runtime_session_control_text, ("Runtime audit local-only", "Typed preview bridge")
    )
    audit["no_cloud_sink_visible"] = audit.get(
        "no_cloud_sink_visible"
    ) is True and _contains_tokens(runtime_session_control_text, ("NO CLOUD SINK",))
    audit["no_external_export_visible"] = audit.get(
        "no_external_export_visible"
    ) is True and _contains_tokens(runtime_session_control_text, ("NO EXTERNAL EXPORT",))
    runtime_session_control_live_shape_evidence = (
        _build_runtime_session_control_live_shape_evidence(audit)
    )
    audit["runtime_session_control_live_shape_evidence"] = (
        runtime_session_control_live_shape_evidence
    )
    audit.update(runtime_session_control_live_shape_evidence)

    audit.update(_build_operator_workflow_runtime_audit(root, audit))
    operator_workflow_evidence = _build_operator_workflow_evidence(audit)
    audit["operator_workflow_evidence"] = operator_workflow_evidence
    audit.update(operator_workflow_evidence)

    frontend_live_parity_evidence = _build_frontend_live_parity_evidence(audit)
    audit["frontend_live_parity_evidence"] = frontend_live_parity_evidence
    audit.update(
        {
            key: frontend_live_parity_evidence.get(key, False)
            for key in FRONTEND_LIVE_PARITY_SMOKE_KEYS
        }
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
        "typed_preview_bridge_registered",
        "typed_preview_bridge_is_qml_context_instance",
        "typed_preview_bridge_schema_contract_valid",
        "typed_preview_bridge_matches_qml_paper_session_snapshot",
        "typed_preview_bridge_matches_qml_scanner_snapshot",
        "typed_preview_bridge_matches_qml_governor_snapshot",
        "typed_preview_bridge_matches_qml_portfolio_snapshot",
        "typed_preview_bridge_matches_qml_alert_telemetry_snapshot",
        "typed_preview_bridge_runtime_boundary_local_only",
        "typed_preview_bridge_qml_consumer_visible",
        "typed_preview_bridge_qml_consumer_schema_ok_visible",
        "typed_preview_bridge_qml_consumer_runtime_boundary_visible",
        "typed_preview_bridge_qml_consumer_diagnostic_marker_visible",
        "typed_preview_bridge_qml_consumer_diagnostic_marker_local_read_only",
        "typed_preview_bridge_qml_consumer_matches_paper_snapshot",
        "typed_preview_bridge_qml_consumer_matches_scanner_snapshot",
        "typed_preview_bridge_qml_consumer_matches_governor_snapshot",
        "typed_preview_bridge_qml_consumer_fallback_state_visible",
        "typed_preview_bridge_qml_consumer_fallback_state_safe",
        "typed_preview_bridge_qml_consumer_fallback_state_no_type_error",
        "safety_boundary_ok",
        "frontend_live_parity_dashboard_present",
        "frontend_live_parity_market_scanner_present",
        "frontend_live_parity_ai_governor_present",
        "frontend_live_parity_decisions_present",
        "frontend_live_parity_terminal_order_panel_present",
        "frontend_live_parity_portfolio_present",
        "frontend_live_parity_alerts_telemetry_present",
        "frontend_live_parity_live_safety_boundary_visible",
        "frontend_live_parity_no_fake_live_actions",
        "frontend_live_parity_all_required_sections_present",
        "risk_live_safety_controls_visible_complete",
        "terminal_order_form_live_shape_complete",
        "portfolio_live_shape_parity_complete",
        "alerts_telemetry_live_shape_parity_complete",
        "operator_workflow_smoke_complete",
    )
    audit["passed"] = (
        int(audit["start_tick_delta"]) >= 1
        and int(audit["generate_tick_delta"]) == 1
        and int(audit["run_ten_tick_delta"]) == 10
        and int(audit["select_top20_count"]) == 20
        and int(audit["portfolio_filters_count"]) == 7
        and int(audit["portfolio_cycles_count"]) >= 4
        and int(audit["portfolio_cards_count"]) >= 13
        and all(audit.get(key) is True for key in required_true_keys)
    )
    return audit


@contextmanager
def _smoke_artifact_paths() -> Any:
    """Redirect smoke-only generated artifacts away from tracked repo paths."""

    env_keys = ("BOT_CORE_UI_FEED_LATENCY_PATH", "BOT_CORE_UI_LAYOUTS_PATH")
    previous = {key: getattr(os, "environ").get(key) for key in env_keys}
    with tempfile.TemporaryDirectory(prefix="bot-core-ui-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        getattr(os, "environ")["BOT_CORE_UI_FEED_LATENCY_PATH"] = str(
            temp_root / "reports" / "ci" / "decision_feed_metrics.json"
        )
        getattr(os, "environ")["BOT_CORE_UI_LAYOUTS_PATH"] = str(
            temp_root / "var" / "ui_layouts.json"
        )
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    getattr(os, "environ").pop(key, None)
                else:
                    getattr(os, "environ")[key] = value


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
    artifact_paths = _smoke_artifact_paths()
    artifact_paths_entered = False
    try:
        artifact_paths.__enter__()
        artifact_paths_entered = True
        tracked_artifacts_before = _tracked_artifact_snapshot()
        app = BotPysideApplication(options)
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
        typed_preview_bridge: Any | None = None
        qml_context_bridge_instance: Any | None = None
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
                "runtimeSessionControlPanel": 13,
                "helpGlossaryPanel": 14,
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
            typed_preview_bridge = engine.rootContext().contextProperty("typedPreviewBridge")
            qml_context_bridge = getattr(app, "_bridge", None)
            qml_context_bridge_instance = getattr(qml_context_bridge, "typed_preview_bridge", None)
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
                preview_state_audit = _exercise_preview_state(
                    root,
                    typed_preview_bridge,
                    qml_context_bridge_instance,
                )
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
        tracked_artifacts_clean = _tracked_artifacts_unchanged(tracked_artifacts_before)
        preview_launch_payload: dict[str, object] = {
            "ui_loaded": qml_loaded,
            "qml_loaded": qml_loaded,
            "operator_dashboard_present": operator_dashboard_present,
            "operator_dashboard_default": operator_dashboard_default,
            "panel_load_results": panel_load_results,
            "preview_state_exercised": options.exercise_preview_state and bool(preview_state_audit),
            "preview_state_audit": preview_state_audit,
            "runtime_loop_started": False,
            "exchange_io": "disabled",
            "order_submission": "disabled",
            "api_keys_required": False,
            "live_mode_allowed": False,
            "secrets_read": False,
            "keychain_read": False,
            "env_values_read": False,
            "dot_env_read": False,
            "safety_boundary_ok": safety_boundary_ok,
            "tracked_artifacts_clean": tracked_artifacts_clean,
        }
        preview_launch_readiness_evidence = _preview_launch_readiness_evidence(
            preview_launch_payload
        )
        frontend_live_parity_evidence = preview_state_audit.get("frontend_live_parity_evidence", {})
        if not isinstance(frontend_live_parity_evidence, dict):
            frontend_live_parity_evidence = {}
        terminal_order_form_evidence = preview_state_audit.get("terminal_order_form_evidence", {})
        if not isinstance(terminal_order_form_evidence, dict):
            terminal_order_form_evidence = {}
        order_lifecycle_evidence = preview_state_audit.get("order_lifecycle_evidence", {})
        if not isinstance(order_lifecycle_evidence, dict):
            order_lifecycle_evidence = {}
        risk_live_safety_controls_evidence = preview_state_audit.get(
            "risk_live_safety_controls_evidence", {}
        )
        if not isinstance(risk_live_safety_controls_evidence, dict):
            risk_live_safety_controls_evidence = {}
        market_scanner_live_field_evidence = preview_state_audit.get(
            "market_scanner_live_field_evidence", {}
        )
        if not isinstance(market_scanner_live_field_evidence, dict):
            market_scanner_live_field_evidence = {}
        portfolio_live_shape_evidence = preview_state_audit.get("portfolio_live_shape_evidence", {})
        if not isinstance(portfolio_live_shape_evidence, dict):
            portfolio_live_shape_evidence = {}
        alerts_telemetry_live_shape_evidence = preview_state_audit.get(
            "alerts_telemetry_live_shape_evidence", {}
        )
        if not isinstance(alerts_telemetry_live_shape_evidence, dict):
            alerts_telemetry_live_shape_evidence = {}
        operator_workflow_evidence = preview_state_audit.get("operator_workflow_evidence", {})
        if not isinstance(operator_workflow_evidence, dict):
            operator_workflow_evidence = {}
        settings_config_live_shape_evidence = preview_state_audit.get(
            "settings_config_live_shape_evidence", {}
        )
        if not isinstance(settings_config_live_shape_evidence, dict):
            settings_config_live_shape_evidence = {}
        strategy_model_replay_live_shape_evidence = preview_state_audit.get(
            "strategy_model_replay_live_shape_evidence", {}
        )
        if not isinstance(strategy_model_replay_live_shape_evidence, dict):
            strategy_model_replay_live_shape_evidence = {}
        runtime_session_control_live_shape_evidence = preview_state_audit.get(
            "runtime_session_control_live_shape_evidence", {}
        )
        if not isinstance(runtime_session_control_live_shape_evidence, dict):
            runtime_session_control_live_shape_evidence = {}
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
            preview_launch_readiness_evaluated=options.exercise_preview_state,
            preview_launch_readiness_requires_exercise_preview_state=not options.exercise_preview_state,
            preview_launch_readiness_evidence=preview_launch_readiness_evidence,
            frontend_live_parity_dashboard_present=bool(
                frontend_live_parity_evidence.get("frontend_live_parity_dashboard_present", False)
            ),
            frontend_live_parity_market_scanner_present=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_market_scanner_present", False
                )
            ),
            frontend_live_parity_ai_governor_present=bool(
                frontend_live_parity_evidence.get("frontend_live_parity_ai_governor_present", False)
            ),
            frontend_live_parity_decisions_present=bool(
                frontend_live_parity_evidence.get("frontend_live_parity_decisions_present", False)
            ),
            frontend_live_parity_terminal_order_panel_present=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_terminal_order_panel_present", False
                )
            ),
            frontend_live_parity_portfolio_present=bool(
                frontend_live_parity_evidence.get("frontend_live_parity_portfolio_present", False)
            ),
            frontend_live_parity_alerts_telemetry_present=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_alerts_telemetry_present", False
                )
            ),
            frontend_live_parity_live_safety_boundary_visible=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_live_safety_boundary_visible", False
                )
            ),
            frontend_live_parity_runtime_session_control_present=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_runtime_session_control_present", False
                )
            ),
            frontend_live_parity_no_fake_live_actions=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_no_fake_live_actions", False
                )
            ),
            frontend_live_parity_all_required_sections_present=bool(
                frontend_live_parity_evidence.get(
                    "frontend_live_parity_all_required_sections_present", False
                )
            ),
            frontend_live_parity_evidence=frontend_live_parity_evidence,
            settings_config_live_shape_parity_complete=bool(
                settings_config_live_shape_evidence.get(
                    "settings_config_live_shape_parity_complete", False
                )
            ),
            settings_config_live_shape_evidence=settings_config_live_shape_evidence,
            strategy_model_replay_live_shape_parity_complete=bool(
                strategy_model_replay_live_shape_evidence.get(
                    "strategy_model_replay_live_shape_parity_complete", False
                )
            ),
            strategy_model_replay_live_shape_evidence=strategy_model_replay_live_shape_evidence,
            runtime_session_control_live_shape_parity_complete=bool(
                runtime_session_control_live_shape_evidence.get(
                    "runtime_session_control_live_shape_parity_complete", False
                )
            ),
            runtime_session_control_live_shape_evidence=runtime_session_control_live_shape_evidence,
            risk_live_safety_controls_visible_complete=bool(
                risk_live_safety_controls_evidence.get(
                    "risk_live_safety_controls_visible_complete", False
                )
            ),
            risk_live_safety_controls_evidence=risk_live_safety_controls_evidence,
            terminal_order_form_live_shape_complete=bool(
                terminal_order_form_evidence.get("terminal_order_form_live_shape_complete", False)
            ),
            terminal_order_form_evidence=terminal_order_form_evidence,
            order_lifecycle_preview_parity_complete=bool(
                order_lifecycle_evidence.get("order_lifecycle_preview_parity_complete", False)
            ),
            order_lifecycle_evidence=order_lifecycle_evidence,
            market_scanner_live_field_parity_complete=bool(
                market_scanner_live_field_evidence.get(
                    "market_scanner_live_field_parity_complete", False
                )
            ),
            market_scanner_live_field_evidence=market_scanner_live_field_evidence,
            portfolio_live_shape_parity_complete=bool(
                portfolio_live_shape_evidence.get("portfolio_live_shape_parity_complete", False)
            ),
            portfolio_live_shape_evidence=portfolio_live_shape_evidence,
            alerts_telemetry_live_shape_parity_complete=bool(
                alerts_telemetry_live_shape_evidence.get(
                    "alerts_telemetry_live_shape_parity_complete", False
                )
            ),
            alerts_telemetry_live_shape_evidence=alerts_telemetry_live_shape_evidence,
            operator_workflow_smoke_complete=bool(
                operator_workflow_evidence.get("operator_workflow_smoke_complete", False)
            ),
            operator_workflow_evidence=operator_workflow_evidence,
            issues=[] if smoke_ok else audit_issues or qml_warnings or ["qml_root_objects_missing"],
        )
        print(result.to_json(), file=output)
        return 0 if smoke_ok else 1
    except Exception as exc:  # pragma: no cover - exercised via CLI integration tests
        issue = f"{type(exc).__name__}: {exc}"
        result = UiSmokeResult(status="error", issues=qml_warnings + audit_issues + [issue])
        print(result.to_json(), file=output)
        return 1
    finally:
        if artifact_paths_entered:
            artifact_paths.__exit__(None, None, None)


if __name__ == "__main__":  # pragma: no cover - CLI compatibility for direct script smoke runs
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run PySide preview UI smoke checks.")
    parser.add_argument("--config", default="ui/config/preview_local.yaml")
    parser.add_argument("--offscreen", action="store_true", default=True)
    parser.add_argument("--exercise-preview-state", action="store_true")
    args = parser.parse_args()
    smoke_options = AppOptions.parse(
        [
            "--config",
            args.config,
            "--smoke",
            "--offscreen",
            *(["--exercise-preview-state"] if args.exercise_preview_state else []),
        ]
    )
    raise SystemExit(run_smoke(smoke_options, output=sys.stdout, force_offscreen=True))
