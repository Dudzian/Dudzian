"""Safe source-level smoke check for the PySide6/QML UI."""

from __future__ import annotations

import json
import os
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
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Render the smoke contract as deterministic CP1252-safe JSON."""

        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


PANEL_AUDIT_IDS = (
    "sidePanel",
    "aiCenterPanel",
    "tradingUniversePanel",
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
        if qml_loaded:
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
            issues=[] if smoke_ok else audit_issues or qml_warnings or ["qml_root_objects_missing"],
        )
        print(result.to_json(), file=output)
        return 0 if smoke_ok else 1
    except Exception as exc:  # pragma: no cover - exercised via CLI integration tests
        issue = f"{type(exc).__name__}: {exc}"
        result = UiSmokeResult(status="error", issues=qml_warnings + audit_issues + [issue])
        print(result.to_json(), file=output)
        return 1
