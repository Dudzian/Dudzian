"""Safe source-level smoke check for the PySide6/QML UI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import TextIO

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
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Render the smoke contract as deterministic CP1252-safe JSON."""

        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


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

    issues: list[str] = []
    app = BotPysideApplication(options)
    try:
        engine = app.load(warning_sink=issues.append)
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
        result = UiSmokeResult(
            status="ok" if qml_loaded else "error",
            ui_loaded=qml_loaded,
            qml_loaded=qml_loaded,
            operator_dashboard_present=operator_dashboard_present,
            operator_dashboard_default=operator_dashboard_default,
            operator_dashboard_visible=operator_dashboard_visible,
            active_panel_id=active_panel_id,
            central_content_empty=central_content_empty,
            issues=[] if qml_loaded else issues or ["qml_root_objects_missing"],
        )
        print(result.to_json(), file=output)
        return 0 if qml_loaded else 1
    except Exception as exc:  # pragma: no cover - exercised via CLI integration tests
        issue = f"{type(exc).__name__}: {exc}"
        result = UiSmokeResult(status="error", issues=issues + [issue])
        print(result.to_json(), file=output)
        return 1
