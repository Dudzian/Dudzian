"""Shared PySide UI test hygiene fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def ui_generated_artifacts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep UI-generated diagnostics/layout artifacts out of tracked repo paths."""

    monkeypatch.setenv(
        "BOT_CORE_UI_FEED_LATENCY_PATH",
        str(tmp_path / "reports" / "ci" / "decision_feed_metrics.json"),
    )
    monkeypatch.setenv("BOT_CORE_UI_LAYOUTS_PATH", str(tmp_path / "var" / "ui_layouts.json"))

@pytest.fixture(scope="session")
def qt_gui_application():
    """Keep the native Qt application alive through the complete QML test session."""
    from PySide6.QtGui import QGuiApplication

    from tests._qt_utils import settle_qt_application

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication([])

    yield app
    # QML engines have per-test finalizers; drain residual native Qt work before
    # Python starts unloading shiboken/PySide6 modules at interpreter exit.
    settle_qt_application(process_events=app.processEvents)
