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


# Keep the Python wrapper alive after pytest releases its session fixture.
# Qt's normal process shutdown owns app teardown; a session-wide DeferredDelete
# flush here can destroy still-referenced QML objects and segfault PySide6.
_SESSION_QT_APPLICATION: object | None = None


@pytest.fixture(scope="session")
def qt_gui_application():
    """Reuse a single Qt application without global native-object destruction."""
    global _SESSION_QT_APPLICATION

    from PySide6.QtGui import QGuiApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication([])

    _SESSION_QT_APPLICATION = app
    return app
