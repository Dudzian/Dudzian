from __future__ import annotations

import sys
from pathlib import Path

import pytest
try:
    from PySide6.QtCore import QObject, Property, QUrl, Signal, Slot
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
except ImportError as exc:  # pragma: no cover - środowisko bez GL/Qt
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from tests.ui_pyside.qml_test_helpers import assert_engine_loaded, collect_engine_warnings


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def _ensure_app() -> QGuiApplication:
    try:
        app = QGuiApplication.instance()
        if app is None:
            app = QGuiApplication([])
        return app
    except Exception as exc:  # pragma: no cover - środowisko bez backendu GL/Qt
        pytest.skip(
            f"Qt runtime unavailable on {sys.platform}: {exc}",
            allow_module_level=True,
        )


class _RuntimeStub(QObject):
    aiGovernorSnapshotChanged = Signal()

    def __init__(self, snapshot: dict[str, object]) -> None:
        super().__init__()
        self._snapshot = snapshot

    @Property("QVariantMap", notify=aiGovernorSnapshotChanged)
    def aiGovernorSnapshot(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._snapshot)

    @Slot(result="QVariantMap")
    def reloadAiGovernorSnapshot(self) -> dict[str, object]:  # type: ignore[override]
        self.aiGovernorSnapshotChanged.emit()
        return dict(self._snapshot)


class _DesignSystemStub(QObject):
    @Slot(str, result=str)
    def color(self, token: str) -> str:  # type: ignore[override]
        return {
            "border": "#3C3F44",
            "accent": "#5BC8FF",
            "gradientHeroStart": "#1f1f35",
            "gradientHeroEnd": "#0d0d14",
            "textPrimary": "#ffffff",
            "textSecondary": "#c5cad3",
        }.get(token, "#ffffff")

    @Slot(str, result=str)
    def iconGlyph(self, _token: str) -> str:  # type: ignore[override]
        return "\uf0c2"

    @Slot(result=str)
    def fontAwesomeFamily(self) -> str:  # type: ignore[override]
        return "Font Awesome 6 Free"


def test_ai_decisions_view_renders_without_live_runtime() -> None:
    _ensure_app()
    snapshot = {
        "lastDecision": {
            "mode": "scalping",
            "reason": "Niskie koszty",
            "confidence": 0.82,
            "riskScore": 0.42,
            "transactionCostBps": 8.5,
            "recommendedModes": ["scalping", "hedge"],
        },
        "history": [
            {
                "mode": "grid",
                "reason": "Wysokie koszty",
                "confidence": 0.61,
                "timestamp": "2025-01-01T12:00:00Z",
            },
            {
                "mode": "hedge",
                "reason": "Guardrail",
                "confidence": 0.73,
                "timestamp": "2025-01-01T12:01:00Z",
            },
        ],
        "telemetry": {
            "riskMetrics": {"risk_score": 0.42},
            "cycleMetrics": {"cycle_latency_p95_ms": 1320.0},
        },
    }
    runtime = _RuntimeStub(snapshot)
    design = _DesignSystemStub()

    engine = QQmlApplicationEngine()
    warnings = collect_engine_warnings(engine)
    engine.rootContext().setContextProperty("runtimeService", runtime)
    engine.rootContext().setContextProperty("designSystem", design)
    qml_path = Path("ui/pyside_app/qml/views/AiDecisionsView.qml").resolve()
    engine.load(QUrl.fromLocalFile(qml_path.as_posix()))
    assert_engine_loaded(engine, warnings, "QML view failed to load")

    view = engine.rootObjects()[0]
    app = _ensure_app()
    app.processEvents()

    assert int(view.property("timelineCount")) == len(snapshot["history"])
    assert int(view.property("recommendationCount")) == len(snapshot["lastDecision"]["recommendedModes"])
    assert view.property("currentMode").lower() == snapshot["lastDecision"]["mode"]
