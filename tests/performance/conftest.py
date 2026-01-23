from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_qt_software_for_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Perf tests: keep Qt deterministic and avoid GPU/ANGLE variability.
    monkeypatch.setenv("QT_QPA_PLATFORM", os.getenv("QT_QPA_PLATFORM", "offscreen"))
    monkeypatch.setenv("QT_QUICK_BACKEND", os.getenv("QT_QUICK_BACKEND", "software"))
    monkeypatch.setenv("QT_OPENGL", os.getenv("QT_OPENGL", "software"))
    # Qt6 / RHI safety net (no-op for older builds).
    monkeypatch.setenv("QSG_RHI_BACKEND", os.getenv("QSG_RHI_BACKEND", "software"))
