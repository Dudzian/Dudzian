from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:
    from PySide6.QtQml import QQmlEngine

# WAŻNE: te zmienne muszą być ustawione ZANIM Qt/PySide6 zainicjalizuje platform plugin / scenegraph.
# conftest z katalogu tests/performance ładuje się przed importem modułów testów w tym katalogu,
# więc jest to najbezpieczniejsze miejsce.
#
# Cel: stabilność (szczególnie na Windows runnerach) i eliminacja zależności od GPU/ANGLE/D3D.

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# W środowisku CI wymuszamy software backend (w tym headless macOS),
# a lokalnie robimy wyjątek tylko dla Windows, gdzie backend GPU jest najbardziej niestabilny.
if sys.platform == "win32" or os.environ.get("CI"):
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QSG_RENDER_LOOP", "basic")

# Wydajność testów QML: wycinamy ciężkie komponenty i animacje, bo to benchmark SLA paneli,
# a nie test QtCharts/animacji.
os.environ.setdefault("DUDZIAN_DISABLE_QTCHARTS", "1")
os.environ.setdefault("DUDZIAN_QML_DISABLE_ANIMATIONS", "1")

# Mniej szumu w logach (zostawiamy ostrzeżenia krytyczne):
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.scenegraph.*=false")


@pytest.fixture(autouse=True)
def _perf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUDZIAN_TEST_MODE", "1")
    monkeypatch.setenv("DUDZIAN_ALLOW_LONG_POLL", "0")


@pytest.fixture(scope="session")
def qml_engine() -> Generator["QQmlEngine", None, None]:
    # Importy PySide6 dopiero w środku, żeby środowisko było ustawione.
    pytest.importorskip("PySide6.QtQml")
    from PySide6.QtQml import QQmlEngine

    engine = QQmlEngine()
    try:
        yield engine
    finally:
        # Dajemy Qt szansę posprzątać zaległe obiekty/GC po stronie C++.
        engine.collectGarbage()
