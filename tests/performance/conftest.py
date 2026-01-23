import os

def pytest_sessionstart(session) -> None:
    """
    Perf testy muszą ustawić backend/offscreen *zanim* powstanie Q(Core|Gui|)Application.
    Fixture (function-scope) jest za późno, jeśli gdziekolwiek istnieje session-scope qt_app.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")

    # QML ma killswitch na QtCharts (context property: disableQtCharts) – w perf/CI domyślnie wyłączamy.
    os.environ.setdefault("DUDZIAN_DISABLE_QTCHARTS", "1")
