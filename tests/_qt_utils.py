from __future__ import annotations

import time
from typing import Callable, Iterable, TypeVar

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtCore import QCoreApplication, QEvent  # type: ignore[attr-defined]
    from PySide6.QtGui import QGuiApplication  # type: ignore[attr-defined]
    from PySide6.QtTest import QTest  # type: ignore[attr-defined]
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - brak Qt
    QCoreApplication = QEvent = QGuiApplication = QApplication = None  # type: ignore[assignment]
    QTest = None  # type: ignore[assignment]


def qt_wait(milliseconds: int, sleep_fn: Callable[[float], None] = time.sleep) -> None:
    if QTest is not None:
        QTest.qWait(milliseconds)
    else:
        sleep_fn(milliseconds / 1000)


T = TypeVar("T")


def wait_for(
    predicate: Callable[[], T | None],
    *,
    timeout_s: float,
    step_ms: int = 10,
    process_events: Callable[[], None] | None = None,
    description: str | None = None,
) -> T:
    """Czekaj na spełnienie warunku i przerwij z TimeoutError po przekroczeniu deadline."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process_events is not None:
            process_events()
        value = predicate()
        if value is not None:
            return value
        qt_wait(step_ms)

    if process_events is not None:
        process_events()
    value = predicate()
    if value is not None:
        return value

    detail = f" ({description})" if description else ""
    raise TimeoutError(f"wait_for timeout after {timeout_s:.3f}s{detail}")


def force_qt_cleanup(
    *,
    process_events: Callable[[], None] | None = None,
    process_rounds: int = 10,
) -> None:
    """Best-effort cleanup obiektów Qt pozostawionych po teście QML."""

    if QCoreApplication is None or QEvent is None:
        return
    for _ in range(max(process_rounds, 1)):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        if process_events is not None:
            process_events()

    # Ostatni przebieg pętli eventów pomaga domknąć obiekty zwolnione po stronie Python GC.
    if process_events is not None:
        process_events()


def settle_qt_application(
    *,
    process_events: Callable[[], None] | None = None,
    process_rounds: int = 8,
) -> None:
    """Best-effort domykanie globalnych zasobów Qt po teście QML."""

    if QCoreApplication is None:
        return

    app = QCoreApplication.instance()
    if app is None:
        return

    # Zamknij top-level windows, bo aktywne okna mogą utrzymywać timery/render-loop.
    gui_app = QGuiApplication.instance() if QGuiApplication is not None else None
    if gui_app is not None:
        try:
            windows = list(gui_app.topLevelWindows())
        except Exception:
            windows = []
        for window in windows:
            try:
                window.close()
            except Exception:
                pass
            try:
                window.deleteLater()
            except Exception:
                pass

    widgets_app = QApplication.instance() if QApplication is not None else None
    if widgets_app is not None:
        try:
            widgets = list(widgets_app.topLevelWidgets())
        except Exception:
            widgets = []
        for widget in widgets:
            try:
                widget.close()
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass

    force_qt_cleanup(process_events=process_events, process_rounds=process_rounds)

    quit_fn = getattr(app, "quit", None)
    if callable(quit_fn):
        try:
            quit_fn()
        except Exception:
            pass

    force_qt_cleanup(process_events=process_events, process_rounds=process_rounds)


def teardown_qml_engine(
    engine: object,
    *,
    process_events: Callable[[], None] | None = None,
    context_properties_to_clear: Iterable[str] = (),
    delete_root_objects: bool = True,
) -> None:
    """Wspólny teardown QQmlApplicationEngine dla testów UI/QML.

    `delete_root_objects=False` pozwala użyć helpera także wtedy, gdy test
    kasuje lub przepina rooty w niestandardowy sposób przed teardownem engine.
    """

    try:
        root_context = getattr(engine, "rootContext", None)
        if callable(root_context):
            context = root_context()
            if context is not None:
                for property_name in context_properties_to_clear:
                    context.setContextProperty(property_name, None)
    except Exception:
        pass

    if delete_root_objects:
        try:
            root_objects = getattr(engine, "rootObjects", None)
            if callable(root_objects):
                for root in list(root_objects() or []):
                    delete_later = getattr(root, "deleteLater", None)
                    if callable(delete_later):
                        delete_later()
        except Exception:
            pass
        # Najpierw domknij DeferredDelete rootów, żeby nie czyścić cache/GC,
        # kiedy obiekty QML nadal są "żywe, ale zaplanowane do usunięcia".
        force_qt_cleanup(process_events=process_events)

    clear_component_cache = getattr(engine, "clearComponentCache", None)
    if callable(clear_component_cache):
        clear_component_cache()

    collect_garbage = getattr(engine, "collectGarbage", None)
    if callable(collect_garbage):
        collect_garbage()

    delete_later = getattr(engine, "deleteLater", None)
    if callable(delete_later):
        delete_later()

    force_qt_cleanup(process_events=process_events)
