"""Konfiguracja wspólna dla testów QML."""
from __future__ import annotations

import io
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List

import pytest

try:  # pragma: no cover - zależne od środowiska CI
    import PySide6  # type: ignore[import-not-found]  # noqa: F401
    _PYSIDE6_AVAILABLE = True
except Exception:  # pragma: no cover - zależne od środowiska CI
    _PYSIDE6_AVAILABLE = False


logger = logging.getLogger(__name__)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if "QML_IMPORT_TRACE" not in os.environ:
    if os.getenv("QML_DIAGNOSTICS_DIR") or (
        os.getenv("CI") and os.getenv("QML_TRACE") == "1"
    ):
        os.environ["QML_IMPORT_TRACE"] = "1"
if sys.platform == "win32":
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QSG_RENDER_LOOP", "basic")


def _sanitize_nodeid(nodeid: str) -> str:
    sanitized = nodeid.replace("::", "__").replace("/", "_").replace("\\", "_")
    return sanitized




def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Pomiń import modułów z tests/ui przy braku PySide6 (unikamy collection errors)."""

    if _PYSIDE6_AVAILABLE:
        return False
    root = Path(str(config.rootpath)).resolve()
    path = Path(str(collection_path)).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if relative.suffix != ".py":
        return False
    return len(relative.parts) >= 2 and relative.parts[0] == "tests" and relative.parts[1] == "ui"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _PYSIDE6_AVAILABLE:
        return
    skip_qml = pytest.mark.skip(reason="UI/QML tests require PySide6")
    for item in items:
        item.add_marker(skip_qml)



@pytest.fixture(scope="session")
def qml_diagnostics_root(pytestconfig: pytest.Config) -> Path:
    base = os.getenv("QML_DIAGNOSTICS_DIR", "test-results/qml")
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "screenshots").mkdir(exist_ok=True)

    # Zapisz podstawowe informacje o środowisku Qt, aby ułatwić diagnostykę.
    info_path = root / "environment.txt"
    if not info_path.exists():
        lines: List[str] = []
        for name in (
            "QT_QPA_PLATFORM",
            "QT_PLUGIN_PATH",
            "QT_QUICK_BACKEND",
            "QML_IMPORT_TRACE",
            "QSG_RHI_BACKEND",
            "QT_OPENGL",
            "QSG_RENDER_LOOP",
        ):
            lines.append(f"{name}={os.getenv(name, '')}")
        try:
            import PySide6
            from PySide6 import QtCore
            from PySide6.QtCore import QLibraryInfo
        except Exception as exc:  # pragma: no cover - zależne od środowiska CI
            lines.append(f"PySide6: import failed -> {exc}")
        else:  # pragma: no cover - tylko logowanie metadanych
            lines.append(f"PySide6: {getattr(PySide6, '__version__', 'unknown')}")
            lines.append(f"Qt wersja: {QtCore.qVersion()}")
            try:  # pragma: no cover - zależne od wersji Qt
                for path_id in QLibraryInfo.LibraryPath:  # type: ignore[attr-defined]
                    try:
                        path_value = QLibraryInfo.path(path_id)
                    except Exception as path_exc:  # pragma: no cover - tylko logowanie
                        lines.append(f"QLibraryInfo[{getattr(path_id, 'name', path_id)}]: {path_exc}")
                    else:
                        lines.append(
                            f"QLibraryInfo[{getattr(path_id, 'name', path_id)}]={path_value}"
                        )
            except Exception as paths_exc:  # pragma: no cover - kompatybilność Qt
                lines.append(f"QLibraryInfo: {paths_exc}")
        info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    freeze_path = root / "pip-freeze.txt"
    if not freeze_path.exists():
        freeze_result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        output = freeze_result.stdout
        if freeze_result.returncode != 0:
            output += "\n" + "# pip freeze exit code " + str(freeze_result.returncode)
            if freeze_result.stderr:
                output += "\n" + freeze_result.stderr
        freeze_path.write_text(output, encoding="utf-8")

    invocation = getattr(pytestconfig, "invocation_params", None)
    command_path = root / "pytest-command.txt"
    if invocation and not command_path.exists():
        cmdline = " ".join(invocation.args)
        command_path.write_text(cmdline + "\n", encoding="utf-8")

    return root


@pytest.fixture(scope="session", autouse=True)
def shutdown_db_background_loop_at_session_end() -> Generator[None, None, None]:
    """
    Background loop DB NIE jest zamykany per-test (Windows UB / aiosqlite thread lifecycle).
    Zamykamy go raz na końcu sesji.
    """
    yield
    try:
        from bot_core.database.manager import DatabaseManager
    except Exception as exc:
        logger.debug(
            "Skipping DatabaseManager session shutdown fixture (import failed): %r", exc
        )
        return

    # deterministycznie domknij instancje zanim zatrzymasz loop/thread
    DatabaseManager.close_all_active(blocking=True, timeout=5.0)
    DatabaseManager.shutdown_background_loop(timeout=5.0)

    still_alive = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name == "DatabaseManagerBackgroundLoopThread"
    ]
    if still_alive:
        logger.debug(
            "DatabaseManager background loop thread still alive at session end; threads=%s",
            [
                (thread.name, thread.ident, thread.daemon)
                for thread in threading.enumerate()
                if thread.is_alive()
            ],
        )

    assert not still_alive, "DatabaseManager background loop thread still alive after session shutdown"


@pytest.fixture(autouse=True)
def enforce_test_mode_for_qml(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return
    previous = os.environ.get("DUDZIAN_TEST_MODE")
    previous_allow_long_poll = os.environ.get("DUDZIAN_ALLOW_LONG_POLL")
    os.environ["DUDZIAN_TEST_MODE"] = "1"
    os.environ["DUDZIAN_ALLOW_LONG_POLL"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DUDZIAN_TEST_MODE", None)
        else:
            os.environ["DUDZIAN_TEST_MODE"] = previous
        if previous_allow_long_poll is None:
            os.environ.pop("DUDZIAN_ALLOW_LONG_POLL", None)
        else:
            os.environ["DUDZIAN_ALLOW_LONG_POLL"] = previous_allow_long_poll


@pytest.fixture(autouse=True)
def shutdown_live_threads_after_qml(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return
    yield
    try:
        from bot_core.events.emitter import EventBus, EventEmitter
        from bot_core.exchanges.streaming import LocalLongPollStream
        from bot_core.execution.live_router import LiveExecutionRouter
        from bot_core.runtime.pipeline import Pipeline
        from bot_core.database.manager import DatabaseManager
    except Exception:
        return

    EventBus.close_all_active()
    EventEmitter.close_all_active()
    Pipeline.close_all_active()
    LocalLongPollStream.close_all_active()
    LiveExecutionRouter.close_all_active()
    prefixes = (
        "LocalLongPollStream[",
        "LiveExecutionRouterLoop",
        "LiveExecutionRouterWorker-",
        "EventEmitter",
        "PipelineStream",
    )
    deadline = time.monotonic() + 2.0
    active: list[threading.Thread] = []
    while True:
        active = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive() and thread.name.startswith(prefixes)
        ]
        if not active or time.monotonic() >= deadline:
            break
        time.sleep(0.05)

    # Po domknięciu komponentów jeszcze raz domknij DB (w razie late teardown).
    DatabaseManager.close_all_active(blocking=True, timeout=2.5)
    if DatabaseManager.active_instances():
        logger.debug(
            "DatabaseManager.close_all_active left instances: %s",
            [id(instance) for instance in DatabaseManager.active_instances()],
        )
    suspicious_threads = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and ("aiosqlite" in thread.name or "AnyIO" in thread.name)
    ]
    if suspicious_threads:
        logger.debug("Suspicious background threads after QML cleanup: %s", suspicious_threads)
    active = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name.startswith(prefixes)
    ]
    assert not active, f"Pozostały aktywne wątki live: {[t.name for t in active]}"
    assert not DatabaseManager.active_instances(), "Pozostały aktywne instancje DatabaseManager"


@pytest.fixture(autouse=True)
def capture_qt_messages(
    request: pytest.FixtureRequest,
    qml_diagnostics_root: Path,
) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return

    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:  # pragma: no cover - brak PySide6 w środowisku
        yield
        return

    messages: list[str] = []

    def _handler(mode, context, message):  # type: ignore[override]
        level = getattr(mode, "name", str(mode))
        file_name = getattr(context, "file", "") if context else ""
        line = getattr(context, "line", 0) if context else 0
        category = getattr(context, "category", "") if context else ""
        formatted = f"[{level}] {file_name}:{line} {category} {message}".strip()
        messages.append(formatted)

    previous_handler = qInstallMessageHandler(_handler)
    setattr(request.node, "_qml_log_messages", messages)
    try:
        yield
    finally:
        logs_dir = qml_diagnostics_root / "logs"
        log_path = logs_dir / f"{_sanitize_nodeid(request.node.nodeid)}.log"
        qInstallMessageHandler(previous_handler)
        reports = getattr(request.node, "_qml_reports", [])
        for report in reports:
            outcome_line = f"[PYTEST] when={report.when} outcome={report.outcome}"
            messages.append(outcome_line)
            if report.failed:
                longrepr = getattr(report, "longreprtext", "") or str(report.longrepr)
                messages.append("[PYTEST] failure=")
                messages.append(longrepr)
            stdout = getattr(report, "capstdout", "")
            if stdout:
                messages.append("[PYTEST] stdout=")
                messages.append(stdout)
            stderr = getattr(report, "capstderr", "")
            if stderr:
                messages.append("[PYTEST] stderr=")
                messages.append(stderr)
            log = getattr(report, "caplog", "")
            if log:
                messages.append("[PYTEST] caplog=")
                messages.append(str(log))
        if not messages:
            messages.append("Brak komunikatów Qt.")
        log_path.write_text("\n".join(messages) + "\n", encoding="utf-8")

        manifest_path = qml_diagnostics_root / "manifest.json"
        entry = {
            "nodeid": request.node.nodeid,
            "log": str(log_path.relative_to(qml_diagnostics_root)),
            "reports": [
                {
                    "when": getattr(report, "when", "call"),
                    "outcome": getattr(report, "outcome", "unknown"),
                }
                for report in reports
            ],
            "status": (reports[-1].outcome if reports else "unknown"),
        }

        try:
            manifest: list[dict[str, object]] = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else []
            )
        except json.JSONDecodeError:
            manifest = []

        manifest = [item for item in manifest if item.get("nodeid") != request.node.nodeid]
        manifest.append(entry)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


@pytest.fixture(autouse=True)
def capture_qml_artifacts(
    request: pytest.FixtureRequest,
    qml_diagnostics_root: Path,
) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return

    reports: list[pytest.TestReport] = []
    setattr(request.node, "_qml_reports", reports)

    log_stream = io.StringIO()
    runtime_logger = logging.getLogger("ui.backend.runtime_service")
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    runtime_logger.addHandler(handler)

    try:
        yield
    finally:
        runtime_logger.removeHandler(handler)
        handler.flush()
        runtime_log = log_stream.getvalue().strip()
        node_id = _sanitize_nodeid(request.node.nodeid)
        logs_dir = qml_diagnostics_root / "logs"
        screenshots_dir = qml_diagnostics_root / "screenshots"
        logs_dir.mkdir(exist_ok=True)
        screenshots_dir.mkdir(exist_ok=True)

        if runtime_log:
            log_path = logs_dir / f"{node_id}.runtime.log"
            log_path.write_text(runtime_log + "\n", encoding="utf-8")
            qt_messages = getattr(request.node, "_qml_log_messages", [])
            if isinstance(qt_messages, list):
                qt_messages.append("[PYTHON] runtime_service logger:")
                qt_messages.extend(runtime_log.splitlines())

        screenshot_count = 0
        try:
            from PySide6.QtGui import QGuiApplication
        except Exception:  # pragma: no cover - brak PySide6
            QGuiApplication = None  # type: ignore[assignment]

        if QGuiApplication is not None:  # pragma: no branch - gałąź diagnostyczna
            app = QGuiApplication.instance()
            if app is not None:
                for index, window in enumerate(QGuiApplication.topLevelWindows()):
                    try:
                        image = window.grabWindow()
                    except Exception:  # pragma: no cover - defensywne przechwycenie
                        continue
                    if image.isNull():
                        continue
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    screenshot_path = screenshots_dir / f"{node_id}_{timestamp}_{index}.png"
                    try:
                        image.save(str(screenshot_path), "PNG")
                    except Exception:  # pragma: no cover - błąd zapisu
                        continue
                    screenshot_count += 1

        if screenshot_count == 0:
            placeholder = screenshots_dir / f"{node_id}_no_window.txt"
            placeholder.write_text(
                "Brak widocznych okien do zrzutu ekranu." "\n", encoding="utf-8"
            )


@pytest.fixture(autouse=True)
def flush_qt_deletes_after_qml(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return
    yield
    if importlib.util.find_spec("PySide6") is None:
        return
    from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop
    from PySide6.QtQml import QQmlEngine

    app = QCoreApplication.instance()
    if app is None:
        return
    for _ in range(3):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents(QEventLoop.AllEvents, 50)
    # PySide6: w niektórych wersjach collectGarbage nie jest statyczne w bindingach
    # i wymaga instancji (TypeError "unbound method ... needs an argument").
    try:
        QQmlEngine.collectGarbage()
    except TypeError:
        try:
            QQmlEngine().collectGarbage()
        except Exception:
            # GC jest best-effort – nie chcemy failować teardownu przez różnice w bindingach
            pass
    import gc

    gc.collect()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Generator[None, None, None]:
    outcome = yield
    report = outcome.get_result()
    if "qml" in item.keywords:
        reports = list(getattr(item, "_qml_reports", []))
        reports.append(report)
        setattr(item, "_qml_reports", reports)
