"""Konfiguracja wspólna dla testów QML."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Generator, List

import pytest


def _sanitize_nodeid(nodeid: str) -> str:
    sanitized = nodeid.replace("::", "__").replace("/", "_").replace("\\", "_")
    return sanitized


@pytest.fixture(scope="session")
def qml_diagnostics_root(pytestconfig: pytest.Config) -> Path:
    base = os.getenv("QML_DIAGNOSTICS_DIR", "test-results/qml")
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)

    # Zapisz podstawowe informacje o środowisku Qt, aby ułatwić diagnostykę.
    info_path = root / "environment.txt"
    if not info_path.exists():
        lines: List[str] = []
        for name in ("QT_QPA_PLATFORM", "QT_PLUGIN_PATH", "QT_QUICK_BACKEND", "QML_IMPORT_TRACE"):
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


@pytest.fixture(autouse=True)
def configure_qt_environment(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    if "qml" not in request.node.keywords:
        yield
        return

    applied: dict[str, None] = {}
    for name, value in (
        ("QT_QPA_PLATFORM", "offscreen"),
        ("QT_QUICK_BACKEND", "software"),
        ("QML_IMPORT_TRACE", "1"),
    ):
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = None

    try:
        yield
    finally:
        for name in applied:
            os.environ.pop(name, None)


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
        log_path = qml_diagnostics_root / f"{_sanitize_nodeid(request.node.nodeid)}.log"
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Generator[None, None, None]:
    outcome = yield
    report = outcome.get_result()
    if "qml" in item.keywords:
        reports = list(getattr(item, "_qml_reports", []))
        reports.append(report)
        setattr(item, "_qml_reports", reports)
