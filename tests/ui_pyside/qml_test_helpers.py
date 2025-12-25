"""Helpers improving diagnostics when QML fails to load during tests."""
from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtQml import QQmlApplicationEngine, QQmlError


def _format_qml_error(error: QQmlError) -> str:
    """Render a single QML error or warning with location."""

    try:
        url = error.url().toString() if error.url().isValid() else "<unknown>"
        return f"{url}:{error.line()}:{error.column()}: {error.description()}"
    except Exception:
        return str(error)


def collect_engine_warnings(engine: QQmlApplicationEngine) -> list[str]:
    """Attach to the engine warnings signal and collect existing ones."""

    collected: list[str] = []

    def _on_warnings(warnings: Iterable[QQmlError]) -> None:
        for warning in warnings:
            collected.append(_format_qml_error(warning))

    try:
        engine.warnings.connect(_on_warnings)
    except Exception:
        # Environments or bindings without the signal
        pass

    try:
        existing = engine.warnings()  # type: ignore[operator]
    except Exception:
        existing = []
    for warning in existing:
        collected.append(_format_qml_error(warning))

    return collected


def collect_engine_errors(engine: QQmlApplicationEngine) -> list[str]:
    """Collect engine errors if the API is available."""

    try:
        errors: Sequence[QQmlError] = engine.errors()  # type: ignore[operator]
    except Exception:
        errors = []
    return [_format_qml_error(error) for error in errors]


def assert_engine_loaded(
    engine: QQmlApplicationEngine,
    warnings: Iterable[str] | None = None,
    message: str = "QML failed to load",
) -> None:
    """Assert that engine.rootObjects() is non-empty and surface diagnostics."""

    collected_warnings = list(warnings or [])
    try:
        existing = engine.warnings()  # type: ignore[operator]
    except Exception:
        existing = []
    for warning in existing:
        formatted = _format_qml_error(warning)
        if formatted not in collected_warnings:
            collected_warnings.append(formatted)

    collected_errors = collect_engine_errors(engine)
    if engine.rootObjects():
        return

    details: list[str] = []
    if collected_warnings:
        details.append("warnings=" + " | ".join(collected_warnings))
    if collected_errors:
        details.append("errors=" + " | ".join(collected_errors))
    detail_msg = "; ".join(details) if details else "no warnings/errors collected"
    raise AssertionError(f"{message}: {detail_msg}")
