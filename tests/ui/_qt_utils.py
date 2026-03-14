from __future__ import annotations

import time
from typing import Callable, TypeVar

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtTest import QTest  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - brak Qt
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
