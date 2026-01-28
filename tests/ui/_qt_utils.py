from __future__ import annotations

import time
from typing import Callable

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtTest import QTest  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - brak Qt
    QTest = None  # type: ignore[assignment]


def qt_wait(milliseconds: int, sleep_fn: Callable[[float], None] = time.sleep) -> None:
    if QTest is not None:
        QTest.qWait(milliseconds)
    else:
        sleep_fn(milliseconds / 1000)
