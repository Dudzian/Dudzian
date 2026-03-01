from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final

import pytest


_TRUTHY: Final = {"1", "true", "yes", "on"}


@pytest.fixture(autouse=True)
def allow_long_poll_in_exchange_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Włącz long-poll dla testów adapterów giełdowych (LocalLongPollStream soft-close w test mode),
    ale zawsze sprzątaj wątki po teście, żeby nie wyciekały do teardownów UI/QML.
    """
    allow_long_poll = os.getenv("DUDZIAN_ALLOW_LONG_POLL", "").strip().lower()
    if allow_long_poll not in _TRUTHY:
        monkeypatch.setenv("DUDZIAN_ALLOW_LONG_POLL", "1")

    try:
        yield
    finally:
        try:
            from bot_core.events.emitter import EventBus, EventEmitter
            from bot_core.runtime.pipeline import Pipeline
        except Exception:
            EventBus = EventEmitter = Pipeline = None

        if EventBus is not None:
            EventBus.close_all_active()
        if EventEmitter is not None:
            EventEmitter.close_all_active()
        if Pipeline is not None:
            Pipeline.close_all_active()

        try:
            from bot_core.execution.live_router import LiveExecutionRouter
        except Exception:
            LiveExecutionRouter = None

        if LiveExecutionRouter is not None:
            LiveExecutionRouter.close_all_active()

        try:
            from bot_core.exchanges.streaming import LocalLongPollStream
        except Exception:
            return

        # best-effort: nie wieszaj suite, ale daj chwilę na join.
        LocalLongPollStream.close_all_active(timeout=5.0)

        # domknij router ponownie po stream-close (rzadkie, ale możliwe referencje wtórne).
        if LiveExecutionRouter is not None:
            LiveExecutionRouter.close_all_active()
