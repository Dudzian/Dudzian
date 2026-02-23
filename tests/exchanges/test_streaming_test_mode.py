from __future__ import annotations

import threading
import time

import pytest

from bot_core.exchanges.streaming import LocalLongPollStream


def test_local_long_poll_stream_does_not_start_worker_when_disabled_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUDZIAN_TEST_MODE", "1")
    monkeypatch.setenv("DUDZIAN_ALLOW_LONG_POLL", "0")

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9110",
        path="/long-poll",
        channels=["ticker"],
        adapter="demo",
        scope="public",
        environment="paper",
        timeout=10.0,
        poll_interval=0.0,
    )

    try:
        stream.start()

        worker = stream._worker_thread
        assert worker is None or not worker.is_alive()

        time.sleep(0.05)
        leaking_workers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("LocalLongPollStream[")
        ]
        assert not leaking_workers
        assert stream.closed
    finally:
        stream.close()
