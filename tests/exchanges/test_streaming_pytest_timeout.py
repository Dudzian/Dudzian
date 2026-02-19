from __future__ import annotations

from urllib.error import URLError

import pytest

from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.streaming import LocalLongPollStream


def test_local_long_poll_stream_clamps_timeout_when_pytest_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[float] = []

    def fake_urlopen(request, timeout=0.0):  # noqa: D401
        observed_timeouts.append(float(timeout))
        raise URLError("simulated network error")

    monkeypatch.delenv("DUDZIAN_TEST_MODE", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/exchanges/test_streaming_pytest_timeout.py::test")
    monkeypatch.setattr("bot_core.exchanges.streaming.urlopen", fake_urlopen)

    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9110",
        path="/timeout-clamp",
        channels=["ticker"],
        adapter="test",
        scope="public",
        environment="paper",
        poll_interval=0.0,
        timeout=10.0,
        max_retries=1,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )

    with pytest.raises(ExchangeNetworkError):
        stream._poll_once()

    assert observed_timeouts
    assert observed_timeouts[0] <= 0.5

    stream.close()
