from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def allow_long_poll_in_exchange_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Włącz long-poll dla testów adapterów giełdowych (LocalLongPollStream soft-close w test mode)."""

    monkeypatch.setenv("DUDZIAN_ALLOW_LONG_POLL", "1")
