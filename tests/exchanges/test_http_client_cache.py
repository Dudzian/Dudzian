import asyncio
from typing import List

import pytest

from bot_core.exchanges import http_client


class _DummyClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:  # noqa: D401
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_client_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_client, "_CLIENT_CACHE", {})
    monkeypatch.setattr(http_client, "_CLIENT_TTLS", {})


def _run_sync(async_fn, *args, **kwargs):
    async def _runner():
        return await async_fn(*args, **kwargs)

    return asyncio.run(_runner())


def test_client_cache_reuses_client_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    created: List[_DummyClient] = []

    def fake_factory(**kwargs):  # noqa: D401
        client = _DummyClient()
        created.append(client)
        return client

    timeline = iter([0.0, 0.1])

    monkeypatch.setattr(http_client, "get_rate_limited_client", fake_factory)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_now", lambda: next(timeline))

    base_url = "https://api.example"
    http_client.configure_client_cache_ttl(base_url, 1.0)

    first = http_client._resolve_client(base_url, 5.0)
    second = http_client._resolve_client(base_url, 5.0)

    assert first is second
    assert len(created) == 1


def test_client_cache_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    created: List[_DummyClient] = []

    def fake_factory(**kwargs):  # noqa: D401
        client = _DummyClient()
        created.append(client)
        return client

    timeline = iter([0.0, 1.0])

    monkeypatch.setattr(http_client, "get_rate_limited_client", fake_factory)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_now", lambda: next(timeline))

    base_url = "https://api.example"
    http_client.configure_client_cache_ttl(base_url, 0.5)

    first = http_client._resolve_client(base_url, 10.0)
    second = http_client._resolve_client(base_url, 10.0)

    assert first is not second
    assert len(created) == 2
    assert first.closed is True
    assert second.closed is False


def test_client_cache_ttl_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    created: List[_DummyClient] = []

    def fake_factory(**kwargs):  # noqa: D401
        client = _DummyClient()
        created.append(client)
        return client

    timeline = iter([0.0, 3600.0])

    monkeypatch.setattr(http_client, "get_rate_limited_client", fake_factory)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_now", lambda: next(timeline))

    base_url = "https://api.example"
    http_client.configure_client_cache_ttl(base_url, 0.0)

    first = http_client._resolve_client(base_url, 5.0)
    second = http_client._resolve_client(base_url, 5.0)

    assert first is second
    assert len(created) == 1
    assert first.closed is False


def test_client_cache_ttl_override_can_be_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    created: List[_DummyClient] = []

    def fake_factory(**kwargs):  # noqa: D401
        client = _DummyClient()
        created.append(client)
        return client

    timeline = iter([0.0, 0.6, 0.7, 1.8])

    monkeypatch.setattr(http_client, "get_rate_limited_client", fake_factory)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_now", lambda: next(timeline))
    monkeypatch.setattr(http_client, "_DEFAULT_CLIENT_TTL", 1.0)

    base_url = "https://api.example"
    http_client.configure_client_cache_ttl(base_url, 0.5)

    first = http_client._resolve_client(base_url, 10.0)
    second = http_client._resolve_client(base_url, 10.0)

    assert first is not second
    assert first.closed is True
    assert second.closed is False

    http_client.configure_client_cache_ttl(base_url, None)

    third = http_client._resolve_client(base_url, 10.0)
    fourth = http_client._resolve_client(base_url, 10.0)

    assert third is second
    assert fourth is not second
    assert third.closed is True
    assert fourth.closed is False
    assert len(created) == 3


def test_client_cache_applies_shorter_ttl_to_existing_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: List[_DummyClient] = []

    def fake_factory(**kwargs):  # noqa: D401
        client = _DummyClient()
        created.append(client)
        return client

    timeline = iter([0.0, 5.0])

    monkeypatch.setattr(http_client, "get_rate_limited_client", fake_factory)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_now", lambda: next(timeline))

    base_url = "https://api.example"
    http_client.configure_client_cache_ttl(base_url, 10.0)

    first = http_client._resolve_client(base_url, 2.0)
    assert len(created) == 1

    http_client.configure_client_cache_ttl(base_url, 2.0)

    second = http_client._resolve_client(base_url, 2.0)

    assert first is not second
    assert first.closed is True
    assert second.closed is False
    assert len(created) == 2
