from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot_core.exchanges import http_client


class _DummyResponse:
    status_code = 200
    content = b"{}"
    reason_phrase = "OK"
    headers = {}


class _TimeoutCapturingClient:
    def __init__(self) -> None:
        self.observed_timeouts: list[float | None] = []

    async def request(self, method, path, **kwargs):  # noqa: ANN001, D401
        self.observed_timeouts.append(kwargs.get("timeout"))
        return _DummyResponse()


def _run_sync(async_fn, *args, **kwargs):
    async def _runner():
        return await async_fn(*args, **kwargs)

    return asyncio.run(_runner())


def _fake_httpx_module() -> object:
    return SimpleNamespace(RequestError=Exception)


def test_urlopen_forwards_per_request_timeout(monkeypatch):
    client = _TimeoutCapturingClient()

    monkeypatch.setattr(http_client, "_resolve_client", lambda base_url, timeout: client)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_require_httpx", _fake_httpx_module)

    response = http_client.urlopen("http://example.com/test", timeout=0.123)

    assert response.status == 200
    assert client.observed_timeouts == [0.123]


def test_urlopen_forwards_zero_timeout_without_defaulting(monkeypatch):
    client = _TimeoutCapturingClient()

    monkeypatch.setattr(http_client, "_resolve_client", lambda base_url, timeout: client)
    monkeypatch.setattr(http_client, "run_sync", _run_sync)
    monkeypatch.setattr(http_client, "_require_httpx", _fake_httpx_module)

    response = http_client.urlopen("http://example.com/test", timeout=0.0)

    assert response.status == 200
    assert client.observed_timeouts == [0.0]
