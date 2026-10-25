"""Retired cached clients close on their owning loop after their last request."""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import anyio
import httpx
import pytest

from bot_core.exchanges import http_client
from core.network import run_sync


class _Client:
    def __init__(self):
        self.close_calls = 0
        self.active = 0
        self.started = 0
        self.hold = None
        self.error = None

    async def request(self, method, path, **kwargs):
        self.started += 1
        self.active += 1
        try:
            if self.hold is not None:
                await self.hold.wait()
            if self.error is not None:
                raise self.error
            return httpx.Response(200, content=b"ok")
        finally:
            self.active -= 1

    async def aclose(self):
        assert not self.active, "an in-flight request lost its client"
        await anyio.sleep(0)
        self.close_calls += 1


@pytest.fixture
def cache(monkeypatch):
    clock = [0.0]
    created = []

    def factory(**kwargs):
        client = _Client()
        created.append(client)
        return client

    monkeypatch.setattr(http_client, "_CLIENT_CACHE", {})
    monkeypatch.setattr(http_client, "_CLIENT_TTLS", {})
    monkeypatch.setattr(http_client, "_now", lambda: clock[0])
    monkeypatch.setattr(http_client, "get_rate_limited_client", factory)
    yield clock, created
    http_client._shutdown_client_cache()


def test_idle_ttl_expiry_closes_on_the_real_portal(cache):
    clock, clients = cache
    assert http_client.urlopen("https://api.example/first").read() == b"ok"
    clock[0] = 301.0
    assert http_client.urlopen("https://api.example/second").read() == b"ok"
    assert len(clients) == 2
    assert [client.close_calls for client in clients] == [1, 0]
    http_client._shutdown_client_cache()
    http_client._shutdown_client_cache()
    assert [client.close_calls for client in clients] == [1, 1]


async def _wait_started(client, count):
    async def wait():
        while client.started != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=3)


@pytest.mark.parametrize("retire", ["ttl", "shutdown"])
def test_retirement_waits_for_every_active_request(cache, retire):
    clock, clients = cache

    async def scenario():
        client = http_client._resolve_client("https://api.example", http_client._DEFAULT_TIMEOUT)
        client.hold = asyncio.Event()
        tasks = [
            asyncio.create_task(http_client._async_open("https://api.example/held"))
            for _ in range(2)
        ]
        try:
            await _wait_started(client, 2)
            if retire == "ttl":
                clock[0] = 301.0
                await http_client._async_open("https://api.example/new")
                assert len(clients) == 2
            await asyncio.to_thread(http_client._shutdown_client_cache)
            await asyncio.to_thread(http_client._shutdown_client_cache)
            assert client.close_calls == 0
        finally:
            client.hold.set()
            await asyncio.gather(*tasks)
        assert client.close_calls == 1
        assert all(item.close_calls == 1 for item in clients)

    run_sync(scenario)


def test_failed_request_releases_its_retired_client_and_preserves_the_error(cache):
    clock, clients = cache

    async def scenario():
        client = http_client._resolve_client("https://api.example", http_client._DEFAULT_TIMEOUT)
        client.hold = asyncio.Event()
        client.error = httpx.ReadError("controlled request failure")
        task = asyncio.create_task(http_client._async_open("https://api.example/held"))
        try:
            await _wait_started(client, 1)
            clock[0] = 301.0
            await http_client._async_open("https://api.example/new")
            assert client.close_calls == 0
        finally:
            client.hold.set()
            with pytest.raises(URLError) as raised:
                await task
            assert raised.value.__cause__ is client.error
        assert client.close_calls == 1

    run_sync(scenario)


def test_cancelled_scope_still_closes_its_retired_client(cache):
    clock, clients = cache

    async def scenario():
        with anyio.CancelScope() as scope:
            async with http_client._client_lease(
                "https://api.example", http_client._DEFAULT_TIMEOUT
            ) as client:
                clock[0] = 301.0
                await http_client._async_open("https://api.example/new")
                scope.cancel()
                await anyio.sleep(0)
        assert scope.cancelled_caught
        assert client.close_calls == 1
        assert (
            http_client._CLIENT_CACHE[
                ("https://api.example", http_client._DEFAULT_TIMEOUT)
            ].active_requests
            == 0
        )

    run_sync(scenario)


def test_factory_failure_keeps_expired_clients_available_for_cleanup(cache, monkeypatch):
    clock, clients = cache
    http_client.urlopen("https://api.example/first")
    clock[0] = 301.0

    def fail(**kwargs):
        raise RuntimeError("controlled factory failure")

    monkeypatch.setattr(http_client, "get_rate_limited_client", fail)
    with pytest.raises(RuntimeError, match="controlled factory failure"):
        http_client.urlopen("https://api.example/second")
    assert (
        http_client._CLIENT_CACHE[("https://api.example", http_client._DEFAULT_TIMEOUT)].client
        is clients[0]
    )
    http_client._shutdown_client_cache()
    assert clients[0].close_calls == 1


def test_expired_real_httpx_connection_closes_without_waiting_for_peer_or_gc(monkeypatch):
    condition = threading.Condition()
    connections = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            with condition:
                connections.add(self.connection)
                condition.notify_all()

        def finish(self):
            try:
                super().finish()
            finally:
                with condition:
                    connections.discard(self.connection)
                    condition.notify_all()

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            self.wfile.flush()

        def log_message(self, *args):
            pass

    clock = [0.0]
    monkeypatch.setattr(http_client, "_CLIENT_CACHE", {})
    monkeypatch.setattr(http_client, "_CLIENT_TTLS", {})
    monkeypatch.setattr(http_client, "_now", lambda: clock[0])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        assert http_client.urlopen(url).read() == b"ok"
        clock[0] = 301.0
        assert http_client.urlopen(url).read() == b"ok"
        with condition:
            assert condition.wait_for(lambda: len(connections) == 1, timeout=3)
        http_client._shutdown_client_cache()
        with condition:
            assert condition.wait_for(lambda: not connections, timeout=3)
    finally:
        http_client._shutdown_client_cache()
        # Keep failure cleanup bounded, including when run against the old leak.
        with condition:
            remaining = list(connections)
        for connection in remaining:
            try:
                connection.shutdown(2)
            except OSError:
                pass
            connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()
