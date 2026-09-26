from __future__ import annotations

import httpx
import pytest

from core.network.async_http import RateLimitedAsyncClient, get_rate_limited_client


@pytest.mark.asyncio
async def test_client_without_base_url_can_be_constructed_and_closed() -> None:
    client = RateLimitedAsyncClient()

    assert isinstance(client.client, httpx.AsyncClient)
    assert not client.client.is_closed

    await client.aclose()

    assert client.client.is_closed


@pytest.mark.asyncio
async def test_factory_without_base_url_can_be_constructed_and_closed() -> None:
    client = get_rate_limited_client()

    assert isinstance(client.client, httpx.AsyncClient)
    assert not client.client.is_closed

    await client.aclose()

    assert client.client.is_closed


@pytest.mark.asyncio
async def test_client_without_base_url_requests_absolute_url() -> None:
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handle)
    async with RateLimitedAsyncClient(transport=transport) as client:
        response = await client.request("GET", "https://example.test/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert [request.url for request in requests] == [httpx.URL("https://example.test/health")]


@pytest.mark.asyncio
async def test_client_with_explicit_base_url_requests_relative_url() -> None:
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    transport = httpx.MockTransport(handle)
    async with RateLimitedAsyncClient(
        base_url="https://example.test/api/", transport=transport
    ) as client:
        assert client.client.base_url == httpx.URL("https://example.test/api/")
        response = await client.request("GET", "status")

    assert response.status_code == 204
    assert [request.url for request in requests] == [httpx.URL("https://example.test/api/status")]
