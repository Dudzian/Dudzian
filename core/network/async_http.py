"""Asynchroniczny klient HTTP z kontrolą limitów i retry."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx


_DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass(slots=True)
class _RetryConfig:
    retries: int = 3
    backoff_factor: float = 0.4
    jitter_range: tuple[float, float] = (0.05, 0.35)
    retry_statuses: frozenset[int] = _DEFAULT_RETRY_STATUSES


class RateLimitedAsyncClient:
    """Lekki wrapper nad :class:`httpx.AsyncClient` z obsługą retry."""

    __slots__ = ("_client", "_retry")

    def __init__(
        self,
        *,
        base_url: str | httpx.URL | None = None,
        timeout: float = 10.0,
        max_connections: int = 20,
        max_keepalive_connections: int = 20,
        retries: int = 3,
        backoff_factor: float = 0.4,
        retry_statuses: Iterable[int] | None = None,
        jitter_range: Sequence[float] = (0.05, 0.35),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=limits,
            transport=transport,
        )
        low, high = (float(jitter_range[0]), float(jitter_range[1])) if len(jitter_range) >= 2 else (0.05, 0.35)
        self._retry = _RetryConfig(
            retries=max(1, int(retries)),
            backoff_factor=max(0.0, float(backoff_factor)),
            jitter_range=(min(low, high), max(low, high)),
            retry_statuses=frozenset(int(code) for code in (retry_statuses or _DEFAULT_RETRY_STATUSES)),
        )

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RateLimitedAsyncClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        await self._client.__aexit__(exc_type, exc, tb)

    async def request(self, method: str, url: str | httpx.URL, **kwargs) -> httpx.Response:
        attempt = 0
        last_response: httpx.Response | None = None
        while attempt < self._retry.retries:
            attempt += 1
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.RequestError:
                if attempt >= self._retry.retries:
                    raise
                await self._sleep_backoff(attempt)
                continue

            if response.status_code not in self._retry.retry_statuses:
                return response

            last_response = response
            if attempt >= self._retry.retries:
                return response
            await self._sleep_backoff(attempt)
        assert last_response is not None  # pragma: no cover - zabezpieczenie
        return last_response

    async def _sleep_backoff(self, attempt: int) -> None:
        base_delay = self._retry.backoff_factor * (2 ** (attempt - 1))
        jitter = random.uniform(*self._retry.jitter_range)
        await asyncio.sleep(base_delay + jitter)


def get_rate_limited_client(
    *,
    base_url: str | httpx.URL | None = None,
    timeout: float = 10.0,
    max_connections: int = 20,
    max_keepalive_connections: int = 20,
    retries: int = 3,
    backoff_factor: float = 0.4,
    retry_statuses: Iterable[int] | None = None,
    jitter_range: Sequence[float] = (0.05, 0.35),
    transport: httpx.AsyncBaseTransport | None = None,
) -> RateLimitedAsyncClient:
    """Zwraca skonfigurowany klient HTTP do wykorzystania w adapterach."""

    return RateLimitedAsyncClient(
        base_url=base_url,
        timeout=timeout,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        retries=retries,
        backoff_factor=backoff_factor,
        retry_statuses=retry_statuses,
        jitter_range=jitter_range,
        transport=transport,
    )


__all__ = ["RateLimitedAsyncClient", "get_rate_limited_client"]
