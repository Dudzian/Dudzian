"""Asynchroniczny wrapper HTTP zastępujący ``urllib.request.urlopen`` w adapterach giełdowych."""
from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request

import httpx

from core.network import RateLimitedAsyncClient, get_rate_limited_client


__all__ = ["urlopen", "AsyncHTTPResponse"]


_MAX_CONNECTIONS = 40
_DEFAULT_TIMEOUT = 15.0


@dataclass(slots=True)
class _HeadersProxy:
    """Udostępnia metody kompatybilne z obiektem nagłówków ``http.client``."""

    _headers: Mapping[str, str]

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name, default)

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name, default)

    def items(self) -> Iterable[tuple[str, str]]:
        return self._headers.items()

    def keys(self) -> Iterable[str]:
        return self._headers.keys()

    def values(self) -> Iterable[str]:
        return self._headers.values()


class AsyncHTTPResponse:
    """Lekka reprezentacja odpowiedzi HTTP dla adapterów giełdowych."""

    __slots__ = ("url", "status", "code", "headers", "reason", "_content")

    def __init__(
        self,
        *,
        url: str,
        status_code: int,
        headers: Mapping[str, str],
        reason: str,
        content: bytes,
    ) -> None:
        self.url = url
        self.status = status_code
        self.code = status_code
        self.reason = reason
        self.headers = _HeadersProxy(headers)
        self._content = content

    def read(self) -> bytes:
        return self._content

    def close(self) -> None:  # pragma: no cover - kompatybilność z API urllib
        self._content = b""

    def __enter__(self) -> "AsyncHTTPResponse":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()


_CLIENT_CACHE: MutableMapping[tuple[str, float], RateLimitedAsyncClient] = {}
_CLIENT_LOCK = threading.Lock()


def _resolve_client(base_url: str, timeout: float) -> RateLimitedAsyncClient:
    key = (base_url, timeout)
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is not None:
            return client
        client = get_rate_limited_client(
            base_url=base_url,
            timeout=timeout,
            max_connections=_MAX_CONNECTIONS,
            max_keepalive_connections=_MAX_CONNECTIONS,
        )
        _CLIENT_CACHE[key] = client
        return client


def _extract_request_components(request: Request | str) -> tuple[str, str, Mapping[str, str], bytes | None]:
    if isinstance(request, Request):
        method = (request.get_method() or "GET").upper()
        url = request.full_url
        headers = {key: value for key, value in request.header_items()}
        data = request.data
        if isinstance(data, str):
            data = data.encode("utf-8")
        return method, url, headers, data
    method = "GET"
    url = str(request)
    return method, url, {}, None


async def _async_open(request: Request | str, timeout: float | None = None) -> AsyncHTTPResponse:
    method, url, headers, data = _extract_request_components(request)
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise URLError(f"Niepoprawny URL: {url}")
    base_url = f"{parts.scheme}://{parts.netloc}"
    path = parts.path or "/"
    params: Sequence[tuple[str, str]] | None = None
    if parts.query:
        params = tuple(parse_qsl(parts.query, keep_blank_values=True))
    client = _resolve_client(base_url, float(timeout or _DEFAULT_TIMEOUT))
    try:
        response = await client.request(
            method,
            path,
            headers=headers or None,
            params=params,
            content=data,
        )
    except httpx.RequestError as exc:
        raise URLError(exc) from exc

    status_code = int(response.status_code)
    content = response.content
    reason = response.reason_phrase or ""
    response_headers = {key: value for key, value in response.headers.items()}

    if status_code >= 400:
        fp = io.BytesIO(content)
        error = HTTPError(url, status_code, reason, response_headers, fp)
        raise error

    return AsyncHTTPResponse(
        url=url,
        status_code=status_code,
        headers=response_headers,
        reason=reason,
        content=content,
    )


def urlopen(request: Request | str, timeout: float | None = None) -> AsyncHTTPResponse:
    """Blokujące API kompatybilne z ``urllib.request.urlopen``."""

    return asyncio.run(_async_open(request, timeout=timeout))
