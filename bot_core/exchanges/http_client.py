"""Asynchroniczny wrapper HTTP zastępujący ``urllib.request.urlopen`` w adapterach giełdowych."""
from __future__ import annotations

import atexit
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request

import httpx

from core.network import RateLimitedAsyncClient, get_rate_limited_client, run_sync


__all__ = ["urlopen", "AsyncHTTPResponse", "configure_client_cache_ttl"]


_MAX_CONNECTIONS = 40
_DEFAULT_TIMEOUT = 15.0
_DEFAULT_CLIENT_TTL = 300.0


_LOGGER = logging.getLogger(__name__)


def _now() -> float:
    return time.monotonic()


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


@dataclass(slots=True)
class _ClientCacheEntry:
    client: RateLimitedAsyncClient
    timeout: float
    last_access: float
    expires_at: float


_CLIENT_CACHE: MutableMapping[tuple[str, float], _ClientCacheEntry] = {}
_CLIENT_TTLS: MutableMapping[str, float] = {}
_CLIENT_LOCK = threading.Lock()


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized or base_url


def configure_client_cache_ttl(base_url: str, ttl: float | None) -> None:
    """Konfiguruje TTL cache'u klientów dla danego ``base_url``.

    Wartość ``ttl`` w sekundach mniejsza lub równa zero oznacza wyłączenie
    wygaszania. Przekazanie ``None`` usuwa nadpisaną wartość przywracając
    domyślny TTL.
    """

    normalized = _normalize_base_url(base_url)
    with _CLIENT_LOCK:
        if ttl is None:
            _CLIENT_TTLS.pop(normalized, None)
            return
        try:
            numeric = float(ttl)
        except (TypeError, ValueError) as exc:
            raise ValueError("TTL klienta musi być liczbą zmiennoprzecinkową") from exc
        _CLIENT_TTLS[normalized] = numeric


def _resolve_ttl(base_url: str) -> float:
    ttl = _CLIENT_TTLS.get(base_url)
    if ttl is None:
        return _DEFAULT_CLIENT_TTL
    return float(ttl)


def _purge_expired_clients(now: float) -> list[RateLimitedAsyncClient]:
    expired: list[RateLimitedAsyncClient] = []
    stale_keys: list[tuple[str, float]] = []
    for key, entry in list(_CLIENT_CACHE.items()):
        base_url = key[0]
        ttl = _resolve_ttl(base_url)
        if ttl <= 0:
            entry.expires_at = float("inf")
            continue
        deadline = entry.last_access + ttl
        if deadline != entry.expires_at:
            entry.expires_at = deadline
        if now >= entry.expires_at:
            stale_keys.append(key)
            expired.append(entry.client)
    for key in stale_keys:
        _CLIENT_CACHE.pop(key, None)
    return expired


def _close_client(client: RateLimitedAsyncClient) -> None:
    try:
        run_sync(client.aclose)
    except Exception:  # pragma: no cover - logowanie diagnostyczne przy wychodzeniu
        _LOGGER.debug("Nie udało się zamknąć klienta HTTP", exc_info=True)


def _shutdown_client_cache() -> None:
    with _CLIENT_LOCK:
        entries = list(_CLIENT_CACHE.values())
        _CLIENT_CACHE.clear()
    for entry in entries:
        _close_client(entry.client)


atexit.register(_shutdown_client_cache)


def _resolve_client(base_url: str, timeout: float) -> RateLimitedAsyncClient:
    normalized_base = _normalize_base_url(base_url)
    key = (normalized_base, timeout)
    now = _now()
    to_close: list[RateLimitedAsyncClient] = []
    with _CLIENT_LOCK:
        to_close = _purge_expired_clients(now)
        entry = _CLIENT_CACHE.get(key)
        if entry is not None:
            ttl = _resolve_ttl(normalized_base)
            entry.last_access = now
            entry.expires_at = now + ttl if ttl > 0 else float("inf")
            client = entry.client
        else:
            client = get_rate_limited_client(
                base_url=normalized_base,
                timeout=timeout,
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_CONNECTIONS,
            )
            ttl = _resolve_ttl(normalized_base)
            expires_at = now + ttl if ttl > 0 else float("inf")
            _CLIENT_CACHE[key] = _ClientCacheEntry(
                client=client,
                timeout=timeout,
                last_access=now,
                expires_at=expires_at,
            )
    for candidate in to_close:
        _close_client(candidate)
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

    return run_sync(_async_open, request, timeout=timeout)
