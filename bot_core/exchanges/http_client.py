"""Asynchroniczny wrapper HTTP zastępujący ``urllib.request.urlopen`` w adapterach giełdowych."""

from __future__ import annotations

import atexit
import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request

if TYPE_CHECKING:
    from core.network import RateLimitedAsyncClient
else:
    RateLimitedAsyncClient = Any

__all__ = [
    "urlopen",
    "AsyncHTTPResponse",
    "configure_client_cache_ttl",
    "get_rate_limited_client",
    "run_sync",
]


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
    active_requests: int = 0
    retired: bool = False
    close_claimed: bool = False


_CLIENT_CACHE: MutableMapping[tuple[str, float], _ClientCacheEntry] = {}
_CLIENT_TTLS: MutableMapping[str, float] = {}
_CLIENT_LOCK = threading.Lock()
_CLIENT_SHUTDOWN_LOCK = threading.Lock()
_CLIENT_SHUTDOWN_REGISTERED = False


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


def _require_httpx():
    try:
        import httpx
    except (ModuleNotFoundError, ImportError) as exc:
        dep = getattr(exc, "name", None) or "dependency"
        message = f"Brak opcjonalnej zależności '{dep}' wymaganej dla bot_core.exchanges.http_client:httpx."
        raise RuntimeError(message) from exc
    return httpx


def _require_core_network():
    global _CLIENT_SHUTDOWN_REGISTERED
    try:
        from core.network import RateLimitedAsyncClient, get_rate_limited_client, run_sync
    except (ModuleNotFoundError, ImportError) as exc:
        dep = getattr(exc, "name", None) or "dependency"
        message = f"Brak opcjonalnej zależności '{dep}' wymaganej dla core.network."
        raise RuntimeError(message) from exc
    # atexit runs in reverse order. Importing core.network first ensures cached
    # clients close on the live portal before that portal's own shutdown.
    with _CLIENT_SHUTDOWN_LOCK:
        if not _CLIENT_SHUTDOWN_REGISTERED:
            atexit.register(_shutdown_client_cache)
            _CLIENT_SHUTDOWN_REGISTERED = True
    return RateLimitedAsyncClient, get_rate_limited_client, run_sync


def get_rate_limited_client(*args, **kwargs):
    _, get_client, _ = _require_core_network()
    return get_client(*args, **kwargs)


def run_sync(*args, **kwargs):
    _, _, run_sync_impl = _require_core_network()
    return run_sync_impl(*args, **kwargs)


def _retire_client_entry(entry: _ClientCacheEntry) -> RateLimitedAsyncClient | None:
    """Retire and claim an unused client exactly once, with _CLIENT_LOCK held."""
    entry.retired = True
    if entry.active_requests or entry.close_claimed:
        return None
    entry.close_claimed = True
    return entry.client


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
            client = _retire_client_entry(entry)
            if client is not None:
                expired.append(client)
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
        to_close = [_retire_client_entry(entry) for entry in entries]
    for client in to_close:
        if client is not None:
            _close_client(client)


def _resolve_client_entry(
    base_url: str, timeout: float, *, acquire: bool
) -> tuple[_ClientCacheEntry, list[RateLimitedAsyncClient]]:
    normalized_base = _normalize_base_url(base_url)
    key = (normalized_base, timeout)
    now = _now()
    with _CLIENT_LOCK:
        entry = _CLIENT_CACHE.get(key)
        ttl = _resolve_ttl(normalized_base)
        if entry is None or (ttl > 0 and now >= entry.last_access + ttl):
            # Construct before retiring entries: a factory failure must not lose
            # the only references to clients still awaiting cleanup.
            client = get_rate_limited_client(
                base_url=normalized_base,
                timeout=timeout,
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_CONNECTIONS,
            )
            entry = _ClientCacheEntry(
                client=client,
                timeout=timeout,
                last_access=now,
                expires_at=now + ttl if ttl > 0 else float("inf"),
            )
        to_close = _purge_expired_clients(now)
        entry.last_access = now
        entry.expires_at = now + ttl if ttl > 0 else float("inf")
        _CLIENT_CACHE[key] = entry
        if acquire:
            entry.active_requests += 1
    return entry, to_close


def _resolve_client(base_url: str, timeout: float) -> RateLimitedAsyncClient:
    entry, to_close = _resolve_client_entry(base_url, timeout, acquire=False)
    for candidate in to_close:
        _close_client(candidate)
    return entry.client


async def _close_clients_async(clients: Iterable[RateLimitedAsyncClient]) -> None:
    # Keep optional native-network dependencies lazy and finish cleanup even
    # when the request's AnyIO cancel scope has already been cancelled.
    from anyio import CancelScope

    with CancelScope(shield=True):
        for client in clients:
            try:
                await client.aclose()
            except Exception:  # Match the synchronous cleanup's diagnostic-only failures.
                _LOGGER.debug("Nie udało się zamknąć klienta HTTP", exc_info=True)


@asynccontextmanager
async def _client_lease(base_url: str, timeout: float) -> AsyncIterator[RateLimitedAsyncClient]:
    entry, to_close = _resolve_client_entry(base_url, timeout, acquire=True)
    try:
        await _close_clients_async(to_close)
        yield entry.client
    finally:
        with _CLIENT_LOCK:
            entry.active_requests -= 1
            client = _retire_client_entry(entry) if entry.retired else None
        if client is not None:
            await _close_clients_async((client,))


def _extract_request_components(
    request: Request | str,
) -> tuple[str, str, Mapping[str, str], bytes | None]:
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
    resolved_timeout = _DEFAULT_TIMEOUT if timeout is None else float(timeout)
    httpx = _require_httpx()
    async with _client_lease(base_url, resolved_timeout) as client:
        try:
            response = await client.request(
                method,
                path,
                headers=headers or None,
                params=params,
                content=data,
                timeout=resolved_timeout,
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
