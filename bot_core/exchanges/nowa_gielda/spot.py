"""Minimalistyczny adapter REST dla rynku spot nowa_gielda."""
from __future__ import annotations

import hmac
import logging
import time
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Deque, Iterable, Mapping, Optional, Protocol, Sequence

import httpx

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.nowa_gielda import symbols
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from core.network import RateLimitedAsyncClient, get_rate_limited_client, run_sync

_PUBLIC_STREAM_CHANNEL_MAP: Mapping[str, str] = {
    "ticker": "ticker",
    "depth": "orderbook",
    "trades": "trades",
}

_PRIVATE_STREAM_CHANNEL_MAP: Mapping[str, str] = {
    "orders": "orders",
    "balances": "account",
    "fills": "fills",
}

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """Opis pojedynczego limitu wagowego endpointu REST."""

    method: str
    path: str
    weight: int
    window_seconds: float
    max_requests: int

    @property
    def key(self) -> str:
        return f"{self.method.upper()} {self.path}".strip()


_RATE_LIMITS: Mapping[str, RateLimitRule] = {
    rule.key: rule
    for rule in (
        RateLimitRule("GET", "/public/ticker", weight=1, window_seconds=1.0, max_requests=20),
        RateLimitRule("GET", "/public/orderbook", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("GET", "/public/ohlcv", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("GET", "/private/account", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("GET", "/private/orders", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/orders/history", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/trades", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("POST", "/private/orders", weight=5, window_seconds=1.0, max_requests=5),
        RateLimitRule("DELETE", "/private/orders", weight=1, window_seconds=1.0, max_requests=10),
    )
}


def _strip_none(data: Mapping[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    return {key: value for key, value in data.items() if value is not None}


def _canonical_payload(data: Mapping[str, Any] | None) -> str:
    if not data:
        return ""
    items: list[tuple[str, str]] = []
    for key, value in data.items():
        items.append((str(key), str(value)))
    items.sort()
    return "&".join(f"{key}={value}" for key, value in items)


class _RateLimiter:
    """Prosty licznik zużycia limitów w oknie czasowym."""

    __slots__ = ("_rules", "_state")

    def __init__(self, rules: Mapping[str, RateLimitRule]) -> None:
        self._rules = rules
        self._state: dict[str, tuple[float, int]] = {}

    def consume(self, method: str, path: str) -> None:
        key = f"{method.upper()} {path}".strip()
        rule = self._rules.get(key)
        if rule is None:
            return

        now = time.monotonic()
        window_start, used = self._state.get(key, (now, 0))
        if now - window_start >= rule.window_seconds:
            window_start = now
            used = 0

        projected = used + rule.weight
        if projected > rule.max_requests:
            raise ExchangeThrottlingError(
                message="Limit zapytań dla endpointu został przekroczony",
                status_code=429,
                payload={"endpoint": key, "used": used, "limit": rule.max_requests},
            )

        self._state[key] = (window_start, projected)


_ERROR_CODE_MAPPING: Mapping[str, type[ExchangeAPIError]] = {
    "INVALID_SIGNATURE": ExchangeAuthError,
    "AUTHENTICATION_REQUIRED": ExchangeAuthError,
    "RATE_LIMIT_EXCEEDED": ExchangeThrottlingError,
    "ORDER_NOT_FOUND": ExchangeAPIError,
    "INVALID_SYMBOL": ExchangeAPIError,
}


class NowaGieldaHTTPClient:
    """Klient HTTP odpowiadający za komunikację z REST API nowa_gielda."""

    __slots__ = ("_base_url", "_client", "_rate_limiter", "_owns_client")

    def __init__(
        self,
        base_url: str,
        *,
        client: RateLimitedAsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._rate_limiter = _RateLimiter(_RATE_LIMITS)
        if client is None:
            self._client = get_rate_limited_client(base_url=self._base_url, timeout=timeout)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    @property
    def rate_limiter(self) -> _RateLimiter:
        return self._rate_limiter

    async def _request_async(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        self._rate_limiter.consume(method, path)
        query = _strip_none(params)
        body = _strip_none(json_body)
        try:
            response = await self._client.request(
                method,
                path,
                params=query or None,
                json=body or None,
                headers=headers,
            )
        except httpx.RequestError as exc:  # pragma: no cover - zabezpieczenie
            raise ExchangeNetworkError("Błąd połączenia z API nowa_gielda", reason=exc) from exc

        return self._parse_response(method, path, response)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        return run_sync(
            self._request_async,
            method,
            path,
            params=params,
            json_body=json_body,
            headers=headers,
        )

    def _parse_response(self, method: str, path: str, response: httpx.Response) -> Mapping[str, Any]:
        status = response.status_code
        if 200 <= status < 300:
            try:
                return response.json()
            except ValueError as exc:  # pragma: no cover - defensywnie
                raise ExchangeAPIError(
                    message="Niepoprawny format JSON odpowiedzi",
                    status_code=status,
                    payload=response.text,
                ) from exc

        payload: Any
        try:
            payload = response.json()
        except ValueError:  # pragma: no cover - fallback
            payload = response.text

        message = ""
        code: str | None = None
        if isinstance(payload, Mapping):
            message = str(payload.get("message", ""))
            raw_code = payload.get("code")
            code = str(raw_code) if raw_code is not None else None

        exc_cls: type[ExchangeAPIError] | None = None
        if code:
            exc_cls = _ERROR_CODE_MAPPING.get(code)

        if exc_cls is None:
            if status in {401, 403}:
                exc_cls = ExchangeAuthError
            elif status == 429:
                exc_cls = ExchangeThrottlingError
            else:
                exc_cls = ExchangeAPIError

        raise exc_cls(
            message=message or f"Błąd API ({status}) przy {method.upper()} {path}",
            status_code=status,
            payload=payload,
        )

    def close(self) -> None:
        if self._owns_client:
            run_sync(self._client.aclose)

    # --- Public helpers -------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Mapping[str, Any]:
        return self._request("GET", "/public/ticker", params={"symbol": symbol})

    def fetch_orderbook(self, symbol: str, depth: int = 50) -> Mapping[str, Any]:
        return self._request(
            "GET",
            "/public/orderbook",
            params={"symbol": symbol, "depth": depth},
        )

    def fetch_account(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/account", params=params, headers=headers)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        query = _strip_none(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        if params:
            query = {**query, **_strip_none(params)}
        return self._request(
            "GET",
            "/public/ohlcv",
            params=query,
            headers=headers,
        )

    def fetch_trades(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/trades", params=params, headers=headers)

    def fetch_open_orders(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/orders", params=params, headers=headers)

    def fetch_order_history(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/orders/history", params=params, headers=headers)

    def create_order(
        self,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        return self._request("POST", "/private/orders", json_body=payload, headers=headers)

    def cancel_order(
        self,
        order_id: str,
        *,
        headers: Mapping[str, str],
        symbol: Optional[str] = None,
    ) -> Mapping[str, Any]:
        params = {"orderId": order_id}
        if symbol is not None:
            params["symbol"] = symbol
        return self._request("DELETE", "/private/orders", params=params, headers=headers)


class NowaGieldaStreamClient(Iterable[StreamBatch]):
    """Lokalny klient streamingu z obsługą reconnectów i bufora."""

    __slots__ = (
        "_adapter",
        "_scope",
        "_channels",
        "_remote_channels",
        "_channel_map",
        "_fallback_factory",
        "_stream",
        "_closed",
        "_pending",
        "_history",
        "_replay_scheduled",
        "_max_reconnects",
        "_backoff_base",
        "_backoff_cap",
        "_clock",
        "_sleep",
        "_last_cursor",
        "_symbol_filter",
        "_reconnect_attempt",
        "_total_batches",
        "_total_events",
        "_heartbeats",
        "_reconnects_total",
        "_metrics",
        "_metric_labels",
        "_metric_pending",
        "_metric_history",
        "_metric_reconnects",
        "_metric_batches",
        "_metric_events",
        "_metric_heartbeats",
        "_metric_replays",
    )

    _CURSOR_UNCHANGED = object()

    def __init__(
        self,
        *,
        adapter: str | None = None,
        scope: str,
        channels: Sequence[str],
        fallback_factory: Callable[[Sequence[str], Optional[str]], LocalLongPollStream],
        channel_mapping: Mapping[str, str] | None = None,
        symbols: Sequence[str] | None = None,
        max_reconnects: int = 3,
        backoff_base: float = 0.5,
        backoff_cap: float = 5.0,
        buffer_size: int = 8,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        normalized_channels = tuple(
            dict.fromkeys(
                channel.strip()
                for channel in (str(item) for item in channels)
                if str(channel).strip()
            )
        )
        if not normalized_channels:
            raise ValueError("Lista kanałów subskrypcji nie może być pusta.")

        self._adapter = (adapter or "nowa_gielda_stream").strip() or "nowa_gielda_stream"
        self._scope = scope
        self._channels = normalized_channels
        self._channel_map = dict(channel_mapping or {})
        remote_channels: list[str] = []
        for channel in normalized_channels:
            if self._channel_map:
                if channel not in self._channel_map:
                    raise ValueError(f"Kanał {channel!r} nie jest wspierany w streamie {scope}.")
                remote_channels.append(str(self._channel_map[channel]))
            else:
                remote_channels.append(str(channel))

        self._remote_channels = tuple(remote_channels)
        self._fallback_factory = fallback_factory
        self._closed = False
        self._pending: Deque[StreamBatch] = deque()
        self._history: Deque[StreamBatch] = deque(maxlen=max(1, int(buffer_size)))
        self._replay_scheduled = False
        self._max_reconnects = max(0, int(max_reconnects)) or 1
        self._backoff_base = max(0.0, float(backoff_base))
        self._backoff_cap = max(self._backoff_base, float(backoff_cap))
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_cursor: str | None = None
        self._symbol_filter = frozenset(
            self._normalize_symbol(symbol)
            for symbol in (symbols or ())
            if self._normalize_symbol(symbol)
        )
        self._reconnect_attempt = 0
        self._total_batches = 0
        self._total_events = 0
        self._heartbeats = 0
        self._reconnects_total = 0
        self._stream = self._fallback_factory(self._remote_channels, None)
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_labels = {"adapter": self._adapter, "scope": self._scope}
        self._metric_pending = self._metrics.gauge(
            "bot_exchange_stream_client_pending_batches",
            "Aktualna liczba paczek oczekujących na wydanie przez klienta streamu.",
        )
        self._metric_history = self._metrics.gauge(
            "bot_exchange_stream_client_history_size",
            "Liczba paczek przechowywanych w historii klienta streamu.",
        )
        self._metric_reconnects = self._metrics.counter(
            "bot_exchange_stream_client_reconnects_total",
            "Łączna liczba restartów fallbackowego streamu.",
        )
        self._metric_batches = self._metrics.counter(
            "bot_exchange_stream_client_batches_total",
            "Łączna liczba paczek zwróconych przez klienta streamu.",
        )
        self._metric_events = self._metrics.counter(
            "bot_exchange_stream_client_events_total",
            "Łączna liczba zdarzeń dostarczonych przez klienta streamu.",
        )
        self._metric_heartbeats = self._metrics.counter(
            "bot_exchange_stream_client_heartbeats_total",
            "Liczba heartbeatów zwróconych przez klienta streamu.",
        )
        self._metric_replays = self._metrics.counter(
            "bot_exchange_stream_client_history_replays_total",
            "Łączna liczba paczek ponownie wysłanych z historii klienta streamu.",
        )
        self._update_queue_metrics()

    def __iter__(self) -> "NowaGieldaStreamClient":
        return self

    def __enter__(self) -> "NowaGieldaStreamClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()
        return None

    def __next__(self) -> StreamBatch:
        while not self._closed:
            if self._pending:
                batch = self._pending.popleft()
                self._update_queue_metrics()
                self._record_yield(batch)
                return batch
            try:
                batch = next(self._stream)
            except StopIteration:
                self.close()
                raise
            except (ExchangeNetworkError, ExchangeThrottlingError) as exc:
                self._handle_disconnect(exc)
                continue
            filtered = self._filter_batch(batch)
            if filtered is None:
                continue
            if filtered.cursor is not None:
                self._last_cursor = filtered.cursor
            elif batch.cursor is not None:
                self._last_cursor = batch.cursor
            self._enqueue_batch(filtered)
        raise StopIteration

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._pending.clear()
            self._history.clear()
            self._update_queue_metrics()

    @property
    def closed(self) -> bool:
        """Informuje, czy klient został zamknięty."""

        return self._closed

    @property
    def channels(self) -> tuple[str, ...]:
        """Zwraca listę kanałów zadeklarowanych podczas tworzenia klienta."""

        return self._channels

    @property
    def remote_channels(self) -> tuple[str, ...]:
        """Zwraca kanały przekazywane do fallbackowego streamu REST."""

        return self._remote_channels

    @property
    def scope(self) -> str:
        """Udostępnia przestrzeń streamu (np. ``public`` lub ``private``)."""

        return self._scope

    @property
    def last_cursor(self) -> str | None:
        """Udostępnia ostatni znany kursor streamu (jeśli został nadany)."""

        return self._last_cursor

    @property
    def max_reconnects(self) -> int:
        """Zwraca limit ponowień połączenia."""

        return self._max_reconnects

    @property
    def reconnect_attempt(self) -> int:
        """Informuje ile kolejnych prób reconnectu zostało już wykonanych."""

        return self._reconnect_attempt

    @property
    def reconnects_total(self) -> int:
        """Zwraca łączną liczbę wykonanych restartów streamu."""

        return self._reconnects_total

    @property
    def buffer_size(self) -> int:
        """Zwraca maksymalną pojemność bufora historii paczek."""

        return int(self._history.maxlen or 0)

    @property
    def history_size(self) -> int:
        """Informuje ile paczek aktualnie znajduje się w historii."""

        return len(self._history)

    @property
    def pending_size(self) -> int:
        """Informuje ile paczek oczekuje w kolejce do zwrócenia."""

        return len(self._pending)

    @property
    def total_batches(self) -> int:
        """Zlicza paczki zwrócone użytkownikowi (łącznie z odtworzoną historią)."""

        return self._total_batches

    @property
    def total_events(self) -> int:
        """Zwraca sumę zdarzeń dostarczonych w paczkach."""

        return self._total_events

    @property
    def heartbeats_received(self) -> int:
        """Zwraca liczbę heartbeatów zwróconych w paczkach."""

        return self._heartbeats

    def reset_counters(self) -> None:
        """Zeruje liczniki diagnostyczne paczek, zdarzeń i restartów."""

        self._total_batches = 0
        self._total_events = 0
        self._heartbeats = 0
        self._reconnects_total = 0

    def replay_history(
        self,
        *,
        include_heartbeats: bool = True,
        force: bool = False,
    ) -> bool:
        """Ponownie emituje zbuforowaną historię paczek.

        Parametr ``include_heartbeats`` pozwala pominąć puste paczki będące
        jedynie heartbeatami. Gdy ``force`` ustawione jest na ``True`` metoda
        wyczyści aktualną kolejkę oczekujących paczek oraz zdejmie blokadę
        ponownego odtwarzania, co umożliwia natychmiastowe ponowne odtworzenie
        historii nawet jeśli klient czeka na świeże dane po reconnectach.
        Zwraca ``True`` jeśli historia została dodana do kolejki, w przeciwnym
        razie ``False``.
        """

        if self._closed or not self._history:
            return False

        if not force and (self._pending or self._replay_scheduled):
            return False

        snapshots = (
            list(self._history)
            if include_heartbeats
            else [batch for batch in self._history if batch.events]
        )
        if not snapshots:
            return False

        if force:
            self._pending.clear()
            self._replay_scheduled = False

        for batch in snapshots:
            self._pending.append(batch)

        self._metric_replays.inc(len(snapshots), labels=self._metric_labels)
        self._update_queue_metrics()

        return True

    def force_reconnect(
        self,
        *,
        cursor: str | None | object = _CURSOR_UNCHANGED,
        replay_history: bool = True,
    ) -> None:
        """Wymusza natychmiastowe ponowne połączenie z fallbackowym streamem.

        Parametr ``cursor`` pozwala wskazać nowy kursor startowy. Gdy nie
        zostanie przekazany, klient użyje ostatnio znanego kursora.
        Jeżeli ``replay_history`` ustawione jest na ``True`` (wartość
        domyślna), zbuforowana historia zostanie umieszczona w kolejce
        oczekujących paczek przed pobraniem nowych danych.
        """

        if self._closed:
            raise RuntimeError("Próba reconnectu zamkniętego klienta streamu.")

        self._restart_stream(cursor=cursor)
        self._reconnect_attempt = 0
        self._reconnects_total += 1
        self._metric_reconnects.inc(labels=self._metric_labels)
        if replay_history and self._history:
            self._pending.extend(self._history)
            self._replay_scheduled = True
            self._metric_replays.inc(len(self._history), labels=self._metric_labels)
        else:
            self._replay_scheduled = False
        self._update_queue_metrics()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _enqueue_batch(self, batch: StreamBatch) -> None:
        self._pending.append(batch)
        if batch.events or batch.heartbeat:
            self._history.append(batch)
        self._replay_scheduled = False
        self._reconnect_attempt = 0
        self._update_queue_metrics()

    def _handle_disconnect(self, exc: Exception) -> None:
        if self._closed:
            return
        self._reconnect_attempt += 1
        if self._reconnect_attempt > self._max_reconnects:
            self.close()
            raise
        delay = min(self._backoff_base * (2 ** (self._reconnect_attempt - 1)), self._backoff_cap)
        if delay > 0:
            self._sleep(delay)
        self._restart_stream()
        self._reconnects_total += 1
        self._metric_reconnects.inc(labels=self._metric_labels)
        if self._history and not self._replay_scheduled:
            # udostępnij ostatnie dane zanim pojawią się nowe paczki
            self._pending.extend(self._history)
            self._replay_scheduled = True
            self._metric_replays.inc(len(self._history), labels=self._metric_labels)
        self._update_queue_metrics()

    def _restart_stream(self, *, cursor: object = _CURSOR_UNCHANGED) -> None:
        if cursor is not self._CURSOR_UNCHANGED:
            if cursor is None:
                self._last_cursor = None
            elif isinstance(cursor, str):
                self._last_cursor = cursor
            else:
                self._last_cursor = str(cursor)
        try:
            self._stream.close()
        except Exception:  # pragma: no cover - defensywnie
            _LOGGER.debug("Błąd podczas zamykania streamu %s", self._scope, exc_info=True)
        self._stream = self._fallback_factory(self._remote_channels, self._last_cursor)

    def _filter_batch(self, batch: StreamBatch) -> StreamBatch | None:
        events = list(batch.events)
        if self._symbol_filter:
            events = [event for event in events if self._matches_symbol(event)]
        if not events and not batch.heartbeat:
            return None
        return StreamBatch(
            channel=batch.channel,
            events=tuple(events),
            received_at=batch.received_at,
            cursor=batch.cursor,
            heartbeat=batch.heartbeat,
            raw=batch.raw,
        )

    def _record_yield(self, batch: StreamBatch) -> None:
        self._total_batches += 1
        self._total_events += len(batch.events)
        if batch.heartbeat:
            self._heartbeats += 1
            self._metric_heartbeats.inc(labels=self._metric_labels)
        if batch.events:
            self._metric_events.inc(len(batch.events), labels=self._metric_labels)
        self._metric_batches.inc(labels=self._metric_labels)
        self._update_queue_metrics()

    def _update_queue_metrics(self) -> None:
        self._metric_pending.set(len(self._pending), labels=self._metric_labels)
        self._metric_history.set(len(self._history), labels=self._metric_labels)

    def _matches_symbol(self, event: Mapping[str, Any]) -> bool:
        if not isinstance(event, Mapping):
            return False
        symbol = None
        for key in ("symbol", "Symbol", "pair", "Pair", "instrument", "s"):
            value = event.get(key)
            if value is None:
                continue
            symbol = self._normalize_symbol(value)
            if symbol:
                break
        if not symbol:
            return False
        return symbol in self._symbol_filter

    @staticmethod
    def _normalize_symbol(value: object) -> str:
        if value is None:
            return ""
        normalized = str(value).strip()
        if not normalized:
            return ""
        normalized = normalized.replace("-", "_").upper()
        return normalized


class NowaGieldaSpotAdapter(ExchangeAdapter):
    """Adapter implementujący podstawowe operacje dla nowa_gielda."""

    __slots__ = (
        "_environment",
        "_ip_allowlist",
        "_metrics",
        "_http_client",
        "_settings",
        "_permission_set",
    )

    name: str = "nowa_gielda_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._ip_allowlist: tuple[str, ...] = ()
        self._metrics: MetricsRegistry = metrics_registry or get_global_metrics_registry()
        self._http_client = NowaGieldaHTTPClient(self._determine_base_url(self._environment))
        self._settings = dict(settings or {})
        self._permission_set = frozenset(perm.lower() for perm in self.credentials.permissions)

    # --- Utilities ---------------------------------------------------------
    @staticmethod
    def _timestamp() -> int:
        return int(time.time() * 1000)

    def _secret(self) -> bytes:
        secret = self.credentials.secret or ""
        return secret.encode("utf-8")

    # --- Configuration -----------------------------------------------------
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        self._ip_allowlist = tuple(ip_allowlist or ())
        if self._ip_allowlist:
            _LOGGER.debug("Skonfigurowano allowlistę IP: %s", self._ip_allowlist)

    # --- Streaming configuration -------------------------------------------
    def _stream_settings(self) -> Mapping[str, Any]:
        raw = self._settings.get("stream")
        if isinstance(raw, Mapping):
            return raw
        return {}

    def _stream_symbol_filter(self, scope: str) -> tuple[str, ...]:
        settings = self._stream_settings()
        scope_key = f"{scope}_symbols"
        raw = settings.get(scope_key, settings.get("symbols"))
        if raw is None:
            return ()
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, Iterable):
            values = list(raw)
        else:
            return ()
        normalized: list[str] = []
        for entry in values:
            symbol = NowaGieldaStreamClient._normalize_symbol(entry)
            if symbol and symbol not in normalized:
                normalized.append(symbol)
        return tuple(normalized)

    def _stream_reconnect_config(self, scope: str) -> tuple[int, float, float, int]:
        settings = self._stream_settings()
        attempts = int(
            settings.get(f"{scope}_reconnect_attempts", settings.get("reconnect_attempts", 3))
        )
        backoff_base = float(
            settings.get(f"{scope}_reconnect_backoff", settings.get("reconnect_backoff", 0.5))
        )
        backoff_cap = float(
            settings.get(
                f"{scope}_reconnect_backoff_cap",
                settings.get("reconnect_backoff_cap", max(backoff_base, 5.0)),
            )
        )
        buffer_size = int(settings.get(f"{scope}_buffer_size", settings.get("buffer_size", 8)))
        return max(1, attempts), max(0.0, backoff_base), max(backoff_base, backoff_cap), max(1, buffer_size)

    def _build_stream(
        self,
        scope: str,
        channels: Sequence[str],
        *,
        initial_cursor: str | None = None,
    ) -> LocalLongPollStream:
        stream_settings = dict(self._stream_settings())
        base_url = str(
            stream_settings.get("base_url", self._settings.get("stream_base_url", "http://127.0.0.1:8765"))
        )
        default_path = f"/stream/{self.name}/{scope}"
        path = str(
            stream_settings.get(
                f"{scope}_path",
                self._settings.get(f"stream_{scope}_path", default_path),
            )
            or default_path
        )
        poll_interval = float(stream_settings.get("poll_interval", 0.5))
        timeout = float(stream_settings.get("timeout", 10.0))
        # Long-pollowe ponawianie po stronie LocalLongPollStream ograniczamy do jednej próby.
        # Dzięki temu logika reconnectów i buforowania pozostaje w NowaGieldaStreamClient,
        # który jest w stanie odtworzyć dane z historii przed pobraniem świeżych paczek.
        max_retries = 1
        backoff_base = float(stream_settings.get("backoff_base", 0.25))
        backoff_cap = float(stream_settings.get("backoff_cap", 2.0))
        jitter = stream_settings.get("jitter", (0.05, 0.30))
        channel_param = stream_settings.get(f"{scope}_channel_param")
        if channel_param is None:
            channel_param = stream_settings.get("channel_param", "channels")
        cursor_param = stream_settings.get(f"{scope}_cursor_param")
        if cursor_param is None:
            cursor_param = stream_settings.get("cursor_param", "cursor")
        initial_cursor_value = initial_cursor
        if initial_cursor_value is None:
            initial_cursor_value = stream_settings.get(f"{scope}_initial_cursor")
        channel_serializer = None
        serializer_candidate = stream_settings.get(f"{scope}_channel_serializer")
        if not callable(serializer_candidate):
            serializer_candidate = stream_settings.get("channel_serializer")
        if callable(serializer_candidate):
            channel_serializer = serializer_candidate
        else:
            separator = stream_settings.get(f"{scope}_channel_separator")
            if separator is None:
                separator = stream_settings.get("channel_separator", ",")
            if isinstance(separator, str):
                channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731
        headers_raw = stream_settings.get("headers")
        header_map = dict(headers_raw) if isinstance(headers_raw, Mapping) else None
        params: dict[str, object] = {}
        base_params = stream_settings.get("params")
        if isinstance(base_params, Mapping):
            params.update(base_params)
        scope_params = stream_settings.get(f"{scope}_params")
        if isinstance(scope_params, Mapping):
            params.update(scope_params)
        token_key = f"{scope}_token"
        if isinstance(stream_settings.get(token_key), str):
            params.setdefault("token", stream_settings[token_key])
        elif isinstance(stream_settings.get("auth_token"), str):
            params.setdefault("token", stream_settings["auth_token"])
        http_method = stream_settings.get(f"{scope}_method", stream_settings.get("method", "GET"))
        params_in_body = bool(stream_settings.get(f"{scope}_params_in_body", stream_settings.get("params_in_body", False)))
        channels_in_body = bool(stream_settings.get(f"{scope}_channels_in_body", stream_settings.get("channels_in_body", False)))
        cursor_in_body = bool(stream_settings.get(f"{scope}_cursor_in_body", stream_settings.get("cursor_in_body", False)))
        body_params: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body_params.update(base_body)
        scope_body = stream_settings.get(f"{scope}_body_params")
        if isinstance(scope_body, Mapping):
            body_params.update(scope_body)
        body_encoder = stream_settings.get(f"{scope}_body_encoder", stream_settings.get("body_encoder"))
        buffer_size = int(
            stream_settings.get(
                f"{scope}_buffer_size", stream_settings.get("buffer_size", 64)
            )
        )

        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=self.name,
            scope=scope,
            environment=self._environment.value,
            params=params,
            headers=header_map,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter if isinstance(jitter, Sequence) else (0.05, 0.30),
            channel_param=str(channel_param).strip() if channel_param not in (None, "") else "",
            cursor_param=str(cursor_param).strip() if cursor_param not in (None, "") else "",
            initial_cursor=initial_cursor_value,
            channel_serializer=channel_serializer,
            http_method=str(http_method or "GET"),
            params_in_body=params_in_body,
            channels_in_body=channels_in_body,
            cursor_in_body=cursor_in_body,
            body_params=body_params or None,
            body_encoder=body_encoder,
            buffer_size=buffer_size,
            metrics_registry=self._metrics,
        )

    # --- ExchangeAdapter API -----------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:
        timestamp = self._timestamp()
        signature = self.sign_request(timestamp, "GET", "/private/account")
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_account(headers=headers)

        raw_balances = payload.get("balances", [])
        if not isinstance(raw_balances, Sequence):
            raise ExchangeAPIError(
                message="Niepoprawny format listy sald konta",
                status_code=200,
                payload=payload,
            )

        balances: dict[str, float] = {}
        for entry in raw_balances:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja salda ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            asset = entry.get("asset")
            total = entry.get("total")
            if not asset:
                raise ExchangeAPIError(
                    message="Brak identyfikatora waluty w saldzie",
                    status_code=200,
                    payload=payload,
                )
            try:
                balances[str(asset)] = float(total)
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Niepoprawna wartość salda",
                    status_code=200,
                    payload=payload,
                ) from exc

        def _float_field(name: str, default: float = 0.0) -> float:
            value = payload.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message=f"Niepoprawna wartość pola {name}",
                    status_code=200,
                    payload=payload,
                ) from exc

        return AccountSnapshot(
            balances=balances,
            total_equity=_float_field("totalEquity"),
            available_margin=_float_field("availableMargin"),
            maintenance_margin=_float_field("maintenanceMargin"),
        )

    def fetch_symbols(self) -> Iterable[str]:
        return symbols.supported_internal_symbols()

    # --- Market data -----------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Mapping[str, Any]:
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_ticker(exchange_symbol)
        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w odpowiedzi API nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )
        return {
            "symbol": symbol,
            "best_bid": float(payload["bestBid"]),
            "best_ask": float(payload["bestAsk"]),
            "last_price": float(payload["lastPrice"]),
            "timestamp": float(payload.get("timestamp", self._timestamp())),
        }

    def fetch_orderbook(self, symbol: str, depth: int = 50) -> Mapping[str, Any]:
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_orderbook(exchange_symbol, depth=depth)
        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w orderbooku nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )
        return payload

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_ohlcv(
            exchange_symbol,
            interval,
            start=start,
            end=end,
            limit=limit,
        )

        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w odpowiedzi OHLCV nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )

        raw_candles = payload.get("candles", [])
        if not isinstance(raw_candles, Sequence):
            raise ExchangeAPIError(
                message="Lista świec ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        candles: list[list[float]] = []
        for candle in raw_candles:
            if not isinstance(candle, Sequence) or len(candle) < 6:
                raise ExchangeAPIError(
                    message="Świeca ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            open_time, open_price, high_price, low_price, close_price, volume = candle[:6]
            try:
                candles.append(
                    [
                        float(open_time),
                        float(open_price),
                        float(high_price),
                        float(low_price),
                        float(close_price),
                        float(volume),
                    ]
                )
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Niepoprawne wartości liczby w świecy",
                    status_code=200,
                    payload=payload,
                ) from exc

        return candles

    def fetch_trades_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/trades",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_trades(headers=headers, params=params)

        raw_trades = payload.get("trades", [])
        if not isinstance(raw_trades, Sequence):
            raise ExchangeAPIError(
                message="Lista transakcji ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        trades: list[Mapping[str, Any]] = []
        for entry in raw_trades:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja historii transakcji ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Transakcja nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol transakcji nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            trade_id_raw = entry.get("tradeId") or entry.get("id")
            if trade_id_raw is None:
                raise ExchangeAPIError(
                    message="Transakcja nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId")
            side_raw = entry.get("side")
            price_raw = entry.get("price")
            qty_raw = entry.get("quantity") or entry.get("qty")
            fee_raw = entry.get("fee")
            timestamp_raw = entry.get("timestamp") or entry.get("time")

            try:
                trade = {
                    "trade_id": str(trade_id_raw),
                    "order_id": str(order_id_raw) if order_id_raw is not None else None,
                    "symbol": internal_symbol,
                    "side": str(side_raw) if side_raw is not None else "",
                    "price": float(price_raw),
                    "quantity": float(qty_raw),
                    "fee": float(fee_raw) if fee_raw is not None else None,
                    "timestamp": float(timestamp_raw),
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Transakcja zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            trades.append(trade)

        return trades

    def fetch_open_orders(
        self,
        *,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "limit": limit,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/orders",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_open_orders(headers=headers, params=params)

        raw_orders = payload.get("orders", [])
        if not isinstance(raw_orders, Sequence):
            raise ExchangeAPIError(
                message="Lista zleceń ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        orders: list[Mapping[str, Any]] = []
        for entry in raw_orders:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja zlecenia ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Zlecenie nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol zlecenia nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId") or entry.get("id")
            status_raw = entry.get("status")
            side_raw = entry.get("side")
            type_raw = entry.get("type")
            price_raw = entry.get("price")
            avg_price_raw = entry.get("avgPrice")
            quantity_raw = entry.get("quantity") or entry.get("qty")
            filled_raw = entry.get("filledQuantity") or entry.get("filled")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")

            if order_id_raw is None:
                raise ExchangeAPIError(
                    message="Zlecenie nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            try:
                order = {
                    "order_id": str(order_id_raw),
                    "symbol": internal_symbol,
                    "status": str(status_raw) if status_raw is not None else "",
                    "side": str(side_raw) if side_raw is not None else "",
                    "type": str(type_raw) if type_raw is not None else "",
                    "price": float(price_raw) if price_raw is not None else None,
                    "avg_price": float(avg_price_raw) if avg_price_raw is not None else None,
                    "quantity": float(quantity_raw),
                    "filled_quantity": float(filled_raw) if filled_raw is not None else 0.0,
                    "timestamp": float(timestamp_raw)
                    if timestamp_raw is not None
                    else float(timestamp),
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Zlecenie zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            orders.append(order)

        return orders

    def fetch_closed_orders(
        self,
        *,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "limit": limit,
                "start": start,
                "end": end,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/orders/history",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_order_history(headers=headers, params=params)

        raw_orders = payload.get("orders", [])
        if not isinstance(raw_orders, Sequence):
            raise ExchangeAPIError(
                message="Lista zamkniętych zleceń ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        orders: list[Mapping[str, Any]] = []
        for entry in raw_orders:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja zamkniętego zlecenia ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol zamkniętego zlecenia nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId") or entry.get("id")
            status_raw = entry.get("status")
            side_raw = entry.get("side")
            type_raw = entry.get("type")
            price_raw = entry.get("price")
            avg_price_raw = entry.get("avgPrice")
            quantity_raw = entry.get("quantity") or entry.get("qty")
            filled_raw = (
                entry.get("executedQuantity")
                or entry.get("filledQuantity")
                or entry.get("filled")
            )
            created_raw = entry.get("timestamp") or entry.get("createdAt")
            closed_raw = entry.get("closedAt") or entry.get("updatedAt")

            if order_id_raw is None:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            try:
                order = {
                    "order_id": str(order_id_raw),
                    "symbol": internal_symbol,
                    "status": str(status_raw) if status_raw is not None else "",
                    "side": str(side_raw) if side_raw is not None else "",
                    "type": str(type_raw) if type_raw is not None else "",
                    "price": float(price_raw) if price_raw is not None else None,
                    "avg_price": float(avg_price_raw) if avg_price_raw is not None else None,
                    "quantity": float(quantity_raw),
                    "filled_quantity": float(filled_raw) if filled_raw is not None else 0.0,
                    "timestamp": float(created_raw)
                    if created_raw is not None
                    else float(timestamp),
                    "closed_timestamp": float(closed_raw)
                    if closed_raw is not None
                    else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            orders.append(order)

        return orders

    def place_order(self, request: OrderRequest) -> OrderResult:
        payload = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(request.symbol),
                "side": request.side,
                "type": request.order_type,
                "quantity": request.quantity,
                "price": request.price,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(timestamp, "POST", "/private/orders", body=payload)
        headers = self.build_auth_headers(timestamp, signature)
        _LOGGER.debug("Składanie zlecenia %s z nagłówkami %s", payload, headers)
        response = self._http_client.create_order(payload, headers=headers)
        order_id = str(response["orderId"])
        status = str(response.get("status", "accepted"))
        filled_qty = float(response.get("filledQuantity", 0.0))
        avg_price_raw = response.get("avgPrice")
        avg_price = float(avg_price_raw) if avg_price_raw is not None else None
        return OrderResult(
            order_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_price=avg_price,
            raw_response=response,
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        _LOGGER.debug("Anulowanie zlecenia %s dla symbolu %s", order_id, symbol)
        timestamp = self._timestamp()
        exchange_symbol = symbols.to_exchange_symbol(symbol) if symbol else None
        params = _strip_none({"orderId": order_id, "symbol": exchange_symbol})
        signature = self.sign_request(
            timestamp,
            "DELETE",
            "/private/orders",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        self._http_client.cancel_order(order_id, headers=headers, symbol=exchange_symbol)

    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        symbol_filter = self._stream_symbol_filter("public")
        attempts, backoff_base, backoff_cap, buffer_size = self._stream_reconnect_config("public")

        def factory(mapped_channels: Sequence[str], cursor: Optional[str]) -> LocalLongPollStream:
            return self._build_stream("public", mapped_channels, initial_cursor=cursor)

        return NowaGieldaStreamClient(
            adapter=self.name,
            scope="public",
            channels=channels,
            fallback_factory=factory,
            channel_mapping=_PUBLIC_STREAM_CHANNEL_MAP,
            symbols=symbol_filter,
            max_reconnects=attempts,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            buffer_size=buffer_size,
            metrics_registry=self._metrics,
        )

    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        if "trade" not in self._permission_set and "write" not in self._permission_set:
            raise PermissionError("Poświadczenia nie pozwalają na prywatny stream nowa_gielda.")

        symbol_filter = self._stream_symbol_filter("private")
        attempts, backoff_base, backoff_cap, buffer_size = self._stream_reconnect_config("private")

        def factory(mapped_channels: Sequence[str], cursor: Optional[str]) -> LocalLongPollStream:
            return self._build_stream("private", mapped_channels, initial_cursor=cursor)

        return NowaGieldaStreamClient(
            adapter=self.name,
            scope="private",
            channels=channels,
            fallback_factory=factory,
            channel_mapping=_PRIVATE_STREAM_CHANNEL_MAP,
            symbols=symbol_filter,
            max_reconnects=attempts,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            buffer_size=buffer_size,
            metrics_registry=self._metrics,
        )

    # --- Custom helpers ----------------------------------------------------
    def sign_request(
        self,
        timestamp: int,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        suffix_parts: list[str] = []
        canonical_params = _canonical_payload(_strip_none(params))
        canonical_body = _canonical_payload(_strip_none(body))
        if canonical_params:
            suffix_parts.append(f"P:{canonical_params}")
        if canonical_body:
            suffix_parts.append(f"B:{canonical_body}")
        suffix = "|".join(suffix_parts)
        message = f"{timestamp}{method.upper()}{path}{suffix}".encode("utf-8")
        return hmac.new(self._secret(), message, sha256).hexdigest()

    def build_auth_headers(self, timestamp: int, signature: str) -> Mapping[str, str]:
        return {
            "X-API-KEY": self.credentials.key_id,
            "X-API-SIGN": signature,
            "X-API-TIMESTAMP": str(timestamp),
        }

    def _determine_base_url(self, environment: Environment) -> str:
        if environment is Environment.LIVE:
            return "https://api.nowa-gielda.example"
        if environment is Environment.PAPER:
            return "https://paper.nowa-gielda.example"
        return "https://testnet.nowa-gielda.example"

    def rate_limit_rule(self, method: str, path: str) -> RateLimitRule | None:
        key = f"{method.upper()} {path}".strip()
        return _RATE_LIMITS.get(key)

    def request_weight(self, method: str, path: str) -> int:
        rule = self.rate_limit_rule(method, path)
        return rule.weight if rule else 1


__all__ = ["NowaGieldaSpotAdapter", "RateLimitRule", "NowaGieldaStreamClient"]
