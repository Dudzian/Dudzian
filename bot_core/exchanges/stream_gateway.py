"""Serwer HTTP udostępniający lokalny stream long-pollowy dla adapterów giełdowych."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, is_dataclass, asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.parse import parse_qs, urlsplit

from bot_core.exchanges.base import ExchangeAdapter
from bot_core.exchanges.errors import ExchangeError

_LOGGER = logging.getLogger(__name__)


class StreamGatewayError(RuntimeError):
    """Wyjątek zgłaszany przy niepoprawnych żądaniach do gateway'a."""

    def __init__(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class _ChannelState:
    """Przechowuje ostatni stan kanału streamingu."""

    cursor: str
    version: int
    canonical: tuple[str, ...]
    updated_at: float


class StreamGateway:
    """Warstwa logiki odpowiedzialna za obsługę streamów long-pollowych."""

    def __init__(
        self,
        *,
        retry_after: float = 0.5,
        default_depth: int = 50,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._retry_after = max(0.0, float(retry_after))
        self._default_depth = max(1, int(default_depth))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._adapters: dict[tuple[str, str | None], ExchangeAdapter] = {}
        self._channels: dict[tuple[Any, ...], _ChannelState] = {}

    # ------------------------------------------------------------------
    # Rejestracja adapterów
    # ------------------------------------------------------------------
    def register_adapter(
        self,
        adapter_name: str,
        *,
        environment: str | None,
        adapter: ExchangeAdapter,
    ) -> None:
        """Rejestruje instancję adaptera obsługiwaną przez gateway."""

        if not adapter_name:
            raise ValueError("Nazwa adaptera nie może być pusta")
        env_key = environment.lower() if isinstance(environment, str) else None
        key = (adapter_name, env_key)
        with self._lock:
            self._adapters[key] = adapter

    # ------------------------------------------------------------------
    # Obsługa żądań
    # ------------------------------------------------------------------
    def handle_request(
        self,
        *,
        adapter_name: str,
        environment: str | None,
        scope: str,
        channels: Sequence[str],
        params: Mapping[str, Sequence[str]],
        cursor: str | None,
        reset: bool = False,
    ) -> Mapping[str, Any]:
        """Buduje odpowiedź JSON dla żądania long-pollowego."""

        if not channels:
            raise StreamGatewayError("Parametr kanałów nie może być pusty")

        normalized_scope = scope.strip().lower()
        if normalized_scope not in {"public", "private"}:
            raise StreamGatewayError("Nieobsługiwany zakres streamu", status=HTTPStatus.NOT_FOUND)

        adapter = self._resolve_adapter(adapter_name, environment)
        normalized_params = {
            str(key): tuple(str(value) for value in values if value not in (None, ""))
            for key, values in params.items()
            if values
        }

        batches: list[dict[str, Any]] = []
        response_cursor = cursor
        now = self._clock()

        reset_performed = False
        for channel in channels:
            normalized_channel = channel.strip()
            if not normalized_channel:
                raise StreamGatewayError("Kanał streamu nie może być pusty")

            events = self._collect_events(
                adapter,
                normalized_scope,
                normalized_channel,
                normalized_params,
            )
            if reset:
                reset_performed = (
                    self._reset_channel(
                        adapter_name,
                        environment,
                        normalized_scope,
                        normalized_channel,
                        normalized_params,
                    )
                    or reset_performed
                )
            canonical = tuple(
                json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                for event in events
            )
            state_key = self._build_state_key(
                adapter_name,
                environment,
                normalized_scope,
                normalized_channel,
                normalized_params,
            )
            with self._lock:
                state = self._channels.get(state_key)
                changed = state is None or state.canonical != canonical
                if changed:
                    version = 1 if state is None else state.version + 1
                    batch_cursor = self._generate_cursor(version, now)
                    self._channels[state_key] = _ChannelState(
                        cursor=batch_cursor,
                        version=version,
                        canonical=canonical,
                        updated_at=now,
                    )
                    batch_events = events
                    heartbeat = False
                    response_cursor = batch_cursor
                else:
                    batch_cursor = state.cursor
                    if cursor is None or cursor != state.cursor:
                        batch_events = [json.loads(item) for item in state.canonical]
                        heartbeat = False
                    else:
                        batch_events = []
                        heartbeat = True
                    response_cursor = state.cursor

            batches.append(
                {
                    "channel": normalized_channel,
                    "events": batch_events,
                    "cursor": batch_cursor,
                    "heartbeat": heartbeat,
                }
            )

        if response_cursor is None and batches:
            response_cursor = batches[-1]["cursor"]

        payload: dict[str, Any] = {
            "cursor": response_cursor,
            "batches": batches,
            "retry_after": self._retry_after,
        }
        if reset_performed:
            payload["reset"] = True
        return payload

    def status_snapshot(self) -> Mapping[str, Any]:
        """Zwraca stan zarejestrowanych kanałów i adapterów."""

        with self._lock:
            now = self._clock()
            adapters = [
                {
                    "adapter": adapter_name,
                    "environment": environment,
                }
                for adapter_name, environment in self._adapters.keys()
            ]
            channels: list[Mapping[str, Any]] = []
            for key, state in self._channels.items():
                adapter_name, environment, scope, channel, params = key
                param_map = {
                    str(param_key): [*values]
                    for param_key, values in params
                }
                channels.append(
                    {
                        "adapter": adapter_name,
                        "environment": environment,
                        "scope": scope,
                        "channel": channel,
                        "params": param_map,
                        "cursor": state.cursor,
                        "version": state.version,
                        "age_seconds": max(0.0, now - state.updated_at),
                    }
                )

        return {"adapters": adapters, "channels": channels}

    def reset_channels(
        self,
        *,
        adapter_name: str,
        environment: str | None,
        scope: str,
        channels: Sequence[str],
        params: Mapping[str, Sequence[str]],
    ) -> bool:
        """Czyści bufory dla wybranych kanałów (wymusza restart kursora)."""

        normalized_params = {
            str(key): tuple(str(value) for value in values)
            for key, values in params.items()
        }
        removed = False
        for channel in channels:
            normalized_channel = channel.strip()
            if not normalized_channel:
                continue
            removed = (
                self._reset_channel(
                    adapter_name,
                    environment,
                    scope.strip().lower(),
                    normalized_channel,
                    normalized_params,
                )
                or removed
            )
        return removed

    # ------------------------------------------------------------------
    # Zamykanie zasobów
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Zamyka wszystkie zarejestrowane adaptery, jeśli obsługują metodę `close`."""

        with self._lock:
            adapters = list(self._adapters.values())
            self._adapters.clear()
            self._channels.clear()
        for adapter in adapters:
            closer = getattr(adapter, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - zamykanie jest best-effort
                    _LOGGER.debug("Błąd podczas zamykania adaptera %s", adapter, exc_info=True)

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia
    # ------------------------------------------------------------------
    def _resolve_adapter(self, adapter_name: str, environment: str | None) -> ExchangeAdapter:
        name = str(adapter_name)
        env_key = environment.lower() if isinstance(environment, str) else None
        with self._lock:
            for candidate in ((name, env_key), (name, None)):
                adapter = self._adapters.get(candidate)
                if adapter is not None:
                    return adapter
        raise StreamGatewayError(
            f"Brak zarejestrowanego adaptera '{name}' dla środowiska '{environment}'.",
            status=HTTPStatus.NOT_FOUND,
        )

    def _build_state_key(
        self,
        adapter_name: str,
        environment: str | None,
        scope: str,
        channel: str,
        params: Mapping[str, Sequence[str]],
    ) -> tuple[Any, ...]:
        env_key = environment.lower() if isinstance(environment, str) else None
        normalized_params: list[tuple[str, tuple[str, ...]]] = []
        for key, values in params.items():
            if key in {"exchange", "environment", "scope"}:
                continue
            normalized_params.append((str(key), tuple(values)))
        normalized_params.sort()
        return (adapter_name, env_key, scope, channel, tuple(normalized_params))

    def _generate_cursor(self, version: int, now: float) -> str:
        timestamp_ms = int(now * 1000)
        return f"{timestamp_ms}-{version}"

    def _collect_events(
        self,
        adapter: ExchangeAdapter,
        scope: str,
        channel: str,
        params: Mapping[str, Sequence[str]],
    ) -> list[Mapping[str, Any]]:
        if scope == "public":
            return self._collect_public_events(adapter, channel, params)
        return self._collect_private_events(adapter, channel, params)

    def _reset_channel(
        self,
        adapter_name: str,
        environment: str | None,
        scope: str,
        channel: str,
        params: Mapping[str, Sequence[str]],
    ) -> bool:
        key = self._build_state_key(adapter_name, environment, scope, channel, params)
        with self._lock:
            return self._channels.pop(key, None) is not None

    def _collect_public_events(
        self,
        adapter: ExchangeAdapter,
        channel: str,
        params: Mapping[str, Sequence[str]],
    ) -> list[Mapping[str, Any]]:
        if channel.lower() in {"ticker", "tickers"}:
            symbols = self._extract_symbols(params)
            fetcher = getattr(adapter, "fetch_ticker", None)
            if not callable(fetcher):
                raise StreamGatewayError("Adapter nie obsługuje kanału ticker")
            events: list[Mapping[str, Any]] = []
            for symbol in symbols:
                result = fetcher(symbol)
                events.append(self._to_mapping(result))
            return events

        if channel.lower() in {"depth", "order_book", "orderbook"}:
            symbol = self._extract_single_symbol(params)
            depth = self._extract_depth(params)
            fetcher = getattr(adapter, "fetch_order_book", None)
            if not callable(fetcher):
                raise StreamGatewayError("Adapter nie obsługuje kanału depth")
            result = fetcher(symbol, depth=depth)
            return [self._to_mapping(result)]

        raise StreamGatewayError(f"Nieobsługiwany kanał publiczny '{channel}'", status=HTTPStatus.NOT_FOUND)

    def _collect_private_events(
        self,
        adapter: ExchangeAdapter,
        channel: str,
        params: Mapping[str, Sequence[str]],
    ) -> list[Mapping[str, Any]]:
        lowered = channel.lower()
        if lowered in {"orders", "open_orders", "openorders"}:
            fetcher = getattr(adapter, "fetch_open_orders", None)
            if not callable(fetcher):
                raise StreamGatewayError("Adapter nie obsługuje kanału orders")
            raw_orders = fetcher()
            events: list[Mapping[str, Any]] = []
            if isinstance(raw_orders, Sequence):
                for entry in raw_orders:
                    events.append(self._to_mapping(entry))
            return events

        if lowered in {"fills", "trades", "executions"}:
            fetcher = getattr(adapter, "fetch_recent_fills", None)
            if not callable(fetcher):
                fetcher = getattr(adapter, "fetch_recent_trades", None)
            if not callable(fetcher):
                raise StreamGatewayError("Adapter nie obsługuje kanału fills")
            symbols = self._maybe_extract_symbols(params)
            limit = self._extract_limit(params, default=50)
            from_id = self._extract_int_param(params, ("from_id", "fromId", "cursor"), minimum=0)

            def _invoke(symbol_value: str | None) -> Sequence[Mapping[str, Any]]:
                kwargs: dict[str, object] = {}
                if symbol_value is not None:
                    kwargs["symbol"] = symbol_value
                if limit is not None:
                    kwargs["limit"] = limit
                if from_id is not None:
                    kwargs["from_id"] = from_id
                try:
                    result = fetcher(**kwargs)  # type: ignore[call-arg]
                except TypeError:
                    kwargs.pop("from_id", None)
                    result = fetcher(**kwargs)  # type: ignore[call-arg]
                if isinstance(result, Sequence):
                    return [self._to_mapping(item) for item in result]
                if result is None:
                    return []
                return [self._to_mapping(result)]

            aggregated: list[Mapping[str, Any]] = []
            if symbols:
                for symbol_value in symbols:
                    aggregated.extend(_invoke(symbol_value))
            else:
                aggregated.extend(_invoke(None))
            return aggregated

        if lowered in {"balances", "account"}:
            snapshot = adapter.fetch_account_snapshot()
            return [self._to_mapping(snapshot)]

        raise StreamGatewayError(f"Nieobsługiwany kanał prywatny '{channel}'", status=HTTPStatus.NOT_FOUND)

    # ------------------------------------------------------------------
    # Parsowanie parametrów
    # ------------------------------------------------------------------
    def _extract_symbols(self, params: Mapping[str, Sequence[str]]) -> list[str]:
        for key in ("symbols", "symbol", "pairs", "pair", "instrument"):
            values = params.get(key)
            if not values:
                continue
            symbols: list[str] = []
            for value in values:
                for token in str(value).split(","):
                    token = token.strip()
                    if token:
                        symbols.append(token)
            if symbols:
                return symbols
        raise StreamGatewayError("Parametr symbol/symbols jest wymagany dla kanału ticker")

    def _extract_single_symbol(self, params: Mapping[str, Sequence[str]]) -> str:
        symbols = self._extract_symbols(params)
        if not symbols:
            raise StreamGatewayError("Parametr symbol jest wymagany")
        return symbols[0]

    def _extract_depth(self, params: Mapping[str, Sequence[str]]) -> int:
        for key in ("depth", "limit"):
            values = params.get(key)
            if not values:
                continue
            candidate = values[-1]
            try:
                depth = int(candidate)
            except (TypeError, ValueError) as exc:
                raise StreamGatewayError("Parametr depth musi być liczbą całkowitą") from exc
            if depth <= 0:
                raise StreamGatewayError("Parametr depth musi być dodatni")
            return depth
        return self._default_depth

    def _extract_limit(self, params: Mapping[str, Sequence[str]], *, default: int) -> int:
        value = self._extract_int_param(params, ("limit",), minimum=1)
        return value if value is not None else default

    def _extract_int_param(
        self,
        params: Mapping[str, Sequence[str]],
        keys: Sequence[str],
        *,
        minimum: int | None = None,
    ) -> int | None:
        for key in keys:
            values = params.get(key)
            if not values:
                continue
            candidate = values[-1]
            try:
                numeric = int(candidate)
            except (TypeError, ValueError) as exc:
                raise StreamGatewayError("Parametr musi być liczbą całkowitą") from exc
            if minimum is not None and numeric < minimum:
                raise StreamGatewayError("Parametr musi być nieujemny")
            return numeric
        return None

    def _maybe_extract_symbols(self, params: Mapping[str, Sequence[str]]) -> list[str]:
        symbols: list[str] = []
        for key in ("symbols", "symbol", "pairs", "pair", "instrument"):
            values = params.get(key)
            if not values:
                continue
            for value in values:
                for token in str(value).split(","):
                    token = token.strip()
                    if token and token not in symbols:
                        symbols.append(token)
        return symbols

    def _to_mapping(self, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "_asdict"):
            return dict(value._asdict())  # type: ignore[attr-defined]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {"values": list(value)}
        return {"value": value}


class _StreamRequestHandler(BaseHTTPRequestHandler):
    """Obsługuje zapytania HTTP kierowane do StreamGateway."""

    server_version = "StreamGateway/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - API bazowej klasy
        _LOGGER.info("STREAM %s - %s", self.client_address[0], format % args)

    def do_GET(self) -> None:  # noqa: D401,N802 - API BaseHTTPRequestHandler
        server = self.server  # type: ignore[assignment]
        assert isinstance(server, StreamGatewayHTTPServer)
        parsed = urlsplit(self.path)
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts == ["stream", "status"]:
            snapshot = server.gateway.status_snapshot()
            self._send_json(snapshot, HTTPStatus.OK)
            return

        if len(path_parts) != 3 or path_parts[0] != "stream":
            self._send_json({"error": {"message": "nieprawidłowa ścieżka"}}, HTTPStatus.NOT_FOUND)
            return

        adapter_name, scope = path_parts[1], path_parts[2]
        query = parse_qs(parsed.query, keep_blank_values=False)
        channels, channel_key = self._extract_channels(query)
        if channel_key:
            query.pop(channel_key, None)
        cursor, cursor_key = self._extract_cursor(query)
        if cursor_key:
            query.pop(cursor_key, None)
        environment = self._first_param(query, ("environment", "env"))
        if environment is not None:
            query.pop("environment", None)
            query.pop("env", None)
        query.pop("exchange", None)
        query.pop("scope", None)

        reset_flag = False
        reset_param = self._first_param(query, ("reset", "action"))
        if reset_param is not None:
            if reset_param.lower() in {"1", "true", "yes", "reset"}:
                reset_flag = True
            query.pop("reset", None)
            if reset_param.lower() == "reset":
                query.pop("action", None)

        try:
            response = server.gateway.handle_request(
                adapter_name=adapter_name,
                environment=environment,
                scope=scope,
                channels=channels,
                params=query,
                cursor=cursor,
                reset=reset_flag,
            )
        except StreamGatewayError as exc:
            self._send_json({"error": {"message": str(exc)}}, exc.status)
            return
        except ExchangeError as exc:  # pragma: no cover - zależy od adapterów
            _LOGGER.debug("Błąd adaptera w streamie %s/%s: %s", adapter_name, scope, exc, exc_info=True)
            self._send_json({"error": {"message": str(exc)}}, HTTPStatus.BAD_GATEWAY)
            return
        except Exception as exc:  # pragma: no cover - bezpieczeństwo
            _LOGGER.exception("Nieoczekiwany wyjątek stream gateway", exc_info=exc)
            self._send_json({"error": {"message": "internal_error"}}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(response, HTTPStatus.OK)

    def do_HEAD(self) -> None:  # noqa: D401,N802
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Metoda HEAD nie jest wspierana")

    # ------------------------------------------------------------------
    # Pomocnicze metody parsera
    # ------------------------------------------------------------------
    def _extract_channels(
        self, params: MutableMapping[str, list[str]]
    ) -> tuple[list[str], str | None]:
        for key in ("channels", "channel", "topics", "topic"):
            values = params.get(key)
            if not values:
                continue
            channels: list[str] = []
            for raw in values:
                for token in str(raw).split(","):
                    token = token.strip()
                    if token:
                        channels.append(token)
            if channels:
                return channels, key
        raise StreamGatewayError("Parametr channels jest wymagany")

    def _extract_cursor(
        self, params: MutableMapping[str, list[str]]
    ) -> tuple[str | None, str | None]:
        for key in ("cursor", "position", "next_cursor", "nextCursor"):
            values = params.get(key)
            if not values:
                continue
            candidate = values[-1].strip()
            return (candidate or None, key)
        return None, None

    def _first_param(
        self, params: Mapping[str, list[str]], keys: Sequence[str]
    ) -> str | None:
        for key in keys:
            values = params.get(key)
            if values:
                return values[-1]
        return None

    def _send_json(self, payload: Mapping[str, Any], status: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


class StreamGatewayHTTPServer(ThreadingMixIn, HTTPServer):
    """Serwer HTTP obsługujący zapytania streamingu long-pollowego."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], *, gateway: StreamGateway) -> None:
        self.gateway = gateway
        super().__init__(server_address, _StreamRequestHandler)


def start_stream_gateway(
    host: str,
    port: int,
    *,
    gateway: StreamGateway,
) -> tuple[StreamGatewayHTTPServer, threading.Thread]:
    """Uruchamia gateway w tle i zwraca parę (server, thread)."""

    server = StreamGatewayHTTPServer((host, port), gateway=gateway)
    thread = threading.Thread(target=server.serve_forever, name="stream-gateway", daemon=True)
    thread.start()
    actual_host, actual_port = server.server_address  # type: ignore[misc]
    _LOGGER.info("Stream gateway nasłuchuje na %s:%s", actual_host, actual_port)
    return server, thread


__all__ = [
    "StreamGateway",
    "StreamGatewayError",
    "StreamGatewayHTTPServer",
    "start_stream_gateway",
]
