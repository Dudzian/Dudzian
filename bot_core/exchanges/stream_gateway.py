"""Serwer HTTP udostępniający lokalny stream long-pollowy dla adapterów giełdowych."""
from __future__ import annotations

import copy
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
    payload: tuple[Mapping[str, Any], ...]
    updated_at: float


class StreamGateway:
    """Warstwa logiki odpowiedzialna za obsługę streamów long-pollowych."""

    def __init__(
        self,
        *,
        retry_after: float = 0.5,
        default_depth: int = 50,
        state_ttl: float | None = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._retry_after = max(0.0, float(retry_after))
        self._default_depth = max(1, int(default_depth))
        if state_ttl is None:
            self._state_ttl: float | None = None
        else:
            try:
                ttl_value = float(state_ttl)
            except (TypeError, ValueError) as exc:
                raise ValueError("state_ttl musi być liczbą rzeczywistą") from exc
            self._state_ttl = ttl_value if ttl_value > 0 else None
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
        latest_cursor = cursor
        now = self._clock()
        self._prune_stale_states(now)

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
            canonical, normalized_events = self._canonicalize_events(events)
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
                        payload=normalized_events,
                        updated_at=now,
                    )
                    batch_events = [copy.deepcopy(event) for event in normalized_events]
                    heartbeat = False
                else:
                    assert state is not None  # dla mypy
                    batch_cursor = state.cursor
                    state.updated_at = now
                    if cursor is None or cursor != state.cursor:
                        batch_events = [copy.deepcopy(event) for event in state.payload]
                        heartbeat = False
                    else:
                        batch_events = []
                        heartbeat = True

            latest_cursor = self._prefer_newer_cursor(latest_cursor, batch_cursor)
            batches.append(
                {
                    "channel": normalized_channel,
                    "events": batch_events,
                    "cursor": batch_cursor,
                    "heartbeat": heartbeat,
                }
            )

        response_cursor = latest_cursor
        if response_cursor is None and batches:
            response_cursor = batches[-1]["cursor"]

        return {
            "cursor": response_cursor,
            "batches": batches,
            "retry_after": self._retry_after,
        }

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

    def _prefer_newer_cursor(self, current: str | None, candidate: str | None) -> str | None:
        if candidate in (None, ""):
            return current
        if current in (None, ""):
            return candidate
        current_key = self._cursor_sort_key(str(current))
        candidate_key = self._cursor_sort_key(str(candidate))
        if candidate_key >= current_key:
            return candidate
        return current

    @staticmethod
    def _cursor_sort_key(cursor: str) -> tuple[int, int]:
        if not cursor:
            return (0, 0)
        try:
            timestamp_part, version_part = cursor.split("-", 1)
            return (int(timestamp_part), int(version_part))
        except (ValueError, TypeError):  # pragma: no cover - niepoprawny format kursora
            return (0, 0)

    def _prune_stale_states(self, now: float) -> None:
        ttl = self._state_ttl
        if ttl is None:
            return
        threshold = now - ttl
        with self._lock:
            stale_keys = [key for key, state in self._channels.items() if state.updated_at < threshold]
            if not stale_keys:
                return
            for key in stale_keys:
                self._channels.pop(key, None)

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

    def _canonicalize_events(
        self, events: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
        """Buduje kanoniczną reprezentację i kopię zdarzeń kanału."""

        canonical: list[str] = []
        normalized: list[Mapping[str, Any]] = []
        for event in events:
            normalized_event = self._normalize_event(event)
            serialized = json.dumps(
                normalized_event,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            canonical.append(serialized)
            normalized.append(normalized_event)
        return tuple(canonical), tuple(normalized)

    def _normalize_event(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        """Zwraca głęboką kopię zdarzenia w postaci słownika."""

        if isinstance(event, Mapping):
            return copy.deepcopy(dict(event))
        return copy.deepcopy(dict(self._to_mapping(event)))

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
        self._handle_stream_request()

    def do_POST(self) -> None:  # noqa: D401,N802 - API BaseHTTPRequestHandler
        self._handle_stream_request()

    def _handle_stream_request(self) -> None:
        server = self.server  # type: ignore[assignment]
        assert isinstance(server, StreamGatewayHTTPServer)
        parsed = urlsplit(self.path)
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) != 3 or path_parts[0] != "stream":
            self._send_json({"error": {"message": "nieprawidłowa ścieżka"}}, HTTPStatus.NOT_FOUND)
            return

        adapter_name, scope = path_parts[1], path_parts[2]
        try:
            body_params = self._parse_body_params()
        except StreamGatewayError as exc:
            self._send_json({"error": {"message": str(exc)}}, exc.status)
            return

        query = self._merge_params(
            parse_qs(parsed.query, keep_blank_values=False),
            body_params,
        )
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

        try:
            response = server.gateway.handle_request(
                adapter_name=adapter_name,
                environment=environment,
                scope=scope,
                channels=channels,
                params=query,
                cursor=cursor,
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

    def _parse_body_params(self) -> MutableMapping[str, list[str]]:
        if self.command not in {"POST", "PUT", "PATCH"}:
            return {}

        length_header = self.headers.get("Content-Length")
        if not length_header:
            return {}
        try:
            length = int(length_header)
        except (TypeError, ValueError):
            raise StreamGatewayError("Nagłówek Content-Length ma niepoprawną wartość")
        if length <= 0:
            return {}

        raw_body = self.rfile.read(length)
        if not raw_body:
            return {}

        content_type = self.headers.get("Content-Type", "")
        mime, charset = self._split_content_type(content_type)

        try:
            decoded_body = raw_body.decode(charset)
        except LookupError as exc:
            raise StreamGatewayError("Nieobsługiwane kodowanie znaków w treści żądania") from exc
        except UnicodeDecodeError as exc:
            raise StreamGatewayError("Treść żądania nie jest poprawnie zakodowana") from exc

        if mime in {"", "application/x-www-form-urlencoded"}:
            return parse_qs(decoded_body, keep_blank_values=False)

        if mime == "application/json":
            try:
                payload = json.loads(decoded_body)
            except json.JSONDecodeError as exc:
                raise StreamGatewayError("Treść żądania JSON jest niepoprawna") from exc
            if not isinstance(payload, Mapping):
                raise StreamGatewayError("Treść żądania JSON musi być obiektem")

            normalized: dict[str, list[str]] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                flattened = self._flatten_body_value(value)
                if flattened:
                    normalized.setdefault(key, []).extend(flattened)
            return normalized

        raise StreamGatewayError("Nieobsługiwany typ treści żądania", status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    def _merge_params(
        self,
        *sources: Mapping[str, Sequence[str]] | None,
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for source in sources:
            if not source:
                continue
            for key, values in source.items():
                if not key:
                    continue
                bucket = merged.setdefault(key, [])
                for value in values:
                    if value in (None, ""):
                        continue
                    bucket.append(str(value))
        return merged

    def _split_content_type(self, header_value: str) -> tuple[str, str]:
        mime = ""
        charset = "utf-8"
        if header_value:
            parts = [part.strip() for part in header_value.split(";") if part.strip()]
            if parts:
                mime = parts[0].lower()
                for part in parts[1:]:
                    if part.lower().startswith("charset="):
                        candidate = part.split("=", 1)[1].strip()
                        if candidate:
                            charset = candidate
                            break
        return mime, charset

    def _flatten_body_value(self, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, Mapping):
            try:
                serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                serialized = str(value)
            return [serialized]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_body_value(item))
            return flattened
        return [str(value)]

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
