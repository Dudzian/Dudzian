"""Wspólne narzędzia do lokalnego streamingu opartego o long-polle REST/gRPC."""
from __future__ import annotations

import asyncio
import functools
import gzip
import json
import logging
import random
import threading
import time
import zlib
from collections import deque
from datetime import timezone
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request
from email.utils import parsedate_to_datetime

from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.http_client import urlopen
from bot_core.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
    get_global_metrics_registry,
)

try:  # pragma: no cover - sprawdzamy dostępność opcjonalnej zależności
    import brotli  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - brak brotli w środowisku testowym
    try:  # pragma: no cover - alternatywa dla środowisk bez brotli
        import brotlicffi as brotli  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - brak obu bibliotek
        brotli = None  # type: ignore[assignment]

try:  # pragma: no cover - zstd jest opcjonalny
    import zstandard  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - brak biblioteki zstd
    zstandard = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

_BASE_ACCEPT_ENCODING_VALUES = ("gzip", "deflate")

_CONTENT_ENCODING_ALIASES: Mapping[str, str] = {
    "x-gzip": "gzip",
    "x-deflate": "deflate",
    "x-brotli": "br",
    "x-zstd": "zstd",
    "zst": "zstd",
}


def _build_accept_encoding_values() -> list[str]:
    values = list(_BASE_ACCEPT_ENCODING_VALUES)
    if brotli is not None:
        values.append("br")
    if zstandard is not None:
        values.append("zstd")
    return values


def _build_default_headers() -> dict[str, str]:
    return {
        "User-Agent": "bot-core/stream/1.0",
        "Accept-Encoding": ", ".join(_build_accept_encoding_values()),
    }


_DEFAULT_HEADERS = _build_default_headers()


def _iter_header_entries(value: object) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, (bytes, bytearray)):
            try:
                return (value.decode("latin-1"),)
            except UnicodeDecodeError:
                return (value.decode("utf-8", errors="ignore"),)
        return (value,)
    if isinstance(value, Mapping):
        items: list[str] = []
        for item in value.values():
            items.extend(_iter_header_entries(item))
        return tuple(items)
    if isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            items.extend(_iter_header_entries(item))
        return tuple(items)
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            items.extend(_iter_header_entries(item))
        return tuple(items)
    return (str(value),)
_DEFAULT_POLL_INTERVAL = 0.5
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 0.25
_DEFAULT_BACKOFF_CAP = 2.0
_DEFAULT_JITTER = (0.05, 0.30)
_DEFAULT_BUFFER_SIZE = 64
_DEFAULT_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
_DEFAULT_LAG_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


@dataclass(slots=True)
class StreamBatch:
    """Pojedyncza paczka danych zwróconych przez stream long-pollowy."""

    channel: str
    events: Sequence[Mapping[str, Any]]
    received_at: float
    cursor: str | None = None
    heartbeat: bool = False
    raw: Mapping[str, Any] | None = None


class LocalLongPollStream(Iterable[StreamBatch]):
    """Iterator korzystający z lokalnego endpointu REST/gRPC opartego o long-polle."""

    def __init__(
        self,
        *,
        base_url: str,
        path: str,
        channels: Sequence[str],
        adapter: str,
        scope: str,
        environment: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_cap: float = _DEFAULT_BACKOFF_CAP,
        jitter: Sequence[float] = _DEFAULT_JITTER,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        channel_param: str | None = "channels",
        cursor_param: str | None = "cursor",
        initial_cursor: str | None = None,
        channel_serializer: Callable[[Sequence[str]], object] | None = None,
        http_method: str = "GET",
        params_in_body: bool = False,
        channels_in_body: bool = False,
        cursor_in_body: bool = False,
        body_params: Mapping[str, object] | None = None,
        body_encoder: str
        | Callable[[Mapping[str, object]], bytes | tuple[bytes, str | None]]
        | None = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
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
            raise ValueError("Lista kanałów streamu nie może być pusta.")

        self._base_url = base_url.rstrip("/") or "http://127.0.0.1:8765"
        self._path = path if path.startswith("/") else f"/{path}"
        self._channels = normalized_channels
        self._adapter = adapter
        self._scope = scope
        self._environment = environment
        self._params: MutableMapping[str, object] = {
            "exchange": adapter,
            "environment": environment,
            "scope": scope,
        }
        if params:
            for key, value in params.items():
                if value in (None, ""):
                    continue
                self._params[str(key)] = value

        self._headers: MutableMapping[str, str] = _build_default_headers()
        if headers:
            for key, value in headers.items():
                self._headers[str(key)] = str(value)

        self._poll_interval = max(0.0, float(poll_interval))
        self._timeout = max(0.1, float(timeout))
        self._max_retries = max(1, int(max_retries))
        self._backoff_base = max(0.0, float(backoff_base))
        self._backoff_cap = max(self._backoff_base, float(backoff_cap))
        jitter_values = tuple(float(item) for item in jitter) if jitter else _DEFAULT_JITTER
        self._jitter = jitter_values if len(jitter_values) == 2 else _DEFAULT_JITTER
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep

        self._buffer_size = max(1, int(buffer_size))
        self._pending: Deque[StreamBatch] = deque()
        self._pending_lock = threading.Lock()
        self._pending_condition = threading.Condition(self._pending_lock)
        self._prefill_enabled = False
        self._prefill_drained_once = False
        self._channel_param = (channel_param or "").strip()
        self._cursor_param = (cursor_param or "").strip()
        self._cursor: str | None = (
            str(initial_cursor).strip() if initial_cursor not in (None, "") else None
        )
        self._channel_serializer = channel_serializer
        self._closed = False
        self._last_poll = 0.0
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._worker_error: Exception | None = None
        self._http_method = (http_method or "GET").upper()
        self._params_in_body = bool(params_in_body)
        self._channels_in_body = bool(channels_in_body)
        self._cursor_in_body = bool(cursor_in_body)
        self._body_params = dict(body_params) if isinstance(body_params, Mapping) else {}
        self._body_encoder = body_encoder
        self._metrics: MetricsRegistry = metrics_registry or get_global_metrics_registry()
        self._metric_labels = {
            "adapter": self._adapter,
            "scope": self._scope,
            "environment": self._environment,
        }
        self._metric_latency: HistogramMetric = self._metrics.histogram(
            "bot_exchange_stream_long_poll_latency_seconds",
            "Rozkład opóźnień zapytań long-pollowych",
            buckets=_DEFAULT_LATENCY_BUCKETS,
        )
        self._metric_delivery_lag: HistogramMetric = self._metrics.histogram(
            "bot_exchange_stream_delivery_lag_seconds",
            "Różnica między pobraniem paczki a jej konsumpcją",
            buckets=_DEFAULT_LAG_BUCKETS,
        )
        self._metric_queue: GaugeMetric = self._metrics.gauge(
            "bot_exchange_stream_pending_batches",
            "Aktualna liczba oczekujących paczek long-pollowych",
        )
        self._metric_backpressure: CounterMetric = self._metrics.counter(
            "bot_exchange_stream_backpressure_total",
            "Liczba paczek usuniętych z kolejki long-pollowej z powodu backpressure",
        )
        self._update_queue_metric()

    def start(self) -> "LocalLongPollStream":
        """Uruchamia wątek odpowiedzialny za polling long-pollowy."""

        with self._pending_condition:
            if self._closed:
                return self
            self._prefill_enabled = True
        self._ensure_worker()
        return self

    def wait_prefill(self, *, min_batches: int = 1, timeout: float | None = None) -> bool:
        """Oczekuje na wypełnienie bufora przez wątek prefetchujący.

        Zwraca ``True`` jeśli w buforze pojawiła się wymagana liczba paczek
        przed upływem limitu czasu. Jeśli limit czasu zostanie przekroczony,
        metoda zwraca ``False``. Wszelkie błędy pracy wątku pobierającego są
        propagowane poprzez ponowne zgłoszenie oryginalnego wyjątku.

        Args:
            min_batches: Minimalna liczba paczek, które powinny znaleźć się w
                buforze zanim metoda zakończy działanie. Wartości mniejsze od 1
                traktowane są jako brak wymagań (metoda zwraca ``True``).
            timeout: Maksymalny czas oczekiwania w sekundach. ``None`` oznacza
                oczekiwanie bez limitu czasowego.
        """

        if min_batches <= 0:
            return True

        self.start()

        deadline: float | None = None
        if timeout is not None:
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError):
                timeout_value = 0.0
            if timeout_value < 0:
                timeout_value = 0.0
            deadline = time.monotonic() + timeout_value

        with self._pending_condition:
            while not self._closed:
                if self._worker_error is not None:
                    raise self._worker_error
                if len(self._pending) >= min_batches:
                    return True
                if deadline is None:
                    self._pending_condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._pending_condition.wait(remaining)

            if self._worker_error is not None:
                raise self._worker_error
            return False

    async def wait_prefill_async(
        self, *, min_batches: int = 1, timeout: float | None = None
    ) -> bool:
        """Asynchroniczna wersja ``wait_prefill`` uruchamiana poza pętlą zdarzeń."""

        loop = asyncio.get_running_loop()
        func = functools.partial(self.wait_prefill, min_batches=min_batches, timeout=timeout)
        return await loop.run_in_executor(None, func)

    def __iter__(self) -> "LocalLongPollStream":
        self._ensure_worker()
        return self

    def __enter__(self) -> "LocalLongPollStream":
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:  # noqa: D401
        self.close()
        return None

    async def __aenter__(self) -> "LocalLongPollStream":
        """Zapewnia wsparcie dla asynchronicznych kontekstów."""

        self.start()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> bool | None:  # noqa: D401
        await self.aclose()
        return None

    @property
    def closed(self) -> bool:
        """Zwraca informację, czy iterator został już zamknięty."""

        return self._closed

    def __next__(self) -> StreamBatch:
        if self._closed:
            raise StopIteration

        self._ensure_worker()
        batch: StreamBatch | None = None
        error: Exception | None = None

        while batch is None and error is None:
            with self._pending_condition:
                if self._worker_error is not None:
                    error = self._worker_error
                    break
                if self._pending:
                    batch = self._pending.popleft()
                    self._prefill_drained_once = True
                    self._update_queue_metric_locked()
                    self._pending_condition.notify_all()
                    break
                if self._closed:
                    break
                self._pending_condition.wait()

        if error is not None:
            raise error
        if batch is None:
            raise StopIteration

        self._record_delivery_lag(batch)
        return batch

    def __aiter__(self) -> "LocalLongPollStream":
        self._ensure_worker()
        return self

    async def __anext__(self) -> StreamBatch:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.__next__)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        """Asynchroniczna wersja zamykania strumienia."""

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)

    def close(self) -> None:
        """Zamyka iterator – kolejne próby pobrania danych zakończą się StopIteration."""

        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with self._pending_condition:
            self._pending.clear()
            self._update_queue_metric_locked()
            self._pending_condition.notify_all()

        worker = self._worker_thread
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        self._worker_thread = None

    # ------------------------------------------------------------------
    # Wewnętrzne operacje long-polla
    # ------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._closed:
            return
        with self._pending_condition:
            worker = self._worker_thread
            if worker and worker.is_alive():
                return
            if self._closed:
                return
            worker = threading.Thread(
                target=self._poller_loop,
                name=f"LocalLongPollStream[{self._adapter}:{self._scope}]",
                daemon=True,
            )
            self._worker_thread = worker
        worker.start()

    def _poller_loop(self) -> None:
        try:
            while not self._stop_event.is_set() and not self._closed:
                with self._pending_condition:
                    while (
                        not self._closed
                        and not self._stop_event.is_set()
                        and (
                            len(self._pending)
                            >= (
                                self._buffer_size
                                if self._prefill_enabled and self._prefill_drained_once
                                else 1
                            )
                        )
                    ):
                        self._pending_condition.wait()
                if self._stop_event.is_set() or self._closed:
                    break
                try:
                    self._poll_once()
                except Exception as exc:  # pragma: no cover - delegujemy do wątku głównego
                    with self._pending_condition:
                        self._worker_error = exc
                        self._closed = True
                        self._pending_condition.notify_all()
                    self._stop_event.set()
                    return
        finally:
            with self._pending_condition:
                self._pending_condition.notify_all()

    def _poll_once(self) -> None:
        now = self._clock()
        elapsed = now - self._last_poll
        if elapsed < self._poll_interval:
            self._sleep(self._poll_interval - elapsed)

        attempt = 0
        last_error: Exception | None = None
        while attempt < self._max_retries and not self._closed:
            request = self._build_request()
            should_backoff = True
            headers = None
            poll_started = self._clock()
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    raw_payload = response.read()
                    headers = getattr(response, "headers", None)
            except HTTPError as exc:
                server_retry_after = self._retry_after(exc.headers)
                if exc.code in {401, 403}:
                    raise ExchangeAuthError(
                        f"Stream {self._adapter}/{self._scope} odrzucił uwierzytelnienie.",
                        exc.code,
                        payload=None,
                    ) from exc

                if exc.code == 429:
                    last_error = ExchangeThrottlingError(
                        f"Stream {self._adapter}/{self._scope} został ograniczony (429).",
                        exc.code,
                        payload=None,
                    )
                    if server_retry_after is not None and server_retry_after > 0:
                        self._sleep(server_retry_after)
                        should_backoff = False
                elif 500 <= exc.code < 600:
                    last_error = ExchangeNetworkError(
                        f"Stream {self._adapter}/{self._scope} zwrócił błąd {exc.code}.",
                        reason=exc,
                    )
                else:
                    raise ExchangeAPIError(
                        f"Stream {self._adapter}/{self._scope} zwrócił kod {exc.code}.",
                        exc.code,
                        payload=None,
                    ) from exc
            except URLError as exc:
                last_error = ExchangeNetworkError(
                    f"Nie udało się połączyć ze streamem {self._adapter}/{self._scope}.",
                    reason=exc,
                )
            else:
                payload = self._parse_payload(raw_payload, headers)
                retry_after = self._enqueue_batches(payload)
                finished = self._clock()
                self._record_poll_latency(max(0.0, finished - poll_started))
                self._last_poll = finished
                if retry_after > 0:
                    self._sleep(retry_after)
                return

            attempt += 1
            if self._closed:
                break
            if self._backoff_base > 0.0 and should_backoff:
                delay = min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_cap)
                jitter = random.uniform(*self._jitter) if self._jitter[1] > 0 else 0.0
                if delay + jitter > 0:
                    self._sleep(delay + jitter)

        if last_error is not None:
            raise last_error
        raise ExchangeNetworkError(
            f"Nie udało się pobrać danych streamu {self._adapter}/{self._scope}.",
            reason=None,
        )

    def _build_request(self) -> Request:
        query_params: list[tuple[str, object]] = []
        if not self._params_in_body:
            query_params = [(str(key), value) for key, value in self._params.items()]

        body_payload: dict[str, object] = {}
        if self._params_in_body:
            for key, value in self._params.items():
                self._assign_body_value(body_payload, str(key), value)
        if self._body_params:
            for key, value in self._body_params.items():
                self._assign_body_value(body_payload, str(key), value)

        if self._channel_param:
            if self._channels_in_body:
                self._assign_body_value(
                    body_payload, self._channel_param, self._serialize_channels()
                )
            else:
                query_params = [item for item in query_params if item[0] != self._channel_param]
                self._extend_params(
                    query_params, self._channel_param, self._serialize_channels()
                )

        if self._cursor is not None and self._cursor_param:
            if self._cursor_in_body:
                self._assign_body_value(body_payload, self._cursor_param, self._cursor)
            else:
                query_params = [item for item in query_params if item[0] != self._cursor_param]
                self._extend_params(query_params, self._cursor_param, self._cursor)

        filtered_params = [
            (key, value)
            for key, value in query_params
            if key and not self._is_empty_value(value)
        ]
        query = urlencode(filtered_params, doseq=True)
        url = f"{self._base_url}{self._path}"
        if query:
            url = f"{url}?{query}"
        _LOGGER.debug("Long-poll %s %s", urlsplit(url).path, query)
        data: bytes | None = None
        content_type: str | None = None
        if body_payload:
            data, content_type = self._encode_body_payload(body_payload)
        headers = dict(self._headers)
        if content_type and "Content-Type" not in headers:
            headers["Content-Type"] = content_type
        return Request(url, headers=headers, data=data, method=self._http_method)

    def _parse_payload(
        self, payload: bytes, headers: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        decoded_bytes = self._decode_payload(payload, headers)
        if not decoded_bytes:
            return {}
        try:
            decoded = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - rzadki przypadek
            raise ExchangeAPIError(
                "Odpowiedź streamu zawiera niepoprawne kodowanie.",
                0,
                payload=str(exc),
            ) from exc
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ExchangeAPIError(
                "Odpowiedź streamu zawiera niepoprawny JSON.",
                0,
                payload=decoded[:200],
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ExchangeAPIError(
                "Stream zwrócił odpowiedź o niepoprawnym formacie.",
                0,
                payload=decoded,
            )
        self._detect_errors(parsed)
        return parsed

    def _decode_payload(
        self, payload: bytes, headers: Mapping[str, Any] | None
    ) -> bytes:
        if not payload:
            return b""
        encoding_header: object | None = None
        if headers:
            try:
                encoding_header = headers.get("Content-Encoding")  # type: ignore[arg-type]
            except AttributeError:  # pragma: no cover - nietypowy typ nagłówków
                encoding_header = None
            if encoding_header is None and isinstance(headers, Mapping):
                for key, value in headers.items():
                    if isinstance(key, str) and key.lower() == "content-encoding":
                        encoding_header = value
                        break
        if not encoding_header:
            return payload
        encoding_values: list[str] = []
        for entry in _iter_header_entries(encoding_header):
            for raw_item in entry.split(","):
                if not raw_item:
                    continue
                token = raw_item.split(";", 1)[0].strip().lower()
                if not token:
                    continue
                token = _CONTENT_ENCODING_ALIASES.get(token, token)
                encoding_values.append(token)
        if not encoding_values:
            return payload
        data = payload
        # Zgodnie z RFC 9110 sekcja 8.4, kodowania są stosowane w kolejności podanej
        # w nagłówku, więc dekodowanie musi przebiegać w odwrotnej kolejności.
        for encoding in reversed(encoding_values):
            if encoding in {"gzip"}:
                try:
                    data = gzip.decompress(data)
                except OSError as exc:  # pragma: no cover - uszkodzone dane gzip
                    raise ExchangeAPIError(
                        "Nie udało się zdekompresować odpowiedzi gzip streamu.",
                        0,
                        payload=str(exc),
                    ) from exc
                continue
            if encoding in {"deflate", "zlib"}:
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    try:
                        data = zlib.decompress(data, -zlib.MAX_WBITS)
                    except zlib.error as exc:  # pragma: no cover - uszkodzone dane deflate
                        raise ExchangeAPIError(
                            "Nie udało się zdekompresować odpowiedzi deflate streamu.",
                            0,
                            payload=str(exc),
                        ) from exc
                continue
            if encoding in {"br", "brotli"}:
                if brotli is None or not hasattr(brotli, "decompress"):
                    _LOGGER.debug(
                        "Odebrano odpowiedź brotli, ale brak opcjonalnej biblioteki 'brotli' – pomijam dekodowanie."
                    )
                    continue
                try:
                    data = brotli.decompress(data)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - zależy od implementacji
                    raise ExchangeAPIError(
                        "Nie udało się zdekompresować odpowiedzi brotli streamu.",
                        0,
                        payload=str(exc),
                    ) from exc
                continue
            if encoding in {"zstd", "zstandard"}:
                if zstandard is None:
                    _LOGGER.debug(
                        "Odebrano odpowiedź zstd, ale brak opcjonalnej biblioteki 'zstandard' – pomijam dekodowanie."
                    )
                    continue
                decompressor = getattr(zstandard, "decompress", None)
                if callable(decompressor):
                    try:
                        data = decompressor(data)
                    except Exception as exc:  # pragma: no cover - zależy od implementacji
                        raise ExchangeAPIError(
                            "Nie udało się zdekompresować odpowiedzi zstd streamu.",
                            0,
                            payload=str(exc),
                        ) from exc
                    continue
                decompressor_cls = getattr(zstandard, "ZstdDecompressor", None)
                if decompressor_cls is not None:
                    try:
                        instance = decompressor_cls()
                    except Exception as exc:  # pragma: no cover - inicjalizacja może rzucić
                        raise ExchangeAPIError(
                            "Nie udało się zainicjalizować dekodera zstd streamu.",
                            0,
                            payload=str(exc),
                        ) from exc
                    decompress_method = getattr(instance, "decompress", None)
                    if callable(decompress_method):
                        try:
                            data = decompress_method(data)
                        except Exception as exc:  # pragma: no cover - zależy od implementacji
                            raise ExchangeAPIError(
                                "Nie udało się zdekompresować odpowiedzi zstd streamu.",
                                0,
                                payload=str(exc),
                            ) from exc
                        continue
                _LOGGER.debug(
                    "Odebrano odpowiedź zstd, ale moduł 'zstandard' nie udostępnia funkcji dekodowania – pomijam dekodowanie."
                )
                continue
            if encoding in {"identity"}:
                continue
            raise ExchangeAPIError(
                f"Stream {self._adapter}/{self._scope} zwrócił nieobsługiwane kodowanie odpowiedzi '{encoding}'.",
                0,
                payload={"encoding": encoding_header},
            )
        return data

    def _detect_errors(self, payload: Mapping[str, Any]) -> None:
        error_entry = payload.get("error")
        if self._is_truthy_error(error_entry):
            message = self._format_error_message(error_entry)
            raise ExchangeAPIError(
                f"Stream {self._adapter}/{self._scope} zwrócił błąd: {message}",
                0,
                payload=payload,
            )

        errors_entry = payload.get("errors")
        if isinstance(errors_entry, Sequence) and errors_entry:
            message = ", ".join(self._format_error_message(item) for item in errors_entry)
            raise ExchangeAPIError(
                f"Stream {self._adapter}/{self._scope} zwrócił błędy: {message}",
                0,
                payload=payload,
            )

        status = payload.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            details = payload.get("message") or payload.get("reason")
            message = self._format_error_message(details) if details else status
            raise ExchangeAPIError(
                f"Stream {self._adapter}/{self._scope} zgłosił status błędu: {message}",
                0,
                payload=payload,
            )

    @staticmethod
    def _is_truthy_error(value: object) -> bool:
        if value in (None, "", False, 0):
            return False
        if isinstance(value, Mapping):
            return any(LocalLongPollStream._is_truthy_error(item) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(LocalLongPollStream._is_truthy_error(item) for item in value)
        return True

    @staticmethod
    def _format_error_message(value: object) -> str:
        if isinstance(value, Mapping):
            message = value.get("message") or value.get("error") or value.get("reason")
            if isinstance(message, str) and message.strip():
                return message.strip()
            if isinstance(value.get("code"), (str, int)):
                return f"code={value['code']}"
            return json.dumps(value, ensure_ascii=False)[:120]
        return str(value)

    def _enqueue_batches(self, payload: Mapping[str, Any]) -> float:
        cursor = self._extract_cursor(payload)
        if cursor is not None:
            self._cursor = cursor

        raw_batches = payload.get("batches")
        if isinstance(raw_batches, Mapping):
            batches_iterable: Sequence[Mapping[str, Any]] = (raw_batches,)
        elif isinstance(raw_batches, Sequence) and not isinstance(raw_batches, (str, bytes, bytearray)):
            batches_iterable = [entry for entry in raw_batches if isinstance(entry, Mapping)]
        elif payload.get("events") is not None:
            batches_iterable = (payload,)  # pojedynczy kanał
        else:
            batches_iterable = ()

        now = self._clock()
        prepared_batches: list[StreamBatch] = []
        for entry in batches_iterable:
            channel = str(entry.get("channel") or self._channels[0])
            events_raw = entry.get("events") or ()
            events: list[Mapping[str, Any]] = []
            if isinstance(events_raw, Mapping):
                events.append(dict(events_raw))
            elif isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes, bytearray)):
                for item in events_raw:
                    if isinstance(item, Mapping):
                        events.append(dict(item))
                    else:
                        events.append({"value": item})
            elif events_raw not in (None, ""):
                events.append({"value": events_raw})

            entry_cursor = self._extract_cursor(entry)
            if entry_cursor is not None:
                self._cursor = entry_cursor

            heartbeat_flag = bool(entry.get("heartbeat", False))
            prepared_batches.append(
                StreamBatch(
                    channel=channel,
                    events=tuple(events),
                    cursor=self._cursor,
                    heartbeat=heartbeat_flag or not events,
                    received_at=now,
                    raw=dict(entry),
                )
            )

        if not prepared_batches:
            prepared_batches.append(
                StreamBatch(
                    channel="*",
                    events=(),
                    cursor=self._cursor,
                    heartbeat=True,
                    received_at=now,
                    raw={},
                )
            )

        with self._pending_condition:
            for batch in prepared_batches:
                self._apply_backpressure_locked()
                self._pending.append(batch)
            self._update_queue_metric_locked()
            self._pending_condition.notify_all()

        retry_after = self._coerce_positive_float(payload.get("retry_after"))
        if retry_after is not None:
            return retry_after
        poll_after = self._coerce_positive_float(payload.get("poll_after"))
        if poll_after is not None:
            return poll_after
        return self._poll_interval

    @staticmethod
    def _retry_after(headers: Mapping[str, str] | None) -> float | None:
        if not headers:
            return None
        retry_after = headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            value = float(retry_after)
        except (TypeError, ValueError):  # pragma: no cover - niepoprawna wartość
            try:
                parsed = parsedate_to_datetime(str(retry_after))
            except (TypeError, ValueError):  # pragma: no cover - niepoprawne RFC2822
                return None
            if parsed is None:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            target_ts = parsed.timestamp()
            current = time.time()
            return max(0.0, target_ts - current)
        return max(0.0, value)

    def _serialize_channels(self) -> object:
        if self._channel_serializer:
            return self._channel_serializer(self._channels)
        return ",".join(self._channels)

    def _assign_body_value(
        self, payload: MutableMapping[str, object], key: str, value: object
    ) -> None:
        if not key or self._is_empty_value(value):
            return
        if isinstance(value, Mapping):
            normalized = {
                str(sub_key): sub_value
                for sub_key, sub_value in value.items()
                if not self._is_empty_value(sub_value)
            }
            if not normalized:
                return
            existing = payload.get(key)
            if isinstance(existing, Mapping):
                merged = dict(existing)
                merged.update(normalized)
                payload[key] = merged
            else:
                payload[key] = normalized
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            normalized_seq = [item for item in value if not self._is_empty_value(item)]
            if not normalized_seq:
                return
            payload[key] = normalized_seq
            return

        existing = payload.get(key)
        if existing is None:
            payload[key] = value
            return
        if isinstance(existing, list):
            existing.append(value)
        else:
            payload[key] = [existing, value]

    def _encode_body_payload(
        self, payload: Mapping[str, object]
    ) -> tuple[bytes, str | None]:
        encoder = self._body_encoder
        if encoder is None:
            encoder_mode = "json"
        elif isinstance(encoder, str):
            encoder_mode = encoder.lower()
        else:
            result = encoder(dict(payload))
            if isinstance(result, tuple):
                data, content_type = result
            else:
                data, content_type = result, None
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, content_type

        if encoder is None or encoder_mode == "json":
            serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return serialized, "application/json"
        if encoder_mode in {"form", "urlencoded"}:
            items: list[tuple[str, object]] = []
            for key, value in payload.items():
                self._extend_params(items, key, value)
            filtered = [
                (key, value)
                for key, value in items
                if key and not self._is_empty_value(value)
            ]
            encoded = urlencode(filtered, doseq=True).encode("utf-8")
            return encoded, "application/x-www-form-urlencoded"
        raise ValueError(f"Nieobsługiwany tryb kodowania body '{self._body_encoder}'.")

    def _extract_cursor(self, payload: Mapping[str, Any]) -> str | None:
        keys: tuple[str, ...]
        if self._cursor_param and self._cursor_param != "cursor":
            keys = (self._cursor_param, "cursor")
        else:
            keys = ("cursor",)
        meta = payload.get("meta") or payload.get("metadata")
        if isinstance(meta, Mapping):
            for key in keys:
                nested = meta.get(key)
                parsed = self._normalize_cursor_value(nested)
                if parsed is not None:
                    return parsed

        for key in keys:
            value = payload.get(key)
            parsed = self._normalize_cursor_value(value)
            if parsed is not None:
                return parsed

        if isinstance(meta, Mapping):
            next_cursor = meta.get("next_cursor") or meta.get("nextCursor")
            parsed = self._normalize_cursor_value(next_cursor)
            if parsed is not None:
                return parsed

        next_cursor = payload.get("next_cursor") or payload.get("nextCursor")
        parsed = self._normalize_cursor_value(next_cursor)
        if parsed is not None:
            return parsed

        return None

    def _normalize_cursor_value(self, value: object) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        elif isinstance(value, (int, float)):
            return str(value)
        return None

    @staticmethod
    def _is_empty_value(value: object) -> bool:
        if value in (None, ""):
            return True
        if isinstance(value, (Sequence, Mapping)) and not isinstance(value, (str, bytes, bytearray)):
            if isinstance(value, Mapping):
                return not any(not LocalLongPollStream._is_empty_value(v) for v in value.values())
            return not any(not LocalLongPollStream._is_empty_value(item) for item in value)
        return False

    def _extend_params(self, params: list[tuple[str, object]], key: str, value: object) -> None:
        if not key and not isinstance(value, Mapping):
            return
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                self._extend_params(params, str(sub_key), sub_value)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    for sub_key, sub_value in item.items():
                        self._extend_params(params, str(sub_key), sub_value)
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
                    sub_key, sub_value = item
                    self._extend_params(params, str(sub_key), sub_value)
                else:
                    self._extend_params(params, key, item)
            return
        if value in (None, ""):
            return
        params.append((str(key), value))

    def _apply_backpressure_locked(self) -> None:
        dropped = 0
        while len(self._pending) >= self._buffer_size:
            self._pending.popleft()
            dropped += 1
        if dropped:
            self._metric_backpressure.inc(dropped, labels=self._metric_labels)

    def _update_queue_metric(self) -> None:
        with self._pending_condition:
            self._update_queue_metric_locked()

    def _update_queue_metric_locked(self) -> None:
        self._metric_queue.set(float(len(self._pending)), labels=self._metric_labels)

    def _record_poll_latency(self, duration: float) -> None:
        self._metric_latency.observe(duration, labels=self._metric_labels)

    def _record_delivery_lag(self, batch: StreamBatch) -> None:
        lag = max(0.0, self._clock() - batch.received_at)
        self._metric_delivery_lag.observe(lag, labels=self._metric_labels)

    @staticmethod
    def _coerce_positive_float(value: object) -> float | None:
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            try:
                numeric = float(value.strip())
            except (TypeError, ValueError):
                return None
        else:
            return None
        if numeric <= 0:
            return None
        return numeric
__all__ = ["LocalLongPollStream", "StreamBatch"]
