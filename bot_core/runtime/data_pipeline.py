"""Komponenty budujące feed danych dla runtime strategii."""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
import weakref
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.data import CachedOHLCVSource, resolve_cache_namespace, create_cached_ohlcv_source
from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.backfill_scheduler import BackfillScheduler
from bot_core.data.ohlcv import OHLCVBackfillService
from bot_core.exchanges.base import ExchangeAdapter, Environment
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.market_intel import MarketIntelAggregator
from bot_core.execution.base import ExecutionService, PriceResolver
from bot_core.execution.paper import MarketMetadata
from bot_core.runtime.multi_strategy_scheduler import StrategyDataFeed, StrategySignalSink
from bot_core.runtime.scheduler import AsyncIOTaskQueue
from bot_core.strategies.base import MarketSnapshot, StrategySignal
from bot_core.observability.metrics import MetricsRegistry

_DEFAULT_LEDGER_SUBDIR = Path("audit/ledger")
_DEFAULT_OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_LOGGER = logging.getLogger(__name__)
_TEST_MODE_ENV = "DUDZIAN_TEST_MODE"
_PIPELINE_THREAD_NAME = "PipelineStream"


def _is_test_mode_enabled() -> bool:
    return os.getenv(_TEST_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def consume_stream(
    stream: Iterable[StreamBatch],
    *,
    handle_batch: Callable[[StreamBatch], object] | None = None,
    stop_condition: Callable[[], bool] | None = None,
    on_heartbeat: Callable[[float], object] | None = None,
    heartbeat_interval: float = 15.0,
    idle_timeout: float | None = 60.0,
    clock: Callable[[], float] | None = None,
) -> None:
    iterator = iter(stream)
    closers: list[Callable[[], object]] = []
    for target in (stream, iterator):
        closer = getattr(target, "close", None)
        if callable(closer):
            closers.append(closer)
    if closers:
        unique: dict[tuple[object | None, object], Callable[[], object]] = {}
        for closer in closers:
            key = (getattr(closer, "__self__", None), getattr(closer, "__func__", closer))
            unique[key] = closer
        closers = list(unique.values())

    time_source = clock or time.monotonic
    last_event_at = time_source()
    last_heartbeat_at = last_event_at
    heartbeat_interval = max(0.0, float(heartbeat_interval))
    timeout_value = None if idle_timeout is None else max(0.0, float(idle_timeout))

    try:
        while True:
            if stop_condition is not None and stop_condition():
                break

            try:
                batch = next(iterator)
            except StopIteration:
                break

            observed_at = float(getattr(batch, "received_at", time_source()))
            if batch.events:
                if handle_batch is not None:
                    handle_batch(batch)
                last_event_at = observed_at
                last_heartbeat_at = observed_at
            else:
                if batch.heartbeat:
                    if on_heartbeat is not None:
                        on_heartbeat(observed_at)
                    last_heartbeat_at = observed_at
                elif on_heartbeat and (observed_at - last_heartbeat_at) >= heartbeat_interval:
                    on_heartbeat(observed_at)
                    last_heartbeat_at = observed_at

            if timeout_value is not None and (observed_at - last_event_at) >= timeout_value:
                raise TimeoutError(
                    f"Brak nowych danych w streamie '{batch.channel}' przez {timeout_value:.1f} s"
                )
    finally:
        for closer in closers:
            try:
                closer()
            except Exception:  # pragma: no cover
                _LOGGER.warning("Nie udało się zamknąć streamu long-pollowego", exc_info=True)


async def consume_stream_async(
    stream: AsyncIterable[StreamBatch],
    *,
    handle_batch: Callable[[StreamBatch], Awaitable[object] | object] | None = None,
    stop_condition: Callable[[], Awaitable[bool] | bool] | None = None,
    on_heartbeat: Callable[[float], Awaitable[object] | object] | None = None,
    heartbeat_interval: float = 15.0,
    idle_timeout: float | None = 60.0,
    clock: Callable[[], float] | None = None,
) -> None:
    iterator = stream.__aiter__()
    closers: list[Callable[[], Awaitable[object] | object]] = []
    for target in (stream, iterator):
        if target is None:
            continue
        closer = getattr(target, "aclose", None)
        if not callable(closer):
            closer = getattr(target, "close", None)
        if callable(closer):
            closers.append(closer)
    if closers:
        unique: dict[tuple[object | None, object], Callable[[], Awaitable[object] | object]] = {}
        for closer in closers:
            key = (getattr(closer, "__self__", None), getattr(closer, "__func__", closer))
            unique[key] = closer
        closers = list(unique.values())

    time_source = clock or time.monotonic
    last_event_at = time_source()
    last_heartbeat_at = last_event_at
    heartbeat_interval = max(0.0, float(heartbeat_interval))
    timeout_value = None if idle_timeout is None else max(0.0, float(idle_timeout))

    async def _call_maybe_async(
        func: Callable[..., object] | None, *args: object
    ) -> object | None:
        if func is None:
            return None
        result = func(*args)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result

    try:
        while True:
            if stop_condition is not None:
                should_stop = await _call_maybe_async(stop_condition)
                if bool(should_stop):
                    break

            try:
                batch = await iterator.__anext__()
            except StopAsyncIteration:
                break

            observed_at = float(getattr(batch, "received_at", time_source()))
            if batch.events:
                await _call_maybe_async(handle_batch, batch)
                last_event_at = observed_at
                last_heartbeat_at = observed_at
            else:
                if batch.heartbeat:
                    await _call_maybe_async(on_heartbeat, observed_at)
                    last_heartbeat_at = observed_at
                elif on_heartbeat and (observed_at - last_heartbeat_at) >= heartbeat_interval:
                    await _call_maybe_async(on_heartbeat, observed_at)
                    last_heartbeat_at = observed_at

            if timeout_value is not None and (observed_at - last_event_at) >= timeout_value:
                raise TimeoutError(
                    f"Brak nowych danych w streamie '{batch.channel}' przez {timeout_value:.1f} s"
                )
    finally:
        for closer in closers:
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result  # type: ignore[func-returns-value]
            except Exception:  # pragma: no cover
                _LOGGER.warning("Nie udało się zamknąć streamu long-pollowego", exc_info=True)


class InMemoryStrategySignalSink(StrategySignalSink):
    """Prosty bufor sygnałów strategii do testów/offline."""

    def __init__(self) -> None:
        self._records: list[tuple[str, Sequence[StrategySignal]]] = []

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: Sequence[StrategySignal],
    ) -> None:
        self._records.append((schedule_name, tuple(signals)))

    def export(self) -> Sequence[tuple[str, Sequence[StrategySignal]]]:
        return tuple(self._records)


class OHLCVStrategyFeed(StrategyDataFeed):
    """Strategiczny feed korzystający z lokalnego cache OHLCV."""

    def __init__(
        self,
        data_source: CachedOHLCVSource,
        *,
        symbols_map: Mapping[str, Sequence[str]],
        interval_map: Mapping[str, str],
    ) -> None:
        self._data_source = data_source
        self._symbols_map = {key: tuple(values) for key, values in symbols_map.items()}
        self._interval_map = dict(interval_map)

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols:
            return ()
        interval = self._interval_map.get(strategy_name) or "1h"
        snapshots: list[MarketSnapshot] = []
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        for symbol in symbols:
            request = OHLCVRequest(
                symbol=symbol,
                interval=interval,
                start=0,
                end=now_ms,
                limit=max(1, int(bars)),
            )
            response = self._data_source.fetch_ohlcv(request)
            snapshots.extend(_response_to_snapshots(symbol, response))
        snapshots.sort(key=lambda item: (item.symbol, item.timestamp))
        return tuple(snapshots[-bars:])

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        return self.load_history(strategy_name, bars=1)


class StreamingStrategyFeed(StrategyDataFeed):
    """Łączy `LocalLongPollStream` z interfejsem StrategyDataFeed."""

    _active_lock = threading.Lock()
    _active_instances: "weakref.WeakSet[StreamingStrategyFeed]" = weakref.WeakSet()

    def __init__(
        self,
        *,
        history_feed: StrategyDataFeed,
        stream_factory: Callable[[], Iterable[StreamBatch] | AsyncIterable[StreamBatch]],
        symbols_map: Mapping[str, Sequence[str]],
        buffer_size: int = 256,
        heartbeat_interval: float = 15.0,
        idle_timeout: float | None = 60.0,
        restart_delay: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._history_feed = history_feed
        self._stream_factory = stream_factory
        self._symbols_map = {key: tuple(values) for key, values in symbols_map.items()}
        self._buffer_size = max(1, int(buffer_size))
        self._heartbeat_interval = max(0.0, float(heartbeat_interval))
        self._idle_timeout = idle_timeout if idle_timeout is None else max(0.0, float(idle_timeout))
        self._restart_delay = max(0.5, float(restart_delay))
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._async_task: asyncio.Task[None] | None = None
        self._disabled = False
        self._buffers: dict[str, deque[MarketSnapshot]] = {}
        self._known_symbols = {
            symbol
            for values in self._symbols_map.values()
            for symbol in values
        }
        for symbol in self._known_symbols:
            self._buffers[symbol] = deque(maxlen=self._buffer_size)
        self._last_event_at: float | None = None

    def start(self) -> None:
        if _is_test_mode_enabled():
            self._disabled = True
            self._stop_event.set()
            return
        if self._disabled:
            return
        if self._async_task and not self._async_task.done():
            raise RuntimeError("StreamingStrategyFeed jest już uruchomiony w trybie asynchronicznym.")
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=_PIPELINE_THREAD_NAME, daemon=True)
        self._thread.start()
        self._register_instance()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        task = self._async_task
        if task and not task.done():
            task.cancel()
        self._unregister_instance()

    def ingest_batch(self, batch: StreamBatch) -> None:
        if not batch.events:
            return
        snapshots: list[MarketSnapshot] = []
        for event in batch.events:
            try:
                snapshot = self._event_to_snapshot(event)
            except Exception:
                self._logger.debug("Nie udało się sparsować zdarzenia streamu: %s", event, exc_info=True)
                continue
            if snapshot is None:
                continue
            if self._known_symbols and snapshot.symbol not in self._known_symbols:
                continue
            snapshots.append(snapshot)
        if not snapshots:
            return
        with self._lock:
            for snapshot in snapshots:
                buffer = self._buffers.setdefault(snapshot.symbol, deque(maxlen=self._buffer_size))
                buffer.append(snapshot)
            self._last_event_at = time.monotonic()

    def start_async(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task[None]:
        if self._disabled:
            if not _is_test_mode_enabled():
                if self._async_task is None:
                    if loop is None:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:  # pragma: no cover
                            loop = asyncio.get_event_loop()
                    self._async_task = loop.create_task(asyncio.sleep(0))
                return self._async_task
            task = self._async_task
            if task is not None and not task.done():
                task.cancel()
            self._async_task = None
            self._disabled = False
        if self._thread and self._thread.is_alive():
            raise RuntimeError("StreamingStrategyFeed jest już uruchomiony w trybie synchronicznym.")
        if self._async_task and not self._async_task.done():
            return self._async_task

        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover
                loop = asyncio.get_event_loop()

        self._stop_event.clear()
        self._async_task = loop.create_task(self._run_loop_async())
        self._register_instance()
        return self._async_task

    async def stop_async(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self.stop()

        task = self._async_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover
            pass
        self._unregister_instance()

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        return self._history_feed.load_history(strategy_name, bars)

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols:
            return ()
        collected: list[MarketSnapshot] = []
        with self._lock:
            for symbol in symbols:
                queue = self._buffers.get(symbol)
                if not queue:
                    continue
                while queue:
                    collected.append(queue.popleft())
        collected.sort(key=lambda item: (item.symbol, item.timestamp))
        return tuple(collected)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stream = self._stream_factory()
                consume_stream(
                    stream,
                    handle_batch=self.ingest_batch,
                    heartbeat_interval=self._heartbeat_interval,
                    idle_timeout=self._idle_timeout,
                    on_heartbeat=self._handle_heartbeat,
                    stop_condition=self._stop_event.is_set,
                )
            except TimeoutError:
                self._logger.warning("Brak nowych danych w streamie strategii przez dłuższy czas")
            except Exception:  # pragma: no cover
                self._logger.exception("Błąd podczas przetwarzania streamu strategii")
            if self._stop_event.is_set():
                break
            time.sleep(self._restart_delay)

    async def _run_loop_async(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    stream = self._stream_factory()
                    is_async_stream = callable(getattr(stream, "__aiter__", None))
                    if not is_async_stream:
                        raise TypeError(
                            "StreamingStrategyFeed.start_async wymaga, aby stream_factory zwracał AsyncIterable[StreamBatch], got: "
                            f"{type(stream)!r}"
                        )
                    await consume_stream_async(
                        stream,
                        handle_batch=self.ingest_batch,
                        heartbeat_interval=self._heartbeat_interval,
                        idle_timeout=self._idle_timeout,
                        on_heartbeat=self._handle_heartbeat,
                        stop_condition=self._stop_event.is_set,
                    )
                except asyncio.CancelledError:
                    raise
                except TypeError:
                    raise
                except TimeoutError:
                    self._logger.warning("Brak nowych danych w streamie strategii przez dłuższy czas")
                except Exception:  # pragma: no cover
                    self._logger.exception("Błąd podczas przetwarzania streamu strategii")

                if self._stop_event.is_set():
                    break
                if self._restart_delay > 0:
                    await asyncio.sleep(self._restart_delay)
        except asyncio.CancelledError:
            raise
        finally:
            self._async_task = None
            self._unregister_instance()

    def _handle_heartbeat(self, timestamp: float) -> None:
        if self._last_event_at is None:
            return
        drift = timestamp - self._last_event_at
        if drift > max(self._heartbeat_interval, 1.0):
            self._logger.debug("Opóźnienie streamu strategii %.2f s", drift)

    def _register_instance(self) -> None:
        with self._active_lock:
            self._active_instances.add(self)

    def _unregister_instance(self) -> None:
        with self._active_lock:
            try:
                self._active_instances.remove(self)
            except KeyError:
                pass

    @classmethod
    def close_all_active(cls) -> None:
        with cls._active_lock:
            active = list(cls._active_instances)
        for pipeline in active:
            try:
                pipeline.stop()
            except Exception:  # pragma: no cover
                _LOGGER.debug("Nie udało się zamknąć StreamingStrategyFeed", exc_info=True)

    @staticmethod
    def _float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_to_snapshot(event: Mapping[str, Any]) -> MarketSnapshot | None:
        symbol_raw = event.get("symbol") or event.get("pair") or event.get("instrument")
        if not symbol_raw:
            return None
        symbol = str(symbol_raw)
        timestamp_raw = (
            event.get("timestamp")
            or event.get("time")
            or event.get("ts")
            or time.time()
        )
        try:
            timestamp_value = float(timestamp_raw)
        except (TypeError, ValueError):
            timestamp_value = time.time()
        if timestamp_value > 1e15:
            timestamp_ms = int(timestamp_value)
        elif timestamp_value > 1e12:
            timestamp_ms = int(timestamp_value)
        else:
            timestamp_ms = int(timestamp_value * 1000.0)

        last_price = StreamingStrategyFeed._float(event.get("last_price") or event.get("price") or event.get("close"))
        if last_price is None:
            return None
        open_price = StreamingStrategyFeed._float(event.get("open_price") or event.get("open")) or last_price
        high_price = StreamingStrategyFeed._float(event.get("high_24h") or event.get("high")) or max(open_price, last_price)
        low_price = StreamingStrategyFeed._float(event.get("low_24h") or event.get("low")) or min(open_price, last_price)
        volume = StreamingStrategyFeed._float(
            event.get("volume_24h_base")
            or event.get("volume")
            or event.get("base_volume")
        ) or 0.0

        indicators: dict[str, float] = {}
        for key in (
            "best_bid",
            "best_ask",
            "price_change_percent",
            "volume_24h_quote",
            "funding_rate",
        ):
            value = StreamingStrategyFeed._float(event.get(key))
            if value is not None:
                indicators[key] = value

        return MarketSnapshot(
            symbol=symbol,
            timestamp=timestamp_ms,
            open=open_price,
            high=high_price,
            low=low_price,
            close=last_price,
            volume=volume,
            indicators=indicators,
        )


def _response_to_snapshots(symbol: str, response: object) -> list[MarketSnapshot]:
    if not response or not getattr(response, "rows", None):
        return []
    columns = [str(column).lower() for column in getattr(response, "columns", [])]
    index = {column: idx for idx, column in enumerate(columns)}
    required = {"open", "high", "low", "close"}
    if not required.issubset(index):
        return []
    time_idx = index.get("open_time")
    if time_idx is None:
        time_idx = index.get("timestamp")
    if time_idx is None:
        return []
    volume_idx = index.get("volume")
    snapshots: list[MarketSnapshot] = []
    for row in getattr(response, "rows", []):
        if len(row) <= index["close"]:
            continue
        snapshots.append(
            MarketSnapshot(
                symbol=symbol,
                timestamp=int(float(row[time_idx])),
                open=float(row[index["open"]]),
                high=float(row[index["high"]]),
                low=float(row[index["low"]]),
                close=float(row[index["close"]]),
                volume=float(row[volume_idx]) if volume_idx is not None else 0.0,
            )
        )
    snapshots.sort(key=lambda item: item.timestamp)
    return snapshots


def _create_cached_source(adapter: ExchangeAdapter, environment) -> CachedOHLCVSource:
    cache_root = Path(environment.data_cache_path)
    data_source_cfg = getattr(environment, "data_source", None)
    enable_snapshots = True
    namespace = resolve_cache_namespace(environment)
    offline_mode = bool(getattr(environment, "offline_mode", False))
    allow_network_upstream = not offline_mode
    if data_source_cfg is not None:
        enable_snapshots = bool(getattr(data_source_cfg, "enable_snapshots", True))
    if offline_mode:
        enable_snapshots = False

    return create_cached_ohlcv_source(
        adapter,
        cache_directory=cache_root / "ohlcv_parquet",
        manifest_path=cache_root / "ohlcv_manifest.sqlite",
        enable_snapshots=enable_snapshots,
        allow_network_upstream=allow_network_upstream,
        namespace=namespace,
    )


def _build_streaming_feed(
    *,
    stream_config: Mapping[str, Any] | None,
    stream_settings: Mapping[str, Any] | None,
    adapter_metrics: MetricsRegistry | None,
    base_feed: StrategyDataFeed,
    symbols_map: Mapping[str, Sequence[str]],
    exchange: str,
    environment_name: str | None = None,
    bootstrap: bool = False,
    **_: Any,
) -> "StreamingStrategyFeed | None":
    if stream_config is None and stream_settings is None:
        return None

    stream_settings = stream_settings or {}
    stream_config = stream_config or {}
    base_url = stream_config.get("base_url") or stream_settings.get("base_url")
    path = stream_config.get("path") or stream_settings.get("path")
    channel_param = (
        stream_config.get("channel_param")
        or stream_settings.get("channel_param")
        or stream_settings.get("param")
    )
    if not base_url or not path or not channel_param:
        return None

    channels = stream_config.get("channels")
    if not channels:
        channels = [
            channel_param.join(values)
            for values in symbols_map.values()
        ]

    exchange_factory = stream_config.get("exchange_factory")
    if callable(exchange_factory):
        try:
            exchange = exchange_factory()
        except Exception:  # pragma: no cover
            _LOGGER.exception("Nie udało się zbudować adaptera do streamu long-poll")
            return None
    else:
        exchange = stream_config.get("exchange") or stream_settings.get("exchange") or exchange
    if not exchange:
        return None

    header_keys = (
        "public_headers",
        "public_header",
        "headers",
        "header",
    )
    headers = None
    for key in header_keys:
        value = stream_settings.get(key)
        if isinstance(value, Mapping):
            headers = value
            break

    params_keys = (
        "public_params",
        "public_param",
        "params",
        "param",
    )
    def _build_params() -> Mapping[str, object] | None:
        base_params = {}
        for key in params_keys:
            raw = stream_settings.get(key)
            if isinstance(raw, Mapping):
                base_params.update(raw)
        scope_params = stream_settings.get("scope_params")
        if isinstance(scope_params, Mapping):
            base_params.update(scope_params)
        return base_params or None

    serializer = stream_settings.get("public_channel_serializer") or stream_settings.get("channel_serializer")
    channel_serializer = serializer if callable(serializer) else None
    if channel_serializer is None:
        separator = stream_settings.get("public_channel_separator")
        if separator is None:
            separator = stream_settings.get("channel_separator")
        if isinstance(separator, str) and separator:
            channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731

    def _build_body_params() -> Mapping[str, object] | None:
        body: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body.update(base_body)
        scope_body = stream_settings.get("public_body_params")
        if isinstance(scope_body, Mapping):
            body.update(scope_body)
        return body or None

    body_encoder = stream_settings.get("public_body_encoder")
    if body_encoder is None:
        body_encoder = stream_settings.get("body_encoder")

    jitter_raw = stream_settings.get("jitter")
    if isinstance(jitter_raw, Sequence):
        try:
            jitter = tuple(float(item) for item in jitter_raw)
        except (TypeError, ValueError):
            jitter = (0.05, 0.30)
        if len(jitter) != 2:
            jitter = (0.05, 0.30)
    else:
        jitter = (0.05, 0.30)

    poll_interval = float(stream_settings.get("poll_interval", 0.5))
    timeout = float(stream_settings.get("timeout", 10.0))
    max_retries = int(stream_settings.get("max_retries", 3))
    backoff_base = float(stream_settings.get("backoff_base", 0.25))
    backoff_cap = float(stream_settings.get("backoff_cap", 2.0))
    http_method = stream_settings.get("public_method") or stream_settings.get("method", "GET")
    params_in_body = bool(stream_settings.get("public_params_in_body", stream_settings.get("params_in_body", False)))
    channels_in_body = bool(stream_settings.get("public_channels_in_body", stream_settings.get("channels_in_body", False)))
    cursor_in_body = bool(stream_settings.get("public_cursor_in_body", stream_settings.get("cursor_in_body", False)))

    buffer_size_raw = stream_settings.get("buffer_size", 256)
    try:
        feed_buffer_size = int(buffer_size_raw)
    except (TypeError, ValueError):
        feed_buffer_size = 256
    if feed_buffer_size < 1:
        feed_buffer_size = 1

    stream_buffer_raw = stream_settings.get("public_buffer_size")
    if stream_buffer_raw is None:
        stream_buffer_size = feed_buffer_size
    else:
        try:
            stream_buffer_size = int(stream_buffer_raw)
        except (TypeError, ValueError):
            stream_buffer_size = feed_buffer_size
        if stream_buffer_size < 1:
            stream_buffer_size = 1

    def _factory() -> LocalLongPollStream:
        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=exchange,
            scope="public",
            environment=environment_name or "",
            params=_build_params(),
            headers=headers,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter,
            channel_param=channel_param,
            cursor_param=stream_settings.get("cursor_param"),
            initial_cursor=stream_settings.get("initial_cursor"),
            channel_serializer=channel_serializer,
            http_method=str(http_method or "GET"),
            params_in_body=params_in_body,
            channels_in_body=channels_in_body,
            cursor_in_body=cursor_in_body,
            body_params=_build_body_params(),
            body_encoder=body_encoder,
            buffer_size=stream_buffer_size,
            metrics_registry=adapter_metrics,
        ).start()

    heartbeat_interval = float(stream_settings.get("heartbeat_interval", 15.0))
    idle_timeout_raw = stream_settings.get("idle_timeout")
    idle_timeout = None if idle_timeout_raw in (None, "") else float(idle_timeout_raw)
    restart_delay = float(stream_settings.get("restart_delay", 5.0))

    feed = StreamingStrategyFeed(
        history_feed=base_feed,
        stream_factory=_factory,
        symbols_map=symbols_map,
        buffer_size=feed_buffer_size,
        heartbeat_interval=heartbeat_interval,
        idle_timeout=idle_timeout,
        restart_delay=restart_delay,
        logger=_LOGGER,
    )

    if bootstrap:
        starter = getattr(feed, "start", None)
        if callable(starter):
            starter()

    return feed


def _ensure_local_market_data_availability(
    environment,
    data_source: CachedOHLCVSource,
    markets: Mapping[str, MarketMetadata],
    interval: str,
    *,
    backfill_service: OHLCVBackfillService | None = None,
    adapter: ExchangeAdapter | None = None,
) -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    scheduler = BackfillScheduler(
        data_source,
        backfill_service=backfill_service,
        adapter=adapter,
        default_columns=_DEFAULT_OHLCV_COLUMNS,
    )
    scheduler.ensure_ohlcv_availability(
        symbols=markets.keys(),
        interval=interval,
        environment=environment,
        now_ms=now_ms,
    )
