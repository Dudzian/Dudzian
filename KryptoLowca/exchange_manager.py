# -*- coding: utf-8 -*-
"""Lekki wrapper zapewniający kompatybilność z historycznym API."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import sys
import time
import types
import weakref
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from KryptoLowca.exchanges import (
    AdapterError,
    BaseExchangeAdapter,
    create_exchange_adapter,
)

# ---- Alerts (z fallbackiem, gdy moduł nie istnieje) -------------------------
try:  # pragma: no cover
    from bot_core.alerts import (
        AlertEvent,
        AlertSeverity,
        BotError,
        emit_alert,
        get_alert_dispatcher,
    )
except Exception:  # pragma: no cover
    class AlertSeverity:
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    class BotError(RuntimeError):
        severity = AlertSeverity.ERROR
        source = "general"

    def emit_alert(message: str, *, severity: str, source: str, context: Optional[Dict[str, Any]] = None) -> None:
        logging.getLogger(__name__).critical("[ALERT:FALLBACK] %s | src=%s | ctx=%s", message, source, context or {})

    class _DummyDispatcher:
        def register(self, callback: Callable[[Any], None], *, name: str) -> str:
            return name
        def unregister(self, token: str) -> None:
            return None

    def get_alert_dispatcher() -> _DummyDispatcher:
        return _DummyDispatcher()

    @dataclass
    class AlertEvent:
        message: str
        severity: str = AlertSeverity.INFO
        source: str = "general"
        context: Dict[str, Any] = field(default_factory=dict)

# ---- Telemetria (opcjonalna) ------------------------------------------------
try:  # pragma: no cover
    from KryptoLowca.telemetry import TelemetryWriter  # type: ignore
except Exception:  # pragma: no cover
    class TelemetryWriter:  # no-op fallback
        def __init__(self, *_, **__): ...
        def write_snapshot(self, *_args, **_kwargs): ...

# ---- ccxt async importy odporne na różne wersje / brak biblioteki -----------
try:  # pragma: no cover - importowany tylko jeśli ccxt jest dostępne
    import ccxt.async_support as ccxt_async  # type: ignore
except Exception:  # pragma: no cover - starsze wersje ccxt
    try:
        import ccxt.asyncio as ccxt_async  # type: ignore
    except Exception:  # pragma: no cover - brak ccxt
        # Utwórz atrapę modułu, aby testy mogły się odwoływać do ccxt.asyncio
        ccxt_async = types.ModuleType("ccxt.asyncio")  # type: ignore
        ccxt_module = sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
        setattr(ccxt_module, "asyncio", ccxt_async)
        sys.modules["ccxt.asyncio"] = ccxt_async

from bot_core.exchanges.core import Mode

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = [
    "ExchangeManager",
    "ExchangeError",
    "AuthenticationError",
    "OrderResult",
]

# ------------------------------- Telemetria ----------------------------------
@dataclass(slots=True)
class _EndpointMetrics:
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0


@dataclass(slots=True)
class _APIMetrics:
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    consecutive_errors: int = 0
    window_calls: int = 0
    window_errors: int = 0
    last_endpoint: Optional[str] = None


@dataclass(slots=True)
class _RateLimitBucket:
    """Opis pojedynczego limitu czasowego API."""
    name: str
    capacity: int
    window_seconds: float
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)
    alert_active: bool = False
    last_usage: float = 0.0
    max_usage: float = 0.0

    def reset(self, *, now: Optional[float] = None, hard: bool = False) -> None:
        self.count = 0
        self.window_start = now if now is not None else time.monotonic()
        self.alert_active = False
        self.last_usage = 0.0
        if hard:
            self.max_usage = 0.0

    def snapshot(self) -> Dict[str, Any]:
        remaining = max(0.0, self.window_seconds - (time.monotonic() - self.window_start))
        return {
            "name": self.name,
            "capacity": self.capacity,
            "window_seconds": self.window_seconds,
            "count": self.count,
            "usage": self.last_usage,
            "max_usage": self.max_usage,
            "reset_in_seconds": remaining,
            "alert_active": self.alert_active,
        }


@dataclass(slots=True)
class _SessionState:
    refresh_callback: Callable[[], Any]
    default_ttl: float
    margin: float
    failure_cooldown: float
    expires_at: float
    lock_until: float = 0.0
    consecutive_failures: int = 0

# ------------------------------- Wyjątki -------------------------------------
class ExchangeError(BotError):
    """Podstawowy wyjątek warstwy wymiany."""
    severity = AlertSeverity.ERROR
    source = "exchange"


class AuthenticationError(ExchangeError):
    """Błąd uwierzytelniania API giełdy."""
    severity = AlertSeverity.CRITICAL

# -------------------------- Wynik zlecenia -----------------------------------
@dataclass(slots=True)
class OrderResult:
    """Minimalna struktura opisująca wynik zlecenia."""
    id: Any
    symbol: str
    side: str
    qty: float
    price: Optional[float]
    status: str

# ---------------------------- ExchangeManager --------------------------------
class ExchangeManager:
    """Asynchroniczny wrapper wykorzystywany w testach jednostkowych."""

    def __init__(self, exchange, *, user_id: Optional[int] = None) -> None:
        self.exchange = exchange
        self._adapter: Optional[BaseExchangeAdapter] = None
        self._current_exchange_name: Optional[str] = getattr(exchange, "id", None)
        self._user_id = user_id
        self._retry_attempts = 1
        self._retry_delay = 0.05
        self.mode = Mode.PAPER
        self._markets: Dict[str, Dict[str, Any]] = {}

        # Telemetria / alerty / throttling
        self._db_manager: Optional[Any] = None
        self._alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._alert_listener_token: Optional[str] = None
        self._alert_dispatcher = get_alert_dispatcher()
        self._alert_listener_ref: Optional[Callable[[AlertEvent], None]] = None

        self._metrics = _APIMetrics()
        self._endpoint_metrics: Dict[str, _EndpointMetrics] = defaultdict(lambda: _EndpointMetrics())
        self._throttle_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._min_interval = 0.0
        self._last_request_ts = 0.0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._rate_limit_window = 60.0
        self._rate_limit_per_minute: Optional[int] = None
        self._max_calls_per_window: Optional[int] = None
        self._rate_limit_buckets: List[_RateLimitBucket] = []
        self._alert_usage_threshold = 0.85
        self._rate_alert_active = False
        self._alert_cooldown_seconds = 5.0
        self._alert_last: Dict[str, float] = {}
        self._error_alert_threshold = 3
        self._metrics_log_interval = 30.0
        self._last_metrics_log = 0.0

        # Telemetria rozszerzona (opcjonalna)
        self._telemetry_schema_version = 1
        self._telemetry_writer: Optional[Any] = None

        self._session_state: Optional[_SessionState] = None

    # -------------------------------- Factory --------------------------------
    @classmethod
    async def create(
        cls,
        config,
        db_manager: Optional[Any] = None,
        security_manager: Optional[Any] = None,
    ) -> "ExchangeManager":
        if ccxt_async is None:
            raise ExchangeError("Biblioteka ccxt nie jest dostępna w trybie asynchronicznym.")

        exchange_id = getattr(config, "exchange_name", "binance")
        adapter_options = {
            "api_key": getattr(config, "api_key", ""),
            "api_secret": getattr(config, "api_secret", ""),
            "sandbox": getattr(config, "testnet", True),
        }
        extra_options = getattr(config, "ccxt_options", None)
        if isinstance(extra_options, dict):
            adapter_options.update(extra_options)

        try:
            adapter = create_exchange_adapter(exchange_id, **adapter_options)
            exchange = await adapter.connect()
        except AdapterError as exc:
            auth_error_cls = None
            try:
                from ccxt.base.errors import AuthenticationError as CCXTAuthError  # type: ignore
                auth_error_cls = CCXTAuthError
            except Exception:  # pragma: no cover - brak ccxt
                auth_error_cls = None

            if auth_error_cls is not None and isinstance(exc.__cause__, auth_error_cls):
                raise AuthenticationError(str(exc.__cause__)) from exc
            raise ExchangeError(str(exc)) from exc

        user_id = None
        if db_manager is not None:
            try:
                user_id = await db_manager.ensure_user("system@bot")
            except Exception:
                logger.warning("Nie udało się utworzyć użytkownika w bazie – kontynuuję bez ID.")

        manager = cls(exchange, user_id=user_id)
        manager._adapter = adapter
        manager._current_exchange_name = str(exchange_id).lower()
        manager._db_manager = db_manager
        manager.mode = getattr(config, "mode", Mode.PAPER)
        if isinstance(manager.mode, str):
            try:
                manager.mode = Mode(manager.mode)
            except ValueError:
                pass

        require_demo = bool(getattr(config, "require_demo_mode", False))
        if require_demo and not getattr(config, "testnet", True):
            raise ExchangeError(
                "Polityka bezpieczeństwa wymaga uruchomienia bota w trybie testnet/paper na tym etapie."
            )

        # Konfiguracja limitów/alertów z configu
        per_minute = max(0, int(getattr(config, "rate_limit_per_minute", 0) or 0))
        window_seconds = float(getattr(config, "rate_limit_window_seconds", 60.0) or 60.0)
        manager._rate_limit_window = max(0.1, window_seconds)
        manager.configure_rate_limits(
            per_minute=per_minute or None,
            window_seconds=manager._rate_limit_window,
            buckets=getattr(config, "rate_limit_buckets", None),
        )
        manager._alert_usage_threshold = max(
            0.1,
            min(1.0, float(getattr(config, "rate_limit_alert_threshold", 0.85) or 0.85)),
        )
        manager._error_alert_threshold = max(1, int(getattr(config, "error_alert_threshold", 3) or 3))

        # Telemetria – interwał logowania i schema version
        telemetry_interval = getattr(config, "metrics_log_interval", None)
        telemetry_interval = getattr(config, "telemetry_log_interval_s", telemetry_interval)
        if telemetry_interval is not None:
            manager.set_metrics_log_interval(telemetry_interval)
        schema_version = getattr(config, "telemetry_schema_version", None)
        if schema_version is not None:
            try:
                manager._telemetry_schema_version = int(schema_version)
            except (TypeError, ValueError):
                logger.warning("Nieprawidłowa wartość telemetry_schema_version: %s", schema_version)

        # Telemetry writer (opcjonalnie przez fabrykę lub ścieżkę)
        storage_factory = getattr(config, "telemetry_writer_factory", None)
        if callable(storage_factory):
            try:
                manager._telemetry_writer = storage_factory()
            except Exception as exc:
                logger.error("Nie udało się utworzyć telemetry writer: %s", exc)
        else:
            storage_path = getattr(config, "telemetry_storage_path", None)
            if storage_path:
                try:
                    mode_str = getattr(manager.mode, "value", manager.mode)
                    manager._telemetry_writer = TelemetryWriter(
                        storage_path=storage_path,
                        exchange=exchange_id,
                        mode=str(mode_str).lower(),
                        grpc_target=getattr(config, "telemetry_grpc_target", None),
                        aggregate_intervals=getattr(config, "telemetry_aggregate_intervals", (1, 10, 60)),
                    )
                except Exception as exc:
                    logger.error("Nie udało się utworzyć lokalnego telemetry writer: %s", exc)

        rate_limit_ms = getattr(exchange, "rateLimit", None)
        try:
            manager._min_interval = max(0.0, float(rate_limit_ms) / 1000.0) if rate_limit_ms else 0.0
        except (TypeError, ValueError):  # pragma: no cover
            manager._min_interval = 0.0
        manager._window_start = time.monotonic()
        manager._window_count = 0
        manager.set_retry_policy(
            attempts=getattr(config, "retry_attempts", manager._retry_attempts),
            delay=getattr(config, "retry_delay", manager._retry_delay),
        )
        return manager

    # --------------------------- konfiguracja ---------------------------------
    def set_retry_policy(self, *, attempts: int, delay: float) -> None:
        """Skonfiguruj politykę ponawiania żądań."""
        try:
            attempts_int = max(0, int(attempts))
        except Exception:
            attempts_int = self._retry_attempts
        try:
            delay_float = max(0.0, float(delay))
        except Exception:
            delay_float = self._retry_delay
        self._retry_attempts = attempts_int
        self._retry_delay = delay_float

    def set_metrics_log_interval(self, seconds: float) -> None:
        """Ustaw minimalny odstęp między zapisami telemetrii API do bazy."""
        try:
            interval = max(0.0, float(seconds))
        except Exception:
            interval = 0.0
        self._metrics_log_interval = interval

    def configure_rate_limits(
        self,
        *,
        per_minute: Optional[int] = None,
        window_seconds: float = 60.0,
        buckets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Zaktualizuj konfigurację limitów API i wyzeruj liczniki."""
        self._rate_limit_window = max(0.1, float(window_seconds or 60.0))
        self._rate_limit_per_minute = None
        self._max_calls_per_window = None
        new_buckets: List[_RateLimitBucket] = []

        if per_minute:
            try:
                per_minute_int = max(0, int(per_minute))
            except Exception:
                per_minute_int = 0
            if per_minute_int > 0:
                calls_per_window = per_minute_int * (self._rate_limit_window / 60.0)
                capacity = max(1, int(math.floor(calls_per_window)))
                new_buckets.append(
                    _RateLimitBucket(name="global", capacity=capacity, window_seconds=self._rate_limit_window)
                )
                self._rate_limit_per_minute = per_minute_int
                self._max_calls_per_window = capacity

        if buckets:
            for raw in buckets:
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("name") or f"bucket_{len(new_buckets) + 1}")
                try:
                    capacity = int(raw.get("capacity", 0))
                    window = float(raw.get("window_seconds", 0.0))
                except Exception:
                    continue
                if capacity <= 0 or window <= 0:
                    continue
                new_buckets.append(
                    _RateLimitBucket(name=name, capacity=capacity, window_seconds=float(window))
                )

        if not new_buckets:
            new_buckets.append(
                _RateLimitBucket(
                    name="unbounded",
                    capacity=1_000_000_000,
                    window_seconds=self._rate_limit_window,
                )
            )

        self._rate_limit_buckets = new_buckets
        self._reset_rate_windows()

    async def switch_exchange(self, exchange_name: str, **options: Any) -> None:
        """Dynamiczne przełączenie aktywnej giełdy w trakcie działania managera."""
        if not exchange_name:
            raise ValueError("exchange_name jest wymagane")

        adapter = create_exchange_adapter(exchange_name, **options)
        exchange = await adapter.connect()

        await self._release_exchange()
        self.exchange = exchange
        self._adapter = adapter
        self._current_exchange_name = str(exchange_name).lower()
        self._markets.clear()
        self._metrics = _APIMetrics()
        self._endpoint_metrics.clear()
        self._reset_rate_windows()

    def _reset_rate_windows(self) -> None:
        now = time.monotonic()
        self._window_start = now
        self._window_count = 0
        self._metrics.window_calls = 0
        self._metrics.window_errors = 0
        self._rate_alert_active = False
        for bucket in self._rate_limit_buckets:
            bucket.reset(now=now, hard=True)

    def _ensure_rate_buckets(self) -> None:
        if not self._rate_limit_buckets:
            self._rate_limit_buckets = [
                _RateLimitBucket(
                    name="unbounded",
                    capacity=1_000_000_000,
                    window_seconds=self._rate_limit_window,
                )
            ]

    def configure_session_monitor(
        self,
        refresh_callback: Callable[[], Any | Awaitable[Any]],
        *,
        ttl_seconds: float,
        margin_seconds: float = 60.0,
        failure_cooldown: float = 180.0,
    ) -> None:
        ttl = max(float(ttl_seconds), 30.0)
        margin = max(float(margin_seconds), 5.0)
        cooldown = max(float(failure_cooldown), 30.0)
        expires_at = time.time() + ttl
        self._session_state = _SessionState(
            refresh_callback=refresh_callback,
            default_ttl=ttl,
            margin=margin,
            failure_cooldown=cooldown,
            expires_at=expires_at,
        )

    def update_session_expiry(self, ttl_seconds: float) -> None:
        state = self._session_state
        if state is None:
            return
        ttl = max(float(ttl_seconds), 1.0)
        state.expires_at = time.time() + ttl
        state.default_ttl = ttl

    async def _ensure_session_valid(self) -> None:
        state = self._session_state
        if state is None:
            return
        now = time.time()
        if now < state.lock_until:
            raise AuthenticationError("Sesja API zablokowana po nieudanych próbach odświeżenia.")
        if now <= state.expires_at - state.margin:
            return
        await self._refresh_session(state)

    async def _refresh_session(self, state: _SessionState) -> None:
        async with self._session_lock:
            now = time.time()
            if now < state.lock_until:
                raise AuthenticationError("Sesja API zablokowana po nieudanych próbach odświeżenia.")
            if now <= state.expires_at - state.margin:
                return
            try:
                result = state.refresh_callback()
                if inspect.isawaitable(result):
                    result = await result
                new_expiry: Optional[float] = None
                if isinstance(result, dict):
                    if "expires_at" in result:
                        new_expiry = float(result["expires_at"])
                    elif "expires_in" in result:
                        new_expiry = time.time() + float(result["expires_in"])
                elif isinstance(result, (int, float)):
                    new_expiry = time.time() + float(result)
                if new_expiry is None:
                    new_expiry = time.time() + state.default_ttl
                state.expires_at = new_expiry
                state.consecutive_failures = 0
            except Exception as exc:
                state.consecutive_failures += 1
                state.lock_until = time.time() + state.failure_cooldown
                context = {
                    "error": str(exc),
                    "failures": state.consecutive_failures,
                    "cooldown_seconds": state.failure_cooldown,
                }
                self._raise_alert("Odświeżanie tokenu API nie powiodło się", context=context, key="session_refresh")
                raise AuthenticationError("Nie udało się odświeżyć sesji API") from exc

    # --------------------------- operacje pomocnicze ---------------------------
    async def _before_call(self, endpoint: str) -> None:
        await self._ensure_session_valid()
        wait_time = 0.0
        alerts: List[Tuple[str, Dict[str, Any], str]] = []
        async with self._throttle_lock:
            now = time.monotonic()
            if now - self._window_start >= self._rate_limit_window:
                self._reset_rate_windows()

            self._ensure_rate_buckets()

            for bucket in self._rate_limit_buckets:
                elapsed = now - bucket.window_start
                if elapsed >= bucket.window_seconds:
                    bucket.reset(now=now, hard=True)
                    elapsed = 0.0

                projected = bucket.count + 1
                usage_pct = projected / bucket.capacity if bucket.capacity > 0 else 0.0
                bucket.last_usage = min(usage_pct, 1.0 if bucket.capacity > 0 else usage_pct)
                bucket.max_usage = max(bucket.max_usage, bucket.last_usage)

                if bucket.capacity > 0 and bucket.count >= bucket.capacity:
                    wait_time = max(wait_time, bucket.window_seconds - elapsed)

                if (
                    bucket.capacity > 0
                    and usage_pct >= self._alert_usage_threshold
                    and not bucket.alert_active
                ):
                    bucket.alert_active = True
                    context = {
                        "endpoint": endpoint,
                        "bucket": bucket.name,
                        "usage": bucket.last_usage,
                        "capacity": bucket.capacity,
                        "window_seconds": bucket.window_seconds,
                        "reset_in_seconds": max(0.0, bucket.window_seconds - elapsed),
                    }
                    alerts.append(
                        (
                            f"Zużyto {bucket.last_usage * 100:.1f}% limitu API dla kubełka '{bucket.name}'.",
                            context,
                            f"rate_limit::{bucket.name}",
                        )
                    )

                bucket.count = projected

            if self._rate_limit_buckets:
                self._window_count = self._rate_limit_buckets[0].count
                self._rate_alert_active = any(bucket.alert_active for bucket in self._rate_limit_buckets)

            if self._min_interval > 0:
                delta = now - self._last_request_ts
                if delta < self._min_interval:
                    wait_time = max(wait_time, self._min_interval - delta)

            self._metrics.window_calls = self._window_count
            self._last_request_ts = now + wait_time if wait_time > 0 else now

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        if self._db_manager and hasattr(self._db_manager, "log_rate_limit_snapshot"):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - brak pętli event
                loop = None
            if loop is not None and self._rate_limit_buckets:
                buckets_snapshot = [bucket.snapshot() for bucket in self._rate_limit_buckets]
                payload = {
                    "endpoint": endpoint,
                    "timestamp": time.time(),
                    "user_id": self._user_id,
                    "context": {
                        "limit_triggered": any(bucket.get("alert_active") for bucket in buckets_snapshot),
                        "buckets": buckets_snapshot,
                    },
                }

                async def _persist_rate_limits() -> None:
                    try:
                        await self._db_manager.log_rate_limit_snapshot(payload)
                    except Exception:
                        logger.exception("Błąd zapisu snapshotu limitów API")

                loop.create_task(_persist_rate_limits())

        for message, context, key in alerts:
            self._raise_alert(message, context=context, key=key)

    def _record_metrics(self, endpoint: str, latency_ms: float, *, success: bool) -> None:
        metrics = self._metrics
        metrics.total_calls += 1
        metrics.avg_latency_ms = (
            ((metrics.avg_latency_ms * (metrics.total_calls - 1)) + latency_ms) / metrics.total_calls
        )
        metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
        metrics.last_latency_ms = latency_ms
        metrics.last_endpoint = endpoint

        endpoint_metrics = self._endpoint_metrics[endpoint]
        endpoint_metrics.total_calls += 1
        endpoint_metrics.avg_latency_ms = (
            (endpoint_metrics.avg_latency_ms * (endpoint_metrics.total_calls - 1) + latency_ms)
            / endpoint_metrics.total_calls
        )
        endpoint_metrics.max_latency_ms = max(endpoint_metrics.max_latency_ms, latency_ms)
        endpoint_metrics.last_latency_ms = latency_ms

        if success:
            metrics.consecutive_errors = 0
        else:
            metrics.total_errors += 1
            metrics.window_errors += 1
            metrics.consecutive_errors += 1
            endpoint_metrics.total_errors += 1
            if self._error_alert_threshold and metrics.consecutive_errors == self._error_alert_threshold:
                context = {
                    "endpoint": endpoint,
                    "consecutive_errors": metrics.consecutive_errors,
                }
                self._raise_alert(
                    f"Przekroczono próg błędów API ({metrics.consecutive_errors}) dla {endpoint}",
                    context=context,
                    key="error",
                )

        self._schedule_metrics_snapshot()

    def _register_error(self, endpoint: str, exc: Exception, *, final: bool) -> None:
        level = logging.ERROR if final else logging.WARNING
        logger.log(level, "Wywołanie %s nie powiodło się: %s", endpoint, exc)
        if final and self._db_manager and self._user_id:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover
                pass
            else:
                loop.create_task(
                    self._db_manager.log(
                        self._user_id,
                        "ERROR",
                        f"Wywołanie {endpoint} zakończyło się błędem: {exc}",
                        category="exchange",
                    )
                )

    def _raise_alert(self, message: str, context: Optional[Dict[str, Any]] = None, *, key: str) -> None:
        now = time.monotonic()
        last = self._alert_last.get(key, 0.0)
        if now - last < self._alert_cooldown_seconds:
            return
        self._alert_last[key] = now
        payload = dict(context or {})
        if self._user_id is not None:
            payload.setdefault("user_id", self._user_id)
        logger.critical("[ALERT] %s | context=%s", message, payload)
        emit_alert(
            message,
            severity=AlertSeverity.CRITICAL,
            source="exchange",
            context=payload,
        )
        if self._db_manager and self._user_id:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover
                return
            loop.create_task(
                self._db_manager.log(
                    self._user_id,
                    "CRITICAL",
                    message,
                    category="exchange",
                    context=payload,
                )
            )

    def _schedule_metrics_snapshot(self) -> None:
        if not self._db_manager or not self._user_id:
            return
        if self._metrics_log_interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_metrics_log < self._metrics_log_interval:
            return
        self._last_metrics_log = now
        snapshot = self.get_api_metrics()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _persist() -> None:
            try:
                await self._db_manager.log(
                    self._user_id,
                    "INFO",
                    "API metrics snapshot",
                    category="exchange_metrics",
                    context=snapshot,
                )
            except Exception:
                logger.exception("Błąd zapisu telemetrii API")

        loop.create_task(_persist())

        if hasattr(self._db_manager, "log_performance_metric"):
            async def _persist_metric() -> None:
                try:
                    await self._db_manager.log_performance_metric(
                        {
                            "metric": "exchange_total_calls",
                            "value": snapshot.get("total_calls", 0),
                            "window": snapshot.get("window_calls", 0),
                            "symbol": snapshot.get("last_endpoint"),
                            "mode": getattr(self.mode, "name", str(self.mode)),
                            "context": snapshot,
                        }
                    )
                except Exception:
                    logger.exception("Błąd zapisu metryk wydajności API")

            loop.create_task(_persist_metric())

        # Opcjonalnie, lokalny zapis telemetryjny
        if self._telemetry_writer is not None:
            try:
                self._telemetry_writer.write_snapshot(snapshot)
            except Exception:
                logger.exception("Nie udało się zapisać snapshotu telemetryjnego lokalnie")

    async def _run_with_retry(self, coro_factory: Callable[[], Any], *, endpoint: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._retry_attempts + 1):
            await self._before_call(endpoint)
            start = time.perf_counter()
            try:
                result = await coro_factory()
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_metrics(endpoint, latency_ms, success=False)
                final = attempt == self._retry_attempts
                self._register_error(endpoint, exc, final=final)
                last_exc = exc
                if final:
                    raise
                await asyncio.sleep(self._retry_delay)
            else:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_metrics(endpoint, latency_ms, success=True)
                return result
        if last_exc:
            raise last_exc
        return None

    async def _call_with_metrics(self, endpoint: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        async def _factory() -> Any:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        return await self._run_with_retry(_factory, endpoint=endpoint)

    def _market(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._markets:
            return self._markets[symbol]
        if symbol.upper() in self._markets:
            return self._markets[symbol.upper()]
        return {}

    # --------------------------------- API ------------------------------------
    async def load_markets(self) -> Dict[str, Any]:
        markets = await self._call_with_metrics("load_markets", self.exchange.load_markets)
        self._markets = markets or {}
        return markets

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[Any]:
        if not symbol:
            raise ValueError("Symbol nie może być pusty.")
        if limit <= 0:
            raise ValueError("Limit musi być dodatni.")
        return await self._call_with_metrics(
            "fetch_ohlcv", self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("Dozwolone strony zlecenia: buy/sell.")
        if quantity <= 0:
            raise ValueError("Ilość musi być dodatnia.")
        raw = await self._call_with_metrics(
            "create_market_order", self.exchange.create_market_order, symbol, side, quantity
        )
        return OrderResult(
            id=raw.get("id"),
            symbol=raw.get("symbol", symbol),
            side=raw.get("side", side),
            qty=float(raw.get("amount", quantity)),
            price=raw.get("price"),
            status=str(raw.get("status", "filled")).lower(),
        )

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("Dozwolone strony zlecenia: buy/sell.")
        if quantity <= 0:
            raise ValueError("Ilość musi być dodatnia.")

        qty = self.quantize_amount(symbol, quantity)
        if qty <= 0:
            raise ValueError("Skwantowana ilość wynosi 0.")

        order_type = order_type.lower()
        px: Optional[float] = None
        if order_type != "market":
            if price is None:
                raise ValueError("Cena wymagana dla zleceń LIMIT/STOP.")
            px = self.quantize_price(symbol, float(price))
        else:
            px = float(price) if price is not None else None

        params = dict(params or {})
        if client_order_id:
            params.setdefault("newClientOrderId", client_order_id)
            params.setdefault("clientOrderId", client_order_id)

        notional_price = px if px is not None else price
        if order_type == "market" and notional_price is None:
            last_price = await self._last_price(symbol)
            notional_price = last_price
        min_notional = self.min_notional(symbol)
        if min_notional and notional_price:
            if qty * float(notional_price) < min_notional - 1e-9:
                raise ValueError("Wartość zlecenia poniżej min_notional")

        response = await self._submit_order(
            symbol, side.lower(), order_type, qty, px, params or {}, client_order_id
        )
        amount = float(response.get("amount") or response.get("quantity") or qty)
        filled_price = response.get("price") or px or notional_price or 0.0
        status = str(response.get("status") or "open").lower()
        order_id = response.get("id") or response.get("orderId") or response.get("order_id")
        return OrderResult(
            id=order_id,
            symbol=response.get("symbol", symbol),
            side=response.get("side", side),
            qty=amount,
            price=float(filled_price) if filled_price is not None else None,
            status=status,
        )

    async def fetch_balance(self) -> Dict[str, Any]:
        data = await self._call_with_metrics("fetch_balance", self.exchange.fetch_balance)
        if isinstance(data, dict) and "total" in data:
            return data["total"]
        return data

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        if symbol:
            rows = await self._call_with_metrics(
                "fetch_open_orders", self.exchange.fetch_open_orders, symbol
            )
        else:
            rows = await self._call_with_metrics("fetch_open_orders", self.exchange.fetch_open_orders)
        results: List[OrderResult] = []
        for row in rows or []:
            results.append(
                OrderResult(
                    id=row.get("id"),
                    symbol=row.get("symbol", symbol or ""),
                    side=row.get("side", ""),
                    qty=float(row.get("amount", 0.0)),
                    price=row.get("price"),
                    status=str(row.get("status", "open")).lower(),
                )
            )
        return results

    async def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if not order_id or not symbol:
            raise ValueError("Identyfikator zlecenia i symbol są wymagane.")
        result = await self._call_with_metrics(
            "cancel_order", self.exchange.cancel_order, order_id, symbol
        )
        return bool(result) if result is not None else True

    async def _release_exchange(self) -> None:
        current = getattr(self, "exchange", None)
        close_cb = getattr(current, "close", None)
        if close_cb:
            maybe = close_cb()
            if inspect.isawaitable(maybe):
                await maybe
        if self._adapter is not None:
            try:
                await self._adapter.close()
            finally:
                self._adapter = None
        self._current_exchange_name = None

    async def close(self) -> None:
        await self._release_exchange()
        if self._alert_listener_token is not None:
            self._alert_dispatcher.unregister(self._alert_listener_token)
            self._alert_listener_token = None
            self._alert_listener_ref = None
            self._alert_callback = None

    # ------------------------------- helpers ----------------------------------
    def quantize_amount(self, symbol: str, amount: float) -> float:
        market = self._market(symbol)
        value = float(amount)
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        step = (limits.get("amount") or {}).get("step")
        if step:
            step_val = float(step)
            if step_val > 0:
                value = math.floor(value / step_val) * step_val
        elif precision.get("amount") is not None:
            value = round(value, int(precision["amount"]))
        return max(value, 0.0)

    def quantize_price(self, symbol: str, price: float) -> float:
        market = self._market(symbol)
        value = float(price)
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        step = (limits.get("price") or {}).get("step")
        if step:
            step_val = float(step)
            if step_val > 0:
                value = math.floor(value / step_val) * step_val
        elif precision.get("price") is not None:
            value = round(value, int(precision["price"]))
        return max(value, 0.0)

    def min_notional(self, symbol: str) -> float:
        market = self._market(symbol)
        limits = market.get("limits") or {}
        cost = limits.get("cost") or {}
        value = cost.get("min")
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):  # pragma: no cover
            return 0.0

    async def _last_price(self, symbol: str) -> Optional[float]:
        ticker_fn = getattr(self.exchange, "fetch_ticker", None)
        if not callable(ticker_fn):
            return None
        try:
            ticker = await self._call_with_metrics("fetch_ticker", ticker_fn, symbol)
        except Exception:
            return None
        if not isinstance(ticker, dict):
            return None
        for key in ("last", "close", "bid", "ask"):
            value = ticker.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    async def _submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        params: Dict[str, Any],
        client_order_id: Optional[str],
    ) -> Dict[str, Any]:
        if order_type == "market":
            method = getattr(self.exchange, "create_order", None)
            if callable(method):
                try:
                    return await self._call_with_metrics(
                        "create_order",
                        method,
                        symbol,
                        "market",
                        side,
                        quantity,
                        None,
                        params,
                    )
                except TypeError:
                    return await self._call_with_metrics(
                        "create_order",
                        method,
                        symbol,
                        "market",
                        side,
                        quantity,
                        None,
                    )

            market_method = getattr(self.exchange, "create_market_order", None)
            if market_method is None:
                raise ExchangeError("Exchange does not provide create_order API")
            try:
                return await self._call_with_metrics(
                    "create_market_order", market_method, symbol, side, quantity, params
                )
            except TypeError:
                return await self._call_with_metrics(
                    "create_market_order", market_method, symbol, side, quantity
                )

        method = getattr(self.exchange, "create_order", None)
        if callable(method):
            try:
                return await self._call_with_metrics(
                    "create_order",
                    method,
                    symbol,
                    order_type,
                    side,
                    quantity,
                    price,
                    params,
                )
            except TypeError:
                return await self._call_with_metrics(
                    "create_order",
                    method,
                    symbol,
                    order_type,
                    side,
                    quantity,
                    price,
                )

        limit_method = getattr(self.exchange, "create_limit_order", None)
        if limit_method is None:
            raise ExchangeError("Exchange does not provide limit order API")
        try:
            return await self._call_with_metrics(
                "create_limit_order", limit_method, symbol, side, quantity, price, params
            )
        except TypeError:
            return await self._call_with_metrics(
                "create_limit_order", limit_method, symbol, side, quantity, price
            )

    # ------------------------------ telemetry ---------------------------------
    def register_alert_handler(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Zarejestruj zewnętrzny handler alertów (np. GUI / moduł powiadomień)."""

        self._alert_callback = callback
        if self._alert_listener_token is not None:
            self._alert_dispatcher.unregister(self._alert_listener_token)
            self._alert_listener_token = None
            self._alert_listener_ref = None

        manager_ref = weakref.ref(self)

        def _listener(event: AlertEvent) -> None:
            manager = manager_ref()
            if manager is None:
                return
            if getattr(event, "source", None) != "exchange":
                return
            cb = manager._alert_callback
            if cb is None:
                return
            try:
                cb(event.message, dict(getattr(event, "context", {}) or {}))
            except Exception:  # pragma: no cover
                logger.exception("Alert callback zgłosił wyjątek")

        self._alert_listener_ref = _listener
        self._alert_listener_token = self._alert_dispatcher.register(
            _listener,
            name=f"exchange-{id(self)}",
        )

    def get_rate_limit_snapshot(self) -> List[Dict[str, Any]]:
        """Szybki podgląd stanu kubełków limitów API."""
        return [bucket.snapshot() for bucket in self._rate_limit_buckets]

    def get_api_metrics(self) -> Dict[str, Any]:
        """Zwróć metryki zużycia API (łącznie i per-endpoint)."""
        usage = None
        if self._max_calls_per_window:
            usage = self._window_count / self._max_calls_per_window
        buckets_snapshot = [bucket.snapshot() for bucket in self._rate_limit_buckets]
        return {
            "schema_version": self._telemetry_schema_version,
            "timestamp_ns": time.time_ns(),
            "total_calls": self._metrics.total_calls,
            "total_errors": self._metrics.total_errors,
            "avg_latency_ms": self._metrics.avg_latency_ms,
            "max_latency_ms": self._metrics.max_latency_ms,
            "last_latency_ms": self._metrics.last_latency_ms,
            "consecutive_errors": self._metrics.consecutive_errors,
            "window_calls": self._metrics.window_calls,
            "window_errors": self._metrics.window_errors,
            "last_endpoint": self._metrics.last_endpoint,
            "rate_limit_per_minute": self._rate_limit_per_minute,
            "rate_limit_window_seconds": self._rate_limit_window,
            "current_window_usage": usage,
            "rate_limit_buckets": buckets_snapshot,
            "endpoints": {name: asdict(data) for name, data in self._endpoint_metrics.items()},
        }

    def reset_api_metrics(self) -> None:
        """Wyzeruj liczniki metryk (np. po raporcie)."""
        self._metrics = _APIMetrics()
        self._endpoint_metrics = defaultdict(lambda: _EndpointMetrics())
        self._metrics.consecutive_errors = 0
        self._alert_last.clear()
        self._reset_rate_windows()
        self._last_metrics_log = 0.0
