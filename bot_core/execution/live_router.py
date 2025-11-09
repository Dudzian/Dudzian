"""Router egzekucji dla środowiska live z fallbackami, metrykami i (opcjonalnie) decision logiem."""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import ExchangeAdapter, OrderRequest, OrderResult

try:  # pragma: no cover - optional dependency in stripped builds
    from bot_core.runtime.scheduler import AsyncIOTaskQueue
except Exception:  # pragma: no cover
    AsyncIOTaskQueue = None  # type: ignore[misc,assignment]

# --- Wyjątki giełdowe (opcjonalne – różne gałęzie mogą je mieć/nie mieć)
try:  # pragma: no cover
    from bot_core.exchanges.errors import (
        ExchangeAPIError,
        ExchangeAuthError,
        ExchangeNetworkError,
        ExchangeThrottlingError,
    )
except Exception:  # pragma: no cover - fallback gdy moduł nie istnieje
    class ExchangeAPIError(Exception):
        status_code: int | None = None
        message: str | None = None

    class ExchangeAuthError(Exception):
        pass

    class ExchangeNetworkError(Exception):
        pass

    class ExchangeThrottlingError(Exception):
        pass


# --- Observability (wymagana w buildach produkcyjnych)
try:  # pragma: no cover
    from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Moduł 'bot_core.observability.metrics' jest wymagany przez LiveExecutionRouter. "
        "Upewnij się, że komponenty observability zostały uwzględnione w pakiecie instalacyjnym"
    ) from exc


# --- Podpisywanie decision logu (opcjonalne, z fallbackiem)
try:  # pragma: no cover
    from bot_core.security.signing import (  # type: ignore
        build_hmac_signature as _build_hmac_signature,
        TransactionSigner,
        TransactionSignerSelector,
    )
except Exception:  # pragma: no cover
    def _build_hmac_signature(
        payload: Mapping[str, object],
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA384",
        key_id: str | None = None,
    ) -> Mapping[str, str]:
        algo = algorithm.upper().replace("-", "")
        if algo not in {"HMACSHA384", "HMACSHA256"}:
            algo = "HMAC-SHA384"
        digestmod = hashlib.sha384 if algo == "HMAC-SHA384" else hashlib.sha256
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(key, canonical, digestmod).digest()
        out = {
            "algorithm": "HMAC-SHA384" if digestmod is hashlib.sha384 else "HMAC-SHA256",
            "value": base64.b64encode(sig).decode("ascii"),
        }
        if key_id is not None:
            out["key_id"] = key_id
        return out

    class TransactionSigner:  # type: ignore[too-many-ancestors]
        algorithm = "HMAC-SHA256"
        key_id: str | None = None
        requires_hardware = False

        def sign(self, payload: Mapping[str, object]) -> Mapping[str, str]:  # pragma: no cover - fallback
            raise NotImplementedError

        def describe(self) -> Mapping[str, object]:  # pragma: no cover - fallback
            data: dict[str, object] = {
                "algorithm": self.algorithm,
                "requires_hardware": getattr(self, "requires_hardware", False),
            }
            if self.key_id is not None:
                data["key_id"] = self.key_id
            return data

    class TransactionSignerSelector:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._default: TransactionSigner | None = None

        def resolve(self, _account_id: str | None) -> TransactionSigner | None:  # pragma: no cover - fallback
            return self._default


_LOGGER = logging.getLogger(__name__)


def _build_api_error(message: str, *, status: int = 422, payload: Mapping[str, object] | None = None) -> ExchangeAPIError:
    """Tworzy instancję ``ExchangeAPIError`` niezależnie od wersji modułu błędów."""

    try:
        return ExchangeAPIError(message=message, status_code=status, payload=payload)
    except TypeError:  # pragma: no cover - fallback dla prostszej implementacji
        error = ExchangeAPIError(message)  # type: ignore[arg-type]
        setattr(error, "status_code", status)
        setattr(error, "message", message)
        if payload is not None:
            setattr(error, "payload", payload)
        return error


# === Backoff & CircuitBreaker =================================================

def _exp_backoff_with_jitter(attempt: int, *, base: float = 0.05, cap: float = 0.5) -> float:
    """Exponential backoff z losowym jitterem (pełny jitter)."""
    exp = min(cap, base * (2 ** max(0, attempt - 1)))
    return random.uniform(0.0, exp)


@dataclass(slots=True)
class CircuitBreaker:
    """Prosty breaker (closed → open → half-open). Per giełda w routerze."""
    failure_threshold: int = 5
    open_seconds: float = 5.0
    half_open_max_calls: int = 1

    state: str = field(default="closed", init=False)
    failures: int = field(default=0, init=False)
    opened_at: float = field(default=0.0, init=False)
    half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self, now: float) -> bool:
        with self._lock:
            if self.state == "open":
                if now - self.opened_at >= self.open_seconds:
                    self.state = "half-open"
                    self.half_open_calls = 0
                else:
                    return False
            if self.state == "half-open":
                if self.half_open_calls >= self.half_open_max_calls:
                    return False
                self.half_open_calls += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.failures = 0
            self.half_open_calls = 0

    def record_failure(self, now: float) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = now
                self.half_open_calls = 0


# === API trasowania (kompatybilność) ========================================

@dataclass(slots=True)
class RoutingPlan:
    """Prosty plan – kolejność giełd podczas egzekucji."""
    exchanges: Sequence[str]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        unique: list[str] = []
        for name in self.exchanges:
            n = str(name).strip()
            if not n or n in seen:
                continue
            unique.append(n)
            seen.add(n)
        self.exchanges = tuple(unique)

    def __bool__(self) -> bool:  # pragma: no cover
        return bool(self.exchanges)


@dataclass(slots=True)
class RouteDefinition:
    """Rozbudowana definicja trasy (nazwa, filtry, retry, budżet, metadane)."""
    name: str
    exchanges: Sequence[str]
    symbols: Sequence[str] = ()
    risk_profiles: Sequence[str] = ()
    # Domyślnie jedna próba – dodatkowe retry należy zadeklarować explicite.
    max_retries_per_exchange: int = 1
    latency_budget_ms: float = 250.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def iter_exchanges(self) -> Iterable[str]:
        for ex in self.exchanges:
            n = str(ex).strip()
            if n:
                yield n


@dataclass(slots=True)
class QoSConfig:
    """Konfiguracja kolejkowania i współbieżności LiveExecutionRouter."""

    max_queue_size: int = 1024
    worker_concurrency: int = 4
    per_exchange_concurrency: Mapping[str, int] = field(default_factory=dict)
    priority_resolver: Callable[[OrderRequest, ExecutionContext], int] | None = None
    max_queue_wait_seconds: float | None = None
    ack_queue_size: int = 512


@dataclass(slots=True)
class RouterRuntimeStats:
    """Zrzut stanu runtime kolejki i limiterów routera live."""

    queue_depth: int
    queue_limit: int
    worker_concurrency: int
    per_exchange_limits: Mapping[str, int]
    inflight_by_exchange: Mapping[str, int]
    queue_timeout: float | None
    closed: bool
    mode: str


@dataclass(slots=True)
class _ExecutionPlan:
    """Przygotowany plan egzekucji wykorzystywany przez worker."""

    route_name: str | None
    route_metadata: Mapping[str, str]
    exchanges_and_retries: tuple[tuple[str, int], ...]
    latency_budget_ms: float | None


@dataclass(slots=True)
class AcknowledgementEvent:
    """Sygnalizuje postęp realizacji zlecenia do warstwy decyzyjnej."""

    ack_id: str
    status: str
    exchange: str | None
    order_id: str | None
    client_order_id: str | None
    symbol: str
    portfolio: str
    timestamp: float
    details: Mapping[str, object]


@dataclass(slots=True)
class _QueuedOrder:
    request: OrderRequest
    context: ExecutionContext
    future: asyncio.Future[OrderResult]
    enqueued_at: float
    plan: _ExecutionPlan
    ack_id: str
    final_status_sent: bool = field(default=False, init=False)


# === LiveExecutionRouter =====================================================

class LiveExecutionRouter(ExecutionService):
    """Asynchroniczny router egzekucji dla środowiska live."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ExchangeAdapter],
        # --- tryb 1 (routes) ---
        routes: Sequence[RouteDefinition] | None = None,
        default_route: str | Iterable[str] | None = None,
        # --- tryb 2 (plan) ---
        route_overrides: Mapping[str, Sequence[str]] | None = None,
        # --- decision log (opcjonalny) ---
        decision_log_path: str | os.PathLike[str] | None = None,
        decision_log_hmac_key: bytes | None = None,
        decision_log_key_id: str | None = None,
        decision_log_rotate_bytes: int = 8 * 1024 * 1024,
        decision_log_keep: int = 3,
        # --- metryki / czas ---
        metrics_registry: MetricsRegistry | None = None,
        metrics: MetricsRegistry | None = None,  # alias zgodności
        latency_buckets: Sequence[float] | None = None,
        time_source: Callable[[], float] | None = None,
        # --- breaker / bindings ---
        circuit_breaker_factory: Callable[[], CircuitBreaker] | None = None,
        bindings_capacity: int = 10_000,
        # --- współbieżność / QoS ---
        qos: QoSConfig | None = None,
        io_dispatcher: AsyncIOTaskQueue | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("LiveExecutionRouter wymaga co najmniej jednego adaptera giełdowego")
        self._adapters: dict[str, ExchangeAdapter] = {str(k): v for k, v in adapters.items()}

        # Tryb trasowania
        self._mode: str
        self._routes_by_name: dict[str, RouteDefinition] = {}
        self._routes: tuple[RouteDefinition, ...] = ()
        self._default_named_route: str | None = None
        self._default_plan: RoutingPlan | None = None
        self._overrides: dict[str, RoutingPlan] = {}

        if routes is not None:
            if not routes:
                raise ValueError("Parametr 'routes' nie może być pusty")
            self._mode = "routes"
            self._routes = tuple(routes)
            self._routes_by_name = {r.name: r for r in self._routes}
            if len(self._routes_by_name) != len(self._routes):
                raise ValueError("Nazwy tras muszą być unikalne")
            if isinstance(default_route, str):
                if default_route not in self._routes_by_name:
                    raise KeyError(f"Domyślna trasa '{default_route}' nie istnieje")
                self._default_named_route = default_route
        else:
            self._mode = "plan"
            if not default_route:
                raise ValueError("W trybie 'plan' wymagany jest parametr default_route")
            self._default_plan = RoutingPlan(
                tuple(default_route if not isinstance(default_route, str) else [default_route])
            )
            for sym, route in (route_overrides or {}).items():
                if route:
                    self._overrides[str(sym)] = RoutingPlan(tuple(route))

        # Decision log
        self._decision_log_path: Path | None = None
        self._decision_log_key: bytes | None = None
        self._decision_log_key_id: str | None = decision_log_key_id
        self._decision_log_rotate_bytes = max(0, int(decision_log_rotate_bytes))
        self._decision_log_keep = max(1, int(decision_log_keep))
        if decision_log_path is not None:
            self._decision_log_path = Path(decision_log_path)
            self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            if decision_log_hmac_key is not None:
                if len(decision_log_hmac_key) < 32:
                    raise ValueError("Klucz HMAC decision logu musi mieć co najmniej 32 bajty")
                self._decision_log_key = bytes(decision_log_hmac_key)
        self._log_lock = threading.Lock()

        # Metryki
        self._metrics = metrics_registry or metrics or get_global_metrics_registry()
        buckets = tuple(latency_buckets or (0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000))
        self._m_latency = self._metrics.histogram(
            "live_execution_latency_seconds",
            "Czas realizacji zleceń live (sekundy).",
            buckets=buckets,
        )
        self._m_attempts = self._metrics.counter(
            "live_orders_attempts_total", "Liczba prób egzekucji zleceń (łącznie z fallbackami)."
        )
        self._m_failures = self._metrics.counter(
            "live_orders_failed_total", "Liczba zleceń nieudanych po wszystkich próbach."
        )
        self._m_fallbacks = self._metrics.counter(
            "live_orders_fallback_total", "Liczba zleceń wymagających fallbacku."
        )
        self._m_success = self._metrics.counter(
            "live_orders_success_total", "Liczba zleceń zrealizowanych."
        )
        self._m_orders_total = self._metrics.counter(
            "live_orders_total", "Łączna liczba zleceń obsłużonych przez router live."
        )
        self._m_router_fallbacks = self._metrics.counter(
            "live_router_fallbacks_total",
            "Liczba zleceń wymagających fallbacku (metryka kompatybilności).",
        )
        self._m_router_failures = self._metrics.counter(
            "live_router_failures_total",
            "Liczba zleceń zakończonych niepowodzeniem (metryka kompatybilności).",
        )
        self._m_fill_ratio = self._metrics.histogram(
            "live_orders_fill_ratio",
            "Rozkład wypełnienia zleceń live (0-1).",
            buckets=(0.25, 0.5, 0.75, 1.0),
        )
        self._m_errors = self._metrics.counter(
            "live_orders_errors_total",
            "Liczba błędów podczas obsługi zleceń live.",
        )
        self._g_breaker_open = getattr(self._metrics, "gauge", self._metrics.counter)(
            "live_breaker_open", "Stan breakerów (1=open, 0=closed)."
        )
        gauge_factory = getattr(self._metrics, "gauge", None)
        self._g_queue_depth = None
        if callable(gauge_factory):
            try:
                self._g_queue_depth = gauge_factory(
                    "live_execution_queue_depth",
                    "Aktualna liczba zleceń oczekujących w kolejce LiveExecutionRouter.",
                )
            except Exception:  # pragma: no cover - zależne od implementacji rejestru
                self._g_queue_depth = None

        # Czas i breaker
        self._time = time_source or time.perf_counter
        factory = circuit_breaker_factory or (lambda: CircuitBreaker())
        self._breakers: dict[str, CircuitBreaker] = {name: factory() for name in self._adapters.keys()}

        # LRU powiązań order_id→exchange
        self._bindings_capacity = max(100, int(bindings_capacity))
        self._bindings: OrderedDict[str, str] = OrderedDict()
        self._bindings_lock = threading.Lock()

        # QoS / asyncio infrastruktura
        self._qos = qos or QoSConfig()
        if self._qos.worker_concurrency <= 0:
            raise ValueError("QoS.worker_concurrency musi być dodatnie")
        if self._qos.max_queue_size <= 0:
            raise ValueError("QoS.max_queue_size musi być dodatnie")
        queue_timeout = self._qos.max_queue_wait_seconds
        if queue_timeout is not None and queue_timeout < 0:
            raise ValueError("QoS.max_queue_wait_seconds musi być nieujemne")
        self._queue_timeout = float(queue_timeout) if queue_timeout is not None else None
        self._priority_resolver = self._qos.priority_resolver or (lambda _req, _ctx: 0)
        self._queue_counter = itertools.count()
        self._queue: asyncio.PriorityQueue[tuple[int, int, _QueuedOrder]] | None = None
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._exchange_semaphores: dict[str, asyncio.Semaphore] = {}
        self._ack_queue: asyncio.Queue[AcknowledgementEvent] | None = None
        self._ack_backlog: deque[AcknowledgementEvent] = deque()

        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="LiveExecutionRouterLoop",
            daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait()

        self._closed = False

        if AsyncIOTaskQueue is not None and isinstance(io_dispatcher, AsyncIOTaskQueue):
            self._io_dispatcher: AsyncIOTaskQueue | None = io_dispatcher
        else:
            self._io_dispatcher = None

        self._inflight_by_exchange: dict[str, int] = {}

        if self._g_queue_depth is not None:
            try:
                self._g_queue_depth.set(0.0)
            except Exception:  # pragma: no cover - zależne od backendu metryk
                pass

    # --- Publiczne pomocnicze API -------------------------------------------

    def list_adapters(self) -> tuple[str, ...]:
        return tuple(self._adapters.keys())

    def set_bindings_capacity(self, capacity: int) -> None:
        self._bindings_capacity = max(100, int(capacity))

    def binding_for_order(self, order_id: str) -> str | None:
        with self._bindings_lock:
            return self._bindings.get(order_id)

    def _resolve_account_identifier(
        self,
        request: OrderRequest,
        context: ExecutionContext,
        exchange_name: str,
    ) -> str | None:
        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        for key in ("account", "account_id", "exchange_account"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
        context_meta = context.metadata if isinstance(context.metadata, Mapping) else {}
        for key in ("account", "account_id", "exchange_account"):
            value = context_meta.get(key)
            if value not in (None, ""):
                return str(value)
        if exchange_name:
            return str(exchange_name)
        return None

    def _maybe_prepare_withdrawal_signature(
        self,
        request: OrderRequest,
        context: ExecutionContext,
        exchange_name: str,
    ) -> None:
        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        normalized = dict(metadata)
        existing_signature = normalized.get("hardware_wallet_signature")
        existing_timestamp = normalized.get("hardware_wallet_signed_at")
        existing_account_raw = normalized.get("hardware_wallet_account")
        existing_algorithm = normalized.get("hardware_wallet_algorithm")
        existing_key_id = normalized.get("hardware_wallet_key_id")
        normalized.pop("hardware_wallet_signature", None)
        normalized.pop("hardware_wallet_algorithm", None)
        normalized.pop("hardware_wallet_key_id", None)
        normalized.pop("hardware_wallet_signed_at", None)
        normalized.pop("hardware_wallet_account", None)

        context_meta = context.metadata if isinstance(context.metadata, Mapping) else {}
        operation = str(normalized.get("operation") or context_meta.get("operation") or "").lower()
        requires_hw = bool(normalized.get("requires_hardware_wallet") or context_meta.get("requires_hardware_wallet"))
        if operation in {"withdrawal", "payout"}:
            requires_hw = True

        if not requires_hw:
            if normalized != metadata:
                request.metadata = normalized
            return

        if self._transaction_signers is None:
            if self._require_hardware_wallet:
                raise RuntimeError(
                    "Licencja wymaga portfela sprzętowego dla wypłat, ale nie skonfigurowano podpisującego."
                )
            if normalized != metadata:
                request.metadata = normalized
            return

        account_id = self._resolve_account_identifier(request, context, exchange_name)
        signer = self._transaction_signers.resolve(account_id)
        if signer is None:
            raise RuntimeError(
                f"Brak podpisującego transakcje dla konta '{account_id or 'default'}' wymagającego portfela sprzętowego."
            )
        if self._require_hardware_wallet and not getattr(signer, "requires_hardware", False):
            raise RuntimeError(
                "Licencja wymaga podpisu z portfela sprzętowego dla wypłat, jednak wybrany podpisujący nie korzysta z urządzenia."
            )

        existing_account = str(existing_account_raw or account_id or "") or None

        if (
            isinstance(existing_signature, Mapping)
            and isinstance(existing_timestamp, str)
            and existing_timestamp.strip()
        ):
            timestamp = existing_timestamp.strip()
            signature_for_verification: Mapping[str, Any]
            if (
                existing_key_id
                and "key_id" not in existing_signature
                and isinstance(existing_key_id, str)
                and existing_key_id.strip()
            ):
                signature_copy = dict(existing_signature)
                signature_copy.setdefault("key_id", existing_key_id.strip())
                signature_for_verification = signature_copy
            else:
                signature_for_verification = existing_signature
            payload: dict[str, object] = {
                "exchange": exchange_name,
                "account": existing_account or account_id,
                "portfolio": context.portfolio_id,
                "risk_profile": context.risk_profile,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "operation": operation or "withdrawal",
                "timestamp": timestamp,
            }
            if request.client_order_id:
                payload["client_order_id"] = request.client_order_id
            if self._transaction_signers.verify(
                existing_account or account_id,
                payload,
                signature_for_verification,
            ):
                normalized_signature = dict(signature_for_verification)
                normalized["operation"] = operation or "withdrawal"
                normalized["requires_hardware_wallet"] = True
                normalized.setdefault("hardware_wallet_account", existing_account or account_id)
                normalized["hardware_wallet_signature"] = normalized_signature
                algorithm_value = existing_algorithm or normalized_signature.get("algorithm")
                if algorithm_value:
                    normalized["hardware_wallet_algorithm"] = str(algorithm_value)
                key_id_value = normalized_signature.get("key_id") or existing_key_id
                if key_id_value:
                    normalized["hardware_wallet_key_id"] = str(key_id_value)
                normalized["hardware_wallet_signed_at"] = timestamp
                request.metadata = normalized
                return

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload: dict[str, object] = {
            "exchange": exchange_name,
            "account": account_id,
            "portfolio": context.portfolio_id,
            "risk_profile": context.risk_profile,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "operation": operation or "withdrawal",
            "timestamp": timestamp,
        }
        if request.client_order_id:
            payload["client_order_id"] = request.client_order_id
        signature = signer.sign(payload)

        normalized["operation"] = operation or "withdrawal"
        normalized["requires_hardware_wallet"] = True
        normalized["hardware_wallet_signature"] = dict(signature)
        normalized["hardware_wallet_algorithm"] = getattr(signer, "algorithm", "unknown")
        if getattr(signer, "key_id", None):
            normalized["hardware_wallet_key_id"] = signer.key_id  # type: ignore[attr-defined]
        normalized["hardware_wallet_signed_at"] = timestamp
        normalized["hardware_wallet_account"] = account_id
        request.metadata = normalized

    # --- ExecutionService API ------------------------------------------------

    def _submit_order_threadsafe(
        self, request: OrderRequest, context: ExecutionContext
    ) -> concurrent.futures.Future[OrderResult]:
        if self._closed:
            raise RuntimeError("LiveExecutionRouter został zamknięty")
        submit_future = asyncio.run_coroutine_threadsafe(
            self._submit_order(request, context),
            self._loop,
        )
        return submit_future

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        submit_future = self._submit_order_threadsafe(request, context)
        try:
            return submit_future.result()
        except concurrent.futures.CancelledError as exc:  # pragma: no cover - defensywnie
            raise RuntimeError("Zlecenie zostało anulowane przed egzekucją") from exc

    async def execute_async(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        submit_future = self._submit_order_threadsafe(request, context)
        wrapped = asyncio.wrap_future(submit_future)
        try:
            return await wrapped
        except asyncio.CancelledError:
            submit_future.cancel()
            raise

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: D401
        exchange_name: str | None = self.binding_for_order(order_id)

        if not exchange_name:
            for adapter in self._adapters.values():
                try:
                    adapter.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.debug(
                        "Anulacja %s na adapterze %s: %s",
                        order_id,
                        getattr(adapter, "name", "?"),
                        exc,
                    )
            return

        adapter = self._adapters.get(exchange_name)
        if adapter is None:
            _LOGGER.warning("Brak adaptera %s do anulacji %s", exchange_name, order_id)
            return

        try:
            adapter.cancel_order(order_id, symbol=None)  # type: ignore[call-arg]
        except TypeError:
            adapter.cancel_order(order_id)  # type: ignore[misc]

    def flush(self) -> None:
        if self._closed or self._loop.is_closed():
            self._flush_log(sync=True)
            return
        try:
            asyncio.run_coroutine_threadsafe(self._wait_for_idle(), self._loop).result()
        except Exception:  # pragma: no cover - w testach ignorujemy błędy wygaszania
            pass
        self._flush_log(sync=True)

    async def flush_async(self) -> None:
        if self._closed or self._loop.is_closed():
            await asyncio.to_thread(self._flush_log, sync=True)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._wait_for_idle(), self._loop)
            await asyncio.wrap_future(future)
        except Exception:  # pragma: no cover - w testach ignorujemy błędy wygaszania
            pass
        await asyncio.to_thread(self._flush_log, sync=True)

    def close(self) -> None:
        if not self._mark_closed():
            return
        self._shutdown_loop_sync()
        self._finalize_shutdown_sync()

    async def close_async(self) -> None:
        if not self._mark_closed():
            return
        await self._shutdown_loop_async()
        await self._finalize_shutdown_async()

    def get_runtime_stats(self) -> RouterRuntimeStats:
        """Zwraca bieżące statystyki kolejki i limiterów routera."""

        if self._loop.is_closed() or self._queue is None:
            return self._build_runtime_stats(queue_depth=0, inflight={})
        future = asyncio.run_coroutine_threadsafe(self._gather_runtime_stats(), self._loop)
        try:
            return future.result()
        except Exception:  # pragma: no cover - defensywnie
            return self._build_runtime_stats(queue_depth=0, inflight={})

    async def get_runtime_stats_async(self) -> RouterRuntimeStats:
        """Asynchroniczny odpowiednik :meth:`get_runtime_stats`."""

        if self._loop.is_closed() or self._queue is None:
            return self._build_runtime_stats(queue_depth=0, inflight={})
        future = asyncio.run_coroutine_threadsafe(self._gather_runtime_stats(), self._loop)
        try:
            return await asyncio.wrap_future(future)
        except Exception:  # pragma: no cover - defensywnie
            return self._build_runtime_stats(queue_depth=0, inflight={})

    def __del__(self) -> None:  # pragma: no cover - defensywne czyszczenie zasobów
        try:
            self.close()
        except Exception:
            pass

    def _mark_closed(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return True

    def _shutdown_loop_sync(self) -> None:
        if self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._wait_for_idle(), self._loop).result()
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

    async def _shutdown_loop_async(self) -> None:
        if self._loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._wait_for_idle(), self._loop)
            await asyncio.wrap_future(future)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        if self._loop_thread.is_alive():
            await asyncio.to_thread(self._loop_thread.join, timeout=5.0)

    def _finalize_shutdown_sync(self) -> None:
        self._update_queue_depth(value=0.0)
        self._flush_log(sync=True)

    async def _finalize_shutdown_async(self) -> None:
        self._update_queue_depth(value=0.0)
        await asyncio.to_thread(self._flush_log, sync=True)

    # --- Asynchroniczny pipeline --------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.PriorityQueue(self._qos.max_queue_size)
        self._ack_queue = asyncio.Queue(self._qos.ack_queue_size)
        self._flush_ack_backlog()
        for idx in range(self._qos.worker_concurrency):
            task = self._loop.create_task(
                self._worker_loop(idx),
                name=f"LiveExecutionRouterWorker-{idx}",
            )
            self._worker_tasks.append(task)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            for task in self._worker_tasks:
                task.cancel()
            if self._worker_tasks:
                self._loop.run_until_complete(
                    asyncio.gather(*self._worker_tasks, return_exceptions=True)
                )
            self._loop.run_until_complete(self._fail_pending_orders())
            self._loop.close()
            self._queue = None
            self._ack_queue = None

    async def _submit_order(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        if self._queue is None:
            raise RuntimeError("Kolejka LiveExecutionRouter nie została zainicjalizowana")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[OrderResult] = loop.create_future()
        priority = self._resolve_priority(request, context)
        plan = self._prepare_execution_plan(request, context)
        ack_id = self._resolve_ack_id(request, context)
        queued = _QueuedOrder(
            request=request,
            context=context,
            future=future,
            enqueued_at=self._time(),
            plan=plan,
            ack_id=ack_id,
        )
        try:
            self._queue.put_nowait((priority, next(self._queue_counter), queued))
        except asyncio.QueueFull:
            error = await self._reject_due_to_queue(
                queued,
                0.0,
                reason="overflow",
                queue_timeout=self._queue_timeout,
            )
            raise error

        self._update_queue_depth()
        self._publish_ack(
            queued,
            "ack",
            exchange=None,
            order_id=None,
            details={"priority": priority},
        )

        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise

    async def _worker_loop(self, worker_id: int) -> None:
        del worker_id
        assert self._queue is not None
        while True:
            try:
                _, _, order = await self._queue.get()
            except asyncio.CancelledError:
                break
            if order.future.done():
                self._queue.task_done()
                self._update_queue_depth()
                continue
            now = self._time()
            queue_wait = max(0.0, now - order.enqueued_at)
            try:
                queue_timeout = self._queue_timeout
                if queue_timeout is not None and queue_wait > queue_timeout:
                    await self._reject_due_to_queue(
                        order,
                        queue_wait,
                        reason="timeout",
                        queue_timeout=queue_timeout,
                    )
                    continue

                result = await self._process_order(order, queue_wait, now)
            except Exception as exc:  # noqa: BLE001
                if not order.future.done():
                    order.future.set_exception(exc)
            else:
                if not order.future.done():
                    order.future.set_result(result)
            finally:
                self._queue.task_done()
                self._update_queue_depth()

    async def _process_order(
        self,
        order: _QueuedOrder,
        queue_wait: float,
        started_at: float,
    ) -> OrderResult:
        plan = order.plan
        request = order.request
        context = order.context
        route_name = plan.route_name
        route_meta = plan.route_metadata
        latency_budget_ms = plan.latency_budget_ms
        exchanges_and_retries = list(plan.exchanges_and_retries)

        attempts_rec: list[dict[str, str]] = []
        fallback_used = False
        allowed_fallback_categories = self._resolve_allowed_fallback_categories(context)
        queue_labels = self._queue_labels(queue_wait)
        # start nie obejmuje czasu kolejki – metryki dostają go osobno w etykietach
        start = started_at
        last_error: Exception | None = None
        route_label = route_name or "default"

        for index, (exchange_name, max_retries) in enumerate(exchanges_and_retries):
            breaker_now = self._time()
            breaker = self._breakers.get(exchange_name)
            if breaker and not breaker.allow(breaker_now):
                attempts_rec.append({"exchange": exchange_name, "status": "breaker_open"})
                self._set_breaker_metric(exchange_name, open_=True)
                continue

            adapter = self._adapters.get(exchange_name)
            if adapter is None:
                _LOGGER.warning("Brak adaptera %s", exchange_name)
                attempts_rec.append({"exchange": exchange_name, "status": "adapter_missing"})
                continue

            attempt = 1
            while attempt <= max_retries:
                attempt_now = self._time()
                if latency_budget_ms is not None:
                    elapsed_ms = (attempt_now - start) * 1000.0
                    if elapsed_ms > latency_budget_ms:
                        attempts_rec.append({"exchange": exchange_name, "status": "latency_budget_exceeded"})
                        last_error = last_error or TimeoutError("Przekroczono budżet opóźnień trasy")
                        break

                attempt_labels = {"exchange": exchange_name, **queue_labels}
                attempt_labels_with_route = {**attempt_labels, "route": route_label}
                try:
                    result = await self._perform_attempt(adapter, order.request, exchange_name)
                except (ExchangeNetworkError, ExchangeThrottlingError) as exc:
                    current_time = self._time()
                    elapsed = max(0.0, current_time - start)
                    error_labels = {**attempt_labels_with_route, "result": "error"}
                    self._m_latency.observe(elapsed, labels=error_labels)
                    self._m_attempts.inc(labels=error_labels)
                    self._m_errors.inc(labels=attempt_labels_with_route)
                    attempts_rec.append(
                        {
                            "exchange": exchange_name,
                            "attempt": str(attempt),
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    last_error = exc
                    if breaker:
                        breaker.record_failure(current_time)
                    if attempt < max_retries:
                        await asyncio.sleep(_exp_backoff_with_jitter(attempt))
                        attempt += 1
                        continue
                    category = "throttling" if isinstance(exc, ExchangeThrottlingError) else "network"
                    if not self._is_fallback_allowed(category, allowed_fallback_categories):
                        self._publish_ack(
                            order,
                            "nak",
                            exchange=exchange_name,
                            order_id=None,
                            details={"error": repr(exc), "kind": category},
                        )
                        raise
                    break
                except ExchangeAuthError as exc:
                    self._m_attempts.inc(labels={**attempt_labels_with_route, "result": "auth_error"})
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "auth_error"})
                    if breaker:
                        breaker.record_failure(self._time())
                    _LOGGER.error("Błąd uwierzytelnienia na %s – przerywam fallback.", exchange_name)
                    self._publish_ack(
                        order,
                        "nak",
                        exchange=exchange_name,
                        order_id=None,
                        details={"error": repr(exc), "kind": "auth"},
                    )
                    raise
                except ExchangeAPIError as exc:
                    error_labels = {**attempt_labels_with_route, "result": "api_error"}
                    self._m_attempts.inc(labels=error_labels)
                    self._m_errors.inc(labels=attempt_labels_with_route)
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "api_error"})
                    last_error = exc
                    if breaker:
                        breaker.record_failure(self._time())
                    _LOGGER.error(
                        "API %s odrzuciło zlecenie (status=%s): %s",
                        exchange_name,
                        getattr(exc, "status_code", "?"),
                        getattr(exc, "message", repr(exc)),
                    )
                    self._publish_ack(
                        order,
                        "nak",
                        exchange=exchange_name,
                        order_id=None,
                        details={"error": repr(exc), "kind": "api"},
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    current_time = self._time()
                    elapsed = max(0.0, current_time - start)
                    error_labels = {**attempt_labels_with_route, "result": "exception"}
                    self._m_latency.observe(elapsed, labels=error_labels)
                    self._m_attempts.inc(labels=error_labels)
                    self._m_errors.inc(labels=attempt_labels_with_route)
                    attempts_rec.append(
                        {
                            "exchange": exchange_name,
                            "attempt": str(attempt),
                            "status": "exception",
                            "error": repr(exc),
                        }
                    )
                    last_error = exc
                    if breaker:
                        breaker.record_failure(current_time)
                    if not self._is_fallback_allowed("unknown", allowed_fallback_categories):
                        self._publish_ack(
                            order,
                            "nak",
                            exchange=exchange_name,
                            order_id=None,
                            details={"error": repr(exc), "kind": "exception"},
                        )
                        raise
                    if attempt < max_retries:
                        await asyncio.sleep(_exp_backoff_with_jitter(attempt))
                        attempt += 1
                        continue
                    break

                # sukces
                self._validate_adapter_result(result, exchange_name, request)
                current_time = self._time()
                elapsed = max(0.0, current_time - start)
                success_labels = {**attempt_labels_with_route, "result": "success"}
                self._m_latency.observe(elapsed, labels=success_labels)
                self._m_attempts.inc(labels=success_labels)
                route_labels = {"exchange": exchange_name, "route": route_label}
                route_only_labels = {"route": route_label}
                router_labels = {
                    "exchange": exchange_name,
                    "symbol": request.symbol,
                    "portfolio": context.portfolio_id,
                }
                self._m_success.inc(labels=route_labels)
                self._m_orders_total.inc(labels=route_labels)
                filled_qty = float(result.filled_quantity or 0.0)
                requested_qty = float(request.quantity or 0.0)
                ratio = 0.0
                if requested_qty > 0:
                    ratio = max(0.0, min(1.0, filled_qty / requested_qty))
                self._m_fill_ratio.observe(ratio, labels=route_labels)
                if breaker:
                    breaker.record_success()
                    self._set_breaker_metric(exchange_name, open_=False)
                if index > 0:
                    self._m_fallbacks.inc(labels=route_only_labels)
                    self._m_router_fallbacks.inc(labels=router_labels)
                    fallback_used = True

                self._remember_binding(result.order_id, exchange_name)
                attempts_rec.append({"exchange": exchange_name, "status": "success", "latency_s": f"{elapsed:.6f}"})
                self._maybe_write_decision_log(
                    route_name=route_name or "default",
                    route_metadata=route_meta,
                    request=request,
                    context=context,
                    result=result,
                    attempts=attempts_rec,
                    latency_seconds=elapsed,
                    fallback_used=fallback_used,
                )
                self._publish_ack(
                    order,
                    "done",
                    exchange=exchange_name,
                    order_id=result.order_id,
                    details={
                        "status": result.status,
                        "filled_quantity": result.filled_quantity,
                        "avg_price": result.avg_price,
                        "fallback_used": fallback_used,
                    },
                )
                return result

        elapsed = max(0.0, self._time() - start)
        attempts_rec.append({"status": "failed", "latency_s": f"{elapsed:.6f}"})
        failure_exchange = None
        for attempt in reversed(attempts_rec):
            exchange = attempt.get("exchange")
            if exchange:
                failure_exchange = str(exchange)
                break
        if failure_exchange is None and exchanges_and_retries:
            failure_exchange = str(exchanges_and_retries[0][0])

        failure_route_labels = {"route": route_label}
        failure_router_labels = {
            "symbol": request.symbol,
            "portfolio": context.portfolio_id,
        }

        self._m_failures.inc(labels=failure_route_labels)
        self._m_router_failures.inc(labels=failure_router_labels)
        error_exchange = failure_exchange or (exchanges_and_retries[0][0] if exchanges_and_retries else "unknown")
        self._m_errors.inc(
            labels={
                "exchange": error_exchange,
                "route": route_label,
                **queue_labels,
            }
        )
        await asyncio.to_thread(
            self._maybe_write_decision_log,
            route_name=(route_name or "default"),
            route_metadata=route_meta,
            request=order.request,
            context=order.context,
            result=None,
            attempts=attempts_rec,
            latency_seconds=elapsed,
            fallback_used=fallback_used,
            error=repr(last_error) if last_error else None,
        )
        if last_error is not None:
            self._publish_ack(
                order,
                "nak",
                exchange=failure_exchange,
                order_id=None,
                details={"error": repr(last_error)},
            )
            raise last_error
        raise RuntimeError("Nie udało się zrealizować zlecenia – brak dostępnych giełd")

    async def _perform_attempt(
        self,
        adapter: ExchangeAdapter,
        request: OrderRequest,
        exchange_name: str,
    ) -> OrderResult:
        async def _call() -> OrderResult:
            return await asyncio.to_thread(adapter.place_order, request)

        if self._io_dispatcher is not None:
            async with self._track_exchange_inflight(exchange_name):
                return await self._io_dispatcher.submit(exchange_name, _call)

        semaphore = await self._get_exchange_semaphore(exchange_name)
        async with semaphore:
            async with self._track_exchange_inflight(exchange_name):
                return await _call()

    async def _get_exchange_semaphore(self, exchange_name: str) -> asyncio.Semaphore:
        semaphore = self._exchange_semaphores.get(exchange_name)
        if semaphore is not None:
            return semaphore
        limit = int(self._qos.per_exchange_concurrency.get(exchange_name, self._qos.worker_concurrency))
        limit = max(1, limit)
        semaphore = asyncio.Semaphore(limit)
        self._exchange_semaphores[exchange_name] = semaphore
        return semaphore

    async def _wait_for_idle(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()

    async def _fail_pending_orders(self) -> None:
        if self._queue is None:
            return
        while not self._queue.empty():
            _, _, order = await self._queue.get()
            if not order.future.done():
                order.future.set_exception(RuntimeError("LiveExecutionRouter zatrzymany"))
            self._queue.task_done()
        self._update_queue_depth(value=0.0)

    def _update_queue_depth(self, *, value: float | None = None) -> None:
        if self._g_queue_depth is None:
            return
        try:
            if value is not None:
                self._g_queue_depth.set(float(value))
                return
            queue = self._queue
            size = float(queue.qsize()) if queue is not None else 0.0
            self._g_queue_depth.set(size)
        except Exception:  # pragma: no cover - zależne od implementacji
            pass

    @staticmethod
    def _queue_labels(queue_wait: float) -> dict[str, str]:
        waited = max(0.0, queue_wait)
        return {
            "queued": "true" if waited > 0.0 else "false",
            "queue_wait_seconds": f"{waited:.6f}",
        }

    def _build_runtime_stats(
        self,
        *,
        queue_depth: int,
        inflight: Mapping[str, int],
    ) -> RouterRuntimeStats:
        limits = {
            name: int(self._qos.per_exchange_concurrency.get(name, self._qos.worker_concurrency))
            for name in self._adapters.keys()
        }
        return RouterRuntimeStats(
            queue_depth=int(queue_depth),
            queue_limit=int(self._qos.max_queue_size),
            worker_concurrency=int(self._qos.worker_concurrency),
            per_exchange_limits=limits,
            inflight_by_exchange=dict(inflight),
            queue_timeout=self._queue_timeout,
            closed=self._closed,
            mode=self._mode,
        )

    async def _gather_runtime_stats(self) -> RouterRuntimeStats:
        queue_size = 0
        if self._queue is not None:
            try:
                queue_size = int(self._queue.qsize())
            except Exception:  # pragma: no cover - zależne od implementacji
                queue_size = 0
        return self._build_runtime_stats(queue_depth=queue_size, inflight=self._inflight_by_exchange)

    @asynccontextmanager
    async def _track_exchange_inflight(self, exchange_name: str):
        self._inflight_by_exchange[exchange_name] = self._inflight_by_exchange.get(exchange_name, 0) + 1
        try:
            yield
        finally:
            remaining = self._inflight_by_exchange.get(exchange_name, 0) - 1
            if remaining <= 0:
                self._inflight_by_exchange.pop(exchange_name, None)
            else:
                self._inflight_by_exchange[exchange_name] = remaining

    async def _reject_due_to_queue(
        self,
        order: _QueuedOrder,
        queue_wait: float,
        *,
        reason: str,
        queue_timeout: float | None,
    ) -> TimeoutError:
        route_name = order.plan.route_name or "default"
        queue_labels = self._queue_labels(queue_wait)
        labels_base = {
            "exchange": "queue",
            "symbol": order.request.symbol,
            "portfolio": order.context.portfolio_id,
            **queue_labels,
        }
        result_label = "queue_timeout" if reason == "timeout" else "queue_overflow"
        self._m_latency.observe(queue_wait, labels={**labels_base, "result": result_label})
        self._m_attempts.inc(labels={**labels_base, "result": result_label})
        self._m_failures.inc(labels={"route": route_name})
        self._m_router_failures.inc(
            labels={"symbol": order.request.symbol, "portfolio": order.context.portfolio_id}
        )

        attempt_payload: dict[str, str] = {
            "status": result_label,
            "queue_wait_s": f"{queue_wait:.6f}",
        }
        error_details: str
        if reason == "timeout" and queue_timeout is not None:
            attempt_payload["queue_timeout_s"] = f"{queue_timeout:.6f}"
            error_details = (
                f"QueueTimeout(wait={queue_wait:.6f}, limit={queue_timeout:.6f})"
            )
            message = (
                f"Przekroczono limit oczekiwania w kolejce ({queue_wait:.3f}s > {queue_timeout:.3f}s)"
            )
        else:
            queue_size = 0
            if self._queue is not None:
                try:
                    queue_size = int(self._queue.qsize())
                except Exception:  # pragma: no cover - defensywnie
                    queue_size = 0
            attempt_payload["queue_size"] = str(queue_size)
            error_details = f"QueueOverflow(wait={queue_wait:.6f}, size={queue_size})"
            message = "Kolejka LiveExecutionRouter jest pełna"

        self._publish_ack(
            order,
            "nak",
            exchange=None,
            order_id=None,
            details={
                "reason": reason,
                "queue_wait_s": f"{queue_wait:.6f}",
                **({"queue_timeout_s": f"{queue_timeout:.6f}"} if reason == "timeout" and queue_timeout is not None else {}),
            },
        )

        await asyncio.to_thread(
            self._maybe_write_decision_log,
            route_name=route_name,
            route_metadata=order.plan.route_metadata,
            request=order.request,
            context=order.context,
            result=None,
            attempts=[attempt_payload],
            latency_seconds=queue_wait,
            fallback_used=False,
            error=error_details,
        )

        error = TimeoutError(message)
        if not order.future.done():
            order.future.set_exception(error)
        return error

    def _resolve_priority(self, request: OrderRequest, context: ExecutionContext) -> int:
        try:
            return int(self._priority_resolver(request, context))
        except Exception:  # pragma: no cover - defensywnie
            _LOGGER.debug("Błąd wyliczenia priorytetu kolejki", exc_info=True)
            return 0

    def _prepare_execution_plan(
        self, request: OrderRequest, context: ExecutionContext
    ) -> _ExecutionPlan:
        if self._mode == "routes":
            selection = self._select_route_definition(request, context)
            exchanges = tuple(
                (exchange, max(1, selection.max_retries_per_exchange))
                for exchange in selection.iter_exchanges()
            )
            if not exchanges:
                raise RuntimeError("Trasa live nie definiuje żadnych giełd do egzekucji")
            return _ExecutionPlan(
                route_name=selection.name,
                route_metadata=selection.metadata,
                exchanges_and_retries=exchanges,
                latency_budget_ms=float(selection.latency_budget_ms),
            )

        routing_plan = self._resolve_plan(request.symbol)
        exchanges = tuple((exchange, 1) for exchange in routing_plan.exchanges)
        if not exchanges:
            raise RuntimeError("Plan live nie zawiera dostępnych giełd dla symbolu")
        return _ExecutionPlan(
            route_name=None,
            route_metadata={},
            exchanges_and_retries=exchanges,
            latency_budget_ms=None,
        )

    # --- Wewnętrzne narzędzia ------------------------------------------------

    def _remember_binding(self, order_id: str, exchange_name: str) -> None:
        with self._bindings_lock:
            self._bindings[order_id] = exchange_name
            self._bindings.move_to_end(order_id)
            if len(self._bindings) > self._bindings_capacity:
                self._bindings.popitem(last=False)

    def _resolve_ack_id(self, request: OrderRequest, context: ExecutionContext) -> str:
        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        for key in ("ack_id", "request_id", "decision_id", "correlation_id", "tracking_id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
        context_meta = context.metadata if isinstance(context.metadata, Mapping) else {}
        for key in ("request_id", "decision_id", "correlation_id", "tracking_id"):
            value = context_meta.get(key)
            if value not in (None, ""):
                return str(value)
        if request.client_order_id not in (None, ""):
            return str(request.client_order_id)
        return uuid.uuid4().hex

    def _flush_ack_backlog(self) -> None:
        queue = self._ack_queue
        if queue is None:
            return
        while self._ack_backlog:
            event = self._ack_backlog[0]
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                break
            else:
                self._ack_backlog.popleft()

    def _publish_ack(
        self,
        order: _QueuedOrder,
        status: str,
        *,
        exchange: str | None,
        order_id: str | None,
        details: Mapping[str, object] | None,
    ) -> None:
        if status in {"done", "nak"} and order.final_status_sent:
            return
        event = AcknowledgementEvent(
            ack_id=order.ack_id,
            status=status,
            exchange=exchange,
            order_id=order_id,
            client_order_id=order.request.client_order_id,
            symbol=order.request.symbol,
            portfolio=order.context.portfolio_id,
            timestamp=self._time(),
            details=dict(details or {}),
        )
        queue = self._ack_queue
        if queue is None:
            self._ack_backlog.append(event)
        else:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    self._ack_backlog.append(event)
        if status in {"done", "nak"}:
            order.final_status_sent = True
        self._flush_ack_backlog()

    async def _wait_for_ack_event(self, timeout: float | None) -> AcknowledgementEvent:
        queue = self._ack_queue
        if queue is None:
            raise RuntimeError("Kolejka ACK LiveExecutionRouter nie jest dostępna")
        try:
            if timeout is None:
                event = await queue.get()
            else:
                event = await asyncio.wait_for(queue.get(), timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Przekroczono limit oczekiwania na potwierdzenie zlecenia") from exc
        finally:
            self._flush_ack_backlog()
        return event

    def get_acknowledgement(self, timeout: float | None = None) -> AcknowledgementEvent:
        future = asyncio.run_coroutine_threadsafe(self._wait_for_ack_event(timeout), self._loop)
        return future.result()

    async def get_acknowledgement_async(self, timeout: float | None = None) -> AcknowledgementEvent:
        future = asyncio.run_coroutine_threadsafe(self._wait_for_ack_event(timeout), self._loop)
        return await asyncio.wrap_future(future)

    def _set_breaker_metric(self, exchange: str, *, open_: bool) -> None:
        try:
            self._g_breaker_open.set(1 if open_ else 0, labels={"exchange": exchange})
        except Exception:
            pass

    @staticmethod
    def _resolve_allowed_fallback_categories(context: ExecutionContext) -> set[str] | None:
        metadata = getattr(context, "metadata", {}) or {}
        raw = metadata.get("risk_allow_fallback_categories")
        if raw is None:
            return None
        if isinstance(raw, str):
            parts = [part.strip().lower() for part in raw.split(",")]
            allowed = {part for part in parts if part}
            return allowed or set()
        try:
            iterator = iter(raw)  # type: ignore[arg-type]
        except TypeError:
            return None
        allowed_set: set[str] = set()
        for entry in iterator:
            if entry is None:
                continue
            allowed_set.add(str(entry).strip().lower())
        return allowed_set

    @staticmethod
    def _is_fallback_allowed(category: str, allowed: set[str] | None) -> bool:
        if allowed is None:
            return True
        if not allowed:
            return False
        return category.lower() in allowed

    def _validate_adapter_result(
        self, result: OrderResult, exchange: str, request: OrderRequest
    ) -> None:
        status = str(getattr(result, "status", ""))
        normalized_status = status.upper()
        if normalized_status in {"REJECTED", "CANCELLED", "CANCELED", "ERROR"}:
            message = f"Adapter {exchange} odrzucił zlecenie (status={normalized_status})."
            raise _build_api_error(message, status=422, payload={"exchange": exchange, "status": status})

        if not getattr(result, "order_id", ""):
            message = f"Adapter {exchange} nie zwrócił identyfikatora zlecenia."
            raise _build_api_error(message, status=500, payload={"exchange": exchange})

        filled_quantity = getattr(result, "filled_quantity", 0.0)
        try:
            filled_value = float(filled_quantity)
        except (TypeError, ValueError):
            raise _build_api_error(
                f"Adapter {exchange} zwrócił nieprawidłowe wypełnienie.",
                status=500,
                payload={"exchange": exchange, "filled_quantity": filled_quantity},
            ) from None

        if math.isnan(filled_value) or filled_value < 0.0:
            raise _build_api_error(
                f"Adapter {exchange} zwrócił ujemne wypełnienie.",
                status=500,
                payload={
                    "exchange": exchange,
                    "filled_quantity": filled_value,
                    "symbol": request.symbol,
                },
            )

    # --- Tryb "routes": wybór trasy -----------------------------------------

    def _select_route_definition(self, request: OrderRequest, context: ExecutionContext) -> RouteDefinition:
        assert self._mode == "routes"
        meta = context.metadata or {}
        requested = meta.get("execution_route")
        if requested:
            route = self._routes_by_name.get(str(requested))
            if route is None:
                raise KeyError(f"Wymagana trasa execution_route={requested} nie istnieje")
            return route

        for route in self._routes:
            if route.symbols and request.symbol not in route.symbols:
                continue
            if route.risk_profiles and context.risk_profile not in route.risk_profiles:
                continue
            return route

        if self._default_named_route:
            return self._routes_by_name[self._default_named_route]
        return self._routes[0]

    # --- Tryb "plan": plan na podstawie symbolu -----------------------------

    def _resolve_plan(self, symbol: str) -> RoutingPlan:
        assert self._mode == "plan"
        override = self._overrides.get(symbol)
        if override:
            return override
        assert self._default_plan is not None
        return self._default_plan

    # --- Decision log --------------------------------------------------------

    def _maybe_write_decision_log(
        self,
        *,
        route_name: str,
        route_metadata: Mapping[str, str],
        request: OrderRequest,
        context: ExecutionContext,
        attempts: Sequence[Mapping[str, str]],
        latency_seconds: float,
        fallback_used: bool,
        result: OrderResult | None,
        error: str | None = None,
    ) -> None:
        if self._decision_log_path is None:
            return

        payload: dict[str, object] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "route": route_name,
            "metadata": dict(route_metadata or {}),
            "request": {
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type,
                "quantity": request.quantity,
                "price": request.price,
                "time_in_force": getattr(request, "time_in_force", None),
                "client_order_id": getattr(request, "client_order_id", None),
            },
            "context": {
                "portfolio_id": context.portfolio_id,
                "risk_profile": context.risk_profile,
                "environment": context.environment,
                "metadata": dict(context.metadata or {}),
            },
            "attempts": list(attempts),
            "latency_seconds": round(latency_seconds, 6),
            "fallback_used": fallback_used,
        }
        if result is not None:
            payload["result"] = {
                "order_id": result.order_id,
                "status": result.status,
                "filled_quantity": result.filled_quantity,
                "avg_price": result.avg_price,
            }
        if error:
            payload["error"] = error

        document: dict[str, object] = {"payload": payload}
        if self._decision_log_key is not None:
            document["signature"] = _build_hmac_signature(
                payload,
                key=self._decision_log_key,
                algorithm="HMAC-SHA384",
                key_id=self._decision_log_key_id,
            )

        serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
        with self._log_lock:
            self._rotate_log_if_needed_unlocked()
            with self._decision_log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")

    def _rotate_log_if_needed_unlocked(self) -> None:
        path = self._decision_log_path
        if path is None or self._decision_log_rotate_bytes <= 0:
            return
        if not path.exists():
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._decision_log_rotate_bytes:
            return

        for idx in range(self._decision_log_keep - 1, 0, -1):
            older = path.with_suffix(path.suffix + f".{idx}")
            newer = path.with_suffix(path.suffix + ("" if idx == 1 else f".{idx-1}"))
            if newer.exists():
                try:
                    if older.exists():
                        older.unlink()
                except OSError:
                    pass
                try:
                    newer.rename(older)
                except OSError:
                    pass
        target = path.with_suffix(path.suffix + ".1")
        try:
            if target.exists():
                target.unlink()
        except OSError:
            pass
        try:
            path.rename(target)
        except OSError:
            pass

    def _flush_log(self, *, sync: bool = False) -> None:
        path = self._decision_log_path
        if path is None or not path.exists():
            return
        with self._log_lock:
            try:
                with path.open("ab", buffering=0) as handle:
                    handle.flush()
                    if sync:
                        try:
                            os.fsync(handle.fileno())
                        except Exception:
                            pass
            except Exception:
                pass


__all__ = [
    "LiveExecutionRouter",
    "RoutingPlan",
    "RouteDefinition",
    "QoSConfig",
    "RouterRuntimeStats",
    "AcknowledgementEvent",
]
