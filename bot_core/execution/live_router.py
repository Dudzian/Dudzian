"""Router egzekucji dla środowiska live z fallbackami, metrykami i (opcjonalnie) decision logiem."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import ExchangeAdapter, OrderRequest, OrderResult

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


# --- Observability (różne nazwy modułów w gałęziach)
try:  # pragma: no cover
    from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
except Exception:  # pragma: no cover
    try:
        from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore
    except Exception:  # pragma: no cover
        class _NoopMetric:
            def inc(self, *_args, **_kwargs) -> None:  # noqa: D401
                return None

            def observe(self, *_args, **_kwargs) -> None:
                return None

            def set(self, *_args, **_kwargs) -> None:
                return None

            def dec(self, *_args, **_kwargs) -> None:
                return None

        class MetricsRegistry:  # type: ignore
            def counter(self, *_args, **_kwargs) -> _NoopMetric:
                return _NoopMetric()

            def histogram(self, *_args, **_kwargs) -> _NoopMetric:
                return _NoopMetric()

            def gauge(self, *_args, **_kwargs) -> _NoopMetric:
                return _NoopMetric()

        def get_global_metrics_registry() -> MetricsRegistry:  # type: ignore
            return MetricsRegistry()


# --- Podpisywanie decision logu (opcjonalne, z fallbackiem)
try:  # pragma: no cover
    from bot_core.security.signing import build_hmac_signature as _build_hmac_signature  # type: ignore
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


_LOGGER = logging.getLogger(__name__)


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


# === LiveExecutionRouter =====================================================

class LiveExecutionRouter(ExecutionService):
    """
    Router egzekucji live. Dwa tryby:

    * 'routes' — przekaż `routes: Sequence[RouteDefinition]` (opcjonalnie `default_route=<nazwa>`).
    * 'plan'   — przekaż `default_route: Iterable[str]` (+ opcjonalnie `route_overrides` per symbol).

    Dodatki operacyjne:
    - circuit-breaker per giełda,
    - exponential backoff z jitterem,
    - budżet opóźnień per trasa,
    - LRU dla powiązań order_id→exchange,
    - decision-log JSONL (opcjonalny) z podpisem HMAC i rotacją rozmiaru,
    - metryki (latencja, próby, fallbacki, błędy, stan breakerów).
    """

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

        # Czas i breaker
        self._time = time_source or time.perf_counter
        factory = circuit_breaker_factory or (lambda: CircuitBreaker())
        self._breakers: dict[str, CircuitBreaker] = {name: factory() for name in self._adapters.keys()}

        # LRU powiązań order_id→exchange
        self._bindings_capacity = max(100, int(bindings_capacity))
        self._bindings: OrderedDict[str, str] = OrderedDict()
        self._bindings_lock = threading.Lock()

    # --- Publiczne pomocnicze API -------------------------------------------

    def list_adapters(self) -> tuple[str, ...]:
        """Zwraca nazwy zarejestrowanych adapterów giełdowych."""

        return tuple(self._adapters.keys())

    def set_bindings_capacity(self, capacity: int) -> None:
        self._bindings_capacity = max(100, int(capacity))

    def binding_for_order(self, order_id: str) -> str | None:
        with self._bindings_lock:
            return self._bindings.get(order_id)

    # --- ExecutionService API ------------------------------------------------

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        """
        Realizuje zlecenie na pierwszej skutecznej giełdzie zgodnie z planem/trasą.
        W trybie "routes" respektuje `max_retries_per_exchange` oraz pisze decision log (gdy skonfigurowany).
        Uwzględnia breaker i budżet opóźnień.
        """
        exchanges_and_retries: list[tuple[str, int]]
        route_name: str | None = None
        route_meta: Mapping[str, str] = {}
        latency_budget_ms: float | None = None

        if self._mode == "routes":
            sel = self._select_route_definition(request, context)
            route_name = sel.name
            route_meta = sel.metadata
            latency_budget_ms = float(sel.latency_budget_ms)
            exchanges_and_retries = [(ex, max(1, sel.max_retries_per_exchange)) for ex in sel.iter_exchanges()]
        else:
            plan = self._resolve_plan(request.symbol)
            exchanges_and_retries = [(ex, 1) for ex in plan.exchanges]

        attempts_rec: list[dict[str, str]] = []
        attempts_counter = 0
        fallback_used = False
        start = self._time()
        last_error: Exception | None = None
        route_label = route_name or "default"

        for exchange_name, max_retries in exchanges_and_retries:
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

            for attempt in range(1, max_retries + 1):
                # budżet opóźnień (per route)
                attempt_now = self._time()
                if latency_budget_ms is not None:
                    elapsed_ms = (attempt_now - start) * 1000.0
                    if elapsed_ms > latency_budget_ms:
                        attempts_rec.append({"exchange": exchange_name, "status": "latency_budget_exceeded"})
                        last_error = last_error or TimeoutError("Przekroczono budżet opóźnień trasy")
                        break

                labels = {
                    "exchange": exchange_name,
                    "symbol": request.symbol,
                    "portfolio": context.portfolio_id,
                    "route": route_label,
                }
                attempts_counter += 1
                try:
                    result = adapter.place_order(request)
                except (ExchangeNetworkError, ExchangeThrottlingError) as exc:
                    current_time = self._time()
                    elapsed = max(0.0, current_time - start)
                    self._m_latency.observe(elapsed, labels={**labels, "result": "error"})
                    self._m_attempts.inc(labels={**labels, "result": "error"})
                    self._m_errors.inc(labels=labels)
                    attempts_rec.append(
                        {"exchange": exchange_name, "attempt": str(attempt), "status": "error", "error": repr(exc)}
                    )
                    last_error = exc
                    if breaker:
                        breaker.record_failure(current_time)
                    # retry jeśli dostępny
                    if attempt < max_retries:
                        time.sleep(_exp_backoff_with_jitter(attempt))
                        continue
                    break
                except ExchangeAuthError as exc:
                    self._m_attempts.inc(labels={**labels, "result": "auth_error"})
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "auth_error"})
                    if breaker:
                        breaker.record_failure(self._time())
                    _LOGGER.error("Błąd uwierzytelnienia na %s – przerywam fallback.", exchange_name)
                    raise
                except ExchangeAPIError as exc:
                    self._m_attempts.inc(labels={**labels, "result": "api_error"})
                    self._m_errors.inc(labels=labels)
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
                    # błąd twardy (bez dalszego fallbacku)
                    raise
                except Exception as exc:  # noqa: BLE001
                    current_time = self._time()
                    elapsed = max(0.0, current_time - start)
                    self._m_latency.observe(elapsed, labels={**labels, "result": "exception"})
                    self._m_attempts.inc(labels={**labels, "result": "exception"})
                    self._m_errors.inc(labels=labels)
                    attempts_rec.append(
                        {"exchange": exchange_name, "attempt": str(attempt), "status": "exception", "error": repr(exc)}
                    )
                    last_error = exc
                    if breaker:
                        breaker.record_failure(current_time)
                    if attempt < max_retries:
                        time.sleep(_exp_backoff_with_jitter(attempt))
                        continue
                    break

                # sukces
                current_time = self._time()
                elapsed = max(0.0, current_time - start)
                self._m_latency.observe(elapsed, labels={**labels, "result": "success"})
                self._m_attempts.inc(labels={**labels, "result": "success"})
                common_labels = {
                    "exchange": exchange_name,
                    "symbol": request.symbol,
                    "portfolio": context.portfolio_id,
                    "route": route_label,
                }
                self._m_success.inc(labels=common_labels)
                self._m_orders_total.inc(labels=common_labels)
                filled_qty = float(result.filled_quantity or 0.0)
                requested_qty = float(request.quantity or 0.0)
                ratio = 0.0
                if requested_qty > 0:
                    ratio = max(0.0, min(1.0, filled_qty / requested_qty))
                self._m_fill_ratio.observe(ratio, labels=common_labels)
                if breaker:
                    breaker.record_success()
                    self._set_breaker_metric(exchange_name, open_=False)
                if attempts_counter > 1:
                    self._m_fallbacks.inc(labels=common_labels)
                    self._m_router_fallbacks.inc(labels=common_labels)
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
                return result

        # brak sukcesu
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

        failure_labels = {
            "route": route_label,
            "symbol": request.symbol,
            "portfolio": context.portfolio_id,
        }
        if failure_exchange is not None:
            failure_labels["exchange"] = failure_exchange

        self._m_failures.inc(labels=failure_labels)
        self._m_router_failures.inc(labels=failure_labels)
        self._m_errors.inc(
            labels={
                "exchange": exchanges_and_retries[0][0] if exchanges_and_retries else "unknown",
                "symbol": request.symbol,
                "portfolio": context.portfolio_id,
                "route": route_label,
            }
        )
        self._maybe_write_decision_log(
            route_name=(route_name or "default"),
            route_metadata=route_meta,
            request=request,
            context=context,
            result=None,
            attempts=attempts_rec,
            latency_seconds=elapsed,
            fallback_used=fallback_used,
            error=repr(last_error) if last_error else None,
        )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Nie udało się zrealizować zlecenia – brak dostępnych giełd")

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: D401
        exchange_name: str | None = self.binding_for_order(order_id)

        if not exchange_name:
            # wyrozumiały fallback (bez wywalania wyjątku)
            for adapter in self._adapters.values():
                try:
                    adapter.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.debug("Anulacja %s na adapterze %s: %s", order_id, getattr(adapter, "name", "?"), exc)
            return

        adapter = self._adapters.get(exchange_name)
        if adapter is None:
            _LOGGER.warning("Brak adaptera %s do anulacji %s", exchange_name, order_id)
            return

        # zgodność: różne sygnatury
        try:
            adapter.cancel_order(order_id, symbol=None)  # type: ignore[call-arg]
        except TypeError:
            adapter.cancel_order(order_id)  # type: ignore[misc]

    def flush(self) -> None:
        self._flush_log(sync=True)

    # --- Wewnętrzne narzędzia ------------------------------------------------

    def _remember_binding(self, order_id: str, exchange_name: str) -> None:
        with self._bindings_lock:
            self._bindings[order_id] = exchange_name
            self._bindings.move_to_end(order_id)
            if len(self._bindings) > self._bindings_capacity:
                self._bindings.popitem(last=False)

    def _set_breaker_metric(self, exchange: str, *, open_: bool) -> None:
        try:
            self._g_breaker_open.set(1 if open_ else 0, labels={"exchange": exchange})
        except Exception:
            # jeśli gauge nie istnieje w danej implementacji, ignorujemy
            pass

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

        # prosta rotacja: .1 .. .N
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
        # przenieś bieżący do .1
        target = path.with_suffix(path.suffix + ".1")
        try:
            if target.exists():
                target.unlink()
        except OSError:
            pass
        try:
            path.rename(target)
        except OSError:
            # w najgorszym razie – brak rotacji, log dalej rośnie
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


__all__ = ["LiveExecutionRouter", "RoutingPlan", "RouteDefinition"]
