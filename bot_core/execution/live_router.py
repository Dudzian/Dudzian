"""Router egzekucji dla środowiska live z fallbackami, metrykami i (opcjonalnie) decision logiem."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
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
        # starsze gałęzie
        from bot_core.observability import MetricsRegistry, get_global_metrics_registry  # type: ignore
    except Exception:
        class _NoopMetric:
            def inc(self, *_args, **_kwargs) -> None:  # noqa: D401
                return None

            def observe(self, *_args, **_kwargs) -> None:  # noqa: D401
                return None

        class MetricsRegistry:  # type: ignore
            def counter(self, *_args, **_kwargs) -> _NoopMetric:
                return _NoopMetric()

            def histogram(self, *_args, **_kwargs) -> _NoopMetric:
                return _NoopMetric()

        def get_global_metrics_registry() -> MetricsRegistry:  # type: ignore
            return MetricsRegistry()


# --- Podpisywanie decision logu (opcjonalne)
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
            algo = "HMACSHA384"
        digestmod = hashlib.sha384 if algo == "HMACSHA384" else hashlib.sha256
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


# --- API trasowania: wspieramy obie nazwy/formaty ---------------------------
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
    max_retries_per_exchange: int = 1
    latency_budget_ms: float = 250.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def iter_exchanges(self) -> Iterable[str]:
        for ex in self.exchanges:
            n = str(ex).strip()
            if n:
                yield n


class LiveExecutionRouter(ExecutionService):
    """
    Router egzekucji live. Wspiera dwa tryby:

    1) TRYB "routes":
       - przekazujesz `routes: Sequence[RouteDefinition]` (opcjonalnie `default_route=<nazwa>`),
       - możesz skonfigurować `decision_log_path` (+ klucz HMAC).

    2) TRYB "plan":
       - przekazujesz `default_route: Iterable[str]` (+ opcjonalne `route_overrides`).
       - prosty fallback bez decision logu (chyba że go jawnie skonfigurujesz).
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
        # --- decision log (opcjonalny w obu trybach) ---
        decision_log_path: str | os.PathLike[str] | None = None,
        decision_log_hmac_key: bytes | None = None,
        decision_log_key_id: str | None = None,
        # --- metryki / czas ---
        metrics_registry: MetricsRegistry | None = None,
        metrics: MetricsRegistry | None = None,  # alias zgodności
        latency_buckets: Sequence[float] | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("LiveExecutionRouter wymaga co najmniej jednego adaptera giełdowego")
        self._adapters: dict[str, ExchangeAdapter] = {str(k): v for k, v in adapters.items()}

        # Rozpoznaj tryb
        self._mode: str
        self._routes_by_name: dict[str, RouteDefinition] = {}
        self._routes: tuple[RouteDefinition, ...] = ()
        self._default_named_route: str | None = None
        self._default_plan: RoutingPlan | None = None
        self._overrides: dict[str, RoutingPlan] = {}

        if routes is not None:
            # tryb "routes"
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
            # tryb "plan"
            self._mode = "plan"
            if not default_route:
                raise ValueError("W trybie 'plan' wymagany jest parametr default_route")
            self._default_plan = RoutingPlan(tuple(default_route if not isinstance(default_route, str) else [default_route]))
            for sym, route in (route_overrides or {}).items():
                if route:
                    self._overrides[str(sym)] = RoutingPlan(tuple(route))

        # Decision log (opcjonalny)
        self._decision_log_path: Path | None = None
        self._decision_log_key: bytes | None = None
        self._decision_log_key_id: str | None = decision_log_key_id
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

        # Inne
        self._time = time_source or time.perf_counter
        self._bindings: MutableMapping[str, str] = {}
        self._bindings_lock = threading.Lock()

    # ------------------------------------------------------------------
    # ExecutionService API
    # ------------------------------------------------------------------
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        """
        Realizuje zlecenie na pierwszej skutecznej giełdzie zgodnie z planem/trasą.
        W trybie "routes" respektuje `max_retries_per_exchange` oraz pisze decision log (gdy skonfigurowany).
        """
        exchanges_and_retries: list[tuple[str, int]]
        route_name: str | None = None
        route_meta: Mapping[str, str] = {}

        if self._mode == "routes":
            sel = self._select_route_definition(request, context)
            route_name = sel.name
            route_meta = sel.metadata
            exchanges_and_retries = [(ex, max(1, sel.max_retries_per_exchange)) for ex in sel.iter_exchanges()]
        else:
            plan = self._resolve_plan(request.symbol)
            exchanges_and_retries = [(ex, 1) for ex in plan.exchanges]

        attempts_rec: list[dict[str, str]] = []
        attempts_counter = 0
        fallback_used = False
        start = self._time()
        last_error: Exception | None = None

        for exchange_name, max_retries in exchanges_and_retries:
            adapter = self._adapters.get(exchange_name)
            if adapter is None:
                _LOGGER.warning("Brak adaptera %s", exchange_name)
                attempts_rec.append({"exchange": exchange_name, "status": "adapter_missing"})
                continue

            for attempt in range(1, max_retries + 1):
                labels = {
                    "exchange": exchange_name,
                    "symbol": request.symbol,
                    "portfolio": context.portfolio_id,
                }
                attempts_counter += 1
                try:
                    result = adapter.place_order(request)
                except (ExchangeNetworkError, ExchangeThrottlingError) as exc:
                    elapsed = max(0.0, self._time() - start)
                    self._m_latency.observe(elapsed, labels={**labels, "result": "error"})
                    self._m_attempts.inc(labels={**labels, "result": "error"})
                    _LOGGER.warning("Błąd sieci/throttling na %s: %s", exchange_name, exc)
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "error", "error": repr(exc)})
                    last_error = exc
                    # retry jeśli dostępny
                    if attempt < max_retries:
                        time.sleep(min(0.05 * attempt, 0.5))
                        continue
                    break
                except ExchangeAuthError as exc:
                    self._m_attempts.inc(labels={**labels, "result": "auth_error"})
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "auth_error"})
                    _LOGGER.error("Błąd uwierzytelnienia na %s – przerywam fallback.", exchange_name)
                    raise
                except ExchangeAPIError as exc:
                    self._m_attempts.inc(labels={**labels, "result": "api_error"})
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "api_error"})
                    last_error = exc
                    _LOGGER.error("API %s odrzuciło zlecenie (status=%s): %s", exchange_name, getattr(exc, "status_code", "?"), getattr(exc, "message", repr(exc)))
                    # traktujemy jako błąd twardy
                    raise
                except Exception as exc:  # noqa: BLE001
                    elapsed = max(0.0, self._time() - start)
                    self._m_latency.observe(elapsed, labels={**labels, "result": "exception"})
                    self._m_attempts.inc(labels={**labels, "result": "exception"})
                    _LOGGER.exception("Nieoczekiwany błąd egzekucji na %s", exchange_name)
                    attempts_rec.append({"exchange": exchange_name, "attempt": str(attempt), "status": "exception", "error": repr(exc)})
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(min(0.05 * attempt, 0.5))
                        continue
                    break

                # sukces
                elapsed = max(0.0, self._time() - start)
                self._m_latency.observe(elapsed, labels={**labels, "result": "success"})
                self._m_attempts.inc(labels={**labels, "result": "success"})
                self._m_success.inc(labels={"symbol": request.symbol, "portfolio": context.portfolio_id})
                if attempts_counter > 1:
                    self._m_fallbacks.inc(labels={"symbol": request.symbol, "portfolio": context.portfolio_id})
                    fallback_used = True

                with self._bindings_lock:
                    self._bindings[result.order_id] = exchange_name

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
        self._m_failures.inc(labels={"symbol": request.symbol, "portfolio": context.portfolio_id})
        elapsed = max(0.0, self._time() - start)
        attempts_rec.append({"status": "failed", "latency_s": f"{elapsed:.6f}"})
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
        exchange_name: str | None = None
        with self._bindings_lock:
            exchange_name = self._bindings.get(order_id)

        if not exchange_name:
            # bardziej wyrozumiały fallback (zamiast KeyError jak w jednej z gałęzi)
            for adapter in self._adapters.values():
                try:
                    adapter.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning("Nie udało się anulować %s na adapterze %s: %s", order_id, getattr(adapter, "name", "?"), exc)
            return

        adapter = self._adapters.get(exchange_name)
        if adapter is None:
            _LOGGER.warning("Brak adaptera %s do anulacji %s", exchange_name, order_id)
            return
        # różne gałęzie miały sygnatury z/bez symbol=None – użyj najbardziej kompatybilnego wywołania
        try:
            adapter.cancel_order(order_id, symbol=None)  # type: ignore[call-arg]
        except TypeError:
            adapter.cancel_order(order_id)  # type: ignore[misc]

    def flush(self) -> None:  # noqa: D401
        # Flush dotyczy tylko decision logu, jeśli włączony
        self._flush_log()

    # ------------------------------------------------------------------
    # Tryb "routes": wybór trasy
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tryb "plan": trasa na podstawie symbolu
    # ------------------------------------------------------------------
    def _resolve_plan(self, symbol: str) -> RoutingPlan:
        assert self._mode == "plan"
        override = self._overrides.get(symbol)
        if override:
            return override
        assert self._default_plan is not None
        return self._default_plan

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------
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
            with self._decision_log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")

    def _flush_log(self) -> None:
        path = self._decision_log_path
        if path is None or not path.exists():
            return
        with self._log_lock:
            with path.open("ab") as handle:
                handle.flush()


__all__ = ["LiveExecutionRouter", "RoutingPlan", "RouteDefinition"]
