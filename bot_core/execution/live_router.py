"""Router zleceń live zapewniający fallbacki, metryki oraz decision log."""
from __future__ import annotations

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
from bot_core.observability import MetricsRegistry, get_global_metrics_registry
from bot_core.security.signing import build_hmac_signature


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RouteDefinition:
    """Pojedyncza definicja trasy egzekucji."""

    name: str
    exchanges: Sequence[str]
    symbols: Sequence[str] = ()
    risk_profiles: Sequence[str] = ()
    max_retries_per_exchange: int = 2
    latency_budget_ms: float = 250.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def iter_exchanges(self) -> Iterable[str]:
        for exchange in self.exchanges:
            normalized = exchange.strip()
            if normalized:
                yield normalized


class LiveExecutionRouter(ExecutionService):
    """Router egzekucji, który próbuje wielu giełd zgodnie z definicją tras."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ExchangeAdapter],
        routes: Sequence[RouteDefinition],
        default_route: str | None = None,
        decision_log_path: str | os.PathLike[str],
        decision_log_hmac_key: bytes | None = None,
        decision_log_key_id: str | None = None,
        metrics: MetricsRegistry | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        if not routes:
            raise ValueError("LiveExecutionRouter wymaga co najmniej jednej trasy routingu")
        self._routes: tuple[RouteDefinition, ...] = tuple(routes)
        self._routes_by_name = {route.name: route for route in self._routes}
        if len(self._routes_by_name) != len(self._routes):
            raise ValueError("Nazwy tras live routera muszą być unikalne")
        if default_route is not None and default_route not in self._routes_by_name:
            raise KeyError(f"Domyślna trasa '{default_route}' nie istnieje w konfiguracji")
        self._default_route = default_route

        self._adapters: dict[str, ExchangeAdapter] = {
            name: adapter for name, adapter in adapters.items()
        }
        if not self._adapters:
            raise ValueError("LiveExecutionRouter wymaga co najmniej jednego adaptera giełdowego")

        self._decision_log_path = Path(decision_log_path)
        self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._decision_log_key = None
        if decision_log_hmac_key is not None:
            if len(decision_log_hmac_key) < 32:
                raise ValueError("Klucz HMAC decision logu musi mieć co najmniej 32 bajty")
            self._decision_log_key = bytes(decision_log_hmac_key)
        self._decision_log_key_id = decision_log_key_id
        self._log_lock = threading.Lock()

        self._metrics = metrics or get_global_metrics_registry()
        self._metric_orders_total = self._metrics.counter(
            "live_orders_total",
            "Liczba zleceń wysłanych przez LiveExecutionRouter",
        )
        self._metric_orders_failed = self._metrics.counter(
            "live_orders_failed_total",
            "Liczba zleceń, które nie zostały zrealizowane przez żadną giełdę",
        )
        self._metric_orders_fallback = self._metrics.counter(
            "live_orders_fallback_total",
            "Liczba zleceń wymagających fallbacku na alternatywną giełdę",
        )
        self._metric_latency = self._metrics.histogram(
            "live_execution_latency_seconds",
            "Czas realizacji zleceń live (sekundy)",
            buckets=(0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000),
        )

        self._time = time_source or time.monotonic
        self._bindings: MutableMapping[str, str] = {}
        self._bindings_lock = threading.Lock()

    # ------------------------------------------------------------------
    # ExecutionService API
    # ------------------------------------------------------------------
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        route = self._select_route(request, context)
        attempts: list[dict[str, str]] = []
        fallback_used = False
        start_time = self._time()
        last_error: Exception | None = None

        for exchange_name in route.iter_exchanges():
            adapter = self._adapters.get(exchange_name)
            if adapter is None:
                _LOGGER.error("Brak adaptera giełdy %s dla trasy %s", exchange_name, route.name)
                attempts.append({"exchange": exchange_name, "status": "adapter_missing"})
                continue

            for attempt in range(1, max(route.max_retries_per_exchange, 1) + 1):
                attempt_record = {"exchange": exchange_name, "attempt": str(attempt)}
                try:
                    result = adapter.place_order(request)
                except Exception as exc:  # noqa: BLE001 - logujemy każdy wyjątek adaptera
                    last_error = exc
                    attempt_record["status"] = "error"
                    attempt_record["error"] = repr(exc)
                    attempts.append(attempt_record)
                    if attempt < route.max_retries_per_exchange:
                        time.sleep(min(0.05 * attempt, 0.5))
                    continue

                latency = max(0.0, self._time() - start_time)
                self._metric_orders_total.inc(labels={"exchange": exchange_name, "route": route.name})
                self._metric_latency.observe(latency, labels={"exchange": exchange_name, "route": route.name})
                if attempts:
                    fallback_used = True
                    self._metric_orders_fallback.inc(labels={"route": route.name})

                with self._bindings_lock:
                    self._bindings[result.order_id] = exchange_name

                attempts.append({"exchange": exchange_name, "status": "success", "latency_s": f"{latency:.6f}"})
                self._write_decision_log(
                    route=route,
                    request=request,
                    context=context,
                    result=result,
                    attempts=attempts,
                    latency_seconds=latency,
                    fallback_used=fallback_used,
                )
                return result

            fallback_used = True

        self._metric_orders_failed.inc(labels={"route": route.name})
        latency = max(0.0, self._time() - start_time)
        attempts.append({"status": "failed", "latency_s": f"{latency:.6f}"})
        self._write_decision_log(
            route=route,
            request=request,
            context=context,
            result=None,
            attempts=attempts,
            latency_seconds=latency,
            fallback_used=fallback_used,
            error=repr(last_error) if last_error else None,
        )
        raise RuntimeError(
            f"Żadna giełda nie zrealizowała zlecenia (route={route.name}, symbol={request.symbol})"
        )

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: ARG002 - kontekst informacyjny
        exchange_name: str | None = None
        with self._bindings_lock:
            exchange_name = self._bindings.get(order_id)
        if exchange_name is None:
            # fallback: spróbujmy anulować na wszystkich giełdach
            for adapter in self._adapters.values():
                try:
                    adapter.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001 - logujemy ostrzeżenie i kontynuujemy
                    _LOGGER.warning("Anulacja %s na adapterze %s nie powiodła się: %s", order_id, adapter.name, exc)
            return

        adapter = self._adapters.get(exchange_name)
        if adapter is None:
            _LOGGER.warning("Brak adaptera %s do anulacji %s", exchange_name, order_id)
            return
        adapter.cancel_order(order_id)

    def flush(self) -> None:
        # Router nie buforuje stanu, ale zapewnia synchronizację decision logu.
        self._flush_log()

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia
    # ------------------------------------------------------------------
    def _select_route(self, request: OrderRequest, context: ExecutionContext) -> RouteDefinition:
        metadata = context.metadata or {}
        requested_route = metadata.get("execution_route")
        if requested_route:
            route = self._routes_by_name.get(str(requested_route))
            if route is None:
                raise KeyError(f"Wymagana trasa execution_route={requested_route} nie istnieje")
            return route

        for route in self._routes:
            if route.symbols and request.symbol not in route.symbols:
                continue
            if route.risk_profiles and context.risk_profile not in route.risk_profiles:
                continue
            return route

        if self._default_route:
            return self._routes_by_name[self._default_route]

        return self._routes[0]

    def _write_decision_log(
        self,
        *,
        route: RouteDefinition,
        request: OrderRequest,
        context: ExecutionContext,
        attempts: Sequence[Mapping[str, str]],
        latency_seconds: float,
        fallback_used: bool,
        result: OrderResult | None,
        error: str | None = None,
    ) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "route": route.name,
            "metadata": dict(route.metadata),
            "request": {
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type,
                "quantity": request.quantity,
                "price": request.price,
                "time_in_force": request.time_in_force,
                "client_order_id": request.client_order_id,
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

        document = {"payload": payload}
        if self._decision_log_key is not None:
            document["signature"] = build_hmac_signature(
                payload,
                key=self._decision_log_key,
                algorithm="HMAC-SHA384",
                key_id=self._decision_log_key_id,
            )

        serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
        with self._log_lock:
            with self._decision_log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")

    def _flush_log(self) -> None:
        if not self._decision_log_path.exists():
            return
        with self._log_lock:
            with self._decision_log_path.open("ab") as handle:
                handle.flush()


__all__ = ["LiveExecutionRouter", "RouteDefinition"]

