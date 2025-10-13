"""Router egzekucji dla środowiska live obsługujący fallbacki i metryki."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import ExchangeAdapter, OrderRequest, OrderResult
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)

# --- Observability -----------------------------------------------------------
try:  # pragma: no cover - moduł metryk jest opcjonalny w części gałęzi
    from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
except Exception:  # pragma: no cover - fallback gdy metryki nie istnieją
    class _NoopMetric:
        def inc(self, *_args, **_kwargs) -> None:  # noqa: D401 - API licznika
            return None

        def observe(self, *_args, **_kwargs) -> None:  # noqa: D401 - API histogramu
            return None

    class MetricsRegistry:  # type: ignore[override]
        def counter(self, *_args, **_kwargs) -> _NoopMetric:
            return _NoopMetric()

        def histogram(self, *_args, **_kwargs) -> _NoopMetric:
            return _NoopMetric()

    def get_global_metrics_registry() -> MetricsRegistry:  # type: ignore[override]
        return MetricsRegistry()


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RoutingPlan:
    """Opisuje kolejność giełd wykorzystywaną podczas egzekucji."""

    exchanges: Sequence[str]

    def __post_init__(self) -> None:
        unique = []
        seen = set()
        for name in self.exchanges:
            normalized = str(name).strip()
            if not normalized or normalized in seen:
                continue
            unique.append(normalized)
            seen.add(normalized)
        self.exchanges = tuple(unique)

    def __bool__(self) -> bool:  # pragma: no cover - pomocnicze
        return bool(self.exchanges)


class LiveExecutionRouter(ExecutionService):
    """Egzekucja live z fallbackami pomiędzy adapterami giełdowymi."""

    def __init__(
        self,
        adapters: Mapping[str, ExchangeAdapter],
        *,
        default_route: Iterable[str],
        route_overrides: Mapping[str, Sequence[str]] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        latency_buckets: Sequence[float] | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("LiveExecutionRouter wymaga co najmniej jednego adaptera giełdowego")

        self._adapters: Mapping[str, ExchangeAdapter] = {str(k): v for k, v in adapters.items()}
        self._default_route = RoutingPlan(tuple(default_route))
        if not self._default_route:
            raise ValueError("default_route musi zawierać co najmniej jedną giełdę")

        overrides: dict[str, RoutingPlan] = {}
        for symbol, route in (route_overrides or {}).items():
            if not route:
                continue
            overrides[str(symbol)] = RoutingPlan(tuple(route))
        self._route_overrides = overrides

        self._metrics = metrics_registry or get_global_metrics_registry()
        buckets = latency_buckets or (0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
        self._metric_latency = self._metrics.histogram(
            "live_router_latency_seconds",
            "Czas obsługi zlecenia przez LiveExecutionRouter.",
            buckets=buckets,
        )
        self._metric_attempts = self._metrics.counter(
            "live_router_attempts_total",
            "Liczba prób egzekucji zleceń (łącznie z fallbackami).",
        )
        self._metric_failures = self._metrics.counter(
            "live_router_failures_total",
            "Liczba nieudanych egzekucji zlecenia po wszystkich próbach.",
        )
        self._metric_fallbacks = self._metrics.counter(
            "live_router_fallbacks_total",
            "Liczba sytuacji, gdy konieczne było użycie fallbacku giełdowego.",
        )

        self._order_lock = threading.Lock()
        self._order_exchange: MutableMapping[str, str] = {}

    # ------------------------------------------------------------------
    # ExecutionService API
    # ------------------------------------------------------------------
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        route = self._resolve_route(request.symbol)
        last_error: Exception | None = None
        attempts = 0

        for exchange_name in route.exchanges:
            adapter = self._adapters.get(exchange_name)
            if adapter is None:
                _LOGGER.warning("Brak adaptera %s w LiveExecutionRouter", exchange_name)
                continue

            attempts += 1
            start = time.perf_counter()
            labels = {
                "exchange": exchange_name,
                "symbol": request.symbol,
                "portfolio": context.portfolio_id,
            }
            try:
                result = adapter.place_order(request)
            except (ExchangeNetworkError, ExchangeThrottlingError) as exc:
                elapsed = max(0.0, time.perf_counter() - start)
                self._metric_latency.observe(elapsed, labels={**labels, "result": "error"})
                self._metric_attempts.inc(labels={**labels, "result": "error"})
                _LOGGER.warning(
                    "Błąd sieci/throttling podczas składania zlecenia na %s: %s",
                    exchange_name,
                    exc,
                )
                last_error = exc
                continue
            except ExchangeAuthError as exc:
                self._metric_attempts.inc(labels={**labels, "result": "auth_error"})
                _LOGGER.error(
                    "Błąd uwierzytelnienia na giełdzie %s – przerywam próby fallback", exchange_name
                )
                raise
            except ExchangeAPIError as exc:
                self._metric_attempts.inc(labels={**labels, "result": "api_error"})
                last_error = exc
                _LOGGER.error(
                    "Giełda %s odrzuciła zlecenie (status=%s): %s",
                    exchange_name,
                    exc.status_code,
                    exc.message,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - logujemy i próbujemy fallbacku
                elapsed = max(0.0, time.perf_counter() - start)
                self._metric_latency.observe(elapsed, labels={**labels, "result": "exception"})
                self._metric_attempts.inc(labels={**labels, "result": "exception"})
                _LOGGER.exception("Nieoczekiwany błąd egzekucji na %s", exchange_name)
                last_error = exc
                continue

            elapsed = max(0.0, time.perf_counter() - start)
            result_labels = {**labels, "result": "success"}
            self._metric_latency.observe(elapsed, labels=result_labels)
            self._metric_attempts.inc(labels=result_labels)

            if attempts > 1:
                self._metric_fallbacks.inc(labels=labels)

            with self._order_lock:
                self._order_exchange[result.order_id] = exchange_name
            return result

        self._metric_failures.inc(labels={"symbol": request.symbol, "portfolio": context.portfolio_id})
        if last_error is not None:
            raise last_error
        raise RuntimeError("Nie udało się zrealizować zlecenia – brak dostępnych giełd")

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: D401
        exchange_name: str | None
        with self._order_lock:
            exchange_name = self._order_exchange.get(order_id)

        if not exchange_name:
            raise KeyError(f"Brak informacji o giełdzie dla zlecenia {order_id}")

        adapter = self._adapters.get(exchange_name)
        if adapter is None:
            raise KeyError(f"Adapter {exchange_name} nie jest zarejestrowany")
        adapter.cancel_order(order_id, symbol=None)

    def flush(self) -> None:  # noqa: D401
        # Adaptery REST/HTTP nie wymagają dodatkowego flush.
        return None

    # ------------------------------------------------------------------
    # Narzędzia pomocnicze
    # ------------------------------------------------------------------
    def _resolve_route(self, symbol: str) -> RoutingPlan:
        override = self._route_overrides.get(symbol)
        if override is not None and override:
            return override
        return self._default_route


__all__ = ["LiveExecutionRouter", "RoutingPlan"]

