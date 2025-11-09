from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, MutableMapping

from bot_core.execution.base import ExecutionContext, ExecutionService, RetryPolicy
from bot_core.execution.paper import MarketMetadata
from bot_core.exchanges.base import ExchangeAdapter, OrderRequest, OrderResult
from bot_core.runtime.journal import TradingDecisionJournal, log_decision_event

try:  # pragma: no cover - moduł może nie istnieć w każdej gałęzi
    from bot_core.exchanges.errors import ExchangeNetworkError, ExchangeThrottlingError
except Exception:  # pragma: no cover
    class _FallbackExchangeNetworkError(Exception):
        """Fallback when giełdowe wyjątki nie są dostępne."""

    class _FallbackExchangeThrottlingError(Exception):
        """Fallback when giełdowe wyjątki nie są dostępne."""

    ExchangeNetworkError = _FallbackExchangeNetworkError
    ExchangeThrottlingError = _FallbackExchangeThrottlingError


_RETRYABLE_EXCEPTIONS = (ExchangeNetworkError, ExchangeThrottlingError, TimeoutError)


def _quantize(value: float, step: float | None) -> float:
    if step is None or step <= 0:
        return value
    if value <= 0:
        return 0.0
    return math.floor(value / step) * step


def _normalise_decision(decision: Any) -> MutableMapping[str, Any]:
    if isinstance(decision, Mapping):
        return dict(decision)
    payload: MutableMapping[str, Any] = {}
    to_mapping = getattr(decision, "to_mapping", None)
    if callable(to_mapping):
        try:
            mapping = to_mapping()
        except Exception:  # pragma: no cover - defensywna osłona
            mapping = None
        if isinstance(mapping, Mapping):
            payload.update(mapping)
    for attribute in ("symbol", "side", "action", "price", "quantity", "notional", "order_type", "metadata"):
        if hasattr(decision, attribute):
            payload[attribute] = getattr(decision, attribute)
    candidate = getattr(decision, "candidate", None)
    if candidate is not None:
        payload.setdefault("candidate", _normalise_decision(candidate))
    return payload


def decision_to_order_request(
    decision: Mapping[str, Any] | Any,
    *,
    price: float | None = None,
    market: MarketMetadata | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OrderRequest:
    """Przekształca decyzję strategii na ``OrderRequest``."""

    payload = _normalise_decision(decision)
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping):
        source = candidate
    else:
        source = payload

    symbol = source.get("symbol") or payload.get("symbol")
    if not symbol:
        raise ValueError("decision payload missing symbol")

    action = str(source.get("action") or payload.get("action") or source.get("side") or payload.get("side") or "buy").lower()
    if action in {"enter", "long", "buy"}:
        side = "buy"
    elif action in {"exit", "sell", "short"}:
        side = "sell"
    else:
        side = str(source.get("side") or payload.get("side") or "buy")

    order_type = str(source.get("order_type") or payload.get("order_type") or "market")
    request_price = source.get("price", payload.get("price", price))
    if request_price is None:
        request_price = price

    notional = source.get("notional", payload.get("notional"))
    quantity = source.get("quantity", payload.get("quantity"))
    if quantity is None and notional is not None and request_price:
        try:
            quantity = float(notional) / float(request_price)
        except (TypeError, ValueError):
            quantity = None
    if quantity is None:
        quantity = 0.0
    try:
        quantity = float(quantity)
    except (TypeError, ValueError):
        quantity = 0.0

    try:
        resolved_price = float(request_price) if request_price is not None else None
    except (TypeError, ValueError):
        resolved_price = None

    meta: MutableMapping[str, object] = {}
    for block in (payload.get("metadata"), source.get("metadata"), metadata):
        if isinstance(block, Mapping):
            meta.update({str(key): value for key, value in block.items()})

    if market is not None:
        quantity = _quantize(quantity, market.step_size)
        if resolved_price is not None:
            resolved_price = _quantize(resolved_price, market.tick_size)
        if market.min_quantity and quantity < market.min_quantity:
            raise ValueError("quantity below market minimum")
        if market.min_notional and resolved_price is not None:
            notional_value = quantity * resolved_price
            if notional_value < market.min_notional:
                raise ValueError("notional below market minimum")

    return OrderRequest(
        symbol=str(symbol),
        side=str(side),
        quantity=max(0.0, quantity),
        order_type=order_type,
        price=resolved_price,
        metadata=meta or None,
    )


@dataclass(slots=True)
class ExchangeAdapterExecutionService(ExecutionService):
    """Most pomiędzy ``OrderRequest`` a aktywnym ``ExchangeAdapterem``."""

    adapter: ExchangeAdapter | Callable[[], ExchangeAdapter]
    journal: TradingDecisionJournal | None = None
    retry_policy: RetryPolicy | None = None
    max_attempts: int = 3
    backoff_base: float = 0.25
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    clock: Callable[[], datetime] = field(
        default_factory=lambda: (lambda: datetime.now(timezone.utc))
    )

    def _resolve_adapter(self) -> ExchangeAdapter:
        adapter = self.adapter
        if callable(adapter):
            adapter = adapter()
        if adapter is None:
            raise RuntimeError("Exchange adapter unavailable")
        return adapter

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        adapter = self._resolve_adapter()
        event_metadata: MutableMapping[str, object] = {}
        if isinstance(context.metadata, Mapping):
            event_metadata.update({str(k): v for k, v in context.metadata.items()})
        if request.metadata:
            event_metadata.update({str(k): v for k, v in request.metadata.items()})

        self._log_event(
            context,
            event="order_submitted",
            request=request,
            status="submitted",
            metadata=event_metadata,
        )

        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                result = adapter.place_order(request)
            except _RETRYABLE_EXCEPTIONS as exc:
                delay = self._compute_backoff(attempt, exc)
                self.logger.warning(
                    "Retryable exchange error on attempt %s: %s", attempt, exc
                )
                if attempt >= max(1, int(self.max_attempts)):
                    self._log_event(
                        context,
                        event="order_failed",
                        request=request,
                        status="failed",
                        metadata={**event_metadata, "error": str(exc)},
                    )
                    raise
                if delay > 0:
                    time.sleep(delay)
                continue
            except Exception as exc:
                self._log_event(
                    context,
                    event="order_failed",
                    request=request,
                    status="failed",
                    metadata={**event_metadata, "error": str(exc)},
                )
                raise
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log_event(
                context,
                event="order_filled",
                request=request,
                status=result.status,
                quantity=result.filled_quantity,
                price=result.avg_price,
                latency_ms=latency_ms,
                metadata={**event_metadata, "order_id": result.order_id},
            )
            return result

    def execute_decision(
        self,
        decision: Mapping[str, Any] | Any,
        *,
        context: ExecutionContext,
        price: float | None = None,
        market: MarketMetadata | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> OrderResult:
        request = decision_to_order_request(decision, price=price, market=market, metadata=metadata)
        return self.execute(request, context)

    def _compute_backoff(self, attempt: int, error: Exception) -> float:
        if self.retry_policy is not None:
            try:
                delay = float(self.retry_policy.on_error(attempt, error))
            except Exception:  # pragma: no cover - polityka retry nie powinna psuć przepływu
                delay = 0.0
        else:
            delay = min(self.backoff_base * (2 ** max(0, attempt - 1)), 5.0)
        return max(0.0, delay)

    def cancel(self, order_id: str, context: ExecutionContext) -> None:
        adapter = self._resolve_adapter()
        cancel = getattr(adapter, "cancel_order", None)
        if not callable(cancel):
            return
        try:
            cancel(order_id)
        except TypeError:
            cancel(order_id, symbol=None)

    def flush(self) -> None:
        return None

    def _log_event(
        self,
        context: ExecutionContext,
        *,
        event: str,
        request: OrderRequest,
        status: str | None,
        metadata: Mapping[str, object] | None = None,
        quantity: float | None = None,
        price: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if self.journal is None:
            return
        merged_meta: MutableMapping[str, object] = {}
        if metadata:
            merged_meta.update({str(k): v for k, v in metadata.items()})
        log_decision_event(
            self.journal,
            event=event,
            timestamp=self.clock(),
            environment=context.environment,
            portfolio=context.portfolio_id,
            risk_profile=context.risk_profile,
            symbol=request.symbol,
            side=request.side,
            quantity=quantity if quantity is not None else request.quantity,
            price=price if price is not None else request.price,
            status=status,
            latency_ms=latency_ms,
            metadata=merged_meta,
        )


__all__ = ["decision_to_order_request", "ExchangeAdapterExecutionService"]
