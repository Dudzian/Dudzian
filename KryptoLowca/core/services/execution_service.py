"""Serwis wykonania zleceń – most łączący stare GUI z nową warstwą bot_core."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Mapping, MutableMapping

from KryptoLowca.exchanges.interfaces import ExchangeAdapter, OrderRequest, OrderStatus
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import StrategyContext, StrategySignal

from .error_policy import guard_exceptions

try:  # pragma: no cover - bot_core jest opcjonalny w środowisku CI
    from bot_core.execution import ExecutionContext as CoreExecutionContext
    from bot_core.execution import LiveExecutionRouter
    from bot_core.exchanges.base import OrderRequest as CoreOrderRequest
except Exception:  # pragma: no cover - fallback gdy bot_core nie jest dostępny
    CoreExecutionContext = None  # type: ignore[assignment]
    LiveExecutionRouter = None  # type: ignore[assignment]
    CoreOrderRequest = None  # type: ignore[assignment]

logger = get_logger(__name__)


class ExecutionService:
    """Warstwa zgodności dla historycznego GUI i AutoTradera."""

    def __init__(
        self,
        adapter: ExchangeAdapter | None,
        *,
        router: "LiveExecutionRouter" | None = None,
        account_manager: Any | None = None,
        context_builder: Callable[[StrategyContext], "CoreExecutionContext" | None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._router = router
        self._account_manager = account_manager
        self._context_builder = context_builder or self._default_context_builder
        self._logger = get_logger(__name__)

    def set_adapter(self, adapter: ExchangeAdapter) -> None:
        """Zachowuje kompatybilność z dotychczasowym API."""

        self._adapter = adapter

    def bind_router(
        self,
        router: "LiveExecutionRouter",
        *,
        context_builder: Callable[[StrategyContext], "CoreExecutionContext" | None] | None = None,
    ) -> None:
        if LiveExecutionRouter is None:
            raise RuntimeError("bot_core.execution.LiveExecutionRouter nie jest dostępny w tej dystrybucji")
        self._router = router
        if context_builder is not None:
            self._context_builder = context_builder

    def bind_account_manager(self, manager: Any) -> None:
        """Pozwala wstrzyknąć MultiExchangeAccountManager do obsługi wielu kont."""

        self._account_manager = manager

    def update_market_data(self, symbol: str, timeframe: str, market_payload: Mapping[str, object]) -> None:
        updater = getattr(self._adapter, "update_market_data", None)
        if callable(updater):
            updater(symbol, timeframe, market_payload)

    def portfolio_snapshot(self, symbol: str) -> Mapping[str, object] | None:
        getter = getattr(self._adapter, "portfolio_snapshot", None)
        if callable(getter):
            result = getter(symbol)
            if result is None or isinstance(result, Mapping):
                return result
            raise TypeError(
                "portfolio_snapshot must return Mapping[str, object] or None"
            )
        return None

    @guard_exceptions("ExecutionService")
    async def execute(self, signal: StrategySignal, context: StrategyContext) -> Any:
        if signal.action == "HOLD":
            self._logger.info("Sygnał HOLD – nie wysyłamy zlecenia")
            return None
        if signal.size is None:
            self._logger.warning("Brak wielkości pozycji w sygnale")
            return None

        if self._account_manager is not None and CoreOrderRequest is not None:
            return await self._dispatch_via_account_manager(signal, context)

        if self._router is not None and CoreOrderRequest is not None:
            return await self._dispatch_via_router(signal, context)

        if self._adapter is None:
            self._logger.error("Brak skonfigurowanego adaptera wykonawczego")
            return None

        side = "buy" if signal.action.upper() == "BUY" else "sell"
        order = OrderRequest(
            symbol=context.symbol,
            side=side,
            quantity=signal.size,
            order_type="MARKET" if signal.action.upper() in {"BUY", "SELL"} else "LIMIT",
            extra_params={
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            },
        )
        submit = getattr(self._adapter, "submit_order")
        try:
            result = submit(order)
        except TypeError:
            result = submit(
                symbol=context.symbol,
                side=side,
                size=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

        payload = await result if inspect.isawaitable(result) else result
        self._logger.info("Zlecenie wysłane: %s", payload)
        return payload.raw if isinstance(payload, OrderStatus) else payload

    async def _dispatch_via_router(self, signal: StrategySignal, context: StrategyContext) -> Any:
        assert self._router is not None
        request = self._build_core_order_request(signal, context)
        exec_context = self._build_execution_context(context)
        if exec_context is None:
            self._logger.warning("Brak ExecutionContext – pomijam wysyłkę przez router")
            return None
        self._logger.info("Wysyłam zlecenie przez LiveExecutionRouter: %s", request)
        result = await asyncio.to_thread(self._router.execute, request, exec_context)
        return result

    async def _dispatch_via_account_manager(self, signal: StrategySignal, context: StrategyContext) -> Any:
        manager = self._account_manager
        request = self._build_core_order_request(signal, context)
        if manager is None:
            self._logger.warning("Brak menedżera kont – pomijam wysyłkę")
            return None
        self._logger.info("Wysyłam zlecenie przez MultiExchangeAccountManager: %s", request)
        result = await manager.dispatch_order(request)
        return result

    def _build_execution_context(self, context: StrategyContext) -> "CoreExecutionContext" | None:
        try:
            return self._context_builder(context)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Nie udało się zbudować ExecutionContext")
            return None

    def _build_core_order_request(
        self, signal: StrategySignal, context: StrategyContext
    ) -> "CoreOrderRequest":
        if CoreOrderRequest is None:
            raise RuntimeError("bot_core.exchanges.base.OrderRequest niedostępny")

        action = signal.action.upper()
        side = "buy" if action == "BUY" else "sell"
        order_type = "market" if action in {"BUY", "SELL"} else "limit"
        metadata: MutableMapping[str, object] = {
            "strategy": context.metadata.name,
            "timeframe": context.timeframe,
            "mode": context.extra.get("mode", "demo"),
        }
        metadata.update({f"signal_{k}": v for k, v in signal.payload.items()})

        price = signal.payload.get("price")
        if price is None:
            price = context.extra.get("price")

        request = CoreOrderRequest(
            symbol=context.symbol,
            side=side,
            quantity=float(signal.size),
            order_type=order_type,
            price=float(price) if price is not None else None,
            client_order_id=signal.payload.get("client_order_id"),
            stop_price=signal.stop_loss,
            metadata=dict(metadata),
        )
        return request

    def _default_context_builder(self, context: StrategyContext) -> "CoreExecutionContext" | None:
        if CoreExecutionContext is None:
            return None
        extra = dict(context.extra or {})
        portfolio = str(extra.get("portfolio_id") or extra.get("portfolio") or "default")
        risk_profile = str(extra.get("risk_profile") or "baseline")
        environment = str(extra.get("environment") or extra.get("mode") or "demo")
        metadata = {
            "symbol": context.symbol,
            "timeframe": context.timeframe,
            "strategy": context.metadata.name,
        }
        for key, value in extra.items():
            if isinstance(key, str):
                metadata[f"extra_{key}"] = value
        return CoreExecutionContext(
            portfolio_id=portfolio,
            risk_profile=risk_profile,
            environment=environment,
            metadata=metadata,
        )


__all__ = ["ExecutionService", "ExchangeAdapter"]
