"""Serwis wykonania zleceń – abstrakcja nad adapterami giełdowymi."""
from __future__ import annotations

import inspect
from typing import Any, Mapping

from KryptoLowca.exchanges.interfaces import ExchangeAdapter, OrderRequest, OrderStatus
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import StrategyContext, StrategySignal

from .error_policy import guard_exceptions

logger = get_logger(__name__)


class ExecutionService:
    def __init__(self, adapter: ExchangeAdapter) -> None:
        self._adapter = adapter
        self._logger = get_logger(__name__)

    def set_adapter(self, adapter: ExchangeAdapter) -> None:
        self._adapter = adapter

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
    async def execute(self, signal: StrategySignal, context: StrategyContext) -> Mapping[str, Any] | None:
        if signal.action == "HOLD":
            self._logger.info("Sygnał HOLD – nie wysyłamy zlecenia")
            return None
        if signal.size is None:
            self._logger.warning("Brak wielkości pozycji w sygnale")
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


__all__ = ["ExecutionService", "ExchangeAdapter"]
