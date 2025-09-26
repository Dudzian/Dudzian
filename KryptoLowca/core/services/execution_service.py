"""Serwis wykonania zleceń – abstrakcja nad adapterami giełdowymi."""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import StrategyContext, StrategySignal

from .error_policy import guard_exceptions

logger = get_logger(__name__)


class ExchangeAdapter(Protocol):
    async def submit_order(self, *, symbol: str, side: str, size: float, **kwargs: Any) -> Mapping[str, Any]:
        ...


class ExecutionService:
    def __init__(self, adapter: ExchangeAdapter) -> None:
        self._adapter = adapter
        self._logger = get_logger(__name__)

    @guard_exceptions("ExecutionService")
    async def execute(self, signal: StrategySignal, context: StrategyContext) -> Mapping[str, Any] | None:
        if signal.action == "HOLD":
            self._logger.info("Sygnał HOLD – nie wysyłamy zlecenia")
            return None
        if signal.size is None:
            self._logger.warning("Brak wielkości pozycji w sygnale")
            return None
        side = "buy" if signal.action.upper() == "BUY" else "sell"
        payload = await self._adapter.submit_order(
            symbol=context.symbol,
            side=side,
            size=signal.size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        self._logger.info("Zlecenie wysłane: %s", payload)
        return payload


__all__ = ["ExecutionService", "ExchangeAdapter"]
