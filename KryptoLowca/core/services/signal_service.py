"""Serwis sygnałów łączący strategie z silnikiem wykonawczym."""
from __future__ import annotations

from datetime import datetime
from typing import Mapping

from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import (
    BaseStrategy,
    DataProvider,
    StrategyContext,
    StrategyMetadata,
    StrategySignal,
    registry as global_registry,
)

from .error_policy import guard_exceptions


class SignalService:
    """Centralizuje uruchamianie strategii i walidację wyników."""

    def __init__(self, *, strategy_registry=global_registry) -> None:
        self._registry = strategy_registry
        self._logger = get_logger(__name__)

    @guard_exceptions("SignalService")
    async def run_strategy(
        self,
        strategy_name: str,
        context: StrategyContext,
        market_payload: Mapping[str, object],
        data_provider: DataProvider,
    ) -> StrategySignal | None:
        strategy_cls = self._registry.get(strategy_name)
        strategy: BaseStrategy = strategy_cls()
        await strategy.prepare(context, data_provider)
        signal = await strategy.handle_market_data(context, market_payload)
        self._logger.debug(
            "Sygnał %s: action=%s confidence=%.2f", strategy_name, signal.action, signal.confidence
        )
        return signal

    def build_context(
        self,
        *,
        symbol: str,
        timeframe: str,
        portfolio_value: float,
        position: float,
        metadata: StrategyMetadata,
        mode: str = "demo",
    ) -> StrategyContext:
        return StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            portfolio_value=portfolio_value,
            position=position,
            timestamp=datetime.utcnow(),
            metadata=metadata,
            extra={"mode": mode},
        )


__all__ = ["SignalService"]
