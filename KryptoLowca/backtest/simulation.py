"""Warstwa zgodności dla modułów backtestowych."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, MutableMapping

import pandas as pd

from KryptoLowca.config_manager import ValidationError
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import (
    BaseStrategy,
    StrategyContext,
    StrategyMetadata,
    StrategySignal,
    registry as strategy_registry,
)
from bot_core.backtest import BacktestFill, MatchingConfig, MatchingEngine as _CoreMatchingEngine
from bot_core.backtest.engine import (
    BacktestEngine as _CoreBacktestEngine,
    BacktestError,
    BacktestReport,
    BacktestTrade,
    HistoricalDataProvider as _CoreHistoricalDataProvider,
    PerformanceMetrics,
)

logger = get_logger(__name__)


class HistoricalDataProvider(_CoreHistoricalDataProvider):
    """Adapter delegujący do natywnego providera i mapujący wyjątki."""

    def __init__(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        try:
            super().__init__(data, symbol, timeframe)
        except BacktestError as exc:  # pragma: no cover - defensywne
            raise ValidationError(str(exc)) from exc


class MatchingEngine:
    """Warstwa zgodności otulająca natywny silnik dopasowujący."""

    def __init__(self, config: MatchingConfig) -> None:
        self._config = config
        self._engine = _CoreMatchingEngine(config)

    def submit_market_order(
        self,
        *,
        side: str,
        size: float,
        index: int,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> int:
        try:
            return self._engine.submit_market_order(
                side=side,
                size=size,
                index=index,
                timestamp=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        except ValueError as exc:  # pragma: no cover - zachowujemy typ legacy
            raise ValidationError(str(exc)) from exc

    def process_bar(
        self,
        *,
        index: int,
        timestamp: datetime,
        bar: Mapping[str, float],
    ) -> list[BacktestFill]:
        return self._engine.process_bar(index=index, timestamp=timestamp, bar=bar)

    def force_fill(
        self,
        *,
        side: str,
        size: float,
        timestamp: datetime,
        bar: Mapping[str, float],
    ) -> BacktestFill:
        quantity = float(size)
        if quantity <= 0:
            raise ValidationError("Wielkość wymuszonego zlecenia musi być dodatnia")
        base_price = float(bar.get("close", bar.get("open", 0.0)))
        if base_price <= 0:
            raise ValidationError("Nie można domknąć pozycji bez prawidłowej ceny")
        direction = (side or "").lower()
        if direction not in {"buy", "sell"}:
            raise ValidationError(f"Nieobsługiwany kierunek zlecenia: {side!r}")
        ts = _CoreMatchingEngine._ensure_timestamp(timestamp)  # type: ignore[attr-defined]
        slip = base_price * (self._config.slippage_bps / 10_000.0)
        if direction == "buy":
            execution_price = base_price + slip
            slippage_per_unit = slip
        else:
            execution_price = max(0.0, base_price - slip)
            slippage_per_unit = -slip
        fee_rate = abs(self._config.fee_bps) / 10_000.0
        fee = abs(execution_price * quantity) * fee_rate
        order_id = getattr(self._engine, "_next_order_id", 1)
        setattr(self._engine, "_next_order_id", order_id + 1)
        return BacktestFill(
            order_id=order_id,
            side=direction,
            size=quantity,
            price=execution_price,
            fee=fee,
            slippage=slippage_per_unit * quantity,
            timestamp=ts,
            partial=False,
        )


class BacktestEngine(_CoreBacktestEngine):
    """Otulacz zapewniający kompatybilność ze starym API."""

    def __init__(
        self,
        *,
        strategy_name: str,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        matching: MatchingConfig,
        allow_short: bool = False,
        context_extra: Mapping[str, object] | None = None,
        strategy_registry=strategy_registry,
    ) -> None:
        if strategy_name.lower() not in strategy_registry:
            raise ValidationError(f"Strategia '{strategy_name}' nie jest zarejestrowana")
        strategy_cls: type[BaseStrategy] = strategy_registry.get(strategy_name)
        metadata = getattr(
            strategy_cls,
            "metadata",
            StrategyMetadata(name=strategy_name, description=""),
        )

        def _strategy_factory() -> BaseStrategy:
            return strategy_cls()

        def _context_builder(payload: Mapping[str, object]) -> StrategyContext:
            timestamp = payload["timestamp"]
            if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            extra_raw = payload.get("extra", {})
            if isinstance(extra_raw, MutableMapping):
                extra: MutableMapping[str, object] = extra_raw  # type: ignore[assignment]
            else:
                extra = dict(extra_raw) if isinstance(extra_raw, Mapping) else {}
            return StrategyContext(
                symbol=str(payload["symbol"]),
                timeframe=str(payload["timeframe"]),
                portfolio_value=float(payload["portfolio_value"]),
                position=float(payload["position"]),
                timestamp=timestamp,
                metadata=metadata,
                extra=extra,
            )

        try:
            super().__init__(
                strategy_factory=_strategy_factory,
                context_builder=_context_builder,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                matching=matching,
                allow_short=allow_short,
                context_extra=context_extra or {},
                metadata=metadata,
                logger=logger,
            )
        except BacktestError as exc:  # pragma: no cover - defensywne
            raise ValidationError(str(exc)) from exc

        self._strategy_name = strategy_name
        self._matching_config = matching
        self._allow_short = bool(allow_short)
        self._initial_balance = float(initial_balance)

    def run(self) -> BacktestReport:
        try:
            report = super().run()
        except BacktestError as exc:  # pragma: no cover - defensywne
            raise ValidationError(str(exc)) from exc
        report.parameters["strategy"] = self._strategy_name
        return report

    def _determine_size(
        self,
        signal: StrategySignal,
        context: StrategyContext,
        market_payload: Mapping[str, object],
    ) -> float:
        return super()._determine_size(signal, context, market_payload)


def evaluate_strategy_backtest(config: Mapping[str, object], report: BacktestReport) -> None:
    if report.metrics is None:
        raise ValidationError("Raport backtestu nie zawiera metryk")
    if not report.trades:
        raise ValidationError("Strategia nie wygenerowała żadnych domkniętych transakcji")
    if report.metrics.total_return_pct <= 0:
        raise ValidationError("Stopa zwrotu strategii jest nieakceptowalna")
    max_drawdown_allowed = float(config.get("max_position_notional_pct", 0.02)) * float(
        config.get("max_leverage", 1.0)
    ) * 100.0
    if report.metrics.max_drawdown_pct > max_drawdown_allowed:
        raise ValidationError("Obsunięcie przekracza limity strategii")


__all__ = [
    "BacktestEngine",
    "BacktestFill",
    "BacktestReport",
    "BacktestTrade",
    "HistoricalDataProvider",
    "MatchingConfig",
    "MatchingEngine",
    "PerformanceMetrics",
    "evaluate_strategy_backtest",
]
