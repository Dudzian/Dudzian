"""Kontroler łączący warstwę danych, strategię, ryzyko i egzekucję."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from bot_core.config.models import CoreConfig, ControllerRuntimeConfig
from bot_core.data.base import OHLCVRequest
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.risk.base import RiskEngine
from bot_core.strategies.base import MarketSnapshot, StrategySignal
from bot_core.strategies.daily_trend import DailyTrendMomentumStrategy

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DailyTrendController:
    """Prosty kontroler realizujący cykl: backfill -> strategia -> ryzyko -> egzekucja."""

    core_config: CoreConfig
    environment_name: str
    controller_name: str
    symbols: Sequence[str]
    backfill_service: OHLCVBackfillService
    data_source: CachedOHLCVSource
    strategy: DailyTrendMomentumStrategy
    risk_engine: RiskEngine
    execution_service: ExecutionService
    account_loader: Callable[[], AccountSnapshot]
    execution_context: ExecutionContext
    position_size: float = 1.0
    _environment: object = field(init=False, repr=False)
    _runtime: ControllerRuntimeConfig = field(init=False, repr=False)
    _risk_profile: str = field(init=False, repr=False)
    _positions: dict[str, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Wymagany jest przynajmniej jeden symbol do obsługi.")
        try:
            self._environment = self.core_config.environments[self.environment_name]
        except KeyError as exc:
            raise KeyError(f"Brak konfiguracji środowiska '{self.environment_name}' w CoreConfig") from exc
        try:
            self._runtime: ControllerRuntimeConfig = self.core_config.runtime_controllers[self.controller_name]
        except KeyError as exc:
            raise KeyError(f"Brak sekcji runtime dla kontrolera '{self.controller_name}' w CoreConfig") from exc
        self._risk_profile = self._environment.risk_profile

    @property
    def tick_seconds(self) -> float:
        """Częstotliwość wywołań kontrolera z konfiguracji."""

        return self._runtime.tick_seconds

    @property
    def interval(self) -> str:
        """Interwał OHLCV wykorzystywany przez kontroler."""

        return self._runtime.interval

    def run_cycle(self, *, start: int, end: int) -> list[OrderResult]:
        """Przeprowadza pojedynczy cykl przetwarzania danych i składania zleceń."""

        if start > end:
            raise ValueError("Parametr start nie może być większy niż end")

        self.backfill_service.synchronize(
            symbols=self.symbols,
            interval=self.interval,
            start=start,
            end=end,
        )

        executed: list[OrderResult] = []
        for symbol in self.symbols:
            response = self.data_source.fetch_ohlcv(
                OHLCVRequest(symbol=symbol, interval=self.interval, start=start, end=end)
            )
            snapshots = self._to_snapshots(symbol, response.columns, response.rows)
            for snapshot in snapshots:
                signals = self.strategy.on_data(snapshot)
                executed.extend(self._handle_signals(snapshot, signals))
        return executed

    # ------------------------------------------------------------------
    # Metody pomocnicze
    # ------------------------------------------------------------------
    def _handle_signals(
        self,
        snapshot: MarketSnapshot,
        signals: Sequence[StrategySignal],
    ) -> list[OrderResult]:
        results: list[OrderResult] = []
        for signal in signals:
            base_request = self._build_order_request(snapshot, signal)
            account_snapshot = self.account_loader()
            risk_result = self.risk_engine.apply_pre_trade_checks(
                base_request,
                account=account_snapshot,
                profile_name=self._risk_profile,
            )
            if not risk_result.allowed:
                _LOGGER.info(
                    "Kontroler %s: sygnał %s dla %s odrzucony przez silnik ryzyka (%s)",
                    self.controller_name,
                    signal.side,
                    snapshot.symbol,
                    risk_result.reason,
                )
                continue

            quantity = base_request.quantity
            if risk_result.adjustments and "quantity" in risk_result.adjustments:
                quantity = float(risk_result.adjustments["quantity"])
            if quantity <= 0:
                _LOGGER.debug(
                    "Kontroler %s: dostosowana wielkość <= 0 dla %s – pomijam egzekucję.",
                    self.controller_name,
                    snapshot.symbol,
                )
                continue

            request = OrderRequest(
                symbol=base_request.symbol,
                side=base_request.side,
                quantity=quantity,
                order_type=base_request.order_type,
                price=base_request.price,
                time_in_force=base_request.time_in_force,
                client_order_id=base_request.client_order_id,
            )
            result = self.execution_service.execute(request, self.execution_context)
            self._post_fill(signal.side, snapshot.symbol, request, result)
            results.append(result)
        return results

    def _build_order_request(self, snapshot: MarketSnapshot, signal: StrategySignal) -> OrderRequest:
        side = signal.side.lower()
        price = snapshot.close
        return OrderRequest(
            symbol=snapshot.symbol,
            side=side,
            quantity=self.position_size,
            order_type="market",
            price=price,
        )

    def _to_snapshots(
        self,
        symbol: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> list[MarketSnapshot]:
        if not rows:
            return []

        index: Mapping[str, int] = {column.lower(): idx for idx, column in enumerate(columns)}
        try:
            open_time_idx = index["open_time"]
        except KeyError as exc:
            if "timestamp" in index:
                open_time_idx = index["timestamp"]
            else:  # pragma: no cover - zabezpieczenie na inne formaty
                raise ValueError("Brak kolumny open_time w danych OHLCV") from exc
        for key in ("open", "high", "low", "close"):
            if key not in index:
                raise ValueError(f"Brak kolumny '{key}' w danych OHLCV")
        open_idx = index["open"]
        high_idx = index["high"]
        low_idx = index["low"]
        close_idx = index["close"]
        volume_idx = index.get("volume")

        snapshots = [
            MarketSnapshot(
                symbol=symbol,
                timestamp=int(float(row[open_time_idx])),
                open=float(row[open_idx]),
                high=float(row[high_idx]),
                low=float(row[low_idx]),
                close=float(row[close_idx]),
                volume=float(row[volume_idx]) if volume_idx is not None else 0.0,
            )
            for row in rows
            if len(row) > close_idx
        ]
        snapshots.sort(key=lambda item: item.timestamp)
        return snapshots

    def _post_fill(
        self,
        side: str,
        symbol: str,
        request: OrderRequest,
        result: OrderResult,
    ) -> None:
        avg_price = result.avg_price or request.price or 0.0
        notional = avg_price * request.quantity
        side_lower = side.lower()
        pnl = 0.0
        if side_lower == "buy":
            self._positions[symbol] = avg_price
            position_value = notional
        else:
            entry_price = self._positions.pop(symbol, avg_price)
            pnl = (avg_price - entry_price) * request.quantity
            position_value = 0.0
        self.risk_engine.on_fill(
            profile_name=self._risk_profile,
            symbol=symbol,
            side=side_lower,
            position_value=position_value,
            pnl=pnl,
        )
        _LOGGER.info(
            "Kontroler %s: wykonano %s %s qty=%s avg_price=%s pnl=%s",
            self.controller_name,
            side_lower,
            symbol,
            request.quantity,
            avg_price,
            pnl,
        )


__all__ = ["DailyTrendController"]
