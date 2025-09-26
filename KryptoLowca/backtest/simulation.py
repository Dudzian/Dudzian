"""Unified backtesting and paper-trading simulation utilities."""
from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from KryptoLowca.config_manager import ValidationError
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import (
    BaseStrategy,
    DataProvider,
    StrategyContext,
    StrategyMetadata,
    StrategySignal,
    registry as strategy_registry,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class BacktestFill:
    order_id: int
    side: str
    size: float
    price: float
    fee: float
    slippage: float
    timestamp: datetime
    partial: bool


@dataclass(slots=True)
class BacktestTrade:
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees_paid: float
    slippage_cost: float


@dataclass(slots=True)
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    hit_ratio_pct: float
    risk_of_ruin_pct: float
    fees_paid: float
    slippage_cost: float


@dataclass(slots=True)
class BacktestReport:
    trades: List[BacktestTrade] = field(default_factory=list)
    fills: List[BacktestFill] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[datetime] = field(default_factory=list)
    starting_balance: float = 0.0
    final_balance: float = 0.0
    metrics: PerformanceMetrics | None = None
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MatchingConfig:
    latency_bars: int = 1
    slippage_bps: float = 5.0
    fee_bps: float = 10.0
    liquidity_share: float = 0.5


class HistoricalDataProvider(DataProvider):
    """Adapter udostępniający dane historyczne strategiom w trakcie backtestu."""

    def __init__(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        if data.empty:
            raise ValidationError("Backtest wymaga danych historycznych")
        if "close" not in data.columns:
            raise ValidationError("Dane historyczne wymagają kolumny 'close'")
        if "volume" not in data.columns:
            data = data.copy()
            data["volume"] = 0.0
        self._data = data.sort_index()
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_ohlcv(
        self, symbol: str, timeframe: str, *, limit: int = 500
    ) -> Mapping[str, object]:
        limit = max(1, int(limit))
        window = self._data.tail(limit)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": window,
            "close": float(window["close"].iloc[-1]),
        }

    async def get_ticker(self, symbol: str) -> Mapping[str, object]:
        last_close = float(self._data["close"].iloc[-1])
        return {"symbol": symbol, "last": last_close}

    def iter_rows(self) -> Iterable[Tuple[datetime, Mapping[str, float]]]:
        for ts, row in self._data.iterrows():
            if isinstance(ts, datetime):
                timestamp = ts
            else:
                timestamp = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
            yield timestamp, {
                "open": float(row.get("open", row["close"])),
                "high": float(row.get("high", row["close"])),
                "low": float(row.get("low", row["close"])),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }

    def history_until(self, index: int) -> pd.DataFrame:
        if index < 0:
            return self._data.iloc[:0]
        return self._data.iloc[: index + 1]


class StrategyBacktestSession:
    """Zarządza cyklem życia instancji strategii w trybie backtestu."""

    def __init__(
        self,
        strategy_cls: type[BaseStrategy],
        context_template: Mapping[str, object],
        data_provider: HistoricalDataProvider,
    ) -> None:
        self._strategy = strategy_cls()
        self._context_template = dict(context_template)
        self._data_provider = data_provider
        self._loop = asyncio.new_event_loop()
        self._prepared = False

    def close(self) -> None:
        try:
            if self._prepared:
                self._run(self._strategy.shutdown())
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop=self._loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:  # pragma: no cover - defensywne
                pass
            finally:
                self._loop.close()

    def _run(self, coro: asyncio.Future | asyncio.Task) -> object:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop.is_running():
            result: Dict[str, object] = {}
            error: Dict[str, BaseException] = {}

            def _worker() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result["value"] = loop.run_until_complete(coro)
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except BaseException as exc:  # pragma: no cover - defensive
                    error["value"] = exc
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error["value"]
            return result.get("value")

        asyncio.set_event_loop(self._loop)
        try:
            return self._loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)

    def build_context(self, *, timestamp: datetime, portfolio_value: float, position: float) -> StrategyContext:
        metadata = self._context_template.get("metadata")
        if not isinstance(metadata, StrategyMetadata):
            metadata = StrategyMetadata(name="Backtest", description="auto")
        extra = dict(self._context_template.get("extra", {}))
        extra.setdefault("mode", "demo")
        extra["backtest"] = True
        return StrategyContext(
            symbol=str(self._context_template.get("symbol", "UNKNOWN")),
            timeframe=str(self._context_template.get("timeframe", "1m")),
            portfolio_value=float(portfolio_value),
            position=float(position),
            timestamp=timestamp,
            metadata=metadata,
            extra=extra,
        )

    def ensure_prepared(self, context: StrategyContext) -> None:
        if self._prepared:
            return
        self._run(self._strategy.prepare(context, self._data_provider))
        self._prepared = True

    def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, object],
    ) -> StrategySignal:
        self.ensure_prepared(context)
        return self._run(self._strategy.handle_market_data(context, market_payload))

    def notify_fill(self, context: StrategyContext, fill: Mapping[str, object]) -> None:
        if not self._prepared:
            return
        try:
            self._run(self._strategy.notify_fill(context, fill))
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Strategy notify_fill failed during backtest")


@dataclass(slots=True)
class _PendingOrder:
    order_id: int
    side: str
    size: float
    remaining: float
    submit_index: int
    timestamp: datetime
    stop_loss: float | None = None
    take_profit: float | None = None


class MatchingEngine:
    """Prosty silnik dopasowujący zlecenia z opóźnieniem, slippage i prowizją."""

    def __init__(self, config: MatchingConfig) -> None:
        self._cfg = config
        self._orders: List[_PendingOrder] = []
        self._next_id = 1

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
        if size <= 0:
            raise ValidationError("Wielkość zlecenia musi być dodatnia")
        order = _PendingOrder(
            order_id=self._next_id,
            side=side.lower(),
            size=size,
            remaining=size,
            submit_index=index,
            timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._orders.append(order)
        self._next_id += 1
        return order.order_id

    def process_bar(
        self,
        *,
        index: int,
        timestamp: datetime,
        bar: Mapping[str, float],
    ) -> List[BacktestFill]:
        fills: List[BacktestFill] = []
        completed: List[_PendingOrder] = []
        for order in list(self._orders):
            if index - order.submit_index < self._cfg.latency_bars:
                continue
            base_price = float(bar.get("open", bar.get("close", 0.0)))
            if base_price <= 0:
                continue
            volume = max(0.0, float(bar.get("volume", 0.0)))
            liquidity = volume * max(0.0, min(self._cfg.liquidity_share, 1.0))
            remaining = order.remaining
            fill_size = remaining if liquidity <= 0 else min(remaining, liquidity)
            if fill_size <= 0:
                continue
            slip = base_price * (self._cfg.slippage_bps / 10_000.0)
            if order.side == "buy":
                price = base_price + slip
                slippage = slip
            else:
                price = max(0.0, base_price - slip)
                slippage = -slip
            fee = price * fill_size * (self._cfg.fee_bps / 10_000.0)
            fills.append(
                BacktestFill(
                    order_id=order.order_id,
                    side=order.side,
                    size=fill_size,
                    price=price,
                    fee=fee,
                    slippage=slippage * fill_size,
                    timestamp=timestamp,
                    partial=fill_size < remaining,
                )
            )
            order.remaining -= fill_size
            if order.remaining <= 1e-12:
                completed.append(order)
        for order in completed:
            self._orders.remove(order)
        return fills


class BacktestEngine:
    """Orkiestruje wykonanie strategii na danych historycznych."""

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
        self._strategy_name = strategy_name
        self._symbol = symbol
        self._timeframe = timeframe
        self._initial_balance = float(initial_balance)
        self._allow_short = bool(allow_short)
        self._matching_config = matching
        self._matching_engine = MatchingEngine(matching)
        self._data_provider = HistoricalDataProvider(data, symbol, timeframe)
        strategy_cls = strategy_registry.get(strategy_name)
        metadata = getattr(strategy_cls, "metadata", StrategyMetadata(name=strategy_name, description=""))
        context_template = {
            "symbol": symbol,
            "timeframe": timeframe,
            "metadata": metadata,
            "extra": dict(context_extra or {}),
        }
        self._session = StrategyBacktestSession(
            strategy_cls,
            context_template,
            self._data_provider,
        )

    def run(self) -> BacktestReport:
        cash = float(self._initial_balance)
        position = 0.0
        avg_price = 0.0
        trades: List[BacktestTrade] = []
        fills: List[BacktestFill] = []
        equity_curve: List[float] = []
        equity_ts: List[datetime] = []
        warnings: List[str] = []

        context = self._session.build_context(
            timestamp=datetime.now(timezone.utc), portfolio_value=cash, position=0.0
        )
        self._session.ensure_prepared(context)

        last_entry_time: datetime | None = None
        last_entry_price = 0.0
        entry_equity = cash
        trade_volume = 0.0
        fees_paid = 0.0
        slippage_cost = 0.0
        returns: List[float] = []
        previous_equity = cash

        last_bar_payload: tuple[int, datetime, Mapping[str, float]] | None = None
        for idx, (timestamp, bar) in enumerate(self._data_provider.iter_rows()):
            last_bar_payload = (idx, timestamp, bar)
            bar_fills = self._matching_engine.process_bar(index=idx, timestamp=timestamp, bar=bar)
            if bar_fills:
                position_before = position
                for fill in bar_fills:
                    fills.append(fill)
                    fees_paid += fill.fee
                    slippage_cost += abs(fill.slippage)
                    direction = 1 if fill.side == "buy" else -1
                    cash -= direction * fill.price * fill.size
                    cash -= fill.fee if fill.side == "buy" else -fill.fee
                    position += direction * fill.size
                    trade_volume += abs(fill.size)
                    if position_before == 0.0 and position != 0.0 and last_entry_time is None:
                        last_entry_time = timestamp
                        last_entry_price = fill.price
                        entry_equity = cash + position * bar["close"]
                    position_before = position
                    if position:
                        avg_price = (
                            (avg_price * (position - direction * fill.size)) + fill.price * fill.size
                        ) / position
                    else:
                        avg_price = 0.0
                    fill_context = self._session.build_context(
                        timestamp=timestamp,
                        portfolio_value=cash + position * bar["close"],
                        position=position,
                    )
                    self._session.notify_fill(
                        fill_context,
                        {
                            "order_id": fill.order_id,
                            "price": fill.price,
                            "size": fill.size,
                            "side": fill.side,
                            "fee": fill.fee,
                            "timestamp": fill.timestamp,
                        },
                    )
                if position == 0.0 and last_entry_time is not None:
                    exit_price = float(bar.get("close", avg_price))
                    trade_pnl = (cash + position * exit_price) - entry_equity
                    trades.append(
                        BacktestTrade(
                            direction="LONG" if avg_price >= 0 else "SHORT",
                            entry_time=last_entry_time,
                            exit_time=timestamp,
                            entry_price=last_entry_price,
                            exit_price=exit_price,
                            quantity=trade_volume,
                            pnl=trade_pnl,
                            pnl_pct=(trade_pnl / self._initial_balance * 100.0)
                            if self._initial_balance
                            else 0.0,
                            fees_paid=fees_paid,
                            slippage_cost=slippage_cost,
                        )
                    )
                    last_entry_time = None
                    trade_volume = 0.0

            equity = cash + position * float(bar.get("close", 0.0))
            equity_curve.append(equity)
            equity_ts.append(timestamp)
            if previous_equity:
                returns.append((equity - previous_equity) / previous_equity)
            previous_equity = equity

            context = self._session.build_context(
                timestamp=timestamp,
                portfolio_value=equity,
                position=position,
            )
            history = self._data_provider.history_until(idx)
            market_payload = {
                "price": float(bar.get("close", 0.0)),
                "bar": dict(bar),
                "ohlcv": history,
            }
            signal = self._session.generate_signal(context, market_payload)
            if signal.action.upper() == "HOLD":
                continue
            if signal.action.upper() not in {"BUY", "SELL"}:
                logger.debug("Ignoruję nieznane działanie sygnału: %s", signal.action)
                continue
            side = signal.action.upper()
            size = self._determine_size(signal, context, market_payload)
            if size <= 0:
                continue
            if side == "SELL" and position <= 0 and not self._allow_short:
                continue
            self._matching_engine.submit_market_order(
                side=side,
                size=size,
                index=idx,
                timestamp=timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            if side == "BUY" and last_entry_time is None:
                last_entry_time = timestamp
                last_entry_price = float(bar.get("close", 0.0))
                entry_equity = equity
                trade_volume = 0.0

        if last_bar_payload is not None:
            final_idx, final_ts, final_bar = last_bar_payload
            final_fills = self._matching_engine.process_bar(
                index=final_idx + 1,
                timestamp=final_ts,
                bar=final_bar,
            )
            for fill in final_fills:
                fills.append(fill)
                fees_paid += fill.fee
                slippage_cost += abs(fill.slippage)
                direction = 1 if fill.side == "buy" else -1
                cash -= direction * fill.price * fill.size
                cash -= fill.fee if fill.side == "buy" else -fill.fee
                position += direction * fill.size
                trade_volume += abs(fill.size)
                if abs(position) < 1e-9:
                    position = 0.0
                fill_context = self._session.build_context(
                    timestamp=final_ts,
                    portfolio_value=cash + position * final_bar["close"],
                    position=position,
                )
                self._session.notify_fill(
                    fill_context,
                    {
                        "order_id": fill.order_id,
                        "price": fill.price,
                        "size": fill.size,
                        "side": fill.side,
                        "fee": fill.fee,
                        "timestamp": fill.timestamp,
                    },
                )
            if position == 0.0 and last_entry_time is not None:
                exit_price = float(final_bar.get("close", avg_price))
                trade_pnl = (cash + position * exit_price) - entry_equity
                trades.append(
                    BacktestTrade(
                        direction="LONG" if avg_price >= 0 else "SHORT",
                        entry_time=last_entry_time,
                        exit_time=final_ts,
                        entry_price=last_entry_price,
                        exit_price=exit_price,
                        quantity=trade_volume,
                        pnl=trade_pnl,
                        pnl_pct=(trade_pnl / self._initial_balance * 100.0)
                        if self._initial_balance
                        else 0.0,
                        fees_paid=fees_paid,
                        slippage_cost=slippage_cost,
                    )
                )
                last_entry_time = None

        if last_bar_payload is not None and last_entry_time is not None and position != 0.0:
            _, final_ts, final_bar = last_bar_payload
            exit_price = float(final_bar.get("close", avg_price))
            cash -= position * exit_price
            trade_pnl = (cash + position * exit_price) - entry_equity
            trades.append(
                BacktestTrade(
                    direction="LONG" if position > 0 else "SHORT",
                    entry_time=last_entry_time,
                    exit_time=final_ts,
                    entry_price=last_entry_price,
                    exit_price=exit_price,
                    quantity=trade_volume or abs(position),
                    pnl=trade_pnl,
                    pnl_pct=(trade_pnl / self._initial_balance * 100.0)
                    if self._initial_balance
                    else 0.0,
                    fees_paid=fees_paid,
                    slippage_cost=slippage_cost,
                )
            )
            position = 0.0
            last_entry_time = None

        self._session.close()
        metrics = self._compute_metrics(
            equity_curve,
            equity_ts,
            returns,
            fees_paid,
            slippage_cost,
            trades,
        )
        if not trades:
            warnings.append("Brak domkniętych transakcji w badanym okresie")
        final_balance = equity_curve[-1] if equity_curve else self._initial_balance
        return BacktestReport(
            trades=trades,
            fills=fills,
            equity_curve=equity_curve,
            equity_timestamps=equity_ts,
            starting_balance=self._initial_balance,
            final_balance=final_balance,
            metrics=metrics,
            warnings=warnings,
            parameters={
                "strategy": self._strategy_name,
                "symbol": self._symbol,
                "timeframe": self._timeframe,
                "initial_balance": self._initial_balance,
                "matching": {
                    "latency_bars": self._matching_config.latency_bars,
                    "slippage_bps": self._matching_config.slippage_bps,
                    "fee_bps": self._matching_config.fee_bps,
                    "liquidity_share": self._matching_config.liquidity_share,
                },
            },
        )

    def _determine_size(
        self,
        signal: StrategySignal,
        context: StrategyContext,
        market_payload: Mapping[str, object],
    ) -> float:
        if signal.size is not None:
            return float(signal.size)
        action = signal.action.upper()
        if action == "SELL" and context.position > 0:
            return abs(context.position)
        if action == "BUY" and context.position < 0 and self._allow_short:
            return abs(context.position)
        price = float(market_payload.get("price") or 0.0)
        if price <= 0:
            return 0.0
        extra = context.extra if isinstance(context.extra, dict) else {}
        risk_pct = extra.get("trade_risk_pct") if isinstance(extra, dict) else None
        trade_risk_pct = float(risk_pct) if isinstance(risk_pct, (int, float)) else 0.01
        stop_loss_pct = max(float(signal.stop_loss or 0.0), 0.0005)
        if stop_loss_pct == 0:
            stop_loss_pct = 0.01
        risk_capital = context.portfolio_value * trade_risk_pct
        qty_by_risk = risk_capital / (stop_loss_pct * price)
        max_notional_pct = float(extra.get("max_position_notional_pct", 1.0))
        max_leverage = float(extra.get("max_leverage", 1.0))
        qty_by_notional = (context.portfolio_value * max_notional_pct) / price
        qty_by_leverage = (context.portfolio_value * max_leverage) / price
        qty = min(qty_by_risk, qty_by_notional, qty_by_leverage)
        return max(0.0, qty)

    def _compute_metrics(
        self,
        equity_curve: Sequence[float],
        timestamps: Sequence[datetime],
        returns: Sequence[float],
        fees: float,
        slippage: float,
        trades: Sequence[BacktestTrade],
    ) -> PerformanceMetrics:
        if not equity_curve:
            return PerformanceMetrics(
                total_return_pct=0.0,
                cagr_pct=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                hit_ratio_pct=0.0,
                risk_of_ruin_pct=100.0,
                fees_paid=fees,
                slippage_cost=slippage,
            )
        start = equity_curve[0]
        end = equity_curve[-1]
        total_return_pct = ((end / start) - 1.0) * 100.0 if start else 0.0
        cagr_pct = 0.0
        if timestamps:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
            if duration > 0:
                years = duration / (365.25 * 24 * 3600)
                if years > 0:
                    cagr_pct = ((end / start) ** (1 / years) - 1.0) * 100.0
        max_drawdown_pct = self._max_drawdown(equity_curve)
        sharpe = self._sharpe_ratio(returns)
        hit_ratio = self._hit_ratio(trades)
        ruin = self._risk_of_ruin(trades)
        return PerformanceMetrics(
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            hit_ratio_pct=hit_ratio,
            risk_of_ruin_pct=ruin,
            fees_paid=fees,
            slippage_cost=slippage,
        )

    @staticmethod
    def _max_drawdown(equity: Sequence[float]) -> float:
        peak = -float("inf")
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            if peak <= 0:
                continue
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100.0

    @staticmethod
    def _sharpe_ratio(returns: Sequence[float]) -> float:
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        annual_factor = math.sqrt(365.25 * 24 * 60)
        return (mean / std) * annual_factor

    @staticmethod
    def _hit_ratio(trades: Sequence[BacktestTrade]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade.pnl > 0)
        return wins / len(trades) * 100.0

    @staticmethod
    def _risk_of_ruin(trades: Sequence[BacktestTrade]) -> float:
        if not trades:
            return 100.0
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [-t.pnl for t in trades if t.pnl < 0]
        if not wins or not losses:
            return 0.0
        win_rate = len(wins) / len(trades)
        loss_rate = 1.0 - win_rate
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            return 0.0
        edge = win_rate - loss_rate * (avg_loss / avg_win)
        if edge <= 0:
            return 100.0
        capital_units = 100
        ruin = ((loss_rate / win_rate) ** capital_units) * 100.0
        return max(0.0, min(100.0, ruin))


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
    "BacktestReport",
    "BacktestTrade",
    "BacktestFill",
    "PerformanceMetrics",
    "HistoricalDataProvider",
    "MatchingConfig",
    "MatchingEngine",
    "evaluate_strategy_backtest",
]
