"""Generic backtesting helpers niezależne od warstwy legacy."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import pandas as pd

from .simulation import BacktestFill, MatchingConfig, MatchingEngine

__all__ = [
    "BacktestError",
    "BacktestTrade",
    "PerformanceMetrics",
    "BacktestReport",
    "HistoricalDataProvider",
    "StrategySignalProtocol",
    "StrategyContextProtocol",
    "StrategyLike",
    "StrategyBacktestSession",
    "BacktestEngine",
]


logger = logging.getLogger(__name__)


class BacktestError(RuntimeError):
    """Ogólny wyjątek zgłaszany w przypadku błędów konfiguracji."""


@runtime_checkable
class DataProviderProtocol(Protocol):
    async def get_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> Mapping[str, Any]:
        ...

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:
        ...


class StrategySignalProtocol(Protocol):
    action: str
    size: float | None
    stop_loss: float | None
    take_profit: float | None


class StrategyContextProtocol(Protocol):
    symbol: str
    timeframe: str
    portfolio_value: float
    position: float
    timestamp: datetime
    metadata: Any
    extra: MutableMapping[str, Any]


ContextT = TypeVar("ContextT", bound=StrategyContextProtocol)
SignalT = TypeVar("SignalT", bound=StrategySignalProtocol)


class StrategyLike(Protocol[ContextT, SignalT]):
    async def prepare(self, context: ContextT, data_provider: DataProviderProtocol) -> None:
        ...

    async def handle_market_data(
        self, context: ContextT, market_payload: Mapping[str, Any]
    ) -> SignalT:
        ...

    async def notify_fill(self, context: ContextT, fill: Mapping[str, Any]) -> None:
        ...

    async def shutdown(self) -> None:
        ...


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _safe_timeframe_to_seconds(value: str) -> int | None:
    if not value:
        return None
    value = value.strip()
    unit = value[-1].lower()
    factor = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}.get(unit)
    if factor is None:
        return None
    try:
        amount = int(value[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    return amount * factor


def _normalize_required_data(value: Any) -> Tuple[str, ...] | None:
    if value is None:
        return None
    collected: List[str] = []
    seen: set[str] = set()

    def _consume(item: Any) -> None:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            collected.append(text)

    if isinstance(value, str):
        for segment in value.split(","):
            _consume(segment)
    elif isinstance(value, Mapping):
        for item in value.values():
            _consume(item)
    elif isinstance(value, Iterable):
        for item in value:
            _consume(item)
    else:
        _consume(value)

    return tuple(collected) if collected else None


def _normalize_strategy_metadata(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        result: Dict[str, Any] = {}
        risk_profile = metadata.get("risk_profile")
        if isinstance(risk_profile, str) and risk_profile.strip():
            result["risk_profile"] = risk_profile.strip()
        elif risk_profile not in (None, ""):
            result["risk_profile"] = str(risk_profile)

        normalized_required = _normalize_required_data(metadata.get("required_data"))
        if normalized_required:
            result["required_data"] = normalized_required

        for key, value in metadata.items():
            if key in {"risk_profile", "required_data"}:
                continue
            result[key] = value

        if not result:
            result["raw"] = dict(metadata)
        return result
    return {"raw": metadata}


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
    sortino_ratio: float
    omega_ratio: float
    hit_ratio_pct: float
    risk_of_ruin_pct: float
    max_exposure_pct: float
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
    parameters: Dict[str, Any] = field(default_factory=dict)
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)


class HistoricalDataProvider(DataProviderProtocol):
    """Adapter udostępniający dane historyczne strategiom."""

    def __init__(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        if data.empty:
            raise BacktestError("Backtest wymaga danych historycznych")
        if "close" not in data.columns:
            raise BacktestError("Dane historyczne wymagają kolumny 'close'")
        if "volume" not in data.columns:
            data = data.copy()
            data["volume"] = 0.0
        self._data = data.sort_index()
        self.symbol = symbol
        self.timeframe = timeframe
        self._history_cache_idx = -1
        self._history_cache: pd.DataFrame = self._data.iloc[:0]

    async def get_ohlcv(
        self, symbol: str, timeframe: str, *, limit: int = 500
    ) -> Mapping[str, Any]:
        limit = max(1, int(limit))
        window = self._data.tail(limit)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": window,
            "close": float(window["close"].iloc[-1]),
        }

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:
        last_close = float(self._data["close"].iloc[-1])
        return {"symbol": symbol, "last": last_close}

    def iter_rows(self) -> Iterable[Tuple[datetime, Mapping[str, float]]]:
        for ts, row in self._data.iterrows():
            timestamp = (
                ts
                if isinstance(ts, datetime)
                else datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
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
        if index == self._history_cache_idx:
            return self._history_cache
        self._history_cache = self._data.head(index + 1)
        self._history_cache_idx = index
        return self._history_cache

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._data


class StrategyBacktestSession:
    """Zarządza cyklem życia strategii podczas backtestu."""

    def __init__(
        self,
        strategy_factory: Callable[[], StrategyLike[ContextT, SignalT]],
        *,
        symbol: str,
        timeframe: str,
        metadata: Any,
        context_extra: Mapping[str, Any],
        context_builder: Callable[[Mapping[str, Any]], ContextT],
        data_provider: HistoricalDataProvider,
    ) -> None:
        self._strategy = strategy_factory()
        self._context_template = {
            "symbol": symbol,
            "timeframe": timeframe,
            "metadata": metadata,
            "extra": dict(context_extra),
        }
        self._context_builder = context_builder
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
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:  # pragma: no cover - defensywne
                pass
            finally:
                self._loop.close()

    def _run(self, coro: Awaitable[Any]) -> Any:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop.is_running():
            result: Dict[str, Any] = {}
            error: Dict[str, BaseException] = {}

            def _callback(task: asyncio.Future[Any]) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    error["value"] = exc
                else:
                    result["value"] = task.result()

            task = running_loop.create_task(coro)
            task.add_done_callback(_callback)
            task.add_done_callback(lambda _: self._loop.call_soon_threadsafe(self._loop.stop))
            self._loop.run_until_complete(asyncio.sleep(0))
            if "value" in error:
                raise error["value"]
            return result.get("value")

        return self._loop.run_until_complete(coro)

    def build_context(
        self, *, timestamp: datetime, portfolio_value: float, position: float
    ) -> ContextT:
        payload = dict(self._context_template)
        payload.update(
            {
                "portfolio_value": float(portfolio_value),
                "position": float(position),
                "timestamp": _ensure_timezone(timestamp),
            }
        )
        return self._context_builder(payload)

    def ensure_prepared(self, context: ContextT) -> None:
        if self._prepared:
            return
        self._run(self._strategy.prepare(context, self._data_provider))
        self._prepared = True

    def generate_signal(
        self,
        context: ContextT,
        market_payload: Mapping[str, Any],
    ) -> SignalT:
        self.ensure_prepared(context)
        return self._run(self._strategy.handle_market_data(context, market_payload))

    def notify_fill(self, context: ContextT, fill: Mapping[str, Any]) -> None:
        if not self._prepared:
            return
        try:
            self._run(self._strategy.notify_fill(context, fill))
        except Exception:  # pragma: no cover
            logger.exception("Strategy notify_fill failed during backtest")


class BacktestEngine:
    """Wykonuje strategię na danych historycznych."""

    def __init__(
        self,
        *,
        strategy_factory: Callable[[], StrategyLike[ContextT, SignalT]],
        context_builder: Callable[[Mapping[str, Any]], ContextT],
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        matching: MatchingConfig,
        allow_short: bool = False,
        context_extra: Mapping[str, Any] | None = None,
        metadata: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._strategy_factory = strategy_factory
        self._symbol = symbol
        self._timeframe = timeframe
        self._initial_balance = float(initial_balance)
        self._allow_short = bool(allow_short)
        self._matching_config = matching
        self._matching_engine = MatchingEngine(matching)
        self._data_provider = HistoricalDataProvider(data, symbol, timeframe)
        self._metadata = metadata
        self._strategy_metadata = _normalize_strategy_metadata(metadata)
        self._available_data = tuple(str(col) for col in self._data_provider.dataframe.columns)
        required = self._strategy_metadata.get("required_data")
        if isinstance(required, (list, tuple)):
            self._required_data = tuple(str(item) for item in required)
        else:
            self._required_data: Tuple[str, ...] = ()
        self._context_extra = dict(context_extra or {})
        self._session = StrategyBacktestSession(
            strategy_factory,
            symbol=symbol,
            timeframe=timeframe,
            metadata=metadata,
            context_extra=self._context_extra,
            context_builder=context_builder,
            data_provider=self._data_provider,
        )

    def run(self) -> BacktestReport:
        cash = float(self._initial_balance)
        position = 0.0
        trades: List[BacktestTrade] = []
        fills: List[BacktestFill] = []
        equity_curve: List[float] = []
        equity_ts: List[datetime] = []
        warnings: List[str] = []
        max_exposure_ratio = 0.0

        strategy_metadata = dict(self._strategy_metadata)
        strategy_metadata["available_data"] = self._available_data
        missing_required = tuple(
            item for item in self._required_data if item not in self._available_data
        )
        if missing_required:
            warnings.append(
                "Strategia wymaga danych, których nie znaleziono w zestawie historycznym: "
                + ", ".join(missing_required)
            )
            strategy_metadata["required_data_missing"] = missing_required

        context = self._session.build_context(
            timestamp=datetime.now(timezone.utc), portfolio_value=cash, position=0.0
        )
        self._session.ensure_prepared(context)

        total_fees = 0.0
        total_slippage = 0.0
        returns: List[float] = []
        previous_equity = cash

        df = self._data_provider.dataframe
        timestamps: List[datetime] = []
        for ts in df.index:
            if isinstance(ts, datetime):
                timestamps.append(ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
            else:
                timestamps.append(datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc))

        timeframe_s = _safe_timeframe_to_seconds(self._timeframe)
        gap_threshold = timeframe_s * 1.5 if timeframe_s else None

        invalid_volume_indices: List[int] = []
        for idx, (_, bar) in enumerate(self._data_provider.iter_rows()):
            volume = bar.get("volume")
            try:
                value = float(volume)
            except (TypeError, ValueError):
                invalid_volume_indices.append(idx)
                continue
            if not math.isfinite(value) or value < 0:
                invalid_volume_indices.append(idx)
        if invalid_volume_indices:
            sample_idx = invalid_volume_indices[:3]
            sample_ts = [timestamps[i].isoformat() for i in sample_idx if i < len(timestamps)]
            warnings.append(
                "Wykryto nieprawidłowe wartości wolumenu na świecach: "
                f"indeksy {sample_idx} (czas: {', '.join(sample_ts)})."
            )

        open_trade: Dict[str, Any] | None = None

        zero_volume_start_idx: int | None = None
        zero_volume_start_ts: datetime | None = None
        zero_volume_last_ts: datetime | None = None
        zero_volume_count = 0
        zero_volume_threshold = 3 if timeframe_s else 10

        def _finalize_zero_volume_warning() -> None:
            nonlocal zero_volume_start_idx, zero_volume_start_ts, zero_volume_last_ts, zero_volume_count
            if zero_volume_count >= zero_volume_threshold and zero_volume_start_idx is not None:
                duration_s: int | None = None
                if zero_volume_start_ts and zero_volume_last_ts:
                    duration_s = int((zero_volume_last_ts - zero_volume_start_ts).total_seconds())
                start_ts = zero_volume_start_ts.isoformat() if zero_volume_start_ts else "?"
                end_ts = zero_volume_last_ts.isoformat() if zero_volume_last_ts else "?"
                duration_fragment = (
                    f", łączny czas ok. {duration_s}s" if duration_s is not None and duration_s > 0 else ""
                )
                warnings.append(
                    "Wykryto długą sekwencję zerowego wolumenu: "
                    f"od indeksu {zero_volume_start_idx} ({start_ts}) do {end_ts} "
                    f"({zero_volume_count} świec{duration_fragment})."
                )
            zero_volume_start_idx = None
            zero_volume_start_ts = None
            zero_volume_last_ts = None
            zero_volume_count = 0

        def apply_fill(fill: BacktestFill, *, bar_close: float) -> None:
            nonlocal cash, position, total_fees, total_slippage, open_trade
            fee_paid = float(fill.fee)
            fills.append(fill)
            total_fees += fee_paid
            total_slippage += abs(fill.slippage)

            direction = 1 if fill.side == "buy" else -1
            position_before = position
            equity_before = cash + position_before * bar_close
            trade_notional = fill.price * fill.size
            cash -= direction * trade_notional
            cash -= fee_paid
            position = position_before + direction * fill.size
            if abs(position) < 1e-9:
                position = 0.0
            portfolio_value = cash + position * bar_close

            fill_context = self._session.build_context(
                timestamp=fill.timestamp,
                portfolio_value=portfolio_value,
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

            if open_trade is None and position_before == 0.0 and position != 0.0:
                open_trade = {
                    "direction": "LONG" if position > 0 else "SHORT",
                    "entry_time": fill.timestamp,
                    "entry_price": fill.price,
                    "entry_equity": equity_before,
                    "fees": 0.0,
                    "slippage": 0.0,
                    "volume": 0.0,
                    "position": abs(position),
                }

            if open_trade is not None:
                open_trade["fees"] = float(open_trade.get("fees", 0.0)) + fill.fee
                open_trade["slippage"] = float(open_trade.get("slippage", 0.0)) + abs(fill.slippage)
                open_trade["volume"] = float(open_trade.get("volume", 0.0)) + abs(fill.size)
                trade_dir = open_trade["direction"]
                if (
                    (trade_dir == "LONG" and direction == 1 and position > 0)
                    or (trade_dir == "SHORT" and direction == -1 and position < 0)
                ):
                    prev_pos = float(open_trade.get("position", 0.0))
                    new_pos = abs(position)
                    if new_pos > 0:
                        open_trade["entry_price"] = (
                            float(open_trade["entry_price"]) * prev_pos + fill.price * abs(fill.size)
                        ) / new_pos
                        open_trade["position"] = new_pos
                open_trade["position"] = abs(position)

            if open_trade is not None and position == 0.0:
                exit_price = fill.price
                pnl = portfolio_value - float(open_trade["entry_equity"])
                trades.append(
                    BacktestTrade(
                        direction=str(open_trade["direction"]),
                        entry_time=open_trade["entry_time"],
                        exit_time=fill.timestamp,
                        entry_price=float(open_trade["entry_price"]),
                        exit_price=exit_price,
                        quantity=float(open_trade["volume"]),
                        pnl=pnl,
                        pnl_pct=(pnl / self._initial_balance * 100.0)
                        if self._initial_balance
                        else 0.0,
                        fees_paid=float(open_trade["fees"]),
                        slippage_cost=float(open_trade["slippage"]),
                    )
                )
                open_trade = None

        last_bar_idx = -1
        last_bar: Mapping[str, float] | None = None
        last_ts: datetime | None = None

        for idx, (timestamp, bar) in enumerate(self._data_provider.iter_rows()):
            if gap_threshold is not None and last_ts is not None and timestamp > last_ts:
                delta_s = (timestamp - last_ts).total_seconds()
                if delta_s > gap_threshold and timeframe_s:
                    missing = max(1, int(round(delta_s / timeframe_s)) - 1)
                    warnings.append(
                        "Wykryto lukę danych: brak "
                        f"{missing} świec pomiędzy {last_ts.isoformat()} a {timestamp.isoformat()} "
                        f"(odstęp {int(delta_s)}s, timeframe {self._timeframe})."
                    )

            bar = {
                "open": float(bar.get("open", bar.get("close", 0.0))),
                "high": float(bar.get("high", bar.get("close", 0.0))),
                "low": float(bar.get("low", bar.get("close", 0.0))),
                "close": float(bar.get("close", 0.0)),
                "volume": float(bar.get("volume", 0.0)),
            }

            if bar["volume"] <= 0.0:
                if zero_volume_start_idx is None:
                    zero_volume_start_idx = idx
                    zero_volume_start_ts = timestamp
                zero_volume_last_ts = timestamp
                zero_volume_count += 1
            else:
                _finalize_zero_volume_warning()

            last_bar_idx = idx
            last_bar = bar
            last_ts = timestamp

            bar_fills = self._matching_engine.process_bar(index=idx, timestamp=timestamp, bar=bar)
            for fill in bar_fills:
                apply_fill(fill, bar_close=bar["close"])

            equity = cash + position * bar["close"]
            equity_curve.append(equity)
            equity_ts.append(timestamp)
            if previous_equity:
                returns.append((equity - previous_equity) / previous_equity)
            previous_equity = equity
            notional = abs(position * bar["close"])
            if notional > 0.0:
                if equity <= 0.0:
                    exposure_ratio = float("inf")
                else:
                    exposure_ratio = notional / equity
                if exposure_ratio > max_exposure_ratio:
                    max_exposure_ratio = exposure_ratio

            context = self._session.build_context(
                timestamp=timestamp, portfolio_value=equity, position=position
            )
            history = self._data_provider.history_until(idx)
            market_payload = {"price": bar["close"], "bar": bar, "ohlcv": history}
            signal = self._session.generate_signal(context, market_payload)
            action = signal.action.upper()
            if action == "HOLD":
                continue
            if action not in {"BUY", "SELL"}:
                self._logger.debug("Ignoruję nieznane działanie sygnału: %s", signal.action)
                continue
            side = action
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

        if last_bar is not None and last_ts is not None:
            final_fills = self._matching_engine.process_bar(
                index=last_bar_idx + 1, timestamp=last_ts, bar=last_bar
            )
            for fill in final_fills:
                apply_fill(fill, bar_close=last_bar["close"])

            if position != 0.0:
                forced_side = "sell" if position > 0 else "buy"
                forced_fill = self._matching_engine.force_fill(
                    side=forced_side,
                    size=abs(position),
                    timestamp=last_ts,
                    bar=last_bar,
                )
                apply_fill(forced_fill, bar_close=last_bar["close"])
                final_equity = cash + position * last_bar["close"]
                equity_curve.append(final_equity)
                equity_ts.append(last_ts)
                if previous_equity:
                    returns.append((final_equity - previous_equity) / previous_equity)
                previous_equity = final_equity
                final_notional = abs(position * last_bar["close"])
                if final_notional > 0.0:
                    if final_equity <= 0.0:
                        exposure_ratio = float("inf")
                    else:
                        exposure_ratio = final_notional / max(final_equity, 1e-9)
                    if exposure_ratio > max_exposure_ratio:
                        max_exposure_ratio = exposure_ratio

        _finalize_zero_volume_warning()

        self._session.close()
        metrics = self._compute_metrics(
            equity_curve,
            equity_ts,
            returns,
            total_fees,
            total_slippage,
            trades,
            max_exposure_ratio,
        )
        if not trades:
            warnings.append("Brak domkniętych transakcji w badanym okresie")
        final_balance = equity_curve[-1] if equity_curve else self._initial_balance
        parameters: Dict[str, Any] = {
            "strategy": self._strategy_factory.__name__
            if hasattr(self._strategy_factory, "__name__")
            else repr(self._strategy_factory),
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "initial_balance": self._initial_balance,
            "matching": {
                "latency_bars": self._matching_config.latency_bars,
                "slippage_bps": self._matching_config.slippage_bps,
                "fee_bps": self._matching_config.fee_bps,
                "liquidity_share": self._matching_config.liquidity_share,
            },
        }
        if metrics is not None:
            parameters["max_exposure_pct"] = metrics.max_exposure_pct
        if strategy_metadata:
            parameters["strategy_metadata"] = dict(strategy_metadata)
        return BacktestReport(
            trades=trades,
            fills=fills,
            equity_curve=equity_curve,
            equity_timestamps=equity_ts,
            starting_balance=self._initial_balance,
            final_balance=final_balance,
            metrics=metrics,
            warnings=warnings,
            parameters=parameters,
            strategy_metadata=strategy_metadata,
        )

    def _determine_size(
        self,
        signal: StrategySignalProtocol,
        context: StrategyContextProtocol,
        market_payload: Mapping[str, Any],
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
        return max(0.0, min(qty_by_risk, qty_by_notional, qty_by_leverage))

    def _compute_metrics(
        self,
        equity_curve: Sequence[float],
        equity_ts: Sequence[datetime],
        returns: Sequence[float],
        total_fees: float,
        total_slippage: float,
        trades: Sequence[BacktestTrade],
        max_exposure_ratio: float,
    ) -> PerformanceMetrics | None:
        if not equity_curve:
            return None
        starting_balance = equity_curve[0]
        final_balance = equity_curve[-1]
        total_return = ((final_balance / starting_balance) - 1) * 100 if starting_balance else 0.0
        if equity_ts and len(equity_ts) > 1:
            duration_days = (equity_ts[-1] - equity_ts[0]).days / 365.25
            duration_days = max(duration_days, 1 / 365.25)
        else:
            duration_days = 1 / 365.25
        cagr = (((final_balance / starting_balance) ** (1 / duration_days)) - 1) * 100 if starting_balance else 0.0
        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        sharpe = 0.0
        sortino = 0.0
        omega = 0.0
        if returns:
            periods = len(returns)
            avg_return = sum(returns) / periods
            variance = sum((r - avg_return) ** 2 for r in returns) / periods
            std_dev = math.sqrt(variance)
            if std_dev > 0:
                sharpe = (avg_return / std_dev) * (252 ** 0.5)
            downside = [r for r in returns if r < 0]
            if downside:
                downside_dev = math.sqrt(sum(r**2 for r in downside) / periods)
                if downside_dev > 0:
                    sortino = (avg_return / downside_dev) * (252 ** 0.5)
            elif avg_return > 0:
                sortino = float("inf")
            gains = [max(0.0, r) for r in returns]
            losses = [max(0.0, -r) for r in returns]
            gain_total = sum(gains)
            loss_total = sum(losses)
            if loss_total > 0:
                omega = gain_total / loss_total
            elif gain_total > 0:
                omega = float("inf")
        wins = sum(1 for trade in trades if trade.pnl > 0)
        hit_ratio = (wins / len(trades)) * 100 if trades else 0.0
        risk_of_ruin = max_drawdown * 100
        if math.isfinite(max_exposure_ratio):
            max_exposure_pct = max_exposure_ratio * 100
        else:
            max_exposure_pct = float("inf")
        return PerformanceMetrics(
            total_return_pct=total_return,
            cagr_pct=cagr,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            omega_ratio=omega,
            hit_ratio_pct=hit_ratio,
            risk_of_ruin_pct=risk_of_ruin,
            max_exposure_pct=max_exposure_pct,
            fees_paid=total_fees,
            slippage_cost=total_slippage,
        )
