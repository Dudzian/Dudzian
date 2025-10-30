"""Walk-forward backtester obsługujący wiele instrumentów i koszty transakcyjne."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.catalog import StrategyCatalog, StrategyDefinition


@dataclass(slots=True)
class TransactionCostModel:
    """Prosty model kosztów transakcyjnych bazujący na stałych stawkach bps."""

    fee_bps: float = 2.5
    slippage_bps: float = 5.0

    def estimate(self, price: float, quantity: float) -> tuple[float, float]:
        notional = abs(quantity) * max(price, 0.0)
        fee = notional * (self.fee_bps / 10_000)
        slippage = notional * (self.slippage_bps / 10_000)
        return fee, slippage


@dataclass(slots=True)
class WalkForwardSegment:
    """Zakres pojedynczego segmentu walk-forward."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(slots=True)
class TransactionCostBreakdown:
    symbol: str
    trades: int
    notional: float
    fees_paid: float
    slippage_paid: float


@dataclass(slots=True)
class SymbolSegmentReport:
    symbol: str
    segment_index: int
    train_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    start_equity: float
    end_equity: float
    return_pct: float
    max_drawdown_pct: float
    equity_curve: tuple[tuple[datetime, float], ...]
    costs: TransactionCostBreakdown


@dataclass(slots=True)
class TransactionCostSummary:
    total_fees: float
    total_slippage: float
    total_notional: float
    total_trades: int
    by_symbol: tuple[TransactionCostBreakdown, ...]


@dataclass(slots=True)
class WalkForwardReport:
    """Podsumowanie sesji walk-forward."""

    strategy: str
    engine: str
    definition: StrategyDefinition
    segments: tuple[SymbolSegmentReport, ...]
    total_return_pct: float
    cost_summary: TransactionCostSummary


class WalkForwardBacktester:
    """Uruchamia walk-forward backtest na wielu instrumentach."""

    def __init__(
        self,
        catalog: StrategyCatalog,
        *,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self._catalog = catalog
        self._cost_model = cost_model or TransactionCostModel()

    def run(
        self,
        definition: StrategyDefinition,
        dataset: Mapping[str, pd.DataFrame],
        segments: Sequence[WalkForwardSegment],
        *,
        initial_balance: float = 100_000.0,
    ) -> WalkForwardReport:
        if not segments:
            raise ValueError("Wymagany jest co najmniej jeden segment walk-forward")
        if not dataset:
            raise ValueError("Brak danych historycznych do backtestu")

        symbol_capital: MutableMapping[str, float] = {}
        per_symbol_costs: dict[str, TransactionCostBreakdown] = {}
        reports: list[SymbolSegmentReport] = []

        symbols = list(dataset.keys())
        initial_per_symbol = initial_balance / max(1, len(symbols))
        for symbol in symbols:
            symbol_capital[symbol] = initial_per_symbol
            per_symbol_costs[symbol] = TransactionCostBreakdown(
                symbol=symbol,
                trades=0,
                notional=0.0,
                fees_paid=0.0,
                slippage_paid=0.0,
            )

        for idx, segment in enumerate(segments):
            for symbol, frame in dataset.items():
                report = self._run_symbol_segment(
                    definition,
                    symbol,
                    frame,
                    segment,
                    capital=symbol_capital[symbol],
                )
                if report is None:
                    continue
                reports.append(SymbolSegmentReport(
                    symbol=symbol,
                    segment_index=idx,
                    train_range=(segment.train_start, segment.train_end),
                    test_range=(segment.test_start, segment.test_end),
                    start_equity=report["start_equity"],
                    end_equity=report["end_equity"],
                    return_pct=report["return_pct"],
                    max_drawdown_pct=report["max_drawdown_pct"],
                    equity_curve=tuple(report["equity_curve"]),
                    costs=report["costs"],
                ))
                symbol_capital[symbol] = report["end_equity"]
                costs = per_symbol_costs[symbol]
                costs.trades += report["costs"].trades
                costs.notional += report["costs"].notional
                costs.fees_paid += report["costs"].fees_paid
                costs.slippage_paid += report["costs"].slippage_paid

        final_total = sum(symbol_capital.values())
        initial_total = initial_per_symbol * len(symbols)
        total_return_pct = ((final_total / initial_total) - 1.0) * 100.0 if initial_total else 0.0

        cost_summary = TransactionCostSummary(
            total_fees=sum(item.fees_paid for item in per_symbol_costs.values()),
            total_slippage=sum(item.slippage_paid for item in per_symbol_costs.values()),
            total_notional=sum(item.notional for item in per_symbol_costs.values()),
            total_trades=sum(item.trades for item in per_symbol_costs.values()),
            by_symbol=tuple(per_symbol_costs[symbol] for symbol in symbols),
        )

        return WalkForwardReport(
            strategy=definition.name,
            engine=definition.engine,
            definition=definition,
            segments=tuple(reports),
            total_return_pct=total_return_pct,
            cost_summary=cost_summary,
        )

    def _run_symbol_segment(
        self,
        definition: StrategyDefinition,
        symbol: str,
        frame: pd.DataFrame,
        segment: WalkForwardSegment,
        *,
        capital: float,
    ) -> Mapping[str, Any] | None:
        train = frame.loc[segment.train_start:segment.train_end]
        test = frame.loc[segment.test_start:segment.test_end]
        if test.empty:
            return None

        engine = self._catalog.create(definition)
        train_snapshots = list(_frame_to_snapshots(train, symbol))
        if train_snapshots:
            engine.warm_up(train_snapshots)

        cash = float(capital)
        position_units = 0.0
        equity_curve: list[tuple[datetime, float]] = []
        trades = 0
        total_notional = 0.0
        total_fees = 0.0
        total_slippage = 0.0

        for snapshot in _frame_to_snapshots(test, symbol):
            price = snapshot.close
            if price <= 0:
                continue

            signals = engine.on_data(snapshot)
            equity = cash + position_units * price
            current_notional = position_units * price
            target_ratio = current_notional / equity if equity > 0 else 0.0

            for signal in signals:
                side = signal.side.lower()
                if side not in {"buy", "sell"}:
                    continue
                confidence = max(0.0, min(1.0, float(signal.confidence)))
                if side == "buy":
                    target_ratio = min(1.0, target_ratio + confidence)
                else:
                    target_ratio = max(0.0, target_ratio - confidence)

                desired_notional = target_ratio * equity
                trade_notional = desired_notional - current_notional
                if trade_notional > 0:
                    trade_notional = min(trade_notional, cash)
                if abs(trade_notional) < 1e-9:
                    continue

                trade_units = trade_notional / price
                fee, slippage = self._cost_model.estimate(price, trade_units)
                cash -= trade_notional
                cash -= fee + slippage
                position_units += trade_units
                current_notional = position_units * price
                equity = cash + current_notional
                trades += 1
                total_notional += abs(trade_notional)
                total_fees += fee
                total_slippage += slippage

            equity_curve.append((
                datetime.fromtimestamp(snapshot.timestamp / 1000.0),
                cash + position_units * price,
            ))

        if not equity_curve:
            return None

        equity_values = [value for _, value in equity_curve]
        start_equity = equity_values[0] if equity_values else capital
        end_equity = equity_values[-1] if equity_values else capital
        return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity else 0.0
        max_drawdown_pct = _max_drawdown(equity_values)

        cost_breakdown = TransactionCostBreakdown(
            symbol=symbol,
            trades=trades,
            notional=total_notional,
            fees_paid=total_fees,
            slippage_paid=total_slippage,
        )

        return {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "return_pct": return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "equity_curve": equity_curve,
            "costs": cost_breakdown,
        }


def _frame_to_snapshots(frame: pd.DataFrame, symbol: str) -> Iterable[MarketSnapshot]:
    if frame.empty:
        return
    base_columns = {"open", "high", "low", "close", "volume"}
    for ts, row in frame.iterrows():
        timestamp = pd.Timestamp(ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        epoch_ms = int(timestamp.timestamp() * 1000)
        price = float(row.get("close", 0.0))
        indicators: dict[str, float] = {}
        for column, value in row.items():
            if column in base_columns:
                continue
            try:
                indicators[str(column)] = float(value)
            except (TypeError, ValueError):
                continue
        yield MarketSnapshot(
            symbol=symbol,
            timestamp=epoch_ms,
            open=float(row.get("open", price)),
            high=float(row.get("high", price)),
            low=float(row.get("low", price)),
            close=price,
            volume=float(row.get("volume", 0.0)),
            indicators=indicators,
        )


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            continue
        drawdown = (value - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return abs(max_drawdown) * 100.0


__all__ = [
    "WalkForwardBacktester",
    "WalkForwardReport",
    "WalkForwardSegment",
    "TransactionCostModel",
    "TransactionCostSummary",
    "TransactionCostBreakdown",
    "SymbolSegmentReport",
]

