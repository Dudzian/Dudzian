"""Minimalistyczny backtester do szybkiej walidacji strategii."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from KryptoLowca.config_manager import StrategyConfig, ValidationError


@dataclass(slots=True)
class BacktestTrade:
    direction: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees_paid: float
    slippage_cost: float


@dataclass(slots=True)
class BacktestReport:
    trades: List[BacktestTrade] = field(default_factory=list)
    starting_balance: float = 0.0
    final_balance: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    fees_paid: float = 0.0
    slippage_paid: float = 0.0
    reduce_only_triggers: int = 0
    violations: List[str] = field(default_factory=list)


class MiniBacktester:
    """Prosty silnik backtestu skupiony na limitch ryzyka i kosztach handlu."""

    def __init__(
        self,
        strategy: StrategyConfig,
        *,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = False,
    ) -> None:
        self.strategy = strategy.validate()
        self.fee_rate = float(max(0.0, fee_rate))
        self.slippage_rate = float(max(0.0, slippage_rate))
        self.allow_short = bool(allow_short)

    def run(
        self,
        data: pd.DataFrame,
        *,
        signal_column: str = "signal",
        price_column: str = "close",
        initial_balance: float = 10_000.0,
    ) -> BacktestReport:
        if data.empty:
            raise ValidationError("Backtest wymaga niepustego DataFrame z danymi OHLCV")
        if signal_column not in data.columns:
            raise ValidationError(f"Brak kolumny sygnału '{signal_column}' w danych")
        if price_column not in data.columns:
            raise ValidationError(f"Brak kolumny ceny '{price_column}' w danych")

        balance = float(initial_balance)
        starting_balance = balance
        peak_balance = balance
        max_drawdown = 0.0
        open_direction: Optional[int] = None
        entry_price = 0.0
        entry_idx = -1
        qty = 0.0
        fees_paid = 0.0
        slippage_paid = 0.0
        trades: List[BacktestTrade] = []
        reduce_only_until_idx = -1
        reduce_only_triggers = 0
        violations: List[str] = []

        # Oszacuj liczbę sekund na świecę
        timeframe_seconds = self._infer_timeframe_seconds(data.index)
        cooldown_bars = max(1, math.ceil(self.strategy.violation_cooldown_s / timeframe_seconds))

        for idx, (ts, row) in enumerate(data.iterrows()):
            price = float(row[price_column])
            if price <= 0.0:
                continue
            signal = float(row[signal_column])

            # Zamknięcie pozycji w zależności od sygnału
            if open_direction is not None:
                exit_now = False
                if open_direction > 0 and signal <= 0:
                    exit_now = True
                elif open_direction < 0 and (signal >= 0 or not self.allow_short):
                    exit_now = True

                if exit_now:
                    exit_price = self._apply_slippage(price, -open_direction)
                    pnl, trade_fees = self._close_trade(
                        entry_price,
                        exit_price,
                        qty,
                        open_direction,
                    )
                    balance += pnl
                    fees_paid += trade_fees
                    slippage_paid += abs(exit_price - price) * qty
                    peak_balance = max(peak_balance, balance)
                    drawdown = (peak_balance - balance) / peak_balance if peak_balance else 0.0
                    max_drawdown = max(max_drawdown, drawdown)
                    trades.append(
                        BacktestTrade(
                            direction="LONG" if open_direction > 0 else "SHORT",
                            entry_index=entry_idx,
                            exit_index=idx,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=qty,
                            pnl=pnl,
                            pnl_pct=(pnl / starting_balance * 100.0) if starting_balance else 0.0,
                            fees_paid=trade_fees,
                            slippage_cost=abs(exit_price - price) * qty,
                        )
                    )
                    open_direction = None
                    entry_idx = -1
                    qty = 0.0
                    continue

            # Otwarcie pozycji
            if open_direction is None:
                if idx <= reduce_only_until_idx:
                    continue
                desired_direction = 1 if signal > 0 else -1 if signal < 0 else 0
                if desired_direction == 0:
                    continue
                if desired_direction < 0 and not self.allow_short:
                    continue
                qty, violation_reason = self._determine_size(balance, price)
                if qty <= 0.0:
                    if violation_reason:
                        violations.append(violation_reason)
                        if self.strategy.reduce_only_after_violation:
                            reduce_only_until_idx = idx + cooldown_bars
                            reduce_only_triggers += 1
                    continue
                if violation_reason and self.strategy.reduce_only_after_violation:
                    reduce_only_until_idx = idx + cooldown_bars
                    reduce_only_triggers += 1
                    violations.append(violation_reason)
                entry_price = self._apply_slippage(price, desired_direction)
                fees = entry_price * qty * self.fee_rate
                balance -= fees
                fees_paid += fees
                slippage_paid += abs(entry_price - price) * qty
                open_direction = desired_direction
                entry_idx = idx

        win_trades = sum(1 for trade in trades if trade.pnl > 0)
        win_rate = (win_trades / len(trades) * 100.0) if trades else 0.0

        return BacktestReport(
            trades=trades,
            starting_balance=starting_balance,
            final_balance=balance,
            total_return_pct=((balance / starting_balance - 1.0) * 100.0) if starting_balance else 0.0,
            max_drawdown_pct=max_drawdown * 100.0,
            win_rate_pct=win_rate,
            fees_paid=fees_paid,
            slippage_paid=slippage_paid,
            reduce_only_triggers=reduce_only_triggers,
            violations=violations,
        )

    # ------------------------------------------------------------------
    def _determine_size(self, balance: float, price: float) -> tuple[float, Optional[str]]:
        risk_capital = balance * self.strategy.trade_risk_pct
        stop_loss_pct = max(self.strategy.default_sl, 1e-4)
        qty_risk = risk_capital / (stop_loss_pct * price)
        qty_notional = (balance * self.strategy.max_position_notional_pct) / price
        qty_leverage = (balance * self.strategy.max_leverage) / price
        qty = min(qty_risk, qty_notional, qty_leverage)
        violation = None
        if qty <= 0.0:
            violation = "Brak środków na otwarcie pozycji zgodnie z limitem"
        else:
            if qty == qty_notional and qty_notional < qty_risk:
                violation = "Zredukowano pozycję do limitu max_position_notional_pct"
            elif qty == qty_leverage and qty_leverage < qty_risk:
                violation = "Zredukowano pozycję do limitu dźwigni"
        return qty, violation

    def _apply_slippage(self, price: float, direction: int) -> float:
        if self.slippage_rate == 0.0:
            return price
        if direction > 0:
            return price * (1.0 + self.slippage_rate)
        return price * (1.0 - self.slippage_rate)

    def _close_trade(self, entry_price: float, exit_price: float, qty: float, direction: int) -> tuple[float, float]:
        gross = (exit_price - entry_price) * qty * direction
        fees = (abs(exit_price) + abs(entry_price)) * qty * self.fee_rate
        return gross - fees, fees

    @staticmethod
    def _infer_timeframe_seconds(index: pd.Index) -> int:
        if len(index) < 2:
            return 60
        try:
            delta = (index[1] - index[0]).total_seconds()  # type: ignore[attr-defined]
        except Exception:
            return 60
        return max(1, int(delta))


def evaluate_strategy_backtest(strategy: StrategyConfig, report: BacktestReport) -> None:
    """Egzekwuje minimalne progi jakości backtestu przed wdrożeniem strategii."""

    if not report.trades:
        raise ValidationError("Backtest nie zawiera żadnych transakcji – strategia wymaga korekty")
    if report.total_return_pct < 0.0:
        raise ValidationError("Backtest wykazuje ujemną stopę zwrotu – strategia odrzucona")
    max_allowed_drawdown = strategy.max_position_notional_pct * strategy.max_leverage * 100.0
    if report.max_drawdown_pct > max_allowed_drawdown:
        raise ValidationError(
            "Maksymalne obsunięcie przekracza akceptowalny próg dla ustawionych limitów ryzyka"
        )
