"""Vectorized backtesting utilities for KryptoLowca."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .indicators import MathUtils
from .core import BacktestExecutionError

if TYPE_CHECKING:  # pragma: no cover
    from .core import EngineConfig, TradingParameters


@dataclass(frozen=True)
class Trade:
    """Individual trade record."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position: int
    quantity: float
    pnl: float
    pnl_pct: float
    duration: pd.Timedelta
    exit_reason: str
    commission: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Enhanced immutable backtest results."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    max_drawdown_duration: pd.Timedelta
    win_rate: float
    profit_factor: float
    tail_ratio: float
    var_95: float
    expected_shortfall_95: float
    total_trades: int
    avg_trade_duration: pd.Timedelta
    largest_win: float
    largest_loss: float


class VectorizedBacktestEngine:
    """Ultra-high-performance vectorized backtesting engine with comprehensive metrics."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._math = MathUtils()

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        params: "TradingParameters",
        config: "EngineConfig",
        initial_capital: float = 10000.0,
        fee_bps: float = 5.0,
    ) -> BacktestResult:
        try:
            aligned_data = data.reindex(signals.index, method="ffill")

            returns = self._calculate_returns_vectorized(aligned_data, signals, params, fee_bps)
            equity_curve = (1 + returns).cumprod() * initial_capital

            trades_df = self._generate_trades_dataframe_vectorized(aligned_data, signals, params)
            metrics = self._calculate_comprehensive_metrics(equity_curve, trades_df, returns, config.risk_free_rate)

            return BacktestResult(
                equity_curve=equity_curve,
                trades=trades_df,
                daily_returns=returns,
                **metrics,
            )
        except Exception as exc:  # pragma: no cover - delegated error handling
            raise BacktestExecutionError(f"Backtest execution failed: {exc}") from exc

    def _calculate_returns_vectorized(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        params: "TradingParameters",
        fee_bps: float,
    ) -> pd.Series:
        position = signals.shift(1).fillna(0)
        position = position * params.position_size

        price_returns = data["close"].pct_change().fillna(0)
        strategy_returns = position * price_returns

        position_changes = position.diff().abs()
        transaction_costs = position_changes * (fee_bps / 10000.0)

        return strategy_returns - transaction_costs

    def _generate_trades_dataframe_vectorized(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        params: "TradingParameters",
    ) -> pd.DataFrame:
        signal_changes = signals.diff().fillna(signals)
        trade_points = signal_changes != 0

        if not trade_points.any():
            return pd.DataFrame()

        trades: list[Dict[str, Any]] = []
        position = 0
        entry_idx: Optional[int] = None

        for i, (timestamp, signal) in enumerate(signals.items()):
            if timestamp not in data.index:
                continue

            current_price = data.loc[timestamp, "close"]

            if signal != position:
                if position != 0 and entry_idx is not None:
                    entry_timestamp = signals.index[entry_idx]
                    entry_price = data.loc[entry_timestamp, "close"]

                    pnl = (current_price - entry_price) * position
                    pnl_pct = pnl / entry_price
                    duration = timestamp - entry_timestamp

                    exit_reason = "signal" if signal == 0 else "reversal"

                    trades.append(
                        {
                            "entry_time": entry_timestamp,
                            "exit_time": timestamp,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": position,
                            "quantity": abs(position) * params.position_size,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "duration": duration,
                            "exit_reason": exit_reason,
                            "commission": abs(position)
                            * params.position_size
                            * current_price
                            * 0.0005,
                        }
                    )

                if signal != 0:
                    position = int(signal)
                    entry_idx = i
                else:
                    position = 0
                    entry_idx = None

        return pd.DataFrame(trades)

    def _calculate_comprehensive_metrics(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        if equity_curve.empty or len(equity_curve) < 2:
            return self._empty_comprehensive_metrics()

        initial_capital = equity_curve.iloc[0]
        final_capital = equity_curve.iloc[-1]

        total_return = (final_capital / initial_capital) - 1.0
        n_years = len(equity_curve) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

        volatility = returns.std() * np.sqrt(252)

        drawdown, max_drawdown, max_dd_duration = self._math.calculate_drawdown_vectorized(equity_curve)

        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = (
            (excess_returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() > 0
            else 0.0
        )

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (returns.mean() * 252) / downside_std if downside_std > 0 else 0.0

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        threshold = 0.0
        positive_returns = returns[returns > threshold]
        negative_returns = returns[returns <= threshold]
        omega_ratio = (
            positive_returns.sum() / abs(negative_returns.sum())
            if len(negative_returns) > 0
            else float("inf")
        )

        var_95 = float(returns.quantile(0.05))
        expected_shortfall_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else 0.0

        tail_ratio = (
            float(returns.quantile(0.95) / abs(returns.quantile(0.05)))
            if returns.quantile(0.05) != 0
            else 0.0
        )

        win_rate = self._calculate_win_rate(trades_df)
        profit_factor = self._calculate_profit_factor(trades_df)
        total_trades = len(trades_df)
        avg_trade_duration = trades_df["duration"].mean() if not trades_df.empty else pd.Timedelta(0)
        largest_win = trades_df["pnl"].max() if not trades_df.empty else 0.0
        largest_loss = trades_df["pnl"].min() if not trades_df.empty else 0.0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "omega_ratio": float(omega_ratio),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_duration": max_dd_duration,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "tail_ratio": float(tail_ratio),
            "var_95": var_95,
            "expected_shortfall_95": expected_shortfall_95,
            "total_trades": total_trades,
            "avg_trade_duration": avg_trade_duration,
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
        }

    def _empty_comprehensive_metrics(self) -> Dict[str, Any]:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": pd.Timedelta(0),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "tail_ratio": 0.0,
            "var_95": 0.0,
            "expected_shortfall_95": 0.0,
            "total_trades": 0,
            "avg_trade_duration": pd.Timedelta(0),
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        if trades_df.empty:
            return 0.0
        wins = trades_df[trades_df["pnl"] > 0]
        return len(wins) / len(trades_df)

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        if trades_df.empty:
            return 0.0
        gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
        return gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")


__all__ = [
    "BacktestResult",
    "Trade",
    "VectorizedBacktestEngine",
]
