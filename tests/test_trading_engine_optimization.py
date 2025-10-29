"""Tests for trading engine optimisation summaries."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot_core.trading.engine import (
    BacktestResult,
    TradingEngine,
    TradingParameters,
)


def _build_backtest_result(score: float) -> BacktestResult:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    if isinstance(score, (float, np.floating)) and np.isfinite(score):
        equity_multiplier = 1.0 + score / 100.0
    else:
        equity_multiplier = 1.0

    equity_curve = pd.Series([100.0, 100.0 * equity_multiplier], index=index, name="equity")
    daily_returns = pd.Series([equity_multiplier - 1.0], index=index[1:], name="returns")
    trades = pd.DataFrame(
        {
            "entry_time": [index[0]],
            "exit_time": [index[1]],
            "entry_price": [100.0],
            "exit_price": [100.0 * equity_multiplier],
            "position": [1],
            "quantity": [1.0],
            "pnl": [100.0 * (equity_multiplier - 1.0)],
            "pnl_pct": [equity_multiplier - 1.0],
            "duration": [pd.Timedelta(days=1)],
            "exit_reason": ["signal"],
            "commission": [0.0],
        }
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        daily_returns=daily_returns,
        total_return=equity_multiplier - 1.0,
        annualized_return=equity_multiplier - 1.0,
        volatility=0.0,
        sharpe_ratio=score,
        sortino_ratio=score,
        calmar_ratio=0.0,
        omega_ratio=1.0,
        max_drawdown=0.0,
        max_drawdown_duration=pd.Timedelta(0),
        win_rate=1.0,
        profit_factor=1.0,
        tail_ratio=1.0,
        var_95=0.0,
        expected_shortfall_95=0.0,
        total_trades=1,
        avg_trade_duration=pd.Timedelta(days=1),
        largest_win=1.0,
        largest_loss=-1.0,
    )


class _DummyOptimizationEngine(TradingEngine):
    def __init__(self, scores: Dict[int, float], *, fail_calls: int = 0, raise_on_base: bool = False) -> None:
        # Bypass heavy initialisation from the production engine.
        self._logger = logging.getLogger("dummy-optimization-engine")
        self._logger.addHandler(logging.NullHandler())
        self._last_optimization_summary = None
        self._last_optimization_result = None
        self._scores = scores
        self._fail_calls = fail_calls
        self._call_count = 0
        self._raise_on_base = raise_on_base

    def run_strategy(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        params: TradingParameters,
        initial_capital: float = 10000.0,
        fee_bps: float = 5.0,
        session_weights: Optional[Dict[str, float]] = None,
    ) -> BacktestResult:
        del data, initial_capital, fee_bps, session_weights
        self._call_count += 1
        if self._raise_on_base and params.rsi_period == TradingParameters().rsi_period:
            raise RuntimeError("base parameters unavailable")
        if self._call_count <= self._fail_calls:
            score = float("nan")
        else:
            score = self._scores.get(params.rsi_period, 0.0)
        return _build_backtest_result(score)


def test_optimize_parameters_updates_summary_with_best_result():
    engine = _DummyOptimizationEngine({10: 3.0, 14: 1.5})
    data = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    best_params, best_score = engine.optimize_parameters(data, {"rsi_period": [10, 14]})

    summary = engine.get_last_optimization_summary()
    assert best_params.rsi_period == 10
    assert best_score == 3.0
    assert summary is not None
    assert summary.params == best_params
    assert summary.score == 3.0
    assert summary.iterations == 2
    assert summary.objective == "sharpe_ratio"
    assert summary.fallback_used is False
    assert summary.error is None
    assert engine.get_last_optimization_result() is summary.result


def test_optimize_parameters_records_fallback_summary():
    engine = _DummyOptimizationEngine({14: 0.5}, fail_calls=2)
    data = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    best_params, best_score = engine.optimize_parameters(data, {"rsi_period": [10, 14]})

    summary = engine.get_last_optimization_summary()
    assert best_params.rsi_period == TradingParameters().rsi_period
    assert best_score == 0.5
    assert summary is not None
    assert summary.params == best_params
    assert summary.score == 0.5
    assert summary.iterations == 2
    assert summary.fallback_used is True
    assert summary.error is None
    assert engine.get_last_optimization_result() is summary.result


def test_optimize_parameters_records_summary_when_search_space_empty():
    engine = _DummyOptimizationEngine({14: 1.0})
    data = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    best_params, best_score = engine.optimize_parameters(data, {"rsi_period": []}, max_iterations=0)

    summary = engine.get_last_optimization_summary()
    assert best_params.rsi_period == TradingParameters().rsi_period
    assert best_score == 1.0
    assert summary is not None
    assert summary.params == best_params
    assert summary.score == 1.0
    assert summary.iterations == 0
    assert summary.fallback_used is True
    assert summary.error is None
    assert engine.get_last_optimization_result() is summary.result


def test_optimize_parameters_preserves_summary_on_fallback_error():
    engine = _DummyOptimizationEngine({}, fail_calls=2, raise_on_base=True)
    data = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    try:
        engine.optimize_parameters(data, {"rsi_period": [10, 14]})
    except RuntimeError as error:
        assert "Unable to evaluate baseline parameters" in str(error)
    else:
        raise AssertionError("Expected RuntimeError from fallback")

    summary = engine.get_last_optimization_summary()
    assert summary is not None
    assert summary.params == TradingParameters()
    assert summary.score is None
    assert summary.fallback_used is True
    assert summary.error is not None
    assert "base parameters unavailable" in summary.error
    assert engine.get_last_optimization_result() is None
