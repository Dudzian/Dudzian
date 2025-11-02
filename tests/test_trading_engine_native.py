"""Testy natywnego silnika tradingowego w pakiecie bot_core."""

from __future__ import annotations

import logging
import sys
import tempfile
import types
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

from bot_core.trading.engine import (
    BacktestResult,
    EngineConfig,
    MultiSessionBacktestResult,
    RiskManagementService,
    TechnicalIndicators,
    TechnicalIndicatorsService,
    TradingEngine,
    TradingParameters,
    TradingSignalService,
    VectorizedBacktestEngine,
    StrategyTester,
    TradingStrategies,
)
from bot_core.data.backtest_library import BacktestDatasetLibrary


class TestNativeTradingEngine(unittest.TestCase):
    """Minimalny zestaw testów sanity-check dla modułu bot_core.trading."""

    def setUp(self) -> None:
        self.config = EngineConfig(log_level="ERROR")
        self.engine = TradingEngine(config=self.config)
        self.params = TradingParameters()
        self.data = self._make_data()

    def _make_data(self) -> pd.DataFrame:
        dates = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
        rng = np.random.default_rng(seed=1234)
        steps = rng.normal(0.0005, 0.02, size=len(dates))
        prices = 100.0 * (1 + steps).cumprod()
        noise = rng.normal(0, 0.3, size=len(dates))
        frame = pd.DataFrame(
            {
                "open": prices + noise,
                "high": prices + np.abs(noise) + rng.uniform(0, 1, len(dates)),
                "low": prices - np.abs(noise) - rng.uniform(0, 1, len(dates)),
                "close": prices,
                "volume": rng.integers(1_000, 5_000, len(dates)),
            },
            index=dates,
        )
        frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
        frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
        return frame

    def _make_constant_indicators(self, index: pd.Index) -> TechnicalIndicators:
        base = pd.Series(1.0, index=index)
        atr = pd.Series(1.0, index=index)
        return TechnicalIndicators(
            rsi=base,
            ema_fast=base,
            ema_slow=base,
            sma_trend=base,
            atr=atr,
            bollinger_upper=base,
            bollinger_lower=base,
            bollinger_middle=base,
            macd=base,
            macd_signal=base,
            stochastic_k=base,
            stochastic_d=base,
        )

    def _make_backtest_result(self, index: pd.Index) -> BacktestResult:
        equity = pd.Series(1.0, index=index)
        trades = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "position",
                "quantity",
                "pnl",
                "pnl_pct",
                "duration",
                "exit_reason",
                "commission",
            ]
        )
        returns = pd.Series(0.0, index=index, dtype=float)
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            daily_returns=returns,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=pd.Timedelta(0),
            win_rate=0.0,
            profit_factor=0.0,
            tail_ratio=0.0,
            var_95=0.0,
            expected_shortfall_95=0.0,
            total_trades=0,
            avg_trade_duration=pd.Timedelta(0),
            largest_win=0.0,
            largest_loss=0.0,
        )

    def test_indicator_pipeline(self) -> None:
        validator = TechnicalIndicatorsService(MagicMock(), self.config)
        indicators = validator.calculate_indicators(self.data, self.params)
        self.assertIsInstance(indicators, TechnicalIndicators)
        self.assertEqual(len(indicators.rsi), len(self.data))

    def test_signal_and_risk_services(self) -> None:
        calculator = TechnicalIndicatorsService(MagicMock(), self.config)
        generator = TradingSignalService(MagicMock())
        risk = RiskManagementService(MagicMock())

        indicators = calculator.calculate_indicators(self.data, self.params)
        signals = generator.generate_signals(indicators, self.params)
        managed = risk.apply_risk_management(self.data, signals, indicators, self.params)

        self.assertEqual(len(signals), len(self.data))
        self.assertIsInstance(managed, pd.DataFrame)
        self.assertEqual(len(managed), len(self.data))
        self.assertIn("direction", managed.columns)
        self.assertIn("size", managed.columns)
        self.assertTrue(set(managed["direction"].unique()).issubset({-1, 0, 1}))
        self.assertTrue((managed["size"] >= 0).all())
        if "exit_reason" in managed.columns:
            recorded = managed["exit_reason"].dropna()
            self.assertTrue(recorded.empty or recorded.str.strip().ne("").all())

    def test_risk_management_exit_reasons_propagate_to_trades(self) -> None:
        dates = pd.date_range("2022-01-01", periods=6, freq="D")
        close_prices = pd.Series([100.0, 99.0, 97.5, 96.0, 97.0, 98.0], index=dates)
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.5,
                "low": close_prices - 0.5,
                "close": close_prices,
                "volume": np.full(len(close_prices), 1_000.0),
            },
            index=dates,
        )

        signals = pd.Series([1, 1, 1, 1, 0, 0], index=dates, dtype=int)
        base = pd.Series(np.ones(len(dates)), index=dates)
        atr = pd.Series(np.ones(len(dates)), index=dates)
        indicators = TechnicalIndicators(
            rsi=base,
            ema_fast=base,
            ema_slow=base,
            sma_trend=base,
            atr=atr,
            bollinger_upper=base,
            bollinger_lower=base,
            bollinger_middle=base,
            macd=base,
            macd_signal=base,
            stochastic_k=base,
            stochastic_d=base,
        )

        params = TradingParameters(max_position_risk=0.05, position_size=1.0)
        risk = RiskManagementService(MagicMock())
        managed = risk.apply_risk_management(data, signals, indicators, params)

        self.assertIn("exit_reason", managed.columns)
        self.assertEqual(managed.loc[dates[2], "exit_reason"], "stop_loss")
        self.assertEqual(managed.loc[dates[4], "exit_reason"], "signal")
        non_na_reasons = managed["exit_reason"].dropna()
        self.assertEqual(len(non_na_reasons), 2)
        self.assertListEqual(list(non_na_reasons.index), [dates[2], dates[4]])

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, managed, params)

        self.assertEqual(len(trades), 2)
        self.assertListEqual(trades["exit_reason"].tolist(), ["stop_loss", "signal"])

    def test_strategy_tester_walk_forward_captures_pandas_warnings(self) -> None:
        tester = StrategyTester(self.engine)
        params = self.params
        data = self.data

        def _warning_run_strategy(test_data: pd.DataFrame, _params: TradingParameters, *args, **kwargs):
            warnings.warn("walk-forward drift", pd.errors.PerformanceWarning)
            return self._make_backtest_result(test_data.index)

        with (
            patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
            patch.object(
                self.engine, "run_strategy", side_effect=_warning_run_strategy
            ),
        ):
            result_frame = tester.walk_forward_analysis(
                data,
                params,
                train_ratio=0.6,
                step_ratio=0.25,
                min_train_periods=60,
            )

        self.assertIsInstance(result_frame, pd.DataFrame)
        observe_warning.assert_called()
        warned_kwargs = observe_warning.call_args.kwargs
        self.assertEqual(warned_kwargs["component"], "trading_engine.walk_forward")
        self.assertIn("walk-forward drift", warned_kwargs["message"])

    def test_strategy_tester_monte_carlo_captures_pandas_warnings(self) -> None:
        tester = StrategyTester(self.engine)
        params = self.params
        data = self.data

        def _warning_run_strategy(synthetic_data: pd.DataFrame, _params: TradingParameters, *args, **kwargs):
            warnings.warn("monte carlo drift", pd.errors.PerformanceWarning)
            return self._make_backtest_result(synthetic_data.index)

        with (
            patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
            patch.object(
                self.engine, "run_strategy", side_effect=_warning_run_strategy
            ),
        ):
            results = tester.monte_carlo_simulation(
                data,
                params,
                n_simulations=3,
                block_length=10,
            )

        self.assertIsInstance(results, dict)
        self.assertIn("returns", results)
        observe_warning.assert_called()
        warned_kwargs = observe_warning.call_args.kwargs
        self.assertEqual(warned_kwargs["component"], "trading_engine.monte_carlo")
        self.assertIn("monte carlo drift", warned_kwargs["message"])

    def test_trade_generation_handles_sparse_exit_metadata(self) -> None:
        dates = pd.date_range("2022-02-01", periods=6, freq="D")
        close_prices = pd.Series([100.0, 101.0, 102.0, 101.5, 100.5, 99.5], index=dates)
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.75,
                "low": close_prices - 0.75,
                "close": close_prices,
                "volume": np.full(len(close_prices), 1_000.0),
            },
            index=dates,
        )

        positions = pd.DataFrame(
            {
                "direction": [0, 1, 1, 0, -1, 0],
                "size": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                "exit_reason": pd.Series(
                    [pd.NA, pd.NA, pd.NA, "stop_loss", pd.NA, pd.NA],
                    dtype="string",
                    index=dates,
                ),
            },
            index=dates,
        )

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, positions, self.params)

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.loc[0, "exit_reason"], "stop_loss")
        self.assertEqual(trades.loc[1, "exit_reason"], "signal")

    def test_trade_generation_without_exit_metadata_defaults_to_signal(self) -> None:
        dates = pd.date_range("2022-03-01", periods=5, freq="D")
        close_prices = pd.Series([50.0, 51.0, 52.0, 51.0, 50.5], index=dates)
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.25,
                "low": close_prices - 0.25,
                "close": close_prices,
                "volume": np.full(len(close_prices), 500.0),
            },
            index=dates,
        )

        positions = pd.DataFrame(
            {
                "direction": [0, 1, 1, 0, 0],
                "size": [0.0, 1.0, 1.0, 0.0, 0.0],
            },
            index=dates,
        )

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, positions, self.params)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.loc[0, "exit_reason"], "signal")

    def test_trade_generation_normalizes_exit_reason_values(self) -> None:
        dates = pd.date_range("2022-04-01", periods=7, freq="D")
        close_prices = pd.Series(
            [200.0, 198.0, 197.5, 198.5, 199.0, 200.5, 201.0],
            index=dates,
        )
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.4,
                "low": close_prices - 0.4,
                "close": close_prices,
                "volume": np.full(len(close_prices), 750.0),
            },
            index=dates,
        )

        positions = pd.DataFrame(
            {
                "direction": [0, 1, 0, 1, 0, 1, 0],
                "size": [0.0, 1.5, 0.0, 2.0, 0.0, 1.0, 0.0],
                "exit_reason": [
                    None,
                    None,
                    "  Stop-Loss  ",
                    None,
                    "TAKE PROFIT",
                    None,
                    "Signal",
                ],
            },
            index=dates,
        )

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, positions, self.params)

        self.assertEqual(len(trades), 3)
        self.assertListEqual(
            trades["exit_reason"].tolist(),
            ["stop_loss", "take_profit", "signal"],
        )

    def test_trade_generation_handles_categorical_exit_reason_column(self) -> None:
        dates = pd.date_range("2022-06-01", periods=7, freq="D")
        prices = pd.Series(
            [75.0, 76.5, 77.0, 76.0, 75.5, 76.2, 76.8],
            index=dates,
        )
        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 0.2,
                "low": prices - 0.2,
                "close": prices,
                "volume": np.full(len(prices), 850.0),
            },
            index=dates,
        )

        exit_reason_data = pd.Categorical(
            [
                None,
                None,
                "StopLoss",
                None,
                "TAKEPROFIT",
                None,
                "Signal",
            ],
            categories=["StopLoss", "TAKEPROFIT", "Signal"],
        )

        positions = pd.DataFrame(
            {
                "direction": [0, 1, 0, 1, 0, 1, 0],
                "size": [0.0, 2.0, 0.0, 1.5, 0.0, 1.0, 0.0],
                "exit_reason": exit_reason_data,
            },
            index=dates,
        )

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, positions, self.params)

        self.assertEqual(len(trades), 3)
        self.assertListEqual(
            trades["exit_reason"].tolist(),
            ["stop_loss", "take_profit", "signal"],
        )

    def test_trade_generation_unknown_exit_reason_defaults_to_signal(self) -> None:
        dates = pd.date_range("2022-05-01", periods=5, freq="D")
        close_prices = pd.Series([150.0, 149.5, 149.0, 148.5, 148.0], index=dates)
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.3,
                "low": close_prices - 0.3,
                "close": close_prices,
                "volume": np.full(len(close_prices), 600.0),
            },
            index=dates,
        )

        positions = pd.DataFrame(
            {
                "direction": [0, 1, 1, 0, 0],
                "size": [0.0, 1.0, 1.0, 0.0, 0.0],
                "exit_reason": [pd.NA, "manual_exit", pd.NA, "unknown", pd.NA],
            },
            index=dates,
        )

        engine = VectorizedBacktestEngine(MagicMock())
        trades = engine._generate_trades_dataframe_vectorized(data, positions, self.params)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.loc[0, "exit_reason"], "signal")

    def test_full_backtest(self) -> None:
        result = self.engine.run_strategy(self.data, self.params)
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIsInstance(result.sharpe_ratio, float)

    def test_optimize_parameters_respects_max_iterations(self) -> None:
        mock_result = MagicMock()
        mock_result.configure_mock(sharpe_ratio=1.0)

        self.engine.run_strategy = MagicMock(return_value=mock_result)
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14, 20],
            "ema_fast_period": [5, 10],
            "ema_slow_period": [15, 25],
            "signal_threshold": [0.05, 0.1],
        }

        max_iterations = 3

        params, score = self.engine.optimize_parameters(
            self.data, param_ranges, max_iterations=max_iterations
        )

        self.assertIsNotNone(params)
        self.assertIsInstance(score, float)

        actual_iterations = self.engine.run_strategy.call_count
        self.assertLessEqual(actual_iterations, max_iterations)

        info_messages = [call.args[0] for call in self.engine._logger.info.call_args_list]
        self.assertTrue(
            any(
                f"Optimization completed after {actual_iterations} iterations" in message
                for message in info_messages
            ),
            "Final log message should report the actual iteration count.",
        )

    def test_optimize_parameters_counts_failed_runs(self) -> None:
        successful_result = MagicMock()
        successful_result.configure_mock(sharpe_ratio=2.0)

        self.engine.run_strategy = MagicMock(
            side_effect=[RuntimeError("boom"), successful_result]
        )
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14, 20],
            "ema_fast_period": [5, 10],
            "ema_slow_period": [15, 25],
            "signal_threshold": [0.05, 0.1],
        }

        max_iterations = 2

        params, score = self.engine.optimize_parameters(
            self.data, param_ranges, max_iterations=max_iterations
        )

        self.assertIsNotNone(params)
        self.assertIsInstance(score, float)

        self.assertEqual(self.engine.run_strategy.call_count, max_iterations)

        info_messages = [call.args[0] for call in self.engine._logger.info.call_args_list]
        self.assertTrue(
            any(
                f"Optimization completed after {max_iterations} iterations" in message
                for message in info_messages
            ),
            "Iteration log should include failed strategy attempts.",
        )

    def test_optimize_parameters_recovers_from_pre_execution_failures(self) -> None:
        successful_result = MagicMock()
        successful_result.configure_mock(sharpe_ratio=1.5)

        self.engine.run_strategy = MagicMock(return_value=successful_result)
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14, 20],
            "ema_fast_period": [5, 10],
            "ema_slow_period": [15, 25],
            "signal_threshold": [0.05, 0.1],
        }

        max_iterations = 2

        import bot_core.trading.engine as engine_module

        original_replace = engine_module.replace

        call_tracker = {"count": 0}

        def flaky_replace(obj, **changes):
            if call_tracker["count"] == 0:
                call_tracker["count"] += 1
                raise ValueError("synthetic failure")
            return original_replace(obj, **changes)

        with patch("bot_core.trading.engine.replace", side_effect=flaky_replace):
            params, score = self.engine.optimize_parameters(
                self.data, param_ranges, max_iterations=max_iterations
            )

        self.assertIsNotNone(params)
        self.assertIsInstance(score, float)
        self.assertEqual(self.engine.run_strategy.call_count, max_iterations)

        info_messages = [call.args[0] for call in self.engine._logger.info.call_args_list]
        self.assertTrue(
            any(
                f"Optimization completed after {max_iterations} iterations" in message
                for message in info_messages
            ),
            "Iteration log should still report the actual iteration count.",
        )

    def test_optimize_parameters_accepts_callable_objective(self) -> None:
        mock_result = MagicMock()
        mock_result.configure_mock(sharpe_ratio=0.0)

        evaluations: list[float] = []

        def objective_fn(result: MagicMock) -> float:
            score = 2.0 - 0.1 * len(evaluations)
            evaluations.append(score)
            return score

        self.engine.run_strategy = MagicMock(return_value=mock_result)
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14, 20],
            "ema_fast_period": [5, 10],
            "ema_slow_period": [15, 25],
            "signal_threshold": [0.05, 0.1],
        }

        max_iterations = 3

        params, score = self.engine.optimize_parameters(
            self.data,
            param_ranges,
            objective=objective_fn,
            max_iterations=max_iterations,
        )

        self.assertIsNotNone(params)
        self.assertAlmostEqual(score, evaluations[0])
        self.assertEqual(len(evaluations), self.engine.run_strategy.call_count)
        self.assertLessEqual(self.engine.run_strategy.call_count, max_iterations)

    def test_optimize_parameters_skips_nan_scores(self) -> None:
        successful_result = MagicMock()
        successful_result.configure_mock(sharpe_ratio=1.25)

        nan_result = MagicMock()
        nan_result.configure_mock(sharpe_ratio=float("nan"))

        self.engine.run_strategy = MagicMock(side_effect=[nan_result, successful_result])
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14, 20],
            "ema_fast_period": [5, 10],
            "ema_slow_period": [15, 25],
            "signal_threshold": [0.05, 0.1],
        }

        max_iterations = 2

        params, score = self.engine.optimize_parameters(
            self.data, param_ranges, max_iterations=max_iterations
        )

        self.assertIsNotNone(params)
        self.assertAlmostEqual(score, successful_result.sharpe_ratio)
        self.assertEqual(self.engine.run_strategy.call_count, max_iterations)

    def test_optimize_parameters_returns_baseline_when_no_valid_scores(self) -> None:
        nan_result = MagicMock()
        nan_result.configure_mock(sharpe_ratio=float("nan"))

        self.engine.run_strategy = MagicMock(return_value=nan_result)
        self.engine._logger = MagicMock()

        param_ranges = {
            "rsi_period": [10, 14],
            "ema_fast_period": [5],
            "ema_slow_period": [15],
            "signal_threshold": [0.05],
        }

        max_iterations = 2

        params, score = self.engine.optimize_parameters(
            self.data, param_ranges, max_iterations=max_iterations
        )

        self.assertEqual(params, TradingParameters())
        self.assertEqual(score, float("-inf"))
        self.assertEqual(self.engine.run_strategy.call_count, max_iterations)

        warning_messages = [call.args[0] for call in self.engine._logger.warning.call_args_list]
        self.assertTrue(
            any("without a valid score" in message for message in warning_messages),
            "Fallback warning should mention absence of valid optimization scores.",
        )

    def test_multi_session_backtest_matches_reference(self) -> None:
        dates = pd.date_range("2022-01-01", periods=6, freq="D")
        closes_a = pd.Series([100.0, 101.5, 102.0, 103.0, 104.5, 105.0], index=dates)
        closes_b = pd.Series([50.0, 49.5, 50.5, 51.0, 52.0, 53.5], index=dates)

        def _build_rows(series: pd.Series, symbol: str) -> pd.DataFrame:
            frame = pd.DataFrame({
                "timestamp": (series.index.view('int64') // 10**9).astype(int),
                "symbol": symbol,
                "open": series.values,
                "high": series.values,
                "low": series.values,
                "close": series.values,
                "volume": np.full(len(series), 1_000.0),
            })
            return frame

        weights = {"ASSET_A": 0.6, "ASSET_B": 0.4}
        returns_a = closes_a.pct_change().fillna(0.0)
        returns_b = closes_b.pct_change().fillna(0.0)
        total_return_a = float((1 + returns_a).cumprod().iloc[-1] - 1.0)
        total_return_b = float((1 + returns_b).cumprod().iloc[-1] - 1.0)
        combined_returns = pd.DataFrame({
            "ASSET_A": returns_a,
            "ASSET_B": returns_b,
        }).fillna(0.0)
        weighted_returns = combined_returns.mul(pd.Series(weights)).sum(axis=1)
        total_return_portfolio = float((1 + weighted_returns).cumprod().iloc[-1] - 1.0)

        manifest_payload = {
            "version": 1,
            "interval_units": {"d": 86400},
            "datasets": {
                "asset_a": {
                    "file": "asset_a.csv",
                    "interval": "1d",
                    "timezone": "UTC",
                    "strategies": ["constant"],
                    "risk_profiles": ["balanced"],
                    "schema": {
                        "timestamp": "int",
                        "symbol": "str",
                        "open": "float",
                        "high": "float",
                        "low": "float",
                        "close": "float",
                        "volume": "float",
                    },
                    "checks": {
                        "reference_results": {
                            "total_return": total_return_a,
                        }
                    },
                },
                "asset_b": {
                    "file": "asset_b.csv",
                    "interval": "1d",
                    "timezone": "UTC",
                    "strategies": ["constant"],
                    "risk_profiles": ["balanced"],
                    "schema": {
                        "timestamp": "int",
                        "symbol": "str",
                        "open": "float",
                        "high": "float",
                        "low": "float",
                        "close": "float",
                        "volume": "float",
                    },
                    "checks": {
                        "reference_results": {
                            "total_return": total_return_b,
                        }
                    },
                },
                "portfolio": {
                    "file": "asset_a.csv",
                    "interval": "1d",
                    "timezone": "UTC",
                    "strategies": ["portfolio"],
                    "risk_profiles": ["balanced"],
                    "schema": {
                        "timestamp": "int",
                        "symbol": "str",
                        "open": "float",
                        "high": "float",
                        "low": "float",
                        "close": "float",
                        "volume": "float",
                    },
                    "checks": {
                        "reference_results": {
                            "total_return": total_return_portfolio,
                            "weights": weights,
                        }
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            asset_a_path = tmp_path / "asset_a.csv"
            asset_b_path = tmp_path / "asset_b.csv"
            manifest_path = tmp_path / "manifest.yaml"

            _build_rows(closes_a, "ASSET_A").to_csv(asset_a_path, index=False)
            _build_rows(closes_b, "ASSET_B").to_csv(asset_b_path, index=False)
            manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding="utf-8")

            library = BacktestDatasetLibrary(manifest_path)
            reference_a = library.describe("asset_a").reference_results.get("total_return")
            reference_b = library.describe("asset_b").reference_results.get("total_return")
            reference_portfolio = library.describe("portfolio").reference_results.get("total_return")
            frame_a = library.load_dataframe(
                "asset_a", index_column="timestamp", datetime_columns={"timestamp": "s"}
            )
            frame_b = library.load_dataframe(
                "asset_b", index_column="timestamp", datetime_columns={"timestamp": "s"}
            )

            session_data = {
                "ASSET_A": frame_a[["open", "high", "low", "close", "volume"]],
                "ASSET_B": frame_b[["open", "high", "low", "close", "volume"]],
            }

            class _DummyIndicatorService:
                def __init__(self, logger, config):
                    self._logger = logger
                    self._config = config

                def calculate_indicators(self, data: pd.DataFrame, params: TradingParameters) -> TechnicalIndicators:
                    base = pd.Series(np.ones(len(data)), index=data.index)
                    atr = pd.Series(np.full(len(data), 1.0), index=data.index)
                    return TechnicalIndicators(
                        rsi=base,
                        ema_fast=base,
                        ema_slow=base,
                        sma_trend=base,
                        atr=atr,
                        bollinger_upper=base,
                        bollinger_lower=base,
                        bollinger_middle=base,
                        macd=base,
                        macd_signal=base,
                        stochastic_k=base,
                        stochastic_d=base,
                    )

            class _ConstantSignalGenerator:
                def __init__(self, logger):
                    self._logger = logger

                def generate_signals(self, indicators: TechnicalIndicators, params: TradingParameters) -> pd.Series:
                    return pd.Series(1, index=indicators.rsi.index, dtype=int)

            class _ConstantRiskManager:
                def __init__(self, logger):
                    self._logger = logger

                def apply_risk_management(
                    self,
                    data: pd.DataFrame,
                    signals: pd.Series,
                    indicators: TechnicalIndicators,
                    params: TradingParameters,
                ) -> pd.DataFrame:
                    return pd.DataFrame(
                        {
                            "direction": np.ones(len(signals), dtype=int),
                            "size": np.ones(len(signals), dtype=float),
                        },
                        index=signals.index,
                    )

            logger = MagicMock()
            config = EngineConfig(log_level="ERROR", min_data_points=1)
            engine = TradingEngine(
                config=config,
                indicator_calculator=_DummyIndicatorService(logger, config),
                signal_generator=_ConstantSignalGenerator(logger),
                risk_manager=_ConstantRiskManager(logger),
                logger=logger,
            )

            multi_result = engine.run_strategy(
                session_data,
                {"ASSET_A": TradingParameters(), "ASSET_B": TradingParameters()},
                initial_capital=10_000.0,
                fee_bps=0.0,
                session_weights=weights,
            )

        self.assertIsInstance(multi_result, MultiSessionBacktestResult)
        self.assertIsNotNone(reference_portfolio)
        self.assertAlmostEqual(
            multi_result.aggregate.total_return,
            reference_portfolio,
            places=6,
        )
        self.assertIsNotNone(reference_a)
        self.assertAlmostEqual(
            multi_result.sessions["ASSET_A"].total_return,
            reference_a,
            places=6,
        )
        self.assertIsNotNone(reference_b)
        self.assertAlmostEqual(
            multi_result.sessions["ASSET_B"].total_return,
            reference_b,
            places=6,
        )

    def test_multi_session_backtest_uses_backtest_engine_summary_contract(self) -> None:
        index = pd.date_range("2024-01-01", periods=2, freq="D")

        def _make_backtest_result(total_return: float) -> BacktestResult:
            daily_returns = pd.Series([0.0, total_return], index=index)
            equity_curve = (1 + daily_returns).cumprod() * 1000.0
            trades = pd.DataFrame(
                {
                    "entry_time": [index[0]],
                    "exit_time": [index[-1]],
                    "entry_price": [100.0],
                    "exit_price": [100.0 * (1 + total_return)],
                    "position": [1],
                    "quantity": [1.0],
                    "pnl": [1000.0 * total_return],
                    "pnl_pct": [total_return],
                    "duration": [index[-1] - index[0]],
                    "exit_reason": ["signal"],
                    "commission": [0.0],
                }
            )

            return BacktestResult(
                equity_curve=equity_curve,
                trades=trades,
                daily_returns=daily_returns,
                total_return=float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                omega_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=pd.Timedelta(0),
                win_rate=0.0,
                profit_factor=0.0,
                tail_ratio=0.0,
                var_95=0.0,
                expected_shortfall_95=0.0,
                total_trades=int(trades.shape[0]),
                avg_trade_duration=pd.Timedelta(0),
                largest_win=0.0,
                largest_loss=0.0,
            )

        class SummaryOnlyEngine:
            def __init__(self) -> None:
                self.calls: list[tuple[pd.Series, pd.DataFrame, pd.Series, float]] = []

            def summarize_backtest(
                self,
                equity_curve: pd.Series,
                trades: pd.DataFrame,
                returns: pd.Series,
                risk_free_rate: float,
            ) -> dict[str, object]:
                self.calls.append((equity_curve, trades, returns, risk_free_rate))
                total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
                return {
                    "total_return": total_return,
                    "annualized_return": total_return,
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
                    "total_trades": int(trades.shape[0]),
                    "avg_trade_duration": pd.Timedelta(0),
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                }

        engine = TradingEngine(config=self.config)
        summary_engine = SummaryOnlyEngine()
        engine._backtest_engine = summary_engine  # type: ignore[assignment]

        results = iter([
            _make_backtest_result(0.02),
            _make_backtest_result(0.01),
        ])
        engine._run_single_strategy = MagicMock(side_effect=lambda *_, **__: next(results))

        sessions = {
            "ASSET_A": self.data.iloc[: len(index)].copy(),
            "ASSET_B": self.data.iloc[: len(index)].copy(),
        }
        params_map = {symbol: self.params for symbol in sessions}

        portfolio = engine._run_multi_symbol_strategy(
            sessions,
            params_map,
            initial_capital=10000.0,
            fee_bps=5.0,
        )

        self.assertEqual(len(summary_engine.calls), 1)
        equity_curve, trades, returns, risk_free_rate = summary_engine.calls[0]
        self.assertTrue(equity_curve.equals(portfolio.aggregate.equity_curve))
        self.assertEqual(trades.shape[0], 2)
        self.assertEqual(risk_free_rate, self.config.risk_free_rate)
        self.assertIsInstance(portfolio, MultiSessionBacktestResult)
        self.assertEqual(portfolio.aggregate.total_trades, 2)
        self.assertEqual(set(portfolio.sessions.keys()), {"ASSET_A", "ASSET_B"})

    def test_trading_pipeline_records_pandas_warnings(self) -> None:
        index = self.data.index
        dummy_indicators = self._make_constant_indicators(index)
        dummy_positions = pd.DataFrame(
            {
                "direction": np.zeros(len(index), dtype=int),
                "size": np.zeros(len(index), dtype=float),
            },
            index=index,
        )
        dummy_result = self._make_backtest_result(index)

        def emit_warning(frame: pd.DataFrame) -> pd.DataFrame:
            warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
            return frame

        with (
            patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
            patch.object(self.engine._validator, "validate_ohlcv", side_effect=emit_warning),
            patch.object(
                self.engine._indicator_calculator,
                "calculate_indicators",
                return_value=dummy_indicators,
            ),
            patch.object(
                self.engine._signal_generator,
                "generate_signals",
                return_value=pd.Series(0, index=index, dtype=int),
            ),
            patch.object(self.engine._risk_manager, "apply_risk_management", return_value=dummy_positions),
            patch.object(self.engine._backtest_engine, "run_backtest", return_value=dummy_result),
        ):
            self.engine._logger.setLevel(logging.WARNING)
            with self.assertLogs(self.engine._logger.name, level="WARNING") as log_cm:
                result = self.engine._run_single_strategy(
                    self.data,
                    self.params,
                    initial_capital=10_000.0,
                    fee_bps=5.0,
                )

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(observe_warning.call_count, 1)
        _, kwargs = observe_warning.call_args
        self.assertEqual(kwargs["component"], "trading_engine.pipeline")
        self.assertEqual(kwargs["category"], "PerformanceWarning")
        self.assertEqual(kwargs["message"], "vectorized fallback")
        self.assertTrue(
            any(
                "Pandas warning captured in trading_engine.pipeline" in message
                for message in log_cm.output
            ),
            "Captured pandas warning should be logged with component context.",
        )

    def test_capture_pandas_warnings_filters_categories(self) -> None:
        import bot_core.trading.engine as engine_module

        logger = logging.getLogger("bot_core.trading.engine.test.capture")
        logger.setLevel(logging.WARNING)

        with patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning:
            with self.assertLogs(logger, level="WARNING") as log_cm:
                with engine_module._capture_pandas_warnings(
                    logger, component="unit.test.component"
                ):
                    warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
                    warnings.warn("pandas will change", FutureWarning)
                    warnings.warn("generic future", FutureWarning)

        self.assertEqual(observe_warning.call_count, 2)
        categories = {call.kwargs["category"] for call in observe_warning.call_args_list}
        self.assertIn("PerformanceWarning", categories)
        self.assertIn("FutureWarning", categories)
        self.assertTrue(
            any(
                "Pandas warning captured in unit.test.component" in message
                for message in log_cm.output
            ),
            "Relevant pandas warnings should emit a log entry.",
        )

    def test_signal_service_records_pandas_warnings(self) -> None:
        logger = logging.getLogger("bot_core.trading.engine.test.signals")
        logger.setLevel(logging.WARNING)

        index = self.data.index[:10]
        indicators = self._make_constant_indicators(index)
        params = TradingParameters(ensemble_weights={"dummy": 1.0})

        class DummyPlugin:
            def generate(self, *_: object, **__: object) -> pd.Series:
                warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
                return pd.Series(0.0, index=index, dtype=float)

        class DummyCatalog:
            def __init__(self) -> None:
                self._plugin = DummyPlugin()

            def create(self, name: str) -> DummyPlugin | None:
                return self._plugin if name == "dummy" else None

            def register(self, _: object) -> None:  # pragma: no cover - unused in test
                return None

        service = TradingSignalService(logger, catalog=DummyCatalog())

        with patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning:
            with self.assertLogs(logger, level="WARNING") as log_cm:
                signals = service.generate_signals(indicators, params)

        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(observe_warning.call_count, 1)
        kwargs = observe_warning.call_args.kwargs
        self.assertEqual(kwargs["component"], "trading_engine.signals")
        self.assertEqual(kwargs["category"], "PerformanceWarning")
        self.assertEqual(kwargs["message"], "vectorized fallback")
        self.assertTrue(
            any(
                "Pandas warning captured in trading_engine.signals" in message
                for message in log_cm.output
            ),
            "Signal service should emit warning log entries.",
        )

    def test_risk_manager_records_pandas_warnings(self) -> None:
        logger = logging.getLogger("bot_core.trading.engine.test.risk")
        logger.setLevel(logging.WARNING)

        index = self.data.index[:5]
        data = self.data.iloc[:5]
        signals = pd.Series([1, 1, 0, 0, 0], index=index, dtype=int)
        indicators = self._make_constant_indicators(index)

        service = RiskManagementService(logger)

        def warn_position_size(*_: object, **__: object) -> float:
            warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
            return 1.0

        with patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning:
            with patch.object(service, "_calculate_position_size", side_effect=warn_position_size):
                with self.assertLogs(logger, level="WARNING") as log_cm:
                    managed = service.apply_risk_management(data, signals, indicators, self.params)

        self.assertIsInstance(managed, pd.DataFrame)
        self.assertEqual(observe_warning.call_count, 1)
        kwargs = observe_warning.call_args.kwargs
        self.assertEqual(kwargs["component"], "trading_engine.risk_management")
        self.assertEqual(kwargs["category"], "PerformanceWarning")
        self.assertEqual(kwargs["message"], "vectorized fallback")
        self.assertTrue(
            any(
                "Pandas warning captured in trading_engine.risk_management" in message
                for message in log_cm.output
            ),
            "Risk manager should emit warning log entries.",
        )

    def test_multi_session_aggregation_records_pandas_warnings(self) -> None:
        engine = TradingEngine(config=self.config)
        engine._logger.setLevel(logging.WARNING)

        class SummaryOnlyEngine:
            def __init__(self) -> None:
                self.calls: list[tuple[pd.Series, pd.DataFrame, pd.Series, float]] = []

            def summarize_backtest(
                self,
                equity_curve: pd.Series,
                trades: pd.DataFrame,
                returns: pd.Series,
                risk_free_rate: float,
            ) -> dict[str, object]:
                self.calls.append((equity_curve, trades, returns, risk_free_rate))
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
                    "total_trades": int(trades.shape[0]),
                    "avg_trade_duration": pd.Timedelta(0),
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                }

        engine._backtest_engine = SummaryOnlyEngine()  # type: ignore[assignment]

        index = pd.date_range("2022-01-01", periods=5, freq="D")
        base_result = self._make_backtest_result(index)
        result_a = replace(
            base_result,
            daily_returns=pd.Series([0.01, 0.0, -0.005, 0.002, 0.003], index=index, dtype=float),
        )
        result_b = replace(
            base_result,
            daily_returns=pd.Series([0.0, 0.004, -0.002, 0.001, 0.0], index=index, dtype=float),
        )

        engine._run_single_strategy = MagicMock(  # type: ignore[assignment]
            side_effect=[result_a, result_b]
        )

        sessions = {
            "AAA": self.data.iloc[: len(index)].copy(),
            "BBB": self.data.iloc[: len(index)].copy(),
        }
        params_map = {symbol: self.params for symbol in sessions}

        original_sort_index = pd.DataFrame.sort_index

        def warn_sort(self: pd.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
            warnings.warn("portfolio realign", pd.errors.PerformanceWarning)
            return original_sort_index(self, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "sort_index", new=warn_sort),
            patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
            self.assertLogs(engine._logger, level="WARNING") as log_cm,
        ):
            result = engine._run_multi_symbol_strategy(
                sessions,
                params_map,
                initial_capital=10_000.0,
                fee_bps=5.0,
            )

        self.assertIsInstance(result, MultiSessionBacktestResult)
        self.assertEqual(observe_warning.call_count, 1)
        kwargs = observe_warning.call_args.kwargs
        self.assertEqual(kwargs["component"], "trading_engine.multi_session")
        self.assertEqual(kwargs["category"], "PerformanceWarning")
        self.assertEqual(kwargs["message"], "portfolio realign")
        self.assertTrue(
            any(
                "Pandas warning captured in trading_engine.multi_session" in message
                for message in log_cm.output
            ),
            "Multi-session aggregation should emit warning log entries.",
        )

    def test_trading_strategies_ai_bridge_records_pandas_warnings(self) -> None:
        shim = TradingStrategies(engine=self.engine, logger=self.engine._logger)
        self.engine._logger.setLevel(logging.WARNING)

        class DummyBridge:
            def __init__(self, *_: object, **__: object) -> None:
                return None

            def integrate(
                self,
                validated_data: pd.DataFrame,
                raw_signals: pd.Series,
            ) -> pd.Series:
                warnings.warn("legacy fusion drift", pd.errors.PerformanceWarning)
                return raw_signals

        bridges_pkg = types.ModuleType("bridges")
        bridge_module = types.ModuleType("bridges.ai_trading_bridge")
        bridge_module.AITradingBridge = DummyBridge
        bridges_pkg.ai_trading_bridge = bridge_module  # type: ignore[attr-defined]

        with (
            patch.dict(
                sys.modules,
                {
                    "bridges": bridges_pkg,
                    "bridges.ai_trading_bridge": bridge_module,
                },
            ),
            patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
            self.assertLogs(self.engine._logger, level="WARNING") as log_cm,
        ):
            metrics, trades, equity = shim.backtest(
                self.data,
                initial_capital=10_000.0,
                allow_short=True,
                ai_model=object(),
                ai_weight=0.5,
            )

        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIsInstance(equity, pd.Series)
        self.assertEqual(observe_warning.call_count, 1)
        kwargs = observe_warning.call_args.kwargs
        self.assertEqual(kwargs["component"], "trading_engine.pipeline")
        self.assertEqual(kwargs["category"], "PerformanceWarning")
        self.assertEqual(kwargs["message"], "legacy fusion drift")
        self.assertTrue(
            any(
                "Pandas warning captured in trading_engine.pipeline" in message
                for message in log_cm.output
            ),
            "Legacy shim should surface pandas warnings through pipeline observability.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
