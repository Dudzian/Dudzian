"""Testy natywnego silnika tradingowego w pakiecie bot_core."""

from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
