"""Unit tests for the trading strategies engine module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from KryptoLowca.trading_strategies import (
    BacktestResult,
    DataValidationError,
    DataValidationService,
    EngineConfig,
    RiskManagementService,
    TechnicalIndicators,
    TechnicalIndicatorsService,
    TradingEngine,
    TradingEngineError,
    TradingParameters,
    TradingSignalService,
)


class TestTradingEngine(unittest.TestCase):
    """Comprehensive test suite for the trading engine runtime."""

    def setUp(self) -> None:
        self.config = EngineConfig(log_level="ERROR")
        self.engine = TradingEngine(config=self.config)
        self.sample_data = self._create_sample_data()
        self.params = TradingParameters()

    def _create_sample_data(self) -> pd.DataFrame:
        dates = pd.date_range(start="2020-01-01", end="2022-01-01", freq="D")
        np.random.seed(42)

        n_points = len(dates)
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n_points)

        prices = base_price * (1 + returns).cumprod()
        noise = np.random.normal(0, 0.5, n_points)

        data = pd.DataFrame(
            {
                "open": prices + noise,
                "close": prices,
                "high": prices + abs(noise) + np.random.exponential(0.5, n_points),
                "low": prices - abs(noise) - np.random.exponential(0.5, n_points),
                "volume": np.random.randint(1000, 10000, n_points),
            },
            index=dates,
        )

        data["high"] = np.maximum.reduce([data["open"], data["high"], data["low"], data["close"]])
        data["low"] = np.minimum.reduce([data["open"], data["high"], data["low"], data["close"]])
        return data

    def test_data_validation(self) -> None:
        validator = DataValidationService(MagicMock(), self.config)
        validated = validator.validate_ohlcv(self.sample_data)

        self.assertFalse(validated.empty)
        self.assertTrue(all(col in validated.columns for col in ["open", "high", "low", "close", "volume"]))
        self.assertIsInstance(validated.index, pd.DatetimeIndex)

    def test_indicator_calculation(self) -> None:
        calculator = TechnicalIndicatorsService(MagicMock(), self.config)
        indicators = calculator.calculate_indicators(self.sample_data, self.params)

        self.assertIsInstance(indicators, TechnicalIndicators)
        self.assertEqual(len(indicators.rsi), len(self.sample_data))
        self.assertTrue(all(0 <= rsi <= 100 for rsi in indicators.rsi.dropna()))

    def test_signal_generation(self) -> None:
        calculator = TechnicalIndicatorsService(MagicMock(), self.config)
        generator = TradingSignalService(MagicMock())

        indicators = calculator.calculate_indicators(self.sample_data, self.params)
        signals = generator.generate_signals(indicators, self.params)

        self.assertEqual(len(signals), len(self.sample_data))
        self.assertTrue(all(signal in [-1, 0, 1] for signal in signals))

    def test_risk_management(self) -> None:
        calculator = TechnicalIndicatorsService(MagicMock(), self.config)
        generator = TradingSignalService(MagicMock())
        risk_manager = RiskManagementService(MagicMock())

        indicators = calculator.calculate_indicators(self.sample_data, self.params)
        raw_signals = generator.generate_signals(indicators, self.params)
        managed_positions = risk_manager.apply_risk_management(
            self.sample_data, raw_signals, indicators, self.params
        )

        self.assertIsInstance(managed_positions, pd.DataFrame)
        self.assertEqual(len(managed_positions), len(raw_signals))
        self.assertIn("direction", managed_positions.columns)
        self.assertIn("size", managed_positions.columns)
        self.assertTrue(all(signal in [-1, 0, 1] for signal in managed_positions["direction"]))

    def test_backtest_execution(self) -> None:
        result = self.engine.run_strategy(self.sample_data, self.params)

        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.total_return, float)
        self.assertIsInstance(result.sharpe_ratio, float)
        self.assertGreaterEqual(len(result.equity_curve), len(self.sample_data))

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            TradingParameters(rsi_period=0)

        with self.assertRaises(ValueError):
            TradingParameters(ema_fast_period=26, ema_slow_period=12)

    def test_empty_data_handling(self) -> None:
        empty_data = pd.DataFrame()

        with self.assertRaises(TradingEngineError) as ctx:
            self.engine.run_strategy(empty_data, self.params)

        self.assertIsInstance(ctx.exception.__cause__, DataValidationError)

    def test_performance_metrics(self) -> None:
        result = self.engine.run_strategy(self.sample_data, self.params)

        self.assertIsInstance(result.sharpe_ratio, float)
        self.assertIsInstance(result.sortino_ratio, float)
        self.assertIsInstance(result.max_drawdown, float)
        self.assertLessEqual(result.max_drawdown, 0)
        self.assertGreaterEqual(result.win_rate, 0)
        self.assertLessEqual(result.win_rate, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
