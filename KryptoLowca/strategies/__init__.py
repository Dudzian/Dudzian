"""High level trading strategy API for KryptoLowca."""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from .backtest import BacktestResult, Trade, VectorizedBacktestEngine
from .core import (
    BacktestEngine,
    BacktestExecutionError,
    ConfigurationError,
    DataValidationError,
    DataValidationService,
    DataValidator,
    EngineConfig,
    IndicatorCalculator,
    IndicatorComputationError,
    InsufficientDataError,
    MarketRegime,
    OrderType,
    PortfolioManager,
    RiskAnalyticsService,
    RiskLimitExceededError,
    RiskManagementService,
    RiskManager,
    SignalGenerator,
    SignalType,
    StrategyTester,
    TradingEngine,
    TradingEngineError,
    TradingEngineFactory,
    TradingParameters,
    TradingSignalService,
    TradingStrategies,
)
from .indicators import MathUtils, TechnicalIndicators, TechnicalIndicatorsService

__all__ = [
    "BacktestEngine",
    "BacktestExecutionError",
    "BacktestResult",
    "ConfigurationError",
    "DataValidationError",
    "DataValidationService",
    "DataValidator",
    "EngineConfig",
    "IndicatorCalculator",
    "IndicatorComputationError",
    "InsufficientDataError",
    "MarketRegime",
    "MathUtils",
    "OrderType",
    "PortfolioManager",
    "RiskAnalyticsService",
    "RiskLimitExceededError",
    "RiskManagementService",
    "RiskManager",
    "SignalGenerator",
    "SignalType",
    "StrategyTester",
    "TechnicalIndicators",
    "TechnicalIndicatorsService",
    "Trade",
    "TradingEngine",
    "TradingEngineError",
    "TradingEngineFactory",
    "TradingParameters",
    "TradingSignalService",
    "TradingStrategies",
    "VectorizedBacktestEngine",
]
