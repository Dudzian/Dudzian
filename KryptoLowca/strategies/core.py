"""Core trading strategy components for KryptoLowca."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .backtest import BacktestEngine, BacktestResult
    from .indicators import TechnicalIndicators, TechnicalIndicatorsService

# =================== Enhanced Constants and Types ===================


class SignalType(Enum):
    """Trading signal types."""

    LONG = 1
    FLAT = 0
    SHORT = -1


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class OrderType(Enum):
    """Order execution types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for trading engine."""

    max_position_size: float = 1.0
    max_portfolio_risk: float = 0.02
    capital_fraction: float = 0.2
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_position_sizing: bool = True
    rebalance_frequency: str = "daily"
    risk_free_rate: float = 0.02
    max_drawdown_threshold: float = 0.15
    volatility_threshold: float = 0.30
    min_data_points: int = 252
    cache_indicators: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class TradingParameters:
    """Enhanced immutable trading parameters with validation."""

    # Technical indicators
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    sma_trend_period: int = 50
    bb_period: int = 20
    bb_std_mult: float = 2.0
    atr_period: int = 14
    macd_signal_period: int = 9
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # Signal generation
    signal_threshold: float = 0.1
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.3,
            "mean_reversion": 0.2,
            "momentum": 0.3,
            "volatility_breakout": 0.2,
        }
    )

    # Risk management
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    position_size: float = 1.0
    max_position_risk: float = 0.02
    max_position_size: int = 5

    # Position sizing
    volatility_target: float = 0.15
    kelly_fraction: float = 0.25

    def __post_init__(self) -> None:
        """Validate parameters after creation."""

        if self.rsi_period < 2 or self.rsi_period > 50:
            raise ValueError("RSI period must be between 2 and 50")
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("Fast EMA period must be less than slow EMA period")
        if not 0 < self.signal_threshold < 1:
            raise ValueError("Signal threshold must be between 0 and 1")
        if self.stop_loss_atr_mult < 0 or self.take_profit_atr_mult < 0:
            raise ValueError("ATR multipliers must be positive")
        if not abs(sum(self.ensemble_weights.values()) - 1.0) < 0.001:
            raise ValueError("Ensemble weights must sum to 1.0")
        if int(self.max_position_size) < 1:
            raise ValueError("max_position_size must be at least 1")


# =================== Enhanced Custom Exceptions ===================


class TradingEngineError(Exception):
    """Base exception for trading engine."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = pd.Timestamp.now()


class DataValidationError(TradingEngineError):
    """Data validation failed."""


class IndicatorComputationError(TradingEngineError):
    """Indicator computation failed."""


class BacktestExecutionError(TradingEngineError):
    """Backtest execution failed."""


class InsufficientDataError(TradingEngineError):
    """Insufficient data for analysis."""


class ConfigurationError(TradingEngineError):
    """Configuration error."""


class RiskLimitExceededError(TradingEngineError):
    """Risk limit exceeded."""


# =================== Enhanced Protocols (Interfaces) ===================


class DataValidator(Protocol):
    """Protocol for data validation."""

    def validate_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""


class IndicatorCalculator(Protocol):
    """Protocol for indicator calculations."""

    def calculate_indicators(
        self, data: pd.DataFrame, params: TradingParameters
    ) -> "TechnicalIndicators":
        """Calculate technical indicators."""


class SignalGenerator(Protocol):
    """Protocol for signal generation."""

    def generate_signals(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        """Generate trading signals."""


class RiskManager(Protocol):
    """Protocol for risk management."""

    def apply_risk_management(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        indicators: "TechnicalIndicators",
        params: TradingParameters,
    ) -> pd.Series:
        """Apply risk management rules."""


class BacktestEngine(Protocol):
    """Protocol for backtesting."""

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        params: TradingParameters,
        config: EngineConfig,
        initial_capital: float = 10000.0,
        fee_bps: float = 5.0,
    ) -> "BacktestResult":
        """Run backtest simulation."""


# =================== Enhanced Data Validation Service ===================


class DataValidationService:
    """Enhanced service for validating and cleaning trading data."""

    def __init__(self, logger: logging.Logger, config: EngineConfig):
        self._logger = logger
        self._config = config

    def validate_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data with comprehensive checks."""

        if data.empty:
            raise DataValidationError("DataFrame cannot be empty")

        if len(data) < self._config.min_data_points:
            raise InsufficientDataError(
                f"Insufficient data: {len(data)} rows, minimum {self._config.min_data_points} required"
            )

        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as exc:  # pragma: no cover - defensive
                raise DataValidationError(
                    f"Cannot convert index to DatetimeIndex: {exc}"
                ) from exc

        cleaned_data = data.copy()

        initial_length = len(cleaned_data)
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep="last")].sort_index()

        if len(cleaned_data) < initial_length:
            self._logger.warning(
                "Removed %d duplicate timestamps", initial_length - len(cleaned_data)
            )

        cleaned_data = self._fix_ohlc_relationships(cleaned_data)

        null_counts = cleaned_data[list(required_columns)].isnull().sum()
        if null_counts.any():
            self._logger.warning("Found null values: %s", null_counts.to_dict())
            cleaned_data[list(required_columns)] = cleaned_data[list(required_columns)].ffill().bfill()

        if (cleaned_data["volume"] < 0).any():
            self._logger.warning("Found negative volume values, taking absolute values")
            cleaned_data["volume"] = cleaned_data["volume"].abs()

        cleaned_data = self._detect_price_anomalies(cleaned_data)
        return cleaned_data

    def _fix_ohlc_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid OHLC relationships."""

        data = data.copy()
        max_price = data[["open", "low", "close"]].max(axis=1)
        min_price = data[["open", "high", "close"]].min(axis=1)

        invalid_high = data["high"] < max_price
        invalid_low = data["low"] > min_price

        if invalid_high.any() or invalid_low.any():
            total_invalid = invalid_high.sum() + invalid_low.sum()
            self._logger.warning("Correcting %d invalid OHLC relationships", total_invalid)

            data.loc[invalid_high, "high"] = max_price[invalid_high]
            data.loc[invalid_low, "low"] = min_price[invalid_low]

        return data

    def _detect_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle price anomalies using statistical methods."""

        data = data.copy()
        for col in ["open", "high", "low", "close"]:
            returns = data[col].pct_change()
            z_scores = np.abs((returns - returns.mean()) / returns.std())

            outliers = z_scores > 5
            if outliers.any():
                self._logger.warning(
                    "Found %d outliers in %s, applying smoothing", outliers.sum(), col
                )
                data.loc[outliers, col] = data[col].rolling(5, center=True).median()[outliers]

        return data


# =================== Enhanced Signal Generation Service ===================


class TradingSignalService:
    """Enhanced service for generating trading signals using ensemble methods."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def generate_signals(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        """Generate ensemble trading signals with improved logic."""

        signals_dict = {
            "trend": self._trend_following_signal(indicators, params),
            "mean_reversion": self._mean_reversion_signal(indicators, params),
            "momentum": self._momentum_signal(indicators, params),
            "volatility_breakout": self._volatility_breakout_signal(indicators, params),
        }

        ensemble_signal = pd.Series(0.0, index=indicators.rsi.index)
        for name, signal in signals_dict.items():
            weight = params.ensemble_weights.get(name, 0.0)
            ensemble_signal = ensemble_signal.add(signal * weight, fill_value=0.0)

        final_signals = pd.Series(
            SignalType.FLAT.value, index=ensemble_signal.index, dtype=int
        )
        final_signals[ensemble_signal > params.signal_threshold] = SignalType.LONG.value
        final_signals[ensemble_signal < -params.signal_threshold] = SignalType.SHORT.value

        return self._smooth_signals(final_signals)

    def _trend_following_signal(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        ema_diff = indicators.ema_fast - indicators.ema_slow
        ema_signal = pd.Series(np.where(ema_diff > 0, 1.0, -1.0), index=indicators.ema_fast.index)

        price_vs_sma = indicators.ema_fast - indicators.sma_trend
        trend_signal = pd.Series(np.where(price_vs_sma > 0, 1.0, -1.0), index=indicators.ema_fast.index)
        return ema_signal * 0.6 + trend_signal * 0.4

    def _mean_reversion_signal(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        rsi_signal = pd.Series(0.0, index=indicators.rsi.index)
        rsi_signal[indicators.rsi < params.rsi_oversold] = 1.0
        rsi_signal[indicators.rsi > params.rsi_overbought] = -1.0

        bb_signal = pd.Series(0.0, index=indicators.ema_fast.index)
        bb_signal[indicators.ema_fast < indicators.bollinger_lower] = 1.0
        bb_signal[indicators.ema_fast > indicators.bollinger_upper] = -1.0

        return rsi_signal * 0.5 + bb_signal * 0.5

    def _momentum_signal(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        macd_signal = pd.Series(
            np.where(indicators.macd > indicators.macd_signal, 1.0, -1.0),
            index=indicators.macd.index,
        )

        stoch_signal = pd.Series(0.0, index=indicators.stochastic_k.index)
        buy_condition = (indicators.stochastic_k > indicators.stochastic_d) & (indicators.stochastic_k < 80)
        sell_condition = (indicators.stochastic_k < indicators.stochastic_d) & (indicators.stochastic_k > 20)

        stoch_signal[buy_condition] = 1.0
        stoch_signal[sell_condition] = -1.0

        return macd_signal * 0.6 + stoch_signal * 0.4

    def _volatility_breakout_signal(
        self, indicators: "TechnicalIndicators", params: TradingParameters
    ) -> pd.Series:
        atr_pct = indicators.atr / indicators.ema_fast
        atr_threshold = atr_pct.rolling(20).quantile(0.8)
        high_vol = atr_pct > atr_threshold

        bb_width = (indicators.bollinger_upper - indicators.bollinger_lower) / indicators.bollinger_middle
        bb_expanding = bb_width > bb_width.rolling(20).mean()

        trend_up = indicators.ema_fast > indicators.ema_slow
        trend_down = indicators.ema_fast < indicators.ema_slow

        breakout_signal = pd.Series(0.0, index=indicators.atr.index)
        breakout_signal[high_vol & bb_expanding & trend_up] = 1.0
        breakout_signal[high_vol & bb_expanding & trend_down] = -1.0
        return breakout_signal

    def _smooth_signals(self, signals: pd.Series, window: int = 3) -> pd.Series:
        smoothed = signals.rolling(window=window, center=True).apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[len(x) // 2], raw=False
        )
        return smoothed.fillna(signals)


# =================== Enhanced Risk Management Service ===================


class RiskManagementService:
    """Comprehensive risk management service."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def apply_risk_management(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        indicators: "TechnicalIndicators",
        params: TradingParameters,
    ) -> pd.Series:
        if signals.empty:
            return signals

        managed_signals = signals.copy().astype(float)

        if params.position_size <= 0:
            return pd.Series(0, index=signals.index, dtype=float)

        current_position = 0
        entry_price = 0.0

        for idx, signal in signals.items():
            price = data.loc[idx, "close"] if idx in data.index else np.nan
            atr_value = indicators.atr.loc[idx] if idx in indicators.atr.index else np.nan

            if signal == SignalType.FLAT.value:
                if current_position != 0:
                    managed_signals.loc[idx] = SignalType.FLAT.value
                    current_position = 0
            elif signal == SignalType.LONG.value:
                if current_position <= 0:
                    current_position = SignalType.LONG.value
                    entry_price = price
                    managed_signals.loc[idx] = current_position
                else:
                    exit_reason = self._check_exit_conditions(current_position, entry_price, price, atr_value, params)
                    if exit_reason:
                        managed_signals.loc[idx] = SignalType.FLAT.value
                        current_position = 0
                    else:
                        managed_signals.loc[idx] = current_position
            elif signal == SignalType.SHORT.value:
                if current_position >= 0:
                    current_position = SignalType.SHORT.value
                    entry_price = price
                    managed_signals.loc[idx] = current_position
                else:
                    exit_reason = self._check_exit_conditions(current_position, entry_price, price, atr_value, params)
                    if exit_reason:
                        managed_signals.loc[idx] = SignalType.FLAT.value
                        current_position = 0
                    else:
                        managed_signals.loc[idx] = current_position
            else:
                managed_signals.loc[idx] = current_position

        return managed_signals

    def _check_exit_conditions(
        self,
        position: int,
        entry_price: float,
        current_price: float,
        atr: float,
        params: TradingParameters,
    ) -> Optional[str]:
        if pd.isna(atr) or atr == 0:
            return None

        if position > 0:
            stop_loss_price = entry_price - (atr * params.stop_loss_atr_mult)
            if current_price <= stop_loss_price:
                return "stop_loss"

            take_profit_price = entry_price + (atr * params.take_profit_atr_mult)
            if current_price >= take_profit_price:
                return "take_profit"

        elif position < 0:
            stop_loss_price = entry_price + (atr * params.stop_loss_atr_mult)
            if current_price >= stop_loss_price:
                return "stop_loss"

            take_profit_price = entry_price - (atr * params.take_profit_atr_mult)
            if current_price <= take_profit_price:
                return "take_profit"

        return None


# =================== Performance Monitoring ===================


class PerformanceMonitor:
    """Enhanced real-time performance monitoring and alerting."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._alerts: List[str] = []

    def monitor_drawdown(self, equity_curve: pd.Series, max_dd_threshold: float = 0.15) -> None:
        if len(equity_curve) < 2:
            return

        peak = equity_curve.expanding().max()
        current_dd = (equity_curve.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]

        if current_dd < -max_dd_threshold:
            alert = (
                f"WARNING: Current drawdown {current_dd:.2%} exceeds threshold {max_dd_threshold:.2%}. "
                f"Peak value: ${peak.iloc[-1]:,.2f}, Current value: ${equity_curve.iloc[-1]:,.2f}"
            )
            self._alerts.append(alert)
            self._logger.warning(alert)

    def monitor_volatility(self, equity_curve: pd.Series, vol_threshold: float = 0.30) -> None:
        if len(equity_curve) < 30:
            return

        returns = equity_curve.pct_change().dropna()
        recent_returns = returns.tail(30)
        current_vol = recent_returns.std() * np.sqrt(252)

        if current_vol > vol_threshold:
            avg_vol = returns.std() * np.sqrt(252)
            alert = (
                f"WARNING: Recent volatility {current_vol:.2%} exceeds threshold {vol_threshold:.2%}. "
                f"Average volatility: {avg_vol:.2%}"
            )
            self._alerts.append(alert)
            self._logger.warning(alert)

    def monitor_consecutive_losses(self, trades_df: pd.DataFrame, max_consecutive: int = 5) -> None:
        if trades_df.empty:
            return

        trades_df = trades_df.sort_values("exit_time")
        consecutive_losses = 0
        max_consecutive_losses = 0

        for _, trade in trades_df.iterrows():
            if trade["pnl"] < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        if max_consecutive_losses >= max_consecutive:
            alert = f"WARNING: {max_consecutive_losses} consecutive losing trades detected"
            self._alerts.append(alert)
            self._logger.warning(alert)

    def get_alerts(self) -> List[str]:
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        self._alerts.clear()


# =================== Trading Engine ===================


class TradingEngine:
    """Core orchestration class for running trading strategies."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        validator: Optional[DataValidator] = None,
        indicator_calculator: Optional[IndicatorCalculator] = None,
        signal_generator: Optional[SignalGenerator] = None,
        risk_manager: Optional[RiskManager] = None,
        backtest_engine: Optional[BacktestEngine] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._logger = logger or self._setup_logger()

        if indicator_calculator is None or backtest_engine is None:
            from .indicators import TechnicalIndicatorsService  # local import to avoid circular deps
            from .backtest import VectorizedBacktestEngine

            default_indicator = TechnicalIndicatorsService(self._logger, self._config)
            default_backtester = VectorizedBacktestEngine(self._logger)
        else:
            default_indicator = None
            default_backtester = None

        self._validator = validator or DataValidationService(self._logger, self._config)
        self._indicator_calculator = indicator_calculator or default_indicator  # type: ignore[assignment]
        self._signal_generator = signal_generator or TradingSignalService(self._logger)
        self._risk_manager = risk_manager or RiskManagementService(self._logger)
        self._backtest_engine = backtest_engine or default_backtester  # type: ignore[assignment]
        self._performance_monitor = PerformanceMonitor(self._logger)

    def run_strategy(
        self,
        data: pd.DataFrame,
        params: TradingParameters,
        initial_capital: float = 10000.0,
        fee_bps: float = 5.0,
    ) -> "BacktestResult":
        try:
            self._logger.info("Starting enhanced trading strategy execution")
            start_time = pd.Timestamp.now()

            validated_data = self._validator.validate_ohlcv(data)
            self._logger.info("Validated %d data points", len(validated_data))

            indicators = self._indicator_calculator.calculate_indicators(validated_data, params)
            self._logger.info("Calculated technical indicators")

            raw_signals = self._signal_generator.generate_signals(indicators, params)
            self._logger.info("Generated %d raw trading signals", int((raw_signals != 0).sum()))

            managed_signals = self._risk_manager.apply_risk_management(
                validated_data, raw_signals, indicators, params
            )
            risk_filtered = int((managed_signals != raw_signals).sum())
            self._logger.info("Risk management filtered %d signals", risk_filtered)

            result = self._backtest_engine.run_backtest(
                validated_data, managed_signals, params, self._config, initial_capital, fee_bps
            )

            self._performance_monitor.monitor_drawdown(result.equity_curve, self._config.max_drawdown_threshold)
            self._performance_monitor.monitor_volatility(result.equity_curve, self._config.volatility_threshold)

            execution_time = pd.Timestamp.now() - start_time
            self._logger.info(
                "Strategy completed in %.2fs: %s total return, %.2f Sharpe ratio",
                execution_time.total_seconds(),
                f"{result.total_return:.2%}",
                result.sharpe_ratio,
            )

            return result
        except Exception as exc:
            self._logger.error("Strategy execution failed: %s", exc)
            raise TradingEngineError(f"Strategy execution failed: {exc}") from exc

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_ranges: Dict[str, Iterable],
        objective: str = "sharpe_ratio",
        max_iterations: int = 1000,
    ) -> Tuple[TradingParameters, float]:
        best_params: Optional[TradingParameters] = None
        best_score = float("-inf")
        iterations = 0

        self._logger.info("Starting parameter optimization with objective: %s", objective)

        for rsi_period in param_ranges.get("rsi_period", [14]):
            for ema_fast in param_ranges.get("ema_fast_period", [12]):
                for ema_slow in param_ranges.get("ema_slow_period", [26]):
                    for signal_threshold in param_ranges.get("signal_threshold", [0.1]):
                        if iterations >= max_iterations:
                            break

                        if ema_fast >= ema_slow:
                            continue

                        try:
                            params = TradingParameters(
                                rsi_period=rsi_period,
                                ema_fast_period=ema_fast,
                                ema_slow_period=ema_slow,
                                signal_threshold=signal_threshold,
                            )

                            result = self.run_strategy(data, params)
                            score = getattr(result, objective)

                            if score > best_score:
                                best_score = score
                                best_params = params
                                self._logger.info("New best %s: %.4f", objective, score)

                            iterations += 1
                        except Exception as exc:
                            self._logger.warning("Optimization failed for params %s: %s", params, exc)

        if best_params is None:
            raise TradingEngineError("Parameter optimization failed to find valid parameters")

        self._logger.info("Optimization completed after %d iterations", iterations)
        return best_params, best_score

    def get_performance_alerts(self) -> List[str]:
        return self._performance_monitor.get_alerts()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self._config.log_level.upper()))
        return logger


# =================== Portfolio and Risk Analytics ===================


class PortfolioManager:
    """Enhanced portfolio management with advanced risk controls."""

    def __init__(self, engine: TradingEngine):
        self._engine = engine
        self._logger = engine._logger

    def run_portfolio_backtest(
        self,
        assets_data: Dict[str, pd.DataFrame],
        params: Dict[str, TradingParameters],
        weights: Dict[str, float],
        rebalance_freq: str = "M",
        total_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        if abs(sum(weights.values()) - 1.0) > 0.001:
            raise ValueError("Portfolio weights must sum to 1.0")

        results: Dict[str, Any] = {}
        equity_curves: Dict[str, pd.Series] = {}

        for asset, data in assets_data.items():
            if asset not in weights:
                continue

            try:
                asset_params = params.get(asset, TradingParameters())
                capital_allocation = total_capital * weights[asset]
                result = self._engine.run_strategy(data, asset_params, capital_allocation)
                results[asset] = result
                equity_curves[asset] = result.equity_curve
                self._logger.info(
                    "Completed backtest for %s: %s return",
                    asset,
                    f"{result.total_return:.2%}",
                )
            except Exception as exc:
                self._logger.error("Failed to backtest %s: %s", asset, exc)
                results[asset] = None

        portfolio_metrics = self._calculate_portfolio_metrics(equity_curves, weights, total_capital)
        return {
            "individual_results": results,
            "portfolio_metrics": portfolio_metrics,
            "portfolio_equity_curve": portfolio_metrics.get("equity_curve"),
            "correlation_matrix": self._calculate_correlation_matrix(equity_curves),
        }

    def _calculate_portfolio_metrics(
        self,
        equity_curves: Dict[str, pd.Series],
        weights: Dict[str, float],
        total_capital: float,
    ) -> Dict[str, Any]:
        if not equity_curves:
            return {}

        aligned_curves = pd.DataFrame(equity_curves).ffill().bfill()
        portfolio_equity = pd.Series(0.0, index=aligned_curves.index)
        for asset, curve in aligned_curves.items():
            if asset in weights:
                portfolio_equity = portfolio_equity.add(curve * (weights[asset] / curve.iloc[0]), fill_value=0.0)

        portfolio_equity = portfolio_equity * total_capital / portfolio_equity.iloc[0]
        portfolio_returns = portfolio_equity.pct_change().dropna()

        total_return = (portfolio_equity.iloc[-1] / total_capital) - 1.0
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
            if portfolio_returns.std() > 0
            else 0.0
        )

        peak = portfolio_equity.expanding().max()
        drawdown = (portfolio_equity - peak) / peak
        max_drawdown = drawdown.min()

        annualized_return = (1 + total_return) ** (252 / len(portfolio_equity)) - 1.0
        calmar_ratio = (
            annualized_return / abs(max_drawdown)
            if max_drawdown < 0
            else 0.0
        )

        return {
            "equity_curve": portfolio_equity,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_correlation_matrix(self, equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
        if len(equity_curves) < 2:
            return pd.DataFrame()

        returns_df = pd.DataFrame({asset: curve.pct_change() for asset, curve in equity_curves.items()})
        return returns_df.corr()

    def optimize_portfolio_weights(
        self, assets_data: Dict[str, pd.DataFrame], params: Dict[str, TradingParameters], objective: str = "sharpe_ratio"
    ) -> Dict[str, float]:
        n_assets = len(assets_data)
        if n_assets < 2:
            return {list(assets_data.keys())[0]: 1.0} if assets_data else {}

        equal_weight = 1.0 / n_assets
        return {asset: equal_weight for asset in assets_data.keys()}


class RiskAnalyticsService:
    """Advanced risk analytics and reporting."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def calculate_risk_metrics(
        self, equity_curve: pd.Series, benchmark: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        if equity_curve.empty:
            return {}

        returns = equity_curve.pct_change().dropna()

        metrics: Dict[str, float] = {
            "var_95": self._calculate_var(returns, 0.05),
            "var_99": self._calculate_var(returns, 0.01),
            "expected_shortfall_95": self._calculate_expected_shortfall(returns, 0.05),
            "expected_shortfall_99": self._calculate_expected_shortfall(returns, 0.01),
            "volatility": returns.std() * np.sqrt(252),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "downside_deviation": self._calculate_downside_deviation(returns),
            "ulcer_index": self._calculate_ulcer_index(equity_curve),
            "pain_index": self._calculate_pain_index(equity_curve),
        }

        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

        return metrics

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        if returns.empty:
            return 0.0
        return float(returns.quantile(confidence))

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.05) -> float:
        if returns.empty:
            return 0.0
        var = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return float(tail_returns.mean()) if not tail_returns.empty else 0.0

    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        downside_returns = returns[returns < target]
        return float(downside_returns.std() * np.sqrt(252)) if not downside_returns.empty else 0.0

    def _calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        peak = equity_curve.expanding().max()
        drawdown = ((equity_curve - peak) / peak) * 100
        return float(np.sqrt((drawdown ** 2).mean()))

    def _calculate_pain_index(self, equity_curve: pd.Series) -> float:
        peak = equity_curve.expanding().max()
        drawdown = ((equity_curve - peak) / peak) * 100
        return float(abs(drawdown.mean()))

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join="inner")
        if aligned_data.empty:
            return {}

        strategy_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        active_returns = strategy_returns - bench_returns

        beta = (
            float(strategy_returns.cov(bench_returns) / bench_returns.var())
            if bench_returns.var() > 0
            else 0.0
        )
        alpha = float((strategy_returns.mean() - bench_returns.mean()) * 252)
        information_ratio = (
            float(active_returns.mean() / active_returns.std() * np.sqrt(252))
            if active_returns.std() > 0
            else 0.0
        )
        tracking_error = float(active_returns.std() * np.sqrt(252))

        up_capture = (
            float(strategy_returns[bench_returns > 0].mean() / bench_returns[bench_returns > 0].mean())
            if (bench_returns > 0).any()
            else 0.0
        )
        down_capture = (
            float(strategy_returns[bench_returns < 0].mean() / bench_returns[bench_returns < 0].mean())
            if (bench_returns < 0).any()
            else 0.0
        )

        return {
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "up_capture": up_capture,
            "down_capture": down_capture,
        }


class StrategyTester:
    """Comprehensive testing framework for trading strategies."""

    def __init__(self, engine: TradingEngine):
        self._engine = engine
        self._logger = engine._logger

    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        params: TradingParameters,
        train_ratio: float = 0.7,
        step_ratio: float = 0.1,
        min_train_periods: int = 252,
    ) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []
        data_length = len(data)

        train_size = max(int(data_length * train_ratio), min_train_periods)
        step_size = max(int(data_length * step_ratio), 30)

        for start_idx in range(0, data_length - train_size - step_size, step_size):
            train_end = start_idx + train_size
            test_end = min(train_end + step_size, data_length)

            if test_end - train_end < 30:
                continue

            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]

            try:
                result = self._engine.run_strategy(test_data, params)
                results.append(
                    {
                        "period_start": test_data.index[0],
                        "period_end": test_data.index[-1],
                        "train_periods": len(train_data),
                        "test_periods": len(test_data),
                        "total_return": result.total_return,
                        "annualized_return": result.annualized_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "sortino_ratio": result.sortino_ratio,
                        "max_drawdown": result.max_drawdown,
                        "volatility": result.volatility,
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                    }
                )
            except Exception as exc:
                self._logger.error("Walk-forward test failed for period %d: %s", start_idx, exc)

        return pd.DataFrame(results)

    def monte_carlo_simulation(
        self,
        data: pd.DataFrame,
        params: TradingParameters,
        n_simulations: int = 1000,
        block_length: int = 30,
    ) -> Dict[str, np.ndarray]:
        results: Dict[str, List[float]] = {
            "returns": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "volatilities": [],
            "win_rates": [],
            "total_trades": [],
        }

        returns = data["close"].pct_change().dropna()

        for i in range(n_simulations):
            try:
                synthetic_returns = self._block_bootstrap(returns, len(data), block_length)
                synthetic_data = self._returns_to_ohlcv(synthetic_returns, data.iloc[0])
                result = self._engine.run_strategy(synthetic_data, params)

                results["returns"].append(result.total_return)
                results["sharpe_ratios"].append(result.sharpe_ratio)
                results["max_drawdowns"].append(result.max_drawdown)
                results["volatilities"].append(result.volatility)
                results["win_rates"].append(result.win_rate)
                results["total_trades"].append(result.total_trades)
            except Exception as exc:
                self._logger.warning("Monte Carlo simulation %d failed: %s", i, exc)

        return {key: np.array(values) for key, values in results.items()}

    def _block_bootstrap(self, returns: pd.Series, target_length: int, block_length: int) -> pd.Series:
        n_blocks = int(np.ceil(target_length / block_length))
        bootstrapped_returns: List[float] = []

        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(returns) - block_length + 1)
            block = returns.iloc[start_idx : start_idx + block_length]
            bootstrapped_returns.extend(block.values)

        bootstrapped_returns = bootstrapped_returns[:target_length]
        return pd.Series(bootstrapped_returns)

    def _returns_to_ohlcv(self, returns: pd.Series, initial_data: pd.Series) -> pd.DataFrame:
        prices = initial_data["close"] * (1 + returns).cumprod()

        data = pd.DataFrame(index=pd.date_range(start="2020-01-01", periods=len(prices), freq="D"))
        data["close"] = prices.values
        data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])

        noise = np.random.normal(0, 0.002, len(prices))
        data["high"] = data["close"] * (1 + np.abs(noise))
        data["low"] = data["close"] * (1 - np.abs(noise))
        data["volume"] = np.random.randint(1000, 10000, len(prices))
        return data


# =================== Factory and High-level Wrappers ===================


class TradingEngineFactory:
    """Factory helpers for creating pre-configured trading engines."""

    @staticmethod
    def create_default_engine(config: Optional[EngineConfig] = None) -> TradingEngine:
        return TradingEngine(config=config or EngineConfig())

    @staticmethod
    def create_conservative_engine() -> TradingEngine:
        config = EngineConfig(
            max_position_size=0.5,
            max_portfolio_risk=0.01,
            max_drawdown_threshold=0.10,
            volatility_threshold=0.20,
        )
        return TradingEngine(config=config)

    @staticmethod
    def create_aggressive_engine() -> TradingEngine:
        config = EngineConfig(
            max_position_size=1.5,
            max_portfolio_risk=0.05,
            max_drawdown_threshold=0.25,
            volatility_threshold=0.40,
        )
        return TradingEngine(config=config)

    @staticmethod
    def create_test_engine() -> TradingEngine:
        config = EngineConfig(cache_indicators=False, log_level="ERROR", min_data_points=50)
        return TradingEngine(config=config)


class _NoShortSignalService(TradingSignalService):
    """Signal generator variant that removes short signals."""

    def generate_signals(self, indicators, params):  # type: ignore[override]
        signals = super().generate_signals(indicators, params)
        return signals.where(signals > 0, 0.0)


class TradingStrategies:
    """Backwards-compatible wrapper expected by GUI and AutoTrader."""

    def __init__(self, engine: Optional[TradingEngine] = None, logger: Optional[logging.Logger] = None):
        self._base_engine = engine or TradingEngineFactory.create_default_engine()
        self._logger = logger or logging.getLogger("TradingStrategiesShim")

    def _mk_engine(self, allow_short: bool) -> TradingEngine:
        if allow_short:
            return self._base_engine
        return TradingEngine(
            config=self._base_engine._config if hasattr(self._base_engine, "_config") else None,
            validator=None,
            indicator_calculator=None,
            signal_generator=_NoShortSignalService(self._logger),
            risk_manager=None,
            backtest_engine=None,
            logger=self._logger,
        )

    def run_strategy(self, *args, **kwargs):
        return {"status": "ok"}, pd.DataFrame(), pd.Series(dtype=float)

    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        fee: float = 0.0004,
        slippage: float = 0.0002,
        fraction: float = 0.05,
        allow_short: bool = False,
        ai_model=None,
        ai_weight: float = 0.0,
        ai_threshold_bps: float = 5.0,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
        params = TradingParameters(
            rsi_period=14,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            ema_fast_period=12,
            ema_slow_period=26,
            sma_trend_period=50,
            bb_period=20,
            bb_std_mult=2.0,
            atr_period=14,
            macd_signal_period=9,
            stoch_k_period=14,
            stoch_d_period=3,
            signal_threshold=0.10,
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            position_size=float(max(0.0, min(1.0, fraction))),
            max_position_risk=0.02,
            volatility_target=0.15,
            kelly_fraction=0.25,
        )

        fee_bps = float((fee + slippage) * 10000.0)
        engine = self._mk_engine(bool(allow_short))

        if ai_model is None or ai_weight <= 0.0:
            result = engine.run_strategy(
                data=data,
                params=params,
                initial_capital=float(initial_capital),
                fee_bps=fee_bps,
            )
        else:
            try:
                from bridges.ai_trading_bridge import AITradingBridge
            except Exception:  # pragma: no cover - optional dependency
                AITradingBridge = None

            if AITradingBridge is None:
                result = engine.run_strategy(
                    data=data,
                    params=params,
                    initial_capital=float(initial_capital),
                    fee_bps=fee_bps,
                )
            else:
                validated_data = engine._validator.validate_ohlcv(data)
                indicators = engine._indicator_calculator.calculate_indicators(validated_data, params)
                raw_signals = engine._signal_generator.generate_signals(indicators, params)
                bridge = AITradingBridge(ai_model, weight_ai=float(ai_weight), threshold_bps=float(ai_threshold_bps))
                fused_signals = bridge.integrate(validated_data, raw_signals)
                managed_signals = engine._risk_manager.apply_risk_management(
                    validated_data, fused_signals, indicators, params
                )
                result = engine._backtest_engine.run_backtest(
                    validated_data,
                    managed_signals,
                    params,
                    engine._config,
                    float(initial_capital),
                    fee_bps,
                )

        metrics = {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "omega_ratio": result.omega_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "trades": int(result.total_trades),
            "tail_ratio": result.tail_ratio,
            "var_95": result.var_95,
            "expected_shortfall_95": result.expected_shortfall_95,
        }
        return metrics, result.trades, result.equity_curve


__all__ = [
    "BacktestEngine",
    "BacktestExecutionError",
    "ConfigurationError",
    "DataValidator",
    "DataValidationError",
    "DataValidationService",
    "EngineConfig",
    "InsufficientDataError",
    "IndicatorComputationError",
    "IndicatorCalculator",
    "MarketRegime",
    "OrderType",
    "PortfolioManager",
    "RiskAnalyticsService",
    "RiskLimitExceededError",
    "RiskManager",
    "RiskManagementService",
    "SignalGenerator",
    "SignalType",
    "StrategyTester",
    "TradingEngine",
    "TradingEngineError",
    "TradingEngineFactory",
    "TradingParameters",
    "TradingSignalService",
    "TradingStrategies",
]
