"""Indicator utilities for KryptoLowca trading strategies."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core import IndicatorComputationError

if TYPE_CHECKING:  # pragma: no cover
    from .core import EngineConfig, TradingParameters


@dataclass(frozen=True)
class TechnicalIndicators:
    """Immutable container for technical indicators with validation."""

    rsi: pd.Series
    ema_fast: pd.Series
    ema_slow: pd.Series
    sma_trend: pd.Series
    atr: pd.Series
    bollinger_upper: pd.Series
    bollinger_lower: pd.Series
    bollinger_middle: pd.Series
    macd: pd.Series
    macd_signal: pd.Series
    stochastic_k: pd.Series
    stochastic_d: pd.Series

    def __post_init__(self) -> None:
        indicators = [
            self.rsi,
            self.ema_fast,
            self.ema_slow,
            self.sma_trend,
            self.atr,
            self.bollinger_upper,
            self.bollinger_lower,
            self.bollinger_middle,
            self.macd,
            self.macd_signal,
            self.stochastic_k,
            self.stochastic_d,
        ]

        if not all(isinstance(ind, pd.Series) for ind in indicators):
            raise ValueError("All indicators must be pandas Series")

        base_index = self.rsi.index
        if not all(ind.index.equals(base_index) for ind in indicators):
            raise ValueError("All indicators must have the same index")


class MathUtils:
    """Vectorized mathematical utilities for indicator and backtest calculations."""

    @staticmethod
    def ema_alpha(span: int) -> float:
        if span <= 0:
            raise ValueError("EMA span must be positive")
        return 2.0 / (span + 1.0)

    @staticmethod
    def safe_divide(numerator: NDArray, denominator: NDArray, fill_value: float = 0.0) -> NDArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(numerator, denominator)
            mask = np.isfinite(result)
            return np.where(mask, result, fill_value)

    @staticmethod
    def rolling_apply_numba(series: pd.Series, window: int, func: callable) -> pd.Series:
        if len(series) < window:
            return pd.Series(index=series.index, dtype=float)

        values = series.values
        result = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            result[i] = func(values[i - window + 1 : i + 1])
        return pd.Series(result, index=series.index)

    @staticmethod
    def calculate_drawdown_vectorized(equity: pd.Series) -> tuple[pd.Series, float, pd.Timedelta]:
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        is_peak = equity == peak
        peak_indices = equity.index[is_peak]

        if len(peak_indices) > 1:
            max_duration = pd.Timedelta(0)
            for i in range(len(peak_indices) - 1):
                duration = peak_indices[i + 1] - peak_indices[i]
                max_duration = max(max_duration, duration)
        else:
            max_duration = pd.Timedelta(0)

        return drawdown, float(max_dd), max_duration


class TechnicalIndicatorsService:
    """Optimized service for calculating technical indicators with caching."""

    def __init__(self, logger: logging.Logger, config: "EngineConfig"):
        self._logger = logger
        self._config = config
        self._math = MathUtils()
        self._cache: Dict[str, TechnicalIndicators] = {}

    def calculate_indicators(self, data: pd.DataFrame, params: "TradingParameters") -> TechnicalIndicators:
        if self._config.cache_indicators:
            cache_key = self._get_cache_key(data, params)
            if cache_key in self._cache:
                self._logger.debug("Using cached indicators")
                return self._cache[cache_key]

        try:
            close = data["close"]
            high = data["high"]
            low = data["low"]

            indicators = TechnicalIndicators(
                rsi=self._calculate_rsi_optimized(close, params.rsi_period),
                ema_fast=self._calculate_ema_optimized(close, params.ema_fast_period),
                ema_slow=self._calculate_ema_optimized(close, params.ema_slow_period),
                sma_trend=self._calculate_sma_optimized(close, params.sma_trend_period),
                atr=self._calculate_atr_optimized(high, low, close, params.atr_period),
                **self._calculate_bollinger_bands_optimized(close, params.bb_period, params.bb_std_mult),
                **self._calculate_macd_optimized(
                    close,
                    params.ema_fast_period,
                    params.ema_slow_period,
                    params.macd_signal_period,
                ),
                **self._calculate_stochastic_optimized(
                    high,
                    low,
                    close,
                    params.stoch_k_period,
                    params.stoch_d_period,
                ),
            )

            if self._config.cache_indicators:
                self._cache[cache_key] = indicators
                if len(self._cache) > 50:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            return indicators
        except Exception as exc:  # pragma: no cover - delegated error handling
            raise IndicatorComputationError(f"Failed to calculate indicators: {exc}") from exc

    def _get_cache_key(self, data: pd.DataFrame, params: "TradingParameters") -> str:
        data_hash = hashlib.md5(
            (str(data.index[0]) + str(data.index[-1]) + str(len(data))).encode()
        ).hexdigest()[:8]
        params_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
        return f"{data_hash}_{params_hash}"

    def _calculate_rsi_optimized(self, series: pd.Series, period: int) -> pd.Series:
        if len(series) < period + 1:
            return pd.Series(50.0, index=series.index)

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = self._math.safe_divide(avg_gain.values, avg_loss.values, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return pd.Series(rsi, index=series.index).fillna(50.0)

    def _calculate_ema_optimized(self, series: pd.Series, span: int) -> pd.Series:
        if span <= 0:
            raise ValueError("EMA span must be positive")
        if len(series) < span:
            return pd.Series(index=series.index, dtype=float)
        return series.ewm(span=span, adjust=False).mean()

    def _calculate_sma_optimized(self, series: pd.Series, window: int) -> pd.Series:
        if window <= 0:
            raise ValueError("SMA window must be positive")
        if len(series) < window:
            return pd.Series(index=series.index, dtype=float)
        return series.rolling(window=window, min_periods=1).mean()

    def _calculate_atr_optimized(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        if period <= 0:
            raise ValueError("ATR period must be positive")

        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()

        ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        true_range = ranges.max(axis=1)

        alpha = 1.0 / period
        return true_range.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_bollinger_bands_optimized(
        self, series: pd.Series, period: int, std_mult: float
    ) -> Dict[str, pd.Series]:
        if period <= 0:
            raise ValueError("Bollinger Bands period must be positive")

        sma = self._calculate_sma_optimized(series, period)
        std = series.rolling(window=period, min_periods=1).std()

        return {
            "bollinger_upper": sma + (std * std_mult),
            "bollinger_lower": sma - (std * std_mult),
            "bollinger_middle": sma,
        }

    def _calculate_macd_optimized(
        self, series: pd.Series, fast_period: int, slow_period: int, signal_period: int
    ) -> Dict[str, pd.Series]:
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")

        ema_fast = self._calculate_ema_optimized(series, fast_period)
        ema_slow = self._calculate_ema_optimized(series, slow_period)
        macd = ema_fast - ema_slow
        signal = self._calculate_ema_optimized(macd, signal_period)

        return {"macd": macd, "macd_signal": signal}

    def _calculate_stochastic_optimized(
        self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int
    ) -> Dict[str, pd.Series]:
        if k_period <= 0 or d_period <= 0:
            raise ValueError("Stochastic periods must be positive")

        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        denominator = highest_high - lowest_low
        k_percent = ((close - lowest_low) / denominator).fillna(0) * 100
        k_percent = k_percent.clip(0, 100)

        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
        return {"stochastic_k": k_percent, "stochastic_d": d_percent}


__all__ = [
    "MathUtils",
    "TechnicalIndicators",
    "TechnicalIndicatorsService",
]
