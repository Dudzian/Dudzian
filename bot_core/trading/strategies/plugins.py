"""Strategy plugin implementations using :class:`TradingParameters`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - hints only
    from bot_core.trading.engine import TechnicalIndicators, TradingParameters

TStrategy = TypeVar("TStrategy", bound="StrategyPlugin")


class StrategyPlugin(ABC):
    """Base class for reusable trading strategy plugins."""

    #: Unique identifier used in ``TradingParameters.ensemble_weights``.
    name: str = "base"

    #: Optional description for UI/help contexts.
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name!r})"

    @abstractmethod
    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Return a signal series in the ``[-1, 1]`` range."""

    def ensure_index(self, series: pd.Series, reference: pd.Index) -> pd.Series:
        """Return ``series`` reindexed to ``reference``.

        Plugins may create derived series using rolling operations which can
        shorten the index. This helper ensures we always return signals aligned
        with the indicator index.
        """

        if series.index.equals(reference):
            return series
        return series.reindex(reference, method="nearest", tolerance=None).fillna(0.0)


class StrategyCatalog:
    """Registry of available strategy plugins."""

    def __init__(
        self, plugins: Optional[Iterable[Union[StrategyPlugin, Type[TStrategy]]]] = None
    ) -> None:
        self._registry: Dict[str, Callable[[], StrategyPlugin]] = {}
        if plugins:
            for plugin in plugins:
                self.register(plugin)

    def register(self, plugin: Union[StrategyPlugin, Type[TStrategy]]) -> None:
        """Register ``plugin`` in the catalog."""

        if isinstance(plugin, type):
            name = getattr(plugin, "name", "")

            def factory(cls: Type[TStrategy] = plugin) -> StrategyPlugin:
                return cls()

            instance_factory = factory
        else:
            name = getattr(plugin, "name", "")

            def factory(instance: StrategyPlugin = plugin) -> StrategyPlugin:
                return instance

            instance_factory = factory

        if not name:
            raise ValueError("Strategy plugin must define a non-empty 'name'")
        key = str(name)
        if key in self._registry:
            raise ValueError(f"Strategy plugin '{key}' is already registered")
        self._registry[key] = instance_factory

    def create(self, name: str) -> StrategyPlugin | None:
        factory = self._registry.get(name)
        if factory is None:
            return None
        return factory()

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._registry))

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._registry

    def describe(self) -> Tuple[Mapping[str, str], ...]:
        """Zwraca uproszczony opis strategii przydatny w UI."""

        summary: list[Mapping[str, str]] = []
        for name in self.available():
            plugin = self.create(name)
            if plugin is None:
                continue
            summary.append({"name": name, "description": plugin.description})
        return tuple(summary)

    @classmethod
    def default(cls) -> "StrategyCatalog":
        """Return catalog populated with built-in strategies."""

        return cls(
            plugins=(
                TrendFollowingStrategy,
                DayTradingStrategy,
                MeanReversionStrategy,
                ArbitrageStrategy,
            )
        )


class TrendFollowingStrategy(StrategyPlugin):
    """Classic trend-following ensemble built on moving averages."""

    name = "trend_following"
    description = "EMA and SMA crossovers highlighting persistent direction."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        ema_diff = indicators.ema_fast - indicators.ema_slow
        ema_signal = pd.Series(np.where(ema_diff > 0, 1.0, -1.0), index=indicators.ema_fast.index)

        price_vs_sma = indicators.ema_fast - indicators.sma_trend
        trend_signal = pd.Series(np.where(price_vs_sma > 0, 1.0, -1.0), index=indicators.ema_fast.index)

        combined = ema_signal * 0.6 + trend_signal * 0.4
        return combined.clip(-1.0, 1.0)


class DayTradingStrategy(StrategyPlugin):
    """Intraday momentum strategy focusing on short-lived swings."""

    name = "day_trading"
    description = "Short momentum bursts with volatility-aware scaling."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        fast_returns = indicators.ema_fast.pct_change(fill_method=None).fillna(0.0)
        momentum_window = max(1, int(params.day_trading_momentum_window))
        momentum_score = fast_returns.rolling(window=momentum_window, min_periods=1).mean()

        volatility = (indicators.atr / (indicators.ema_fast.abs() + 1e-12)).clip(lower=0.0)
        volatility_window = max(1, int(params.day_trading_volatility_window))
        volatility_score = volatility.rolling(window=volatility_window, min_periods=1).mean().replace(0.0, np.nan)
        median_vol = float(volatility_score.median(skipna=True))
        if not np.isfinite(median_vol) or median_vol <= 0:
            median_vol = 1.0
        baseline = volatility_score.fillna(median_vol)

        raw_signal = momentum_score / baseline
        scaled = pd.Series(np.tanh(raw_signal * 2.5), index=indicators.ema_fast.index)

        band_mid = getattr(indicators, "bollinger_middle", indicators.ema_fast)
        intraday_bias = (indicators.ema_fast - band_mid) / (band_mid.abs() + 1e-12)
        bias_adjustment = intraday_bias.clip(-1.0, 1.0) * 0.2

        return (scaled + bias_adjustment).clip(-1.0, 1.0)


class MeanReversionStrategy(StrategyPlugin):
    """RSI/Bollinger-based contrarian strategy."""

    name = "mean_reversion"
    description = "Fade extremes using RSI and Bollinger Bands confirmation."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        rsi_signal = pd.Series(0.0, index=indicators.rsi.index)
        rsi_signal[indicators.rsi < params.rsi_oversold] = 1.0
        rsi_signal[indicators.rsi > params.rsi_overbought] = -1.0

        bb_signal = pd.Series(0.0, index=indicators.ema_fast.index)
        bb_signal[indicators.ema_fast < indicators.bollinger_lower] = 1.0
        bb_signal[indicators.ema_fast > indicators.bollinger_upper] = -1.0

        combined = (rsi_signal * 0.6 + bb_signal * 0.4).clip(-1.0, 1.0)
        return combined


class ArbitrageStrategy(StrategyPlugin):
    """Light-weight spread arbitrage approximation using a synthetic benchmark."""

    name = "arbitrage"
    description = "Exploit deviations from the Bollinger mid-band as proxy spreads."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        reference = getattr(indicators, "bollinger_middle", indicators.ema_fast)
        spread = (indicators.ema_fast - reference) / (reference.abs() + 1e-12)
        confirmation_window = max(1, int(params.arbitrage_confirmation_window))
        confirmed = spread.rolling(window=confirmation_window, min_periods=1).mean()

        threshold = float(params.arbitrage_spread_threshold)
        signal = pd.Series(0.0, index=spread.index)
        signal[confirmed > threshold] = -1.0
        signal[confirmed < -threshold] = 1.0

        dampening = np.tanh(np.abs(confirmed) / (threshold * 3.0))
        return (signal * dampening).clip(-1.0, 1.0)

