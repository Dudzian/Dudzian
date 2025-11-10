"""Strategy plugin implementations using :class:`TradingParameters`."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional import for regime helpers
    from bot_core.ai.regime import MarketRegime
except Exception:  # pragma: no cover - fallback for stripped builds
    MarketRegime = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:  # pragma: no cover - hints only
    from bot_core.ai.adaptive import AdaptiveStrategyLearner
    from bot_core.trading.engine import TechnicalIndicators, TradingParameters

from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG

from .cross_exchange_hedge import (
    CrossExchangeHedgeConfig,
    compute_cross_exchange_hedge_signal,
)
from .futures_spread import (
    FuturesSpreadSignalConfig,
    compute_futures_spread_signal,
)
from .options_income import OptionsIncomeSignalConfig, compute_options_income_signal

_BUILTIN_PLUGIN_REGISTRY: MutableMapping[str, Type["StrategyPlugin"]] = {}
_REGISTERED_ENGINE_KEYS: set[str] = set()

TStrategy = TypeVar("TStrategy", bound="StrategyPlugin")

_LOGGER = logging.getLogger(__name__)

def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sequence(values: Iterable[str] | None) -> Tuple[str, ...]:
    if not values:
        return ()
    seen: Dict[str, None] = {}
    result: list[str] = []
    for raw in values:
        text = _normalize_text(str(raw))
        if text is None or text in seen:
            continue
        seen[text] = None
        result.append(text)
    return tuple(result)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_metrics_map(metrics: Mapping[str, float] | None) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not metrics:
        return normalized
    for key, value in metrics.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _regime_key(value: object) -> str:
    if MarketRegime is not None:
        try:
            return MarketRegime(value).value  # type: ignore[arg-type]
        except Exception:
            pass
    if isinstance(value, str) and value:
        return value.lower()
    return "trend"


class StrategyPlugin(ABC):
    """Base class for reusable trading strategy plugins."""

    #: Unique identifier used in ``TradingParameters.ensemble_weights``.
    name: str = "base"

    #: Optional description for UI/help contexts.
    description: str = ""

    #: Optional metadata describing licensing and risk constraints.
    license_tier: str | None = None
    risk_classes: Tuple[str, ...] = ()
    required_data: Tuple[str, ...] = ()
    capability: str | None = None
    tags: Tuple[str, ...] = ()
    engine_key: str | None = None
    extra_tags: Tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        engine_key = _normalize_text(getattr(cls, "engine_key", None))
        if engine_key is None:
            return

        cls.engine_key = engine_key

        try:
            spec = DEFAULT_STRATEGY_CATALOG.get(engine_key)
        except KeyError as exc:  # pragma: no cover - sanity guard
            raise ValueError(
                "Strategy plugin '%s' declares engine_key='%s' which is not registered "
                "in DEFAULT_STRATEGY_CATALOG" % (cls.__name__, engine_key)
            ) from exc

        cls.license_tier = spec.license_tier
        cls.risk_classes = tuple(spec.risk_classes)
        cls.required_data = tuple(spec.required_data)
        cls.capability = spec.capability

        base_tags = _normalize_sequence(spec.default_tags)
        extra_tags = _normalize_sequence(getattr(cls, "extra_tags", ()))
        override_tags = _normalize_sequence(getattr(cls, "tags", ()))

        if override_tags:
            tags = tuple(dict.fromkeys((*base_tags, *override_tags)))
        elif extra_tags:
            tags = tuple(dict.fromkeys((*base_tags, *extra_tags)))
        else:
            tags = base_tags

        cls.tags = tags

        if engine_key in _REGISTERED_ENGINE_KEYS:
            existing = _BUILTIN_PLUGIN_REGISTRY.get(engine_key)
            existing_name = existing.__name__ if existing else "<external>"
            raise ValueError(
                "Zduplikowano plugin dla engine_key='%s' (zarejestrowany: %s, nowy: %s)"
                % (engine_key, existing_name, cls.__name__)
            )
        _REGISTERED_ENGINE_KEYS.add(engine_key)

        if cls.__module__ == __name__:
            _BUILTIN_PLUGIN_REGISTRY[engine_key] = cls

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

    def metadata(self) -> Mapping[str, object]:
        """Return normalized metadata describing the strategy plugin."""

        payload: Dict[str, object] = {}
        license_tier = _normalize_text(self.license_tier)
        if license_tier:
            payload["license_tier"] = license_tier
        risk_classes = _normalize_sequence(self.risk_classes)
        if risk_classes:
            payload["risk_classes"] = risk_classes
        required_data = _normalize_sequence(self.required_data)
        if required_data:
            payload["required_data"] = required_data
        capability = _normalize_text(self.capability)
        if capability:
            payload["capability"] = capability
        tags = _normalize_sequence(self.tags)
        if tags:
            payload["tags"] = tags
        engine_key = _normalize_text(getattr(type(self), "engine_key", None))
        if engine_key:
            payload["engine"] = engine_key
        payload["description"] = str(self.description or "")
        return MappingProxyType(payload)


class StrategyCatalog:
    """Registry of available strategy plugins."""

    def __init__(
        self, plugins: Optional[Iterable[Union[StrategyPlugin, Type[TStrategy]]]] = None
    ) -> None:
        self._registry: Dict[str, Callable[[], StrategyPlugin]] = {}
        self._adaptive: "AdaptiveStrategyLearner | None" = None
        self._dynamic_presets: MutableMapping[str, Mapping[str, object]] = {}
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
        adaptive = self._adaptive
        if adaptive is not None:
            try:
                adaptive.register_strategies("trend", (key,))
            except Exception:  # pragma: no cover - defensywne logowanie
                _LOGGER.debug("Adaptive learner rejected strategy registration", exc_info=True)

    def create(self, name: str) -> StrategyPlugin | None:
        factory = self._registry.get(name)
        if factory is None:
            return None
        return factory()

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._registry))

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._registry

    def describe(self) -> Tuple[Mapping[str, object], ...]:
        """Zwraca pełny opis metadanych strategii dla warstw klienckich."""

        summary: list[Mapping[str, object]] = []
        for name in self.available():
            plugin = self.create(name)
            if plugin is None:
                continue
            metadata = dict(plugin.metadata())
            metadata.setdefault("name", name)
            summary.append(metadata)
        return tuple(summary)

    def metadata_for(self, name: str) -> Mapping[str, object]:
        """Return normalized metadata for strategy ``name``."""

        plugin = self.create(name)
        if plugin is None:
            return MappingProxyType({})
        metadata = dict(plugin.metadata())
        metadata.setdefault("name", name)
        return MappingProxyType(metadata)

    def attach_adaptive_learner(self, learner: "AdaptiveStrategyLearner | None") -> None:
        """Connect runtime adaptive learner providing dynamic presets."""

        self._adaptive = learner
        self._dynamic_presets.clear()

    def dynamic_preset_for(
        self,
        regime: object,
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> Mapping[str, object] | None:
        learner = self._adaptive
        if learner is None:
            return None
        metrics_map = _normalize_metrics_map(metrics)
        regime_key = _regime_key(regime)
        try:
            preset = learner.build_dynamic_preset(
                regime_key, metrics=metrics_map or None
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            _LOGGER.debug("Adaptive learner failed to build preset", exc_info=True)
            return None
        if preset is None:
            return None
        key = regime_key
        payload = dict(preset)
        if metrics_map and "metrics" not in payload:
            payload["metrics"] = dict(metrics_map)
        generated_at = payload.get("generated_at")
        if not isinstance(generated_at, str) or not generated_at.strip():
            payload["generated_at"] = _now_iso()
        self._dynamic_presets[key] = MappingProxyType(payload)
        return self._dynamic_presets[key]

    def last_dynamic_preset(self, regime: object) -> Mapping[str, object] | None:
        return self._dynamic_presets.get(_regime_key(regime))

    def dynamic_presets_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        return {key: dict(value) for key, value in self._dynamic_presets.items()}

    @classmethod
    def default(cls) -> "StrategyCatalog":
        """Return catalog populated with built-in strategies."""

        return cls(
            plugins=(
                AdaptiveMarketMakingPlugin,
                TrendFollowingStrategy,
                DayTradingStrategy,
                MeanReversionStrategy,
                ArbitrageStrategy,
                GridTradingStrategy,
                VolatilityTargetStrategy,
                ScalpingStrategy,
                TriangularArbitragePlugin,
                OptionsIncomeStrategy,
                StatisticalArbitrageStrategy,
                FuturesSpreadStrategy,
                CrossExchangeHedgeStrategy,
            ),
        )


class AdaptiveMarketMakingPlugin(StrategyPlugin):
    """Reaguje na inventory i zmienność, aby skalować ekspozycję MM."""

    engine_key = "adaptive_market_making"
    name = "adaptive_market_making"
    description = "Inventory-aware market making sygnalizujący ekspozycję względem zmienności."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        price_index = indicators.ema_fast.index

        volatility_series: Optional[pd.Series] = None
        inventory_series: Optional[pd.Series] = None

        if market_data is not None:
            if "realized_volatility" in market_data:
                volatility_series = market_data["realized_volatility"]
            if "inventory_skew" in market_data:
                inventory_series = market_data["inventory_skew"]

        if volatility_series is None:
            atr = indicators.atr.replace(0.0, np.nan)
            volatility_series = (atr / (indicators.ema_fast.abs() + 1e-12)).rolling(window=5, min_periods=1).mean()

        if inventory_series is None:
            inventory_series = (indicators.stochastic_k - 50.0) / 50.0

        volatility_series = volatility_series.reindex(price_index)
        volatility_default = float(volatility_series.mean(skipna=True))
        if not np.isfinite(volatility_default):
            volatility_default = 0.0
        volatility_series = volatility_series.ffill().fillna(volatility_default)

        inventory_series = inventory_series.reindex(price_index)
        inventory_series = inventory_series.ffill().fillna(0.0)

        target_inventory = float(getattr(params, "adaptive_mm_target_inventory", 0.0) or 0.0)
        inventory_scale = max(float(getattr(params, "adaptive_mm_inventory_scale", 0.75) or 0.75), 1e-6)
        volatility_target = max(float(getattr(params, "adaptive_mm_target_volatility", 0.18) or 0.18), 1e-6)
        volatility_sensitivity = float(getattr(params, "adaptive_mm_volatility_sensitivity", 1.6) or 1.6)

        inventory_bias = (inventory_series - target_inventory) / inventory_scale
        inventory_component = -np.tanh(inventory_bias)

        volatility_gap = (volatility_target - volatility_series) / volatility_target
        volatility_component = np.tanh(volatility_gap * volatility_sensitivity)

        combined = inventory_component * 0.6 + volatility_component * 0.4
        return pd.Series(combined.clip(-1.0, 1.0), index=price_index)


class TrendFollowingStrategy(StrategyPlugin):
    """Classic trend-following ensemble built on moving averages."""

    engine_key = "daily_trend_momentum"
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

    engine_key = "day_trading"
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

    engine_key = "mean_reversion"
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

    engine_key = "cross_exchange_arbitrage"
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


class GridTradingStrategy(StrategyPlugin):
    """Market-making grid reacting to deviations around a slow anchor."""

    name = "grid_trading"
    description = "Neutral grid around SMA with ATR-aware band sizing."
    license_tier = "professional"
    risk_classes = ("market_making",)
    required_data = ("order_book", "ohlcv")
    capability = "grid_trading"
    tags = ("grid", "market_making")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        anchor = indicators.sma_trend
        price = indicators.ema_fast
        atr = indicators.atr.replace(0.0, np.nan)
        deviation = ((price - anchor) / atr).fillna(0.0)

        grid_signal = pd.Series(0.0, index=price.index)
        grid_signal[deviation > 0.75] = -1.0
        grid_signal[deviation < -0.75] = 1.0

        soft_bias = deviation.clip(-2.0, 2.0) / 2.0
        combined = (grid_signal * 0.6 + soft_bias * 0.4).clip(-1.0, 1.0)
        return combined


class VolatilityTargetStrategy(StrategyPlugin):
    """Adjust exposure to steer realised volatility toward target."""

    name = "volatility_target"
    description = "Dynamically scales exposure to hit target volatility."
    license_tier = "enterprise"
    risk_classes = ("risk_control", "volatility")
    required_data = ("ohlcv", "realized_volatility")
    capability = "volatility_target"
    tags = ("volatility", "risk")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        price = indicators.ema_fast
        atr = indicators.atr.replace(0.0, np.nan)
        realized_vol = (atr / (price.abs() + 1e-12)).rolling(window=5, min_periods=1).mean()
        target = max(1e-4, float(params.volatility_target))
        vol_gap = (target - realized_vol).fillna(0.0) / target

        trend_bias = (price - indicators.ema_slow) / (atr * 2.0)
        trend_bias = trend_bias.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)

        adjustment = pd.Series(np.tanh(vol_gap * 2.5), index=price.index)
        combined = (adjustment * 0.7 + trend_bias * 0.3).clip(-1.0, 1.0)
        return combined


class ScalpingStrategy(StrategyPlugin):
    """Fast mean-reversion around short momentum for low-latency trading."""

    name = "scalping"
    description = "MACD micro-divergence with stochastic confirmation."
    license_tier = "professional"
    risk_classes = ("intraday", "scalping")
    required_data = ("ohlcv", "order_book")
    capability = "scalping"
    tags = ("intraday", "scalping")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        macd_diff = indicators.macd - indicators.macd_signal
        atr = indicators.atr.replace(0.0, np.nan)
        momentum = (macd_diff / (atr / 2.0 + 1e-12)).clip(-4.0, 4.0)
        stochastic_bias = (indicators.stochastic_k - 50.0) / 50.0

        signal = (np.tanh(momentum) * 0.65 + stochastic_bias * 0.35).clip(-1.0, 1.0)
        return pd.Series(signal, index=indicators.macd.index)


class OptionsIncomeStrategy(StrategyPlugin):
    """Simplified theta harvesting profile informed by volatility spreads."""

    name = "options_income"
    description = "Harvests premium when implied volatility outruns realised."
    license_tier = "enterprise"
    risk_classes = ("derivatives", "income")
    required_data = ("options_chain", "greeks", "ohlcv")
    capability = "options_income"
    tags = ("options", "income")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        config = OptionsIncomeSignalConfig(
            spread_scale=float(getattr(params, "options_income_spread_scale", 0.35) or 0.35),
            delta_anchor_weight=float(getattr(params, "options_income_delta_weight", 0.3) or 0.3),
            theta_weight=float(getattr(params, "options_income_theta_weight", 0.7) or 0.7),
            anchor_window=int(getattr(params, "options_income_anchor_window", 12) or 12),
        )

        series = compute_options_income_signal(
            fast_price=indicators.ema_fast,
            atr=indicators.atr,
            slow_anchor=indicators.sma_trend,
            market_data=market_data,
            config=config,
        )
        return self.ensure_index(series, indicators.ema_fast.index)


class TriangularArbitragePlugin(StrategyPlugin):
    """Ocena przewagi arbitrażowej dla uproszczonych sygnałów portfelowych."""

    engine_key = "triangular_arbitrage"
    name = "triangular_arbitrage"
    description = "Wykrywa sygnały arbitrażu trójkątnego po korekcie o latency."

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        index = indicators.ema_fast.index

        if market_data is None:
            forward_edge = None
            reverse_edge = None
            latency_ms = None
        else:
            forward_edge = market_data.get("triangular_edge_bps")
            reverse_edge = market_data.get("triangular_reverse_edge_bps")
            latency_ms = market_data.get("latency_ms")

        if forward_edge is None:
            forward_edge = pd.Series(0.0, index=index)
        if reverse_edge is None:
            reverse_edge = pd.Series(0.0, index=index)
        if latency_ms is None:
            latency_ms = pd.Series(0.0, index=index)

        forward_edge = forward_edge.reindex(index).ffill().fillna(0.0)
        reverse_edge = reverse_edge.reindex(index).ffill().fillna(0.0)
        latency_ms = latency_ms.reindex(index).ffill().fillna(0.0)

        min_edge = max(float(getattr(params, "triangular_min_edge_bps", 4.0) or 4.0), 1e-6)
        latency_penalty = float(getattr(params, "triangular_latency_penalty", 0.0015) or 0.0015)

        best_edge = forward_edge.copy()
        direction = pd.Series(1.0, index=index)
        better_mask = reverse_edge > best_edge
        best_edge[better_mask] = reverse_edge[better_mask]
        direction[better_mask] = -1.0

        normalized = (best_edge - min_edge) / min_edge
        latency_adjustment = latency_penalty * (latency_ms / 100.0)
        edge_signal = normalized - latency_adjustment
        edge_signal = np.maximum(edge_signal, 0.0)

        activation = np.tanh(edge_signal * 2.0)
        series = direction * activation
        return pd.Series(series.clip(-1.0, 1.0), index=index)


class StatisticalArbitrageStrategy(StrategyPlugin):
    """Pairs-style mean reversion using MACD and Bollinger spreads."""

    name = "statistical_arbitrage"
    description = "Pairs spreads using MACD z-score and Bollinger confirmation."
    license_tier = "professional"
    risk_classes = ("statistical", "mean_reversion")
    required_data = ("ohlcv", "spread_history")
    capability = "stat_arbitrage"
    tags = ("stat_arbitrage", "pairs_trading")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        macd_z = indicators.macd - indicators.macd_signal
        atr = indicators.atr.replace(0.0, np.nan)
        macd_norm = (macd_z / (atr + 1e-12)).clip(-5.0, 5.0)

        mid = indicators.bollinger_middle
        spread = (indicators.ema_fast - mid) / (
            (indicators.bollinger_upper - mid).replace(0.0, np.nan)
        )
        spread = spread.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        signal = (-np.tanh(macd_norm) * 0.6 - spread * 0.4).clip(-1.0, 1.0)
        return pd.Series(signal, index=indicators.macd.index)


class FuturesSpreadStrategy(StrategyPlugin):
    """Kontroluje rozjazd kontraktów futures względem siebie oraz rynku kasowego."""

    engine_key = "futures_spread"
    name = "futures_spread"
    description = "Hedguje carry oraz funding poprzez adaptacyjny trading spreadów."
    license_tier = "enterprise"
    risk_classes = ("derivatives", "market_neutral")
    required_data = ("futures_curve", "funding_rates", "ohlcv")
    capability = "futures_spread"
    tags = ("futures", "hedge", "basis")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        spread_source: Optional[pd.Series] = None
        if market_data is not None:
            for column in ("spread_z", "spread_zscore", "futures_spread_zscore"):
                if column in market_data:
                    spread_source = market_data[column]
                    break
        if spread_source is None:
            spread_source = (
                (indicators.macd - indicators.macd_signal)
                / (indicators.atr.replace(0.0, np.nan).ffill().replace(0.0, 1.0))
            )
        spread_source = spread_source.reindex(indicators.macd.index).ffill().fillna(0.0)

        config = FuturesSpreadSignalConfig(
            entry_z=float(getattr(params, "futures_spread_entry_z", 1.25) or 1.25),
            exit_z=float(getattr(params, "futures_spread_exit_z", 0.4) or 0.4),
            basis_scale=float(getattr(params, "futures_spread_basis_scale", 0.015) or 0.015),
            funding_scale=float(getattr(params, "futures_spread_funding_scale", 0.0008) or 0.0008),
            carry_weight=float(getattr(params, "futures_spread_carry_weight", 0.35) or 0.35),
            funding_weight=float(getattr(params, "futures_spread_funding_weight", 0.25) or 0.25),
        )

        series = compute_futures_spread_signal(
            spread_zscore=spread_source,
            market_data=market_data,
            config=config,
        )
        return self.ensure_index(series, indicators.macd.index)


class CrossExchangeHedgeStrategy(StrategyPlugin):
    """Neutralizuje ekspozycję poprzez dynamiczne hedgingi między venue."""

    engine_key = "cross_exchange_hedge"
    name = "cross_exchange_hedge"
    description = "Rebalansuje delta pomiędzy spot a futures biorąc pod uwagę latency i inventory."
    license_tier = "enterprise"
    risk_classes = ("hedging", "liquidity")
    required_data = ("spot_basis", "inventory_skew", "latency_metrics")
    capability = "cross_exchange_hedge"
    tags = ("hedge", "multi_venue", "delta")

    def generate(
        self,
        indicators: "TechnicalIndicators",
        params: "TradingParameters",
        *,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        basis_series: Optional[pd.Series] = None
        inventory_series: Optional[pd.Series] = None
        if market_data is not None:
            if "spot_basis" in market_data:
                basis_series = market_data["spot_basis"]
            if "inventory_skew" in market_data:
                inventory_series = market_data["inventory_skew"]

        if basis_series is None:
            basis_series = (indicators.ema_fast - indicators.sma_trend) / (
                indicators.atr.replace(0.0, np.nan).ffill().replace(0.0, 1.0)
            )
        if inventory_series is None:
            inventory_series = (indicators.stochastic_k - 50.0) / 50.0

        basis_series = basis_series.reindex(indicators.ema_fast.index).ffill().fillna(0.0)
        inventory_series = inventory_series.reindex(indicators.ema_fast.index).ffill().fillna(0.0)

        config = CrossExchangeHedgeConfig(
            basis_scale=float(getattr(params, "cross_exchange_basis_scale", 0.01) or 0.01),
            inventory_scale=float(getattr(params, "cross_exchange_inventory_scale", 0.35) or 0.35),
            latency_penalty=float(getattr(params, "cross_exchange_latency_penalty", 0.2) or 0.2),
            hedge_weight=float(getattr(params, "cross_exchange_hedge_weight", 0.55) or 0.55),
            inventory_weight=float(getattr(params, "cross_exchange_inventory_weight", 0.25) or 0.25),
        )

        series = compute_cross_exchange_hedge_signal(
            spot_basis=basis_series,
            inventory_skew=inventory_series,
            market_data=market_data,
            config=config,
        )
        return self.ensure_index(series, indicators.ema_fast.index)

