"""Utilities for classifying market regimes based on recent price history."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

from bot_core.ai.config_loader import load_risk_thresholds


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` if it is mapping-like, otherwise an empty mapping."""

    return value if isinstance(value, Mapping) else {}


class MarketRegime(str, Enum):
    """Supported market regimes recognised by the classifier."""

    TREND = "trend"
    DAILY = "daily"
    MEAN_REVERSION = "mean_reversion"


class RiskLevel(str, Enum):
    """Discrete warstwy ryzyka wykorzystywane przy podejmowaniu decyzji."""

    CALM = "calm"
    BALANCED = "balanced"
    WATCH = "watch"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(slots=True)
class MarketRegimeAssessment:
    """Result returned by :class:`MarketRegimeClassifier`."""

    regime: MarketRegime
    confidence: float
    risk_score: float
    metrics: Mapping[str, float]
    symbol: str | None = None

    def to_dict(self) -> Mapping[str, float | str | None]:
        """Serialize the assessment to a dictionary for logging/debugging."""

        payload: MutableMapping[str, float | str | None] = {
            "regime": self.regime.value,
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
        }
        payload.update({str(key): float(value) for key, value in self.metrics.items()})
        payload["symbol"] = self.symbol
        return payload


class MarketRegimeClassifier:
    """Heuristic classifier translating OHLCV history into market regimes."""

    def __init__(
        self,
        *,
        min_history: int = 30,
        trend_window: int = 50,
        daily_window: int = 20,
        trend_strength_threshold: float = 0.01,
        momentum_threshold: float = 0.0015,
        volatility_threshold: float = 0.015,
        intraday_threshold: float = 0.02,
        autocorr_threshold: float = -0.2,
        volume_trend_threshold: float = 0.15,
        thresholds_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if min_history < 10:
            raise ValueError("min_history must be at least 10 observations")
        if trend_window <= 1:
            raise ValueError("trend_window must be greater than 1")
        if daily_window <= 1:
            raise ValueError("daily_window must be greater than 1")
        if trend_strength_threshold <= 0:
            raise ValueError("trend_strength_threshold must be positive")
        if momentum_threshold <= 0:
            raise ValueError("momentum_threshold must be positive")
        if volatility_threshold <= 0:
            raise ValueError("volatility_threshold must be positive")
        if intraday_threshold <= 0:
            raise ValueError("intraday_threshold must be positive")
        if volume_trend_threshold <= 0:
            raise ValueError("volume_trend_threshold must be positive")
        self.min_history = int(min_history)
        self.trend_window = int(trend_window)
        self.daily_window = int(daily_window)
        self.trend_strength_threshold = float(trend_strength_threshold)
        self.momentum_threshold = float(momentum_threshold)
        self.volatility_threshold = float(volatility_threshold)
        self.intraday_threshold = float(intraday_threshold)
        self.autocorr_threshold = float(autocorr_threshold)
        self.volume_trend_threshold = float(volume_trend_threshold)
        self._thresholds_loader: Callable[[], Mapping[str, Any]] = (
            thresholds_loader or load_risk_thresholds
        )
        self._thresholds: Mapping[str, Any] = {}
        self.reload_thresholds()

    @property
    def thresholds_loader(self) -> Callable[[], Mapping[str, Any]]:
        """Return the callable used to fetch threshold configuration."""

        return self._thresholds_loader

    def reload_thresholds(self) -> None:
        """Reload risk thresholds from the configured loader."""

        self._thresholds = _ensure_mapping(
            deepcopy(dict(self._thresholds_loader()))
        )

    def thresholds_snapshot(self) -> Mapping[str, Any]:
        """Return a copy of the currently loaded thresholds."""

        return deepcopy(dict(self._thresholds))

    def metrics_config(self) -> Mapping[str, Any]:
        """Expose the sanitised metrics configuration."""

        return deepcopy(dict(self._metrics_config()))

    def risk_score_config(self) -> Mapping[str, Any]:
        """Expose the sanitised risk score configuration."""

        return deepcopy(dict(self._risk_score_config()))

    def _market_regime_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._thresholds.get("market_regime"))

    def _metrics_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._market_regime_config().get("metrics"))

    def _risk_score_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._market_regime_config().get("risk_score"))

    def _risk_level_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._market_regime_config().get("risk_level"))

    def assess(
        self,
        market_data: pd.DataFrame,
        *,
        price_col: str = "close",
        symbol: str | None = None,
    ) -> MarketRegimeAssessment:
        """Return the most likely regime together with risk heuristics."""

        if market_data is None or market_data.empty:
            raise ValueError("market_data must contain price history")
        if price_col not in market_data.columns:
            raise ValueError(f"Column {price_col!r} missing from market data")

        ordered_data = market_data
        if not ordered_data.index.is_monotonic_increasing:
            try:
                ordered_data = ordered_data.sort_index(kind="mergesort")
            except TypeError:
                ordered_data = ordered_data.reset_index(drop=True)

        close = ordered_data[price_col].astype(float)
        returns = close.pct_change(fill_method=None).dropna()
        if returns.size < self.min_history:
            raise ValueError("Not enough observations to classify market regime")

        metrics = self._compute_metrics(ordered_data, close, returns)
        scores = self._score_regimes(metrics)
        regime = max(scores, key=scores.get)
        total = float(sum(scores.values())) or 1.0
        confidence = float(scores[regime] / total)
        risk_score = self._compute_risk_score(metrics)
        return MarketRegimeAssessment(
            regime=regime,
            confidence=confidence,
            risk_score=risk_score,
            metrics=metrics,
            symbol=symbol,
        )

    def _compute_metrics(
        self,
        market_data: pd.DataFrame,
        close: pd.Series,
        returns: pd.Series,
    ) -> Mapping[str, float]:
        market_cfg = self._thresholds.get("market_regime", {})
        if not isinstance(market_cfg, Mapping):
            market_cfg = {}
        metrics_cfg = market_cfg.get("metrics", {})
        if not isinstance(metrics_cfg, Mapping):
            metrics_cfg = {}
        short_span_min = int(metrics_cfg.get("short_span_min", 5))
        short_span_divisor = int(metrics_cfg.get("short_span_divisor", 3))
        long_span_min = int(metrics_cfg.get("long_span_min", 10))
        window = min(self.trend_window, close.size)
        short = close.ewm(span=max(short_span_min, window // short_span_divisor), adjust=False).mean()
        long = close.ewm(span=max(long_span_min, window), adjust=False).mean()
        trend_strength = float(np.abs(short.iloc[-1] - long.iloc[-1]) / (np.abs(long.iloc[-1]) + 1e-12))

        volatility = float(np.nan_to_num(returns.std(), nan=0.0, posinf=0.0, neginf=0.0))
        momentum = float(
            np.nan_to_num(returns.tail(window).mean(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        autocorr_raw = returns.autocorr(lag=1)
        autocorr = float(np.nan_to_num(autocorr_raw if autocorr_raw is not None else 0.0, nan=0.0))

        if {"high", "low"}.issubset(market_data.columns):
            intraday_series = (
                (market_data["high"] - market_data["low"])
                .div(close)
                .rolling(self.daily_window)
                .mean()
            )
            intraday_series = intraday_series.dropna()
            if intraday_series.empty:
                intraday_vol = float(
                    np.nan_to_num(
                        (market_data["high"] - market_data["low"]).div(close).abs().mean(),
                        nan=0.0,
                    )
                )
            else:
                intraday_vol = float(np.nan_to_num(intraday_series.iloc[-1], nan=0.0))
        else:
            intraday_vol = float(
                np.nan_to_num(returns.tail(self.daily_window).abs().mean(), nan=0.0)
            )

        drawdown = float(
            np.nan_to_num((close.cummax() - close).div(close.cummax() + 1e-12).max(), nan=0.0)
        )
        volatility_window = min(max(self.daily_window * 5, self.trend_window), returns.size)
        rolling_vol = returns.rolling(volatility_window, min_periods=max(volatility_window // 2, 10)).std()
        rolling_clean = rolling_vol.dropna()
        baseline_vol = (
            float(np.nan_to_num(rolling_clean.iloc[-1], nan=0.0, posinf=0.0, neginf=0.0))
            if not rolling_clean.empty
            else volatility
        )
        volatility_ratio = float(volatility / (baseline_vol + 1e-12)) if baseline_vol else 1.0

        if "volume" in market_data.columns:
            volume_series = market_data["volume"].astype(float).sort_index()
            short_vol_ma = volume_series.rolling(self.daily_window, min_periods=1).mean()
            long_window = max(self.daily_window * 3, 1)
            long_vol_ma = volume_series.rolling(long_window, min_periods=1).mean()
            volume_trend = float(
                np.nan_to_num(
                    (short_vol_ma.iloc[-1] - long_vol_ma.iloc[-1])
                    / (np.abs(long_vol_ma.iloc[-1]) + 1e-12),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        else:
            volume_trend = 0.0

        skewness = float(np.nan_to_num(returns.skew(), nan=0.0, posinf=0.0, neginf=0.0))
        kurtosis = float(np.nan_to_num(returns.kurt(), nan=0.0, posinf=0.0, neginf=0.0))

        if "volume" in market_data.columns:
            volume_series = market_data["volume"].astype(float).reindex(close.index)
            change = returns.reindex(volume_series.index, method="ffill").fillna(0.0)
            positive_volume = float(
                np.nan_to_num(volume_series.where(change > 0.0).mean(), nan=0.0)
            )
            negative_volume = float(
                np.nan_to_num(volume_series.where(change <= 0.0).mean(), nan=0.0)
            )
            denom = np.abs(positive_volume) + np.abs(negative_volume) + 1e-12
            volume_imbalance = float(np.nan_to_num((positive_volume - negative_volume) / denom))
        else:
            volume_imbalance = 0.0

        metrics: MutableMapping[str, float] = {
            "trend_strength": trend_strength,
            "volatility": volatility,
            "momentum": momentum,
            "autocorr": autocorr,
            "intraday_vol": intraday_vol,
            "drawdown": drawdown,
            "volatility_ratio": volatility_ratio,
            "volume_trend": volume_trend,
            "return_skew": skewness,
            "return_kurtosis": kurtosis,
            "volume_imbalance": volume_imbalance,
        }
        return metrics

    def _score_regimes(self, metrics: Mapping[str, float]) -> Mapping[MarketRegime, float]:
        trend_strength = float(metrics.get("trend_strength", 0.0))
        momentum_metric = float(metrics.get("momentum", 0.0))
        intraday_metric = float(metrics.get("intraday_vol", 0.0))
        autocorr_metric = float(metrics.get("autocorr", 0.0))
        volume_metric = float(metrics.get("volume_trend", 0.0))

        trend_norm = trend_strength / (self.trend_strength_threshold + 1e-12)
        momentum_norm = momentum_metric / (self.momentum_threshold + 1e-12)
        intraday_norm = intraday_metric / (self.intraday_threshold + 1e-12)
        autocorr_norm = -autocorr_metric / (abs(self.autocorr_threshold) + 1e-12)
        volume_norm = max(0.0, volume_metric) / (self.volume_trend_threshold + 1e-12)

        range_bias = max(0.0, 1.0 - min(1.0, abs(trend_norm)))
        balanced_momentum = max(0.0, 1.0 - min(1.0, abs(momentum_norm)))
        intraday_clamped = max(0.0, min(1.0, intraday_norm))

        trend_score = float(
            np.clip(0.6 * max(0.0, min(1.0, trend_norm)) + 0.3 * max(0.0, momentum_norm) + 0.1 * min(1.0, volume_norm), 0.0, 1.0)
        )
        daily_score = float(
            np.clip(0.55 * intraday_clamped + 0.3 * range_bias + 0.15 * balanced_momentum, 0.0, 1.0)
        )
        mean_reversion_score = float(
            np.clip(
                0.55 * max(0.0, min(1.0, autocorr_norm))
                + 0.25 * range_bias
                + 0.2 * max(0.0, 1.0 - intraday_clamped),
                0.0,
                1.0,
            )
        )

        return {
            MarketRegime.TREND: trend_score,
            MarketRegime.DAILY: daily_score,
            MarketRegime.MEAN_REVERSION: mean_reversion_score,
        }

    def _compute_risk_score(self, metrics: Mapping[str, float]) -> float:
        market_cfg = self._thresholds.get("market_regime", {})
        if not isinstance(market_cfg, Mapping):
            market_cfg = {}
        score_cfg = market_cfg.get("risk_score", {})
        if not isinstance(score_cfg, Mapping):
            score_cfg = {}
        volatility_component = min(
            1.0, float(metrics.get("volatility", 0.0)) / self.volatility_threshold
        )
        intraday_component = min(
            1.0,
            float(metrics.get("intraday_vol", 0.0))
            / (self.intraday_threshold * float(score_cfg.get("intraday_multiplier", 1.5))),
        )
        drawdown_component = min(
            1.0,
            float(metrics.get("drawdown", 0.0)) / float(score_cfg.get("drawdown_threshold", 0.2)),
        )
        volatility_ratio_component = min(1.0, float(metrics.get("volatility_ratio", 1.0)))
        volume_component = min(
            1.0,
            abs(float(metrics.get("volume_trend", 0.0))) / self.volume_trend_threshold,
        )
        return float(
            np.clip(
                float(score_cfg.get("volatility_weight", 0.35)) * volatility_component
                + float(score_cfg.get("intraday_weight", 0.25)) * intraday_component
                + float(score_cfg.get("drawdown_weight", 0.2)) * drawdown_component
                + float(score_cfg.get("volatility_mix_weight", 0.2))
                * max(volatility_ratio_component, volume_component),
                0.0,
                1.0,
            )
        )


@dataclass(frozen=True)
class RegimeStrategyWeights:
    """Mapping between market regimes and strategy allocations."""

    weights: Mapping[MarketRegime, Mapping[str, float]]

    @classmethod
    def default(cls) -> "RegimeStrategyWeights":
        return cls(
            weights={
                MarketRegime.TREND: {"trend_following": 0.6, "daily_breakout": 0.3, "mean_reversion": 0.1},
                MarketRegime.DAILY: {"trend_following": 0.2, "daily_breakout": 0.6, "mean_reversion": 0.2},
                MarketRegime.MEAN_REVERSION: {"trend_following": 0.1, "daily_breakout": 0.2, "mean_reversion": 0.7},
            }
        )

    def weights_for(self, regime: MarketRegime, *, normalize: bool = True) -> Dict[str, float]:
        allocation = dict(self.weights.get(regime, {}))
        if not allocation:
            allocation = dict(self.weights.get(MarketRegime.TREND, {}))
        if not allocation:
            return {}
        if not normalize:
            return allocation
        total = float(sum(allocation.values()))
        if total == 0.0:
            return {name: 0.0 for name in allocation}
        return {name: float(value) / total for name, value in allocation.items()}


@dataclass(slots=True)
class RegimeSnapshot:
    """Pojedynczy wpis w historii klasyfikacji rynku."""

    regime: MarketRegime
    confidence: float
    risk_score: float
    drawdown: float = 0.0
    volatility: float = 0.0
    volume_trend: float = 0.0
    volatility_ratio: float = 1.0
    return_skew: float = 0.0
    return_kurtosis: float = 0.0
    volume_imbalance: float = 0.0

    def to_dict(self) -> Mapping[str, float | str]:
        return {
            "regime": self.regime.value,
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
            "drawdown": float(self.drawdown),
            "volatility": float(self.volatility),
            "volume_trend": float(self.volume_trend),
            "volatility_ratio": float(self.volatility_ratio),
            "return_skew": float(self.return_skew),
            "return_kurtosis": float(self.return_kurtosis),
            "volume_imbalance": float(self.volume_imbalance),
        }


@dataclass(slots=True)
class RegimeSummary:
    """Wygładzony obraz rynku obliczony na podstawie ostatnich ocen."""

    regime: MarketRegime
    confidence: float
    risk_score: float
    stability: float
    risk_trend: float
    risk_level: RiskLevel
    risk_volatility: float
    regime_persistence: float
    transition_rate: float
    confidence_trend: float
    confidence_volatility: float
    regime_streak: int
    instability_score: float
    confidence_decay: float
    avg_drawdown: float
    avg_volume_trend: float
    drawdown_pressure: float
    liquidity_pressure: float
    volatility_ratio: float
    regime_entropy: float
    tail_risk_index: float
    shock_frequency: float
    volatility_of_volatility: float
    stress_index: float
    severe_event_rate: float
    cooldown_score: float
    recovery_potential: float
    resilience_score: float
    stress_balance: float
    liquidity_gap: float
    confidence_resilience: float
    stress_projection: float
    stress_momentum: float
    liquidity_trend: float
    confidence_fragility: float
    volatility_trend: float
    drawdown_trend: float
    volume_trend_volatility: float
    stability_projection: float
    degradation_score: float
    skewness_bias: float
    kurtosis_excess: float
    volume_imbalance: float
    distribution_pressure: float
    history: Tuple[RegimeSnapshot, ...] = ()

    def to_dict(self) -> Mapping[str, float | str | Tuple[Mapping[str, float | str], ...]]:
        return {
            "regime": self.regime.value,
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
            "stability": float(self.stability),
            "risk_trend": float(self.risk_trend),
            "risk_level": self.risk_level.value,
            "risk_volatility": float(self.risk_volatility),
            "regime_persistence": float(self.regime_persistence),
            "transition_rate": float(self.transition_rate),
            "confidence_trend": float(self.confidence_trend),
            "confidence_volatility": float(self.confidence_volatility),
            "regime_streak": int(self.regime_streak),
            "instability_score": float(self.instability_score),
            "confidence_decay": float(self.confidence_decay),
            "avg_drawdown": float(self.avg_drawdown),
            "avg_volume_trend": float(self.avg_volume_trend),
            "drawdown_pressure": float(self.drawdown_pressure),
            "liquidity_pressure": float(self.liquidity_pressure),
            "volatility_ratio": float(self.volatility_ratio),
            "regime_entropy": float(self.regime_entropy),
            "tail_risk_index": float(self.tail_risk_index),
            "shock_frequency": float(self.shock_frequency),
            "volatility_of_volatility": float(self.volatility_of_volatility),
            "stress_index": float(self.stress_index),
            "severe_event_rate": float(self.severe_event_rate),
            "cooldown_score": float(self.cooldown_score),
            "recovery_potential": float(self.recovery_potential),
            "resilience_score": float(self.resilience_score),
            "stress_balance": float(self.stress_balance),
            "liquidity_gap": float(self.liquidity_gap),
            "confidence_resilience": float(self.confidence_resilience),
            "stress_projection": float(self.stress_projection),
            "stress_momentum": float(self.stress_momentum),
            "liquidity_trend": float(self.liquidity_trend),
            "confidence_fragility": float(self.confidence_fragility),
            "volatility_trend": float(self.volatility_trend),
            "drawdown_trend": float(self.drawdown_trend),
            "volume_trend_volatility": float(self.volume_trend_volatility),
            "stability_projection": float(self.stability_projection),
            "degradation_score": float(self.degradation_score),
            "skewness_bias": float(self.skewness_bias),
            "kurtosis_excess": float(self.kurtosis_excess),
            "volume_imbalance": float(self.volume_imbalance),
            "distribution_pressure": float(self.distribution_pressure),
            "history": tuple(snapshot.to_dict() for snapshot in self.history),
        }


class RegimeHistory:
    """Utrzymuje okno ruchome klasyfikacji i oblicza wygładzone miary."""

    def __init__(
        self,
        *,
        maxlen: int = 5,
        decay: float = 0.65,
        thresholds_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in the (0, 1] range")
        self.maxlen = int(maxlen)
        self.decay = float(decay)
        self._snapshots: deque[RegimeSnapshot] = deque(maxlen=self.maxlen)
        self._thresholds_loader: Callable[[], Mapping[str, Any]] = (
            thresholds_loader or load_risk_thresholds
        )
        self._thresholds: Mapping[str, Any] = {}
        self.reload_thresholds()

    @property
    def thresholds_loader(self) -> Callable[[], Mapping[str, Any]]:
        """Return the callable used to obtain thresholds."""

        return self._thresholds_loader

    def thresholds_snapshot(self) -> Mapping[str, Any]:
        """Return a copy of the currently active thresholds."""

        return deepcopy(dict(self._thresholds))

    def reload_thresholds(
        self,
        *,
        thresholds: Mapping[str, Any] | None = None,
        loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """Refresh threshold configuration used by history calculations."""

        if thresholds is not None:
            self._thresholds = _ensure_mapping(deepcopy(dict(thresholds)))
            return

        active_loader: Callable[[], Mapping[str, Any]] = loader or self._thresholds_loader
        self._thresholds_loader = active_loader
        self._thresholds = _ensure_mapping(deepcopy(dict(active_loader())))

    def _market_regime_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._thresholds.get("market_regime"))

    def _risk_level_config(self) -> Mapping[str, Any]:
        return _ensure_mapping(self._market_regime_config().get("risk_level"))

    def __len__(self) -> int:  # pragma: no cover - prosta metoda pomocnicza
        return len(self._snapshots)

    def clear(self) -> None:
        self._snapshots.clear()

    def update(self, assessment: MarketRegimeAssessment) -> RegimeSnapshot:
        if not isinstance(assessment, MarketRegimeAssessment):
            raise TypeError("assessment must be a MarketRegimeAssessment instance")
        metrics = dict(assessment.metrics)
        snapshot = RegimeSnapshot(
            regime=assessment.regime,
            confidence=float(assessment.confidence),
            risk_score=float(assessment.risk_score),
            drawdown=float(metrics.get("drawdown", assessment.risk_score)),
            volatility=float(metrics.get("volatility", assessment.risk_score)),
            volume_trend=float(metrics.get("volume_trend", 0.0)),
            volatility_ratio=float(metrics.get("volatility_ratio", 1.0)),
            return_skew=float(metrics.get("return_skew", 0.0)),
            return_kurtosis=float(metrics.get("return_kurtosis", 0.0)),
            volume_imbalance=float(metrics.get("volume_imbalance", 0.0)),
        )
        self._snapshots.append(snapshot)
        return snapshot

    @property
    def snapshots(self) -> Tuple[RegimeSnapshot, ...]:
        return tuple(self._snapshots)

    def reconfigure(
        self,
        *,
        maxlen: int | None = None,
        decay: float | None = None,
        keep_history: bool = True,
    ) -> None:
        """Aktualizuj parametry wygładzania historii.

        Parametr ``maxlen`` kontroluje rozmiar okna deque przechowującego
        migawki reżimu. Zmiana długości może opcjonalnie przyciąć istniejące
        obserwacje (``keep_history=True``) lub wyczyścić stan, jeżeli
        ``keep_history`` ustawiono na ``False``.

        Parametr ``decay`` steruje wagą przypisywaną starszym obserwacjom
        podczas agregacji metryk.
        """

        new_maxlen = self.maxlen if maxlen is None else int(maxlen)
        if new_maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        new_decay = self.decay if decay is None else float(decay)
        if not (0.0 < new_decay <= 1.0):
            raise ValueError("decay must be in the (0, 1] range")

        self.decay = new_decay

        if new_maxlen != self.maxlen:
            preserved: Tuple[RegimeSnapshot, ...]
            if keep_history:
                preserved = tuple(self._snapshots)[-new_maxlen:]
            else:
                preserved = tuple()
            self.maxlen = new_maxlen
            self._snapshots = deque(preserved, maxlen=self.maxlen)
        elif not keep_history:
            self._snapshots.clear()

    def summarise(self) -> RegimeSummary | None:
        if not self._snapshots:
            return None

        weight = 1.0
        counts: MutableMapping[MarketRegime, float] = {}
        dominant_regime: MarketRegime | None = None
        dominant_score = -1.0
        total_weight = 0.0
        risk_total = 0.0
        confidence_total = 0.0
        drawdown_total = 0.0
        volume_trend_total = 0.0
        volatility_ratio_total = 0.0
        skew_total = 0.0
        kurtosis_total = 0.0
        volume_imbalance_total = 0.0

        for snapshot in reversed(self._snapshots):
            counts[snapshot.regime] = counts.get(snapshot.regime, 0.0) + weight
            score = counts[snapshot.regime]
            if score >= dominant_score:
                dominant_regime = snapshot.regime
                dominant_score = score
            risk_total += snapshot.risk_score * weight
            confidence_total += snapshot.confidence * weight
            drawdown_total += snapshot.drawdown * weight
            volume_trend_total += snapshot.volume_trend * weight
            volatility_ratio_total += snapshot.volatility_ratio * weight
            skew_total += snapshot.return_skew * weight
            kurtosis_total += snapshot.return_kurtosis * weight
            volume_imbalance_total += snapshot.volume_imbalance * weight
            total_weight += weight
            weight *= self.decay

        assert dominant_regime is not None  # pragma: no cover - nieosiągalne przy niepustej historii
        if counts:
            distribution = np.array(list(counts.values()), dtype=float)
            probabilities = distribution / (distribution.sum() or 1.0)
            entropy = 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = float(
                    -np.nansum(probabilities * np.log(probabilities + 1e-12))
                )
            max_entropy = float(np.log(len(MarketRegime)) or 1.0)
            regime_entropy = float(
                np.clip(entropy / (max_entropy or 1.0), 0.0, 1.0)
            )
        else:  # pragma: no cover - counts niepuste przy spełnionym asercie
            regime_entropy = 0.0
        weighted_risk = float(risk_total / (total_weight or 1.0))
        avg_confidence = float(confidence_total / (total_weight or 1.0))
        regime_bias = float(np.clip(dominant_score / (total_weight or 1.0), 0.0, 1.0))
        combined_confidence = float(np.clip((avg_confidence + regime_bias) / 2.0, 0.0, 1.0))
        avg_drawdown = float(max(drawdown_total / (total_weight or 1.0), 0.0))
        avg_volume_trend = float(volume_trend_total / (total_weight or 1.0))
        avg_volatility_ratio = float(max(volatility_ratio_total / (total_weight or 1.0), 0.0))
        avg_skewness = float(skew_total / (total_weight or 1.0))
        avg_kurtosis = float(kurtosis_total / (total_weight or 1.0))
        avg_volume_imbalance = float(volume_imbalance_total / (total_weight or 1.0))

        snapshots = self.snapshots
        risk_values = [snapshot.risk_score for snapshot in snapshots]
        confidence_values = [snapshot.confidence for snapshot in snapshots]
        volatility_values = [snapshot.volatility for snapshot in snapshots]
        drawdown_values = [snapshot.drawdown for snapshot in snapshots]
        volume_trend_values = [snapshot.volume_trend for snapshot in snapshots]
        skew_values = [snapshot.return_skew for snapshot in snapshots]
        kurtosis_values = [snapshot.return_kurtosis for snapshot in snapshots]
        volume_imbalance_values = [snapshot.volume_imbalance for snapshot in snapshots]
        tail_hits = 0
        for snapshot in snapshots:
            if (
                snapshot.drawdown >= 0.22
                or snapshot.volatility >= 0.035
                or snapshot.volatility_ratio >= 1.4
            ):
                tail_hits += 1
        if len(snapshots) >= 2:
            oldest_risk = snapshots[0].risk_score
            latest_risk = snapshots[-1].risk_score
            risk_trend = float(np.clip(latest_risk - oldest_risk, -1.0, 1.0))
            deltas = [
                0.0
                if snapshots[idx].regime == snapshots[idx - 1].regime
                else 1.0
                for idx in range(1, len(snapshots))
            ]
            regime_persistence = float(1.0 - (sum(deltas) / (len(deltas) or 1.0)))
            risk_volatility = float(np.std(risk_values, dtype=float))
            confidence_trend = float(
                np.clip(confidence_values[-1] - confidence_values[0], -1.0, 1.0)
            )
            confidence_volatility = float(np.std(confidence_values, dtype=float))
            risk_deltas = [
                abs(risk_values[idx] - risk_values[idx - 1])
                for idx in range(1, len(risk_values))
            ]
            shock_events = sum(1 for delta in risk_deltas if delta >= 0.12)
            shock_events += sum(
                1
                for idx in range(1, len(snapshots))
                if snapshots[idx].regime is not snapshots[idx - 1].regime
            )
            shock_frequency = float(
                np.clip(shock_events / (max(len(snapshots) - 1, 1)), 0.0, 1.0)
            )
            volatility_of_volatility = float(np.std(volatility_values, dtype=float))
        else:
            risk_trend = 0.0
            regime_persistence = 1.0
            risk_volatility = 0.0
            confidence_trend = 0.0
            confidence_volatility = 0.0
            shock_frequency = 0.0
            volatility_of_volatility = 0.0
        regime_persistence = float(np.clip(regime_persistence, 0.0, 1.0))
        transition_rate = float(np.clip(1.0 - regime_persistence, 0.0, 1.0))
        risk_volatility = float(max(risk_volatility, 0.0))
        confidence_volatility = float(max(confidence_volatility, 0.0))
        confidence_decay = float(max(0.0, -confidence_trend))
        risk_vol_norm = float(np.clip(risk_volatility / 0.3, 0.0, 1.0))
        conf_vol_norm = float(np.clip(confidence_volatility / 0.2, 0.0, 1.0))
        tail_risk_index = float(np.clip(tail_hits / (len(snapshots) or 1.0), 0.0, 1.0))
        volatility_of_volatility = float(max(volatility_of_volatility, 0.0))
        vol_of_vol_norm = float(np.clip(volatility_of_volatility / 0.025, 0.0, 1.0))
        shock_frequency = float(max(shock_frequency, 0.0))
        instability_score = float(
            np.clip(
                0.4 * risk_vol_norm
                + 0.3 * transition_rate
                + 0.2 * conf_vol_norm
                + 0.1 * min(confidence_decay, 1.0),
                0.0,
                1.0,
            )
        )
        stress_index = float(
            np.clip(
                0.45 * tail_risk_index
                + 0.3 * shock_frequency
                + 0.25 * max(risk_vol_norm, vol_of_vol_norm),
                0.0,
                1.0,
            )
        )
        severe_events = sum(
            1
            for snapshot in snapshots
            if (
                snapshot.risk_score >= 0.75
                or snapshot.drawdown >= 0.22
                or snapshot.volatility >= 0.035
                or snapshot.volatility_ratio >= 1.45
            )
        )
        severe_event_rate = float(
            np.clip(severe_events / (len(snapshots) or 1.0), 0.0, 1.0)
        )
        drawdown_pressure = float(np.clip(avg_drawdown / 0.25, 0.0, 1.0))
        liquidity_pressure = float(
            np.clip(
                max(0.0, -avg_volume_trend) / 0.4
                + max(0.0, avg_volatility_ratio - 1.0) * 0.35
                + max(0.0, instability_score - 0.4) * 0.2,
                0.0,
                1.0,
            )
        )
        recovery_potential = float(
            np.clip(
                0.45 * max(0.0, -risk_trend)
                + 0.25 * max(0.0, combined_confidence - 0.5)
                + 0.2 * max(0.0, regime_persistence - 0.55)
                + 0.15 * max(0.0, 0.4 - risk_volatility)
                + 0.15 * max(0.0, 0.35 - drawdown_pressure)
                + 0.15 * max(0.0, 0.35 - liquidity_pressure),
                0.0,
                1.0,
            )
        )
        cooldown_score = float(
            np.clip(
                0.4 * severe_event_rate
                + 0.25 * stress_index
                + 0.2 * max(0.0, instability_score - 0.5)
                + 0.15 * max(0.0, confidence_decay)
                + 0.15 * max(0.0, drawdown_pressure - 0.5)
                + 0.15 * max(0.0, liquidity_pressure - 0.5),
                0.0,
                1.0,
            )
        )

        liquidity_gap = float(
            np.clip(
                0.45 * liquidity_pressure
                + 0.25 * np.clip(max(0.0, -avg_volume_trend) / 0.35, 0.0, 1.0)
                + 0.2 * np.clip(abs(avg_volume_imbalance) / 0.6, 0.0, 1.0)
                + 0.1 * max(0.0, instability_score - 0.45)
                + 0.1 * max(0.0, avg_volatility_ratio - 1.0)
                - 0.2 * recovery_potential,
                0.0,
                1.0,
            )
        )

        if len(volatility_values) >= 2:
            volatility_trend = float(
                np.clip(volatility_values[-1] - volatility_values[0], -0.05, 0.05)
            )
        else:
            volatility_trend = 0.0
        if len(drawdown_values) >= 2:
            drawdown_trend = float(
                np.clip(drawdown_values[-1] - drawdown_values[0], -0.4, 0.4)
            )
        else:
            drawdown_trend = 0.0
        if len(volume_trend_values) >= 2:
            volume_trend_volatility = float(
                np.std(volume_trend_values, dtype=float)
            )
        else:
            volume_trend_volatility = 0.0
        volume_trend_volatility = float(max(volume_trend_volatility, 0.0))
        if len(skew_values) >= 2:
            skew_trend = float(np.clip(skew_values[-1] - skew_values[0], -3.0, 3.0))
        else:
            skew_trend = 0.0
        if len(kurtosis_values) >= 2:
            kurtosis_trend = float(np.clip(kurtosis_values[-1] - kurtosis_values[0], -6.0, 6.0))
        else:
            kurtosis_trend = 0.0
        if len(volume_imbalance_values) >= 2:
            volume_imbalance_trend = float(
                np.clip(volume_imbalance_values[-1] - volume_imbalance_values[0], -1.0, 1.0)
            )
        else:
            volume_imbalance_trend = 0.0

        skewness_bias = float(np.clip(avg_skewness, -5.0, 5.0))
        kurtosis_excess = float(np.clip(avg_kurtosis, -5.0, 10.0))
        volume_imbalance = float(np.clip(avg_volume_imbalance, -1.0, 1.0))
        risk_level_cfg = self._risk_level_config()
        scales = _ensure_mapping(risk_level_cfg.get("scales"))
        skew_scale = float(scales.get("skewness_bias", 1.5)) or 1.5
        kurtosis_scale = float(scales.get("kurtosis_excess", 3.0)) or 3.0
        volume_scale = float(scales.get("volume_imbalance", 0.6)) or 0.6
        skew_pressure = float(np.clip(abs(skewness_bias) / skew_scale, 0.0, 1.0))
        kurtosis_pressure = float(
            np.clip(max(0.0, kurtosis_excess) / kurtosis_scale, 0.0, 1.0)
        )
        volume_imbalance_pressure = float(
            np.clip(abs(volume_imbalance) / volume_scale, 0.0, 1.0)
        )
        distribution_pressure = float(
            np.clip(
                0.28 * skew_pressure
                + 0.28 * kurtosis_pressure
                + 0.18 * volume_imbalance_pressure
                + 0.08 * max(0.0, instability_score - 0.45)
                + 0.08 * max(0.0, abs(skew_trend) / 2.5)
                + 0.06 * max(0.0, max(0.0, kurtosis_trend) / 4.0)
                + 0.04 * max(0.0, abs(volume_imbalance_trend)),
                0.0,
                1.0,
            )
        )
        volatility_trend_up = float(max(0.0, volatility_trend))
        drawdown_trend_up = float(max(0.0, drawdown_trend))
        volatility_trend_intensity = float(np.clip(volatility_trend_up / 0.03, 0.0, 1.0))
        drawdown_trend_intensity = float(np.clip(drawdown_trend_up / 0.18, 0.0, 1.0))
        volume_trend_volatility_norm = float(
            np.clip(volume_trend_volatility / 0.25, 0.0, 1.0)
        )

        degradation_score = float(
            np.clip(
                0.35 * volatility_trend_intensity
                + 0.35 * drawdown_trend_intensity
                + 0.2 * volume_trend_volatility_norm
                + 0.1 * max(0.0, instability_score - 0.45)
                + 0.1 * max(0.0, tail_risk_index - 0.4)
                + 0.1 * max(0.0, stress_index - 0.45),
                0.0,
                1.0,
            )
        )
        stability_projection = float(
            np.clip(
                0.45 * regime_persistence
                + 0.25 * max(0.0, 1.0 - transition_rate)
                + 0.2 * max(0.0, 1.0 - risk_vol_norm)
                + 0.15 * recovery_potential
                - 0.25 * volatility_trend_intensity
                - 0.2 * drawdown_trend_intensity
                - 0.15 * volume_trend_volatility_norm,
                0.0,
                1.0,
            )
        )

        stress_balance = float(
            np.clip(0.5 + 0.5 * (recovery_potential - stress_index), 0.0, 1.0)
        )
        resilience_score = float(
            np.clip(
                0.3 * recovery_potential
                + 0.25 * stability_projection
                + 0.2 * max(0.0, 1.0 - drawdown_pressure)
                + 0.12 * max(0.0, 1.0 - liquidity_pressure)
                + 0.08 * max(0.0, 1.0 - liquidity_gap)
                + 0.05 * max(0.0, 1.0 - confidence_decay)
                + 0.05 * max(0.0, 1.0 - distribution_pressure),
                0.0,
                1.0,
            )
        )

        confidence_resilience = float(
            np.clip(
                0.35 * combined_confidence
                + 0.2 * max(0.0, 1.0 - confidence_decay)
                + 0.15 * max(0.0, 1.0 - conf_vol_norm)
                + 0.15 * max(0.0, confidence_trend + 0.15)
                + 0.15 * resilience_score
                - 0.1 * distribution_pressure
                - 0.1 * liquidity_gap,
                0.0,
                1.0,
            )
        )

        stress_projection = float(
            np.clip(
                0.35 * stress_index
                + 0.25 * degradation_score
                + 0.15 * tail_risk_index
                + 0.1 * shock_frequency
                + 0.1 * distribution_pressure
                + 0.1 * liquidity_gap
                - 0.2 * recovery_potential
                - 0.1 * resilience_score,
                0.0,
                1.0,
            )
        )

        stress_momentum = float(
            np.clip(
                0.4 * stress_index
                + 0.3 * stress_projection
                + 0.2 * tail_risk_index
                + 0.15 * shock_frequency
                + 0.1 * max(0.0, risk_trend)
                - 0.2 * recovery_potential,
                0.0,
                1.0,
            )
        )
        liquidity_trend_component = float(
            np.clip(max(0.0, -avg_volume_trend) / 0.35, 0.0, 1.0)
        )
        liquidity_trend = float(
            np.clip(
                0.5 * liquidity_pressure
                + 0.3 * liquidity_gap
                + 0.2 * liquidity_trend_component
                + 0.1 * max(0.0, volume_trend_volatility_norm - 0.4)
                - 0.2 * recovery_potential,
                0.0,
                1.0,
            )
        )
        confidence_fragility = float(
            np.clip(
                0.35 * conf_vol_norm
                + 0.3 * confidence_decay
                + 0.2 * max(0.0, -confidence_trend)
                + 0.1 * regime_entropy
                + 0.1 * distribution_pressure,
                0.0,
                1.0,
            )
        )

        regime_streak = 0
        latest_regime = snapshots[-1].regime
        for snapshot in reversed(snapshots):
            if snapshot.regime is latest_regime:
                regime_streak += 1
            else:
                break

        risk_level = self._resolve_risk_level(
            weighted_risk,
            risk_trend,
            regime_bias,
            combined_confidence,
            risk_volatility,
            regime_persistence,
            transition_rate,
            confidence_trend,
            confidence_volatility,
            regime_streak,
            instability_score,
            confidence_decay,
            drawdown_pressure,
            liquidity_pressure,
            avg_volatility_ratio,
            tail_risk_index,
            shock_frequency,
            volatility_of_volatility,
            stress_index,
            severe_event_rate,
            cooldown_score,
            recovery_potential,
            volatility_trend,
            drawdown_trend,
            volume_trend_volatility,
            stability_projection,
            degradation_score,
            skewness_bias,
            kurtosis_excess,
            volume_imbalance,
            distribution_pressure,
            regime_entropy,
            resilience_score,
            stress_balance,
            liquidity_gap,
            confidence_resilience,
            stress_projection,
            stress_momentum,
            liquidity_trend,
            confidence_fragility,
        )

        return RegimeSummary(
            regime=dominant_regime,
            confidence=combined_confidence,
            risk_score=weighted_risk,
            stability=regime_bias,
            risk_trend=risk_trend,
            risk_level=risk_level,
            risk_volatility=risk_volatility,
            regime_persistence=regime_persistence,
            transition_rate=transition_rate,
            confidence_trend=confidence_trend,
            confidence_volatility=confidence_volatility,
            regime_streak=regime_streak,
            instability_score=instability_score,
            confidence_decay=confidence_decay,
            avg_drawdown=avg_drawdown,
            avg_volume_trend=avg_volume_trend,
            drawdown_pressure=drawdown_pressure,
            liquidity_pressure=liquidity_pressure,
            volatility_ratio=avg_volatility_ratio,
            regime_entropy=regime_entropy,
            tail_risk_index=tail_risk_index,
            shock_frequency=shock_frequency,
            volatility_of_volatility=volatility_of_volatility,
            stress_index=stress_index,
            severe_event_rate=severe_event_rate,
            cooldown_score=cooldown_score,
            recovery_potential=recovery_potential,
            resilience_score=resilience_score,
            stress_balance=stress_balance,
            liquidity_gap=liquidity_gap,
            confidence_resilience=confidence_resilience,
            stress_projection=stress_projection,
            stress_momentum=stress_momentum,
            liquidity_trend=liquidity_trend,
            confidence_fragility=confidence_fragility,
            volatility_trend=volatility_trend,
            drawdown_trend=drawdown_trend,
            volume_trend_volatility=volume_trend_volatility,
            stability_projection=stability_projection,
            degradation_score=degradation_score,
            skewness_bias=skewness_bias,
            kurtosis_excess=kurtosis_excess,
            volume_imbalance=volume_imbalance,
            distribution_pressure=distribution_pressure,
            history=snapshots,
        )

    def _resolve_risk_level(
        self,
        risk_score: float,
        risk_trend: float,
        stability: float,
        confidence: float,
        risk_volatility: float,
        regime_persistence: float,
        transition_rate: float,
        confidence_trend: float,
        confidence_volatility: float,
        regime_streak: int,
        instability_score: float,
        confidence_decay: float,
        drawdown_pressure: float,
        liquidity_pressure: float,
        volatility_ratio: float,
        tail_risk_index: float,
        shock_frequency: float,
        volatility_of_volatility: float,
        stress_index: float,
        severe_event_rate: float,
        cooldown_score: float,
        recovery_potential: float,
        volatility_trend: float,
        drawdown_trend: float,
        volume_trend_volatility: float,
        stability_projection: float,
        degradation_score: float,
        skewness_bias: float,
        kurtosis_excess: float,
        volume_imbalance: float,
        distribution_pressure: float,
        regime_entropy: float,
        resilience_score: float,
        stress_balance: float,
        liquidity_gap: float,
        confidence_resilience: float,
        stress_projection: float,
        stress_momentum: float,
        liquidity_trend: float,
        confidence_fragility: float,
    ) -> RiskLevel:
        """Przypisz poziom ryzyka na bazie zagregowanych metryk."""

        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        risk_trend = float(np.clip(risk_trend, -1.0, 1.0))
        stability = float(np.clip(stability, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        risk_volatility = float(max(risk_volatility, 0.0))
        regime_persistence = float(np.clip(regime_persistence, 0.0, 1.0))
        transition_rate = float(np.clip(transition_rate, 0.0, 1.0))
        confidence_trend = float(np.clip(confidence_trend, -1.0, 1.0))
        confidence_volatility = float(max(confidence_volatility, 0.0))
        regime_streak = int(max(regime_streak, 0))
        instability_score = float(np.clip(instability_score, 0.0, 1.0))
        confidence_decay = float(np.clip(confidence_decay, 0.0, 1.0))
        drawdown_pressure = float(np.clip(drawdown_pressure, 0.0, 1.0))
        liquidity_pressure = float(np.clip(liquidity_pressure, 0.0, 1.0))
        volatility_ratio = float(max(volatility_ratio, 0.0))
        tail_risk_index = float(np.clip(tail_risk_index, 0.0, 1.0))
        shock_frequency = float(np.clip(shock_frequency, 0.0, 1.0))
        volatility_of_volatility = float(max(volatility_of_volatility, 0.0))
        stress_index = float(np.clip(stress_index, 0.0, 1.0))
        severe_event_rate = float(np.clip(severe_event_rate, 0.0, 1.0))
        cooldown_score = float(np.clip(cooldown_score, 0.0, 1.0))
        recovery_potential = float(np.clip(recovery_potential, 0.0, 1.0))
        volatility_trend = float(np.clip(volatility_trend, -1.0, 1.0))
        drawdown_trend = float(np.clip(drawdown_trend, -1.0, 1.0))
        volume_trend_volatility = float(max(volume_trend_volatility, 0.0))
        stability_projection = float(np.clip(stability_projection, 0.0, 1.0))
        degradation_score = float(np.clip(degradation_score, 0.0, 1.0))
        skewness_bias = float(np.clip(skewness_bias, -5.0, 5.0))
        kurtosis_excess = float(np.clip(kurtosis_excess, -5.0, 10.0))
        volume_imbalance = float(np.clip(volume_imbalance, -1.0, 1.0))
        distribution_pressure = float(np.clip(distribution_pressure, 0.0, 1.0))
        regime_entropy = float(np.clip(regime_entropy, 0.0, 1.0))
        resilience_score = float(np.clip(resilience_score, 0.0, 1.0))
        stress_balance = float(np.clip(stress_balance, 0.0, 1.0))
        liquidity_gap = float(np.clip(liquidity_gap, 0.0, 1.0))
        confidence_resilience = float(np.clip(confidence_resilience, 0.0, 1.0))
        stress_projection = float(np.clip(stress_projection, 0.0, 1.0))
        stress_momentum = float(np.clip(stress_momentum, 0.0, 1.0))
        liquidity_trend = float(np.clip(liquidity_trend, 0.0, 1.0))
        confidence_fragility = float(np.clip(confidence_fragility, 0.0, 1.0))
        vol_trend_intensity = float(max(0.0, volatility_trend))
        drawdown_trend_intensity = float(max(0.0, drawdown_trend))
        risk_level_cfg = self._risk_level_config()
        scales = _ensure_mapping(risk_level_cfg.get("scales"))
        skew_scale = float(scales.get("skewness_bias", 1.5)) or 1.5
        kurtosis_scale = float(scales.get("kurtosis_excess", 3.0)) or 3.0
        volume_scale = float(scales.get("volume_imbalance", 0.6)) or 0.6
        skew_pressure = float(np.clip(abs(skewness_bias) / skew_scale, 0.0, 1.0))
        kurtosis_pressure = float(
            np.clip(max(0.0, kurtosis_excess) / kurtosis_scale, 0.0, 1.0)
        )
        volume_imbalance_pressure = float(
            np.clip(abs(volume_imbalance) / volume_scale, 0.0, 1.0)
        )

        critical = _ensure_mapping(risk_level_cfg.get("critical"))
        if (
            risk_score >= float(critical.get("risk_score", 0.85))
            or risk_trend >= float(critical.get("risk_trend", 0.25))
            or (
                instability_score >= float(critical.get("instability_score", 0.85))
                and (
                    risk_score >= float(critical.get("instability_risk_score", 0.55))
                    or transition_rate >= float(critical.get("transition_rate", 0.7))
                )
            )
            or drawdown_pressure >= float(critical.get("drawdown_pressure", 0.9))
            or (
                drawdown_pressure >= float(critical.get("drawdown_pressure_support", 0.75))
                and risk_score >= float(critical.get("drawdown_risk_score", 0.55))
            )
            or (
                liquidity_pressure >= float(critical.get("liquidity_pressure", 0.85))
                and risk_score >= float(critical.get("liquidity_risk_score", 0.5))
            )
            or stress_index >= float(critical.get("stress_index", 0.85))
            or (
                tail_risk_index >= float(critical.get("tail_risk_index", 0.7))
                and shock_frequency >= float(critical.get("shock_frequency", 0.6))
            )
            or (
                volatility_of_volatility
                >= float(critical.get("volatility_of_volatility", 0.035))
                and risk_score >= float(critical.get("volatility_risk_score", 0.55))
            )
            or cooldown_score >= float(critical.get("cooldown_score", 0.75))
            or severe_event_rate >= float(critical.get("severe_event_rate", 0.6))
            or degradation_score >= float(critical.get("degradation_score", 0.75))
            or distribution_pressure >= float(critical.get("distribution_pressure", 0.8))
            or regime_entropy >= float(critical.get("regime_entropy", 0.85))
            or resilience_score <= float(critical.get("resilience_score", 0.2))
            or stress_balance <= float(critical.get("stress_balance", 0.2))
            or liquidity_gap >= float(critical.get("liquidity_gap", 0.85))
            or stress_projection >= float(critical.get("stress_projection", 0.8))
            or stress_momentum >= float(critical.get("stress_momentum", 0.8))
            or liquidity_trend >= float(critical.get("liquidity_trend", 0.85))
            or confidence_fragility >= float(critical.get("confidence_fragility", 0.8))
            or (
                confidence_resilience
                <= float(critical.get("confidence_resilience", 0.25))
                and risk_score >= float(critical.get("liquidity_risk_score", 0.5))
            )
            or (
                stability_projection
                <= float(critical.get("stability_projection", 0.2))
                and (
                    risk_score >= float(critical.get("drawdown_risk_score", 0.55))
                    or instability_score >= float(critical.get("instability_support", 0.6))
                )
            )
            or (
                vol_trend_intensity >= float(critical.get("vol_trend_intensity", 0.025))
                and risk_score >= float(critical.get("vol_trend_risk_score", 0.6))
            )
            or (
                drawdown_trend_intensity
                >= float(critical.get("drawdown_trend_intensity", 0.12))
                and risk_score >= float(critical.get("vol_trend_risk_score", 0.6))
            )
            or (
                skew_pressure >= float(critical.get("skew_pressure", 0.8))
                and risk_score >= float(critical.get("pressure_risk_score", 0.55))
            )
            or (
                kurtosis_pressure >= float(critical.get("kurtosis_pressure", 0.8))
                and risk_score >= float(critical.get("pressure_risk_score", 0.55))
            )
            or (
                volume_imbalance_pressure
                >= float(critical.get("volume_imbalance_pressure", 0.85))
                and liquidity_pressure
                >= float(critical.get("liquidity_pressure_support", 0.45))
            )
        ):
            return RiskLevel.CRITICAL
        elevated = _ensure_mapping(risk_level_cfg.get("elevated"))
        if (
            risk_score >= float(elevated.get("risk_score", 0.65))
            or risk_trend >= float(elevated.get("risk_trend", 0.08))
            or (
                risk_volatility >= float(elevated.get("risk_volatility", 0.2))
                and risk_score >= float(elevated.get("risk_support_score", 0.5))
            )
            or (
                confidence_volatility
                >= float(elevated.get("confidence_volatility", 0.2))
                and risk_score >= float(elevated.get("risk_support_score", 0.5))
            )
            or (
                confidence_trend <= float(elevated.get("confidence_trend", -0.3))
                and risk_score >= float(elevated.get("risk_support_score", 0.5))
            )
            or instability_score >= float(elevated.get("instability_score", 0.7))
            or (
                transition_rate >= float(elevated.get("transition_rate", 0.6))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or drawdown_pressure >= float(elevated.get("drawdown_pressure", 0.6))
            or (
                liquidity_pressure >= float(elevated.get("liquidity_pressure", 0.6))
                and risk_score >= float(elevated.get("liquidity_risk_score", 0.35))
            )
            or volatility_ratio >= float(elevated.get("volatility_ratio", 1.45))
            or stress_index >= float(elevated.get("stress_index", 0.65))
            or tail_risk_index >= float(elevated.get("tail_risk_index", 0.55))
            or shock_frequency >= float(elevated.get("shock_frequency", 0.55))
            or volatility_of_volatility
            >= float(elevated.get("volatility_of_volatility", 0.03))
            or cooldown_score >= float(elevated.get("cooldown_score", 0.55))
            or severe_event_rate >= float(elevated.get("severe_event_rate", 0.45))
            or degradation_score >= float(elevated.get("degradation_score", 0.55))
            or distribution_pressure >= float(elevated.get("distribution_pressure", 0.6))
            or regime_entropy >= float(elevated.get("regime_entropy", 0.75))
            or resilience_score <= float(elevated.get("resilience_score", 0.35))
            or stress_balance <= float(elevated.get("stress_balance", 0.35))
            or liquidity_gap >= float(elevated.get("liquidity_gap", 0.6))
            or stress_projection >= float(elevated.get("stress_projection", 0.6))
            or stress_momentum >= float(elevated.get("stress_momentum", 0.6))
            or liquidity_trend >= float(elevated.get("liquidity_trend", 0.65))
            or confidence_fragility >= float(elevated.get("confidence_fragility", 0.6))
            or confidence_resilience
            <= float(elevated.get("confidence_resilience", 0.35))
            or stability_projection <= float(elevated.get("stability_projection", 0.35))
            or (
                vol_trend_intensity
                >= float(elevated.get("vol_trend_intensity", 0.018))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or (
                drawdown_trend_intensity
                >= float(elevated.get("drawdown_trend_intensity", 0.08))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or (
                volume_trend_volatility
                >= float(elevated.get("volume_trend_volatility", 0.18))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or (
                skew_pressure >= float(elevated.get("skew_pressure", 0.6))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or (
                kurtosis_pressure >= float(elevated.get("kurtosis_pressure", 0.6))
                and risk_score >= float(elevated.get("transition_risk_score", 0.45))
            )
            or (
                volume_imbalance_pressure
                >= float(elevated.get("volume_imbalance_pressure", 0.65))
                and liquidity_pressure
                >= float(elevated.get("liquidity_pressure_support", 0.4))
            )
        ):
            return RiskLevel.ELEVATED
        calm = _ensure_mapping(risk_level_cfg.get("calm"))
        if (
            risk_score <= float(calm.get("risk_score", 0.25))
            and risk_trend <= float(calm.get("risk_trend", 0.0))
            and stability >= float(calm.get("stability", 0.55))
            and confidence >= float(calm.get("confidence", 0.5))
            and risk_volatility <= float(calm.get("risk_volatility", 0.12))
            and regime_persistence >= float(calm.get("regime_persistence", 0.5))
            and confidence_trend >= float(calm.get("confidence_trend", -0.05))
            and confidence_volatility <= float(calm.get("confidence_volatility", 0.1))
            and instability_score <= float(calm.get("instability_score", 0.35))
            and transition_rate <= float(calm.get("transition_rate", 0.35))
            and drawdown_pressure <= float(calm.get("drawdown_pressure", 0.35))
            and liquidity_pressure <= float(calm.get("liquidity_pressure", 0.4))
            and volatility_ratio <= float(calm.get("volatility_ratio", 1.15))
            and stress_index <= float(calm.get("stress_index", 0.35))
            and tail_risk_index <= float(calm.get("tail_risk_index", 0.3))
            and shock_frequency <= float(calm.get("shock_frequency", 0.35))
            and volatility_of_volatility <= float(calm.get("volatility_of_volatility", 0.02))
            and cooldown_score <= float(calm.get("cooldown_score", 0.35))
            and severe_event_rate <= float(calm.get("severe_event_rate", 0.35))
            and recovery_potential >= float(calm.get("recovery_potential", 0.3))
            and degradation_score <= float(calm.get("degradation_score", 0.3))
            and stability_projection >= float(calm.get("stability_projection", 0.45))
            and distribution_pressure <= float(calm.get("distribution_pressure", 0.35))
            and regime_entropy <= float(calm.get("regime_entropy", 0.55))
            and resilience_score >= float(calm.get("resilience_score", 0.55))
            and stress_balance >= float(calm.get("stress_balance", 0.5))
            and liquidity_gap <= float(calm.get("liquidity_gap", 0.35))
            and stress_projection <= float(calm.get("stress_projection", 0.35))
            and stress_momentum <= float(calm.get("stress_momentum", 0.35))
            and liquidity_trend <= float(calm.get("liquidity_trend", 0.35))
            and confidence_fragility <= float(calm.get("confidence_fragility", 0.35))
            and confidence_resilience >= float(calm.get("confidence_resilience", 0.55))
            and skew_pressure <= float(calm.get("skew_pressure", 0.45))
            and kurtosis_pressure <= float(calm.get("kurtosis_pressure", 0.45))
            and volume_imbalance_pressure
            <= float(calm.get("volume_imbalance_pressure", 0.45))
        ):
            return RiskLevel.CALM
        balanced = _ensure_mapping(risk_level_cfg.get("balanced"))
        if (
            risk_score <= float(balanced.get("risk_score", 0.45))
            and risk_trend <= float(balanced.get("risk_trend", 0.05))
            and risk_volatility <= float(balanced.get("risk_volatility", 0.18))
            and regime_persistence >= float(balanced.get("regime_persistence", 0.35))
            and confidence_trend >= float(balanced.get("confidence_trend", -0.15))
            and transition_rate <= float(balanced.get("transition_rate", 0.55))
            and instability_score <= float(balanced.get("instability_score", 0.6))
            and drawdown_pressure <= float(balanced.get("drawdown_pressure", 0.55))
            and liquidity_pressure <= float(balanced.get("liquidity_pressure", 0.55))
            and volatility_ratio <= float(balanced.get("volatility_ratio", 1.35))
            and stress_index <= float(balanced.get("stress_index", 0.5))
            and tail_risk_index <= float(balanced.get("tail_risk_index", 0.45))
            and shock_frequency <= float(balanced.get("shock_frequency", 0.45))
            and volatility_of_volatility
            <= float(balanced.get("volatility_of_volatility", 0.028))
            and cooldown_score <= float(balanced.get("cooldown_score", 0.5))
            and severe_event_rate <= float(balanced.get("severe_event_rate", 0.45))
            and degradation_score <= float(balanced.get("degradation_score", 0.45))
            and stability_projection >= float(balanced.get("stability_projection", 0.4))
            and distribution_pressure <= float(balanced.get("distribution_pressure", 0.5))
            and regime_entropy <= float(balanced.get("regime_entropy", 0.7))
            and resilience_score >= float(balanced.get("resilience_score", 0.4))
            and stress_balance >= float(balanced.get("stress_balance", 0.4))
            and liquidity_gap <= float(balanced.get("liquidity_gap", 0.5))
            and stress_projection <= float(balanced.get("stress_projection", 0.5))
            and stress_momentum <= float(balanced.get("stress_momentum", 0.5))
            and liquidity_trend <= float(balanced.get("liquidity_trend", 0.5))
            and confidence_fragility <= float(balanced.get("confidence_fragility", 0.5))
            and confidence_resilience >= float(balanced.get("confidence_resilience", 0.45))
            and skew_pressure <= float(balanced.get("skew_pressure", 0.6))
            and kurtosis_pressure <= float(balanced.get("kurtosis_pressure", 0.6))
            and volume_imbalance_pressure
            <= float(balanced.get("volume_imbalance_pressure", 0.55))
        ):
            return RiskLevel.BALANCED
        watch = _ensure_mapping(risk_level_cfg.get("watch"))
        if (
            confidence_volatility >= float(watch.get("confidence_volatility", 0.18))
            or (
                confidence_trend <= float(watch.get("confidence_trend", -0.2))
                and regime_streak <= int(watch.get("regime_streak", 2))
            )
            or transition_rate >= float(watch.get("transition_rate", 0.4))
            or instability_score >= float(watch.get("instability_score", 0.5))
            or confidence_decay >= float(watch.get("confidence_decay", 0.2))
            or drawdown_pressure >= float(watch.get("drawdown_pressure", 0.45))
            or liquidity_pressure >= float(watch.get("liquidity_pressure", 0.5))
            or volatility_ratio >= float(watch.get("volatility_ratio", 1.25))
            or tail_risk_index >= float(watch.get("tail_risk_index", 0.4))
            or shock_frequency >= float(watch.get("shock_frequency", 0.4))
            or stress_index >= float(watch.get("stress_index", 0.45))
            or volatility_of_volatility
            >= float(watch.get("volatility_of_volatility", 0.024))
            or cooldown_score >= float(watch.get("cooldown_score", 0.45))
            or severe_event_rate >= float(watch.get("severe_event_rate", 0.4))
            or degradation_score >= float(watch.get("degradation_score", 0.4))
            or stability_projection <= float(watch.get("stability_projection", 0.45))
            or vol_trend_intensity >= float(watch.get("vol_trend_intensity", 0.015))
            or drawdown_trend_intensity
            >= float(watch.get("drawdown_trend_intensity", 0.05))
            or distribution_pressure >= float(watch.get("distribution_pressure", 0.45))
            or regime_entropy >= float(watch.get("regime_entropy", 0.6))
            or resilience_score <= float(watch.get("resilience_score", 0.45))
            or stress_balance <= float(watch.get("stress_balance", 0.45))
            or liquidity_gap >= float(watch.get("liquidity_gap", 0.45))
            or stress_projection >= float(watch.get("stress_projection", 0.45))
            or confidence_resilience <= float(watch.get("confidence_resilience", 0.5))
            or stress_momentum >= float(watch.get("stress_momentum", 0.45))
            or liquidity_trend >= float(watch.get("liquidity_trend", 0.5))
            or confidence_fragility >= float(watch.get("confidence_fragility", 0.5))
            or skew_pressure >= float(watch.get("skew_pressure", 0.55))
            or kurtosis_pressure >= float(watch.get("kurtosis_pressure", 0.55))
            or volume_imbalance_pressure
            >= float(watch.get("volume_imbalance_pressure", 0.5))
        ):
            return RiskLevel.WATCH
        if (
            risk_volatility >= float(watch.get("risk_volatility", 0.25))
            and regime_persistence <= float(watch.get("regime_persistence", 0.4))
        ):
            return RiskLevel.WATCH
        return RiskLevel.WATCH


__all__ = [
    "MarketRegime",
    "MarketRegimeAssessment",
    "MarketRegimeClassifier",
    "RegimeHistory",
    "RegimeSnapshot",
    "RegimeSummary",
    "RiskLevel",
]

