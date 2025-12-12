from __future__ import annotations

"""Klasyfikacja reżimu rynku oparta o kontrakt feature set."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
import pandas as pd

from bot_core.ai.config_loader import load_risk_thresholds
from .features import RegimeFeatureSet, build_regime_features


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class MarketRegime(str, Enum):
    TREND = "trend"
    DAILY = "daily"
    MEAN_REVERSION = "mean_reversion"


class RiskLevel(str, Enum):
    CALM = "calm"
    BALANCED = "balanced"
    WATCH = "watch"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(slots=True)
class MarketRegimeAssessment:
    regime: MarketRegime
    confidence: float
    risk_score: float
    metrics: Mapping[str, float]
    symbol: str | None = None

    def to_dict(self) -> Mapping[str, float | str | None]:
        payload: MutableMapping[str, float | str | None] = {
            "regime": self.regime.value,
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
        }
        payload.update({str(key): float(value) for key, value in self.metrics.items()})
        payload["symbol"] = self.symbol
        return payload


class MarketRegimeClassifier:
    """Heurystyczny klasyfikator wykorzystujący wyodrębnione cechy rynku."""

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
        return self._thresholds_loader

    def reload_thresholds(self) -> None:
        self._thresholds = _ensure_mapping(
            dict(self._thresholds_loader())
        )

    def thresholds_snapshot(self) -> Mapping[str, Any]:
        return dict(self._thresholds)

    def metrics_config(self) -> Mapping[str, Any]:
        return dict(self._metrics_config())

    def risk_score_config(self) -> Mapping[str, Any]:
        return dict(self._risk_score_config())

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
        feature_set = build_regime_features(
            market_data,
            price_col=price_col,
            symbol=symbol,
            min_history=self.min_history,
            trend_window=self.trend_window,
            daily_window=self.daily_window,
            metrics_config=self._metrics_config(),
            autocorr_lag=1,
        )

        metrics = feature_set.metrics
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
            np.clip(
                0.6 * max(0.0, min(1.0, trend_norm))
                + 0.3 * max(0.0, momentum_norm)
                + 0.1 * min(1.0, volume_norm),
                0.0,
                1.0,
            )
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
        market_cfg = self._market_regime_config()
        score_cfg = _ensure_mapping(market_cfg.get("risk_score"))
        volatility_component = min(
            1.0, float(metrics.get("volatility", 0.0)) / self.volatility_threshold
        )
        intraday_component = min(
            1.0,
            float(metrics.get("intraday_vol", 0.0))
            / (
                self.intraday_threshold
                * float(score_cfg.get("intraday_multiplier", 1.5))
            ),
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


__all__ = [
    "MarketRegime",
    "MarketRegimeAssessment",
    "MarketRegimeClassifier",
    "RegimeFeatureSet",
    "RiskLevel",
]
