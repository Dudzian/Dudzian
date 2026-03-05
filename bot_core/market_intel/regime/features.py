from __future__ import annotations

"""Kontrakt feature engineering dla klasyfikacji reżimu rynku."""

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np
import pandas as pd


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(slots=True)
class RegimeFeatureSet:
    """Ustrukturyzowany zestaw cech na podstawie danych OHLCV."""

    metrics: Mapping[str, float]
    price_column: str
    symbol: str | None = None

    def to_mapping(self) -> Mapping[str, float]:
        return dict(self.metrics)


def build_regime_features(
    market_data: pd.DataFrame,
    *,
    price_col: str = "close",
    symbol: str | None = None,
    min_history: int = 30,
    trend_window: int = 50,
    daily_window: int = 20,
    metrics_config: Mapping[str, Any] | None = None,
    autocorr_lag: int = 1,
) -> RegimeFeatureSet:
    """Przekształca dane OHLCV w zestaw cech używany przez klasyfikator."""

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
    if returns.size < min_history:
        raise ValueError("Not enough observations to classify market regime")

    metrics_cfg = _ensure_mapping(metrics_config)
    short_span_min = int(metrics_cfg.get("short_span_min", 5))
    short_span_divisor = int(metrics_cfg.get("short_span_divisor", 3))
    long_span_min = int(metrics_cfg.get("long_span_min", 10))
    window = min(trend_window, close.size)
    short = close.ewm(span=max(short_span_min, window // short_span_divisor), adjust=False).mean()
    long = close.ewm(span=max(long_span_min, window), adjust=False).mean()
    trend_strength = float(np.abs(short.iloc[-1] - long.iloc[-1]) / (np.abs(long.iloc[-1]) + 1e-12))

    volatility = float(np.nan_to_num(returns.std(), nan=0.0, posinf=0.0, neginf=0.0))
    momentum = float(np.nan_to_num(returns.tail(window).mean(), nan=0.0, posinf=0.0, neginf=0.0))
    autocorr_raw = returns.autocorr(lag=autocorr_lag)
    autocorr = float(np.nan_to_num(autocorr_raw if autocorr_raw is not None else 0.0, nan=0.0))

    if {"high", "low"}.issubset(ordered_data.columns):
        intraday_series = (
            (ordered_data["high"] - ordered_data["low"]).div(close).rolling(daily_window).mean()
        )
        intraday_series = intraday_series.dropna()
        if intraday_series.empty:
            intraday_vol = float(
                np.nan_to_num(
                    (ordered_data["high"] - ordered_data["low"]).div(close).abs().mean(),
                    nan=0.0,
                )
            )
        else:
            intraday_vol = float(np.nan_to_num(intraday_series.iloc[-1], nan=0.0))
    else:
        intraday_vol = float(np.nan_to_num(returns.tail(daily_window).abs().mean(), nan=0.0))

    drawdown = float(
        np.nan_to_num((close.cummax() - close).div(close.cummax() + 1e-12).max(), nan=0.0)
    )
    volatility_window = min(max(daily_window * 5, trend_window), returns.size)
    rolling_vol = returns.rolling(
        volatility_window, min_periods=max(volatility_window // 2, 10)
    ).std()
    rolling_clean = rolling_vol.dropna()
    baseline_vol = (
        float(np.nan_to_num(rolling_clean.iloc[-1], nan=0.0, posinf=0.0, neginf=0.0))
        if not rolling_clean.empty
        else volatility
    )
    volatility_ratio = float(volatility / (baseline_vol + 1e-12)) if baseline_vol else 1.0

    if "volume" in ordered_data.columns:
        volume_series = ordered_data["volume"].astype(float).sort_index()
        short_vol_ma = volume_series.rolling(daily_window, min_periods=1).mean()
        long_window = max(daily_window * 3, 1)
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

    if "volume" in ordered_data.columns:
        volume_series = ordered_data["volume"].astype(float).reindex(close.index)
        change = returns.reindex(volume_series.index, method="ffill").fillna(0.0)
        positive_volume = float(np.nan_to_num(volume_series.where(change > 0.0).mean(), nan=0.0))
        negative_volume = float(np.nan_to_num(volume_series.where(change <= 0.0).mean(), nan=0.0))
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
    return RegimeFeatureSet(metrics=metrics, price_column=price_col, symbol=symbol)


__all__ = ["RegimeFeatureSet", "build_regime_features"]
