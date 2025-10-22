from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot_core.ai.regime import MarketRegime, MarketRegimeClassifier


def _build_dataframe(close: np.ndarray, *, noise: float = 0.0) -> pd.DataFrame:
    base = pd.Series(close, index=pd.date_range("2023-01-01", periods=close.size, freq="h"))
    high = base * (1.0 + 0.002 + noise)
    low = base * (1.0 - 0.002 - noise)
    volume = pd.Series(np.linspace(10_000, 15_000, close.size), index=base.index)
    return pd.DataFrame({
        "open": base,
        "high": high,
        "low": low,
        "close": base,
        "volume": volume,
    })


def test_market_regime_classifier_detects_trend() -> None:
    prices = np.linspace(100, 140, 150)
    df = _build_dataframe(prices)
    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(df, symbol="trend")
    assert assessment.regime is MarketRegime.TREND
    assert 0.0 <= assessment.risk_score <= 1.0
    assert assessment.symbol == "trend"


def test_market_regime_classifier_detects_mean_reversion() -> None:
    prices = [120.0]
    for _ in range(199):
        deviation = prices[-1] - 120.0
        prices.append(120.0 - deviation * 0.8)
    rng = np.random.default_rng(1)
    noisy = np.asarray(prices) + rng.normal(0, 0.2, size=len(prices))
    df = _build_dataframe(noisy)
    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(df, symbol="mr")
    assert assessment.regime is MarketRegime.MEAN_REVERSION
    assert assessment.confidence > 0.2


def test_market_regime_classifier_detects_daily_regime() -> None:
    base = np.full(180, 200.0)
    noise = np.random.default_rng(42).normal(0, 0.8, size=180)
    prices = base + noise
    df = _build_dataframe(prices, noise=0.01)
    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(df, symbol="daily")
    assert assessment.regime is MarketRegime.DAILY
    scores = classifier._score_regimes(assessment.metrics)
    assert scores[MarketRegime.DAILY] >= scores[MarketRegime.MEAN_REVERSION]
    assert "volatility_ratio" in assessment.metrics
    assert "volume_trend" in assessment.metrics
    assert "return_skew" in assessment.metrics
    assert "return_kurtosis" in assessment.metrics
    assert "volume_imbalance" in assessment.metrics


def test_market_regime_classifier_flags_high_risk_environment() -> None:
    rng = np.random.default_rng(7)
    base = np.linspace(100, 120, 200)
    noise = rng.normal(0, 3.5, size=200)
    prices = base + noise
    df = _build_dataframe(prices, noise=0.03)
    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(df, symbol="volatile")
    assert assessment.risk_score > 0.6


def test_market_regime_classifier_provides_distribution_metrics() -> None:
    rng = np.random.default_rng(21)
    base = np.linspace(100, 110, 160)
    noise = rng.normal(0, 1.5, size=160)
    # Introduce asymmetry
    noise[:80] += 1.2
    noise[80:] -= 0.8
    prices = base + noise
    volumes = np.linspace(5000, 9000, 160)
    df = _build_dataframe(prices)
    df["volume"] = volumes
    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(df, symbol="asym")
    assert "return_skew" in assessment.metrics
    assert "return_kurtosis" in assessment.metrics
    assert "volume_imbalance" in assessment.metrics
    assert isinstance(assessment.metrics["return_skew"], float)
    assert isinstance(assessment.metrics["return_kurtosis"], float)
    assert isinstance(assessment.metrics["volume_imbalance"], float)


def test_market_regime_classifier_assess_matches_manual_metrics() -> None:
    index = pd.date_range("2024-01-01", periods=160, freq="h")
    close = pd.Series(np.linspace(100.0, 140.0, index.size), index=index)
    high = close * 1.004
    low = close * 0.996
    volume = pd.Series(np.linspace(5000.0, 8000.0, index.size), index=index)
    ohlcv = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    classifier = MarketRegimeClassifier(min_history=30)
    assessment = classifier.assess(ohlcv, symbol="btc")

    close_series = ohlcv["close"].astype(float)
    returns = close_series.pct_change(fill_method=None).dropna()

    metrics_cfg = classifier._thresholds["market_regime"]["metrics"]
    short_span_min = int(metrics_cfg.get("short_span_min", 5))
    short_span_divisor = int(metrics_cfg.get("short_span_divisor", 3))
    long_span_min = int(metrics_cfg.get("long_span_min", 10))
    window = min(classifier.trend_window, close_series.size)
    short = close_series.ewm(
        span=max(short_span_min, window // short_span_divisor), adjust=False
    ).mean()
    long = close_series.ewm(span=max(long_span_min, window), adjust=False).mean()
    trend_strength = float(
        np.abs(short.iloc[-1] - long.iloc[-1]) / (np.abs(long.iloc[-1]) + 1e-12)
    )

    volatility = float(returns.std())
    momentum = float(returns.tail(window).mean())
    autocorr = float(returns.autocorr(lag=1) or 0.0)

    intraday_series = (
        (ohlcv["high"] - ohlcv["low"])
        .div(close_series)
        .rolling(classifier.daily_window)
        .mean()
    ).dropna()
    if intraday_series.empty:
        intraday_vol = float(
            (ohlcv["high"] - ohlcv["low"]).div(close_series).abs().mean()
        )
    else:
        intraday_vol = float(intraday_series.iloc[-1])

    drawdown = float(
        (close_series.cummax() - close_series).div(close_series.cummax() + 1e-12).max()
    )
    volatility_window = min(
        max(classifier.daily_window * 5, classifier.trend_window), returns.size
    )
    rolling_vol = returns.rolling(
        volatility_window, min_periods=max(volatility_window // 2, 10)
    ).std()
    rolling_clean = rolling_vol.dropna()
    baseline_vol = float(rolling_clean.iloc[-1]) if not rolling_clean.empty else volatility
    volatility_ratio = (
        float(volatility / (baseline_vol + 1e-12)) if baseline_vol else 1.0
    )

    volume_series = ohlcv["volume"].astype(float).sort_index()
    short_vol_ma = volume_series.rolling(classifier.daily_window, min_periods=1).mean()
    long_window = max(classifier.daily_window * 3, 1)
    long_vol_ma = volume_series.rolling(long_window, min_periods=1).mean()
    volume_trend = float(
        (short_vol_ma.iloc[-1] - long_vol_ma.iloc[-1])
        / (np.abs(long_vol_ma.iloc[-1]) + 1e-12)
    )

    skewness = float(np.nan_to_num(returns.skew(), nan=0.0, posinf=0.0, neginf=0.0))
    kurtosis = float(np.nan_to_num(returns.kurt(), nan=0.0, posinf=0.0, neginf=0.0))

    volume_series = ohlcv["volume"].astype(float).reindex(close_series.index)
    change = returns.reindex(volume_series.index, method="ffill").fillna(0.0)
    positive_volume = volume_series.where(change > 0.0).mean()
    negative_volume = volume_series.where(change <= 0.0).mean()
    denom = np.abs(positive_volume) + np.abs(negative_volume) + 1e-12
    volume_imbalance = float(
        np.nan_to_num((positive_volume - negative_volume) / denom)
    )

    expected_metrics = {
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

    for key, value in expected_metrics.items():
        assert key in assessment.metrics
        assert assessment.metrics[key] == pytest.approx(value, rel=1e-6, abs=1e-8)

    score_cfg = classifier._thresholds["market_regime"]["risk_score"]
    volatility_component = min(1.0, expected_metrics["volatility"] / classifier.volatility_threshold)
    intraday_component = min(
        1.0,
        expected_metrics["intraday_vol"]
        / (
            classifier.intraday_threshold
            * float(score_cfg.get("intraday_multiplier", 1.5))
        ),
    )
    drawdown_component = min(
        1.0,
        expected_metrics["drawdown"] / float(score_cfg.get("drawdown_threshold", 0.2))
    )
    volatility_ratio_component = min(1.0, expected_metrics.get("volatility_ratio", 1.0))
    volume_component = min(
        1.0,
        abs(expected_metrics.get("volume_trend", 0.0)) / classifier.volume_trend_threshold,
    )
    expected_risk = float(
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

    assert assessment.risk_score == pytest.approx(expected_risk, rel=1e-6, abs=1e-8)
