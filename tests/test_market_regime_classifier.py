from __future__ import annotations

import numpy as np
import pandas as pd

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
