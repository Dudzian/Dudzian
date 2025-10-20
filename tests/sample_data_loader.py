"""Helpers for loading calibrated OHLCV samples used in integration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

import logging
import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSummary,
    RiskLevel,
)

LOGGER = logging.getLogger(__name__)

SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv"

DATASET_BY_REGIME: Mapping[MarketRegime, str] = {
    MarketRegime.TREND: "trend",
    MarketRegime.DAILY: "daily_high_risk",
    MarketRegime.MEAN_REVERSION: "mean_reversion",
}

SAMPLE_EXPECTATIONS: Mapping[str, Mapping[str, object]] = {
    "trend": {
        "regime": MarketRegime.TREND,
        "risk_level": RiskLevel.BALANCED,
        "risk_range": (0.28, 0.38),
        "confidence_min": 0.7,
    },
    "daily_high_risk": {
        "regime": MarketRegime.DAILY,
        "risk_level": RiskLevel.CRITICAL,
        "risk_range": (0.8, 0.95),
        "confidence_min": 0.55,
    },
    "mean_reversion": {
        "regime": MarketRegime.MEAN_REVERSION,
        "risk_level": RiskLevel.BALANCED,
        "risk_range": (0.23, 0.35),
        "confidence_min": 0.6,
    },
}


def load_sample_ohlcv(name: str) -> pd.DataFrame:
    """Load OHLCV history for the named sample."""

    path = SAMPLE_DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def iterate_assessments(
    name: str,
    *,
    classifier: MarketRegimeClassifier | None = None,
    history_window: int = 5,
    step: int = 24,
) -> Iterable[MarketRegimeAssessment]:
    """Yield regime assessments for expanding windows of the sample data."""

    classifier = classifier or MarketRegimeClassifier(min_history=30)
    df = load_sample_ohlcv(name)
    start = classifier.min_history + 1
    for end in range(start, len(df) + 1, step):
        yield classifier.assess(df.iloc[:end])


def summarise_sample(
    name: str,
    *,
    classifier: MarketRegimeClassifier | None = None,
    history_window: int = 5,
    step: int = 24,
) -> RegimeSummary:
    """Build a regime summary from the sample assessments."""

    classifier = classifier or MarketRegimeClassifier(min_history=30)
    history = RegimeHistory(maxlen=history_window)
    summary: RegimeSummary | None = None
    for assessment in iterate_assessments(
        name,
        classifier=classifier,
        history_window=history_window,
        step=step,
    ):
        history.update(assessment)
        summary = history.summarise()
    if summary is None:
        raise RuntimeError(f"Not enough history to summarise sample {name!r}")
    return summary


def load_summary_for_regime(
    regime: MarketRegime,
    *,
    classifier: MarketRegimeClassifier | None = None,
    overrides: Mapping[str, object] | None = None,
    dataset: str | None = None,
    step: int = 24,
) -> RegimeSummary:
    """Return a calibrated summary for the desired regime with optional overrides."""

    dataset_name = dataset or DATASET_BY_REGIME[regime]
    summary = summarise_sample(dataset_name, classifier=classifier, step=step)
    if overrides:
        summary = replace(summary, **overrides)
    return summary


def log_deviation(
    name: str,
    summary: RegimeSummary,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log deviations between expectations and the obtained summary."""

    logger = logger or LOGGER
    expectations = SAMPLE_EXPECTATIONS.get(name)
    if not expectations:
        return
    low, high = expectations["risk_range"]  # type: ignore[assignment]
    if not (low <= summary.risk_score <= high):
        logger.warning(
            "Risk score for sample %s outside calibrated range: %.4f not in [%.3f, %.3f]",
            name,
            summary.risk_score,
            low,
            high,
        )
    confidence_min = expectations.get("confidence_min")
    if confidence_min is not None and summary.confidence < float(confidence_min):
        logger.warning(
            "Confidence for sample %s below expectation: %.4f < %.3f",
            name,
            summary.confidence,
            confidence_min,
        )
