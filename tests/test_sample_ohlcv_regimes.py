from __future__ import annotations

import logging

import pytest

from bot_core.ai.regime import MarketRegimeClassifier
from tests.sample_data_loader import (
    SAMPLE_EXPECTATIONS,
    load_sample_ohlcv,
    log_deviation,
    summarise_sample,
)


@pytest.mark.parametrize("name", sorted(SAMPLE_EXPECTATIONS))
def test_sample_series_align_with_calibrated_regimes(name: str, caplog: pytest.LogCaptureFixture) -> None:
    expectations = SAMPLE_EXPECTATIONS[name]
    classifier = MarketRegimeClassifier(min_history=30)
    df = load_sample_ohlcv(name)
    assessment = classifier.assess(df, symbol=name)
    with caplog.at_level(logging.INFO):
        summary = summarise_sample(name, classifier=classifier)
        log_deviation(name, summary, logger=logging.getLogger("tests.sample_data"))
    assert assessment.regime is expectations["regime"]
    assert summary.regime is expectations["regime"]
    assert summary.risk_level is expectations["risk_level"]
    low, high = expectations["risk_range"]
    assert low <= summary.risk_score <= high
    assert summary.confidence >= expectations["confidence_min"]
