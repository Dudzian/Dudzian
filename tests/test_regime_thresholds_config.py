from __future__ import annotations

import yaml

from bot_core.ai.regime import MarketRegimeClassifier
from pathlib import Path


def test_classifier_defaults_match_calibrated_config() -> None:
    config_path = Path("config/regime_thresholds.yaml")
    data = yaml.safe_load(config_path.read_text())
    defaults = data["defaults"]
    classifier = MarketRegimeClassifier()
    assert classifier.min_history == defaults["min_history"]
    assert classifier.trend_window == defaults["trend_window"]
    assert classifier.daily_window == defaults["daily_window"]
    assert classifier.trend_strength_threshold == defaults["trend_strength_threshold"]
    assert classifier.momentum_threshold == defaults["momentum_threshold"]
    assert classifier.volatility_threshold == defaults["volatility_threshold"]
    assert classifier.intraday_threshold == defaults["intraday_threshold"]
    assert classifier.autocorr_threshold == defaults["autocorr_threshold"]
    assert classifier.volume_trend_threshold == defaults["volume_trend_threshold"]
