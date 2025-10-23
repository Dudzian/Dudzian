from datetime import timedelta

import numpy as np
import pandas as pd

from bot_core.ai.monitoring import (
    DataCompletenessWatcher,
    FeatureBoundsValidator,
    FeatureDriftAnalyzer,
)


def test_data_completeness_watcher_detects_gaps() -> None:
    index = pd.date_range("2024-06-01", periods=12, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": index.delete([4, 5]),
            "close": np.linspace(100.0, 102.0, 10),
        }
    )
    watcher = DataCompletenessWatcher(frequency=timedelta(minutes=1), warning_gap_ratio=0.05, critical_gap_ratio=0.2)

    assessment = watcher.assess(frame)

    assert assessment.status in {"warning", "critical"}
    assert assessment.summary["total_gaps"] == 2
    issue_payloads = assessment.issues_payload()
    assert issue_payloads and issue_payloads[0]["code"] == "missing_bars"
    assert issue_payloads[0]["details"]["missing_bars"] == 2


def test_data_completeness_watcher_handles_empty_frame() -> None:
    watcher = DataCompletenessWatcher("1min")

    assessment = watcher.assess(pd.DataFrame(columns=["timestamp", "close"]))

    assert assessment.status == "warning"
    assert assessment.summary["observed_rows"] == 0
    assert assessment.issues_payload()[0]["code"] == "no_data"


def test_feature_drift_analyzer_computes_metrics() -> None:
    baseline = pd.DataFrame(
        {
            "close": np.linspace(100.0, 110.0, 200),
            "volume": np.linspace(1_000.0, 2_000.0, 200),
        }
    )
    production = pd.DataFrame(
        {
            "close": np.linspace(110.0, 120.0, 200),
            "volume": np.linspace(1_500.0, 2_500.0, 200),
        }
    )
    analyzer = FeatureDriftAnalyzer(psi_threshold=0.05, ks_threshold=0.05)

    assessment = analyzer.compare(baseline, production)

    assert assessment.triggered is True
    assert assessment.summary["max_psi"] > 0.0
    metrics = assessment.metrics_payload()
    assert set(metrics) == {"close", "volume"}
    assert metrics["close"]["psi"] >= assessment.summary["max_psi"]


def test_feature_bounds_validator_flags_outliers() -> None:
    validator = FeatureBoundsValidator(sigma_multiplier=2.0)
    scalers = {"momentum": (0.0, 0.5), "volume_ratio": (1.0, 0.1)}
    features = {"momentum": 2.0, "volume_ratio": 1.15}

    issues = validator.validate(features, scalers)

    assert issues, "Validator should detect out-of-bounds feature"
    assert any(issue.code == "feature_out_of_bounds" for issue in issues)
    assert validator.is_within_bounds({"momentum": 0.1, "volume_ratio": 1.05}, scalers)

