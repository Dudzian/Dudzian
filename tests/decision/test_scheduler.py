from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bot_core.ai import FeatureDataset, ModelTrainer, RetrainingScheduler, WalkForwardValidator
from bot_core.ai.feature_engineering import FeatureVector


def test_retraining_scheduler_marks_runs() -> None:
    scheduler = RetrainingScheduler(interval=timedelta(hours=6))
    now = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)

    assert scheduler.should_retrain(now)
    scheduler.mark_executed(now)
    assert scheduler.next_run(now) == now + timedelta(hours=6)
    assert scheduler.should_retrain(now + timedelta(hours=5)) is False
    assert scheduler.should_retrain(now + timedelta(hours=6)) is True


def test_walk_forward_validator_produces_metrics() -> None:
    vectors: list[FeatureVector] = []
    for idx in range(20):
        features = {"momentum": float(idx) / 10.0, "volume_ratio": 1.0 + (idx % 3) * 0.1}
        target = (idx % 4 - 1.5) * 5.0
        vectors.append(
            FeatureVector(
                timestamp=1_700_000_000 + idx * 60,
                symbol="BTCUSDT",
                features=features,
                target_bps=target,
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": ["BTCUSDT"]})

    validator = WalkForwardValidator(dataset, train_window=8, test_window=4)

    def factory() -> ModelTrainer:
        return ModelTrainer(learning_rate=0.15, n_estimators=5)

    result = validator.validate(factory)

    assert result.windows, "walk-forward powinien zwrócić okna walidacyjne"
    assert result.average_mae >= 0.0
    assert 0.0 <= result.average_directional_accuracy <= 1.0
