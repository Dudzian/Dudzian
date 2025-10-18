from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bot_core.ai import (
    FeatureDataset,
    ModelTrainer,
    RetrainingScheduler,
    TrainingScheduler,
    WalkForwardValidator,
)
from bot_core.ai.feature_engineering import FeatureVector
from bot_core.ai.scheduler import ScheduledTrainingJob
from bot_core.ai.training import ExternalModelAdapter, ExternalTrainingContext, ExternalTrainingResult, register_external_model_adapter


class _MeanModel:
    def __init__(self, mean: float, feature_names: Sequence[str], scalers: Mapping[str, tuple[float, float]]):
        self.mean = float(mean)
        self.feature_names = list(feature_names)
        self.feature_scalers = dict(scalers)

    def predict(self, features: Mapping[str, float]) -> float:
        return self.mean

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> Sequence[float]:
        return [self.mean for _ in samples]


def _mean_adapter_train(context: ExternalTrainingContext) -> ExternalTrainingResult:
    mean_target = sum(context.train_targets) / len(context.train_targets)
    model = _MeanModel(mean_target, context.feature_names, context.scalers)
    return ExternalTrainingResult(state={"mean": mean_target}, trained_model=model)


def _mean_adapter_load(
    state: Mapping[str, object],
    feature_names: Sequence[str],
    metadata: Mapping[str, object],
):
    scalers_raw = metadata.get("feature_scalers", {})
    scalers: dict[str, tuple[float, float]] = {}
    if isinstance(scalers_raw, Mapping):
        for name, payload in scalers_raw.items():
            if not isinstance(payload, Mapping):
                continue
            mean = float(payload.get("mean", 0.0))
            stdev = float(payload.get("stdev", 0.0))
            scalers[str(name)] = (mean, stdev)
    return _MeanModel(float(state.get("mean", 0.0)), feature_names, scalers)


register_external_model_adapter(
    ExternalModelAdapter(
        backend="mean_regressor",
        train=_mean_adapter_train,
        load=_mean_adapter_load,
    )
)


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


def test_training_scheduler_executes_external_backend() -> None:
    vectors: list[FeatureVector] = []
    for idx in range(12):
        features = {"momentum": float(idx) / 5.0, "volume_ratio": 1.0 + (idx % 2) * 0.2}
        vectors.append(
            FeatureVector(
                timestamp=1_700_100_000 + idx * 60,
                symbol="ETHUSDT",
                features=features,
                target_bps=float(idx - 5),
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": ["ETHUSDT"]})

    schedule = RetrainingScheduler(interval=timedelta(minutes=5))
    job = ScheduledTrainingJob(
        name="eth-mean",
        scheduler=schedule,
        trainer_factory=lambda: ModelTrainer(backend="mean_regressor"),
        dataset_provider=lambda: dataset,
    )
    scheduler = TrainingScheduler()
    scheduler.register(job)

    now = datetime(2024, 5, 1, 10, tzinfo=timezone.utc)
    results = scheduler.run_due_jobs(now)

    assert len(results) == 1
    run_job, artifact = results[0]
    assert run_job is job
    assert artifact.metadata["backend"] == "mean_regressor"
    assert artifact.metrics["mae"] >= 0.0
    assert job.history, "historia uruchomień powinna być zaktualizowana"
    record = job.history[-1]
    assert record.backend == "mean_regressor"
    assert record.dataset_rows == len(dataset.vectors)
