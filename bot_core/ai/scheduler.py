"""Scheduler retreningu i walidacji walk-forward dla Decision Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, Sequence

from ._license import ensure_ai_signals_enabled
from .feature_engineering import FeatureDataset
from .models import ModelArtifact
from .training import ModelTrainer


@dataclass(slots=True)
class RetrainingScheduler:
    """Utrzymuje harmonogram ponownego trenowania modeli."""

    interval: timedelta
    last_run: datetime | None = None

    def __post_init__(self) -> None:
        ensure_ai_signals_enabled("harmonogramu retreningu modeli AI")

    def should_retrain(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.last_run is None:
            return True
        return now - self.last_run >= self.interval

    def mark_executed(self, when: datetime | None = None) -> None:
        self.last_run = when or datetime.now(timezone.utc)

    def next_run(self, now: datetime | None = None) -> datetime:
        now = now or datetime.now(timezone.utc)
        if self.last_run is None:
            return now
        return self.last_run + self.interval


@dataclass(slots=True)
class WalkForwardWindow:
    train_indices: Sequence[int]
    test_indices: Sequence[int]


@dataclass(slots=True)
class WalkForwardResult:
    """Zbiorcze metryki walidacji walk-forward."""

    windows: Sequence[Mapping[str, float]]
    average_mae: float
    average_directional_accuracy: float


class WalkForwardValidator:
    """Realizuje walidację walk-forward na zbiorze cech."""

    def __init__(
        self,
        dataset: FeatureDataset,
        *,
        train_window: int,
        test_window: int,
        step: int | None = None,
    ) -> None:
        if train_window <= 0:
            raise ValueError("train_window musi być dodatni")
        if test_window <= 0:
            raise ValueError("test_window musi być dodatni")
        if len(dataset.vectors) < train_window + test_window:
            raise ValueError("Za mało danych do przeprowadzenia walidacji walk-forward")
        self._dataset = dataset
        self._train_window = train_window
        self._test_window = test_window
        self._step = step or test_window

    def windows(self) -> Iterable[WalkForwardWindow]:
        total = len(self._dataset.vectors)
        start = 0
        while start + self._train_window + self._test_window <= total:
            train_indices = list(range(start, start + self._train_window))
            test_indices = list(
                range(start + self._train_window, start + self._train_window + self._test_window)
            )
            yield WalkForwardWindow(train_indices=train_indices, test_indices=test_indices)
            start += self._step

    def validate(self, trainer_factory: Callable[[], ModelTrainer]) -> WalkForwardResult:
        windows_metrics: list[Mapping[str, float]] = []
        maes: list[float] = []
        directional: list[float] = []

        for window in self.windows():
            train_dataset = self._dataset.subset(window.train_indices)
            trainer = trainer_factory()
            artifact = trainer.train(train_dataset)
            model = artifact.build_model()
            test_dataset = self._dataset.subset(window.test_indices)
            preds = [float(model.predict(vector.features)) for vector in test_dataset.vectors]
            mae = 0.0
            if preds:
                mae = sum(
                    abs(vector.target_bps - preds[pos])
                    for pos, vector in enumerate(test_dataset.vectors)
                ) / len(preds)
            maes.append(mae)
            hits = 0
            for pos, vector in enumerate(test_dataset.vectors):
                target = vector.target_bps
                pred = preds[pos]
                if (target >= 0 and pred >= 0) or (target < 0 and pred < 0):
                    hits += 1
            accuracy = hits / len(test_dataset.vectors) if test_dataset.vectors else 0.0
            directional.append(accuracy)
            windows_metrics.append(
                {
                    "start_timestamp": test_dataset.metadata.get("start_timestamp", 0.0),
                    "end_timestamp": test_dataset.metadata.get("end_timestamp", 0.0),
                    "mae": mae,
                    "directional_accuracy": accuracy,
                }
            )

        avg_mae = sum(maes) / len(maes) if maes else 0.0
        avg_dir = sum(directional) / len(directional) if directional else 0.0
        return WalkForwardResult(
            windows=tuple(windows_metrics),
            average_mae=avg_mae,
            average_directional_accuracy=avg_dir,
        )


@dataclass(slots=True)
class TrainingRunRecord:
    """Informacje o pojedynczym uruchomieniu treningu."""

    trained_at: datetime
    metrics: Mapping[str, float]
    backend: str
    dataset_rows: int
    validation: WalkForwardResult | None = None


@dataclass(slots=True)
class ScheduledTrainingJob:
    """Definicja zadania treningowego zarządzanego przez harmonogram."""

    name: str
    scheduler: RetrainingScheduler
    trainer_factory: Callable[[], ModelTrainer]
    dataset_provider: Callable[[], FeatureDataset]
    validator_factory: Callable[[FeatureDataset], WalkForwardValidator] | None = None
    on_completed: Callable[[ModelArtifact, WalkForwardResult | None], None] | None = None
    history: list[TrainingRunRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        ensure_ai_signals_enabled("zadań treningowych AI")
        if not callable(self.trainer_factory):
            raise TypeError("trainer_factory musi być wywoływalny")
        if not callable(self.dataset_provider):
            raise TypeError("dataset_provider musi być wywoływalny")

    def is_due(self, now: datetime | None = None) -> bool:
        return self.scheduler.should_retrain(now)

    def run(self, now: datetime | None = None) -> ModelArtifact:
        dataset = self.dataset_provider()
        trainer = self.trainer_factory()
        artifact = trainer.train(dataset)
        validation_result: WalkForwardResult | None = None
        if self.validator_factory is not None:
            validator = self.validator_factory(dataset)
            validation_result = validator.validate(self.trainer_factory)
        self.scheduler.mark_executed(now)
        record = TrainingRunRecord(
            trained_at=artifact.trained_at,
            metrics=dict(artifact.metrics),
            backend=getattr(trainer, "backend", "builtin"),
            dataset_rows=len(dataset.vectors),
            validation=validation_result,
        )
        self.history.append(record)
        if self.on_completed is not None:
            self.on_completed(artifact, validation_result)
        return artifact


class TrainingScheduler:
    """Prosty harmonogram zarządzający wieloma zadaniami treningowymi."""

    def __init__(self) -> None:
        ensure_ai_signals_enabled("harmonogramu treningowego AI")
        self._jobs: dict[str, ScheduledTrainingJob] = {}

    def register(self, job: ScheduledTrainingJob) -> None:
        self._jobs[job.name] = job

    def unregister(self, name: str) -> None:
        self._jobs.pop(name, None)

    def due_jobs(self, now: datetime | None = None) -> Sequence[ScheduledTrainingJob]:
        return [job for job in self._jobs.values() if job.is_due(now)]

    def run_due_jobs(
        self, now: datetime | None = None
    ) -> Sequence[tuple[ScheduledTrainingJob, ModelArtifact]]:
        results: list[tuple[ScheduledTrainingJob, ModelArtifact]] = []
        for job in self.due_jobs(now):
            artifact = job.run(now)
            results.append((job, artifact))
        return results


__all__ = [
    "RetrainingScheduler",
    "ScheduledTrainingJob",
    "TrainingRunRecord",
    "TrainingScheduler",
    "WalkForwardResult",
    "WalkForwardValidator",
]
