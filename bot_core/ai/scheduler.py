"""Scheduler retreningu i walidacji walk-forward dla Decision Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, Sequence

from .feature_engineering import FeatureDataset
from .training import ModelTrainer


@dataclass(slots=True)
class RetrainingScheduler:
    """Utrzymuje harmonogram ponownego trenowania modeli."""

    interval: timedelta
    last_run: datetime | None = None

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


__all__ = [
    "RetrainingScheduler",
    "WalkForwardResult",
    "WalkForwardValidator",
]
