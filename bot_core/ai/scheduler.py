"""Scheduler retreningu i walidacji walk-forward dla Decision Engine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, ClassVar, Iterable, Mapping, Sequence

from ..runtime.journal import TradingDecisionEvent, TradingDecisionJournal
from ._license import ensure_ai_signals_enabled
from .feature_engineering import FeatureDataset
from .models import ModelArtifact
from .training import ModelTrainer


@dataclass(slots=True)
class RetrainingScheduler:
    """Utrzymuje harmonogram ponownego trenowania modeli."""

    STATE_VERSION: ClassVar[int] = 5
    MAX_FAILURE_BACKOFF_MULTIPLIER: ClassVar[int] = 6
    interval: timedelta
    last_run: datetime | None = None
    updated_at: datetime | None = None
    last_failure: datetime | None = None
    failure_streak: int = 0
    last_failure_reason: str | None = None
    cooldown_until: datetime | None = None
    paused_until: datetime | None = None
    paused_reason: str | None = None
    persistence_path: str | Path | None = None
    _persistence_enabled: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        ensure_ai_signals_enabled("harmonogramu retreningu modeli AI")
        self._validate_interval(self.interval)
        path = self.persistence_path or Path("audit/ai_decision/scheduler.json")
        self.persistence_path = Path(path)
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_enabled = True
        if self.last_run is not None:
            self.last_run = self._ensure_utc(self.last_run)
            self.updated_at = self.last_run
        if self.updated_at is not None:
            self.updated_at = self._ensure_utc(self.updated_at)
        if self.last_failure is not None:
            self.last_failure = self._ensure_utc(self.last_failure)
        if self.cooldown_until is not None:
            self.cooldown_until = self._ensure_utc(self.cooldown_until)
        if self.paused_until is not None:
            self.paused_until = self._ensure_utc(self.paused_until)
        self._load_state()

    def should_retrain(self, now: datetime | None = None) -> bool:
        now = self._ensure_utc(now)
        self._clear_expired_pause(now)
        if self.paused_until is not None and now < self.paused_until:
            return False
        return now >= self.next_run(now)

    def mark_executed(self, when: datetime | None = None) -> None:
        timestamp = self._ensure_utc(when)
        self.last_run = timestamp
        self.updated_at = timestamp
        self.failure_streak = 0
        self.cooldown_until = None
        if self.paused_until is not None or self.paused_reason is not None:
            self.paused_until = None
            self.paused_reason = None
        self._persist_state()

    def next_run(self, now: datetime | None = None) -> datetime:
        now = self._ensure_utc(now)
        self._clear_expired_pause(now)
        if self.last_run is None:
            baseline = now
        else:
            baseline = self.last_run + self.interval
        candidate = baseline
        if self.cooldown_until is not None and self.cooldown_until > candidate:
            candidate = self.cooldown_until
        if self.paused_until is not None and self.paused_until > candidate:
            candidate = self.paused_until
        return candidate

    def update_interval(self, interval: timedelta) -> None:
        """Zmienia interwał retreningu i zapisuje nowy stan."""

        self._validate_interval(interval)
        self.interval = interval
        if self.last_run is not None:
            self.updated_at = self.last_run
        else:
            self.updated_at = self._ensure_utc(None)
        if self.cooldown_until is not None and self.last_failure is not None:
            self.cooldown_until = self._compute_failure_cooldown(
                failure_time=self.last_failure,
                previous_cooldown=None,
            )
        self._persist_state()

    def mark_failure(self, when: datetime | None = None, reason: str | None = None) -> None:
        """Rejestruje nieudane uruchomienie retreningu."""

        timestamp = self._ensure_utc(when)
        self.last_failure = timestamp
        self.updated_at = timestamp
        previous_cooldown = self.cooldown_until
        self.failure_streak = max(0, int(self.failure_streak)) + 1
        if reason is not None:
            self.last_failure_reason = str(reason)[:1024]
        self.cooldown_until = self._compute_failure_cooldown(
            failure_time=timestamp,
            previous_cooldown=previous_cooldown,
        )
        self._persist_state()

    def pause(
        self,
        *,
        until: datetime | None = None,
        duration: timedelta | None = None,
        reason: str | None = None,
    ) -> None:
        """Wstrzymuje harmonogram do wskazanego czasu."""

        if until is not None and duration is not None:
            raise ValueError("Podaj tylko until albo duration")
        if until is None and duration is None:
            raise ValueError("Wymagany jest until lub duration")

        now_utc = self._ensure_utc(None)

        if duration is not None:
            if duration.total_seconds() <= 0:
                raise ValueError("duration musi być dodatni")
            paused_until = now_utc + duration
        else:
            paused_until = self._ensure_utc(until)
            if paused_until <= now_utc:
                raise ValueError("until musi wskazywać czas w przyszłości")

        self.paused_until = paused_until
        self.paused_reason = str(reason)[:1024] if reason is not None else None
        self.updated_at = now_utc
        self._persist_state()

    def resume(self, when: datetime | None = None) -> None:
        """Przywraca harmonogram po pauzie."""

        if self.paused_until is None and self.paused_reason is None:
            return
        timestamp = self._ensure_utc(when)
        self.paused_until = None
        self.paused_reason = None
        self.updated_at = timestamp
        self._persist_state()

    def _ensure_utc(self, value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _validate_interval(self, interval: timedelta) -> None:
        seconds = interval.total_seconds()
        if seconds <= 0:
            raise ValueError("interval musi być dodatnim przedziałem czasu")

    def _compute_failure_cooldown(
        self, *, failure_time: datetime, previous_cooldown: datetime | None
    ) -> datetime:
        multiplier = min(max(1, int(self.failure_streak)), self.MAX_FAILURE_BACKOFF_MULTIPLIER)
        reference = previous_cooldown if previous_cooldown is not None else failure_time
        reference = max(reference, failure_time)
        baseline = (
            self.last_run + self.interval
            if self.last_run is not None
            else failure_time + self.interval
        )
        cooldown = self.interval * multiplier
        candidate = reference + cooldown
        if candidate < baseline:
            candidate = baseline
        return candidate

    def _load_state(self) -> None:
        if not self._persistence_enabled:
            return
        if not self.persistence_path.exists():
            return
        try:
            with self.persistence_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        version_raw = payload.get("version")
        try:
            version = int(version_raw)
        except (TypeError, ValueError):
            version = None
        if version is not None and version > self.STATE_VERSION:
            return
        interval_value = payload.get("interval")
        file_interval = self._parse_interval(interval_value)

        if file_interval is not None:
            self.interval = file_interval

        interval_for_timestamp = file_interval or self.interval

        parsed_last_run = self._parse_timestamp(payload.get("last_run"))

        if parsed_last_run is None:
            parsed_next = self._parse_timestamp(payload.get("next_run"))
            if parsed_next is not None:
                parsed_last_run = parsed_next - interval_for_timestamp

        if parsed_last_run is not None:
            parsed_last_run = self._ensure_utc(parsed_last_run)
            if self.last_run is None or parsed_last_run > self.last_run:
                self.last_run = parsed_last_run
                self.updated_at = parsed_last_run

        parsed_updated_at = self._parse_timestamp(payload.get("updated_at"))
        if parsed_updated_at is not None:
            parsed_updated_at = self._ensure_utc(parsed_updated_at)
            if self.updated_at is None or parsed_updated_at > self.updated_at:
                self.updated_at = parsed_updated_at

        parsed_failure = self._parse_timestamp(payload.get("last_failure"))
        if parsed_failure is not None:
            parsed_failure = self._ensure_utc(parsed_failure)
            if self.last_failure is None or parsed_failure > self.last_failure:
                self.last_failure = parsed_failure

        failure_streak_raw = payload.get("failure_streak")
        try:
            streak = int(failure_streak_raw)
        except (TypeError, ValueError):
            streak = None
        if streak is not None and streak >= 0:
            self.failure_streak = streak

        reason_raw = payload.get("last_failure_reason")
        if reason_raw is not None:
            self.last_failure_reason = str(reason_raw)

        parsed_cooldown = self._parse_timestamp(payload.get("cooldown_until"))
        if parsed_cooldown is not None:
            parsed_cooldown = self._ensure_utc(parsed_cooldown)
            self.cooldown_until = parsed_cooldown
        elif self.failure_streak <= 0:
            self.cooldown_until = None
        elif self.last_failure is not None:
            self.cooldown_until = self._compute_failure_cooldown(
                failure_time=self.last_failure,
                previous_cooldown=None,
            )

        parsed_paused_until = self._parse_timestamp(payload.get("paused_until"))
        if parsed_paused_until is not None:
            parsed_paused_until = self._ensure_utc(parsed_paused_until)
            self.paused_until = parsed_paused_until
        else:
            self.paused_until = None

        paused_reason_raw = payload.get("paused_reason")
        if paused_reason_raw is not None:
            self.paused_reason = str(paused_reason_raw)
        elif parsed_paused_until is None:
            self.paused_reason = None

    def _persist_state(self) -> None:
        if not self._persistence_enabled:
            return
        state = self.export_state()
        temp_name: str | None = None
        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.persistence_path.parent, delete=False
            ) as tmp:
                temp_name = tmp.name
                json.dump(state, tmp, ensure_ascii=False, indent=2)
                tmp.write("\n")
            os.replace(temp_name, self.persistence_path)
        except OSError:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            return

    def export_state(self) -> Mapping[str, object]:
        """Zwraca bieżący stan harmonogramu w formacie zgodnym z audytem."""

        return {
            "version": self.STATE_VERSION,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run().isoformat(),
            "interval": self.interval.total_seconds(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "failure_streak": int(self.failure_streak),
            "last_failure_reason": self.last_failure_reason,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "paused_reason": self.paused_reason,
        }

    def _parse_interval(self, value: object) -> timedelta | None:
        if isinstance(value, (int, float)):
            seconds = float(value)
        elif isinstance(value, str):
            try:
                seconds = float(value.strip())
            except (TypeError, ValueError):
                return None
        else:
            return None
        if seconds < 0:
            return None
        return timedelta(seconds=seconds)

    def _parse_timestamp(self, value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _clear_expired_pause(self, now: datetime | None = None) -> None:
        if self.paused_until is None:
            return
        reference = self._ensure_utc(now)
        if reference >= self.paused_until:
            self.paused_until = None
            self.paused_reason = None
            self.updated_at = reference
            self._persist_state()


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
    decision_journal: TradingDecisionJournal | None = None
    journal_environment: str = "ai_decision"
    journal_portfolio: str = "ai_training"
    journal_risk_profile: str = "model_retraining"

    def __post_init__(self) -> None:
        ensure_ai_signals_enabled("zadań treningowych AI")
        if not callable(self.trainer_factory):
            raise TypeError("trainer_factory musi być wywoływalny")
        if not callable(self.dataset_provider):
            raise TypeError("dataset_provider musi być wywoływalny")

    def is_due(self, now: datetime | None = None) -> bool:
        return self.scheduler.should_retrain(now)

    def run(self, now: datetime | None = None) -> ModelArtifact:
        try:
            dataset = self.dataset_provider()
            trainer = self.trainer_factory()
            artifact = trainer.train(dataset)
            validation_result: WalkForwardResult | None = None
            if self.validator_factory is not None:
                validator = self.validator_factory(dataset)
                validation_result = validator.validate(self.trainer_factory)
            execution_time = getattr(artifact, "trained_at", None)
            if not isinstance(execution_time, datetime):
                execution_time = now
            execution_time = self.scheduler._ensure_utc(execution_time)
            artifact.trained_at = execution_time
        except Exception as exc:
            self._handle_failure(now, exc)
            raise

        self.scheduler.mark_executed(artifact.trained_at)
        record = TrainingRunRecord(
            trained_at=artifact.trained_at,
            metrics=dict(artifact.metrics),
            backend=getattr(trainer, "backend", "builtin"),
            dataset_rows=len(dataset.vectors),
            validation=validation_result,
        )
        self.history.append(record)
        if self.decision_journal is not None:
            metadata: dict[str, str] = {
                "next_run": self.scheduler.next_run().isoformat(),
                "interval_seconds": str(self.scheduler.interval.total_seconds()),
                "dataset_rows": str(len(dataset.vectors)),
                "scheduler_version": str(self.scheduler.STATE_VERSION),
                "failure_streak": str(self.scheduler.failure_streak),
            }
            if self.scheduler.last_run is not None:
                metadata["last_run"] = self.scheduler.last_run.isoformat()
            if self.scheduler.updated_at is not None:
                metadata["state_updated_at"] = self.scheduler.updated_at.isoformat()
            if self.scheduler.last_failure is not None:
                metadata["last_failure"] = self.scheduler.last_failure.isoformat()
            if self.scheduler.last_failure_reason:
                metadata["last_failure_reason"] = self.scheduler.last_failure_reason
            if self.scheduler.cooldown_until is not None:
                metadata["cooldown_until"] = self.scheduler.cooldown_until.isoformat()
            if self.scheduler.paused_until is not None:
                metadata["paused_until"] = self.scheduler.paused_until.isoformat()
            if self.scheduler.paused_reason:
                metadata["paused_reason"] = self.scheduler.paused_reason
            for key, value in artifact.metrics.items():
                metadata[f"metric_{key}"] = str(value)
            event = TradingDecisionEvent(
                event_type="ai_retraining",
                timestamp=artifact.trained_at,
                environment=self.journal_environment,
                portfolio=self.journal_portfolio,
                risk_profile=self.journal_risk_profile,
                schedule=self.name,
                strategy=self.name,
                schedule_run_id=f"{self.name}:{artifact.trained_at.isoformat()}",
                telemetry_namespace=f"ai.scheduler.{self.name}",
                metadata=metadata,
            )
            self.decision_journal.record(event)
        if self.on_completed is not None:
            self.on_completed(artifact, validation_result)
        return artifact

    def _handle_failure(self, now: datetime | None, exc: Exception) -> None:
        failure_time = self.scheduler._ensure_utc(now)
        reason = f"{exc.__class__.__name__}: {exc}"
        self.scheduler.mark_failure(failure_time, reason=reason)
        if self.decision_journal is None:
            return
        metadata: dict[str, str] = {
            "next_run": self.scheduler.next_run(failure_time).isoformat(),
            "interval_seconds": str(self.scheduler.interval.total_seconds()),
            "scheduler_version": str(self.scheduler.STATE_VERSION),
            "failure_streak": str(self.scheduler.failure_streak),
            "error_type": exc.__class__.__name__,
        }
        if self.scheduler.last_run is not None:
            metadata["last_run"] = self.scheduler.last_run.isoformat()
        if self.scheduler.updated_at is not None:
            metadata["state_updated_at"] = self.scheduler.updated_at.isoformat()
        if self.scheduler.last_failure is not None:
            metadata["last_failure"] = self.scheduler.last_failure.isoformat()
        if self.scheduler.last_failure_reason:
            metadata["last_failure_reason"] = self.scheduler.last_failure_reason
        if self.scheduler.cooldown_until is not None:
            metadata["cooldown_until"] = self.scheduler.cooldown_until.isoformat()
        if self.scheduler.paused_until is not None:
            metadata["paused_until"] = self.scheduler.paused_until.isoformat()
        if self.scheduler.paused_reason:
            metadata["paused_reason"] = self.scheduler.paused_reason
        event = TradingDecisionEvent(
            event_type="ai_retraining_failed",
            timestamp=failure_time,
            environment=self.journal_environment,
            portfolio=self.journal_portfolio,
            risk_profile=self.journal_risk_profile,
            schedule=self.name,
            strategy=self.name,
            schedule_run_id=f"{self.name}:{failure_time.isoformat()}:failure",
            telemetry_namespace=f"ai.scheduler.{self.name}",
            metadata=metadata,
        )
        self.decision_journal.record(event)


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
