"""Scheduler retreningu i walidacji walk-forward dla Decision Engine."""

from __future__ import annotations

import json
import os
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, ClassVar, Iterable, Mapping, MutableMapping, Sequence

from ._license import ensure_ai_signals_enabled
from .feature_engineering import FeatureDataset
from .models import ModelArtifact
from .training import ModelTrainer

from .audit import DEFAULT_AUDIT_ROOT, save_walk_forward_report
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal


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

    @property
    def train_window(self) -> int:
        return self._train_window

    @property
    def test_window(self) -> int:
        return self._test_window

    @property
    def step(self) -> int:
        return self._step

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
    audit_root: str | Path | None = None
    decision_journal: TradingDecisionJournal | None = None
    decision_journal_context: Mapping[str, str] | None = None
    _journal_environment: str = field(
        init=False, repr=False, default=DEFAULT_JOURNAL_ENVIRONMENT
    )
    _journal_portfolio: str | None = field(init=False, repr=False, default=None)
    _journal_risk_profile: str = field(
        init=False, repr=False, default=DEFAULT_JOURNAL_RISK_PROFILE
    )
    journal_environment: InitVar[str | None] = None
    journal_portfolio: InitVar[str | None] = None
    journal_risk_profile: InitVar[str | None] = None

    def __post_init__(
        self,
        journal_environment: str | None,
        journal_portfolio: str | None,
        journal_risk_profile: str | None,
    ) -> None:
        ensure_ai_signals_enabled("zadań treningowych AI")
        if not callable(self.trainer_factory):
            raise TypeError("trainer_factory musi być wywoływalny")
        if not callable(self.dataset_provider):
            raise TypeError("dataset_provider musi być wywoływalny")
        if self.decision_journal_context is not None and not isinstance(
            self.decision_journal_context, Mapping
        ):
            raise TypeError("decision_journal_context musi być mapowaniem lub None")
        env_value = object.__getattribute__(self, "journal_environment")
        if isinstance(env_value, property):
            env_value = "ai-training"
        object.__setattr__(self, "journal_environment", str(env_value or "ai-training"))

        risk_value = object.__getattribute__(self, "journal_risk_profile")
        if isinstance(risk_value, property):
            risk_value = "ai-research"
        object.__setattr__(self, "journal_risk_profile", str(risk_value or "ai-research"))

        portfolio_value = object.__getattribute__(self, "journal_portfolio")
        if isinstance(portfolio_value, property) or portfolio_value is None:
            object.__setattr__(self, "journal_portfolio", self.name)
        else:
            object.__setattr__(self, "journal_portfolio", str(portfolio_value))

    def is_due(self, now: datetime | None = None) -> bool:
        return self.scheduler.should_retrain(now)

    def _journal_context_value(self, key: str, default: str) -> str:
        if self.decision_journal_context is None:
            return default
        value = self.decision_journal_context.get(key)
        if value is None:
            return default
        return self._normalize_journal_value(value, default=default)

    @staticmethod
    def _normalize_journal_value(value: object | None, *, default: str) -> str:
        if value is None:
            return default
        coerced = str(value)
        if coerced:
            return coerced
        return default

    @property
    def journal_environment(self) -> str:
        return self._journal_context_value("environment", self._journal_environment)

    @journal_environment.setter
    def journal_environment(self, value: str | None) -> None:
        self._journal_environment = self._normalize_journal_value(
            value, default=DEFAULT_JOURNAL_ENVIRONMENT
        )

    @property
    def journal_portfolio(self) -> str:
        default_portfolio = self._journal_portfolio or self.name
        return self._journal_context_value("portfolio", default_portfolio)

    @journal_portfolio.setter
    def journal_portfolio(self, value: str | None) -> None:
        if value is None:
            self._journal_portfolio = None
        else:
            self._journal_portfolio = self._normalize_journal_value(
                value, default=self.name
            )

    @property
    def journal_risk_profile(self) -> str:
        return self._journal_context_value("risk_profile", self._journal_risk_profile)

    @journal_risk_profile.setter
    def journal_risk_profile(self, value: str | None) -> None:
        self._journal_risk_profile = self._normalize_journal_value(
            value, default=DEFAULT_JOURNAL_RISK_PROFILE
        )

    @property
    def journal_strategy(self) -> str:
        return self._journal_context_value("strategy", self.name)

    def run(self, now: datetime | None = None) -> ModelArtifact:
        planned_timestamp = (
            self.scheduler._ensure_utc(now) if now is not None else None
        )
        dataset: FeatureDataset | None = None
        trainer: ModelTrainer | None = None
        try:
            dataset = self.dataset_provider()
            trainer = self.trainer_factory()
            artifact = trainer.train(dataset)
        except Exception as exc:
            failure_time = planned_timestamp or self.scheduler._ensure_utc(None)
            self.scheduler.mark_failure(failure_time, reason=f"{type(exc).__name__}: {exc}")
            self._record_retraining_failure_journal(
                failed_at=failure_time,
                error=exc,
                dataset=dataset,
            )
            raise

        trained_at = planned_timestamp or artifact.trained_at
        if trained_at != artifact.trained_at:
            artifact = replace(artifact, trained_at=trained_at)

        validation_result: WalkForwardResult | None = None
        report_path: Path | None = None
        validator: WalkForwardValidator | None = None
        if self.validator_factory is not None:
            validator = self.validator_factory(dataset)
            validation_result = validator.validate(self.trainer_factory)
            report_path = save_walk_forward_report(
                validation_result,
                job_name=self.name,
                dataset=dataset,
                validator=validator,
                artifact=artifact,
                audit_root=self.audit_root if self.audit_root is not None else DEFAULT_AUDIT_ROOT,
                generated_at=artifact.trained_at,
                trainer_backend=getattr(trainer, "backend", "builtin"),
            )

        self.scheduler.mark_executed(artifact.trained_at)
        summary_metrics: dict[str, object]
        try:
            summary_callable = getattr(artifact.metrics, "summary")
        except AttributeError:
            summary_callable = None
        if callable(summary_callable):
            try:
                summary_source = summary_callable()
            except Exception:  # pragma: no cover - defensywnie na nietypowe metryki
                summary_source = {}
        else:
            summary_source = {}
        if isinstance(summary_source, Mapping):
            summary_metrics = dict(summary_source)
        else:
            try:
                summary_metrics = dict(summary_source)
            except Exception:
                summary_metrics = {}
        record = TrainingRunRecord(
            trained_at=artifact.trained_at,
            metrics=dict(summary_metrics),
            backend=getattr(trainer, "backend", "builtin"),
            dataset_rows=len(dataset.vectors),
            validation=validation_result,
        )
        self.history.append(record)
        if self.decision_journal is not None:
            context = self._build_journal_context()
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
            for key, value in summary_metrics.items():
                try:
                    numeric_value = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                metadata[f"metric_{key}"] = f"{numeric_value:.10f}"
            block_metrics_source: Mapping[str, Mapping[str, float]] | None = None
            try:
                blocks_callable = getattr(artifact.metrics, "blocks")
            except AttributeError:
                blocks_callable = None
            if callable(blocks_callable):
                try:
                    block_metrics_source = blocks_callable()
                except Exception:  # pragma: no cover - defensywnie na nietypowe metryki
                    block_metrics_source = {}
            elif hasattr(artifact.metrics, "splits"):
                splits_callable = getattr(artifact.metrics, "splits")
                if callable(splits_callable):
                    try:
                        block_metrics_source = splits_callable()
                    except Exception:  # pragma: no cover - defensywnie na nietypowe metryki
                        block_metrics_source = {}
            if isinstance(block_metrics_source, Mapping):
                for split_name, values in block_metrics_source.items():
                    if split_name == "summary" or not isinstance(values, Mapping):
                        continue
                    for metric_name, metric_value in values.items():
                        try:
                            formatted_value = float(metric_value)
                        except (TypeError, ValueError):
                            continue
                        metadata[
                            f"metric_{split_name}_{metric_name}"
                        ] = f"{formatted_value:.10f}"
            event = TradingDecisionEvent(
                event_type="ai_retraining",
                timestamp=artifact.trained_at,
                environment=context.get("environment", self.journal_environment),
                portfolio=context.get("portfolio", self.journal_portfolio),
                risk_profile=context.get("risk_profile", self.journal_risk_profile),
                schedule=context.get("schedule", self.name),
                strategy=context.get("strategy", self.name),
                schedule_run_id=context.get(
                    "schedule_run_id", f"{self.name}:{artifact.trained_at.isoformat()}"
                ),
                strategy_instance_id=context.get("strategy_instance_id"),
                telemetry_namespace=context.get(
                    "telemetry_namespace", f"ai.scheduler.{self.name}"
                ),
                metadata=metadata,
            )
            self.decision_journal.record(event)
            if validation_result is not None:
                self._record_walk_forward_journal(
                    validation=validation_result,
                    report_path=report_path,
                    validator=validator,
                    dataset=dataset,
                    trained_at=artifact.trained_at,
                    context=context,
                )
        if self.on_completed is not None:
            self.on_completed(artifact, validation_result)
        return artifact

    def _record_walk_forward_journal(
        self,
        *,
        validation: WalkForwardResult,
        report_path: Path | None,
        validator: WalkForwardValidator | None,
        dataset: FeatureDataset,
        trained_at: datetime,
        context: Mapping[str, str] | None = None,
    ) -> None:
        if self.decision_journal is None:
            return

        context_view: Mapping[str, str]
        if context is None:
            context_view = self._build_journal_context()
        else:
            context_view = context

        metadata: MutableMapping[str, str] = {
            "job": self.name,
            "average_mae": f"{validation.average_mae:.10f}",
            "average_directional_accuracy": f"{validation.average_directional_accuracy:.10f}",
            "dataset_rows": str(len(dataset.vectors)),
        }
        if validator is not None:
            metadata["train_window"] = str(getattr(validator, "train_window", ""))
            metadata["test_window"] = str(getattr(validator, "test_window", ""))
            metadata["step"] = str(getattr(validator, "step", ""))
        if report_path is not None:
            metadata["report_path"] = str(report_path)
        metadata["trained_at"] = trained_at.astimezone(timezone.utc).isoformat()

        event = TradingDecisionEvent(
            event_type="ai_walk_forward_report",
            timestamp=trained_at.astimezone(timezone.utc),
            environment=context_view.get("environment", self.journal_environment),
            portfolio=context_view.get("portfolio", self.journal_portfolio),
            risk_profile=context_view.get("risk_profile", self.journal_risk_profile),
            schedule=context_view.get("schedule", self.name),
            strategy=context_view.get("strategy", self.name),
            schedule_run_id=context_view.get("schedule_run_id"),
            strategy_instance_id=context_view.get("strategy_instance_id"),
            metadata=metadata,
        )
        try:
            self.decision_journal.record(event)
        except Exception:  # pragma: no cover - audyt nie może blokować treningu
            pass

    def _build_journal_context(self) -> MutableMapping[str, str]:
        context: MutableMapping[str, str] = {
            "environment": self.journal_environment,
            "portfolio": self.journal_portfolio,
            "risk_profile": self.journal_risk_profile,
            "schedule": self.name,
            "strategy": self.name,
            "telemetry_namespace": f"ai.scheduler.{self.name}",
        }
        if self.decision_journal_context is not None:
            for key, value in self.decision_journal_context.items():
                if value is None:
                    continue
                key_str = str(key)
                if key_str in {"environment", "portfolio", "risk_profile", "strategy", "schedule"}:
                    fallback = context.get(key_str, self.name)
                    context[key_str] = self._normalize_journal_value(
                        value, default=fallback
                    )
                else:
                    context[key_str] = str(value)
        return context

    def _record_retraining_failure_journal(
        self,
        *,
        failed_at: datetime,
        error: BaseException,
        dataset: FeatureDataset | None,
    ) -> None:
        if self.decision_journal is None:
            return

        context = self._build_journal_context()
        metadata: MutableMapping[str, str] = {
            "scheduler_version": str(self.scheduler.STATE_VERSION),
            "failure_streak": str(self.scheduler.failure_streak),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "last_failure": failed_at.astimezone(timezone.utc).isoformat(),
            "next_run": self.scheduler.next_run().isoformat(),
        }
        if dataset is not None:
            metadata["dataset_rows"] = str(len(dataset.vectors))
        if self.scheduler.last_run is not None:
            metadata["last_run"] = self.scheduler.last_run.isoformat()
        if self.scheduler.updated_at is not None:
            metadata["state_updated_at"] = self.scheduler.updated_at.isoformat()
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
            timestamp=failed_at,
            environment=context.get("environment", self.journal_environment),
            portfolio=context.get("portfolio", self.journal_portfolio),
            risk_profile=context.get("risk_profile", self.journal_risk_profile),
            schedule=context.get("schedule", self.name),
            strategy=context.get("strategy", self.name),
            schedule_run_id=context.get(
                "schedule_run_id", f"{self.name}:{failed_at.isoformat()}"
            ),
            strategy_instance_id=context.get("strategy_instance_id"),
            telemetry_namespace=context.get(
                "telemetry_namespace", f"ai.scheduler.{self.name}"
            ),
            metadata=metadata,
        )
        try:
            self.decision_journal.record(event)
        except Exception:  # pragma: no cover - audyt nie może blokować treningu
            pass

    def _handle_failure(
        self,
        *,
        failed_at: datetime,
        error: Exception,
        dataset: FeatureDataset | None,
    ) -> None:
        self.scheduler.mark_failure(failed_at, reason=f"{type(error).__name__}: {error}")
        if self.decision_journal is None:
            return

        metadata: MutableMapping[str, str] = {
            "scheduler_version": str(self.scheduler.STATE_VERSION),
            "failure_streak": str(self.scheduler.failure_streak),
            "error_type": type(error).__name__,
            "next_run": self.scheduler.next_run().isoformat(),
        }
        if self.scheduler.last_run is not None:
            metadata["last_run"] = self.scheduler.last_run.isoformat()
        metadata["last_failure"] = failed_at.isoformat()
        if self.scheduler.last_failure_reason:
            metadata["last_failure_reason"] = self.scheduler.last_failure_reason
        if self.scheduler.cooldown_until is not None:
            metadata["cooldown_until"] = self.scheduler.cooldown_until.isoformat()
        if self.scheduler.updated_at is not None:
            metadata["state_updated_at"] = self.scheduler.updated_at.isoformat()
        if dataset is not None:
            metadata["dataset_rows"] = str(len(dataset.vectors))

        event = TradingDecisionEvent(
            event_type="ai_retraining_failed",
            timestamp=failed_at,
            environment=self.journal_environment,
            portfolio=self.journal_portfolio,
            risk_profile=self.journal_risk_profile,
            schedule=self.name,
            strategy=self.journal_strategy,
            schedule_run_id=f"{self.name}:{failed_at.isoformat()}",
            telemetry_namespace=f"ai.scheduler.{self.name}",
            metadata=metadata,
        )
        try:
            self.decision_journal.record(event)
        except Exception:  # pragma: no cover - audyt nie może blokować treningu
            pass


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
