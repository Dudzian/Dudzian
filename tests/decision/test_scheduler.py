from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Mapping, Sequence

import pytest


from bot_core.ai import (
    FeatureDataset,
    ModelTrainer,
    RetrainingScheduler,
    TrainingScheduler,
    WalkForwardValidator,
)
from bot_core.ai.audit import (
    list_audit_reports,
    load_latest_data_quality_report,
    load_latest_drift_report,
    load_latest_walk_forward_report,
    load_scheduler_state,
    save_data_quality_report,
    save_drift_report,
    save_scheduler_state,
    save_walk_forward_report,
    summarize_walk_forward_reports,
)
from bot_core.ai.feature_engineering import FeatureVector
from bot_core.ai.scheduler import (
    DEFAULT_JOURNAL_ENVIRONMENT,
    DEFAULT_JOURNAL_RISK_PROFILE,
    ScheduledTrainingJob,
    WalkForwardResult,
)
from bot_core.ai.data_monitoring import update_sign_off
from bot_core.ai.training import ExternalModelAdapter, ExternalTrainingContext, ExternalTrainingResult, register_external_model_adapter
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


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


def test_retraining_scheduler_marks_runs(tmp_path) -> None:
    scheduler = RetrainingScheduler(
        interval=timedelta(hours=6),
        persistence_path=tmp_path / "scheduler.json",
    )
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


def test_training_scheduler_executes_external_backend(tmp_path) -> None:
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

    schedule = RetrainingScheduler(
        interval=timedelta(minutes=5),
        persistence_path=tmp_path / "scheduler.json",
    )
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
    assert record.metrics
    assert record.metrics == dict(artifact.metrics.summary())


def test_scheduled_job_persists_state_and_records_journal(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_300_000,
                symbol="BTCUSDT",
                features={"momentum": 1.0},
                target_bps=0.5,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=30),
        persistence_path=persistence_path,
    )
    journal = InMemoryTradingDecisionJournal()
    job = ScheduledTrainingJob(
        name="btc-retrain",
        scheduler=scheduler,
        trainer_factory=lambda: ModelTrainer(learning_rate=0.1, n_estimators=5),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
    )
    now = datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc)

    artifact = job.run(now=now)

    assert persistence_path.exists(), "scheduler powinien zapisać stan do pliku"
    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["version"] == RetrainingScheduler.STATE_VERSION
    assert payload["last_run"] == artifact.trained_at.isoformat()
    assert payload["next_run"] == (
        artifact.trained_at + timedelta(minutes=30)
    ).isoformat()
    assert payload["interval"] == timedelta(minutes=30).total_seconds()
    assert payload["updated_at"] == artifact.trained_at.isoformat()
    assert payload["last_failure"] is None
    assert payload["failure_streak"] == 0
    assert payload["last_failure_reason"] is None
    assert payload["cooldown_until"] is None
    assert payload["paused_until"] is None
    assert payload["paused_reason"] is None

    exported_state = scheduler.export_state()
    assert exported_state["version"] == RetrainingScheduler.STATE_VERSION
    assert exported_state["last_run"] == artifact.trained_at.isoformat()
    assert exported_state["next_run"] == (
        artifact.trained_at + timedelta(minutes=30)
    ).isoformat()
    assert exported_state["interval"] == timedelta(minutes=30).total_seconds()
    assert exported_state["updated_at"] == artifact.trained_at.isoformat()
    assert exported_state["last_failure"] is None
    assert exported_state["failure_streak"] == 0
    assert exported_state["last_failure_reason"] is None
    assert exported_state["cooldown_until"] is None
    assert exported_state["paused_until"] is None
    assert exported_state["paused_reason"] is None

    exported = tuple(journal.export())
    assert len(exported) == 1, "powinien powstać wpis w decision journal"
    entry = exported[0]
    assert entry["event"] == "ai_retraining"
    assert entry["environment"] == DEFAULT_JOURNAL_ENVIRONMENT
    assert entry["portfolio"] == "btc-retrain"
    assert entry["risk_profile"] == DEFAULT_JOURNAL_RISK_PROFILE
    assert entry["schedule"] == "btc-retrain"
    assert entry["schedule_run_id"].startswith("btc-retrain:")
    expected_summary_metrics = dict(artifact.metrics.summary())
    assert expected_summary_metrics, "powinny istnieć metryki podsumowujące"
    assert entry["metric_mae"] == f"{expected_summary_metrics['mae']:.10f}"
    for metric_name, metric_value in expected_summary_metrics.items():
        key = f"metric_{metric_name}"
        assert key in entry
        assert entry[key] == f"{metric_value:.10f}"
    block_metrics = artifact.metrics.splits()
    for split_name, values in block_metrics.items():
        if split_name == "summary" or not values:
            continue
        for metric_name, metric_value in values.items():
            key = f"metric_{split_name}_{metric_name}"
            assert key in entry
            assert entry[key] == f"{metric_value:.10f}"
    assert entry["last_run"] == artifact.trained_at.isoformat()
    assert entry["next_run"] == (
        artifact.trained_at + timedelta(minutes=30)
    ).isoformat()
    assert entry["scheduler_version"] == str(RetrainingScheduler.STATE_VERSION)
    assert entry["state_updated_at"] == artifact.trained_at.isoformat()
    assert entry["failure_streak"] == "0"
    assert "cooldown_until" not in entry


def test_scheduled_job_respects_custom_journal_defaults(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_350_000,
                symbol="BTCUSDT",
                features={"momentum": 1.5},
                target_bps=0.25,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=45),
        persistence_path=tmp_path / "scheduler.json",
    )
    journal = InMemoryTradingDecisionJournal()
    job = ScheduledTrainingJob(
        name="btc-custom",
        scheduler=scheduler,
        trainer_factory=lambda: ModelTrainer(learning_rate=0.2, n_estimators=3),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
        journal_environment="ai-lab",
        journal_portfolio="btc-experiment",
        journal_risk_profile="balanced",
    )

    job.run(now=datetime(2024, 6, 2, 9, tzinfo=timezone.utc))

    exported = tuple(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["environment"] == "ai-lab"
    assert entry["portfolio"] == "btc-experiment"
    assert entry["risk_profile"] == "balanced"


def test_scheduled_job_applies_context_overrides(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_355_000,
                symbol="BTCUSDT",
                features={"momentum": 1.25},
                target_bps=0.35,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=15),
        persistence_path=tmp_path / "scheduler.json",
    )
    journal = InMemoryTradingDecisionJournal()
    context = {
        "environment": "ctx-env",
        "portfolio": "ctx-portfolio",
        "risk_profile": "ctx-risk",
        "strategy": "ctx-strategy",
        "schedule": "ctx-schedule",
        "schedule_run_id": "ctx-run",
        "telemetry_namespace": "ctx.namespace",
        "strategy_instance_id": "ctx-instance",
    }
    job = ScheduledTrainingJob(
        name="btc-context",
        scheduler=scheduler,
        trainer_factory=lambda: ModelTrainer(learning_rate=0.2, n_estimators=3),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
        decision_journal_context=context,
        journal_environment="ignored-env",
        journal_portfolio="ignored-portfolio",
        journal_risk_profile="ignored-risk",
    )

    assert job.journal_environment == "ctx-env"
    assert job.journal_portfolio == "ctx-portfolio"
    assert job.journal_risk_profile == "ctx-risk"
    assert job.journal_strategy == "ctx-strategy"

    now = datetime(2024, 6, 2, 11, tzinfo=timezone.utc)
    job.run(now=now)

    exported = tuple(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["environment"] == "ctx-env"
    assert entry["portfolio"] == "ctx-portfolio"
    assert entry["risk_profile"] == "ctx-risk"
    assert entry["schedule"] == "ctx-schedule"
    assert entry["strategy"] == "ctx-strategy"
    assert entry["schedule_run_id"] == "ctx-run"
    assert entry["telemetry_namespace"] == "ctx.namespace"
    assert entry["strategy_instance_id"] == "ctx-instance"


def test_scheduled_job_ignores_blank_context_overrides(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_356_000,
                symbol="BTCUSDT",
                features={"momentum": 0.75},
                target_bps=0.2,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=25),
        persistence_path=tmp_path / "scheduler.json",
    )
    journal = InMemoryTradingDecisionJournal()
    job = ScheduledTrainingJob(
        name="btc-blank",
        scheduler=scheduler,
        trainer_factory=lambda: ModelTrainer(learning_rate=0.15, n_estimators=4),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
        decision_journal_context={
            "environment": "",
            "portfolio": "",
            "risk_profile": "",
        },
        journal_environment="custom-env",
        journal_portfolio="custom-portfolio",
        journal_risk_profile="custom-risk",
    )

    assert job.journal_environment == "custom-env"
    assert job.journal_portfolio == "custom-portfolio"
    assert job.journal_risk_profile == "custom-risk"

    now = datetime(2024, 6, 2, 12, tzinfo=timezone.utc)
    job.run(now=now)

    exported = tuple(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["environment"] == "custom-env"
    assert entry["portfolio"] == "custom-portfolio"
    assert entry["risk_profile"] == "custom-risk"


def test_scheduled_job_handles_falsy_journal_values(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_360_000,
                symbol="BTCUSDT",
                features={"momentum": 0.0},
                target_bps=0.1,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=30),
        persistence_path=tmp_path / "scheduler.json",
    )
    journal = InMemoryTradingDecisionJournal()
    job = ScheduledTrainingJob(
        name="btc-falsy",
        scheduler=scheduler,
        trainer_factory=lambda: ModelTrainer(learning_rate=0.1, n_estimators=2),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
        journal_environment=0,
        journal_portfolio=0,
        journal_risk_profile=False,
    )

    job.run(now=datetime(2024, 6, 2, 10, tzinfo=timezone.utc))

    exported = tuple(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["environment"] == "0"
    assert entry["portfolio"] == "0"
    assert entry["risk_profile"] == "False"


def test_scheduled_job_records_failure_and_updates_state(tmp_path) -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1_700_400_000,
                symbol="BTCUSDT",
                features={"momentum": 2.0},
                target_bps=-0.75,
            ),
        ),
        metadata={"symbols": ["BTCUSDT"]},
    )
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=20),
        persistence_path=persistence_path,
    )
    previous_run = datetime(2024, 6, 30, 15, 0, tzinfo=timezone.utc)
    scheduler.mark_executed(previous_run)
    journal = InMemoryTradingDecisionJournal()

    class _FailingTrainer:
        backend = "failing"

        def train(self, _dataset: FeatureDataset):
            raise RuntimeError("boom during training")

    job = ScheduledTrainingJob(
        name="btc-fail",
        scheduler=scheduler,
        trainer_factory=lambda: _FailingTrainer(),
        dataset_provider=lambda: dataset,
        decision_journal=journal,
    )

    failure_time = datetime(2024, 6, 30, 18, 30, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError):
        job.run(now=failure_time)

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["version"] == RetrainingScheduler.STATE_VERSION
    assert payload["last_run"] == previous_run.isoformat()
    assert payload["updated_at"] == failure_time.isoformat()
    assert payload["last_failure"] == failure_time.isoformat()
    assert payload["failure_streak"] == 1
    assert payload["last_failure_reason"].startswith("RuntimeError: boom during training")
    expected_cooldown = failure_time + timedelta(minutes=20)
    assert payload["cooldown_until"] == expected_cooldown.isoformat()
    assert payload["paused_until"] is None
    assert payload["paused_reason"] is None

    exported_state = scheduler.export_state()
    assert exported_state["last_run"] == previous_run.isoformat()
    assert exported_state["updated_at"] == failure_time.isoformat()
    assert exported_state["last_failure"] == failure_time.isoformat()
    assert exported_state["failure_streak"] == 1
    assert exported_state["last_failure_reason"].startswith("RuntimeError: boom during training")
    assert exported_state["cooldown_until"] == expected_cooldown.isoformat()
    assert exported_state["paused_until"] is None
    assert exported_state["paused_reason"] is None

    assert scheduler.failure_streak == 1

    exported = tuple(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["event"] == "ai_retraining_failed"
    assert entry["environment"] == DEFAULT_JOURNAL_ENVIRONMENT
    assert entry["portfolio"] == "btc-fail"
    assert entry["risk_profile"] == DEFAULT_JOURNAL_RISK_PROFILE
    assert entry["schedule"] == "btc-fail"
    assert entry["error_type"] == "RuntimeError"
    assert "metric_mae" not in entry
    assert entry["failure_streak"] == "1"
    assert entry["last_failure"] == failure_time.isoformat()
    assert entry["last_run"] == previous_run.isoformat()
    assert entry["next_run"] == expected_cooldown.isoformat()
    assert entry["scheduler_version"] == str(RetrainingScheduler.STATE_VERSION)
    assert entry["last_failure_reason"].startswith("RuntimeError: boom during training")
    assert entry["cooldown_until"] == expected_cooldown.isoformat()


def test_scheduler_reload_uses_persisted_state(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    original_interval = timedelta(hours=2)
    scheduler = RetrainingScheduler(
        interval=original_interval,
        persistence_path=persistence_path,
    )
    executed_at = datetime(2024, 7, 1, 12, 30, tzinfo=timezone.utc)
    scheduler.mark_executed(executed_at)

    reloaded = RetrainingScheduler(
        interval=timedelta(minutes=15),
        persistence_path=persistence_path,
    )

    assert reloaded.interval == original_interval
    assert reloaded.last_run == executed_at
    assert reloaded.next_run(executed_at - timedelta(minutes=1)) == executed_at + original_interval


def test_scheduler_applies_failure_backoff(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=10),
        persistence_path=persistence_path,
    )
    last_run = datetime(2024, 7, 4, 12, 0, tzinfo=timezone.utc)
    scheduler.mark_executed(last_run)

    first_failure = last_run + timedelta(minutes=5)
    scheduler.mark_failure(first_failure, reason="transient")

    assert scheduler.cooldown_until == first_failure + timedelta(minutes=10)
    assert scheduler.should_retrain(first_failure + timedelta(minutes=9)) is False
    assert scheduler.should_retrain(first_failure + timedelta(minutes=11)) is True

    second_failure = first_failure + timedelta(minutes=11)
    scheduler.mark_failure(second_failure, reason="still failing")

    expected_second_cooldown = second_failure + timedelta(minutes=20)
    assert scheduler.cooldown_until == expected_second_cooldown
    assert scheduler.failure_streak == 2
    assert scheduler.should_retrain(second_failure + timedelta(minutes=19)) is False
    assert scheduler.should_retrain(second_failure + timedelta(minutes=21)) is True

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["cooldown_until"] == expected_second_cooldown.isoformat()
    assert payload["failure_streak"] == 2
    assert payload["paused_until"] is None
    assert payload["paused_reason"] is None


def test_scheduler_pause_and_resume(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=15),
        persistence_path=persistence_path,
    )

    scheduler.pause(duration=timedelta(minutes=45), reason="maintenance")

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    paused_until = datetime.fromisoformat(payload["paused_until"]).astimezone(timezone.utc)

    assert payload["paused_reason"] == "maintenance"
    assert scheduler.should_retrain(paused_until - timedelta(minutes=5)) is False
    assert scheduler.next_run(paused_until - timedelta(minutes=5)) == paused_until

    scheduler.resume(paused_until)

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["paused_until"] is None
    assert payload["paused_reason"] is None

    after_resume = paused_until + timedelta(minutes=1)
    assert scheduler.should_retrain(after_resume)


def test_scheduler_clears_pause_after_expiration(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=30),
        persistence_path=persistence_path,
    )

    scheduler.pause(duration=timedelta(minutes=15), reason="ops-window")
    paused_until = scheduler.paused_until
    assert paused_until is not None

    past_pause = paused_until + timedelta(minutes=10)
    assert scheduler.should_retrain(past_pause)

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["paused_until"] is None
    assert payload["paused_reason"] is None
    assert payload["updated_at"] == past_pause.isoformat()


def test_scheduler_pause_validation(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=10),
        persistence_path=persistence_path,
    )

    with pytest.raises(ValueError):
        scheduler.pause()

    with pytest.raises(ValueError):
        scheduler.pause(duration=timedelta(seconds=-1))

    in_past = datetime.now(timezone.utc) - timedelta(minutes=5)
    with pytest.raises(ValueError):
        scheduler.pause(until=in_past)

    with pytest.raises(ValueError):
        scheduler.pause(duration=timedelta(minutes=5), until=datetime.now(timezone.utc) + timedelta(minutes=10))

def test_scheduler_recovers_last_run_from_next_run(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=45),
        persistence_path=persistence_path,
    )
    executed_at = datetime(2024, 7, 2, 10, 0, tzinfo=timezone.utc)
    scheduler.mark_executed(executed_at)

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    payload.pop("last_run", None)
    payload["interval"] = str(payload["interval"])  # wymusza parsowanie z napisu
    payload["next_run"] = (executed_at + timedelta(minutes=45)).timestamp()
    payload["updated_at"] = executed_at.timestamp()
    persistence_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = RetrainingScheduler(
        interval=timedelta(hours=5),
        persistence_path=persistence_path,
    )

    assert reloaded.interval == timedelta(seconds=float(payload["interval"]))
    assert reloaded.last_run == executed_at
    assert reloaded.next_run(executed_at) == executed_at + reloaded.interval
    assert reloaded.updated_at == executed_at


def test_scheduler_update_interval_persists_and_updates_state(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=10),
        persistence_path=persistence_path,
    )
    executed_at = datetime(2024, 7, 3, 9, tzinfo=timezone.utc)
    scheduler.mark_executed(executed_at)

    scheduler.update_interval(timedelta(minutes=90))

    payload = json.loads(persistence_path.read_text(encoding="utf-8"))
    assert payload["interval"] == timedelta(minutes=90).total_seconds()
    assert payload["last_run"] == executed_at.isoformat()
    assert payload["updated_at"] == executed_at.isoformat()
    assert payload["last_failure"] is None
    assert payload["failure_streak"] == 0


def test_scheduler_rejects_non_positive_interval(tmp_path) -> None:
    persistence_path = tmp_path / "scheduler.json"
    with pytest.raises(ValueError):
        RetrainingScheduler(interval=timedelta(0), persistence_path=persistence_path)

    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=5),
        persistence_path=persistence_path,
    )
    with pytest.raises(ValueError):
        scheduler.update_interval(timedelta(seconds=-5))


def test_cross_validation_uses_external_adapter_for_each_fold() -> None:
    calls: List[int] = []

    def _tracking_train(context: ExternalTrainingContext) -> ExternalTrainingResult:
        calls.append(len(context.train_matrix))
        mean_target = sum(context.train_targets) / len(context.train_targets)
        model = _MeanModel(mean_target, context.feature_names, context.scalers)
        return ExternalTrainingResult(state={"mean": mean_target}, trained_model=model)

    def _tracking_load(
        state: Mapping[str, object],
        feature_names: Sequence[str],
        metadata: Mapping[str, object],
    ) -> _MeanModel:
        scalers_raw = metadata.get("feature_scalers", {})
        scalers: dict[str, tuple[float, float]] = {}
        if isinstance(scalers_raw, Mapping):
            for name, payload in scalers_raw.items():
                if not isinstance(payload, Mapping):
                    continue
                scalers[str(name)] = (
                    float(payload.get("mean", 0.0)),
                    float(payload.get("stdev", 0.0)),
                )
        return _MeanModel(float(state.get("mean", 0.0)), feature_names, scalers)

    register_external_model_adapter(
        ExternalModelAdapter(
            backend="tracking_regressor",
            train=_tracking_train,
            load=_tracking_load,
        )
    )

    vectors: list[FeatureVector] = []
    for idx in range(16):
        features = {"momentum": float(idx) / 4.0, "volume_ratio": 1.0 + (idx % 3) * 0.15}
        vectors.append(
            FeatureVector(
                timestamp=1_700_200_000 + idx * 120,
                symbol="BTCUSDT",
                features=features,
                target_bps=float(idx % 5 - 2),
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": ["BTCUSDT"]})

    trainer = ModelTrainer(backend="tracking_regressor", validation_split=0.2)
    artifact = trainer.train(dataset)

    cv_meta = artifact.metadata["cross_validation"]
    folds = cv_meta.get("folds", 0)
    assert folds > 0
    assert len(calls) >= folds + 1


def test_training_job_writes_walk_forward_audit(tmp_path: Path) -> None:
    vectors: list[FeatureVector] = []
    for idx in range(16):
        features = {"momentum": float(idx) / 4.0, "volume_ratio": 1.0 + (idx % 2) * 0.1}
        vectors.append(
            FeatureVector(
                timestamp=1_700_300_000 + idx * 60,
                symbol="BTCUSDT",
                features=features,
                target_bps=float(idx - 7),
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": ["BTCUSDT"]})

    def _validator_factory(data: FeatureDataset) -> WalkForwardValidator:
        return WalkForwardValidator(data, train_window=8, test_window=4)

    schedule = RetrainingScheduler(interval=timedelta(minutes=10))
    journal = InMemoryTradingDecisionJournal()
    job = ScheduledTrainingJob(
        name="btc-wf",
        scheduler=schedule,
        trainer_factory=lambda: ModelTrainer(backend="mean_regressor"),
        dataset_provider=lambda: dataset,
        validator_factory=_validator_factory,
        audit_root=tmp_path / "ai_decision",
        decision_journal=journal,
        decision_journal_context={
            "environment": "lab",
            "portfolio": "ai-lab",
            "risk_profile": "research",
            "strategy": "wf-validation",
        },
    )

    job.run(now=datetime(2024, 6, 1, 12, tzinfo=timezone.utc))

    root_dir = tmp_path / "ai_decision"
    for name in ("walk_forward", "data_quality", "drift"):
        assert (root_dir / name).is_dir(), f"powinien istnieć katalog audytu {name}"

    walk_dir = root_dir / "walk_forward"
    files = sorted(walk_dir.glob("*.json"))
    assert len(files) == 1, "powinien zostać wygenerowany pojedynczy raport walk-forward"

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["job_name"] == "btc-wf"
    assert payload["walk_forward"]["train_window"] == 8
    assert payload["walk_forward"]["test_window"] == 4
    assert payload["walk_forward"]["average_mae"] >= 0.0
    assert payload["dataset"]["rows"] == len(dataset.vectors)

    events = list(journal.export())
    assert events, "journal powinien otrzymać wpis o audycie"
    audit_event = events[-1]
    assert audit_event["event"] == "ai_walk_forward_report"
    assert audit_event["schedule"] == "btc-wf"
    assert Path(audit_event["report_path"]) == files[0]
    assert float(audit_event["average_mae"]) >= 0.0


def test_save_data_quality_report_serializes_payload(tmp_path: Path) -> None:
    dataset = FeatureDataset(
        vectors=(),
        metadata={"symbols": ["BTCUSDT"], "window_minutes": 60},
    )
    path = save_data_quality_report(
        [{"issue": "missing_rows", "count": 12, "symbols": ["BTCUSDT"]}],
        job_name="btc-quality",
        dataset=dataset,
        audit_root=tmp_path,
        summary={"total_gaps": 12, "max_gap_minutes": 30},
        source="ohlcv-lake",
        tags=("ohlcv", "gap-detection"),
    )

    assert path.parent == tmp_path / "data_quality"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "btc-quality"
    assert payload["source"] == "ohlcv-lake"
    assert payload["dataset"]["metadata"]["window_minutes"] == 60
    assert payload["summary"]["total_gaps"] == 12
    assert payload["issues"][0]["issue"] == "missing_rows"
    assert payload["tags"] == ["ohlcv", "gap-detection"]


def test_save_drift_report_serializes_metrics(tmp_path: Path) -> None:
    dataset = FeatureDataset(
        vectors=(),
        metadata={"symbols": ["ETHUSDT"]},
    )
    path = save_drift_report(
        {
            "return_distribution": {
                "p_value": 0.0125,
                "test_statistic": 3.1,
                "method": "ks_test",
            },
            "feature_scaling": {"psi": 0.4},
        },
        job_name="eth-drift",
        dataset=dataset,
        audit_root=tmp_path,
        baseline_window={"start": "2024-05-01", "end": "2024-05-07"},
        production_window={"start": "2024-06-01", "end": "2024-06-07"},
        detector="psi",
        threshold=0.2,
    )

    assert path.parent == tmp_path / "drift"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "eth-drift"
    assert payload["detector"] == "psi"
    assert payload["threshold"] == 0.2
    assert payload["baseline_window"]["start"] == "2024-05-01"
    assert payload["production_window"]["end"] == "2024-06-07"
    assert payload["metrics"]["return_distribution"]["method"] == "ks_test"
    assert payload["metrics"]["feature_scaling"]["psi"] == 0.4


def test_list_and_load_latest_audit_reports(tmp_path: Path) -> None:
    dataset = FeatureDataset(
        vectors=(),
        metadata={"symbols": ["BTCUSDT"], "window_minutes": 15},
    )
    first = save_walk_forward_report(
        WalkForwardResult(
            windows=({"mae": 0.5, "directional_accuracy": 0.6},),
            average_mae=0.5,
            average_directional_accuracy=0.6,
        ),
        job_name="wf-job",
        dataset=dataset,
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 1, 12, tzinfo=timezone.utc),
    )
    second = save_walk_forward_report(
        WalkForwardResult(
            windows=({"mae": 0.4, "directional_accuracy": 0.65},),
            average_mae=0.4,
            average_directional_accuracy=0.65,
        ),
        job_name="wf-job",
        dataset=dataset,
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 2, 12, tzinfo=timezone.utc),
    )

    paths = list_audit_reports("walk_forward", audit_root=tmp_path)
    assert paths == [second, first]
    payload = load_latest_walk_forward_report(audit_root=tmp_path)
    assert payload is not None
    assert payload["walk_forward"]["average_mae"] == 0.4

    save_data_quality_report(
        {"issue": "missing_rows", "count": 3},
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 1, 12, tzinfo=timezone.utc),
    )
    dq_path = save_data_quality_report(
        {"issue": "gap", "count": 1},
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 3, 10, tzinfo=timezone.utc),
    )
    dq_payload = load_latest_data_quality_report(audit_root=tmp_path)
    assert dq_payload is not None
    assert dq_payload["issues"][0]["issue"] == "gap"
    dq_paths = list_audit_reports("data_quality", audit_root=tmp_path)
    assert dq_paths[0] == dq_path

    save_drift_report(
        {"feature_drift": {"psi": 0.2}},
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 1, 9, tzinfo=timezone.utc),
    )
    drift_path = save_drift_report(
        {"feature_drift": {"psi": 0.1}},
        audit_root=tmp_path,
        generated_at=datetime(2024, 6, 5, 9, tzinfo=timezone.utc),
    )
    drift_payload = load_latest_drift_report(audit_root=tmp_path)
    assert drift_payload is not None
    assert drift_payload["metrics"]["feature_drift"]["psi"] == 0.1
    drift_paths = list_audit_reports("drift", audit_root=tmp_path)
    assert drift_paths[0] == drift_path


def test_save_and_load_scheduler_state(tmp_path: Path) -> None:
    state = {
        "version": 5,
        "last_run": "2024-06-01T12:00:00+00:00",
        "next_run": "2024-06-01T12:30:00+00:00",
        "interval": 1800.0,
        "updated_at": "2024-06-01T12:00:05+00:00",
        "failure_streak": 0,
    }

    path = save_scheduler_state(state, audit_root=tmp_path)
    assert path == tmp_path / "scheduler.json"
    loaded = load_scheduler_state(audit_root=tmp_path)
    assert loaded is not None
    assert loaded["next_run"] == state["next_run"]
    for subdir in ("walk_forward", "data_quality", "drift"):
        assert (tmp_path / subdir).is_dir()
