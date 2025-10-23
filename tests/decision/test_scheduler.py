from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    save_data_quality_report,
    save_drift_report,
    save_walk_forward_report,
)
from bot_core.ai.feature_engineering import FeatureVector
from bot_core.ai.scheduler import ScheduledTrainingJob, WalkForwardResult
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
