from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import pytest

from bot_core.ai import (
    AIManager,
    FeatureDataset,
    FeatureVector,
    ModelTrainer,
    RetrainingScheduler,
)
from bot_core.ai.scheduler import ScheduledTrainingJob
from bot_core.alerts.dispatcher import AlertSeverity
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator


class RecordingDecisionOrchestrator(DecisionOrchestrator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.performance_updates: list[tuple[str, dict[str, float], str | None, str | None, datetime | None]] = []

    def update_model_performance(
        self,
        name: str,
        metrics: Mapping[str, float],
        *,
        strategy: str | None = None,
        risk_profile: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.performance_updates.append(
            (name, dict(metrics), strategy, risk_profile, timestamp)
        )
        super().update_model_performance(
            name,
            metrics,
            strategy=strategy,
            risk_profile=risk_profile,
            timestamp=timestamp,
        )


def _dataset(symbol: str = "BTCUSDT") -> FeatureDataset:
    vectors: list[FeatureVector] = []
    for idx in range(16):
        vectors.append(
            FeatureVector(
                timestamp=1_700_000_000 + idx * 60,
                symbol=symbol,
                features={"momentum": float(idx) / 10.0, "volume_ratio": 1.0 + (idx % 3) * 0.1},
                target_bps=float(idx % 5 - 2),
            )
        )
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={"symbols": [symbol]})
    dataset.metadata["target_scale"] = dataset.target_scale
    return dataset


def _config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=20.0,
        min_net_edge_bps=1.5,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.2,
        max_position_ratio=0.8,
        max_open_positions=8,
        max_latency_ms=500.0,
        max_trade_notional=50_000.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.4,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def test_ai_manager_registers_and_runs_training_job(tmp_path: Path) -> None:
    ai_manager = AIManager(model_dir=tmp_path / "cache")
    dataset = _dataset()
    schedule = RetrainingScheduler(interval=timedelta(minutes=15))

    job = ai_manager.register_training_job(
        "btc-trend",
        schedule,
        lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=6, validation_split=0.25),
        symbol="BTCUSDT",
        model_type="trend_following",
        repository_base=tmp_path / "repo",
        artifact_metadata={"risk_profile": "balanced"},
    )

    assert isinstance(job, ScheduledTrainingJob)
    assert ai_manager.list_training_jobs() == ("btc-trend",)

    now = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)

    assert len(results) == 1
    run_job, artifact, path = results[0]
    assert run_job is job
    assert path is not None and path.exists()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload["metadata"]
    assert metadata["job_name"] == "btc-trend"
    assert metadata["model_type"] == "trend_following"
    assert metadata["symbol"] == "btcusdt"
    assert metadata["risk_profile"] == "balanced"
    assert metadata["schedule_interval_seconds"] == pytest.approx(900.0)
    ai_manager.require_real_models()
    assert ai_manager.get_last_trained_artifact_path("btc-trend") == path


def test_training_job_attaches_decision_inference(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ai_manager = AIManager(model_dir=cache_dir)
    orchestrator = DecisionOrchestrator(_config())
    ai_manager.attach_decision_orchestrator(orchestrator)

    dataset = _dataset("ETHUSDT")
    schedule = RetrainingScheduler(interval=timedelta(minutes=30))

    ai_manager.register_training_job(
        "eth-decision",
        schedule,
        lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=4),
        symbol="ETHUSDT",
        model_type="mean_reversion",
        attach_to_decision=True,
        decision_name="eth-live",
        decision_repository_root=tmp_path / "decision_repo",
        set_default_decision=True,
    )

    now = datetime(2024, 6, 1, 10, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)
    assert results, "zadanie powinno zostać uruchomione"

    score = ai_manager.score_decision_features(
        {"momentum": 0.5, "volume_ratio": 1.1},
        model_name="eth-live",
    )
    assert score.expected_return_bps == pytest.approx(float(score.expected_return_bps))
    assert 0.0 <= score.success_probability <= 1.0


def test_training_job_updates_strategy_performance(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ai_manager = AIManager(model_dir=cache_dir)
    orchestrator = RecordingDecisionOrchestrator(_config())
    ai_manager.attach_decision_orchestrator(orchestrator)

    dataset = _dataset("LTCUSDT")
    schedule = RetrainingScheduler(interval=timedelta(minutes=45))

    ai_manager.register_training_job(
        "ltc-decision",
        schedule,
        lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=5),
        symbol="LTCUSDT",
        model_type="mean_reversion",
        attach_to_decision=True,
        decision_name="ltc-live",
        decision_repository_root=tmp_path / "decision_repo",
        set_default_decision=True,
        artifact_metadata={"risk_profile": "aggressive"},
    )

    now = datetime(2024, 7, 1, 10, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)
    assert results, "zadanie powinno zostać uruchomione"
    assert orchestrator.performance_updates, "powinny zostać zapisane metryki skuteczności"

    name, metrics, strategy, risk_profile, timestamp = orchestrator.performance_updates[-1]
    assert name == "ltc-live"
    assert strategy == "mean_reversion"
    assert risk_profile == "aggressive"
    assert timestamp is not None
    assert metrics["mae"] >= 0.0
    assert metrics["directional_accuracy"] >= 0.0
    assert "cv_mae_mean" in metrics
    assert "cv_directional_accuracy_mean" in metrics


def test_schedule_walk_forward_retraining_persists_quality_thresholds(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ai_manager = AIManager(model_dir=cache_dir)
    dataset = _dataset("ADAUSDT")

    ai_manager.schedule_walk_forward_retraining(
        "ada-quality",
        interval=timedelta(minutes=20),
        dataset_provider=lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=4),
        quality_thresholds={
            "min_directional_accuracy": 0.7,
            "max_mae": 5.5,
        },
        repository_base=tmp_path / "quality_repo",
        model_type="trend_following",
        symbol="ADAUSDT",
    )

    now = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)
    assert results, "zadanie retrainingu powinno zostać wykonane"

    artifact_path = ai_manager.get_last_trained_artifact_path("ada-quality")
    assert artifact_path is not None and artifact_path.exists()

    metadata = json.loads(artifact_path.read_text(encoding="utf-8"))["metadata"]
    thresholds = metadata.get("quality_thresholds")
    assert thresholds is not None
    assert thresholds["min_directional_accuracy"] == pytest.approx(0.7)
    assert thresholds["max_mae"] == pytest.approx(5.5)


def test_schedule_walk_forward_retraining_emits_quality_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    ai_manager = AIManager(model_dir=cache_dir)
    dataset = _dataset("SOLUSDT")
    captured: list[tuple[str, AlertSeverity, str, dict[str, str]]] = []

    def _capture(
        message: str,
        *,
        severity: AlertSeverity,
        source: str,
        context: Mapping[str, str],
        exception: object | None = None,
    ) -> None:
        captured.append((message, severity, source, dict(context)))

    monkeypatch.setattr("bot_core.ai.manager.emit_alert", _capture)

    ai_manager.schedule_walk_forward_retraining(
        "sol-quality",
        interval=timedelta(minutes=10),
        dataset_provider=lambda: dataset,
        trainer_factory=lambda: ModelTrainer(n_estimators=3),
        quality_thresholds={
            "min_directional_accuracy": 0.99,
            "max_mae": 0.001,
        },
        repository_base=tmp_path / "alert_repo",
        model_type="momentum",
        symbol="SOLUSDT",
    )

    now = datetime(2024, 9, 1, 9, tzinfo=timezone.utc)
    results = ai_manager.run_due_training_jobs(now)
    assert results, "zadanie retrainingu powinno zostać wykonane"

    error_events = [event for event in captured if event[1] is AlertSeverity.ERROR]
    assert error_events, "powinien zostać wygenerowany alert degradacji jakości"
    assert any(event[3].get("model") == "momentum" for event in error_events)
