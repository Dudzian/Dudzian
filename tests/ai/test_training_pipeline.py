from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.scheduler import RetrainingScheduler, WalkForwardValidator
from bot_core.ai.training import ModelTrainer, WalkForwardTrainingCoordinator
from bot_core.ai.validation import (
    ModelQualityMonitor,
    ModelQualityReport,
    load_latest_quality_report,
    record_model_quality_report,
)
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator


def _build_dataset() -> FeatureDataset:
    vectors: list[FeatureVector] = []
    for idx in range(40):
        timestamp = float(idx)
        features = {"close": float(idx), "volume": float(idx % 7)}
        target = 0.8 if idx % 3 == 0 else -0.4
        vectors.append(
            FeatureVector(
                timestamp=timestamp,
                symbol="BTC/USDT",
                features=features,
                target_bps=target,
            )
        )
    return FeatureDataset(vectors=tuple(vectors), metadata={})


@pytest.fixture()
def dataset() -> FeatureDataset:
    return _build_dataset()


def test_walk_forward_coordinator_runs_and_publishes(tmp_path: Path, dataset: FeatureDataset) -> None:
    scheduler_state = tmp_path / "scheduler.json"
    scheduler = RetrainingScheduler(
        interval=timedelta(hours=2),
        persistence_path=scheduler_state,
    )

    monitor = ModelQualityMonitor(
        model_name="demo",
        history_root=tmp_path / "quality",
        directional_tolerance=0.01,
        mae_tolerance=0.2,
    )

    coordinator = WalkForwardTrainingCoordinator(
        job_name="demo",
        trainer_factory=lambda: ModelTrainer(validation_split=0.1, test_split=0.1),
        dataset_provider=lambda: dataset,
        base_path=tmp_path / "models",
        scheduler=scheduler,
        validator_factory=lambda data: WalkForwardValidator(
            data,
            train_window=24,
            test_window=12,
        ),
        quality_monitor=monitor,
        aliases=("latest",),
        directional_tolerance=0.01,
        mae_tolerance=0.2,
    )

    outcome_first = coordinator.tick()
    assert outcome_first is not None
    assert outcome_first.report.status in {"ok", "improved"}
    assert outcome_first.champion_decision is not None
    assert outcome_first.champion_decision.decision == "champion"

    manifest = coordinator.repository.get_manifest()
    assert manifest.get("active") == outcome_first.version

    latest_report = load_latest_quality_report("demo", history_root=tmp_path / "quality")
    assert latest_report is not None
    assert latest_report.version == outcome_first.version

    # Harmonogram nie powinien uruchomić kolejnego treningu natychmiast po pierwszym przebiegu.
    assert coordinator.tick() is None

    # Sztuczny dryf metryk powinien wymusić retrening niezależnie od harmonogramu.
    degraded_metrics = {
        "directional_accuracy": 0.0,
        "mae": outcome_first.report.metrics.get("mae", 0.0) + 1.0,
    }
    outcome_second = coordinator.tick(production_metrics=degraded_metrics)
    assert outcome_second is not None
    assert outcome_second.version != outcome_first.version

    latest_report = load_latest_quality_report("demo", history_root=tmp_path / "quality")
    assert latest_report is not None
    assert latest_report.version == outcome_second.version

    degraded_report = ModelQualityReport(
        model_name="demo",
        version=outcome_second.version,
        evaluated_at=datetime.now(timezone.utc),
        metrics={"mae": outcome_second.report.metrics.get("mae", 0.0) + 1.0, "directional_accuracy": 0.1},
        status="degraded",
        baseline_version=outcome_first.version,
        delta={"mae": 1.0, "directional_accuracy": -0.5},
    )
    decision = record_model_quality_report(degraded_report, history_root=tmp_path / "quality")
    assert decision.decision == "challenger"

    orchestrator = DecisionOrchestrator(
        DecisionEngineConfig(
            orchestrator=DecisionOrchestratorThresholds(
                max_cost_bps=100.0,
                min_net_edge_bps=0.0,
                max_daily_loss_pct=10.0,
                max_drawdown_pct=20.0,
                max_position_ratio=1.0,
                max_open_positions=5,
            )
        )
    )
    selected_version, loaded_report = orchestrator.load_repository_inference(
        coordinator.repository,
        model_name="demo",
        quality_history=tmp_path / "quality",
    )
    assert loaded_report is not None and loaded_report.status == "degraded"
    assert selected_version == outcome_first.version
