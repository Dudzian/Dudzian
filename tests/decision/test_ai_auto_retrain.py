import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot_core.ai.manager import AIManager
from bot_core.ai.pipeline import QualityThresholds, load_training_manifest
from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.runtime.schedulers import AutoRetrainResult, AutoRetrainScheduler


def _build_manifest(tmp_path: Path) -> tuple[Path, Path]:
    frame = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.05, 0.3, 0.15, 0.22, 0.11, 0.05],
            "f2": [1.0, 0.9, 1.1, 0.95, 1.05, 0.97, 1.02, 0.93],
            "target": [0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.45, 0.25],
        }
    )
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    manifest_payload = {
        "profiles": {
            "demo": {
                "datasets": {"ohlcv": {"path": str(dataset_path)}},
                "models": [
                    {
                        "name": "alpha",
                        "dataset": "ohlcv",
                        "target": "target",
                        "features": ["f1", "f2"],
                        "metadata": {"symbol": "BTCUSDT"},
                    },
                    {
                        "name": "beta",
                        "dataset": "ohlcv",
                        "target": "target",
                        "features": ["f1"],
                        "metadata": {"symbol": "BTCUSDT"},
                    },
                ],
                "ensembles": [
                    {
                        "name": "combo",
                        "components": ["alpha", "beta"],
                        "aggregation": "weighted",
                        "weights": [0.6, 0.4],
                    }
                ],
                "auto_retrain": {
                    "interval_seconds": 60.0,
                    "quality": {
                        "min_directional_accuracy": 0.0,
                        "max_mae": 10.0,
                    },
                    "journal": {
                        "environment": "paper",
                        "portfolio": "ai",
                        "risk_profile": "ai-test",
                        "strategy": "combo",
                    },
                },
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return manifest_path, dataset_path


def _build_manager(tmp_path: Path, journal: InMemoryTradingDecisionJournal) -> AIManager:
    return AIManager(
        ai_threshold_bps=5.0,
        model_dir=tmp_path / "models",
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper",
            "portfolio": "ai",
            "risk_profile": "ai-test",
        },
    )


def test_auto_retrain_scheduler_registers_models_and_logs(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    manifest = load_training_manifest(manifest_path)
    journal = InMemoryTradingDecisionJournal()
    manager = _build_manager(tmp_path, journal)
    scheduler = AutoRetrainScheduler(
        manifest,
        output_dir=tmp_path / "artifacts",
        ai_manager=manager,
        journal=journal,
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    scheduler.register_profile("demo")
    results = scheduler.run_pending()
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, AutoRetrainResult)
    assert result.success is True
    assert not result.failures
    ensembles = manager.list_ensembles()
    assert "combo" in ensembles
    events = tuple(journal.export())
    assert events, "decision journal should contain at least one event"
    last_event = events[-1]
    assert last_event["event"] == "ai_auto_retrain_succeeded"
    assert last_event["status"] == "success"


def test_auto_retrain_scheduler_flags_quality_failure(tmp_path: Path) -> None:
    manifest_path, _ = _build_manifest(tmp_path)
    manifest = load_training_manifest(manifest_path)
    base_policy = manifest.profile("demo").auto_retrain
    assert base_policy is not None
    strict_policy = replace(
        base_policy,
        quality=QualityThresholds(
            min_directional_accuracy=0.95,
            max_mae=0.01,
            max_rmse=None,
        ),
    )
    journal = InMemoryTradingDecisionJournal()
    scheduler = AutoRetrainScheduler(
        manifest,
        output_dir=tmp_path / "strict-artifacts",
        ai_manager=None,
        journal=journal,
        clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    scheduler.register_profile("demo", policy=strict_policy)
    results = scheduler.run_pending()
    assert results and results[0].success is False
    assert results[0].failures
    events = tuple(journal.export())
    assert events and events[-1]["event"] == "ai_auto_retrain_failed"
    assert "quality_failures" in events[-1]
