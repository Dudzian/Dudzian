from __future__ import annotations

import os

import pytest

os.environ.setdefault("BOT_CORE_MINIMAL_EXCHANGES", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_CORE", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_DECISION", "1")

from bot_core.ai import FeatureDataset, FeatureVector, ModelArtifact, ModelRepository, ModelTrainer
from bot_core.ai.manager import AIManager


def test_require_real_models_raises_when_repository_empty(tmp_path) -> None:
    ai_manager = AIManager(model_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError):
        ai_manager.require_real_models()


def test_require_real_models_passes_when_repository_loaded(tmp_path) -> None:
    repo_path = tmp_path / "repo"
    repo = ModelRepository(repo_path)
    vectors = [
        FeatureVector(
            timestamp=1_700_000_000 + idx,
            symbol="BTCUSDT",
            features={"momentum": float(idx), "volume_ratio": 1.0 + idx * 0.1},
            target_bps=float(idx % 3 - 1),
        )
        for idx in range(12)
    ]
    dataset = FeatureDataset(vectors=tuple(vectors), metadata={})
    trainer = ModelTrainer(n_estimators=5, validation_split=0.25)
    artifact = trainer.train(dataset)
    metadata = dict(artifact.metadata)
    metadata.update({"symbol": "BTCUSDT", "model_type": "trend_following"})
    enriched = ModelArtifact(
        feature_names=artifact.feature_names,
        model_state=artifact.model_state,
        trained_at=artifact.trained_at,
        metrics=artifact.metrics,
        metadata=metadata,
        target_scale=artifact.target_scale,
        training_rows=artifact.training_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        feature_scalers=artifact.feature_scalers,
        decision_journal_entry_id=artifact.decision_journal_entry_id,
        backend=artifact.backend,
    )
    repo.save(enriched, "btc_trend.json")

    ai_manager = AIManager(model_dir=tmp_path / "cache")
    loaded = ai_manager.ingest_model_repository(repo_path)
    assert loaded == 1
    ai_manager.require_real_models()
