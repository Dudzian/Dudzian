from __future__ import annotations

from datetime import datetime, timezone

from bot_core.ai import ModelMetrics
from bot_core.ai.models import ModelArtifact


def test_model_metrics_import_and_usage() -> None:
    metrics_payload = {
        "summary": {"mae": 1.23, "directional_accuracy": 0.6},
        "train": {"mae": 1.23, "directional_accuracy": 0.6},
        "validation": {"mae": 1.25},
        "test": {"mae": 1.27},
    }

    artifact = ModelArtifact(
        feature_names=("f1", "f2"),
        model_state={"weights": [0.1, 0.2], "bias": 0.0},
        trained_at=datetime.now(timezone.utc),
        metrics=metrics_payload,
        metadata={},
        target_scale=1.0,
        training_rows=128,
        validation_rows=64,
        test_rows=32,
        feature_scalers={"f1": (0.0, 1.0), "f2": (0.0, 1.0)},
        decision_journal_entry_id=None,
        backend="builtin",
    )

    assert isinstance(artifact.metrics, ModelMetrics)
    assert artifact.metrics["mae"] == 1.23
    assert artifact.metrics.blocks()["validation"]["mae"] == 1.25
    assert "directional_accuracy" in artifact.metrics.summary()
    metrics_dict = artifact.metrics.to_dict()
    assert metrics_dict["summary"]["mae"] == 1.23
    assert set(metrics_dict) == {"summary", "train", "validation", "test"}


def test_model_metrics_flattening_without_summary() -> None:
    metrics_payload = {
        "train": {"mae": 2.5, "rmse": 3.1},
        "validation": {"mae": 2.6},
    }

    artifact = ModelArtifact(
        feature_names=("x",),
        model_state={},
        trained_at=datetime.now(timezone.utc),
        metrics=metrics_payload,
        metadata={},
        target_scale=1.0,
        training_rows=10,
        validation_rows=5,
        test_rows=0,
        feature_scalers={},
        decision_journal_entry_id=None,
        backend="builtin",
    )

    assert "mae" in artifact.metrics
    assert artifact.metrics["mae"] == 2.6
    assert artifact.metrics.as_flat_dict() == {"mae": 2.6, "rmse": 3.1}
    assert artifact.metrics.as_flat_dict(prefer=("train",)) == {
        "mae": 2.5,
        "rmse": 3.1,
    }


def test_model_metrics_summary_is_hydrated_from_other_splits() -> None:
    metrics_payload = {
        "train": {"mae": 1.0},
        "validation": {"rmse": 2.0},
        "test": {"directional_accuracy": 0.52},
    }

    metrics = ModelMetrics.from_raw(metrics_payload)

    summary = metrics.summary()
    # ``summary`` powinno preferować walidację jako główne źródło.
    assert summary["rmse"] == 2.0
    # Brakujące metryki są uzupełniane wartościami z innych bloków.
    assert summary["mae"] == 1.0
    assert summary["directional_accuracy"] == 0.52


def test_model_metrics_summary_respects_existing_values() -> None:
    metrics_payload = {
        "summary": {"mae": 1.5},
        "validation": {"mae": 1.7, "rmse": 2.5},
    }

    metrics = ModelMetrics.from_raw(metrics_payload)

    summary = metrics.summary()
    # ``mae`` pochodzi z oryginalnego podsumowania.
    assert summary["mae"] == 1.5
    # Dodatkowe metryki są uzupełniane z innych bloków.
    assert summary["rmse"] == 2.5
