from __future__ import annotations

import importlib

from bot_core.ai.models import ModelArtifact, ModelMetrics


def _sample_metrics_payload() -> dict[str, dict[str, float]]:
    return {
        "summary": {"mae": 1.5, "directional_accuracy": 0.6},
        "train": {"mae": 1.4},
        "validation": {},
        "test": {},
    }


def test_models_module_exports_model_metrics() -> None:
    module = importlib.import_module("bot_core.ai.models")

    assert "ModelMetrics" in getattr(module, "__all__", ())
    metrics_cls = getattr(module, "ModelMetrics")
    metrics = metrics_cls(_sample_metrics_payload())

    assert metrics["mae"] == 1.5
    assert metrics.splits()["train"]["mae"] == 1.4
    assert metrics.summary()["directional_accuracy"] == 0.6


def test_ai_package_exports_model_metrics() -> None:
    package = importlib.import_module("bot_core.ai")

    assert "ModelMetrics" in getattr(package, "__all__", ())
    metrics_cls = getattr(package, "ModelMetrics")
    metrics = metrics_cls(_sample_metrics_payload())

    assert metrics.summary()["mae"] == 1.5


def test_model_artifact_uses_model_metrics() -> None:
    metrics_instance = ModelMetrics(_sample_metrics_payload())

    artifact = ModelArtifact(
        feature_names=("f1",),
        model_state={},
        trained_at=0,
        metrics=metrics_instance,
        metadata={},
        target_scale=1.0,
        training_rows=1,
        validation_rows=0,
        test_rows=0,
        feature_scalers={"f1": (0.0, 1.0)},
    )

    assert artifact.metrics.summary()["mae"] == 1.5
    assert artifact.metrics.splits()["train"]["mae"] == 1.4
    assert artifact.metrics is metrics_instance


def test_model_metrics_accepts_flat_summary() -> None:
    metrics = ModelMetrics({"mae": 2.0, "directional_accuracy": "0.75"})

    assert metrics.summary()["mae"] == 2.0
    assert metrics["directional_accuracy"] == 0.75
    assert set(metrics.splits().keys()) == {"summary", "train", "validation", "test"}
