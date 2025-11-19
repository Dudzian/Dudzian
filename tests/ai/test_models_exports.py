from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot_core.ai.models import AIModels, ModelArtifact, ModelMetrics


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


def test_model_metrics_accepts_stage6_payload() -> None:
    payload = {
        "summary": {"mae": 2.0, "directional_accuracy": "0.75"},
        "train": {"mae": 1.8},
        "validation": {},
        "test": {},
    }
    metrics = ModelMetrics(payload)

    assert metrics.summary()["mae"] == 2.0
    assert metrics["directional_accuracy"] == 0.75
    assert metrics.splits()["train"]["mae"] == 1.8
    assert set(metrics.splits().keys()) == {"summary", "train", "validation", "test"}


def test_model_metrics_rejects_legacy_payload() -> None:
    payload = {
        "mae": 1.5,
        "validation": {"mae": 1.4},
    }

    with pytest.raises(ValueError) as excinfo:
        ModelMetrics(payload)

    message = str(excinfo.value)
    assert "Stage6" in message
    assert "mae" in message


def test_ai_models_train_predict_save(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    samples = 32
    seq_len = 4
    features = 3
    X = rng.normal(size=(samples, seq_len, features)).astype(np.float32)
    y = rng.normal(loc=0.0, scale=0.5, size=(samples,)).astype(np.float32)

    model = AIModels(input_size=features, seq_len=seq_len, model_type="gb")
    artifact = model.train(X, y)

    assert artifact.feature_names
    preds = model.predict(X)
    assert preds.shape == (samples,)

    frame = pd.DataFrame(
        rng.normal(size=(samples + seq_len, features)),
        columns=[f"f{i}" for i in range(features)],
    )
    series = model.predict_series(frame, list(frame.columns))
    assert len(series) == len(frame)

    path = tmp_path / "model.json"
    model.save_model(path)

    loaded = AIModels.load_model(path)
    loaded_preds = loaded.predict(X)
    assert loaded_preds.shape == preds.shape
