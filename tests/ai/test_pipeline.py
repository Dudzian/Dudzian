"""Regression tests for training pipeline meta-labeling support."""

from __future__ import annotations

import pandas as pd

from bot_core.ai.inference import DecisionModelInference, ModelRepository
from bot_core.ai.pipeline import train_gradient_boosting_model


def test_train_gradient_boosting_model_emits_meta_labeling(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": [0.5, -0.3, 0.4, -0.2, 0.3, -0.1],
        }
    )
    artifact_path = train_gradient_boosting_model(
        frame,
        ["feature"],
        "target",
        output_dir=tmp_path,
        model_name="demo",
    )

    repository = ModelRepository(tmp_path)
    inference = DecisionModelInference(repository)
    inference.load_weights(artifact_path.name)

    meta_payload = inference.meta_labeling_payload
    assert meta_payload is not None
    assert "classifier" in meta_payload
    assert "subsets" in meta_payload
    subsets = meta_payload["subsets"]
    train_summary = subsets["train"]
    assert train_summary["samples"] > 0
    assert 0.0 <= train_summary["hit_rate"] <= 1.0
    total_samples = sum(summary["samples"] for summary in subsets.values())
    assert total_samples >= train_summary["samples"]

    score = inference.score({"feature": 0.25})
    assert 0.0 <= score.success_probability <= 1.0
    assert inference.last_meta_probability is not None


def test_train_gradient_boosting_model_meta_payload_without_classifier(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [0.0, 0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 0.0, 0.0],
        }
    )
    artifact_path = train_gradient_boosting_model(
        frame,
        ["feature"],
        "target",
        output_dir=tmp_path,
        model_name="flat",
    )

    repository = ModelRepository(tmp_path)
    inference = DecisionModelInference(repository)
    inference.load_weights(artifact_path.name)

    meta_payload = inference.meta_labeling_payload
    assert meta_payload is not None
    subsets = meta_payload["subsets"]
    assert subsets["train"]["hit_rate"] == 0.0
    classifier_payload = meta_payload.get("classifier")
    assert classifier_payload in (None, {})
    assert inference.meta_labeling_confidence == 0.0
    score = inference.score({"feature": 0.0})
    assert inference.last_meta_probability is None
    assert 0.0 <= score.success_probability <= 1.0
