from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from bot_core.ai.inference import DecisionModelInference, ModelRepository
from bot_core.ai.pipeline import train_gradient_boosting_model


def _prepare_artifact(tmp_path: Path, threshold: float) -> Path:
    frame = pd.DataFrame(
        {
            "f1": [float(i) for i in range(80)],
            "f2": [float((i % 3) * 2) for i in range(80)],
            "target": [float((i % 5) - 2) for i in range(80)],
        }
    )
    return train_gradient_boosting_model(
        frame,
        ["f1", "f2"],
        "target",
        output_dir=tmp_path,
        model_name="drift",
        metadata={
            "drift_monitor": {
                "threshold": threshold,
                "window": 2,
                "min_observations": 1,
                "cooldown": 1,
                "backend": "decision_engine",
            }
        },
    )


def test_drift_monitor_emits_alert(monkeypatch, tmp_path: Path) -> None:
    artifact_path = _prepare_artifact(tmp_path, threshold=0.1)
    repository = ModelRepository(tmp_path)
    inference = DecisionModelInference(repository)
    inference.model_label = "drift"
    captured: List[object] = []

    def _capture(payload) -> None:
        captured.append(payload)

    monkeypatch.setattr("bot_core.ai.inference.emit_model_drift_alert", _capture)
    inference.load_weights(Path(artifact_path))
    for _ in range(3):
        inference.score({"f1": 10_000.0, "f2": 10_000.0})
    assert captured, "drift monitor should emit alert when z-score crosses threshold"
    assert inference.last_drift_score is not None
