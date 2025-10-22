from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bot_core.ai.pipeline import register_model_artifact, train_gradient_boosting_model


def _build_sample_frame(rows: int = 60) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        data.append({"f1": float(idx), "f2": float(idx % 5), "target": float(idx % 3 - 1)})
    return pd.DataFrame(data)


def test_train_gradient_boosting_model_generates_split_metrics(tmp_path: Path) -> None:
    frame = _build_sample_frame()
    artifact_path = train_gradient_boosting_model(
        frame,
        ["f1", "f2"],
        "target",
        output_dir=tmp_path,
        model_name="demo",
    )
    payload = json.loads(Path(artifact_path).read_text())
    metrics = payload["metrics"]
    assert "train_mae" in metrics
    assert "validation_mae" in metrics
    assert "test_mae" in metrics
    assert metrics["mae"] == metrics["train_mae"]
    metadata = payload["metadata"]
    assert metadata["dataset_split"]["validation_ratio"] == 0.15
    assert metadata["dataset_split"]["test_ratio"] == 0.15
    assert metadata["drift_monitor"]["threshold"] == 3.5
    assert metadata["quality_thresholds"]["min_directional_accuracy"] == 0.55


def test_register_model_artifact_reports_metrics(tmp_path: Path) -> None:
    frame = _build_sample_frame()
    artifact_path = train_gradient_boosting_model(
        frame,
        ["f1", "f2"],
        "target",
        output_dir=tmp_path,
        model_name="demo",
    )

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.attached: dict[str, bool] = {}
            self.metrics: dict[str, dict[str, float]] = {}

        def attach_named_inference(self, name: str, inference, *, set_default: bool = False) -> None:
            self.attached[name] = bool(set_default)

        def update_model_performance(self, name: str, metrics) -> None:
            self.metrics[name] = dict(metrics)

    orchestrator = _StubOrchestrator()
    inference = register_model_artifact(
        orchestrator,
        Path(artifact_path),
        name="demo",
        repository_root=tmp_path,
        set_default=True,
    )
    assert orchestrator.attached["demo"] is True
    assert "demo" in orchestrator.metrics
    assert "test_mae" in orchestrator.metrics["demo"]
    assert getattr(inference, "model_label", "") == "demo"
