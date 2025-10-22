from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
    assert "train_mse" in metrics
    assert "validation_mae" in metrics
    assert "validation_mse" in metrics
    assert "test_mae" in metrics
    assert "test_mse" in metrics
    assert "train_mape" in metrics
    assert "validation_mape" in metrics
    assert "test_mape" in metrics
    assert "train_mase" in metrics
    assert "validation_mase" in metrics
    assert "test_mase" in metrics
    assert "train_r2" in metrics
    assert "validation_r2" in metrics
    assert "test_r2" in metrics
    assert "train_median_absolute_error" in metrics
    assert "validation_median_absolute_error" in metrics
    assert "test_median_absolute_error" in metrics
    assert "train_explained_variance" in metrics
    assert "validation_explained_variance" in metrics
    assert "test_explained_variance" in metrics
    assert "train_max_error" in metrics
    assert "validation_max_error" in metrics
    assert "test_max_error" in metrics
    assert "train_smape" in metrics
    assert "validation_smape" in metrics
    assert "test_smape" in metrics
    assert "train_mean_bias_error" in metrics
    assert "validation_mean_bias_error" in metrics
    assert "test_mean_bias_error" in metrics
    assert "train_wmape" in metrics
    assert "validation_wmape" in metrics
    assert "test_wmape" in metrics
    assert "train_mpe" in metrics
    assert "validation_mpe" in metrics
    assert "test_mpe" in metrics
    assert "train_rmspe" in metrics
    assert "validation_rmspe" in metrics
    assert "test_rmspe" in metrics
    assert "train_median_percentage_error" in metrics
    assert "validation_median_percentage_error" in metrics
    assert "test_median_percentage_error" in metrics
    assert "train_median_absolute_percentage_error" in metrics
    assert "validation_median_absolute_percentage_error" in metrics
    assert "test_median_absolute_percentage_error" in metrics
    assert "train_msle" in metrics
    assert "validation_msle" in metrics
    assert "test_msle" in metrics
    assert "train_mean_absolute_log_error" in metrics
    assert "validation_mean_absolute_log_error" in metrics
    assert "test_mean_absolute_log_error" in metrics
    assert "train_mean_poisson_deviance" in metrics
    assert "validation_mean_poisson_deviance" in metrics
    assert "test_mean_poisson_deviance" in metrics
    assert "train_mean_gamma_deviance" in metrics
    assert "validation_mean_gamma_deviance" in metrics
    assert "test_mean_gamma_deviance" in metrics
    assert "train_mean_tweedie_deviance" in metrics
    assert "validation_mean_tweedie_deviance" in metrics
    assert "test_mean_tweedie_deviance" in metrics
    assert "mape" in metrics
    assert "r2" in metrics
    assert "median_absolute_error" in metrics
    assert "explained_variance" in metrics
    assert "max_error" in metrics
    assert "smape" in metrics
    assert "mean_bias_error" in metrics
    assert "wmape" in metrics
    assert "mpe" in metrics
    assert "rmspe" in metrics
    assert "median_percentage_error" in metrics
    assert "median_absolute_percentage_error" in metrics
    assert "mse" in metrics
    assert "mase" in metrics
    assert "msle" in metrics
    assert "mean_absolute_log_error" in metrics
    assert "mean_poisson_deviance" in metrics
    assert "mean_gamma_deviance" in metrics
    assert "mean_tweedie_deviance" in metrics
    assert metrics["mae"] == metrics["train_mae"]
    metadata = payload["metadata"]
    assert "train_metrics" in metadata
    assert "validation_metrics" in metadata
    assert "test_metrics" in metadata
    assert metadata["dataset_split"]["validation_ratio"] == 0.15
    assert metadata["dataset_split"]["test_ratio"] == 0.15
    assert metadata["drift_monitor"]["threshold"] == 3.5
    assert metadata["quality_thresholds"]["min_directional_accuracy"] == 0.55
    assert metadata["train_metrics"]["r2"] == pytest.approx(metrics["train_r2"])
    assert metadata["validation_metrics"]["r2"] == pytest.approx(metrics["validation_r2"])
    assert metadata["test_metrics"]["r2"] == pytest.approx(metrics["test_r2"])
    assert metadata["train_metrics"]["mse"] == pytest.approx(metrics["train_mse"])
    assert metadata["validation_metrics"]["mse"] == pytest.approx(metrics["validation_mse"])
    assert metadata["test_metrics"]["mse"] == pytest.approx(metrics["test_mse"])
    assert metadata["train_metrics"]["median_absolute_error"] == pytest.approx(
        metrics["train_median_absolute_error"]
    )
    assert metadata["validation_metrics"]["median_absolute_error"] == pytest.approx(
        metrics["validation_median_absolute_error"]
    )
    assert metadata["test_metrics"]["median_absolute_error"] == pytest.approx(
        metrics["test_median_absolute_error"]
    )
    assert metadata["train_metrics"]["explained_variance"] == pytest.approx(
        metrics["train_explained_variance"]
    )
    assert metadata["validation_metrics"]["explained_variance"] == pytest.approx(
        metrics["validation_explained_variance"]
    )
    assert metadata["test_metrics"]["explained_variance"] == pytest.approx(
        metrics["test_explained_variance"]
    )
    assert metadata["train_metrics"]["max_error"] == pytest.approx(
        metrics["train_max_error"]
    )
    assert metadata["validation_metrics"]["max_error"] == pytest.approx(
        metrics["validation_max_error"]
    )
    assert metadata["test_metrics"]["max_error"] == pytest.approx(
        metrics["test_max_error"]
    )
    assert metadata["train_metrics"]["smape"] == pytest.approx(metrics["train_smape"])
    assert metadata["validation_metrics"]["smape"] == pytest.approx(
        metrics["validation_smape"]
    )
    assert metadata["test_metrics"]["smape"] == pytest.approx(metrics["test_smape"])
    assert metadata["train_metrics"]["mean_bias_error"] == pytest.approx(
        metrics["train_mean_bias_error"]
    )
    assert metadata["validation_metrics"]["mean_bias_error"] == pytest.approx(
        metrics["validation_mean_bias_error"]
    )
    assert metadata["test_metrics"]["mean_bias_error"] == pytest.approx(
        metrics["test_mean_bias_error"]
    )
    assert metadata["train_metrics"]["wmape"] == pytest.approx(metrics["train_wmape"])
    assert metadata["validation_metrics"]["wmape"] == pytest.approx(
        metrics["validation_wmape"]
    )
    assert metadata["test_metrics"]["wmape"] == pytest.approx(metrics["test_wmape"])
    assert metadata["train_metrics"]["mpe"] == pytest.approx(metrics["train_mpe"])
    assert metadata["validation_metrics"]["mpe"] == pytest.approx(
        metrics["validation_mpe"]
    )
    assert metadata["test_metrics"]["mpe"] == pytest.approx(metrics["test_mpe"])
    assert metadata["train_metrics"]["rmspe"] == pytest.approx(metrics["train_rmspe"])
    assert metadata["validation_metrics"]["rmspe"] == pytest.approx(
        metrics["validation_rmspe"]
    )
    assert metadata["test_metrics"]["rmspe"] == pytest.approx(metrics["test_rmspe"])
    assert metadata["train_metrics"]["median_percentage_error"] == pytest.approx(
        metrics["train_median_percentage_error"]
    )
    assert metadata["validation_metrics"]["median_percentage_error"] == pytest.approx(
        metrics["validation_median_percentage_error"]
    )
    assert metadata["test_metrics"]["median_percentage_error"] == pytest.approx(
        metrics["test_median_percentage_error"]
    )
    assert metadata["train_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        metrics["train_median_absolute_percentage_error"]
    )
    assert metadata["train_metrics"]["msle"] == pytest.approx(metrics["train_msle"])
    assert metadata["train_metrics"]["mean_absolute_log_error"] == pytest.approx(
        metrics["train_mean_absolute_log_error"]
    )
    assert metadata["train_metrics"]["mean_poisson_deviance"] == pytest.approx(
        metrics["train_mean_poisson_deviance"]
    )
    assert metadata["train_metrics"]["mean_gamma_deviance"] == pytest.approx(
        metrics["train_mean_gamma_deviance"]
    )
    assert metadata["train_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        metrics["train_mean_tweedie_deviance"]
    )
    assert metadata["validation_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        metrics["validation_median_absolute_percentage_error"]
    )
    assert metadata["validation_metrics"]["msle"] == pytest.approx(
        metrics["validation_msle"]
    )
    assert metadata["validation_metrics"]["mean_absolute_log_error"] == pytest.approx(
        metrics["validation_mean_absolute_log_error"]
    )
    assert metadata["validation_metrics"]["mean_poisson_deviance"] == pytest.approx(
        metrics["validation_mean_poisson_deviance"]
    )
    assert metadata["validation_metrics"]["mean_gamma_deviance"] == pytest.approx(
        metrics["validation_mean_gamma_deviance"]
    )
    assert metadata["validation_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        metrics["validation_mean_tweedie_deviance"]
    )
    assert metadata["test_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        metrics["test_median_absolute_percentage_error"]
    )
    assert metadata["train_metrics"]["mase"] == pytest.approx(metrics["train_mase"])
    assert metadata["validation_metrics"]["mase"] == pytest.approx(metrics["validation_mase"])
    assert metadata["test_metrics"]["mase"] == pytest.approx(metrics["test_mase"])
    assert metadata["test_metrics"]["msle"] == pytest.approx(metrics["test_msle"])
    assert metadata["test_metrics"]["mean_absolute_log_error"] == pytest.approx(
        metrics["test_mean_absolute_log_error"]
    )
    assert metadata["test_metrics"]["mean_poisson_deviance"] == pytest.approx(
        metrics["test_mean_poisson_deviance"]
    )
    assert metadata["test_metrics"]["mean_gamma_deviance"] == pytest.approx(
        metrics["test_mean_gamma_deviance"]
    )
    assert metadata["test_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        metrics["test_mean_tweedie_deviance"]
    )


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
    assert "test_r2" in orchestrator.metrics["demo"]
    assert "test_median_absolute_error" in orchestrator.metrics["demo"]
    assert "test_explained_variance" in orchestrator.metrics["demo"]
    assert "test_max_error" in orchestrator.metrics["demo"]
    assert "test_smape" in orchestrator.metrics["demo"]
    assert "test_mean_bias_error" in orchestrator.metrics["demo"]
    assert "test_wmape" in orchestrator.metrics["demo"]
    assert "test_mpe" in orchestrator.metrics["demo"]
    assert "test_rmspe" in orchestrator.metrics["demo"]
    assert "test_median_percentage_error" in orchestrator.metrics["demo"]
    assert "test_median_absolute_percentage_error" in orchestrator.metrics["demo"]
    assert "test_msle" in orchestrator.metrics["demo"]
    assert "test_mean_absolute_log_error" in orchestrator.metrics["demo"]
    assert "test_mean_poisson_deviance" in orchestrator.metrics["demo"]
    assert "test_mean_gamma_deviance" in orchestrator.metrics["demo"]
    assert "test_mean_tweedie_deviance" in orchestrator.metrics["demo"]
    assert getattr(inference, "model_label", "") == "demo"
