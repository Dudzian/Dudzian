from __future__ import annotations

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.training import ModelTrainer


def _build_dataset(rows: int = 60) -> FeatureDataset:
    vectors = []
    for idx in range(rows):
        vectors.append(
            FeatureVector(
                timestamp=float(idx),
                symbol="BTC",
                features={"f1": float(idx), "f2": float(idx % 5)},
                target_bps=float((idx % 7) - 3),
            )
        )
    return FeatureDataset(vectors=tuple(vectors), metadata={})


def test_model_trainer_emits_test_metrics() -> None:
    dataset = _build_dataset()
    trainer = ModelTrainer(
        learning_rate=0.1,
        n_estimators=10,
        validation_split=0.2,
        test_split=0.1,
    )
    artifact = trainer.train(dataset)

    total_rows = (
        artifact.metadata["training_rows"]
        + artifact.metadata["validation_rows"]
        + artifact.metadata["test_rows"]
    )
    assert total_rows == len(dataset.vectors)
    assert artifact.metadata["validation_rows"] > 0
    assert artifact.metadata["test_rows"] > 0
    assert "dataset_split" in artifact.metadata
    assert artifact.metadata["dataset_split"]["validation_ratio"] == pytest.approx(0.2)
    assert artifact.metadata["dataset_split"]["test_ratio"] == pytest.approx(0.1)

    assert "validation_mae" in artifact.metrics
    assert "test_mae" in artifact.metrics
    assert "train_mse" in artifact.metrics
    assert "validation_mse" in artifact.metrics
    assert "test_mse" in artifact.metrics
    assert "train_mape" in artifact.metrics
    assert "validation_mape" in artifact.metrics
    assert "test_mape" in artifact.metrics
    assert "train_mase" in artifact.metrics
    assert "validation_mase" in artifact.metrics
    assert "test_mase" in artifact.metrics
    assert "train_r2" in artifact.metrics
    assert "validation_r2" in artifact.metrics
    assert "test_r2" in artifact.metrics
    assert "train_median_absolute_error" in artifact.metrics
    assert "validation_median_absolute_error" in artifact.metrics
    assert "test_median_absolute_error" in artifact.metrics
    assert "train_explained_variance" in artifact.metrics
    assert "validation_explained_variance" in artifact.metrics
    assert "test_explained_variance" in artifact.metrics
    assert "train_max_error" in artifact.metrics
    assert "validation_max_error" in artifact.metrics
    assert "test_max_error" in artifact.metrics
    assert "train_smape" in artifact.metrics
    assert "validation_smape" in artifact.metrics
    assert "test_smape" in artifact.metrics
    assert "train_mean_bias_error" in artifact.metrics
    assert "validation_mean_bias_error" in artifact.metrics
    assert "test_mean_bias_error" in artifact.metrics
    assert "train_wmape" in artifact.metrics
    assert "validation_wmape" in artifact.metrics
    assert "test_wmape" in artifact.metrics
    assert "mpe" in artifact.metrics
    assert "rmspe" in artifact.metrics
    assert "train_mpe" in artifact.metrics
    assert "validation_mpe" in artifact.metrics
    assert "test_mpe" in artifact.metrics
    assert "train_rmspe" in artifact.metrics
    assert "validation_rmspe" in artifact.metrics
    assert "test_rmspe" in artifact.metrics
    assert "median_percentage_error" in artifact.metrics
    assert "median_absolute_percentage_error" in artifact.metrics
    assert "mse" in artifact.metrics
    assert "mase" in artifact.metrics
    assert "mean_poisson_deviance" in artifact.metrics
    assert "mean_gamma_deviance" in artifact.metrics
    assert "mean_tweedie_deviance" in artifact.metrics
    assert "train_median_percentage_error" in artifact.metrics
    assert "validation_median_percentage_error" in artifact.metrics
    assert "test_median_percentage_error" in artifact.metrics
    assert "train_median_absolute_percentage_error" in artifact.metrics
    assert "validation_median_absolute_percentage_error" in artifact.metrics
    assert "test_median_absolute_percentage_error" in artifact.metrics
    assert "train_msle" in artifact.metrics
    assert "validation_msle" in artifact.metrics
    assert "test_msle" in artifact.metrics
    assert "train_mean_absolute_log_error" in artifact.metrics
    assert "validation_mean_absolute_log_error" in artifact.metrics
    assert "test_mean_absolute_log_error" in artifact.metrics
    assert "train_mean_poisson_deviance" in artifact.metrics
    assert "validation_mean_poisson_deviance" in artifact.metrics
    assert "test_mean_poisson_deviance" in artifact.metrics
    assert "train_mean_gamma_deviance" in artifact.metrics
    assert "validation_mean_gamma_deviance" in artifact.metrics
    assert "test_mean_gamma_deviance" in artifact.metrics
    assert "train_mean_tweedie_deviance" in artifact.metrics
    assert "validation_mean_tweedie_deviance" in artifact.metrics
    assert "test_mean_tweedie_deviance" in artifact.metrics
    assert "test_metrics" in artifact.metadata
    assert artifact.metadata["test_metrics"]["mae"] == pytest.approx(
        artifact.metrics["test_mae"]
    )
    assert artifact.metadata["test_metrics"]["mape"] == pytest.approx(
        artifact.metrics["test_mape"]
    )
    assert artifact.metadata["test_metrics"]["r2"] == pytest.approx(
        artifact.metrics["test_r2"]
    )
    assert artifact.metadata["test_metrics"]["mse"] == pytest.approx(
        artifact.metrics["test_mse"]
    )
    assert artifact.metadata["test_metrics"]["median_absolute_error"] == pytest.approx(
        artifact.metrics["test_median_absolute_error"]
    )
    assert artifact.metadata["test_metrics"]["explained_variance"] == pytest.approx(
        artifact.metrics["test_explained_variance"]
    )
    assert artifact.metadata["test_metrics"]["max_error"] == pytest.approx(
        artifact.metrics["test_max_error"]
    )
    assert artifact.metadata["test_metrics"]["smape"] == pytest.approx(
        artifact.metrics["test_smape"]
    )
    assert artifact.metadata["test_metrics"]["mean_bias_error"] == pytest.approx(
        artifact.metrics["test_mean_bias_error"]
    )
    assert artifact.metadata["test_metrics"]["wmape"] == pytest.approx(
        artifact.metrics["test_wmape"]
    )
    assert artifact.metadata["test_metrics"]["mpe"] == pytest.approx(
        artifact.metrics["test_mpe"]
    )
    assert artifact.metadata["test_metrics"]["rmspe"] == pytest.approx(
        artifact.metrics["test_rmspe"]
    )
    assert artifact.metadata["test_metrics"]["median_percentage_error"] == pytest.approx(
        artifact.metrics["test_median_percentage_error"]
    )
    assert artifact.metadata["test_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        artifact.metrics["test_median_absolute_percentage_error"]
    )
    assert artifact.metadata["test_metrics"]["mase"] == pytest.approx(
        artifact.metrics["test_mase"]
    )
    assert artifact.metadata["test_metrics"]["msle"] == pytest.approx(
        artifact.metrics["test_msle"]
    )
    assert artifact.metadata["test_metrics"]["mean_absolute_log_error"] == pytest.approx(
        artifact.metrics["test_mean_absolute_log_error"]
    )
    assert artifact.metadata["test_metrics"]["mean_poisson_deviance"] == pytest.approx(
        artifact.metrics["test_mean_poisson_deviance"]
    )
    assert artifact.metadata["test_metrics"]["mean_gamma_deviance"] == pytest.approx(
        artifact.metrics["test_mean_gamma_deviance"]
    )
    assert artifact.metadata["test_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        artifact.metrics["test_mean_tweedie_deviance"]
    )
    assert artifact.metadata["validation_metrics"]["r2"] == pytest.approx(
        artifact.metrics["validation_r2"]
    )
    assert artifact.metadata["validation_metrics"]["mse"] == pytest.approx(
        artifact.metrics["validation_mse"]
    )
    assert artifact.metadata["train_metrics"]["r2"] == pytest.approx(
        artifact.metrics["train_r2"]
    )
    assert artifact.metadata["train_metrics"]["mse"] == pytest.approx(
        artifact.metrics["train_mse"]
    )
    assert artifact.metadata["validation_metrics"]["median_absolute_error"] == pytest.approx(
        artifact.metrics["validation_median_absolute_error"]
    )
    assert artifact.metadata["validation_metrics"]["explained_variance"] == pytest.approx(
        artifact.metrics["validation_explained_variance"]
    )
    assert artifact.metadata["validation_metrics"]["max_error"] == pytest.approx(
        artifact.metrics["validation_max_error"]
    )
    assert artifact.metadata["validation_metrics"]["smape"] == pytest.approx(
        artifact.metrics["validation_smape"]
    )
    assert artifact.metadata["validation_metrics"]["mean_bias_error"] == pytest.approx(
        artifact.metrics["validation_mean_bias_error"]
    )
    assert artifact.metadata["validation_metrics"]["wmape"] == pytest.approx(
        artifact.metrics["validation_wmape"]
    )
    assert artifact.metadata["validation_metrics"]["mpe"] == pytest.approx(
        artifact.metrics["validation_mpe"]
    )
    assert artifact.metadata["validation_metrics"]["rmspe"] == pytest.approx(
        artifact.metrics["validation_rmspe"]
    )
    assert artifact.metadata["validation_metrics"]["median_percentage_error"] == pytest.approx(
        artifact.metrics["validation_median_percentage_error"]
    )
    assert artifact.metadata["validation_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        artifact.metrics["validation_median_absolute_percentage_error"]
    )
    assert artifact.metadata["validation_metrics"]["mase"] == pytest.approx(
        artifact.metrics["validation_mase"]
    )
    assert artifact.metadata["validation_metrics"]["msle"] == pytest.approx(
        artifact.metrics["validation_msle"]
    )
    assert artifact.metadata["validation_metrics"]["mean_absolute_log_error"] == pytest.approx(
        artifact.metrics["validation_mean_absolute_log_error"]
    )
    assert artifact.metadata["validation_metrics"]["mean_poisson_deviance"] == pytest.approx(
        artifact.metrics["validation_mean_poisson_deviance"]
    )
    assert artifact.metadata["validation_metrics"]["mean_gamma_deviance"] == pytest.approx(
        artifact.metrics["validation_mean_gamma_deviance"]
    )
    assert artifact.metadata["validation_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        artifact.metrics["validation_mean_tweedie_deviance"]
    )
    assert artifact.metadata["train_metrics"]["median_absolute_error"] == pytest.approx(
        artifact.metrics["train_median_absolute_error"]
    )
    assert artifact.metadata["train_metrics"]["explained_variance"] == pytest.approx(
        artifact.metrics["train_explained_variance"]
    )
    assert artifact.metadata["train_metrics"]["max_error"] == pytest.approx(
        artifact.metrics["train_max_error"]
    )
    assert artifact.metadata["train_metrics"]["smape"] == pytest.approx(
        artifact.metrics["train_smape"]
    )
    assert artifact.metadata["train_metrics"]["mean_bias_error"] == pytest.approx(
        artifact.metrics["train_mean_bias_error"]
    )
    assert artifact.metadata["train_metrics"]["wmape"] == pytest.approx(
        artifact.metrics["train_wmape"]
    )
    assert artifact.metadata["train_metrics"]["mean_poisson_deviance"] == pytest.approx(
        artifact.metrics["train_mean_poisson_deviance"]
    )
    assert artifact.metadata["train_metrics"]["mean_gamma_deviance"] == pytest.approx(
        artifact.metrics["train_mean_gamma_deviance"]
    )
    assert artifact.metadata["train_metrics"]["mean_tweedie_deviance"] == pytest.approx(
        artifact.metrics["train_mean_tweedie_deviance"]
    )
    assert artifact.metadata["train_metrics"]["mpe"] == pytest.approx(
        artifact.metrics["train_mpe"]
    )
    assert artifact.metadata["train_metrics"]["rmspe"] == pytest.approx(
        artifact.metrics["train_rmspe"]
    )
    assert artifact.metadata["train_metrics"]["median_percentage_error"] == pytest.approx(
        artifact.metrics["train_median_percentage_error"]
    )
    assert artifact.metadata["train_metrics"]["median_absolute_percentage_error"] == pytest.approx(
        artifact.metrics["train_median_absolute_percentage_error"]
    )
    assert artifact.metadata["train_metrics"]["mase"] == pytest.approx(
        artifact.metrics["train_mase"]
    )
    assert artifact.metadata["train_metrics"]["msle"] == pytest.approx(
        artifact.metrics["train_msle"]
    )
    assert artifact.metadata["train_metrics"]["mean_absolute_log_error"] == pytest.approx(
        artifact.metrics["train_mean_absolute_log_error"]
    )

