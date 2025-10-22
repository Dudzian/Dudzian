"""Pipelines for training and registering Decision Engine models."""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .inference import DecisionModelInference, ModelRepository
from .models import ModelArtifact
from .training import SimpleGradientBoostingModel

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _LearningSet:
    """Zbiór danych wejściowych używany w treningu / ewaluacji."""

    name: str
    features: list[Mapping[str, float]]
    targets: list[float]


def _validate_training_frame(
    frame: pd.DataFrame, feature_cols: Sequence[str], target_col: str
) -> None:
    if frame.empty:
        raise ValueError("training frame is empty")
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"training frame missing feature columns: {missing}")
    if target_col not in frame.columns:
        raise ValueError(f"training frame missing target column {target_col!r}")


def _extract_learning_set(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    name: str,
) -> _LearningSet:
    features: list[Mapping[str, float]] = []
    targets: list[float] = []
    for _, row in frame.iterrows():
        sample = {str(col): float(row[col]) for col in feature_cols}
        features.append(sample)
        targets.append(float(row[target_col]))
    return _LearningSet(name=name, features=features, targets=targets)


def _split_training_frame(
    frame: pd.DataFrame,
    *,
    validation_ratio: float,
    test_ratio: float,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be within <0.0, 1.0)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be within <0.0, 1.0)")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1.0")
    if frame.empty:
        return frame, frame.iloc[0:0], frame.iloc[0:0]
    shuffled = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    total = len(shuffled)
    test_count = int(round(total * test_ratio))
    validation_count = int(round(total * validation_ratio))
    # ensure non-empty train split
    if test_count + validation_count >= total:
        # shrink validation first, then test if needed
        overflow = test_count + validation_count - (total - 1)
        if overflow > 0 and validation_count > 0:
            reduction = min(validation_count, overflow)
            validation_count -= reduction
            overflow -= reduction
        if overflow > 0 and test_count > 0:
            reduction = min(test_count, overflow)
            test_count -= reduction
    split_train_end = total - (validation_count + test_count)
    if split_train_end <= 0:
        raise ValueError("Not enough rows for requested validation/test split")
    validation_start = split_train_end
    validation_end = validation_start + validation_count
    train_frame = shuffled.iloc[:split_train_end].copy()
    validation_frame = shuffled.iloc[validation_start:validation_end].copy()
    test_frame = shuffled.iloc[validation_end:].copy()
    return train_frame, validation_frame, test_frame


def _compute_metrics(
    model: SimpleGradientBoostingModel,
    features: Sequence[Mapping[str, float]],
    targets: Sequence[float],
) -> Mapping[str, float]:
    if not features:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "directional_accuracy": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "median_absolute_error": 0.0,
            "explained_variance": 0.0,
            "max_error": 0.0,
            "smape": 0.0,
            "mean_bias_error": 0.0,
            "wmape": 0.0,
            "mpe": 0.0,
            "rmspe": 0.0,
            "median_percentage_error": 0.0,
            "median_absolute_percentage_error": 0.0,
            "mase": 0.0,
            "msle": 0.0,
            "mean_absolute_log_error": 0.0,
            "mean_poisson_deviance": 0.0,
            "mean_gamma_deviance": 0.0,
            "mean_tweedie_deviance": 0.0,
        }
    predictions = np.asarray(model.batch_predict(features), dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    if len(predictions) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "directional_accuracy": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "median_absolute_error": 0.0,
            "explained_variance": 0.0,
            "max_error": 0.0,
            "smape": 0.0,
            "mean_bias_error": 0.0,
            "wmape": 0.0,
            "mpe": 0.0,
            "rmspe": 0.0,
            "median_percentage_error": 0.0,
            "median_absolute_percentage_error": 0.0,
            "mase": 0.0,
            "msle": 0.0,
            "mean_absolute_log_error": 0.0,
            "mean_poisson_deviance": 0.0,
            "mean_gamma_deviance": 0.0,
            "mean_tweedie_deviance": 0.0,
        }
    absolute_errors = np.abs(predictions - target_arr)
    mae = float(np.mean(absolute_errors))
    mse = float(np.mean((predictions - target_arr) ** 2))
    medae = float(np.median(absolute_errors))
    max_error = float(np.max(absolute_errors)) if absolute_errors.size else 0.0
    hits = float(np.mean(np.sign(predictions) == np.sign(target_arr)))
    rmse = float(np.sqrt(np.mean((predictions - target_arr) ** 2)))
    mean_bias_error = float(np.mean(predictions - target_arr))
    denominator = np.abs(target_arr)
    safe_denominator = np.where(denominator < 1e-8, np.nan, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_terms = np.abs(predictions - target_arr) / safe_denominator
        mpe_terms = (predictions - target_arr) / safe_denominator
        mspe_terms = ((predictions - target_arr) / safe_denominator) ** 2
    valid_mape_terms = mape_terms[~np.isnan(mape_terms)]
    if valid_mape_terms.size == 0:
        mape = 0.0
        median_absolute_percentage_error = 0.0
    else:
        mape = float(np.mean(valid_mape_terms))
        median_absolute_percentage_error = float(np.median(valid_mape_terms))
    valid_mpe_terms = mpe_terms[~np.isnan(mpe_terms)]
    if valid_mpe_terms.size == 0:
        mpe = 0.0
        median_percentage_error = 0.0
    else:
        mpe = float(np.mean(valid_mpe_terms))
        median_percentage_error = float(np.median(valid_mpe_terms))
    if np.isnan(mspe_terms).all():
        rmspe = 0.0
    else:
        rmspe = float(np.sqrt(np.nanmean(mspe_terms)))
    denominator = np.abs(predictions) + np.abs(target_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_terms = np.where(denominator < 1e-8, np.nan, 2.0 * absolute_errors / denominator)
    if np.isnan(smape_terms).all():
        smape = 0.0
    else:
        smape = float(np.nanmean(smape_terms))
    valid_log_mask = (predictions > -1.0) & (target_arr > -1.0)
    if not np.any(valid_log_mask):
        msle = 0.0
        mean_absolute_log_error = 0.0
    else:
        log_diff = np.log1p(predictions[valid_log_mask]) - np.log1p(
            target_arr[valid_log_mask]
        )
        msle = float(np.mean(log_diff**2))
        mean_absolute_log_error = float(np.mean(np.abs(log_diff)))
    positive_prediction_mask = predictions > 1e-12
    poisson_mask = positive_prediction_mask & (target_arr >= 0.0)
    if np.any(poisson_mask):
        poisson_targets = target_arr[poisson_mask]
        poisson_preds = predictions[poisson_mask]
        safe_preds = np.clip(poisson_preds, 1e-12, None)
        ratio = np.divide(
            poisson_targets,
            safe_preds,
            out=np.ones_like(poisson_targets),
            where=poisson_targets != 0.0,
        )
        log_term = np.where(
            poisson_targets == 0.0,
            0.0,
            np.log(np.clip(ratio, 1e-12, None)),
        )
        poisson_deviance = 2.0 * (
            safe_preds - poisson_targets + poisson_targets * log_term
        )
        mean_poisson_deviance = float(np.mean(poisson_deviance))
    else:
        mean_poisson_deviance = 0.0
    gamma_mask = positive_prediction_mask & (target_arr > 0.0)
    if np.any(gamma_mask):
        gamma_targets = target_arr[gamma_mask]
        gamma_preds = predictions[gamma_mask]
        safe_preds = np.clip(gamma_preds, 1e-12, None)
        ratio = np.clip(gamma_targets / safe_preds, 1e-12, None)
        gamma_deviance = 2.0 * (
            (gamma_targets - safe_preds) / safe_preds - np.log(ratio)
        )
        mean_gamma_deviance = float(np.mean(gamma_deviance))
    else:
        mean_gamma_deviance = 0.0
    tweedie_mask = positive_prediction_mask & (target_arr >= 0.0)
    if np.any(tweedie_mask):
        tweedie_targets = target_arr[tweedie_mask]
        tweedie_preds = np.clip(predictions[tweedie_mask], 1e-12, None)
        power = 1.5
        one_minus_power = 1.0 - power
        two_minus_power = 2.0 - power
        term_a = np.power(tweedie_targets, two_minus_power) / (
            one_minus_power * two_minus_power
        )
        term_b = (
            tweedie_targets
            * np.power(tweedie_preds, one_minus_power)
            / one_minus_power
        )
        term_c = np.power(tweedie_preds, two_minus_power) / two_minus_power
        tweedie_deviance = 2.0 * (term_a - term_b + term_c)
        mean_tweedie_deviance = float(np.mean(tweedie_deviance))
    else:
        mean_tweedie_deviance = 0.0
    if target_arr.size > 1:
        naive_diffs = np.abs(np.diff(target_arr))
        mase_denominator = float(np.mean(naive_diffs)) if naive_diffs.size else 0.0
    else:
        mase_denominator = 0.0
    if mase_denominator < 1e-8:
        mase = 0.0
    else:
        mase = mae / mase_denominator
    total_abs_target = float(np.sum(np.abs(target_arr)))
    if total_abs_target < 1e-8:
        wmape = 0.0
    else:
        wmape = float(np.sum(absolute_errors) / total_abs_target)
    centered_targets = target_arr - np.mean(target_arr)
    total_variance = float(np.sum(centered_targets**2))
    if total_variance <= 1e-12:
        r2 = 0.0
        explained_variance = 0.0
    else:
        residual = float(np.sum((predictions - target_arr) ** 2))
        r2 = 1.0 - residual / total_variance
        prediction_diff = predictions - target_arr
        denominator = float(np.var(target_arr))
        if denominator <= 1e-12:
            explained_variance = 0.0
        else:
            explained_variance = 1.0 - float(np.var(prediction_diff)) / denominator
    if not math.isfinite(explained_variance):
        explained_variance = 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "directional_accuracy": hits,
        "mape": mape,
        "r2": float(r2),
        "median_absolute_error": medae,
        "explained_variance": float(explained_variance),
        "max_error": max_error,
        "smape": smape,
        "mean_bias_error": mean_bias_error,
        "wmape": wmape,
        "mpe": mpe,
        "rmspe": rmspe,
        "median_percentage_error": median_percentage_error,
        "median_absolute_percentage_error": median_absolute_percentage_error,
        "mase": mase,
        "msle": msle,
        "mean_absolute_log_error": mean_absolute_log_error,
        "mean_poisson_deviance": mean_poisson_deviance,
        "mean_gamma_deviance": mean_gamma_deviance,
        "mean_tweedie_deviance": mean_tweedie_deviance,
    }


def train_gradient_boosting_model(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    output_dir: Path,
    model_name: str,
    metadata: Mapping[str, object] | None = None,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int | None = 42,
) -> Path:
    """Train a :class:`SimpleGradientBoostingModel` and persist an artifact."""

    _validate_training_frame(frame, feature_cols, target_col)
    train_frame, validation_frame, test_frame = _split_training_frame(
        frame,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    learning_sets: list[_LearningSet] = [
        _extract_learning_set(train_frame, feature_cols, target_col, name="train"),
    ]
    if not validation_frame.empty:
        learning_sets.append(
            _extract_learning_set(validation_frame, feature_cols, target_col, name="validation")
        )
    if not test_frame.empty:
        learning_sets.append(
            _extract_learning_set(test_frame, feature_cols, target_col, name="test")
        )
    train_set = learning_sets[0]
    model = SimpleGradientBoostingModel()
    model.fit(train_set.features, train_set.targets)
    metrics: dict[str, float] = {}
    subset_metrics: dict[str, dict[str, float]] = {}
    for subset in learning_sets:
        computed = _compute_metrics(model, subset.features, subset.targets)
        subset_metrics[subset.name] = {str(k): float(v) for k, v in computed.items()}
        for key, value in computed.items():
            metrics[f"{subset.name}_{key}"] = float(value)
        if subset.name == "train":
            metrics.update(
                {
                    "mae": float(computed.get("mae", 0.0)),
                    "mse": float(computed.get("mse", 0.0)),
                    "rmse": float(computed.get("rmse", 0.0)),
                    "directional_accuracy": float(
                        computed.get("directional_accuracy", 0.0)
                    ),
                    "mape": float(computed.get("mape", 0.0)),
                    "r2": float(computed.get("r2", 0.0)),
                    "median_absolute_error": float(
                        computed.get("median_absolute_error", 0.0)
                    ),
                    "explained_variance": float(
                        computed.get("explained_variance", 0.0)
                    ),
                    "max_error": float(computed.get("max_error", 0.0)),
                    "smape": float(computed.get("smape", 0.0)),
                    "mean_bias_error": float(computed.get("mean_bias_error", 0.0)),
                    "wmape": float(computed.get("wmape", 0.0)),
                    "mpe": float(computed.get("mpe", 0.0)),
                    "rmspe": float(computed.get("rmspe", 0.0)),
                    "median_percentage_error": float(
                        computed.get("median_percentage_error", 0.0)
                    ),
                    "median_absolute_percentage_error": float(
                        computed.get("median_absolute_percentage_error", 0.0)
                    ),
                    "mase": float(computed.get("mase", 0.0)),
                    "msle": float(computed.get("msle", 0.0)),
                    "mean_absolute_log_error": float(
                        computed.get("mean_absolute_log_error", 0.0)
                    ),
                    "mean_poisson_deviance": float(
                        computed.get("mean_poisson_deviance", 0.0)
                    ),
                    "mean_gamma_deviance": float(
                        computed.get("mean_gamma_deviance", 0.0)
                    ),
                    "mean_tweedie_deviance": float(
                        computed.get("mean_tweedie_deviance", 0.0)
                    ),
                }
            )
    metrics["training_samples"] = float(len(train_set.features))
    metrics["validation_samples"] = float(len(validation_frame))
    metrics["test_samples"] = float(len(test_frame))
    meta_payload = {
        "feature_scalers": {
            name: {"mean": mean, "stdev": stdev}
            for name, (mean, stdev) in model.feature_scalers.items()
        },
        "dataset_split": {
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(test_ratio),
        },
        "training_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "drift_monitor": {
            "threshold": 3.5,
            "window": 50,
            "min_observations": 10,
            "cooldown": 60,
            "backend": "decision_engine",
        },
        "quality_thresholds": {
            "min_directional_accuracy": 0.55,
            "max_mae": 25.0,
        },
    }
    if metadata:
        meta_payload.update(metadata)
    if "train" in subset_metrics:
        meta_payload["train_metrics"] = dict(subset_metrics["train"])
    if "validation" in subset_metrics:
        meta_payload["validation_metrics"] = dict(subset_metrics["validation"])
    if "test" in subset_metrics:
        meta_payload["test_metrics"] = dict(subset_metrics["test"])
    artifact = ModelArtifact(
        feature_names=tuple(model.feature_names),
        model_state=model.to_state(),
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        metadata=meta_payload,
        backend="builtin",
    )
    repository = ModelRepository(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}.json"
    destination = repository.save(artifact, filename)
    LOGGER.info("Saved decision model artifact to %s", destination)
    return destination


def register_model_artifact(
    orchestrator: object,
    artifact_path: Path,
    *,
    name: str,
    repository_root: Path | None = None,
    set_default: bool = False,
) -> DecisionModelInference:
    """Register a model artifact with ``DecisionOrchestrator``."""

    repository_base = repository_root or artifact_path.parent
    repository = ModelRepository(repository_base)
    inference = DecisionModelInference(repository)
    # load_weights expects a path relative to repository root when str provided
    path_obj = Path(artifact_path)
    try:
        relative = path_obj.relative_to(repository_base)
    except ValueError:
        relative = path_obj.name if path_obj.is_absolute() else path_obj
    inference.load_weights(str(relative))
    try:
        setattr(inference, "model_label", name)
    except Exception:  # pragma: no cover - backward compatibility fallback
        LOGGER.debug("DecisionModelInference does not expose model_label setter", exc_info=True)
    attach = getattr(orchestrator, "attach_named_inference", None)
    if not callable(attach):  # pragma: no cover - defensive fallback
        raise TypeError("orchestrator does not expose attach_named_inference")
    attach(name, inference, set_default=set_default)
    artifact = getattr(inference, "_artifact", None)
    update_metrics = getattr(orchestrator, "update_model_performance", None)
    if artifact is not None and callable(update_metrics):
        try:
            update_metrics(name, getattr(artifact, "metrics", {}) or {})
        except Exception:  # pragma: no cover - orchestrator extensions may fail
            LOGGER.exception("Failed to push metrics to DecisionOrchestrator for %s", name)
    return inference


def _load_frame_from_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.DataFrame(payload)
    return pd.read_csv(path)


def _run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Decision Engine models")
    parser.add_argument("input", type=Path, help="Path to CSV or JSON training data")
    parser.add_argument("output", type=Path, help="Directory for saved model artifacts")
    parser.add_argument("model", help="Model artifact name")
    parser.add_argument("target", help="Target column name")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns used for training",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Fraction of data reserved for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data reserved for hold-out testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling the dataset before splitting",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model in DecisionOrchestrator",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to Decision Engine config JSON/YAML for registration",
    )
    parser.add_argument(
        "--model-name",
        default="autotrader",
        help="Name used when registering the inference",
    )
    args = parser.parse_args(argv)

    frame = _load_frame_from_path(args.input)
    artifact_path = train_gradient_boosting_model(
        frame,
        args.features,
        args.target,
        output_dir=args.output,
        model_name=args.model,
        metadata={"training_rows": len(frame)},
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    if args.register:
        if args.config is None:
            parser.error("--config is required when using --register")
        from bot_core.config.loader import load_decision_engine_config
        from bot_core.decision import DecisionOrchestrator

        engine_config = load_decision_engine_config(args.config)
        orchestrator = DecisionOrchestrator(engine_config)
        register_model_artifact(
            orchestrator,
            artifact_path,
            name=args.model_name,
            repository_root=args.output,
            set_default=True,
        )
        LOGGER.info("Registered model %s with DecisionOrchestrator", args.model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_run_cli())
