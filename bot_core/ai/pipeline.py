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
from .models import ModelArtifact, ModelScore
from .training import SimpleGradientBoostingModel


def _calibrate_predictions(
    targets: Sequence[float], predictions: Sequence[float]
) -> tuple[float, float]:
    """Return a simple linear calibration (slope, intercept).

    The calibration rescales raw model outputs to better match the
    distribution of training targets.  We intentionally keep the
    implementation minimal – a closed form least squares regression – so
    that it works in constrained environments (CI, smoke tests) without
    requiring additional dependencies.
    """

    if not targets or not predictions:
        return 1.0, 0.0

    y = np.asarray(targets, dtype=float)
    x = np.asarray(predictions, dtype=float)
    if x.size != y.size:
        raise ValueError("Targets and predictions must have the same size for calibration")

    if np.allclose(x, x[0]):
        # Degenerate case – the model produced a constant prediction.  In
        # that scenario the best calibration we can do is to shift the
        # output towards the empirical mean of the targets.
        return 0.0, float(np.mean(y))

    A = np.vstack([x, np.ones_like(x)]).T
    solution, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(solution[0])
    intercept = float(solution[1])
    return slope, intercept


def _cross_validate(
    features: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    *,
    folds: int = 5,
) -> Mapping[str, Sequence[float]]:
    """Compute walk-forward style cross-validation metrics."""

    total = len(features)
    if total == 0:
        return {"mae": (), "directional_accuracy": ()}
    folds = max(2, min(int(folds), total))
    fold_size = max(1, total // folds)
    maes: list[float] = []
    accuracies: list[float] = []

    for fold in range(folds):
        start = fold * fold_size
        end = total if fold == folds - 1 else min(total, start + fold_size)
        validation_features = features[start:end]
        validation_targets = targets[start:end]
        train_features = features[:start] + features[end:]
        train_targets = targets[:start] + targets[end:]

        if not validation_features or not train_features:
            continue

        model = SimpleGradientBoostingModel()
        model.fit(train_features, train_targets)
        metrics = _compute_metrics(model, validation_features, validation_targets)
        maes.append(float(metrics.get("mae", 0.0)))
        accuracies.append(float(metrics.get("directional_accuracy", 0.0)))

    return {"mae": tuple(maes), "directional_accuracy": tuple(accuracies)}

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
        return {"mae": 0.0, "directional_accuracy": 0.0}
    predictions = np.asarray(model.batch_predict(features), dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    mae = float(np.mean(np.abs(predictions - target_arr))) if len(predictions) else 0.0
    rmse = float(np.sqrt(np.mean((predictions - target_arr) ** 2))) if len(predictions) else 0.0
    hits = float(np.mean(np.sign(predictions) == np.sign(target_arr))) if len(predictions) else 0.0
    pnl = float(np.mean(predictions * target_arr)) if len(predictions) else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": hits,
        "expected_pnl": pnl,
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
    model_version: str | None = None,
    publish_aliases: Sequence[str] | None = None,
    activate_version: bool = True,
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
    train_features = train_set.features
    train_targets = train_set.targets
    min_leaf = max(1, min(5, max(len(train_features) // 3, 1)))
    model = SimpleGradientBoostingModel(min_samples_leaf=min_leaf)
    model.fit(train_features, train_targets)
    raw_predictions = list(model.batch_predict(train_features))
    calibration_slope, calibration_intercept = _calibrate_predictions(train_targets, raw_predictions)
    cv_payload = _cross_validate(
        train_features, train_targets, folds=max(3, min(5, len(train_features)))
    )
    subset_metrics: dict[str, Mapping[str, float]] = {
        subset.name: _compute_metrics(model, subset.features, subset.targets)
        for subset in learning_sets
    }
    metrics = dict(subset_metrics.get(train_set.name, {}))
    if not metrics:
        metrics = _compute_metrics(model, train_features, train_targets)
    for subset_name, values in subset_metrics.items():
        prefix = subset_name.lower()
        for metric_name, value in values.items():
            metrics[f"{prefix}_{metric_name}"] = value
    meta_payload = {
        "feature_scalers": {
            name: {"mean": mean, "stdev": stdev}
            for name, (mean, stdev) in model.feature_scalers.items()
        },
        "calibration": {
            "slope": calibration_slope,
            "intercept": calibration_intercept,
        },
        "cross_validation": {
            "folds": len(cv_payload["mae"]),
            "mae": list(cv_payload["mae"]),
            "directional_accuracy": list(cv_payload["directional_accuracy"]),
        },
        "dataset_split": {
            "train_ratio": max(0.0, 1.0 - float(validation_ratio) - float(test_ratio)),
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(test_ratio),
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "test_rows": len(test_frame),
        },
        "drift_monitor": {
            "threshold": 3.5,
            "window": 50,
            "min_observations": 10,
            "cooldown": 25,
            "backend": "decision_engine",
        },
        "quality_thresholds": {
            "min_directional_accuracy": 0.55,
            "max_mae": 20.0,
        },
    }
    meta_payload.setdefault("model_name", model_name)
    if model_version:
        meta_payload.setdefault("model_version", model_version)
    if metadata:
        for key, value in metadata.items():
            if (
                isinstance(value, Mapping)
                and key in meta_payload
                and isinstance(meta_payload[key], Mapping)
            ):
                merged = dict(meta_payload[key])
                merged.update(value)  # type: ignore[arg-type]
                meta_payload[key] = merged
            else:
                meta_payload[key] = value
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
    if model_version:
        destination = repository.publish(
            artifact,
            version=model_version,
            filename=filename,
            aliases=publish_aliases,
            activate=activate_version,
        )
    else:
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


def score_with_data_monitoring(
    inference: DecisionModelInference,
    features: Mapping[str, float],
    *,
    context: Mapping[str, object] | None = None,
) -> tuple[ModelScore, Mapping[str, Mapping[str, object]]]:
    """Score a candidate ensuring monitoring executes and returns the report."""

    score = inference.score(features, context=context)
    report = inference.last_data_quality_report or {}
    return score, report


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
