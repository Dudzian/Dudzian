"""Pipelines for training and registering Decision Engine models."""
from __future__ import annotations

import argparse
import json
import logging
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
        return {"mae": 0.0, "directional_accuracy": 0.0}
    predictions = np.asarray(model.batch_predict(features), dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    if len(predictions) == 0:
        return {"mae": 0.0, "directional_accuracy": 0.0}
    mae = float(np.mean(np.abs(predictions - target_arr)))
    hits = float(np.mean(np.sign(predictions) == np.sign(target_arr)))
    rmse = float(np.sqrt(np.mean((predictions - target_arr) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": hits,
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
    for subset in learning_sets:
        computed = _compute_metrics(model, subset.features, subset.targets)
        for key, value in computed.items():
            metrics[f"{subset.name}_{key}"] = float(value)
        if subset.name == "train":
            metrics.update(
                {
                    "mae": float(computed.get("mae", 0.0)),
                    "rmse": float(computed.get("rmse", 0.0)),
                    "directional_accuracy": float(
                        computed.get("directional_accuracy", 0.0)
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
