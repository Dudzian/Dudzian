"""Pipelines for training and registering Decision Engine models."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .inference import DecisionModelInference, ModelRepository
from .models import ModelArtifact
from .training import SimpleGradientBoostingModel

LOGGER = logging.getLogger(__name__)


def _prepare_training_sets(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[list[Mapping[str, float]], list[float]]:
    if frame.empty:
        raise ValueError("training frame is empty")
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"training frame missing feature columns: {missing}")
    if target_col not in frame.columns:
        raise ValueError(f"training frame missing target column {target_col!r}")
    features: list[Mapping[str, float]] = []
    targets: list[float] = []
    for _, row in frame.iterrows():
        sample = {}
        for name in feature_cols:
            value = row[name]
            sample[name] = float(value)
        target_value = float(row[target_col])
        features.append(sample)
        targets.append(target_value)
    return features, targets


def _compute_metrics(
    model: SimpleGradientBoostingModel,
    features: Sequence[Mapping[str, float]],
    targets: Sequence[float],
) -> Mapping[str, float]:
    predictions = np.asarray(model.batch_predict(features), dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    mae = float(np.mean(np.abs(predictions - target_arr))) if len(predictions) else 0.0
    hits = float(np.mean(np.sign(predictions) == np.sign(target_arr))) if len(predictions) else 0.0
    return {
        "mae": mae,
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
) -> Path:
    """Train a :class:`SimpleGradientBoostingModel` and persist an artifact."""

    features, targets = _prepare_training_sets(frame, feature_cols, target_col)
    model = SimpleGradientBoostingModel()
    model.fit(features, targets)
    metrics = _compute_metrics(model, features, targets)
    meta_payload = {
        "feature_scalers": {
            name: {"mean": mean, "stdev": stdev}
            for name, (mean, stdev) in model.feature_scalers.items()
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
    attach = getattr(orchestrator, "attach_named_inference", None)
    if not callable(attach):  # pragma: no cover - defensive fallback
        raise TypeError("orchestrator does not expose attach_named_inference")
    attach(name, inference, set_default=set_default)
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
