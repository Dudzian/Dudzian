"""Sequential/RL style training pipeline with feature engineering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .feature_engineering import FeatureDataset, FeatureVector
from .models import ModelArtifact, ModelScore
from .training import SupportsInference


# ---------------------------------------------------------------------------
# Data repository
# ---------------------------------------------------------------------------


def _serialize_dataset(dataset: FeatureDataset) -> Mapping[str, object]:
    vectors: list[Mapping[str, object]] = []
    for vector in dataset.vectors:
        vectors.append(
            {
                "timestamp": float(vector.timestamp),
                "symbol": vector.symbol,
                "features": {str(name): float(value) for name, value in vector.features.items()},
                "target_bps": float(vector.target_bps),
            }
        )

    payload: MutableMapping[str, object] = {
        "vectors": vectors,
        "metadata": dict(dataset.metadata),
    }
    return payload


def _deserialize_dataset(payload: Mapping[str, object]) -> FeatureDataset:
    raw_vectors = payload.get("vectors", [])
    vectors: list[FeatureVector] = []
    if isinstance(raw_vectors, Sequence):
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, Mapping):
                continue
            vectors.append(
                FeatureVector(
                    timestamp=float(raw_vector.get("timestamp", 0.0)),
                    symbol=str(raw_vector.get("symbol", "")),
                    features={
                        str(name): float(value)
                        for name, value in dict(raw_vector.get("features", {})).items()
                    },
                    target_bps=float(raw_vector.get("target_bps", 0.0)),
                )
            )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return FeatureDataset(vectors=tuple(vectors), metadata=dict(metadata))


@dataclass(slots=True)
class HistoricalFeatureRepository:
    """Simple on-disk repository storing historical feature datasets."""

    root: Path
    retention: int = 25

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention = max(int(self.retention), 1)

    def save(self, dataset: FeatureDataset, *, label: str | None = None) -> Path:
        label = label or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        destination = self.root / f"{label}.json"
        payload = _serialize_dataset(dataset)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self._prune()
        return destination

    def load(self, label: str) -> FeatureDataset:
        path = self.root / f"{label}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _deserialize_dataset(payload)

    def latest(self) -> tuple[str, FeatureDataset] | None:
        entries = sorted(self.root.glob("*.json"))
        if not entries:
            return None
        latest_path = entries[-1]
        with latest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        label = latest_path.stem
        return label, _deserialize_dataset(payload)

    def list(self) -> Sequence[str]:
        return tuple(sorted(path.stem for path in self.root.glob("*.json")))

    def _prune(self) -> None:
        entries = sorted(self.root.glob("*.json"))
        if len(entries) <= self.retention:
            return
        excess = entries[: -self.retention]
        for path in excess:
            try:
                path.unlink()
            except FileNotFoundError:  # pragma: no cover - race condition safe guard
                continue


# ---------------------------------------------------------------------------
# Sequential RL style model
# ---------------------------------------------------------------------------


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(slots=True)
class TemporalDifferencePolicy(SupportsInference):
    """Minimal temporal-difference style policy approximator."""

    feature_names: Sequence[str]
    learning_rate: float = 0.05
    discount_factor: float = 0.9
    weight_decay: float = 1e-4
    _weights: list[float] = field(default_factory=list, init=False, repr=False)
    _bias: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.feature_names = tuple(str(name) for name in self.feature_names)
        if not self._weights:
            self._weights = [0.0 for _ in self.feature_names]

    def fit(self, samples: Sequence[Mapping[str, float]], targets: Sequence[float]) -> None:
        if not samples:
            return
        self._fit_numpy(samples, targets)

    # ------------------------------------------------------------------ helpers --
    def _fit_numpy(self, samples: Sequence[Mapping[str, float]], targets: Sequence[float]) -> None:
        feature_matrix = np.asarray([self._vectorize_numpy(sample) for sample in samples], dtype=float)
        target_arr = np.asarray(targets, dtype=float)
        weights = np.asarray(self._weights, dtype=float)
        bias = float(self._bias)
        for feature_row, reward in zip(feature_matrix, target_arr):
            value = float(np.dot(weights, feature_row) + bias)
            td_target = float(reward + self.discount_factor * value)
            td_error = td_target - value
            weights = weights + self.learning_rate * (td_error * feature_row - self.weight_decay * weights)
            bias = bias + self.learning_rate * td_error * 0.5
        self._weights = [float(w) for w in weights]
        self._bias = bias

    def predict(self, features: Mapping[str, float]) -> float:
        vector = [float(features.get(name, 0.0)) for name in self.feature_names]
        return sum(weight * value for weight, value in zip(self._weights, vector)) + self._bias

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> Sequence[float]:
        return [self.predict(sample) for sample in samples]

    def to_state(self) -> Mapping[str, object]:
        return {
            "weights": list(self._weights),
            "bias": float(self._bias),
            "learning_rate": float(self.learning_rate),
            "discount_factor": float(self.discount_factor),
            "weight_decay": float(self.weight_decay),
        }

    @classmethod
    def from_state(cls, feature_names: Sequence[str], payload: Mapping[str, object]) -> "TemporalDifferencePolicy":
        model = cls(
            feature_names=feature_names,
            learning_rate=float(payload.get("learning_rate", 0.05)),
            discount_factor=float(payload.get("discount_factor", 0.9)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
        )
        weights = payload.get("weights")
        if isinstance(weights, Sequence):
            model._weights = [float(value) for value in weights[: len(model.feature_names)]]
        bias = payload.get("bias")
        if bias is not None:
            model._bias = float(bias)
        return model

    # ------------------------------------------------------------------ utils --
    def _vectorize_numpy(self, sample: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(sample.get(name, 0.0)) for name in self.feature_names], dtype=float)


# ---------------------------------------------------------------------------
# Feature selection and walk-forward training
# ---------------------------------------------------------------------------


def _rank_features(dataset: FeatureDataset) -> Sequence[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    targets = dataset.targets
    if not targets:
        return ()
    target_arr = np.asarray(targets, dtype=float)
    for name in dataset.feature_names:
        values = np.asarray([float(vector.features.get(name, 0.0)) for vector in dataset.vectors], dtype=float)
        if np.allclose(values, values[0]):
            score = 0.0
        else:
            corr = float(np.corrcoef(values, target_arr)[0, 1]) if values.size > 1 else 0.0
            score = abs(corr)
        scores.append((name, score))
    return tuple(sorted(scores, key=lambda item: item[1], reverse=True))


def _split_walk_forward(total: int, folds: int) -> Sequence[tuple[int, int]]:
    folds = max(int(folds), 1)
    fold_size = max(total // folds, 1)
    splits: list[tuple[int, int]] = []
    for fold in range(folds):
        start = fold * fold_size
        end = total if fold == folds - 1 else min(total, start + fold_size)
        if start >= total:
            break
        splits.append((start, end))
    return tuple(splits)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def momentum_heuristic(features: Mapping[str, float]) -> float:
    return float(features.get("momentum_1", features.get("return_1", 0.0)))


def volatility_penalized_momentum(features: Mapping[str, float]) -> float:
    momentum = float(features.get("momentum_1", 0.0))
    volatility = float(features.get("volatility_1", abs(features.get("return_1", 0.0))))
    if volatility <= 0:
        return momentum
    return momentum / (1.0 + abs(volatility))


BUILTIN_HEURISTICS: Mapping[str, Callable[[Mapping[str, float]], float]] = {
    "momentum": momentum_heuristic,
    "volatility": volatility_penalized_momentum,
}


@dataclass(slots=True)
class WalkForwardMetrics:
    directional_accuracy: Sequence[float]
    mae: Sequence[float]
    rmse: Sequence[float]
    pnl: Sequence[float]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "directional_accuracy": list(self.directional_accuracy),
            "mae": list(self.mae),
            "rmse": list(self.rmse),
            "expected_pnl": list(self.pnl),
            "mean_directional_accuracy": _mean(self.directional_accuracy),
            "mean_mae": _mean(self.mae),
            "mean_rmse": _mean(self.rmse),
            "mean_expected_pnl": _mean(self.pnl),
        }


@dataclass(slots=True)
class SequentialTrainingReport:
    artifact: ModelArtifact
    selected_features: Sequence[str]
    feature_ranking: Sequence[tuple[str, float]]
    walk_forward_metrics: WalkForwardMetrics
    heuristic_metrics: WalkForwardMetrics


class SequentialTrainingPipeline:
    """Offline sequential training with feature selection and walk-forward validation."""

    def __init__(
        self,
        *,
        repository: HistoricalFeatureRepository,
        heuristics: Mapping[str, Callable[[Mapping[str, float]], float]] | None = None,
        min_directional_accuracy: float = 0.5,
    ) -> None:
        self._repository = repository
        self._heuristics = heuristics or BUILTIN_HEURISTICS
        self._min_directional_accuracy = float(min_directional_accuracy)

    def train_offline(
        self,
        dataset: FeatureDataset,
        *,
        top_k_features: int = 24,
        folds: int = 4,
        learning_rate: float = 0.05,
        discount_factor: float = 0.9,
    ) -> SequentialTrainingReport:
        ranking = _rank_features(dataset)
        selected = [name for name, _ in ranking[: max(int(top_k_features), 1)]]
        trimmed_dataset = dataset.subset(range(len(dataset.vectors)))
        features = [
            {name: float(vector.features.get(name, 0.0)) for name in selected}
            for vector in trimmed_dataset.vectors
        ]
        targets = trimmed_dataset.targets

        wf_metrics = self._walk_forward(features, targets, selected, folds, learning_rate, discount_factor)
        heuristic_metrics = self._evaluate_heuristics(features, targets, folds)

        final_model = TemporalDifferencePolicy(
            feature_names=selected,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        final_model.fit(features, targets)

        metrics_payload = wf_metrics.to_dict()
        metadata: dict[str, object] = {
            "target_scale": dataset.target_scale,
            "walk_forward": metrics_payload,
            "heuristics": heuristic_metrics.to_dict(),
            "min_directional_accuracy": self._min_directional_accuracy,
        }
        if metrics_payload.get("mean_directional_accuracy", 0.0) < self._min_directional_accuracy:
            metadata["training_warning"] = "directional_accuracy_below_threshold"
        feature_scalers = {
            name: (
                float(stats.get("mean", 0.0)),
                float(stats.get("stdev", 0.0)),
            )
            for name, stats in dataset.feature_stats.items()
            if isinstance(stats, Mapping)
        }
        summary_metrics = {
            "directional_accuracy": float(metrics_payload["mean_directional_accuracy"]),
            "mae": float(metrics_payload["mean_mae"]),
            "rmse": float(metrics_payload["mean_rmse"]),
            "expected_pnl": float(metrics_payload["mean_expected_pnl"]),
        }
        decision_journal_entry = metadata.get("decision_journal_entry_id") or metadata.get(
            "decision_journal_entry"
        )
        decision_journal_entry_id = (
            str(decision_journal_entry) if decision_journal_entry is not None else None
        )
        artifact = ModelArtifact(
            feature_names=tuple(selected),
            model_state=final_model.to_state(),
            trained_at=datetime.now(timezone.utc),
            metrics={
                "summary": dict(summary_metrics),
                "train": dict(summary_metrics),
                "validation": {},
                "test": {},
            },
            metadata=metadata,
            target_scale=float(dataset.target_scale),
            training_rows=len(features),
            validation_rows=0,
            test_rows=0,
            feature_scalers=feature_scalers,
            decision_journal_entry_id=decision_journal_entry_id,
            backend="sequential_td",
        )

        self._repository.save(dataset)
        return SequentialTrainingReport(
            artifact=artifact,
            selected_features=tuple(selected),
            feature_ranking=tuple(ranking),
            walk_forward_metrics=wf_metrics,
            heuristic_metrics=heuristic_metrics,
        )

    # ------------------------------------------------------------------ helpers --
    def _walk_forward(
        self,
        features: Sequence[Mapping[str, float]],
        targets: Sequence[float],
        feature_names: Sequence[str],
        folds: int,
        learning_rate: float,
        discount_factor: float,
    ) -> WalkForwardMetrics:
        splits = _split_walk_forward(len(features), folds)
        accuracies: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        pnls: list[float] = []
        for start, end in splits:
            validation_features = features[start:end]
            validation_targets = targets[start:end]
            train_features = features[:start]
            train_targets = targets[:start]
            if not train_features or not validation_features:
                continue
            model = TemporalDifferencePolicy(
                feature_names=feature_names,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            model.fit(train_features, train_targets)
            predictions = model.batch_predict(validation_features)
            accuracies.append(self._directional_accuracy(predictions, validation_targets))
            maes.append(self._mae(predictions, validation_targets))
            rmses.append(self._rmse(predictions, validation_targets))
            pnls.append(self._pnl(predictions, validation_targets))
        return WalkForwardMetrics(
            directional_accuracy=tuple(accuracies),
            mae=tuple(maes),
            rmse=tuple(rmses),
            pnl=tuple(pnls),
        )

    def _evaluate_heuristics(
        self,
        features: Sequence[Mapping[str, float]],
        targets: Sequence[float],
        folds: int,
    ) -> WalkForwardMetrics:
        splits = _split_walk_forward(len(features), folds)
        accuracies: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        pnls: list[float] = []
        for start, end in splits:
            validation_features = features[start:end]
            validation_targets = targets[start:end]
            if not validation_features:
                continue
            predictions = [self._heuristic_prediction(sample) for sample in validation_features]
            accuracies.append(self._directional_accuracy(predictions, validation_targets))
            maes.append(self._mae(predictions, validation_targets))
            rmses.append(self._rmse(predictions, validation_targets))
            pnls.append(self._pnl(predictions, validation_targets))
        return WalkForwardMetrics(
            directional_accuracy=tuple(accuracies),
            mae=tuple(maes),
            rmse=tuple(rmses),
            pnl=tuple(pnls),
        )

    def _heuristic_prediction(self, features: Mapping[str, float]) -> float:
        if not self._heuristics:
            return 0.0
        values = [heuristic(features) for heuristic in self._heuristics.values()]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _directional_accuracy(predictions: Sequence[float], targets: Sequence[float]) -> float:
        if not predictions:
            return 0.0
        hits = sum(math.copysign(1.0, p) == math.copysign(1.0, t) for p, t in zip(predictions, targets))
        return hits / len(predictions)

    @staticmethod
    def _mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
        if not predictions:
            return 0.0
        errors = [abs(p - t) for p, t in zip(predictions, targets)]
        return sum(errors) / len(errors)

    @staticmethod
    def _rmse(predictions: Sequence[float], targets: Sequence[float]) -> float:
        if not predictions:
            return 0.0
        squared = [(p - t) ** 2 for p, t in zip(predictions, targets)]
        return math.sqrt(sum(squared) / len(squared))

    @staticmethod
    def _pnl(predictions: Sequence[float], targets: Sequence[float]) -> float:
        if not predictions:
            return 0.0
        pnl = [p * t for p, t in zip(predictions, targets)]
        return sum(pnl) / len(pnl)


# ---------------------------------------------------------------------------
# Online scoring with heuristics fallback
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OnlineScoringResult:
    score: ModelScore
    source: str
    diagnostics: Mapping[str, float]


class SequentialOnlineScorer:
    """Online scoring helper with fallback heuristics."""

    def __init__(
        self,
        *,
        model: SupportsInference | None,
        heuristics: Mapping[str, Callable[[Mapping[str, float]], float]] | None = None,
        min_probability: float = 0.55,
    ) -> None:
        self._model = model
        self._heuristics = heuristics or BUILTIN_HEURISTICS
        self._min_probability = float(min_probability)

    def score(self, features: Mapping[str, float]) -> OnlineScoringResult:
        model_score: ModelScore | None = None
        if self._model is not None:
            prediction = float(self._model.predict(features))
            probability = _sigmoid(prediction)
            model_score = ModelScore(expected_return_bps=prediction, success_probability=probability)
            if probability >= self._min_probability and math.isfinite(prediction):
                return OnlineScoringResult(
                    score=model_score,
                    source="model",
                    diagnostics={"probability": probability},
                )

        heuristic_prediction = self._heuristic_prediction(features)
        heuristic_probability = max(0.5, min(0.5 + abs(heuristic_prediction) / 100.0, 0.99))
        fallback_score = ModelScore(
            expected_return_bps=heuristic_prediction,
            success_probability=heuristic_probability,
        )
        diagnostics: dict[str, float] = {"heuristic_probability": heuristic_probability}
        if model_score is not None:
            diagnostics["model_probability"] = model_score.success_probability
            diagnostics["model_prediction"] = model_score.expected_return_bps
        return OnlineScoringResult(score=fallback_score, source="heuristic", diagnostics=diagnostics)

    def _heuristic_prediction(self, features: Mapping[str, float]) -> float:
        if not self._heuristics:
            return 0.0
        values = [heuristic(features) for heuristic in self._heuristics.values()]
        if not values:
            return 0.0
        return sum(values) / len(values)


__all__ = [
    "BUILTIN_HEURISTICS",
    "HistoricalFeatureRepository",
    "OnlineScoringResult",
    "SequentialOnlineScorer",
    "SequentialTrainingPipeline",
    "SequentialTrainingReport",
    "TemporalDifferencePolicy",
    "WalkForwardMetrics",
    "momentum_heuristic",
    "volatility_penalized_momentum",
]

