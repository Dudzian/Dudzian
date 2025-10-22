"""Sequential/RL style training pipeline with feature engineering and fallbacks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - numpy is optional in constrained builds
    import numpy as np
except Exception:  # pragma: no cover - fallback to Python implementation
    np = None  # type: ignore[assignment]

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
        if np is None:
            self._fit_python(samples, targets)
        else:
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

    def _fit_python(self, samples: Sequence[Mapping[str, float]], targets: Sequence[float]) -> None:
        weights = list(self._weights)
        bias = float(self._bias)
        for sample, reward in zip(samples, targets):
            feature_row = [float(sample.get(name, 0.0)) for name in self.feature_names]
            value = sum(w * x for w, x in zip(weights, feature_row)) + bias
            td_target = reward + self.discount_factor * value
            td_error = td_target - value
            for idx, x in enumerate(feature_row):
                weights[idx] += self.learning_rate * (td_error * x - self.weight_decay * weights[idx])
            bias += self.learning_rate * td_error * 0.5
        self._weights = weights
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
    if np is not None:
        target_arr = np.asarray(targets, dtype=float)
        for name in dataset.feature_names:
            values = np.asarray([float(vector.features.get(name, 0.0)) for vector in dataset.vectors], dtype=float)
            if np.allclose(values, values[0]):
                score = 0.0
            else:
                corr = float(np.corrcoef(values, target_arr)[0, 1]) if values.size > 1 else 0.0
                score = abs(corr)
            scores.append((name, score))
    else:
        mean_target = sum(targets) / len(targets)
        for name in dataset.feature_names:
            values = [float(vector.features.get(name, 0.0)) for vector in dataset.vectors]
            mean_feature = sum(values) / len(values) if values else 0.0
            numerator = sum((x - mean_feature) * (y - mean_target) for x, y in zip(values, targets))
            denom_feature = math.sqrt(sum((x - mean_feature) ** 2 for x in values))
            denom_target = math.sqrt(sum((y - mean_target) ** 2 for y in targets))
            if denom_feature <= 0.0 or denom_target <= 0.0:
                score = 0.0
            else:
                score = abs(numerator / (denom_feature * denom_target))
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


def build_heuristic_registry(
    *registries: Mapping[str, Callable[[Mapping[str, float]], float]] | None,
) -> Mapping[str, Callable[[Mapping[str, float]], float]]:
    """Łączy wbudowane heurystyki z dodatkowymi rejestrami użytkownika."""

    combined: dict[str, Callable[[Mapping[str, float]], float]] = dict(BUILTIN_HEURISTICS)
    for registry in registries:
        if not registry:
            continue
        for name, func in registry.items():
            if callable(func):
                combined[str(name)] = func
    return combined


def select_heuristics(
    names: Sequence[str] | None,
    *,
    registry: Mapping[str, Callable[[Mapping[str, float]], float]] | None = None,
) -> Mapping[str, Callable[[Mapping[str, float]], float]]:
    """Wybiera podzbiór heurystyk, walidując nazwy względem dostępnego rejestru."""

    available = (
        {str(name): func for name, func in registry.items() if callable(func)}
        if registry is not None
        else dict(BUILTIN_HEURISTICS)
    )
    if not names:
        return available

    selected: dict[str, Callable[[Mapping[str, float]], float]] = {}
    missing: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        func = available.get(name)
        if func is None:
            missing.append(name)
        else:
            selected[name] = func

    if missing:
        raise ValueError(
            "Nieznane heurystyki: " + ", ".join(sorted(missing))
        )

    return selected


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
class HeuristicSummary:
    """Zbiorcza informacja o metrykach dla pojedynczej heurystyki."""

    name: str
    metrics: WalkForwardMetrics

    def to_dict(self) -> Mapping[str, object]:
        payload = dict(self.metrics.to_dict())
        payload["name"] = self.name
        return payload


@dataclass(slots=True)
class SequentialTrainingReport:
    artifact: ModelArtifact
    selected_features: Sequence[str]
    feature_ranking: Sequence[tuple[str, float]]
    walk_forward_metrics: WalkForwardMetrics
    heuristic_metrics: WalkForwardMetrics
    heuristic_details: Sequence[HeuristicSummary]
    heuristic_weights: Mapping[str, float]
    suppressed_heuristics: Mapping[str, float]
    heuristic_confidence: Mapping[str, float]


class SequentialTrainingPipeline:
    """Offline sequential training with feature selection and walk-forward validation."""

    def __init__(
        self,
        *,
        repository: HistoricalFeatureRepository,
        heuristics: Mapping[str, Callable[[Mapping[str, float]], float]] | None = None,
        heuristic_names: Sequence[str] | None = None,
        min_directional_accuracy: float = 0.5,
    ) -> None:
        self._repository = repository
        registry = build_heuristic_registry(heuristics)
        self._heuristics = select_heuristics(heuristic_names, registry=registry)
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

        wf_metrics = self._walk_forward(
            features, targets, selected, folds, learning_rate, discount_factor
        )
        (
            heuristic_metrics,
            heuristic_details,
            heuristic_weights,
            suppressed_heuristics,
            heuristic_confidence,
        ) = self._evaluate_heuristics(features, targets, folds)

        final_model = TemporalDifferencePolicy(
            feature_names=selected,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        final_model.fit(features, targets)

        metrics_payload = wf_metrics.to_dict()
        metadata: dict[str, object] = {
            "target_scale": dataset.target_scale,
            "feature_scalers": dataset.feature_stats,
            "walk_forward": metrics_payload,
            "heuristics": heuristic_metrics.to_dict(),
            "heuristics_used": [
                name for name, weight in heuristic_weights.items() if weight > 0.0
            ],
            "heuristics_detail": {
                summary.name: summary.to_dict() for summary in heuristic_details
            },
            "heuristics_weights": dict(heuristic_weights),
            "min_directional_accuracy": self._min_directional_accuracy,
        }
        if suppressed_heuristics:
            metadata["heuristics_suppressed"] = dict(suppressed_heuristics)
        if heuristic_confidence:
            metadata["heuristics_confidence"] = {
                name: float(value) for name, value in heuristic_confidence.items()
            }
        if metrics_payload.get("mean_directional_accuracy", 0.0) < self._min_directional_accuracy:
            metadata["training_warning"] = "directional_accuracy_below_threshold"
        artifact = ModelArtifact(
            feature_names=tuple(selected),
            model_state=final_model.to_state(),
            trained_at=datetime.now(timezone.utc),
            metrics={
                "directional_accuracy": metrics_payload["mean_directional_accuracy"],
                "mae": metrics_payload["mean_mae"],
                "rmse": metrics_payload["mean_rmse"],
                "expected_pnl": metrics_payload["mean_expected_pnl"],
            },
            metadata=metadata,
            backend="sequential_td",
        )

        self._repository.save(dataset)
        return SequentialTrainingReport(
            artifact=artifact,
            selected_features=tuple(selected),
            feature_ranking=tuple(ranking),
            walk_forward_metrics=wf_metrics,
            heuristic_metrics=heuristic_metrics,
            heuristic_details=tuple(heuristic_details),
            heuristic_weights=dict(heuristic_weights),
            suppressed_heuristics=dict(suppressed_heuristics),
            heuristic_confidence=dict(heuristic_confidence),
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
    ) -> tuple[
        WalkForwardMetrics,
        Sequence[HeuristicSummary],
        Mapping[str, float],
        Mapping[str, float],
        Mapping[str, float],
    ]:
        splits = _split_walk_forward(len(features), folds)
        accuracies: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        pnls: list[float] = []
        per_heuristic: list[HeuristicSummary] = []
        per_metric: dict[str, WalkForwardMetrics] = {}
        suppressed: dict[str, float] = {}
        confidence: dict[str, float] = {}
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
        if self._heuristics:
            for name, heuristic in self._heuristics.items():
                metrics = self._evaluate_single_heuristic(heuristic, features, targets, splits)
                per_heuristic.append(HeuristicSummary(name=name, metrics=metrics))
                per_metric[name] = metrics
                accuracy = _mean(metrics.directional_accuracy)
                confidence[name] = accuracy
                if accuracy < self._min_directional_accuracy:
                    suppressed[name] = accuracy
        overall = WalkForwardMetrics(
            directional_accuracy=tuple(accuracies),
            mae=tuple(maes),
            rmse=tuple(rmses),
            pnl=tuple(pnls),
        )
        if not per_metric:
            return overall, tuple(per_heuristic), {}, {}, confidence

        eligible_metrics: dict[str, WalkForwardMetrics] = {
            name: metric
            for name, metric in per_metric.items()
            if name not in suppressed
        }
        # Jeśli wszystkie heurystyki zostały odrzucone, zachowaj najlepszą z nich jako awaryjną.
        if not eligible_metrics and per_metric:
            best_name = max(
                per_metric.items(),
                key=lambda item: _mean(item[1].directional_accuracy),
            )[0]
            eligible_metrics[best_name] = per_metric[best_name]
            suppressed.pop(best_name, None)

        weights = self._compute_heuristic_weights(eligible_metrics)
        # Dołącz zerowe wagi dla heurystyk usuniętych podczas walidacji.
        for name in per_metric:
            weights.setdefault(name, 0.0)
        return overall, tuple(per_heuristic), weights, suppressed, confidence

    def _evaluate_single_heuristic(
        self,
        heuristic: Callable[[Mapping[str, float]], float],
        features: Sequence[Mapping[str, float]],
        targets: Sequence[float],
        splits: Sequence[tuple[int, int]],
    ) -> WalkForwardMetrics:
        accuracies: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        pnls: list[float] = []
        for start, end in splits:
            validation_features = features[start:end]
            validation_targets = targets[start:end]
            if not validation_features:
                continue
            predictions = [heuristic(sample) for sample in validation_features]
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

    @staticmethod
    def _compute_heuristic_weights(
        metrics: Mapping[str, WalkForwardMetrics],
    ) -> Mapping[str, float]:
        if not metrics:
            return {}
        raw_scores: dict[str, float] = {}
        for name, metric in metrics.items():
            accuracy = _mean(metric.directional_accuracy)
            excess = max(0.0, accuracy - 0.5)
            score = excess ** 2
            if score <= 0.0:
                continue
            raw_scores[name] = score
        if not raw_scores:
            uniform = 1.0 / len(metrics)
            return {name: uniform for name in metrics}
        total = sum(raw_scores.values())
        return {name: value / total for name, value in raw_scores.items()}

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
        heuristic_names: Sequence[str] | None = None,
        min_probability: float = 0.55,
        heuristic_weights: Mapping[str, float] | None = None,
        heuristic_confidence: Mapping[str, float] | None = None,
    ) -> None:
        self._model = model
        registry = build_heuristic_registry(heuristics)
        self._heuristics = select_heuristics(heuristic_names, registry=registry)
        self._min_probability = float(min_probability)
        self._heuristic_weights = {
            name: float(weight)
            for name, weight in (heuristic_weights or {}).items()
            if name in self._heuristics
        }
        self._heuristic_confidence = {
            name: float(confidence)
            for name, confidence in (heuristic_confidence or {}).items()
            if name in self._heuristics
        }

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
        heuristic_probability = self._heuristic_probability(heuristic_prediction)
        fallback_score = ModelScore(
            expected_return_bps=heuristic_prediction,
            success_probability=heuristic_probability,
        )
        diagnostics: dict[str, float] = {"heuristic_probability": heuristic_probability}
        for name, confidence in self._heuristic_confidence.items():
            if self._heuristic_weights.get(name, 0.0) <= 0.0:
                continue
            diagnostics[f"heuristic_confidence.{name}"] = float(
                max(0.0, min(confidence, 1.0))
            )
        if model_score is not None:
            diagnostics["model_probability"] = model_score.success_probability
            diagnostics["model_prediction"] = model_score.expected_return_bps
        return OnlineScoringResult(score=fallback_score, source="heuristic", diagnostics=diagnostics)

    def _heuristic_prediction(self, features: Mapping[str, float]) -> float:
        if not self._heuristics:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for name, heuristic in self._heuristics.items():
            weight = self._heuristic_weights.get(name, 1.0)
            if weight <= 0.0:
                continue
            value = heuristic(features)
            weighted_sum += weight * value
            weight_total += weight
        if weight_total <= 0.0:
            return 0.0
        return weighted_sum / weight_total

    def _heuristic_probability(self, prediction: float) -> float:
        if not self._heuristics:
            return 0.5
        weighted_confidence = 0.0
        total_weight = 0.0
        for name in self._heuristics:
            weight = self._heuristic_weights.get(name, 1.0)
            if weight <= 0.0:
                continue
            confidence = self._heuristic_confidence.get(name)
            if confidence is None or not math.isfinite(confidence):
                continue
            bounded = max(0.0, min(confidence, 1.0))
            weighted_confidence += weight * bounded
            total_weight += weight
        if total_weight > 0.0:
            return max(0.5, min(weighted_confidence / total_weight, 0.99))
        return max(0.5, min(0.5 + abs(prediction) / 100.0, 0.99))


__all__ = [
    "BUILTIN_HEURISTICS",
    "build_heuristic_registry",
    "HistoricalFeatureRepository",
    "HeuristicSummary",
    "OnlineScoringResult",
    "select_heuristics",
    "SequentialOnlineScorer",
    "SequentialTrainingPipeline",
    "SequentialTrainingReport",
    "TemporalDifferencePolicy",
    "WalkForwardMetrics",
    "momentum_heuristic",
    "volatility_penalized_momentum",
]

