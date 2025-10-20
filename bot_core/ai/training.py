"""Moduł trenowania prostego modelu gradient boosting dla Decision Engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ._license import ensure_ai_signals_enabled
from .feature_engineering import FeatureDataset
from .models import ModelArtifact


@runtime_checkable
class SupportsInference(Protocol):
    """Minimalny interfejs wymagany od modelu inference."""

    feature_names: Sequence[str]

    def predict(self, features: Mapping[str, float]) -> float: ...

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> Sequence[float]: ...


@dataclass(slots=True)
class ExternalTrainingContext:
    """Dane przekazywane adapterowi modelu zewnętrznego."""

    feature_names: Sequence[str]
    scalers: Mapping[str, tuple[float, float]]
    train_matrix: Sequence[Sequence[float]]
    train_targets: Sequence[float]
    validation_matrix: Sequence[Sequence[float]]
    validation_targets: Sequence[float]
    options: Mapping[str, object]


@dataclass(slots=True)
class ExternalTrainingResult:
    """Wynik treningu modelu zewnętrznego."""

    state: Mapping[str, object]
    trained_model: SupportsInference | None = None
    metrics: Mapping[str, float] | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class ExternalModelAdapter:
    """Adapter obsługujący cykl życia modeli trenowanych poza modułem wbudowanym."""

    backend: str
    train: Callable[[ExternalTrainingContext], ExternalTrainingResult]
    load: Callable[[Mapping[str, object], Sequence[str], Mapping[str, object]], SupportsInference]


_EXTERNAL_ADAPTERS: dict[str, ExternalModelAdapter] = {}


def register_external_model_adapter(adapter: ExternalModelAdapter) -> None:
    """Rejestruje adapter modelu zewnętrznego."""

    ensure_ai_signals_enabled("rejestracji adapterów modeli AI")
    backend = adapter.backend.lower()
    _EXTERNAL_ADAPTERS[backend] = ExternalModelAdapter(
        backend=backend,
        train=adapter.train,
        load=adapter.load,
    )


def get_external_model_adapter(name: str) -> ExternalModelAdapter:
    """Zwraca adapter dla wskazanego backendu."""

    backend = name.lower()
    if backend not in _EXTERNAL_ADAPTERS:
        raise KeyError(f"Brak adaptera dla backendu '{name}'")
    return _EXTERNAL_ADAPTERS[backend]


@dataclass(slots=True)
class DecisionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def to_dict(self) -> Mapping[str, float]:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "DecisionStump":
        return cls(
            feature_index=int(raw.get("feature_index", 0)),
            threshold=float(raw.get("threshold", 0.0)),
            left_value=float(raw.get("left_value", 0.0)),
            right_value=float(raw.get("right_value", 0.0)),
        )


class SimpleGradientBoostingModel:
    """Minimalna implementacja gradient boosting na pniach decyzyjnych."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_estimators: int = 20,
        min_samples_leaf: int = 5,
        max_bins: int = 12,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_bins = int(max_bins)
        self.initial_prediction: float = 0.0
        self.feature_names: list[str] = []
        self.feature_scalers: dict[str, tuple[float, float]] = {}
        self.stumps: list[DecisionStump] = []

    def fit(self, features: Sequence[Mapping[str, float]], targets: Sequence[float]) -> None:
        if not features:
            raise ValueError("Brak danych do trenowania")
        if len(features) != len(targets):
            raise ValueError("Liczba cech i targetów musi być taka sama")
        feature_names = sorted({name for sample in features for name in sample.keys()})
        matrix = [[float(sample.get(name, 0.0)) for name in feature_names] for sample in features]
        scalers = self._compute_feature_scalers(matrix, feature_names)
        self.fit_matrix(matrix, feature_names, targets, feature_scalers=scalers)

    def fit_matrix(
        self,
        matrix: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        targets: Sequence[float],
        *,
        feature_scalers: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        if not matrix:
            raise ValueError("Brak danych do trenowania")
        if len(matrix) != len(targets):
            raise ValueError("Liczba cech i targetów musi być taka sama")
        if not feature_names:
            raise ValueError("Brak nazw cech dla macierzy treningowej")
        width = len(feature_names)
        if any(len(row) != width for row in matrix):
            raise ValueError("Wiersze macierzy muszą mieć tyle samo kolumn co feature_names")

        self.feature_names = [str(name) for name in feature_names]
        if feature_scalers is None:
            feature_scalers = self._compute_feature_scalers(matrix, self.feature_names)
        self.feature_scalers = {
            str(name): (float(values[0]), float(values[1]))
            for name, values in feature_scalers.items()
        }
        target_values = [float(value) for value in targets]
        self.initial_prediction = sum(target_values) / len(target_values)
        residuals = [y - self.initial_prediction for y in target_values]
        self.stumps = []

        float_matrix = [self._normalize_row([float(value) for value in row]) for row in matrix]

        for _ in range(self.n_estimators):
            stump = self._find_best_stump(float_matrix, residuals)
            if stump is None:
                break
            self.stumps.append(stump)
            predictions = [self._stump_value(row, stump) for row in float_matrix]
            residuals = [r - self.learning_rate * pred for r, pred in zip(residuals, predictions)]
            if all(abs(value) < 1e-6 for value in residuals):
                break

    def predict(self, features: Mapping[str, float]) -> float:
        if not self.feature_names:
            raise RuntimeError("Model nie został wytrenowany")
        vector = self._normalize_features(features)
        value = self.initial_prediction
        for stump in self.stumps:
            value += self.learning_rate * self._stump_value(vector, stump)
        return value

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> list[float]:
        return [self.predict(sample) for sample in samples]

    def to_state(self) -> Mapping[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "n_estimators": len(self.stumps),
            "initial_prediction": self.initial_prediction,
            "feature_names": list(self.feature_names),
            "feature_scalers": {
                name: {"mean": mean, "stdev": stdev}
                for name, (mean, stdev) in self.feature_scalers.items()
            },
            "stumps": [stump.to_dict() for stump in self.stumps],
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.initial_prediction = float(state.get("initial_prediction", 0.0))
        self.feature_names = [str(name) for name in state.get("feature_names", [])]
        scalers_raw = state.get("feature_scalers", {})
        self.feature_scalers = {}
        if isinstance(scalers_raw, Mapping):
            for name, raw in scalers_raw.items():
                if not isinstance(raw, Mapping):
                    continue
                mean = float(raw.get("mean", 0.0))
                stdev = float(raw.get("stdev", 0.0))
                self.feature_scalers[str(name)] = (mean, stdev)
        self.stumps = [DecisionStump.from_dict(raw) for raw in state.get("stumps", [])]

    # ------------------------------------------------------------------ helpers --
    def _find_best_stump(
        self, matrix: list[list[float]], residuals: list[float]
    ) -> DecisionStump | None:
        best: DecisionStump | None = None
        best_error = math.inf
        feature_count = len(self.feature_names)
        if feature_count == 0:
            return None
        n_samples = len(matrix)
        for feature_index in range(feature_count):
            candidates = self._candidate_thresholds(matrix, feature_index)
            for threshold in candidates:
                left_indices = [i for i, row in enumerate(matrix) if row[feature_index] <= threshold]
                right_indices = [i for i, row in enumerate(matrix) if row[feature_index] > threshold]
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                left_value = self._mean(residuals, left_indices)
                right_value = self._mean(residuals, right_indices)
                error = self._squared_error(residuals, left_indices, left_value)
                error += self._squared_error(residuals, right_indices, right_value)
                if error < best_error:
                    best_error = error
                    best = DecisionStump(
                        feature_index=feature_index,
                        threshold=threshold,
                        left_value=left_value,
                        right_value=right_value,
                    )
        return best

    def _candidate_thresholds(
        self, matrix: list[list[float]], feature_index: int
    ) -> Sequence[float]:
        values = sorted({row[feature_index] for row in matrix})
        if len(values) <= self.max_bins:
            return values
        step = max(1, len(values) // self.max_bins)
        return [values[i] for i in range(0, len(values), step)]

    def _stump_value(self, row: Sequence[float], stump: DecisionStump) -> float:
        return stump.left_value if row[stump.feature_index] <= stump.threshold else stump.right_value

    @staticmethod
    def _mean(values: Sequence[float], indices: Iterable[int]) -> float:
        selected = [values[i] for i in indices]
        if not selected:
            return 0.0
        return sum(selected) / len(selected)

    @staticmethod
    def _squared_error(values: Sequence[float], indices: Iterable[int], ref: float) -> float:
        return sum((values[i] - ref) ** 2 for i in indices)

    def _compute_feature_scalers(
        self, matrix: Sequence[Sequence[float]], feature_names: Sequence[str]
    ) -> dict[str, tuple[float, float]]:
        scalers: dict[str, tuple[float, float]] = {}
        width = len(feature_names)
        for idx in range(width):
            column = [float(row[idx]) for row in matrix]
            mean_value = sum(column) / len(column)
            variance = sum((value - mean_value) ** 2 for value in column) / len(column)
            stdev_value = math.sqrt(variance)
            scalers[str(feature_names[idx])] = (float(mean_value), float(stdev_value))
        return scalers

    def _normalize_features(self, features: Mapping[str, float]) -> list[float]:
        vector = []
        for name in self.feature_names:
            mean, stdev = self.feature_scalers.get(name, (0.0, 1.0))
            raw = features.get(name)
            value = float(raw) if raw is not None else mean
            if stdev > 0:
                vector.append((value - mean) / stdev)
            else:
                vector.append(value - mean)
        return vector

    def _normalize_row(self, row: list[float]) -> list[float]:
        normalized: list[float] = []
        for idx, name in enumerate(self.feature_names):
            mean, stdev = self.feature_scalers.get(name, (0.0, 1.0))
            value = row[idx]
            if stdev > 0:
                normalized.append((value - mean) / stdev)
            else:
                normalized.append(value - mean)
        return normalized


class ModelTrainer:
    """Łączy FeatureDataset i wybrany backend modelu, zwracając artefakt."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_estimators: int = 25,
        validation_split: float = 0.0,
        backend: str = "builtin",
        adapter_options: Mapping[str, object] | None = None,
    ) -> None:
        ensure_ai_signals_enabled("trenowania modeli AI")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split musi zawierać się w przedziale [0, 1)")
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.validation_split = float(validation_split)
        self.backend = backend.lower().strip() or "builtin"
        self.adapter_options = dict(adapter_options or {})
        if self.backend != "builtin" and self.backend not in _EXTERNAL_ADAPTERS:
            raise KeyError(f"Nieznany backend modelu: {backend}")

    def train(self, dataset: FeatureDataset) -> ModelArtifact:
        if not dataset.vectors:
            raise ValueError("Dataset nie zawiera danych")
        matrix, targets, feature_names = dataset.to_learning_arrays()
        (
            train_matrix,
            train_targets,
            validation_matrix,
            validation_targets,
        ) = self._split_learning_arrays(matrix, targets)
        if not train_matrix:
            raise ValueError("Za mało danych do trenowania po podziale walidacyjnym")

        scalers = self._compute_scalers(train_matrix, feature_names)
        metadata: MutableMapping[str, object] = dict(dataset.metadata)
        metadata.setdefault("target_scale", dataset.target_scale)
        metadata["feature_scalers"] = {
            name: {"mean": mean, "stdev": stdev}
            for name, (mean, stdev) in scalers.items()
        }
        metadata["training_rows"] = len(train_matrix)
        metadata["validation_rows"] = len(validation_matrix)
        metadata["backend"] = self.backend
        if self.validation_split > 0.0:
            metadata["validation_split"] = self.validation_split

        if self.backend == "builtin":
            return self._train_builtin(
                train_matrix,
                train_targets,
                validation_matrix,
                validation_targets,
                feature_names,
                scalers,
                metadata,
            )
        return self._train_external(
            train_matrix,
            train_targets,
            validation_matrix,
            validation_targets,
            feature_names,
            scalers,
            metadata,
        )

    def _train_builtin(
        self,
        train_matrix: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        validation_matrix: Sequence[Sequence[float]],
        validation_targets: Sequence[float],
        feature_names: Sequence[str],
        scalers: Mapping[str, tuple[float, float]],
        metadata: MutableMapping[str, object],
    ) -> ModelArtifact:
        model = SimpleGradientBoostingModel(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
        )
        model.fit_matrix(
            train_matrix,
            feature_names,
            train_targets,
            feature_scalers=scalers,
        )
        train_samples = self._rows_to_samples(train_matrix, feature_names)
        train_predictions = model.batch_predict(train_samples)
        train_metrics = self._compute_metrics(train_targets, train_predictions)
        metrics = self._compose_metrics(train_metrics)
        validation_metrics: Mapping[str, float] | None = None
        if validation_matrix:
            validation_samples = self._rows_to_samples(validation_matrix, feature_names)
            validation_predictions = model.batch_predict(validation_samples)
            validation_metrics = self._compute_metrics(validation_targets, validation_predictions)
            metrics = self._compose_metrics(train_metrics, validation_metrics)
        if validation_metrics:
            metadata["validation_metrics"] = dict(validation_metrics)
        artifact = ModelArtifact(
            feature_names=tuple(model.feature_names),
            model_state=model.to_state(),
            trained_at=datetime.now(timezone.utc),
            metrics=metrics,
            metadata=metadata,
            backend=self.backend,
        )
        return artifact

    def _train_external(
        self,
        train_matrix: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        validation_matrix: Sequence[Sequence[float]],
        validation_targets: Sequence[float],
        feature_names: Sequence[str],
        scalers: Mapping[str, tuple[float, float]],
        metadata: MutableMapping[str, object],
    ) -> ModelArtifact:
        adapter = get_external_model_adapter(self.backend)
        context = ExternalTrainingContext(
            feature_names=feature_names,
            scalers=scalers,
            train_matrix=train_matrix,
            train_targets=train_targets,
            validation_matrix=validation_matrix,
            validation_targets=validation_targets,
            options=self.adapter_options,
        )
        result = adapter.train(context)
        model = result.trained_model
        if model is None:
            model = adapter.load(result.state, feature_names, metadata)
        train_samples = self._rows_to_samples(train_matrix, feature_names)
        train_predictions = list(model.batch_predict(train_samples))
        train_metrics = self._compute_metrics(train_targets, train_predictions)
        metrics = self._compose_metrics(train_metrics)
        validation_metrics: Mapping[str, float] | None = None
        if validation_matrix:
            validation_samples = self._rows_to_samples(validation_matrix, feature_names)
            validation_predictions = list(model.batch_predict(validation_samples))
            validation_metrics = self._compute_metrics(validation_targets, validation_predictions)
            metrics = self._compose_metrics(train_metrics, validation_metrics)
        if validation_metrics:
            metadata["validation_metrics"] = dict(validation_metrics)
        if result.metrics:
            metrics = {**metrics, **result.metrics}
        if result.metadata:
            metadata.update(result.metadata)
        artifact = ModelArtifact(
            feature_names=tuple(feature_names),
            model_state=dict(result.state),
            trained_at=datetime.now(timezone.utc),
            metrics=metrics,
            metadata=metadata,
            backend=self.backend,
        )
        return artifact

    def _compute_metrics(
        self, targets: Sequence[float], predictions: Sequence[float]
    ) -> Mapping[str, float]:
        if not targets:
            return {"mae": 0.0, "rmse": 0.0}
        errors = [abs(t - p) for t, p in zip(targets, predictions)]
        mae = sum(errors) / len(errors)
        mse = sum((t - p) ** 2 for t, p in zip(targets, predictions)) / len(targets)
        rmse = math.sqrt(mse)
        directional_hits = sum(
            1
            for t, p in zip(targets, predictions)
            if (t >= 0 and p >= 0) or (t < 0 and p < 0)
        )
        accuracy = directional_hits / len(targets)
        return {"mae": mae, "rmse": rmse, "directional_accuracy": accuracy}

    def _split_learning_arrays(
        self,
        matrix: Sequence[Sequence[float]],
        targets: Sequence[float],
    ) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
        total = len(matrix)
        if total == 0:
            return [], [], [], []
        if self.validation_split <= 0.0 or total < 2:
            return (
                [list(row) for row in matrix],
                [float(value) for value in targets],
                [],
                [],
            )
        validation_count = int(total * self.validation_split)
        if validation_count <= 0:
            validation_count = 1
        if validation_count >= total:
            validation_count = total - 1
        split_index = total - validation_count
        return (
            [list(row) for row in matrix[:split_index]],
            [float(targets[idx]) for idx in range(split_index)],
            [list(row) for row in matrix[split_index:]],
            [float(targets[idx]) for idx in range(split_index, total)],
        )

    def _compute_scalers(
        self, matrix: Sequence[Sequence[float]], feature_names: Sequence[str]
    ) -> dict[str, tuple[float, float]]:
        scalers: dict[str, tuple[float, float]] = {}
        if not matrix:
            return scalers
        width = len(feature_names)
        for idx in range(width):
            column = [float(row[idx]) for row in matrix]
            mean_value = sum(column) / len(column)
            variance = sum((value - mean_value) ** 2 for value in column) / len(column)
            scalers[str(feature_names[idx])] = (
                float(mean_value),
                float(math.sqrt(variance)),
            )
        return scalers

    def _rows_to_samples(
        self, matrix: Sequence[Sequence[float]], feature_names: Sequence[str]
    ) -> list[Mapping[str, float]]:
        return [
            {name: float(row[idx]) for idx, name in enumerate(feature_names)}
            for row in matrix
        ]

    def _compose_metrics(
        self,
        train_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float] | None = None,
    ) -> Mapping[str, float]:
        metrics: MutableMapping[str, float] = {
            "mae": float(train_metrics.get("mae", 0.0)),
            "rmse": float(train_metrics.get("rmse", 0.0)),
            "directional_accuracy": float(train_metrics.get("directional_accuracy", 0.0)),
            "train_mae": float(train_metrics.get("mae", 0.0)),
            "train_rmse": float(train_metrics.get("rmse", 0.0)),
            "train_directional_accuracy": float(train_metrics.get("directional_accuracy", 0.0)),
        }
        if validation_metrics:
            metrics.update(
                {
                    "validation_mae": float(validation_metrics.get("mae", 0.0)),
                    "validation_rmse": float(validation_metrics.get("rmse", 0.0)),
                    "validation_directional_accuracy": float(
                        validation_metrics.get("directional_accuracy", 0.0)
                    ),
                }
            )
        return metrics


class _LinearAdapterModel:
    """Lekki model liniowy wykorzystywany przez domyślne adaptery zewnętrzne."""

    def __init__(
        self,
        feature_names: Sequence[str],
        scalers: Mapping[str, tuple[float, float]],
        weights: Sequence[float],
        bias: float,
    ) -> None:
        self.feature_names = [str(name) for name in feature_names]
        self.feature_scalers = {
            str(name): (float(pair[0]), float(pair[1])) for name, pair in scalers.items()
        }
        self._weights = [float(value) for value in weights]
        self._bias = float(bias)

    def predict(self, features: Mapping[str, float]) -> float:
        vector = self._vector_from_mapping(features)
        return self._bias + sum(weight * value for weight, value in zip(self._weights, vector))

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> Sequence[float]:
        return [self.predict(sample) for sample in samples]

    def _vector_from_mapping(self, features: Mapping[str, float]) -> list[float]:
        vector: list[float] = []
        for name in self.feature_names:
            mean, stdev = self.feature_scalers.get(name, (0.0, 1.0))
            raw = features.get(name, mean)
            value = float(raw)
            if stdev > 0:
                vector.append((value - mean) / stdev)
            else:
                vector.append(value - mean)
        return vector


def _normalize_matrix(
    matrix: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    scalers: Mapping[str, tuple[float, float]],
) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        vector: list[float] = []
        for idx, name in enumerate(feature_names):
            mean, stdev = scalers.get(name, (0.0, 1.0))
            value = float(row[idx])
            if stdev > 0:
                vector.append((value - mean) / stdev)
            else:
                vector.append(value - mean)
        normalized.append(vector)
    return normalized


def _fit_linear_regression(
    matrix: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], float]:
    if not matrix:
        return [], 0.0
    width = len(matrix[0])
    weights = [0.0 for _ in range(width)]
    bias = sum(targets) / len(targets) if targets else 0.0
    count = len(matrix)
    for _ in range(iterations):
        grad_w = [0.0 for _ in range(width)]
        grad_b = 0.0
        for row, target in zip(matrix, targets):
            prediction = bias + sum(weight * value for weight, value in zip(weights, row))
            error = prediction - target
            grad_b += error
            for idx, value in enumerate(row):
                grad_w[idx] += error * value
        count_safe = max(count, 1)
        bias -= learning_rate * (grad_b / count_safe)
        for idx in range(width):
            grad = grad_w[idx] / count_safe + l2 * weights[idx]
            weights[idx] -= learning_rate * grad
    return weights, bias


def _linear_adapter_train(context: ExternalTrainingContext) -> ExternalTrainingResult:
    iterations = int(context.options.get("iterations", 300))
    learning_rate = float(context.options.get("learning_rate", 0.05))
    l2 = float(context.options.get("l2", 0.0))
    normalized = _normalize_matrix(context.train_matrix, context.feature_names, context.scalers)
    weights, bias = _fit_linear_regression(
        normalized,
        context.train_targets,
        iterations=max(iterations, 1),
        learning_rate=max(learning_rate, 1e-4),
        l2=max(l2, 0.0),
    )
    model = _LinearAdapterModel(
        feature_names=context.feature_names,
        scalers=context.scalers,
        weights=weights,
        bias=bias,
    )
    state: MutableMapping[str, object] = {
        "weights": list(weights),
        "bias": bias,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "l2": l2,
    }
    return ExternalTrainingResult(
        state=state,
        trained_model=model,
        metadata={"external_adapter": "linear_regression"},
    )


def _linear_adapter_load(
    state: Mapping[str, object],
    feature_names: Sequence[str],
    metadata: Mapping[str, object],
) -> SupportsInference:
    scalers_raw = metadata.get("feature_scalers", {})
    scalers: dict[str, tuple[float, float]] = {}
    if isinstance(scalers_raw, Mapping):
        for name, payload in scalers_raw.items():
            if not isinstance(payload, Mapping):
                continue
            mean = float(payload.get("mean", 0.0))
            stdev = float(payload.get("stdev", 0.0))
            scalers[str(name)] = (mean, stdev)
    weights = [float(value) for value in state.get("weights", [])]
    bias = float(state.get("bias", 0.0))
    return _LinearAdapterModel(feature_names, scalers, weights, bias)


def _ensure_default_external_adapters() -> None:
    for name in ("pytorch", "lightgbm", "linear"):
        if name not in _EXTERNAL_ADAPTERS:
            register_external_model_adapter(
                ExternalModelAdapter(
                    backend=name,
                    train=_linear_adapter_train,
                    load=_linear_adapter_load,
                )
            )


_ensure_default_external_adapters()


__all__ = [
    "DecisionStump",
    "ExternalModelAdapter",
    "ExternalTrainingContext",
    "ExternalTrainingResult",
    "ModelTrainer",
    "SimpleGradientBoostingModel",
    "get_external_model_adapter",
    "register_external_model_adapter",
]
