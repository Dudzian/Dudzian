"""Pipelines for training and registering Decision Engine models."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .inference import DecisionModelInference, ModelRepository
from .meta import build_meta_labeling_payload, train_meta_classifier
from .models import ModelArtifact, ModelScore
from .training import SimpleGradientBoostingModel

try:  # pragma: no cover - opcjonalna zależność
    import yaml
except Exception:  # pragma: no cover - środowiska minimalne mogą nie mieć PyYAML
    yaml = None

try:  # pragma: no cover - import na potrzeby typowania
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python <3.8 fallback
    TYPE_CHECKING = False  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - tylko dla statycznych analizatorów
    from .manager import AIManager


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
class TrainingDatasetSpec:
    """Opis pojedynczego zbioru danych wykorzystywanego w profilu treningowym."""

    name: str
    path: Path | None = None
    format: str = "csv"

    def load_frame(self, overrides: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        if overrides and self.name in overrides:
            frame = overrides[self.name]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"override dla datasetu {self.name!r} nie jest DataFrame")
            return frame.copy()
        if self.path is None:
            raise FileNotFoundError(f"Dataset {self.name!r} nie posiada ścieżki do danych")
        return _load_frame_from_path(self.path)


@dataclass(slots=True)
class TrainingModelSpec:
    """Specyfikacja modelu budowanego w ramach profilu treningowego."""

    name: str
    dataset: str
    target: str
    features: tuple[str, ...]
    trainer: str = "gradient_boosting"
    options: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    publish_aliases: tuple[str, ...] = field(default_factory=tuple)
    activate_version: bool = True
    decision_name: str | None = None
    set_default: bool = False

    def normalized_trainer(self) -> str:
        return self.trainer.strip().lower()


@dataclass(slots=True)
class TrainingEnsembleSpec:
    """Definicja zespołu modeli rejestrowanego po treningu."""

    name: str
    components: tuple[str, ...]
    aggregation: str = "mean"
    weights: tuple[float, ...] | None = None
    regime_weights: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    meta_weight_floor: float = 0.0


@dataclass(slots=True)
class QualityThresholds:
    """Progi jakości wykorzystywane podczas automatycznego retrainingu."""

    min_directional_accuracy: float | None = None
    max_mae: float | None = None
    max_rmse: float | None = None


@dataclass(slots=True)
class AutoRetrainPolicy:
    """Parametry automatycznego retrainingu dla profilu."""

    interval_seconds: float
    quality: QualityThresholds | None = None
    journal_environment: str = "paper-trading"
    journal_portfolio: str = "ai-pipeline"
    journal_risk_profile: str = "ai-research"
    journal_strategy: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def interval_timedelta(self) -> "timedelta | None":
        from datetime import timedelta

        if self.interval_seconds <= 0:
            return None
        return timedelta(seconds=float(self.interval_seconds))


@dataclass(slots=True)
class TrainingProfile:
    """Kompletna definicja profilu treningowego."""

    name: str
    datasets: Mapping[str, TrainingDatasetSpec]
    models: tuple[TrainingModelSpec, ...]
    ensembles: tuple[TrainingEnsembleSpec, ...] = ()
    auto_retrain: AutoRetrainPolicy | None = None
    description: str | None = None

    def dataset(self, name: str) -> TrainingDatasetSpec:
        try:
            return self.datasets[name]
        except KeyError as exc:
            raise KeyError(f"Dataset {name!r} nie istnieje w profilu {self.name!r}") from exc


@dataclass(slots=True)
class TrainingManifest:
    """Zbiór profili treningowych dostępnych dla pipeline'u."""

    profiles: Mapping[str, TrainingProfile]

    def profile(self, name: str) -> TrainingProfile:
        key = name.strip().lower()
        for candidate_name, profile in self.profiles.items():
            if candidate_name.lower() == key:
                return profile
        available = ", ".join(sorted(self.profiles)) or "<brak>"
        raise KeyError(f"Profil {name!r} nie został odnaleziony (dostępne: {available})")

    def list_profiles(self) -> tuple[str, ...]:
        return tuple(sorted(self.profiles))


@dataclass(slots=True)
class ModelTrainingResult:
    """Szczegóły pojedynczego modelu wytrenowanego w profilu."""

    spec: TrainingModelSpec
    artifact_path: Path
    metrics: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(slots=True)
class ProfileTrainingSummary:
    """Wynik wykonania profilu treningowego."""

    profile: TrainingProfile
    results: tuple[ModelTrainingResult, ...]

    @property
    def models(self) -> tuple[ModelTrainingResult, ...]:
        return self.results


_GBM_OPTIONS: frozenset[str] = frozenset(
    {
        "validation_ratio",
        "test_ratio",
        "random_state",
        "model_version",
        "publish_aliases",
        "activate_version",
    }
)


def _ensure_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise TypeError(f"Pole {field_name} musi być sekwencją lub stringiem")


def _ensure_optional_sequence_of_float(value: object, *, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Iterable):
        return tuple(float(item) for item in value)
    raise TypeError(f"Pole {field_name} musi być sekwencją liczb rzeczywistych")


def _load_structured_payload(source: Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    payload_text = Path(source).read_text(encoding="utf-8")
    suffix = Path(source).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("Obsługa YAML wymaga pakietu PyYAML")
        loaded = yaml.safe_load(payload_text) or {}
        if not isinstance(loaded, MutableMapping):
            raise TypeError("Manifest YAML musi być obiektem mapującym u korzenia")
        return loaded
    data = json.loads(payload_text or "{}")
    if not isinstance(data, MutableMapping):
        raise TypeError("Manifest JSON musi być obiektem mapującym u korzenia")
    return data


def _parse_datasets(profile_name: str, payload: Mapping[str, object]) -> dict[str, TrainingDatasetSpec]:
    datasets_raw = payload.get("datasets") or {}
    if not isinstance(datasets_raw, Mapping):
        raise TypeError(f"profiles.{profile_name}.datasets musi być mapowaniem")
    datasets: dict[str, TrainingDatasetSpec] = {}
    for dataset_name, dataset_payload in datasets_raw.items():
        if not isinstance(dataset_payload, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.datasets.{dataset_name} musi być mapowaniem"
            )
        path_raw = dataset_payload.get("path")
        path = Path(path_raw).expanduser() if path_raw is not None else None
        fmt = str(dataset_payload.get("format", "csv")).lower()
        datasets[str(dataset_name)] = TrainingDatasetSpec(
            name=str(dataset_name),
            path=path,
            format=fmt,
        )
    return datasets


def _parse_models(profile_name: str, payload: Mapping[str, object]) -> tuple[TrainingModelSpec, ...]:
    models_raw = payload.get("models") or ()
    if not isinstance(models_raw, Iterable):
        raise TypeError(f"profiles.{profile_name}.models musi być sekwencją")
    models: list[TrainingModelSpec] = []
    for index, model_payload in enumerate(models_raw):
        if not isinstance(model_payload, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.models[{index}] musi być mapowaniem"
            )
        name = str(model_payload.get("name") or f"model_{index}")
        dataset = str(model_payload.get("dataset") or "")
        if not dataset:
            raise ValueError(f"profiles.{profile_name}.models[{index}] wymaga pola dataset")
        target = str(model_payload.get("target") or "")
        if not target:
            raise ValueError(f"profiles.{profile_name}.models[{index}] wymaga pola target")
        features_raw = model_payload.get("features")
        features = _ensure_sequence(features_raw, field_name=f"profiles.{profile_name}.models[{index}].features")
        if not features:
            raise ValueError(
                f"profiles.{profile_name}.models[{index}] wymaga co najmniej jednej cechy"
            )
        trainer = str(model_payload.get("trainer", "gradient_boosting"))
        options_raw = model_payload.get("options") or {}
        if not isinstance(options_raw, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.models[{index}].options musi być mapowaniem"
            )
        options = {str(key): value for key, value in options_raw.items() if key in _GBM_OPTIONS}
        metadata_raw = model_payload.get("metadata") or {}
        if not isinstance(metadata_raw, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.models[{index}].metadata musi być mapowaniem"
            )
        publish_aliases = _ensure_sequence(
            model_payload.get("publish_aliases"),
            field_name=f"profiles.{profile_name}.models[{index}].publish_aliases",
        )
        activate_version = bool(model_payload.get("activate_version", True))
        decision_name_raw = model_payload.get("decision_name")
        decision_name = str(decision_name_raw) if decision_name_raw is not None else None
        set_default = bool(model_payload.get("set_default", False))
        models.append(
            TrainingModelSpec(
                name=name,
                dataset=dataset,
                target=target,
                features=features,
                trainer=trainer,
                options=options,
                metadata=dict(metadata_raw),
                publish_aliases=publish_aliases,
                activate_version=activate_version,
                decision_name=decision_name,
                set_default=set_default,
            )
        )
    return tuple(models)


def _parse_ensembles(profile_name: str, payload: Mapping[str, object]) -> tuple[TrainingEnsembleSpec, ...]:
    raw = payload.get("ensembles") or ()
    if not raw:
        return ()
    if not isinstance(raw, Iterable):
        raise TypeError(f"profiles.{profile_name}.ensembles musi być sekwencją")
    ensembles: list[TrainingEnsembleSpec] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.ensembles[{index}] musi być mapowaniem"
            )
        name = str(entry.get("name") or f"ensemble_{index}")
        components_raw = entry.get("components")
        components = _ensure_sequence(
            components_raw,
            field_name=f"profiles.{profile_name}.ensembles[{index}].components",
        )
        if not components:
            raise ValueError(
                f"profiles.{profile_name}.ensembles[{index}] wymaga listy komponentów"
            )
        aggregation = str(entry.get("aggregation", "mean"))
        weights = _ensure_optional_sequence_of_float(
            entry.get("weights"),
            field_name=f"profiles.{profile_name}.ensembles[{index}].weights",
        )
        regime_weights_raw = entry.get("regime_weights") or {}
        if regime_weights_raw and not isinstance(regime_weights_raw, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.ensembles[{index}].regime_weights musi być mapowaniem"
            )
        regime_weights: dict[str, tuple[float, ...]] = {}
        if isinstance(regime_weights_raw, Mapping):
            for regime_name, values in regime_weights_raw.items():
                vector = _ensure_optional_sequence_of_float(
                    values,
                    field_name=(
                        f"profiles.{profile_name}.ensembles[{index}].regime_weights.{regime_name}"
                    ),
                )
                if vector is None:
                    continue
                if len(vector) != len(components):
                    raise ValueError(
                        "Długość wektora regime_weights musi odpowiadać liczbie komponentów"
                    )
                regime_weights[str(regime_name).lower()] = tuple(vector)
        meta_weight_floor_raw = entry.get("meta_weight_floor", 0.0)
        meta_weight_floor = float(meta_weight_floor_raw)
        ensembles.append(
            TrainingEnsembleSpec(
                name=name,
                components=components,
                aggregation=aggregation,
                weights=weights,
                regime_weights=regime_weights,
                meta_weight_floor=meta_weight_floor,
            )
        )
    return tuple(ensembles)


def _parse_auto_retrain(profile_name: str, payload: Mapping[str, object]) -> AutoRetrainPolicy | None:
    raw = payload.get("auto_retrain")
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"profiles.{profile_name}.auto_retrain musi być mapowaniem")
    interval = float(raw.get("interval_seconds", 0.0))
    if interval <= 0:
        raise ValueError(
            f"profiles.{profile_name}.auto_retrain.interval_seconds musi być dodatni"
        )
    quality_raw = raw.get("quality") or {}
    quality: QualityThresholds | None
    if quality_raw:
        if not isinstance(quality_raw, Mapping):
            raise TypeError(
                f"profiles.{profile_name}.auto_retrain.quality musi być mapowaniem"
            )
        quality = QualityThresholds(
            min_directional_accuracy=(
                float(quality_raw["min_directional_accuracy"])
                if "min_directional_accuracy" in quality_raw
                else None
            ),
            max_mae=(
                float(quality_raw["max_mae"])
                if "max_mae" in quality_raw
                else None
            ),
            max_rmse=(
                float(quality_raw["max_rmse"])
                if "max_rmse" in quality_raw
                else None
            ),
        )
    else:
        quality = None
    journal_raw = raw.get("journal") or {}
    if not isinstance(journal_raw, Mapping):
        raise TypeError(
            f"profiles.{profile_name}.auto_retrain.journal musi być mapowaniem"
        )
    metadata_raw = raw.get("metadata") or {}
    if metadata_raw and not isinstance(metadata_raw, Mapping):
        raise TypeError(
            f"profiles.{profile_name}.auto_retrain.metadata musi być mapowaniem"
        )
    return AutoRetrainPolicy(
        interval_seconds=interval,
        quality=quality,
        journal_environment=str(journal_raw.get("environment", "paper-trading")),
        journal_portfolio=str(journal_raw.get("portfolio", "ai-pipeline")),
        journal_risk_profile=str(journal_raw.get("risk_profile", "ai-research")),
        journal_strategy=(
            str(journal_raw.get("strategy")) if journal_raw.get("strategy") is not None else None
        ),
        metadata={str(key): str(value) for key, value in dict(metadata_raw).items()},
    )


def load_training_manifest(source: Path | Mapping[str, object]) -> TrainingManifest:
    """Wczytaj manifest profili treningowych z pliku YAML/JSON lub mapowania."""

    payload = _load_structured_payload(source)
    profiles_raw = payload.get("profiles") or {}
    if not isinstance(profiles_raw, Mapping):
        raise TypeError("Pole 'profiles' musi być mapowaniem")
    profiles: dict[str, TrainingProfile] = {}
    for profile_name, profile_payload in profiles_raw.items():
        if not isinstance(profile_payload, Mapping):
            raise TypeError(f"profiles.{profile_name} musi być mapowaniem")
        datasets = _parse_datasets(str(profile_name), profile_payload)
        models = _parse_models(str(profile_name), profile_payload)
        ensembles = _parse_ensembles(str(profile_name), profile_payload)
        auto_retrain = _parse_auto_retrain(str(profile_name), profile_payload)
        description_raw = profile_payload.get("description")
        description = str(description_raw) if description_raw is not None else None
        profiles[str(profile_name)] = TrainingProfile(
            name=str(profile_name),
            datasets=datasets,
            models=models,
            ensembles=ensembles,
            auto_retrain=auto_retrain,
            description=description,
        )
    return TrainingManifest(profiles=profiles)


def run_training_profile(
    manifest: TrainingManifest,
    profile_name: str,
    *,
    output_dir: Path,
    dataset_overrides: Mapping[str, pd.DataFrame] | None = None,
) -> ProfileTrainingSummary:
    """Uruchom trening dla wskazanego profilu i zwróć podsumowanie wyników."""

    profile = manifest.profile(profile_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    overrides = dict(dataset_overrides) if dataset_overrides else {}
    frames: dict[str, pd.DataFrame] = {}
    for dataset_name, dataset_spec in profile.datasets.items():
        frames[dataset_name] = dataset_spec.load_frame(overrides)

    results: list[ModelTrainingResult] = []
    for spec in profile.models:
        if spec.dataset not in frames:
            raise KeyError(
                f"Dataset {spec.dataset!r} wymagany przez model {spec.name!r} nie został zdefiniowany"
            )
        frame = frames[spec.dataset]
        trainer = spec.normalized_trainer()
        if trainer != "gradient_boosting":
            raise ValueError(f"Nieobsługiwany typ trenera: {spec.trainer!r}")
        options: dict[str, object] = {}
        for key, value in spec.options.items():
            if key in _GBM_OPTIONS:
                options[key] = value
        if spec.publish_aliases:
            options["publish_aliases"] = tuple(spec.publish_aliases)
        elif "publish_aliases" in options and not options["publish_aliases"]:
            options.pop("publish_aliases")
        if "activate_version" not in options:
            options["activate_version"] = bool(spec.activate_version)
        artifact_path = train_gradient_boosting_model(
            frame,
            spec.features,
            spec.target,
            output_dir=output_path,
            model_name=spec.name,
            metadata=dict(spec.metadata),
            publish_aliases=options.pop("publish_aliases", None),
            activate_version=bool(options.pop("activate_version", True)),
            **options,
        )
        artifact_file = Path(artifact_path)
        payload = json.loads(artifact_file.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        results.append(
            ModelTrainingResult(
                spec=spec,
                artifact_path=artifact_file,
                metrics=dict(metrics),
                metadata=dict(metadata),
            )
        )
    return ProfileTrainingSummary(profile=profile, results=tuple(results))


def register_profile_results(
    summary: ProfileTrainingSummary,
    manager: "AIManager",
    *,
    repository_root: Path,
    register_ensembles: bool = True,
    set_default_if_missing: bool = True,
) -> None:
    """Zarejestruj wyniki profilu w AIManagerze i ewentualnie zaktualizuj zespoły."""

    repo_root = Path(repository_root)
    manager.configure_decision_repository(repo_root)
    default_assigned = False
    for result in summary.models:
        decision_name = result.spec.decision_name or result.spec.name
        mark_default = result.spec.set_default or (set_default_if_missing and not default_assigned)
        manager.load_decision_artifact(
            decision_name,
            result.artifact_path,
            repository_root=repo_root,
            set_default=mark_default,
        )
        if mark_default:
            default_assigned = True
    if register_ensembles and summary.profile.ensembles:
        for ensemble in summary.profile.ensembles:
            manager.register_ensemble(
                ensemble.name,
                ensemble.components,
                aggregation=ensemble.aggregation,
                weights=ensemble.weights,
                regime_weights=ensemble.regime_weights,
                meta_weight_floor=ensemble.meta_weight_floor,
                override=True,
            )

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
    subset_lookup = {subset.name: subset for subset in learning_sets}
    subset_predictions: dict[str, Sequence[float]] = {"train": raw_predictions}
    for subset in learning_sets[1:]:
        subset_predictions[subset.name] = list(model.batch_predict(subset.features))
    train_rows = len(train_frame)
    validation_rows = len(validation_frame)
    test_rows = len(test_frame)
    structured_metrics: dict[str, dict[str, float]] = {
        "train": {str(name): float(value) for name, value in subset_metrics.get("train", {}).items()},
        "validation": {
            str(name): float(value) for name, value in subset_metrics.get("validation", {}).items()
        },
        "test": {str(name): float(value) for name, value in subset_metrics.get("test", {}).items()},
    }
    summary_block = dict(structured_metrics["train"])
    if structured_metrics["validation"]:
        summary_block.setdefault("validation_mae", structured_metrics["validation"].get("mae", 0.0))
        summary_block.setdefault(
            "validation_directional_accuracy",
            structured_metrics["validation"].get("directional_accuracy", 0.0),
        )
        if "expected_pnl" in structured_metrics["validation"]:
            summary_block.setdefault(
                "validation_expected_pnl", structured_metrics["validation"]["expected_pnl"]
            )
    if structured_metrics["test"]:
        summary_block.setdefault("test_mae", structured_metrics["test"].get("mae", 0.0))
        summary_block.setdefault(
            "test_directional_accuracy",
            structured_metrics["test"].get("directional_accuracy", 0.0),
        )
        if "expected_pnl" in structured_metrics["test"]:
            summary_block.setdefault(
                "test_expected_pnl", structured_metrics["test"]["expected_pnl"]
            )
    if "expected_pnl" not in summary_block and "expected_pnl" in structured_metrics["train"]:
        summary_block["expected_pnl"] = structured_metrics["train"]["expected_pnl"]
    structured_metrics["summary"] = summary_block
    target_array = np.asarray(frame[target_col], dtype=float)
    if target_array.size == 0:
        target_scale = 1.0
    else:
        scale = float(np.std(target_array))
        if not np.isfinite(scale) or scale <= 0.0:
            mean_target = float(np.mean(target_array)) if target_array.size else 0.0
            scale = max(abs(mean_target) or 1.0, 1.0)
        target_scale = scale
    classifier = train_meta_classifier(raw_predictions, train_targets)
    validation_meta: tuple[Sequence[float], Sequence[float]] | None = None
    test_meta: tuple[Sequence[float], Sequence[float]] | None = None
    validation_subset = subset_lookup.get("validation")
    if validation_subset is not None and "validation" in subset_predictions:
        validation_meta = (subset_predictions["validation"], validation_subset.targets)
    test_subset = subset_lookup.get("test")
    if test_subset is not None and "test" in subset_predictions:
        test_meta = (subset_predictions["test"], test_subset.targets)
    meta_payload = {
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
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
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
    meta_payload["meta_labeling"] = build_meta_labeling_payload(
        classifier,
        train_data=(raw_predictions, train_targets),
        validation_data=validation_meta,
        test_data=test_meta,
    )
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
    decision_journal_entry = meta_payload.get("decision_journal_entry_id") or meta_payload.get(
        "decision_journal_entry"
    )
    decision_journal_entry_id = (
        str(decision_journal_entry) if decision_journal_entry is not None else None
    )
    feature_scalers = {
        name: (float(mean), float(stdev)) for name, (mean, stdev) in model.feature_scalers.items()
    }
    artifact = ModelArtifact(
        feature_names=tuple(model.feature_names),
        model_state=model.to_state(),
        trained_at=datetime.now(timezone.utc),
        metrics=structured_metrics,
        metadata=meta_payload,
        target_scale=target_scale,
        training_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        feature_scalers=feature_scalers,
        decision_journal_entry_id=decision_journal_entry_id,
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
            raw_metrics = getattr(artifact, "metrics", {}) or {}
            summary_source: Mapping[str, float] = {}
            if isinstance(raw_metrics, Mapping):
                summary_candidate = raw_metrics.get("summary") if isinstance(raw_metrics.get("summary"), Mapping) else None
                source = summary_candidate if isinstance(summary_candidate, Mapping) else raw_metrics
                summary_source = {
                    str(key): float(value)
                    for key, value in source.items()
                    if isinstance(value, (int, float))
                }
            update_metrics(name, summary_source)
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
