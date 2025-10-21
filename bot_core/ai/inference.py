"""Moduł inference Decision Engine korzystający z wytrenowanych artefaktów."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ._license import ensure_ai_signals_enabled
from .models import ModelArtifact, ModelScore


@dataclass(slots=True)
class ModelRepository:
    """Odpowiada za ładowanie i zapisywanie artefaktów modeli."""

    base_path: Path

    def load(self, artifact: str | Path | Mapping[str, object]) -> ModelArtifact:
        if isinstance(artifact, Mapping):
            return ModelArtifact.from_dict(artifact)
        path = self.base_path.joinpath(Path(artifact)) if not isinstance(artifact, Path) else artifact
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ModelArtifact.from_dict(payload)

    def save(self, artifact: ModelArtifact, name: str) -> Path:
        destination = self.base_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2)
        return destination


class DecisionModelInference:
    """Wykonuje scoring kandydatów Decision Engine."""

    def __init__(self, repository: ModelRepository) -> None:
        ensure_ai_signals_enabled("inference modeli AI")
        self._repository = repository
        self._artifact: ModelArtifact | None = None
        self._model: Any | None = None
        self._target_scale: float = 1.0
        self._feature_scalers: dict[str, tuple[float, float]] = {}
        self._calibration: tuple[float, float] | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load_weights(self, artifact: str | Path | Mapping[str, object]) -> None:
        self._artifact = self._repository.load(artifact)
        self._model = self._artifact.build_model()
        metadata = dict(self._artifact.metadata)
        self._target_scale = float(metadata.get("target_scale", 1.0))
        self._feature_scalers = self._extract_scalers(metadata)
        self._calibration = self._extract_calibration(metadata)
        if hasattr(self._model, "feature_scalers"):
            model_scalers = getattr(self._model, "feature_scalers")
            if not self._feature_scalers and isinstance(model_scalers, Mapping):
                self._feature_scalers = {
                    str(name): (float(pair[0]), float(pair[1]))
                    for name, pair in model_scalers.items()
                }
            elif self._feature_scalers:
                self._model.feature_scalers = dict(self._feature_scalers)

    def score(self, features: Mapping[str, float]) -> ModelScore:
        if self._model is None:
            raise RuntimeError("Model inference nie został załadowany")
        prepared = self._prepare_features(features)
        prediction = float(self._model.predict(prepared))
        if self._calibration is not None:
            slope, intercept = self._calibration
            prediction = prediction * slope + intercept
        probability = self._to_probability(prediction)
        return ModelScore(expected_return_bps=prediction, success_probability=probability)

    def explain(self, features: Mapping[str, float]) -> Mapping[str, float]:
        if self._model is None:
            raise RuntimeError("Model inference nie został załadowany")
        importances: MutableMapping[str, float] = {}
        prepared = self._prepare_features(features)
        baseline = float(self._model.predict(prepared))
        for name in self._model.feature_names:
            perturbed = dict(prepared)
            mean = self._feature_scalers.get(name, (0.0, 0.0))[0]
            perturbed[name] = mean
            delta = baseline - float(self._model.predict(perturbed))
            importances[name] = delta
        return dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))

    def _to_probability(self, value: float) -> float:
        scale = self._target_scale if self._target_scale > 0 else 1.0
        normalized = max(min(value / (scale * 6.0), 10.0), -10.0)
        prob = 1.0 / (1.0 + math.exp(-normalized))
        return max(0.0, min(1.0, prob))

    def _prepare_features(self, features: Mapping[str, float]) -> Mapping[str, float]:
        if self._model is None:
            return features
        prepared: MutableMapping[str, float] = {}
        provided = {str(key): value for key, value in features.items()}
        for name in getattr(self._model, "feature_names", ()):  # type: ignore[attr-defined]
            raw = provided.get(name)
            if raw is None:
                mean = self._feature_scalers.get(name, (0.0, 0.0))[0]
                prepared[name] = float(mean)
            else:
                try:
                    prepared[name] = float(raw)
                except (TypeError, ValueError):
                    prepared[name] = float(
                        self._feature_scalers.get(name, (0.0, 0.0))[0]
                    )
        for key, value in provided.items():
            if key not in prepared:
                try:
                    prepared[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return prepared

    def _extract_scalers(
        self, metadata: Mapping[str, object]
    ) -> dict[str, tuple[float, float]]:
        raw = metadata.get("feature_scalers")
        if not isinstance(raw, Mapping):
            return {}
        scalers: dict[str, tuple[float, float]] = {}
        for name, payload in raw.items():
            if not isinstance(payload, Mapping):
                continue
            mean = float(payload.get("mean", 0.0))
            stdev = float(payload.get("stdev", 0.0))
            scalers[str(name)] = (mean, stdev)
        return scalers

    def _extract_calibration(
        self, metadata: Mapping[str, object]
    ) -> tuple[float, float] | None:
        payload = metadata.get("calibration")
        if not isinstance(payload, Mapping):
            return None
        slope = float(payload.get("slope", 1.0))
        intercept = float(payload.get("intercept", 0.0))
        return slope, intercept


__all__ = ["DecisionModelInference", "ModelRepository"]
