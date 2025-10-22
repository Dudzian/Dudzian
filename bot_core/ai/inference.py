"""Moduł inference Decision Engine korzystający z wytrenowanych artefaktów."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Mapping, MutableMapping

from bot_core.alerts import DriftAlertPayload, emit_model_drift_alert

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


@dataclass(slots=True)
class _FeatureDriftMonitor:
    """Simple drift monitor based on moving average of feature z-scores."""

    threshold: float = 3.0
    window: int = 50
    min_observations: int = 10
    cooldown: int = 25
    backend: str = "decision_engine"
    _values: Deque[float] = field(init=False, repr=False)
    _last_alert_score: float | None = field(init=False, default=None, repr=False)
    _last_alert_time: float | None = field(init=False, default=None, repr=False)
    _model_name: str = field(init=False, default="unknown", repr=False)
    _enabled: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=max(self.window, 1))

    def configure(
        self,
        *,
        model_name: str,
        threshold: float | None = None,
        window: int | None = None,
        min_observations: int | None = None,
        cooldown: int | None = None,
        backend: str | None = None,
    ) -> None:
        self._model_name = model_name
        if threshold is not None:
            self.threshold = max(float(threshold), 0.0)
        if window is not None:
            self.window = max(int(window), 1)
        if min_observations is not None:
            self.min_observations = max(int(min_observations), 1)
        if cooldown is not None:
            self.cooldown = max(int(cooldown), 1)
        if backend is not None:
            self.backend = str(backend)
        self._values = deque(self._values, maxlen=self.window)

    def disable(self) -> None:
        self._enabled = False

    def observe(
        self,
        features: Mapping[str, float],
        scalers: Mapping[str, tuple[float, float]],
    ) -> float | None:
        if not self._enabled or not scalers:
            return None
        total = 0.0
        count = 0
        for name, (mean, stdev) in scalers.items():
            if stdev <= 0:
                continue
            value = float(features.get(name, mean))
            total += abs(value - mean) / stdev
            count += 1
        if count == 0:
            return None
        score = total / count
        self._values.append(score)
        avg = sum(self._values) / len(self._values)
        if len(self._values) >= self.min_observations and avg > self.threshold:
            now = time.monotonic()
            should_alert = False
            if self._last_alert_score is None:
                should_alert = True
            elif abs(avg - self._last_alert_score) >= 0.1:
                should_alert = True
            elif self._last_alert_time is not None and now - self._last_alert_time > self.cooldown:
                should_alert = True
            if should_alert:
                payload = DriftAlertPayload(
                    model_name=self._model_name,
                    drift_score=avg,
                    threshold=self.threshold,
                    window=len(self._values),
                    backend=self.backend,
                    extra={"recent_score": score},
                )
                emit_model_drift_alert(payload)
                self._last_alert_score = avg
                self._last_alert_time = now
        return avg


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
        self._model_label: str = "unknown"
        self._drift_monitor = _FeatureDriftMonitor()
        self._last_drift_score: float | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_label(self) -> str:
        return self._model_label

    @model_label.setter
    def model_label(self, value: str) -> None:
        self._model_label = str(value)
        self._drift_monitor.configure(model_name=self._model_label)

    @property
    def last_drift_score(self) -> float | None:
        return self._last_drift_score

    def load_weights(self, artifact: str | Path | Mapping[str, object]) -> None:
        self._artifact = self._repository.load(artifact)
        self._model = self._artifact.build_model()
        metadata = dict(self._artifact.metadata)
        self._target_scale = float(metadata.get("target_scale", 1.0))
        self._feature_scalers = self._extract_scalers(metadata)
        self._calibration = self._extract_calibration(metadata)
        drift_config = metadata.get("drift_monitor")
        if isinstance(drift_config, Mapping):
            self._drift_monitor.configure(
                model_name=getattr(self, "_model_label", "unknown"),
                threshold=drift_config.get("threshold"),
                window=drift_config.get("window"),
                min_observations=drift_config.get("min_observations"),
                cooldown=drift_config.get("cooldown"),
                backend=drift_config.get("backend"),
            )
        else:
            self._drift_monitor.configure(model_name=getattr(self, "_model_label", "unknown"))
        self._last_drift_score = None
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
        drift_score = self._drift_monitor.observe(prepared, self._feature_scalers)
        if drift_score is not None:
            self._last_drift_score = drift_score
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
