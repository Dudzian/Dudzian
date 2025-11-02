"""Utilities for meta-label generation and lightweight classifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Sequence

import numpy as np


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 50.0), -50.0)
    result = 1.0 / (1.0 + math.exp(-clipped))
    return float(max(0.0, min(1.0, result)))


@dataclass(frozen=True, slots=True)
class MetaClassifierModel:
    """Parametry prostego klasyfikatora logistycznego dla meta-etykiet."""

    slope: float
    intercept: float

    def predict_probability(self, prediction: float) -> float:
        """Zwróć prawdopodobieństwo sukcesu dla prognozy bazowego modelu."""

        score = self.slope * float(prediction) + self.intercept
        return _sigmoid(score)

    def to_metadata(self) -> Mapping[str, float]:
        return {
            "type": "logistic",
            "slope": float(self.slope),
            "intercept": float(self.intercept),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, object] | None) -> "MetaClassifierModel | None":
        if not isinstance(payload, Mapping):
            return None
        model_type = str(payload.get("type", "")).strip().lower()
        if model_type and model_type not in {"logistic", "logit"}:
            return None
        try:
            slope = float(payload.get("slope", 0.0))
            intercept = float(payload.get("intercept", 0.0))
        except (TypeError, ValueError):
            return None
        if math.isnan(slope) or math.isnan(intercept):
            return None
        return cls(slope=slope, intercept=intercept)


def generate_meta_labels(predictions: Sequence[float], targets: Sequence[float]) -> np.ndarray:
    """Return binary meta labels (1 – poprawny kierunek, 0 – w przeciwnym razie)."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return np.asarray([], dtype=float)
    pred = np.asarray(predictions, dtype=float)
    targ = np.asarray(targets, dtype=float)
    sign_pred = np.sign(pred)
    sign_targ = np.sign(targ)
    hits = (sign_pred == sign_targ) & (sign_pred != 0.0)
    return hits.astype(float)


def _logistic_grad_descent(features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    weight0 = 0.0
    weight1 = 0.0
    learning_rate = 0.1
    l2 = 1e-4
    for _ in range(400):
        scores = np.clip(weight0 + weight1 * features, -50.0, 50.0)
        probs = 1.0 / (1.0 + np.exp(-scores))
        error = probs - labels
        grad0 = float(np.mean(error) + l2 * weight0)
        grad1 = float(np.mean(error * features) + l2 * weight1)
        if abs(grad0) < 1e-6 and abs(grad1) < 1e-6:
            break
        weight0 -= learning_rate * grad0
        weight1 -= learning_rate * grad1
    return float(weight0), float(weight1)


def train_meta_classifier(
    predictions: Sequence[float], targets: Sequence[float]
) -> MetaClassifierModel | None:
    """Dopasuj prosty klasyfikator logistyczny do meta-etykiet."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if len(predictions) < 2:
        return None
    labels = generate_meta_labels(predictions, targets)
    if labels.size == 0:
        return None
    positives = float(labels.sum())
    if positives <= 0.0 or positives >= labels.size:
        return None
    raw = np.asarray(predictions, dtype=float)
    mean = float(np.mean(raw))
    std = float(np.std(raw))
    if not math.isfinite(std) or std < 1e-6:
        return None
    scaled = (raw - mean) / std
    weight0, weight1 = _logistic_grad_descent(scaled, labels)
    slope = weight1 / std
    intercept = weight0 - (mean * slope)
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        return None
    return MetaClassifierModel(slope=slope, intercept=intercept)


def summarise_meta_labeling(
    predictions: Sequence[float],
    targets: Sequence[float],
    classifier: MetaClassifierModel | None,
) -> Mapping[str, float]:
    """Oblicz podstawowe metryki jakości meta-etykiet."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    labels = generate_meta_labels(predictions, targets)
    total = float(labels.size)
    summary: MutableMapping[str, float] = {}
    summary["samples"] = total
    summary["positives"] = float(labels.sum())
    summary["hit_rate"] = float(labels.mean()) if total else 0.0
    if classifier is None or total == 0:
        return summary
    probs = np.array([classifier.predict_probability(value) for value in predictions])
    predicted = probs >= 0.5
    accuracy = float(np.mean(predicted == labels)) if total else 0.0
    brier = float(np.mean((probs - labels) ** 2)) if total else 0.0
    summary["probability_mean"] = float(np.mean(probs)) if total else 0.0
    summary["probability_std"] = float(np.std(probs)) if total else 0.0
    summary["accuracy"] = accuracy
    summary["brier_score"] = brier
    return summary


def build_meta_labeling_payload(
    classifier: MetaClassifierModel | None,
    *,
    train_data: tuple[Sequence[float], Sequence[float]],
    validation_data: tuple[Sequence[float], Sequence[float]] | None = None,
    test_data: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Mapping[str, object]:
    trained_at = datetime.now(timezone.utc).isoformat()
    subsets: MutableMapping[str, Mapping[str, float]] = {}
    subsets["train"] = summarise_meta_labeling(*train_data, classifier)
    if validation_data is not None:
        subsets["validation"] = summarise_meta_labeling(*validation_data, classifier)
    if test_data is not None:
        subsets["test"] = summarise_meta_labeling(*test_data, classifier)
    payload: MutableMapping[str, object] = {
        "trained_at": trained_at,
        "subsets": dict(subsets),
        "generator": "directional_hit_rate",
    }
    if classifier is not None:
        payload["classifier"] = classifier.to_metadata()
    return payload


def select_meta_confidence(payload: Mapping[str, object]) -> float | None:
    """Wybierz najlepszą dostępną ocenę trafności meta-etykiet."""

    subsets = payload.get("subsets")
    if not isinstance(subsets, Mapping):
        return None
    for key in ("validation", "test", "train"):
        block = subsets.get(key)
        if not isinstance(block, Mapping):
            continue
        hit_rate = block.get("hit_rate")
        try:
            value = float(hit_rate)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        return max(0.0, min(1.0, value))
    return None

