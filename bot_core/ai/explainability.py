"""Narzędzia do generowania i serializacji raportów explainability modeli AI."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

try:  # pragma: no cover - zależność opcjonalna
    import numpy as _np
except Exception:  # pragma: no cover - brak wsparcia NumPy
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - SHAP nie jest wymagany w każdym środowisku
    import shap as _shap
except Exception:  # pragma: no cover - fallback bez SHAP
    _shap = None  # type: ignore[assignment]

try:  # pragma: no cover - moduły inference mogą być niedostępne w lekkich buildach
    from .inference import DecisionModelInference
    from .models import ModelScore
except Exception:  # pragma: no cover - typy tylko dla adnotacji
    DecisionModelInference = Any  # type: ignore[assignment]
    ModelScore = Any  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


@dataclass(slots=True)
class FeatureAttribution:
    """Pojedynczy wkład cechy do decyzji modelu."""

    name: str
    contribution: float
    baseline: float | None = None
    rank: int | None = None


@dataclass(slots=True)
class ExplainabilityReport:
    """Ujednolicony raport explainability przekazywany do UI i dzienników."""

    model_name: str | None
    method: str
    attributions: tuple[FeatureAttribution, ...]
    expected_return_bps: float | None = None
    success_probability: float | None = None
    baseline_prediction: float | None = None
    summary: str | None = None
    _top_positive: tuple[str, ...] = field(default_factory=tuple, repr=False)
    _top_negative: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def as_metadata(self) -> Mapping[str, object]:
        """Zwraca strukturę gotową do serializacji (JSON-friendly)."""

        return {
            "model": self.model_name or "default",
            "method": self.method,
            "expected_return_bps": self.expected_return_bps,
            "success_probability": self.success_probability,
            "baseline_prediction": self.baseline_prediction,
            "top_features": [attr.name for attr in self.attributions[:5]],
            "top_positive": list(self._top_positive),
            "top_negative": list(self._top_negative),
            "feature_importance": {
                attr.name: attr.contribution for attr in self.attributions
            },
            "summary": self.summary,
        }


def _build_from_mapping(
    importances: Mapping[str, float],
    *,
    method: str,
    model_name: str | None,
    expected_return: float | None,
    success_probability: float | None,
    baseline: float | None,
    top_limit: int = 5,
) -> ExplainabilityReport:
    ranked: list[FeatureAttribution] = []
    for index, (name, value) in enumerate(
        sorted(
            (
                (str(key), float(value))
                for key, value in importances.items()
                if _safe_float(value) is not None
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    ):
        ranked.append(
            FeatureAttribution(
                name=name,
                contribution=value,
                baseline=baseline,
                rank=index + 1,
            )
        )

    positives = tuple(
        attr.name for attr in ranked if attr.contribution > 0
    )[:top_limit]
    negatives = tuple(
        attr.name for attr in ranked if attr.contribution < 0
    )[:top_limit]

    summary_parts: list[str] = []
    if ranked:
        summary_parts.append(
            f"Top cecha: {ranked[0].name} ({ranked[0].contribution:+.4f})"
        )
    if positives:
        summary_parts.append(
            "Pozytywne: " + ", ".join(positives)
        )
    if negatives:
        summary_parts.append(
            "Negatywne: " + ", ".join(negatives)
        )

    return ExplainabilityReport(
        model_name=model_name,
        method=method,
        attributions=tuple(ranked),
        expected_return_bps=expected_return,
        success_probability=success_probability,
        baseline_prediction=baseline,
        summary="; ".join(summary_parts) if summary_parts else None,
        _top_positive=positives,
        _top_negative=negatives,
    )


def _try_shap_importances(
    inference: DecisionModelInference,
    features: Mapping[str, float],
) -> Mapping[str, float] | None:
    if _shap is None or _np is None:
        return None
    model = getattr(inference, "_model", None)
    if model is None:
        return None
    try:
        feature_names = list(features.keys())
        vector = _np.array([list(features.values())], dtype=float)
        background = _np.repeat(vector, 10, axis=0)
        explainer = _shap.Explainer(model.predict, background)
        explanation = explainer(vector)
    except Exception:  # pragma: no cover - SHAP bywa niestabilny
        _LOGGER.debug("SHAP explainability fallback", exc_info=True)
        return None

    shap_values = getattr(explanation, "values", None)
    if shap_values is None:
        return None

    try:
        contributions = shap_values[0]
    except Exception:  # pragma: no cover - różne formaty SHAP
        return None

    if len(contributions) != len(feature_names):
        return None

    return {
        feature_names[index]: float(contribution)
        for index, contribution in enumerate(contributions)
    }


def _fallback_importances(
    inference: DecisionModelInference,
    features: Mapping[str, float],
) -> Mapping[str, float] | None:
    explain = getattr(inference, "explain", None)
    if callable(explain):
        try:
            result = explain(features)
        except Exception:
            _LOGGER.debug("DecisionModelInference.explain() failed", exc_info=True)
            return None
        if isinstance(result, Mapping):
            return {str(k): float(v) for k, v in result.items() if _safe_float(v) is not None}
    return None


def build_explainability_report(
    inference: DecisionModelInference | None,
    features: Mapping[str, float] | None,
    *,
    model_name: str | None = None,
    score: ModelScore | None = None,
    baseline_prediction: float | None = None,
    top_limit: int = 5,
) -> ExplainabilityReport | None:
    """Buduje raport explainability dla przekazanych cech.

    Funkcja próbuje najpierw wykorzystać SHAP (jeśli dostępny), a w razie
    problemów fallbackuje do metody perturbacyjnej dostępnej w inference.
    """

    if inference is None or not isinstance(features, Mapping) or not features:
        return None

    importances = _try_shap_importances(inference, features)
    method = "shap" if importances else "perturbation"
    if not importances:
        importances = _fallback_importances(inference, features)
        if not importances:
            return None

    expected_return = getattr(score, "expected_return_bps", None) if score else None
    success_probability = getattr(score, "success_probability", None) if score else None
    return _build_from_mapping(
        importances,
        method=method,
        model_name=model_name,
        expected_return=expected_return,
        success_probability=success_probability,
        baseline=baseline_prediction,
        top_limit=top_limit,
    )


def serialize_explainability(report: ExplainabilityReport) -> str:
    """Zwraca raport w formacie JSON (do przechowywania w dzienniku)."""

    return json.dumps(report.as_metadata(), ensure_ascii=False, sort_keys=True)


def flatten_explainability(
    report: ExplainabilityReport | Mapping[str, Any] | str | None,
    *,
    prefix: str = "ai_explainability",
) -> Mapping[str, str]:
    """Spłaszcza raport explainability do prostych stringów dla dzienników/metryk."""

    if report is None:
        return {}
    if isinstance(report, ExplainabilityReport):
        metadata = report.as_metadata()
    elif isinstance(report, str):
        try:
            metadata = json.loads(report)
        except json.JSONDecodeError:
            return {prefix: report}
    elif isinstance(report, Mapping):
        metadata = dict(report)
    else:
        return {}

    flattened: dict[str, str] = {}
    for key, value in metadata.items():
        name = f"{prefix}_{key}" if key else prefix
        if isinstance(value, (list, tuple, set)):
            flattened[name] = ",".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            try:
                flattened[name] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                flattened[name] = str(value)
        elif value is None:
            continue
        else:
            flattened[name] = str(value)
    if prefix not in flattened:
        flattened[prefix] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return flattened


def parse_explainability_payload(payload: object) -> ExplainabilityReport | None:
    """Konwertuje zapisany raport (dict/JSON) na obiekt ExplainabilityReport."""

    data: Mapping[str, Any]
    if isinstance(payload, ExplainabilityReport):
        return payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
    elif isinstance(payload, Mapping):
        data = payload
    else:
        return None

    feature_map = data.get("feature_importance")
    if not isinstance(feature_map, Mapping):
        return None

    ranked = _build_from_mapping(
        {str(k): float(v) for k, v in feature_map.items() if _safe_float(v) is not None},
        method=str(data.get("method") or "unknown"),
        model_name=str(data.get("model")) if data.get("model") is not None else None,
        expected_return=_safe_float(data.get("expected_return_bps")),
        success_probability=_safe_float(data.get("success_probability")),
        baseline=_safe_float(data.get("baseline_prediction")),
        top_limit=5,
    )
    return ranked


__all__ = [
    "ExplainabilityReport",
    "FeatureAttribution",
    "build_explainability_report",
    "serialize_explainability",
    "flatten_explainability",
    "parse_explainability_payload",
]

