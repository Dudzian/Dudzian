"""Artefakty modeli oraz struktury inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Sequence


def _parse_trained_at(value: object) -> datetime:
    """Bezpiecznie konwertuje wartość na znacznik czasu w UTC."""

    if isinstance(value, datetime):
        trained_at = value
    elif isinstance(value, (int, float)):
        trained_at = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return datetime.now(timezone.utc)
        try:
            trained_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    elif value is None:
        return datetime.now(timezone.utc)
    else:
        return datetime.now(timezone.utc)

    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)
    return trained_at.astimezone(timezone.utc)


def _mapping_or_empty(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    return {}


@dataclass(slots=True)
class ModelScore:
    """Wynik inference modelu: prognoza zwrotu i prawdopodobieństwo sukcesu."""

    expected_return_bps: float
    success_probability: float


@dataclass(slots=True)
class ModelArtifact:
    """Zapisywalny artefakt modelu AI Decision Engine."""

    feature_names: Sequence[str]
    model_state: Mapping[str, object]
    trained_at: datetime
    metrics: Mapping[str, float]
    metadata: Mapping[str, object]
    backend: str = "builtin"

    def to_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "feature_names": list(self.feature_names),
            "model_state": dict(self.model_state),
            "trained_at": self.trained_at.replace(tzinfo=timezone.utc).isoformat(),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "backend": self.backend,
        }
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ModelArtifact":
        trained_at = _parse_trained_at(raw.get("trained_at"))
        feature_names = tuple(str(name) for name in raw.get("feature_names", ()))
        model_state = MappingProxy(dict(_mapping_or_empty(raw.get("model_state"))))
        metrics = MappingProxy(dict(_mapping_or_empty(raw.get("metrics"))))
        metadata = MappingProxy(dict(_mapping_or_empty(raw.get("metadata"))))
        backend = str(raw.get("backend", "builtin"))
        return cls(
            feature_names=feature_names,
            model_state=model_state,
            trained_at=trained_at,
            metrics=metrics,
            metadata=metadata,
            backend=backend,
        )

    def build_model(self) -> "SupportsInference":
        from .training import SimpleGradientBoostingModel
        from .training import get_external_model_adapter

        if self.backend == "builtin":
            model = SimpleGradientBoostingModel()
            model.load_state(self.model_state)
            if not model.feature_names:
                model.feature_names = list(self.feature_names)
            if not getattr(model, "feature_scalers", None):
                scalers_raw = self.metadata.get("feature_scalers")
                if isinstance(scalers_raw, Mapping):
                    scalers: dict[str, tuple[float, float]] = {}
                    for name, payload in scalers_raw.items():
                        if not isinstance(payload, Mapping):
                            continue
                        mean = float(payload.get("mean", 0.0))
                        stdev = float(payload.get("stdev", 0.0))
                        scalers[str(name)] = (mean, stdev)
                    if scalers:
                        model.feature_scalers = scalers
            return model

        adapter = get_external_model_adapter(self.backend)
        return adapter.load(self.model_state, self.feature_names, self.metadata)


class MappingProxy(dict):
    """Lekka kopia tylko do odczytu."""

    def __init__(self, raw: Mapping[str, object]) -> None:
        super().__init__({str(key): value for key, value in raw.items()})

    def __hash__(self) -> int:  # pragma: no cover - słownik niemutowalny nie wymaga haszowania
        return id(self)


__all__ = ["ModelArtifact", "ModelScore"]
