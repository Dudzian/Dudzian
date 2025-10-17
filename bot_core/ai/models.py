"""Artefakty modeli oraz struktury inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Sequence


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

    def to_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "feature_names": list(self.feature_names),
            "model_state": dict(self.model_state),
            "trained_at": self.trained_at.replace(tzinfo=timezone.utc).isoformat(),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ModelArtifact":
        trained_raw = str(raw.get("trained_at"))
        trained_at = datetime.fromisoformat(trained_raw.replace("Z", "+00:00"))
        feature_names = tuple(str(name) for name in raw.get("feature_names", ()))
        model_state = MappingProxy(dict(raw.get("model_state", {})))
        metrics = MappingProxy(dict(raw.get("metrics", {})))
        metadata = MappingProxy(dict(raw.get("metadata", {})))
        return cls(
            feature_names=feature_names,
            model_state=model_state,
            trained_at=trained_at,
            metrics=metrics,
            metadata=metadata,
        )

    def build_model(self) -> "SimpleGradientBoostingModel":
        from .training import SimpleGradientBoostingModel

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


class MappingProxy(dict):
    """Lekka kopia tylko do odczytu."""

    def __init__(self, raw: Mapping[str, object]) -> None:
        super().__init__({str(key): value for key, value in raw.items()})

    def __hash__(self) -> int:  # pragma: no cover - słownik niemutowalny nie wymaga haszowania
        return id(self)


__all__ = ["ModelArtifact", "ModelScore"]
