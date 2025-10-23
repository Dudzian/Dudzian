"""Artefakty modeli oraz struktury inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Iterator, Mapping, MutableMapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .training import SupportsInference


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


def _normalize_feature_scalers(
    raw: Mapping[str, object] | None,
) -> Mapping[str, tuple[float, float]]:
    if not isinstance(raw, Mapping):
        return MappingProxyType({})
    normalized: dict[str, tuple[float, float]] = {}
    for name, payload in raw.items():
        if isinstance(payload, Mapping):
            mean = float(payload.get("mean", 0.0))
            stdev = float(payload.get("stdev", 0.0))
            normalized[str(name)] = (mean, stdev)
        else:
            try:
                mean, stdev = payload  # type: ignore[misc]
            except Exception:  # pragma: no cover - defensywnie dla nietypowych danych
                continue
            normalized[str(name)] = (float(mean), float(stdev))
    return MappingProxyType(normalized)



class ModelMetrics(Mapping[str, object]):
    """Niezmienna struktura udostępniająca metryki modelu."""

    def __init__(self, blocks: Mapping[str, Mapping[str, float]]) -> None:
        self._blocks = MappingProxyType(dict(blocks))
        summary = self._blocks.get("summary")
        if summary is None:
            summary = MappingProxyType({})
        self._summary = summary

    def __getitem__(self, key: str) -> object:
        if key in self._blocks:
            return self._blocks[key]
        if key in self._summary:
            return self._summary[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yielded: set[str] = set()
        for key in self._blocks.keys():
            yielded.add(key)
            yield key
        for key in self._summary.keys():
            if key in yielded:
                continue
            yielded.add(key)
            yield key

    def __len__(self) -> int:
        keys = set(self._blocks.keys())
        keys.update(self._summary.keys())
        return len(keys)

    def blocks(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca pierwotne bloki metryk (summary/train/validation/test)."""

        return self._blocks

    def summary(self) -> Mapping[str, float]:
        """Zwraca blok podsumowania metryk."""

        return self._summary


def _normalize_metrics(raw: Mapping[str, object] | None) -> ModelMetrics:
    required_keys = ("summary", "train", "validation", "test")

    def _coerce_block(values: Mapping[str, object]) -> Mapping[str, float]:
        normalized: dict[str, float] = {}
        for key, value in values.items():
            try:
                normalized[str(key)] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return MappingProxyType(normalized)

    structured: dict[str, Mapping[str, float]] = {}

    if isinstance(raw, Mapping) and raw:
        if all(isinstance(value, Mapping) for value in raw.values()):
            for split, payload in raw.items():
                if isinstance(payload, Mapping):
                    structured[str(split)] = _coerce_block(payload)
        else:
            legacy: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    legacy[str(key)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
            structured["summary"] = MappingProxyType(legacy)

    for key in required_keys:
        if key not in structured:
            structured[key] = MappingProxyType({})

    return ModelMetrics(structured)


@dataclass(slots=True)
class ModelArtifact:
    """Zapisywalny artefakt modelu AI Decision Engine."""

    feature_names: Sequence[str]
    model_state: Mapping[str, object]
    trained_at: datetime
    metrics: ModelMetrics
    metadata: Mapping[str, object]
    target_scale: float
    training_rows: int
    validation_rows: int
    test_rows: int
    feature_scalers: Mapping[str, tuple[float, float]]
    decision_journal_entry_id: str | None = None
    backend: str = "builtin"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_names",
            tuple(str(name) for name in self.feature_names),
        )
        object.__setattr__(self, "trained_at", _parse_trained_at(self.trained_at))
        object.__setattr__(self, "model_state", MappingProxyType(dict(self.model_state)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "feature_scalers",
            MappingProxyType(
                {
                    str(name): (float(pair[0]), float(pair[1]))
                    for name, pair in self.feature_scalers.items()
                }
            ),
        )
        metrics_source = self.metrics if isinstance(self.metrics, Mapping) else None
        object.__setattr__(self, "metrics", _normalize_metrics(metrics_source))
        object.__setattr__(self, "target_scale", float(self.target_scale))
        object.__setattr__(self, "training_rows", int(self.training_rows))
        object.__setattr__(self, "validation_rows", int(self.validation_rows))
        object.__setattr__(self, "test_rows", int(self.test_rows))
        if self.decision_journal_entry_id is not None:
            object.__setattr__(self, "decision_journal_entry_id", str(self.decision_journal_entry_id))
        object.__setattr__(self, "backend", str(self.backend))

    def to_dict(self) -> Mapping[str, object]:
        metrics_payload: dict[str, dict[str, float]] = {}
        for split, values in self.metrics.blocks().items():
            if isinstance(values, Mapping):
                metrics_payload[str(split)] = dict(values)
        for required in ("summary", "train", "validation", "test"):
            metrics_payload.setdefault(required, {})

        payload: MutableMapping[str, object] = {
            "feature_names": list(self.feature_names),
            "model_state": dict(self.model_state),
            "trained_at": self.trained_at.replace(tzinfo=timezone.utc).isoformat(),
            "metrics": dict(metrics_payload),
            "metadata": dict(self.metadata),
            "target_scale": float(self.target_scale),
            "training_rows": int(self.training_rows),
            "validation_rows": int(self.validation_rows),
            "test_rows": int(self.test_rows),
            "feature_scalers": {
                name: {"mean": mean, "stdev": stdev}
                for name, (mean, stdev) in self.feature_scalers.items()
            },
            "backend": self.backend,
        }
        if self.decision_journal_entry_id:
            payload["decision_journal_entry_id"] = str(self.decision_journal_entry_id)
        return payload



    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ModelArtifact":
        trained_at = _parse_trained_at(raw.get("trained_at"))
        feature_names = tuple(str(name) for name in raw.get("feature_names", ()))
        model_state = MappingProxyType(dict(_mapping_or_empty(raw.get("model_state"))))
        metadata = MappingProxyType(dict(_mapping_or_empty(raw.get("metadata"))))
        backend = str(raw.get("backend", "builtin"))

        metrics_raw = raw.get("metrics")
        metrics = _normalize_metrics(metrics_raw if isinstance(metrics_raw, Mapping) else None)

        target_scale = float(raw.get("target_scale", metadata.get("target_scale", 1.0)))
        training_rows = int(raw.get("training_rows", metadata.get("training_rows", 0)))
        validation_rows = int(raw.get("validation_rows", metadata.get("validation_rows", 0)))
        test_rows = int(raw.get("test_rows", metadata.get("test_rows", 0)))

        scalers_raw = raw.get("feature_scalers")
        scalers = _normalize_feature_scalers(
            scalers_raw if isinstance(scalers_raw, Mapping) else metadata.get("feature_scalers")
        )

        decision_journal_entry_id = raw.get("decision_journal_entry_id")
        if decision_journal_entry_id is None:
            decision_journal_entry_id = metadata.get("decision_journal_entry")
        if decision_journal_entry_id is not None:
            decision_journal_entry_id = str(decision_journal_entry_id)

        return cls(
            feature_names=feature_names,
            model_state=model_state,
            trained_at=trained_at,
            metrics=metrics,
            metadata=metadata,
            target_scale=target_scale,
            training_rows=training_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            feature_scalers=MappingProxyType(dict(scalers)),
            decision_journal_entry_id=decision_journal_entry_id,
            backend=backend,
        )

    def build_model(self) -> SupportsInference:
        from .training import SimpleGradientBoostingModel
        from .training import get_external_model_adapter

        if self.backend == "builtin":
            model = SimpleGradientBoostingModel()
            model.load_state(self.model_state)
            if not model.feature_names:
                model.feature_names = list(self.feature_names)
            if self.feature_scalers and getattr(model, "feature_scalers", None) != self.feature_scalers:
                model.feature_scalers = dict(self.feature_scalers)
            elif not getattr(model, "feature_scalers", None):
                scalers_raw = self.metadata.get("feature_scalers")
                if isinstance(scalers_raw, Mapping):
                    scalers = _normalize_feature_scalers(scalers_raw)
                    if scalers:
                        model.feature_scalers = dict(scalers)
            return model

        if self.backend == "sequential_td":
            from .sequential import TemporalDifferencePolicy

            model = TemporalDifferencePolicy.from_state(self.feature_names, self.model_state)
            return model

        adapter = get_external_model_adapter(self.backend)
        return adapter.load(self.model_state, self.feature_names, self.metadata)


__all__ = ["ModelArtifact", "ModelMetrics", "ModelScore"]
