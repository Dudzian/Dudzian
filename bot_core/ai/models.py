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



class _MetricsView(Mapping[str, object]):
    """Widok na metryki artefaktu zachowujący kompatybilność wsteczną."""

    def __init__(self, base: Mapping[str, Mapping[str, float]]) -> None:
        self._base = base
        summary = base.get("summary")
        keys: list[str] = []
        seen: set[str] = set()

        if summary:
            for metric in summary.keys():
                name = str(metric)
                if name not in seen:
                    keys.append(name)
                    seen.add(name)
        else:
            for block in base.values():
                if isinstance(block, Mapping):
                    for metric in block.keys():
                        name = str(metric)
                        if name not in seen:
                            keys.append(name)
                            seen.add(name)

        for key, value in base.items():
            name = str(key)
            if name in seen:
                continue
            if isinstance(value, Mapping) and not value:
                continue
            keys.append(name)
            seen.add(name)

        self._keys = tuple(keys)

    def __getitem__(self, key: str) -> object:
        summary = self._base.get("summary")
        if summary is not None and key in summary:
            return summary[key]
        if key in self._base:
            return self._base[key]
        for block_name in ("validation", "test", "train"):
            block = self._base.get(block_name)
            if isinstance(block, Mapping) and key in block:
                return block[key]
        for block in self._base.values():
            if isinstance(block, Mapping) and key in block:
                return block[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - trywialne
        if isinstance(key, str):
            summary = self._base.get("summary")
            if summary is not None and key in summary:
                return True
            for block_name in ("validation", "test", "train"):
                block = self._base.get(block_name)
                if isinstance(block, Mapping) and key in block:
                    return True
            for block in self._base.values():
                if isinstance(block, Mapping) and key in block:
                    return True
        return key in self._base

    def splits(self) -> Mapping[str, Mapping[str, float]]:
        return self._base

    def blocks(self) -> Mapping[str, Mapping[str, float]]:
        return self._base

    def summary(self) -> Mapping[str, float]:
        return self._base.get("summary", MappingProxyType({}))


class ModelMetrics(_MetricsView):
    """Ustandaryzowany widok metryk modeli eksportowany dla użytkowników."""

    def __init__(
        self,
        raw: Mapping[str, object] | None = None,
        *,
        _structured: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        if _structured is None:
            structured = self._structure_raw(raw)
        else:
            structured = _structured
        self._structured = structured
        super().__init__(structured)

    @staticmethod
    def _structure_raw(
        raw: Mapping[str, object] | None,
    ) -> Mapping[str, Mapping[str, float]]:
        required_keys = ("summary", "train", "validation", "test")

        def _coerce_block(values: Mapping[str, object]) -> dict[str, float]:
            normalized: dict[str, float] = {}
            for key, value in values.items():
                try:
                    normalized[str(key)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
            return normalized

        structured: dict[str, dict[str, float]] = {}

        if isinstance(raw, ModelMetrics):
            raw = raw.blocks()

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
                if legacy:
                    structured["summary"] = legacy

        for key in required_keys:
            structured.setdefault(key, {})

        summary = dict(structured.get("summary", {}))
        if not summary:
            for block_name in ("validation", "test", "train"):
                block = structured.get(block_name, {})
                if block:
                    summary = dict(block)
                    break

        for block_name in ("validation", "test", "train"):
            block = structured.get(block_name, {})
            if block:
                for metric, value in block.items():
                    summary.setdefault(metric, value)

        structured["summary"] = summary

        hydrated: dict[str, Mapping[str, float]] = {}
        for split, values in structured.items():
            hydrated[str(split)] = MappingProxyType(dict(values))

        for key in required_keys:
            hydrated.setdefault(key, MappingProxyType({}))

        return MappingProxyType(hydrated)

    @classmethod
    def from_raw(cls, raw: Mapping[str, object] | None) -> "ModelMetrics":
        return cls(raw)

    def blocks(self) -> Mapping[str, Mapping[str, float]]:
        return self._structured

    def splits(self) -> Mapping[str, Mapping[str, float]]:
        return self._structured

    def __eq__(self, other: object) -> bool:  # pragma: no cover - prosty operator
        if isinstance(other, Mapping):
            if not other:
                return all(
                    not isinstance(block, Mapping) or not block
                    for block in self._base.values()
                )
            return dict(self.items()) == dict(other.items())
        return dict(self.items()) == other

    def __eq__(self, other: object) -> bool:  # pragma: no cover - prosty operator
        if isinstance(other, Mapping):
            if not other:
                return all(
                    not isinstance(block, Mapping) or not block
                    for block in self._base.values()
                )
            return dict(self.items()) == dict(other.items())
        return dict(self.items()) == other

    def __eq__(self, other: object) -> bool:  # pragma: no cover - prosty operator
        if isinstance(other, Mapping):
            if not other:
                return all(
                    not isinstance(block, Mapping) or not block
                    for block in self._base.values()
                )
            return dict(self.items()) == dict(other.items())
        return dict(self.items()) == other

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Zwraca metryki jako głęboko kopiowalny słownik gotowy do serializacji."""

        payload: dict[str, dict[str, float]] = {}
        for split, values in self.blocks().items():
            if isinstance(values, Mapping):
                payload[str(split)] = {
                    str(metric): float(score) for metric, score in values.items()
                }
            else:  # pragma: no cover - defensywnie dla niestandardowych danych
                payload[str(split)] = {}

        for key in ("summary", "train", "validation", "test"):
            payload.setdefault(key, {})

        return payload

    def as_flat_dict(self, *, prefer: Sequence[str] | None = None) -> dict[str, float]:
        """Zwraca spłaszczone metryki preferując wybrane bloki.

        Funkcja zachowuje kompatybilność z historycznym API, które oczekiwało
        prostego słownika metryk. Domyślnie wybierany jest blok "summary", a
        następnie kolejne bloki w kolejności validation, test, train. Jeżeli
        ``prefer`` zostanie podane, wskazane bloki są sprawdzane w pierwszej
        kolejności (pojawienie się "summary" na liście jest opcjonalne).
        """

        if prefer is None:
            prefer_order: Sequence[str] = ("summary", "validation", "test", "train")
        else:
            prefer_order = tuple(prefer)

        visited: set[str] = set()
        for block_name in prefer_order:
            visited.add(block_name)
            values = self.blocks().get(block_name)
            if isinstance(values, Mapping) and values:
                return {str(metric): float(score) for metric, score in values.items()}

        for block_name in ("summary", "validation", "test", "train"):
            if block_name in visited:
                continue
            values = self.blocks().get(block_name)
            if isinstance(values, Mapping) and values:
                return {str(metric): float(score) for metric, score in values.items()}

        for block_name, values in self.blocks().items():
            if block_name in visited:
                continue
            if isinstance(values, Mapping) and values:
                return {str(metric): float(score) for metric, score in values.items()}

        return {}


def _normalize_metrics(raw: Mapping[str, object] | None) -> ModelMetrics:
    if isinstance(raw, ModelMetrics):
        return raw
    return ModelMetrics.from_raw(raw)


@dataclass(slots=True)
class ModelArtifact:
    """Zapisywalny artefakt modelu AI Decision Engine."""

    feature_names: Sequence[str]
    model_state: Mapping[str, object]
    trained_at: datetime
    metrics: Mapping[str, object]
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
        object.__setattr__(self, "metrics", _normalize_metrics(self.metrics))
        object.__setattr__(self, "target_scale", float(self.target_scale))
        object.__setattr__(self, "training_rows", int(self.training_rows))
        object.__setattr__(self, "validation_rows", int(self.validation_rows))
        object.__setattr__(self, "test_rows", int(self.test_rows))
        if self.decision_journal_entry_id is not None:
            object.__setattr__(self, "decision_journal_entry_id", str(self.decision_journal_entry_id))
        object.__setattr__(self, "backend", str(self.backend))

    def to_dict(self) -> Mapping[str, object]:
        metrics_payload = self.metrics.to_dict()

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
        metrics = _normalize_metrics(metrics_raw)

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
