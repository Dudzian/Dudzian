"""Artefakty modeli oraz struktury inference."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from bot_core.security.signing import build_hmac_signature, validate_hmac_signature

if TYPE_CHECKING:
    from .training import SupportsInference


logger = logging.getLogger(__name__)


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


@dataclass(slots=True, frozen=True)
class ModelArtifactBundle:
    """Opis zapisanych plików artefaktu modelu AI."""

    artifact: ModelArtifact
    artifact_path: Path
    metadata_path: Path
    checksums_path: Path
    signature_path: Path | None
    metadata_payload: Mapping[str, object]
    checksums: Mapping[str, str]


class ModelArtifactIntegrityError(RuntimeError):
    """Wyjątek sygnalizujący naruszenie integralności pakietu modelu."""


def _read_checksums(path: Path) -> Mapping[str, str]:
    if not path.exists():
        raise ModelArtifactIntegrityError(f"Brak pliku sum kontrolnych: {path}")
    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                digest, filename = line.split(None, 1)
            except ValueError as exc:  # pragma: no cover - ścieżki diagnostyczne
                raise ModelArtifactIntegrityError(
                    f"Niepoprawny wpis w checksums.sha256: {line!r}"
                ) from exc
            checksums[filename.strip()] = digest.strip()
    return MappingProxyType(checksums)


def _verify_checksums(base_dir: Path, checksums: Mapping[str, str]) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    for filename, expected_digest in checksums.items():
        target = base_dir / filename
        if not target.exists():
            missing.append(filename)
            continue
        actual_digest = _hash_file(target)
        if actual_digest != expected_digest:
            mismatched.append(filename)
    if missing:
        raise ModelArtifactIntegrityError(
            "Brak plików pakietu modeli: " + ", ".join(sorted(missing))
        )
    if mismatched:
        raise ModelArtifactIntegrityError(
            "Niezgodne sumy kontrolne plików: " + ", ".join(sorted(mismatched))
        )


def load_model_artifact_bundle(
    bundle_dir: str | Path,
    *,
    expected_artifact: str | None = None,
    signing_key: bytes | None = None,
    signing_keys: Mapping[str, bytes] | None = None,
) -> ModelArtifactBundle:
    """Ładuje pakiet artefaktu modelu i weryfikuje integralność plików."""

    base_dir = Path(bundle_dir).expanduser().resolve()
    if not base_dir.exists():
        raise ModelArtifactIntegrityError(f"Katalog pakietu modeli nie istnieje: {base_dir}")

    checksums_path = base_dir / "checksums.sha256"
    checksums = _read_checksums(checksums_path)
    _verify_checksums(base_dir, checksums)

    artifact_path: Path | None = None
    metadata_path: Path | None = None
    signature_path: Path | None = None

    for candidate in base_dir.glob("*.json"):
        if candidate.name.endswith(".metadata.json"):
            metadata_path = candidate
        else:
            artifact_path = candidate

    if expected_artifact is not None:
        artifact_candidate = base_dir / expected_artifact
        if artifact_candidate.exists():
            artifact_path = artifact_candidate

    if artifact_path is None or not artifact_path.exists():
        raise ModelArtifactIntegrityError("Brak pliku artefaktu modelu w pakiecie")
    if metadata_path is None or not metadata_path.exists():
        raise ModelArtifactIntegrityError("Brak pliku metadanych artefaktu modelu")

    possible_signatures = list(base_dir.glob("*.sig"))
    if possible_signatures:
        signature_path = possible_signatures[0]

    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    artifact = ModelArtifact.from_dict(artifact_payload)

    version = metadata_payload.get("model_version") or artifact.metadata.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ModelArtifactIntegrityError(
            f"Artefakt {artifact_path.name} nie zawiera metadanej 'model_version'"
        )

    if signature_path is not None:
        signature_doc = json.loads(signature_path.read_text(encoding="utf-8"))
        signature_section = signature_doc.get("signature")
        key_identifier: str | None = None
        if isinstance(signature_section, Mapping):
            key_identifier_obj = signature_section.get("key_id")
            if key_identifier_obj is not None:
                key_identifier = str(key_identifier_obj)

        key_material: bytes | None = signing_key
        resolved_key_id: str | None = None

        if signing_keys:
            if key_identifier is None:
                if len(signing_keys) == 1:
                    resolved_key_id, key_material = next(iter(signing_keys.items()))
                else:
                    logger.error(
                        "Podpis pakietu modeli %s nie zawiera identyfikatora klucza, "
                        "a dostępnych jest %d zaufanych kluczy",
                        artifact_path.name,
                        len(signing_keys),
                    )
                    raise ModelArtifactIntegrityError(
                        "Podpis pakietu modeli nie zawiera identyfikatora klucza"
                    )
            else:
                resolved_key_id = key_identifier
                key_material = signing_keys.get(key_identifier)
                if key_material is None:
                    logger.error(
                        "Brak zaufanego klucza HMAC %s dla pakietu modeli %s",
                        key_identifier,
                        artifact_path.name,
                    )
                    raise ModelArtifactIntegrityError(
                        f"Brak zaufanego klucza HMAC {key_identifier!r} do weryfikacji podpisu"
                    )
        elif key_identifier is not None and signing_key is None:
            logger.error(
                "Podpis pakietu modeli %s wymaga klucza %s, ale nie dostarczono magazynu kluczy",
                artifact_path.name,
                key_identifier,
            )
            raise ModelArtifactIntegrityError(
                "Brak zaufanego klucza HMAC do weryfikacji podpisu pakietu modeli"
            )

        if key_material is None and signature_section is not None:
            logger.error(
                "Nie można zweryfikować podpisu pakietu modeli %s – brak klucza HMAC",
                artifact_path.name,
            )
            raise ModelArtifactIntegrityError(
                "Brak klucza HMAC do weryfikacji podpisu pakietu modeli"
            )

        if key_material is not None:
            errors = validate_hmac_signature(
                artifact_payload,
                signature_doc,
                key=key_material,
            )
            if errors:
                logger.error(
                    "Niepoprawny podpis pakietu modeli %s (klucz: %s): %s",
                    artifact_path.name,
                    resolved_key_id or key_identifier or "<domyślny>",
                    "; ".join(errors),
                )
                raise ModelArtifactIntegrityError(
                    "Niepoprawny podpis HMAC pakietu modeli: " + "; ".join(errors)
                )

    from .validation import validate_model_artifact_schema  # import lokalny, aby uniknąć cyklu

    validate_model_artifact_schema(artifact)

    return ModelArtifactBundle(
        artifact=artifact,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        checksums_path=checksums_path,
        signature_path=signature_path,
        metadata_payload=MappingProxyType(dict(metadata_payload)),
        checksums=checksums,
    )


def _bundle_base_name(name: str | Path | None) -> str:
    if name is None:
        return "model-artifact"
    candidate = Path(str(name).strip())
    if candidate.name and candidate.name not in {".", ".."}:
        if candidate.suffix == ".json":
            return candidate.stem or "model-artifact"
        return candidate.name
    normalized = str(name).strip()
    return normalized or "model-artifact"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_metadata_summary(
    artifact: ModelArtifact,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "trained_at": artifact.trained_at.astimezone(timezone.utc).isoformat(),
        "backend": artifact.backend,
        "feature_names": list(artifact.feature_names),
        "rows": {
            "train": artifact.training_rows,
            "validation": artifact.validation_rows,
            "test": artifact.test_rows,
        },
        "target_scale": artifact.target_scale,
        "feature_scalers": {
            name: {"mean": mean, "stdev": stdev}
            for name, (mean, stdev) in artifact.feature_scalers.items()
        },
        "metrics": artifact.metrics.to_dict(),
    }
    if artifact.decision_journal_entry_id is not None:
        summary["decision_journal_entry_id"] = artifact.decision_journal_entry_id
    metadata_block = dict(artifact.metadata)
    if metadata_block:
        summary["source_metadata"] = metadata_block
    if overrides:
        summary.update({str(key): value for key, value in overrides.items()})
    return summary


def generate_model_artifact_bundle(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    name: str | Path | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
    metadata_overrides: Mapping[str, object] | None = None,
) -> ModelArtifactBundle:
    """Zapisuje artefakt modelu wraz z metadanymi, checksumami i podpisem HMAC."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = _bundle_base_name(name)
    artifact_path = output_path / f"{base_name}.json"
    metadata_path = output_path / f"{base_name}.metadata.json"
    checksums_path = output_path / "checksums.sha256"
    signature_path: Path | None = output_path / f"{base_name}.sig" if signing_key else None

    artifact_payload = artifact.to_dict()
    artifact_path.write_text(
        json.dumps(artifact_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata_payload_dict = _build_metadata_summary(artifact, metadata_overrides)
    metadata_path.write_text(
        json.dumps(metadata_payload_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    written_signature_path: Path | None = None
    if signing_key is not None:
        signature_payload = {
            "target": artifact_path.name,
            "signed_at": datetime.now(timezone.utc).isoformat(),
            "signature": build_hmac_signature(
                artifact_payload,
                key=signing_key,
                key_id=signing_key_id,
            ),
        }
        signature_path = signature_path or output_path / f"{base_name}.sig"
        signature_path.write_text(
            json.dumps(signature_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written_signature_path = signature_path

    checksum_entries: list[tuple[str, str]] = [
        (artifact_path.name, _hash_file(artifact_path)),
        (metadata_path.name, _hash_file(metadata_path)),
    ]
    if written_signature_path is not None:
        checksum_entries.append(
            (written_signature_path.name, _hash_file(written_signature_path))
        )

    with checksums_path.open("w", encoding="utf-8") as handle:
        for filename, digest in checksum_entries:
            handle.write(f"{digest}  {filename}\n")

    checksums_payload = MappingProxyType(dict(checksum_entries))

    return ModelArtifactBundle(
        artifact=artifact,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        checksums_path=checksums_path,
        signature_path=written_signature_path,
        metadata_payload=MappingProxyType(dict(metadata_payload_dict)),
        checksums=checksums_payload,
    )


__all__ = [
    "ModelArtifact",
    "ModelArtifactBundle",
    "ModelArtifactIntegrityError",
    "ModelMetrics",
    "ModelScore",
    "generate_model_artifact_bundle",
    "load_model_artifact_bundle",
]
