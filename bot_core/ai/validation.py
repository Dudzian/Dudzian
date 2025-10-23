"""Walidacje artefaktów modeli AI Decision Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

import jsonschema

from .models import ModelArtifact

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "model_artifact.schema.json"


@dataclass(slots=True)
class ModelArtifactValidationError(RuntimeError):
    """Błąd walidacji artefaktu modelu przeciwko schematowi JSON."""

    message: str
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:  # pragma: no cover - konstruktor RuntimeError
        RuntimeError.__init__(self, self.message)


def _load_schema(schema_path: str | Path | None = None) -> Mapping[str, object]:
    target = Path(schema_path) if schema_path is not None else _SCHEMA_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - I/O zależne od środowiska
        raise ModelArtifactValidationError(
            f"Nie można odczytać schematu ModelArtifact z {target!s}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - schemat dostarczany z repo
        raise ModelArtifactValidationError(
            f"Schemat ModelArtifact w {target!s} zawiera niepoprawny JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelArtifactValidationError(
            f"Schemat ModelArtifact w {target!s} nie jest mapowaniem JSON"
        )
    return payload


def validate_model_artifact_schema(
    artifact: ModelArtifact | Mapping[str, object], *, schema_path: str | Path | None = None
) -> None:
    """Waliduje artefakt modelu względem schematu JSON używanego w audycie."""

    if isinstance(artifact, ModelArtifact):
        payload: MutableMapping[str, object] = dict(artifact.to_dict())
    elif isinstance(artifact, Mapping):
        payload = dict(artifact)
    else:
        raise ModelArtifactValidationError(
            f"Oczekiwano ModelArtifact lub Mapping, otrzymano {type(artifact)!r}"
        )

    schema = _load_schema(schema_path)
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ModelArtifactValidationError(
            "Artefakt modelu nie spełnia schematu JSON",
            errors=(exc.message,),
        ) from exc
    except jsonschema.SchemaError as exc:  # pragma: no cover - defensywnie
        raise ModelArtifactValidationError(
            "Schemat ModelArtifact jest niepoprawny",
            errors=(str(exc),),
        ) from exc


__all__ = ["ModelArtifactValidationError", "validate_model_artifact_schema"]

