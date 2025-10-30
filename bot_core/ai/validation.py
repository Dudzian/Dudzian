"""Walidacje artefaktów modeli AI Decision Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

import jsonschema

from .models import ModelArtifact
from bot_core.reporting.model_quality import (
    DEFAULT_QUALITY_DIR,
    load_latest_quality_payload,
    persist_quality_report,
)

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


def _coerce_float_map(payload: Mapping[str, object]) -> MutableMapping[str, float]:
    normalized: MutableMapping[str, float] = {}
    for key, value in payload.items():
        try:
            normalized[str(key)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return normalized


@dataclass(slots=True)
class ModelQualityReport:
    """Raport jakości modelu wykorzystywany do audytu i monitoringu dryfu."""

    model_name: str
    version: str
    evaluated_at: datetime
    metrics: Mapping[str, float]
    status: str
    baseline_version: str | None = None
    delta: Mapping[str, float] = field(default_factory=dict)
    validation: Mapping[str, object] | None = None
    dataset_rows: int | None = None
    trained_at: datetime | None = None

    def to_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "model_name": self.model_name,
            "version": self.version,
            "evaluated_at": self.evaluated_at.astimezone(timezone.utc).isoformat(),
            "metrics": dict(self.metrics),
            "status": self.status,
            "delta": dict(self.delta),
        }
        if self.baseline_version is not None:
            payload["baseline_version"] = self.baseline_version
        if self.validation is not None:
            payload["validation"] = dict(self.validation)
        if self.dataset_rows is not None:
            payload["dataset_rows"] = int(self.dataset_rows)
        if self.trained_at is not None:
            payload["trained_at"] = self.trained_at.astimezone(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelQualityReport":
        metrics_raw = payload.get("metrics")
        if not isinstance(metrics_raw, Mapping):
            metrics_raw = {}
        delta_raw = payload.get("delta")
        if not isinstance(delta_raw, Mapping):
            delta_raw = {}
        evaluated_raw = payload.get("evaluated_at")
        evaluated_at = datetime.now(timezone.utc)
        if isinstance(evaluated_raw, str) and evaluated_raw.strip():
            try:
                evaluated_at = datetime.fromisoformat(evaluated_raw.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except ValueError:
                evaluated_at = datetime.now(timezone.utc)
        trained_raw = payload.get("trained_at")
        trained_at: datetime | None = None
        if isinstance(trained_raw, str) and trained_raw.strip():
            try:
                trained_at = datetime.fromisoformat(trained_raw.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except ValueError:
                trained_at = None
        validation_payload = payload.get("validation")
        if isinstance(validation_payload, Mapping):
            validation_data = {str(key): value for key, value in validation_payload.items()}
        else:
            validation_data = None
        dataset_rows_raw = payload.get("dataset_rows")
        dataset_rows = None
        if isinstance(dataset_rows_raw, (int, float)):
            dataset_rows = int(dataset_rows_raw)
        return cls(
            model_name=str(payload.get("model_name", "")),
            version=str(payload.get("version", "")),
            evaluated_at=evaluated_at,
            metrics=_coerce_float_map(metrics_raw),
            status=str(payload.get("status", "ok")),
            baseline_version=str(payload["baseline_version"]).strip()
            if isinstance(payload.get("baseline_version"), str) and payload.get("baseline_version", "").strip()
            else None,
            delta=_coerce_float_map(delta_raw),
            validation=validation_data,
            dataset_rows=dataset_rows,
            trained_at=trained_at,
        )


def record_model_quality_report(
    report: ModelQualityReport,
    *,
    history_root: Path | str | None = None,
) -> Path:
    """Zapisuje raport jakości modelu do archiwum."""

    base_dir = Path(history_root) if history_root is not None else DEFAULT_QUALITY_DIR
    return persist_quality_report(
        report.to_dict(),
        model_name=report.model_name,
        version=report.version,
        evaluated_at=report.evaluated_at,
        base_dir=base_dir,
    )


def load_latest_quality_report(
    model_name: str,
    *,
    history_root: Path | str | None = None,
) -> ModelQualityReport | None:
    base_dir = Path(history_root) if history_root is not None else DEFAULT_QUALITY_DIR
    payload = load_latest_quality_payload(model_name, base_dir=base_dir)
    if payload is None:
        return None
    return ModelQualityReport.from_dict(payload)


@dataclass(slots=True)
class ModelQualityMonitor:
    """Monitor jakości modeli odpowiedzialny za wykrywanie dryfu metryk."""

    model_name: str
    history_root: Path | str = field(default_factory=lambda: DEFAULT_QUALITY_DIR)
    directional_tolerance: float = 0.03
    mae_tolerance: float = 0.15
    _baseline: ModelQualityReport | None = field(init=False, repr=False, default=None)
    _model_dir: Path | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        root = Path(self.history_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "history_root", root)
        self._model_dir = root / self.model_name
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._baseline = load_latest_quality_report(self.model_name, history_root=root)

    @property
    def baseline_report(self) -> ModelQualityReport | None:
        return self._baseline

    def should_retrain(self, production_metrics: Mapping[str, float]) -> bool:
        metrics = _coerce_float_map(production_metrics)
        if self._baseline is None:
            return True
        baseline_metrics = self._baseline.metrics
        base_directional = baseline_metrics.get("directional_accuracy")
        current_directional = metrics.get("directional_accuracy")
        if (
            base_directional is not None
            and current_directional is not None
            and current_directional + self.directional_tolerance < base_directional
        ):
            return True
        base_mae = baseline_metrics.get("mae")
        current_mae = metrics.get("mae")
        if (
            base_mae is not None
            and current_mae is not None
            and current_mae - base_mae > self.mae_tolerance
        ):
            return True
        return False

    def record_training_run(
        self,
        *,
        version: str,
        metrics: Mapping[str, float],
        trained_at: datetime,
        dataset_rows: int,
        validation: Mapping[str, object] | None = None,
    ) -> ModelQualityReport:
        normalized_metrics = _coerce_float_map(metrics)
        baseline_metrics = self._baseline.metrics if self._baseline is not None else {}
        delta: MutableMapping[str, float] = {}
        for key, value in normalized_metrics.items():
            baseline_value = baseline_metrics.get(key)
            if baseline_value is None:
                continue
            delta[key] = value - baseline_value

        status = self._evaluate_status(baseline_metrics, normalized_metrics)
        report = ModelQualityReport(
            model_name=self.model_name,
            version=version,
            evaluated_at=datetime.now(timezone.utc),
            metrics=normalized_metrics,
            status=status,
            baseline_version=self._baseline.version if self._baseline else None,
            delta=delta,
            validation=validation,
            dataset_rows=int(dataset_rows),
            trained_at=trained_at.astimezone(timezone.utc),
        )

        if status != "degraded" or self._baseline is None:
            self._baseline = report
        return report

    def _evaluate_status(
        self,
        baseline: Mapping[str, float],
        current: Mapping[str, float],
    ) -> str:
        if not baseline:
            return "improved"
        improved = False
        degraded = False

        base_directional = baseline.get("directional_accuracy")
        current_directional = current.get("directional_accuracy")
        if base_directional is not None and current_directional is not None:
            if current_directional >= base_directional + self.directional_tolerance:
                improved = True
            elif current_directional + self.directional_tolerance < base_directional:
                degraded = True

        base_mae = baseline.get("mae")
        current_mae = current.get("mae")
        if base_mae is not None and current_mae is not None:
            if base_mae - current_mae >= self.mae_tolerance:
                improved = True
            elif current_mae - base_mae > self.mae_tolerance:
                degraded = True

        if improved and not degraded:
            return "improved"
        if degraded and not improved:
            return "degraded"
        return "ok"


__all__ = [
    "ModelArtifactValidationError",
    "ModelQualityMonitor",
    "ModelQualityReport",
    "load_latest_quality_report",
    "record_model_quality_report",
    "validate_model_artifact_schema",
]

