"""Modele i narzędzia do pracy z raportami jakości modeli decyzyjnych."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from bot_core.reporting.model_quality import (
    ChampionDecision,
    DEFAULT_QUALITY_DIR,
    load_latest_quality_payload,
    persist_quality_report,
    update_champion_registry,
)


def _coerce_float_map(payload: Mapping[str, object]) -> MutableMapping[str, float]:
    normalized: MutableMapping[str, float] = {}
    for key, value in payload.items():
        try:
            normalized[str(key)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return normalized


def _parse_datetime(raw: object) -> datetime | None:
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


@dataclass
class ModelQualityReport:
    """Raport jakości modelu wykorzystywany do audytu i monitoringu dryfu."""

    model_name: str
    version: str
    evaluated_at: datetime
    metrics: Mapping[str, object]
    status: str
    baseline_version: str | None = None
    delta: Mapping[str, float] = None  # type: ignore[assignment]
    validation: Mapping[str, object] | None = None
    dataset_rows: int | None = None
    trained_at: datetime | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __post_init__(self) -> None:
        if self.delta is None:
            object.__setattr__(self, "delta", {})

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

        evaluated_at = _parse_datetime(payload.get("evaluated_at")) or datetime.now(timezone.utc)
        trained_at = _parse_datetime(payload.get("trained_at"))

        validation_payload = payload.get("validation")
        validation_data = None
        if isinstance(validation_payload, Mapping):
            validation_data = {str(key): value for key, value in validation_payload.items()}

        dataset_rows = None
        dataset_rows_raw = payload.get("dataset_rows")
        if isinstance(dataset_rows_raw, (int, float)):
            dataset_rows = int(dataset_rows_raw)

        baseline_raw = payload.get("baseline_version")
        baseline_version = None
        if isinstance(baseline_raw, str) and baseline_raw.strip():
            baseline_version = baseline_raw

        return cls(
            model_name=str(payload.get("model_name", "")),
            version=str(payload.get("version", "")),
            evaluated_at=evaluated_at,
            metrics=_coerce_float_map(metrics_raw),
            status=str(payload.get("status", "ok")),
            baseline_version=baseline_version,
            delta=_coerce_float_map(delta_raw),
            validation=validation_data,
            dataset_rows=dataset_rows,
            trained_at=trained_at,
        )


def record_model_quality_report(
    report: ModelQualityReport,
    *,
    history_root: Path | str | None = None,
) -> ChampionDecision:
    """Zapisuje raport jakości modelu do archiwum i aktualizuje rejestr champion."""

    base_dir = Path(history_root) if history_root is not None else DEFAULT_QUALITY_DIR
    report_path = persist_quality_report(
        report.to_dict(),
        model_name=report.model_name,
        version=report.version,
        evaluated_at=report.evaluated_at,
        base_dir=base_dir,
    )
    return update_champion_registry(
        report.to_dict(),
        model_name=report.model_name,
        base_dir=base_dir,
        report_path=report_path,
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


__all__ = [
    "ChampionDecision",
    "ModelQualityReport",
    "load_latest_quality_report",
    "record_model_quality_report",
]
