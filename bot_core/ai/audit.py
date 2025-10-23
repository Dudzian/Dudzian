"""Helpery audytu modeli AI zapisujące raporty walidacji walk-forward i jakości danych."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - tylko dla typowania
    from .feature_engineering import FeatureDataset
    from .models import ModelArtifact
    from .scheduler import WalkForwardResult, WalkForwardValidator


DEFAULT_AUDIT_ROOT = Path("audit/ai_decision")
AUDIT_SUBDIRECTORIES: tuple[str, ...] = ("walk_forward", "data_quality", "drift")
SCHEDULER_STATE_FILENAME = "scheduler.json"


def ensure_audit_structure(audit_root: str | Path | None = None) -> Path:
    """Tworzy strukturę katalogów audytu AI i zwraca ścieżkę główną."""

    root = Path(audit_root) if audit_root is not None else DEFAULT_AUDIT_ROOT
    for name in AUDIT_SUBDIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def scheduler_state_path(audit_root: str | Path | None = None) -> Path:
    """Zwraca ścieżkę do pliku stanu harmonogramu retreningu."""

    root = ensure_audit_structure(audit_root)
    return root / SCHEDULER_STATE_FILENAME


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _serialize_windows(windows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    serialized: list[Mapping[str, Any]] = []
    for window in windows:
        payload: dict[str, Any] = {}
        for key, value in window.items():
            if isinstance(value, (int, float)):
                payload[str(key)] = float(value)
            else:
                payload[str(key)] = _json_safe(value)
        serialized.append(payload)
    return serialized


def _persist_report(
    *,
    subdirectory: str,
    payload: Mapping[str, Any],
    generated_at: datetime,
    audit_root: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    root = ensure_audit_structure(audit_root)
    target_dir = root / subdirectory
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at.astimezone(timezone.utc)
    file_name = filename or f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    target_path = target_dir / file_name
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target_path


def save_walk_forward_report(
    result: "WalkForwardResult",
    *,
    job_name: str,
    dataset: "FeatureDataset",
    validator: "WalkForwardValidator" | None = None,
    artifact: "ModelArtifact" | None = None,
    audit_root: str | Path | None = None,
    generated_at: datetime | None = None,
    trainer_backend: str | None = None,
) -> Path:
    """Zapisuje raport z walidacji walk-forward do katalogu audytowego."""

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    report: dict[str, Any] = {
        "job_name": job_name,
        "generated_at": timestamp.isoformat(),
        "trainer_backend": trainer_backend or "builtin",
        "dataset": {
            "rows": len(dataset.vectors),
            "metadata": _json_safe(dataset.metadata),
        },
        "walk_forward": {
            "average_mae": float(result.average_mae),
            "average_directional_accuracy": float(result.average_directional_accuracy),
            "windows": _serialize_windows(result.windows),
        },
    }

    if validator is not None:
        report["walk_forward"].update(
            {
                "train_window": getattr(validator, "train_window", None),
                "test_window": getattr(validator, "test_window", None),
                "step": getattr(validator, "step", None),
            }
        )

    if artifact is not None:
        report["model_artifact"] = {
            "trained_at": artifact.trained_at.astimezone(timezone.utc).isoformat(),
            "metrics": _json_safe(artifact.metrics),
            "metadata": _json_safe(artifact.metadata),
        }

    return _persist_report(
        subdirectory="walk_forward",
        payload=report,
        generated_at=timestamp,
        audit_root=audit_root,
    )


def save_data_quality_report(
    issues: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    job_name: str | None = None,
    dataset: "FeatureDataset" | None = None,
    audit_root: str | Path | None = None,
    generated_at: datetime | None = None,
    source: str | None = None,
    summary: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> Path:
    """Zapisuje raport jakości danych z monitoringu cech lub źródeł surowych."""

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    normalized_issues: Sequence[Mapping[str, Any]]
    if isinstance(issues, Mapping):
        normalized_issues = [issues]
    else:
        normalized_issues = list(issues)

    payload: MutableMapping[str, Any] = {
        "generated_at": timestamp.isoformat(),
        "issues": [_json_safe(item) for item in normalized_issues],
    }
    if job_name is not None:
        payload["job_name"] = job_name
    if source is not None:
        payload["source"] = source
    if tags is not None:
        payload["tags"] = list(tags)
    if summary is not None:
        payload["summary"] = _json_safe(summary)
    if dataset is not None:
        payload["dataset"] = {
            "rows": len(dataset.vectors),
            "metadata": _json_safe(dataset.metadata),
        }

    return _persist_report(
        subdirectory="data_quality",
        payload=payload,
        generated_at=timestamp,
        audit_root=audit_root,
    )


def save_drift_report(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    job_name: str | None = None,
    dataset: "FeatureDataset" | None = None,
    audit_root: str | Path | None = None,
    generated_at: datetime | None = None,
    baseline_window: Mapping[str, Any] | None = None,
    production_window: Mapping[str, Any] | None = None,
    detector: str | None = None,
    threshold: float | None = None,
) -> Path:
    """Zapisuje raport dryfu danych/cech wykryty podczas monitoringu produkcyjnego."""

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload: MutableMapping[str, Any] = {
        "generated_at": timestamp.isoformat(),
        "metrics": {str(name): _json_safe(values) for name, values in metrics.items()},
    }
    if job_name is not None:
        payload["job_name"] = job_name
    if dataset is not None:
        payload["dataset"] = {
            "rows": len(dataset.vectors),
            "metadata": _json_safe(dataset.metadata),
        }
    if baseline_window is not None:
        payload["baseline_window"] = _json_safe(baseline_window)
    if production_window is not None:
        payload["production_window"] = _json_safe(production_window)
    if detector is not None:
        payload["detector"] = detector
    if threshold is not None:
        payload["threshold"] = float(threshold)

    return _persist_report(
        subdirectory="drift",
        payload=payload,
        generated_at=timestamp,
        audit_root=audit_root,
    )


def save_scheduler_state(
    state: Mapping[str, Any], *, audit_root: str | Path | None = None
) -> Path:
    """Zapisuje stan harmonogramu retreningu do ``scheduler.json``."""

    target_path = scheduler_state_path(audit_root)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump({str(k): _json_safe(v) for k, v in state.items()}, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target_path


def load_scheduler_state(
    *, audit_root: str | Path | None = None
) -> Mapping[str, Any] | None:
    """Ładuje zapisany stan harmonogramu retreningu."""

    target_path = scheduler_state_path(audit_root)
    if not target_path.exists():
        return None
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("Stan harmonogramu w scheduler.json musi być mapowaniem JSON")

def _iter_json_reports(directory: Path) -> list[Path]:
    files: list[Path] = []
    if not directory.exists():
        return files
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            files.append(candidate)
    return files


def list_audit_reports(
    subdirectory: str,
    *,
    audit_root: str | Path | None = None,
    limit: int | None = None,
    newest_first: bool = True,
) -> list[Path]:
    """Zwraca posortowaną listę ścieżek raportów audytu JSON."""

    root = ensure_audit_structure(audit_root)
    target_dir = root / subdirectory
    files = _iter_json_reports(target_dir)
    files.sort(key=lambda path: path.name, reverse=newest_first)
    if limit is not None and limit >= 0:
        return files[:limit]
    return files


def load_audit_report(path: str | Path) -> Mapping[str, Any]:
    """Ładuje wskazany raport audytowy JSON i zwraca jego strukturę."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("Raport audytowy musi być mapowaniem JSON")


def load_latest_walk_forward_report(
    *, audit_root: str | Path | None = None
) -> Mapping[str, Any] | None:
    """Ładuje najnowszy raport walk-forward lub zwraca ``None``."""

    paths = list_audit_reports("walk_forward", audit_root=audit_root, limit=1)
    if not paths:
        return None
    return load_audit_report(paths[0])


def load_latest_data_quality_report(
    *, audit_root: str | Path | None = None
) -> Mapping[str, Any] | None:
    """Ładuje najnowszy raport jakości danych."""

    paths = list_audit_reports("data_quality", audit_root=audit_root, limit=1)
    if not paths:
        return None
    return load_audit_report(paths[0])


def load_latest_drift_report(
    *, audit_root: str | Path | None = None
) -> Mapping[str, Any] | None:
    """Ładuje najnowszy raport dryfu danych."""

    paths = list_audit_reports("drift", audit_root=audit_root, limit=1)
    if not paths:
        return None
    return load_audit_report(paths[0])


__all__ = [
    "AUDIT_SUBDIRECTORIES",
    "DEFAULT_AUDIT_ROOT",
    "ensure_audit_structure",
    "load_scheduler_state",
    "save_scheduler_state",
    "save_walk_forward_report",
    "save_data_quality_report",
    "save_drift_report",
    "scheduler_state_path",
    "list_audit_reports",
    "load_audit_report",
    "load_latest_walk_forward_report",
    "load_latest_data_quality_report",
    "load_latest_drift_report",
]
