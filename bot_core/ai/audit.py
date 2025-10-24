"""Helpery audytu modeli AI zapisujące raporty walidacji walk-forward i jakości danych."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from .data_monitoring import (
    _COMPLETED_SIGN_OFF_STATUSES,
    _SIGN_OFF_DEFAULT_NOTES,
    _SIGN_OFF_ROLES,
    _SIGN_OFF_STATUSES,
)


_DEFAULT_SIGN_OFF_ROLE_ORDER = _SIGN_OFF_ROLES

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


_DEFAULT_SIGN_OFF_NOTES = MappingProxyType(dict(_SIGN_OFF_DEFAULT_NOTES))


def _normalize_role(role: object) -> str | None:
    if isinstance(role, str):
        normalized = role.strip().lower()
        if normalized:
            return normalized
    return None


def _default_sign_off(
    *, extra_roles: Sequence[str] | None = None
) -> dict[str, dict[str, object]]:
    base_roles: list[str] = list(_SIGN_OFF_ROLES)
    extra: set[str] = set()
    for role in extra_roles or ():
        normalized = _normalize_role(role)
        if normalized and normalized not in base_roles:
            extra.add(normalized)
    ordered_roles = (*base_roles, *sorted(extra))
    sign_off: dict[str, dict[str, object]] = {}
    for role in ordered_roles:
        note = _DEFAULT_SIGN_OFF_NOTES.get(
            role, f"Awaiting {role.replace('_', ' ').title()} sign-off"
        )
        sign_off[role] = {
            "status": "pending",
            "signed_by": None,
            "timestamp": None,
            "notes": note,
        }
    return sign_off


def _normalize_sign_off(
    sign_off: Mapping[str, Mapping[str, Any]] | None
) -> Mapping[str, Mapping[str, Any]]:
    extra_roles: list[str] = []
    if isinstance(sign_off, Mapping):
        for role in sign_off.keys():
            role_key = _normalize_role(role)
            if role_key:
                extra_roles.append(role_key)
    normalized: dict[str, dict[str, Any]] = _default_sign_off(extra_roles=extra_roles)
    if not isinstance(sign_off, Mapping):
        return normalized
    for role, payload in sign_off.items():
        role_key = _normalize_role(role)
        if role_key is None:
            continue
        base = dict(normalized.get(role_key, {}))
        if isinstance(payload, Mapping):
            status_raw = payload.get("status")
            if isinstance(status_raw, str):
                status = status_raw.strip().lower()
                if status in _SIGN_OFF_STATUSES:
                    base["status"] = status
            signed_by = payload.get("signed_by")
            if signed_by is not None:
                base["signed_by"] = str(signed_by)
            notes = payload.get("notes")
            if notes is not None:
                base["notes"] = str(notes)
            timestamp = payload.get("timestamp")
            if isinstance(timestamp, str):
                base["timestamp"] = timestamp
        normalized[role_key] = base
    return normalized


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
    sign_off: Mapping[str, Mapping[str, Any]] | None = None,
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
        "sign_off": _json_safe(_normalize_sign_off(sign_off)),
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


def _load_recent_reports(
    *,
    subdir: str,
    limit: int = 20,
    audit_root: str | Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    root = ensure_audit_structure(audit_root)
    directory = root / subdir
    files = _iter_json_reports(directory)
    files.sort(key=lambda path: path.name, reverse=True)
    if limit >= 0:
        files = files[:limit]
    records: list[Mapping[str, Any]] = []
    for path in files:
        try:
            payload = load_audit_report(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, Mapping):
            record: MutableMapping[str, Any] = dict(payload)
        else:
            record = {"raw_payload": payload}
        record.setdefault("report_path", str(path))
        records.append(record)
    return tuple(records)


def load_recent_walk_forward_reports(
    *, limit: int = 20, audit_root: str | Path | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Ładuje najnowsze raporty walk-forward z audytu."""

    return _load_recent_reports(subdir="walk_forward", limit=limit, audit_root=audit_root)


def _collect_pending_sign_off(
    report: Mapping[str, Any],
    *,
    pending: MutableMapping[str, list[Mapping[str, Any]]],
    category: str,
) -> None:
    sign_off = report.get("sign_off")
    timestamp = report.get("generated_at") or report.get("timestamp")
    report_path = report.get("report_path")
    path_str = None
    if isinstance(report_path, (str, PathLike)):
        path_str = str(report_path)

    normalized_entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(sign_off, Mapping):
        for raw_role, payload in sign_off.items():
            role_key = _normalize_role(raw_role)
            if role_key:
                normalized_entries[role_key] = payload

    roles = set(_SIGN_OFF_ROLES)
    roles.update(normalized_entries.keys())

    if not normalized_entries and not isinstance(sign_off, Mapping):
        for role in roles:
            bucket = pending.setdefault(role, [])
            bucket.append(
                {
                    "category": category,
                    "status": "pending",
                    "report_path": path_str,
                    "timestamp": timestamp,
                }
            )
        return

    for role in sorted(roles):
        bucket = pending.setdefault(role, [])
        entry = normalized_entries.get(role)
        status = "pending"
        if isinstance(entry, Mapping):
            status_raw = entry.get("status")
            if isinstance(status_raw, str):
                status = status_raw.strip().lower()
        if status not in _COMPLETED_SIGN_OFF_STATUSES:
            bucket.append(
                {
                    "category": category,
                    "status": status,
                    "report_path": path_str,
                    "timestamp": timestamp,
                }
            )


def summarize_walk_forward_reports(
    reports: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """Buduje podsumowanie raportów walk-forward i brakujących podpisów."""

    normalized = [report for report in reports or () if isinstance(report, Mapping)]
    mae_values: list[float] = []
    accuracy_values: list[float] = []
    worst_mae: float | None = None
    best_mae: float | None = None
    worst_accuracy: float | None = None
    best_accuracy: float | None = None
    total_windows = 0
    pending: MutableMapping[str, list[Mapping[str, Any]]] = {
        role: [] for role in _DEFAULT_SIGN_OFF_ROLE_ORDER
    }

    summary: MutableMapping[str, Any] = {
        "total": len(normalized),
        "latest_report_path": None,
        "latest_generated_at": None,
        "average_mae": {"mean": None, "min": None, "max": None},
        "average_directional_accuracy": {"mean": None, "min": None, "max": None},
        "window_extremes": {
            "total_windows": 0,
            "worst_mae": None,
            "best_mae": None,
            "worst_directional_accuracy": None,
            "best_directional_accuracy": None,
        },
        "pending_sign_off": {role: () for role in _DEFAULT_SIGN_OFF_ROLE_ORDER},
    }

    for index, report in enumerate(normalized):
        if index == 0:
            report_path = report.get("report_path")
            if isinstance(report_path, (str, PathLike)):
                summary["latest_report_path"] = str(report_path)
            generated = report.get("generated_at")
            if isinstance(generated, str):
                summary["latest_generated_at"] = generated
        wf = report.get("walk_forward")
        if isinstance(wf, Mapping):
            average_mae = wf.get("average_mae")
            average_dir = wf.get("average_directional_accuracy")
            try:
                mae_value = float(average_mae)
            except (TypeError, ValueError):
                mae_value = None
            if mae_value is not None:
                mae_values.append(mae_value)
                worst_mae = mae_value if worst_mae is None else max(worst_mae, mae_value)
                best_mae = mae_value if best_mae is None else min(best_mae, mae_value)
            try:
                acc_value = float(average_dir)
            except (TypeError, ValueError):
                acc_value = None
            if acc_value is not None:
                accuracy_values.append(acc_value)
                worst_accuracy = (
                    acc_value if worst_accuracy is None else min(worst_accuracy, acc_value)
                )
                best_accuracy = (
                    acc_value if best_accuracy is None else max(best_accuracy, acc_value)
                )
            windows = wf.get("windows")
            if isinstance(windows, Sequence):
                for window in windows:
                    if not isinstance(window, Mapping):
                        continue
                    total_windows += 1
                    try:
                        mae = float(window.get("mae"))
                        worst_mae = mae if worst_mae is None else max(worst_mae, mae)
                        best_mae = mae if best_mae is None else min(best_mae, mae)
                    except (TypeError, ValueError):
                        pass
                    try:
                        acc = float(window.get("directional_accuracy"))
                        worst_accuracy = (
                            acc if worst_accuracy is None else min(worst_accuracy, acc)
                        )
                        best_accuracy = (
                            acc if best_accuracy is None else max(best_accuracy, acc)
                        )
                    except (TypeError, ValueError):
                        pass
        _collect_pending_sign_off(
            report,
            pending=pending,
            category=str(report.get("job_name") or "walk_forward"),
        )

    if mae_values:
        summary["average_mae"] = {
            "mean": float(sum(mae_values) / len(mae_values)),
            "min": float(min(mae_values)),
            "max": float(max(mae_values)),
        }
    if accuracy_values:
        summary["average_directional_accuracy"] = {
            "mean": float(sum(accuracy_values) / len(accuracy_values)),
            "min": float(min(accuracy_values)),
            "max": float(max(accuracy_values)),
        }
    summary["window_extremes"] = {
        "total_windows": total_windows,
        "worst_mae": float(worst_mae) if worst_mae is not None else None,
        "best_mae": float(best_mae) if best_mae is not None else None,
        "worst_directional_accuracy": (
            float(worst_accuracy) if worst_accuracy is not None else None
        ),
        "best_directional_accuracy": (
            float(best_accuracy) if best_accuracy is not None else None
        ),
    }
    ordered_roles = list(_DEFAULT_SIGN_OFF_ROLE_ORDER)
    ordered_roles.extend(
        sorted(role for role in pending.keys() if role not in _SIGN_OFF_ROLES)
    )
    summary["pending_sign_off"] = {
        role: tuple(pending.get(role, ())) for role in ordered_roles
    }
    return MappingProxyType(summary)


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
    "load_recent_walk_forward_reports",
    "summarize_walk_forward_reports",
    "save_data_quality_report",
    "save_drift_report",
    "scheduler_state_path",
    "list_audit_reports",
    "load_audit_report",
    "load_latest_walk_forward_report",
    "load_latest_data_quality_report",
    "load_latest_drift_report",
]
