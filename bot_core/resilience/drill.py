"""Obsługa scenariuszy failover Stage6 oraz raportowania wyników ćwiczeń."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.security.signing import build_hmac_signature

from .audit import BundleAuditResult


_SUMMARY_SCHEMA = "stage6.resilience.failover_drill.summary"
_SUMMARY_SIGNATURE_SCHEMA = "stage6.resilience.failover_drill.summary.signature"
_SCHEMA_VERSION = "1.0"


def _timestamp() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} powinien być obiektem mapującym")
    return value  # type: ignore[return-value]


def _ensure_sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{context} nie może być napisem")
    if not isinstance(value, Sequence):
        raise ValueError(f"{context} powinien być sekwencją")
    return value


def _ensure_non_negative_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except Exception as exc:  # noqa: BLE001 - błąd walidacji wejścia
        raise ValueError(f"{field} musi być liczbą") from exc
    if number < 0:
        raise ValueError(f"{field} nie może być ujemne")
    return number


def _optional_non_negative_float(value: object | None, *, field: str) -> float | None:
    if value is None:
        return None
    return _ensure_non_negative_float(value, field=field)


def _ensure_patterns(value: object, *, field: str) -> tuple[str, ...]:
    sequence = _ensure_sequence(value, context=field)
    patterns: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise ValueError(f"Element {field} musi być napisem")
        if not item:
            raise ValueError(f"Element {field} nie może być pusty")
        patterns.append(item)
    if not patterns:
        raise ValueError(f"Lista {field} nie może być pusta")
    return tuple(patterns)


@dataclass(slots=True)
class FailoverServicePlan:
    name: str
    max_rto_minutes: float
    max_rpo_minutes: float
    observed_rto_minutes: float | None
    observed_rpo_minutes: float | None
    required_artifacts: tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(slots=True)
class FailoverDrillPlan:
    drill_name: str
    executed_at: str | None
    services: tuple[FailoverServicePlan, ...]
    metadata: Mapping[str, object]

    @staticmethod
    def from_mapping(document: Mapping[str, object]) -> "FailoverDrillPlan":
        drill_name = document.get("drill_name")
        if not isinstance(drill_name, str) or not drill_name.strip():
            raise ValueError("Plan musi zawierać pole 'drill_name'")
        executed_at = document.get("executed_at")
        if executed_at is not None and (not isinstance(executed_at, str) or not executed_at.strip()):
            raise ValueError("Pole 'executed_at' musi być niepustym napisem lub None")
        services_value = document.get("services")
        services_seq = _ensure_sequence(services_value, context="services")
        services: list[FailoverServicePlan] = []
        for raw in services_seq:
            mapping = _ensure_mapping(raw, context="service")
            name = mapping.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Każda usługa musi posiadać nazwę")
            max_rto = _ensure_non_negative_float(mapping.get("max_rto_minutes"), field="max_rto_minutes")
            max_rpo = _ensure_non_negative_float(mapping.get("max_rpo_minutes"), field="max_rpo_minutes")
            observed_rto = _optional_non_negative_float(mapping.get("observed_rto_minutes"), field="observed_rto_minutes")
            observed_rpo = _optional_non_negative_float(mapping.get("observed_rpo_minutes"), field="observed_rpo_minutes")
            required = _ensure_patterns(mapping.get("required_artifacts"), field="required_artifacts")
            metadata = mapping.get("metadata")
            metadata_mapping = dict(_ensure_mapping(metadata, context="metadata")) if metadata else {}
            services.append(
                FailoverServicePlan(
                    name=name,
                    max_rto_minutes=max_rto,
                    max_rpo_minutes=max_rpo,
                    observed_rto_minutes=observed_rto,
                    observed_rpo_minutes=observed_rpo,
                    required_artifacts=required,
                    metadata=metadata_mapping,
                )
            )
        if not services:
            raise ValueError("Plan powinien zawierać co najmniej jedną usługę")
        metadata_value = document.get("metadata")
        metadata = dict(_ensure_mapping(metadata_value, context="metadata")) if metadata_value else {}
        return FailoverDrillPlan(
            drill_name=drill_name,
            executed_at=executed_at,
            services=tuple(services),
            metadata=metadata,
        )


def load_failover_plan(path: Path) -> FailoverDrillPlan:
    path = path.expanduser()
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise ValueError(f"Nie znaleziono pliku planu: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise ValueError(f"Plan nie jest poprawnym JSON: {exc}") from exc
    if not isinstance(content, Mapping):
        raise ValueError("Plan failover musi być obiektem JSON")
    return FailoverDrillPlan.from_mapping(content)


@dataclass(slots=True)
class FailoverServiceResult:
    name: str
    status: str
    max_rto_minutes: float
    observed_rto_minutes: float | None
    max_rpo_minutes: float
    observed_rpo_minutes: float | None
    missing_artifacts: tuple[str, ...]
    matched_artifacts: tuple[str, ...]
    issues: tuple[str, ...]
    metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "max_rto_minutes": self.max_rto_minutes,
            "observed_rto_minutes": self.observed_rto_minutes,
            "max_rpo_minutes": self.max_rpo_minutes,
            "observed_rpo_minutes": self.observed_rpo_minutes,
            "missing_artifacts": list(self.missing_artifacts),
            "matched_artifacts": list(self.matched_artifacts),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class FailoverDrillSummary:
    drill_name: str
    executed_at: str | None
    generated_at: str
    services: tuple[FailoverServiceResult, ...]
    status: str
    counts: Mapping[str, int]
    metadata: Mapping[str, object]
    bundle_audit: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SUMMARY_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "drill_name": self.drill_name,
            "executed_at": self.executed_at,
            "generated_at": self.generated_at,
            "status": self.status,
            "counts": dict(self.counts),
            "services": [service.to_dict() for service in self.services],
            "metadata": dict(self.metadata),
            "bundle_audit": dict(self.bundle_audit) if self.bundle_audit else None,
        }


def _manifest_files(manifest: Mapping[str, object]) -> tuple[str, ...]:
    files = manifest.get("files")
    if not isinstance(files, Sequence):
        return ()
    paths: list[str] = []
    for entry in files:
        if isinstance(entry, Mapping):
            path_value = entry.get("path")
            if isinstance(path_value, str):
                paths.append(path_value)
    return tuple(sorted(paths))


def _match_patterns(patterns: Iterable[str], files: Sequence[str]) -> tuple[str, ...]:
    matches: list[str] = []
    for pattern in patterns:
        for candidate in files:
            if fnmatch(candidate, pattern):
                matches.append(candidate)
    return tuple(sorted(set(matches)))


def _evaluate_service(plan: FailoverServicePlan, files: Sequence[str]) -> FailoverServiceResult:
    missing: list[str] = []
    issues: list[str] = []
    matched = _match_patterns(plan.required_artifacts, files)
    for pattern in plan.required_artifacts:
        if not any(fnmatch(item, pattern) for item in files):
            missing.append(pattern)
    status = "ok"
    if missing:
        status = "failed"
        issues.append("Brak wymaganych artefaktów")
    if plan.observed_rto_minutes is None:
        if status == "ok":
            status = "warning"
        issues.append("Brak pomiaru RTO")
    elif plan.observed_rto_minutes > plan.max_rto_minutes:
        status = "failed"
        issues.append(
            f"RTO {plan.observed_rto_minutes} min przekracza limit {plan.max_rto_minutes} min"
        )
    if plan.observed_rpo_minutes is None:
        if status == "ok":
            status = "warning"
        issues.append("Brak pomiaru RPO")
    elif plan.observed_rpo_minutes > plan.max_rpo_minutes:
        status = "failed"
        issues.append(
            f"RPO {plan.observed_rpo_minutes} min przekracza limit {plan.max_rpo_minutes} min"
        )
    return FailoverServiceResult(
        name=plan.name,
        status=status,
        max_rto_minutes=plan.max_rto_minutes,
        observed_rto_minutes=plan.observed_rto_minutes,
        max_rpo_minutes=plan.max_rpo_minutes,
        observed_rpo_minutes=plan.observed_rpo_minutes,
        missing_artifacts=tuple(sorted(missing)),
        matched_artifacts=matched,
        issues=tuple(issues),
        metadata=plan.metadata,
    )


def evaluate_failover_drill(
    plan: FailoverDrillPlan,
    manifest: Mapping[str, object],
    *,
    bundle_audit: BundleAuditResult | None = None,
) -> FailoverDrillSummary:
    files = _manifest_files(manifest)
    services = tuple(_evaluate_service(service, files) for service in plan.services)
    counts = {
        "total": len(services),
        "ok": sum(1 for service in services if service.status == "ok"),
        "warning": sum(1 for service in services if service.status == "warning"),
        "failed": sum(1 for service in services if service.status == "failed"),
    }
    status = "ok"
    if counts["failed"]:
        status = "failed"
    elif counts["warning"]:
        status = "warning"
    bundle_snapshot: Mapping[str, object] | None = None
    if bundle_audit is not None:
        bundle_snapshot = bundle_audit.to_dict()
        if bundle_audit.errors:
            status = "failed"
        elif bundle_audit.warnings and status == "ok":
            status = "warning"
    summary = FailoverDrillSummary(
        drill_name=plan.drill_name,
        executed_at=plan.executed_at,
        generated_at=_timestamp(),
        services=services,
        status=status,
        counts=counts,
        metadata=plan.metadata,
        bundle_audit=bundle_snapshot,
    )
    return summary


def write_summary_json(summary: FailoverDrillSummary, output_path: Path) -> Mapping[str, object]:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_summary_csv(summary: FailoverDrillSummary, output_path: Path) -> None:
    import csv

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "service",
                "status",
                "max_rto_minutes",
                "observed_rto_minutes",
                "max_rpo_minutes",
                "observed_rpo_minutes",
                "missing_artifacts",
                "issues",
            ]
        )
        for service in summary.services:
            writer.writerow(
                [
                    service.name,
                    service.status,
                    service.max_rto_minutes,
                    service.observed_rto_minutes if service.observed_rto_minutes is not None else "",
                    service.max_rpo_minutes,
                    service.observed_rpo_minutes if service.observed_rpo_minutes is not None else "",
                    " | ".join(service.missing_artifacts),
                    " | ".join(service.issues),
                ]
            )


def write_summary_signature(
    summary_payload: Mapping[str, object],
    output_path: Path,
    *,
    key: bytes,
    key_id: str | None = None,
    target: str | None = None,
) -> Mapping[str, object]:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": _SUMMARY_SIGNATURE_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "signed_at": _timestamp(),
        "target": target or output_path.name,
        "signature": build_hmac_signature(summary_payload, key=key, key_id=key_id),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "FailoverServicePlan",
    "FailoverDrillPlan",
    "FailoverServiceResult",
    "FailoverDrillSummary",
    "load_failover_plan",
    "evaluate_failover_drill",
    "write_summary_json",
    "write_summary_csv",
    "write_summary_signature",
]
