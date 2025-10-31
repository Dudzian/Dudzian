"""Generowanie raportów audytu licencji OEM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


class LicenseAuditError(RuntimeError):
    """Podstawowy wyjątek modułu audytu licencji."""


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(slots=True)
class LicenseActivationRecord:
    """Pojedynczy wpis audytu licencji."""

    timestamp: datetime
    license_id: str | None
    edition: str | None
    local_hwid_hash: str | None
    activation_count: int
    repeat_activation: bool
    bundle_path: str | None
    payload_sha256: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LicenseActivationRecord | None:
        timestamp = _parse_iso_datetime(payload.get("timestamp"))
        if timestamp is None:
            return None

        event = str(payload.get("event") or "").strip()
        if event and event != "license_snapshot":
            return None

        license_id = str(payload.get("license_id") or "").strip() or None
        edition = str(payload.get("edition") or "").strip() or None
        hwid_hash = str(payload.get("local_hwid_hash") or "").strip() or None

        try:
            activation_count = int(payload.get("activation_count") or 0)
        except (TypeError, ValueError):
            activation_count = 0

        repeat_activation = bool(payload.get("repeat_activation"))
        bundle_path = str(payload.get("bundle_path") or "").strip() or None
        payload_sha256 = str(payload.get("payload_sha256") or "").strip() or None

        return cls(
            timestamp=timestamp,
            license_id=license_id,
            edition=edition,
            local_hwid_hash=hwid_hash,
            activation_count=activation_count,
            repeat_activation=repeat_activation,
            bundle_path=bundle_path,
            payload_sha256=payload_sha256,
        )

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "license_id": self.license_id,
            "edition": self.edition,
            "local_hwid_hash": self.local_hwid_hash,
            "activation_count": self.activation_count,
            "repeat_activation": self.repeat_activation,
            "bundle_path": self.bundle_path,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(slots=True)
class LicenseAuditSummary:
    total_activations: int = 0
    unique_devices: int = 0
    latest_activation: datetime | None = None
    license_id: str | None = None
    edition: str | None = None

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "total_activations": self.total_activations,
            "unique_devices": self.unique_devices,
            "latest_activation": (
                self.latest_activation.isoformat().replace("+00:00", "Z")
                if self.latest_activation
                else None
            ),
            "license_id": self.license_id,
            "edition": self.edition,
        }


@dataclass(slots=True)
class LicenseAuditReport:
    """Struktura wyniku audytu licencji."""

    generated_at: datetime
    status_path: Path
    audit_log_path: Path
    summary: LicenseAuditSummary
    status_document: Mapping[str, object] | None = None
    activations: Sequence[LicenseActivationRecord] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "generated_at": self.generated_at.isoformat().replace("+00:00", "Z"),
            "status_path": str(self.status_path),
            "audit_log_path": str(self.audit_log_path),
            "summary": self.summary.to_dict(),
            "status_document": self.status_document,
            "activations": [record.to_dict() for record in self.activations],
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Raport audytu licencji",
            "",
            f"- Data wygenerowania: {self.generated_at.astimezone().isoformat(timespec='seconds')}",
            f"- Plik statusu: {self.status_path}",
            f"- Plik dziennika audytu: {self.audit_log_path}",
            "",
            "## Podsumowanie",
            "",
            f"- Liczba aktywacji: {self.summary.total_activations}",
            f"- Liczba urządzeń: {self.summary.unique_devices}",
            f"- Ostatnia aktywacja: {self.summary.latest_activation.isoformat().replace('+00:00', 'Z') if self.summary.latest_activation else 'brak'}",
            f"- ID licencji: {self.summary.license_id or 'brak'}",
            f"- Edycja: {self.summary.edition or 'brak'}",
        ]

        if self.status_document:
            lines.extend(["", "## Bieżący status", ""])
            for key in ("license_id", "edition", "effective_date", "issued_at", "maintenance", "trial"):
                if key in self.status_document:
                    value = self.status_document[key]
                    if isinstance(value, Mapping):
                        value = json.dumps(value, ensure_ascii=False)
                    lines.append(f"- {key}: {value}")

        if self.activations:
            lines.extend(["", "## Historia aktywacji", "", "| Data | Licencja | Edycja | Urządzenie | Powtórka |", "| --- | --- | --- | --- | --- |"])
            for record in self.activations:
                hwid = record.local_hwid_hash or "-"
                repeated = "tak" if record.repeat_activation else "nie"
                lines.append(
                    "| {timestamp} | {license_id} | {edition} | {hwid} | {repeat} |".format(
                        timestamp=record.timestamp.isoformat().replace("+00:00", "Z"),
                        license_id=record.license_id or "-",
                        edition=record.edition or "-",
                        hwid=hwid,
                        repeat=repeated,
                    )
                )

        if self.warnings:
            lines.extend(["", "## Ostrzeżenia", ""])
            lines.extend(f"- {message}" for message in self.warnings)

        return "\n".join(lines).strip() + "\n"


def _load_status_document(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LicenseAuditError(f"Nie udało się odczytać statusu licencji: {exc}") from exc
    try:
        document = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LicenseAuditError(f"Status licencji zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise LicenseAuditError("Status licencji powinien zawierać obiekt JSON.")
    return document


def _load_audit_records(path: Path) -> Iterable[LicenseActivationRecord]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = list(handle)
    except OSError as exc:
        raise LicenseAuditError(f"Nie udało się odczytać dziennika audytu: {exc}") from exc

    records: list[LicenseActivationRecord] = []
    for raw in lines:
        entry = raw.strip()
        if not entry:
            continue
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        record = LicenseActivationRecord.from_mapping(payload)
        if record is not None:
            records.append(record)
    return records


def generate_license_audit_report(
    *,
    status_path: str | Path | None = None,
    audit_log_path: str | Path | None = None,
    activation_limit: int | None = None,
) -> LicenseAuditReport:
    """Buduje raport audytu licencji na podstawie lokalnych plików."""

    status_file = Path(status_path).expanduser() if status_path else Path("var/security/license_status.json")
    audit_file = Path(audit_log_path).expanduser() if audit_log_path else Path("logs/security_admin.log")

    warnings: list[str] = []

    status_exists = status_file.exists()
    try:
        status_document = _load_status_document(status_file) if status_exists else None
    except LicenseAuditError as exc:
        status_document = None
        warnings.append(str(exc))

    audit_exists = audit_file.exists()
    try:
        records = list(_load_audit_records(audit_file))
    except LicenseAuditError as exc:
        records = []
        warnings.append(str(exc))

    records.sort(key=lambda item: item.timestamp, reverse=True)
    if activation_limit is not None and activation_limit >= 0:
        records = records[: activation_limit or None]

    summary = LicenseAuditSummary()
    summary.total_activations = len(records)
    summary.unique_devices = len({record.local_hwid_hash for record in records if record.local_hwid_hash})
    summary.latest_activation = records[0].timestamp if records else None

    if status_document:
        summary.license_id = str(status_document.get("license_id") or "").strip() or None
        summary.edition = str(status_document.get("edition") or "").strip() or None
    elif records:
        summary.license_id = records[0].license_id
        summary.edition = records[0].edition

    if not status_document and not status_exists:
        warnings.append("Brak pliku statusu licencji – raport bazuje wyłącznie na dzienniku audytu.")
    if not records:
        if not audit_exists:
            warnings.append("Brak dziennika audytu licencji.")
        else:
            warnings.append("Brak wpisów w dzienniku audytu licencji.")

    generated_at = datetime.now(timezone.utc)

    return LicenseAuditReport(
        generated_at=generated_at,
        status_path=status_file,
        audit_log_path=audit_file,
        summary=summary,
        status_document=status_document,
        activations=tuple(records),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "LicenseAuditError",
    "LicenseActivationRecord",
    "LicenseAuditSummary",
    "LicenseAuditReport",
    "generate_license_audit_report",
]

