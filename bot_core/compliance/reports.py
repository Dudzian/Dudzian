"""Walidacja raportów zgodności Stage5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, MutableMapping, Sequence

from bot_core.security.signing import build_hmac_signature


_ALLOWED_STATUS = {"pass", "warn", "fail"}


@dataclass(slots=True)
class ComplianceControl:
    control_id: str
    status: str
    description: str | None = None
    evidence: Sequence[str] | None = None
    metadata: Mapping[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        payload: MutableMapping[str, object] = {
            "control_id": str(self.control_id),
            "status": str(self.status).lower(),
        }
        if self.description:
            payload["description"] = str(self.description)
        if self.evidence:
            payload["evidence"] = [str(item) for item in self.evidence if str(item)]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return dict(payload)


@dataclass(slots=True)
class ComplianceReport:
    report_id: str
    generated_at: datetime
    controls: Sequence[ComplianceControl]
    report_type: str = "stage5_compliance"
    version: str | None = None
    metadata: Mapping[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        payload: MutableMapping[str, object] = {
            "report_id": str(self.report_id),
            "report_type": str(self.report_type),
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "controls": [control.to_payload() for control in self.controls],
        }
        if self.version:
            payload["version"] = str(self.version)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return dict(payload)


@dataclass(slots=True)
class ComplianceReportValidation:
    report_path: Path
    issues: list[str]
    warnings: list[str]
    failed_controls: list[str]
    passed_controls: list[str]
    metadata: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return not self.issues and not self.failed_controls


def load_compliance_report(path: str | Path) -> dict[str, object]:
    content = Path(path).read_text(encoding="utf-8")
    return json.loads(content)


def _parse_timestamp(value: object, *, issues: list[str], field: str) -> datetime | None:
    if not isinstance(value, str):
        issues.append(f"Pole {field} musi być tekstem w formacie ISO8601.")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        issues.append(f"Pole {field} zawiera niepoprawną datę: {value}")
        return None
    return parsed


def _validate_control(raw: Mapping[str, object], *, issues: list[str], warnings: list[str]) -> str:
    control_id = raw.get("control_id")
    if not isinstance(control_id, str) or not control_id.strip():
        issues.append("Kontrola bez prawidłowego pola control_id.")
        control_id = "<unknown>"
    status = raw.get("status")
    if not isinstance(status, str) or status.lower() not in _ALLOWED_STATUS:
        issues.append(f"Kontrola {control_id} ma niepoprawny status: {status!r}")
        status_value = "fail"
    else:
        status_value = status.lower()
    evidence = raw.get("evidence")
    if evidence is not None and not (
        isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes))
    ):
        issues.append(f"Kontrola {control_id} ma niepoprawne pole evidence (oczekiwano listy).")
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        warnings.append(f"Kontrola {control_id} posiada opis w niepoprawnym formacie (konwertowano).")
    return control_id, status_value


def _verify_signature(
    payload: Mapping[str, object],
    *,
    signature: Mapping[str, object] | None,
    signing_key: bytes | None,
    require_signature: bool,
    issues: list[str],
) -> None:
    if not signature:
        if require_signature:
            issues.append("Raport nie zawiera podpisu HMAC.")
        return
    if signing_key is None:
        return
    expected = build_hmac_signature(payload, key=signing_key, key_id=signature.get("key_id"))
    if signature.get("value") != expected.get("value"):
        issues.append("Niepoprawny podpis HMAC dla raportu.")


def validate_compliance_report(
    raw: Mapping[str, object],
    *,
    expected_controls: Iterable[str] | None = None,
    signing_key: bytes | None = None,
    require_signature: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    failed: list[str] = []
    passed: list[str] = []

    report_type = raw.get("report_type")
    if report_type not in {"stage5_compliance", "compliance"}:
        warnings.append(f"Oczekiwano report_type 'stage5_compliance', otrzymano {report_type!r}.")

    generated_at = _parse_timestamp(raw.get("generated_at"), issues=issues, field="generated_at")
    if generated_at and generated_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        warnings.append("Zarejestrowano raport z przyszłości – sprawdź zegar systemowy.")

    controls_raw = raw.get("controls")
    if not isinstance(controls_raw, Sequence):
        issues.append("Raport musi zawierać listę kontroli w polu controls.")
        controls_raw = []
    elif not controls_raw:
        issues.append("Raport nie zawiera żadnych kontroli zgodności.")

    observed_ids: set[str] = set()
    for item in controls_raw:
        if not isinstance(item, Mapping):
            issues.append("Element listy controls nie jest słownikiem.")
            continue
        control_id, status = _validate_control(item, issues=issues, warnings=warnings)
        if control_id in observed_ids:
            warnings.append(f"Kontrola {control_id} występuje wielokrotnie – rozważ deduplikację.")
        observed_ids.add(control_id)
        if status == "fail":
            failed.append(control_id)
        elif status == "pass":
            passed.append(control_id)

    if expected_controls:
        missing = sorted(set(expected_controls) - observed_ids)
        if missing:
            issues.append("Brakuje kontroli: " + ", ".join(missing))

    signature_payload = {k: v for k, v in raw.items() if k != "signature"}
    signature = raw.get("signature") if isinstance(raw.get("signature"), Mapping) else None
    _verify_signature(
        signature_payload,
        signature=signature,
        signing_key=signing_key,
        require_signature=require_signature,
        issues=issues,
    )

    return issues, warnings, failed, passed


from datetime import timedelta  # noqa: E402  (używane w validate_compliance_report)


def validate_compliance_reports(
    paths: Sequence[Path],
    *,
    expected_controls: Iterable[str] | None = None,
    signing_key: bytes | None = None,
    require_signature: bool = False,
) -> list[ComplianceReportValidation]:
    results: list[ComplianceReportValidation] = []
    for path in paths:
        try:
            data = load_compliance_report(path)
        except Exception as exc:  # pragma: no cover - ścieżka błędu IO
            results.append(
                ComplianceReportValidation(
                    report_path=Path(path),
                    issues=[f"Nie udało się wczytać raportu: {exc}"],
                    warnings=[],
                    failed_controls=[],
                    passed_controls=[],
                    metadata={},
                )
            )
            continue
        issues, warnings, failed, passed = validate_compliance_report(
            data,
            expected_controls=expected_controls,
            signing_key=signing_key,
            require_signature=require_signature,
        )
        results.append(
            ComplianceReportValidation(
                report_path=Path(path),
                issues=issues,
                warnings=warnings,
                failed_controls=failed,
                passed_controls=passed,
                metadata={
                    "report_id": data.get("report_id"),
                    "generated_at": data.get("generated_at"),
                    "control_count": len(data.get("controls", []) if isinstance(data.get("controls"), Sequence) else []),
                },
            )
        )
    return results


__all__ = [
    "ComplianceControl",
    "ComplianceReport",
    "ComplianceReportValidation",
    "load_compliance_report",
    "validate_compliance_report",
    "validate_compliance_reports",
]
