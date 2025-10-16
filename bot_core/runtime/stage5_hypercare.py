"""Agregacja artefaktów hypercare Stage5 i generowanie podpisanego raportu."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from bot_core.compliance.reports import validate_compliance_reports
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


def _ensure_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value if value is not None else {}


def _utcnow() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_flags(issues: Sequence[str], warnings: Sequence[str]) -> str:
    if issues:
        return "fail"
    if warnings:
        return "warn"
    return "ok"


@dataclass(slots=True)
class Stage5TcoConfig:
    summary_path: Path
    signature_path: Path | None = None
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5RotationConfig:
    summary_path: Path
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5ComplianceConfig:
    reports: Sequence[Path]
    expected_controls: Iterable[str] | None = None
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5TrainingConfig:
    logs: Sequence[Path]
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5SloConfig:
    report_path: Path
    signature_path: Path | None = None
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5OemAcceptanceConfig:
    summary_path: Path
    required_steps: Sequence[str] = ("bundle", "license", "risk", "mtls")
    signature_path: Path | None = None
    signing_key: bytes | None = None
    require_signature: bool = False


@dataclass(slots=True)
class Stage5HypercareConfig:
    output_path: Path
    signature_path: Path | None = None
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    tco: Stage5TcoConfig | None = None
    rotation: Stage5RotationConfig | None = None
    compliance: Stage5ComplianceConfig | None = None
    training: Stage5TrainingConfig | None = None
    slo: Stage5SloConfig | None = None
    oem: Stage5OemAcceptanceConfig | None = None


@dataclass(slots=True)
class Stage5HypercareResult:
    payload: Mapping[str, Any]
    output_path: Path
    signature_path: Path | None


@dataclass(slots=True)
class Stage5HypercareVerificationResult:
    """Wynik weryfikacji raportu hypercare Stage5."""

    summary_path: Path
    signature_path: Path | None
    summary: Mapping[str, Any]
    signature_valid: bool
    overall_status: str
    issues: list[str]
    warnings: list[str]
    artifact_statuses: Mapping[str, str]


class Stage5HypercareCycle:
    """Buduje zbiorczy raport hypercare Stage5."""

    def __init__(self, config: Stage5HypercareConfig) -> None:
        self._config = config

    def run(self) -> Stage5HypercareResult:
        issues: list[str] = []
        warnings: list[str] = []
        artifacts: MutableMapping[str, Mapping[str, Any]] = {}

        if self._config.tco:
            tco_result = self._evaluate_tco(self._config.tco)
            artifacts["tco"] = tco_result
            issues.extend(tco_result["issues"])
            warnings.extend(tco_result["warnings"])

        if self._config.rotation:
            rotation_result = self._evaluate_rotation(self._config.rotation)
            artifacts["key_rotation"] = rotation_result
            issues.extend(rotation_result["issues"])
            warnings.extend(rotation_result["warnings"])

        if self._config.compliance:
            compliance_result = self._evaluate_compliance(self._config.compliance)
            artifacts["compliance"] = compliance_result
            issues.extend(compliance_result["issues"])
            warnings.extend(compliance_result["warnings"])

        if self._config.training:
            training_result = self._evaluate_training(self._config.training)
            artifacts["training"] = training_result
            issues.extend(training_result["issues"])
            warnings.extend(training_result["warnings"])

        if self._config.slo:
            slo_result = self._evaluate_slo(self._config.slo)
            artifacts["slo_monitor"] = slo_result
            issues.extend(slo_result["issues"])
            warnings.extend(slo_result["warnings"])

        if self._config.oem:
            oem_result = self._evaluate_oem(self._config.oem)
            artifacts["oem_acceptance"] = oem_result
            issues.extend(oem_result["issues"])
            warnings.extend(oem_result["warnings"])

        overall_status = _status_from_flags(issues, warnings)
        payload: MutableMapping[str, Any] = {
            "type": "stage5_hypercare_summary",
            "generated_at": _utcnow().isoformat().replace("+00:00", "Z"),
            "overall_status": overall_status,
            "issues": list(issues),
            "warnings": list(warnings),
            "artifacts": dict(artifacts),
        }

        output_path = self._config.output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        signature_path = None
        if self._config.signing_key:
            signature_path = self._config.signature_path or output_path.with_suffix(output_path.suffix + ".sig")
            signature = build_hmac_signature(
                payload,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
            )
            signature_path.parent.mkdir(parents=True, exist_ok=True)
            signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")

        return Stage5HypercareResult(payload=dict(payload), output_path=output_path, signature_path=signature_path)

    # ------------------------------------------------------------------
    # Poszczególne sekcje raportu
    # ------------------------------------------------------------------
    def _evaluate_tco(self, config: Stage5TcoConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        details: MutableMapping[str, Any] = {
            "summary_path": str(config.summary_path.expanduser()),
        }

        try:
            payload = _load_json(config.summary_path.expanduser())
        except FileNotFoundError:
            issues.append("Brak pliku z podsumowaniem TCO.")
            return self._build_section_payload(issues, warnings, details)
        except json.JSONDecodeError:
            issues.append("Nieprawidłowy format JSON w raporcie TCO.")
            return self._build_section_payload(issues, warnings, details)

        details.update(
            {
                "currency": payload.get("currency"),
                "monthly_total": payload.get("monthly_total"),
                "annual_total": payload.get("annual_total"),
                "items_count": len(payload.get("items", [])) if isinstance(payload.get("items"), Sequence) else 0,
            }
        )

        signature_info: MutableMapping[str, Any] | None = None
        if config.signature_path:
            signature_path = config.signature_path.expanduser()
            signature_info = {"path": str(signature_path)}
            if not signature_path.exists():
                issues.append("Brak pliku z podpisem TCO.")
            else:
                try:
                    signature_payload = _load_json(signature_path)
                except json.JSONDecodeError:
                    issues.append("Nieprawidłowy podpis HMAC dla raportu TCO.")
                else:
                    verified = verify_hmac_signature(payload, signature_payload, key=config.signing_key)
                    signature_info["verified"] = bool(verified)
                    if config.signing_key and not verified:
                        issues.append("Podpis HMAC dla raportu TCO nie zgadza się z zawartością.")
                    elif not config.signing_key:
                        signature_info["verified"] = None
                        warnings.append("Brak klucza do weryfikacji podpisu TCO.")
        elif config.require_signature:
            issues.append("Raport TCO musi być podpisany HMAC.")

        if signature_info:
            details["signature"] = dict(signature_info)

        return self._build_section_payload(issues, warnings, details)

    def _evaluate_rotation(self, config: Stage5RotationConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        details: MutableMapping[str, Any] = {
            "summary_path": str(config.summary_path.expanduser()),
        }

        try:
            payload = _load_json(config.summary_path.expanduser())
        except FileNotFoundError:
            issues.append("Brak raportu z rotacji kluczy.")
            return self._build_section_payload(issues, warnings, details)
        except json.JSONDecodeError:
            issues.append("Nieprawidłowy format raportu rotacji kluczy.")
            return self._build_section_payload(issues, warnings, details)

        signature_payload = payload.get("signature") if isinstance(payload.get("signature"), Mapping) else None
        if signature_payload:
            verification_payload = {k: v for k, v in payload.items() if k != "signature"}
            verified = verify_hmac_signature(verification_payload, signature_payload, key=config.signing_key)
            details["signature"] = {
                "present": True,
                "verified": bool(verified) if config.signing_key else None,
            }
            if config.signing_key and not verified:
                issues.append("Podpis HMAC raportu rotacji jest niepoprawny.")
            if not config.signing_key:
                warnings.append("Brak klucza do weryfikacji podpisu raportu rotacji.")
        elif config.require_signature:
            issues.append("Raport rotacji kluczy musi zawierać podpis HMAC.")
            details["signature"] = {"present": False, "verified": False}

        records = payload.get("records") if isinstance(payload.get("records"), Sequence) else []
        stats = payload.get("stats") if isinstance(payload.get("stats"), Mapping) else {}
        details.update(
            {
                "records": len(records),
                "due_before": stats.get("due_before"),
                "overdue_before": stats.get("overdue_before"),
                "operator": payload.get("operator"),
            }
        )

        if not records:
            issues.append("Raport rotacji nie zawiera żadnych wpisów.")

        return self._build_section_payload(issues, warnings, details)

    def _evaluate_compliance(self, config: Stage5ComplianceConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        details: MutableMapping[str, Any] = {"reports": []}

        if not config.reports:
            issues.append("Nie dostarczono raportów zgodności Stage5.")
            return self._build_section_payload(issues, warnings, details)

        results = validate_compliance_reports(
            [path.expanduser() for path in config.reports],
            expected_controls=config.expected_controls,
            signing_key=config.signing_key,
            require_signature=config.require_signature,
        )

        for result in results:
            entry_details: MutableMapping[str, Any] = {
                "path": str(result.report_path),
                "issues": list(result.issues),
                "warnings": list(result.warnings),
                "failed_controls": list(result.failed_controls),
                "passed_controls": len(result.passed_controls),
                "metadata": dict(result.metadata),
            }
            details["reports"].append(entry_details)
            if result.issues or result.failed_controls:
                issues.append(f"Raport {result.report_path.name} zawiera błędy lub kontrole niezdane.")
            if result.warnings:
                warnings.append(f"Raport {result.report_path.name} posiada ostrzeżenia.")

        return self._build_section_payload(issues, warnings, details)

    def _evaluate_training(self, config: Stage5TrainingConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        sessions: list[Mapping[str, Any]] = []

        if not config.logs:
            warnings.append("Brak logów szkoleń Stage5 do analizy.")
            return self._build_section_payload(issues, warnings, {"sessions": sessions})

        for path in config.logs:
            entry_issues: list[str] = []
            entry_warnings: list[str] = []
            try:
                payload = _load_json(path.expanduser())
            except FileNotFoundError:
                entry_issues.append("Nie znaleziono pliku logu szkolenia.")
                payload = {}
            except json.JSONDecodeError:
                entry_issues.append("Niepoprawny format JSON logu szkolenia.")
                payload = {}

            metadata: MutableMapping[str, Any] = {
                "path": str(path.expanduser()),
                "issues": entry_issues,
                "warnings": entry_warnings,
            }

            if isinstance(payload, Mapping):
                metadata.update(
                    {
                        "session_id": payload.get("session_id"),
                        "title": payload.get("title"),
                        "occurred_at": payload.get("occurred_at"),
                        "participants": len(payload.get("participants", []))
                        if isinstance(payload.get("participants"), Sequence)
                        else 0,
                    }
                )

                for field in ("session_id", "title", "trainer", "occurred_at", "summary"):
                    if not payload.get(field):
                        entry_issues.append(f"Pole {field} jest wymagane w logu szkolenia.")

                signature_payload = payload.get("signature") if isinstance(payload.get("signature"), Mapping) else None
                if signature_payload:
                    verification_payload = {k: v for k, v in payload.items() if k != "signature"}
                    verified = verify_hmac_signature(verification_payload, signature_payload, key=config.signing_key)
                    metadata["signature"] = {
                        "present": True,
                        "verified": bool(verified) if config.signing_key else None,
                    }
                    if config.signing_key and not verified:
                        entry_issues.append("Niepoprawny podpis HMAC logu szkolenia.")
                    if not config.signing_key:
                        entry_warnings.append("Brak klucza do weryfikacji podpisu szkolenia.")
                elif config.require_signature:
                    entry_issues.append("Log szkolenia wymaga podpisu HMAC.")
                    metadata["signature"] = {"present": False, "verified": False}
            else:
                entry_issues.append("Log szkolenia ma niepoprawną strukturę.")

            sessions.append(metadata)
            issues.extend(entry_issues)
            warnings.extend(entry_warnings)

        return self._build_section_payload(issues, warnings, {"sessions": sessions})

    def _evaluate_slo(self, config: Stage5SloConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        details: MutableMapping[str, Any] = {
            "report_path": str(config.report_path.expanduser()),
        }

        try:
            payload = _load_json(config.report_path.expanduser())
        except FileNotFoundError:
            issues.append("Brak raportu SLO Stage5.")
            return self._build_section_payload(issues, warnings, details)
        except json.JSONDecodeError:
            issues.append("Niepoprawny format JSON raportu SLO.")
            return self._build_section_payload(issues, warnings, details)

        signature_info: MutableMapping[str, Any] | None = None
        if config.signature_path:
            signature_path = config.signature_path.expanduser()
            signature_info = {"path": str(signature_path)}
            if not signature_path.exists():
                issues.append("Brak pliku z podpisem raportu SLO.")
            else:
                try:
                    signature_payload = _load_json(signature_path)
                except json.JSONDecodeError:
                    issues.append("Niepoprawny plik podpisu raportu SLO.")
                else:
                    verified = verify_hmac_signature(payload, signature_payload, key=config.signing_key)
                    signature_info["verified"] = bool(verified) if config.signing_key else None
                    if config.signing_key and not verified:
                        issues.append("Podpis HMAC raportu SLO jest niepoprawny.")
                    if not config.signing_key:
                        warnings.append("Brak klucza do weryfikacji podpisu raportu SLO.")
        elif config.require_signature:
            issues.append("Raport SLO wymaga podpisu HMAC.")

        summary_raw = payload.get("summary")
        summary = summary_raw if isinstance(summary_raw, Mapping) else {}

        slo_section = summary.get("slo") if isinstance(summary, Mapping) else None
        slo_counts = _ensure_mapping(slo_section.get("status_counts")) if isinstance(slo_section, Mapping) else {}

        composite_section = summary.get("composites") if isinstance(summary, Mapping) else None
        composites_counts = (
            _ensure_mapping(composite_section.get("status_counts"))
            if isinstance(composite_section, Mapping)
            else {}
        )

        breaches = int(slo_counts.get("breach", 0) or 0)
        fails = int(slo_counts.get("fail", 0) or 0)
        warnings_count = int(slo_counts.get("warning", 0) or 0)

        if fails > 0:
            issues.append(f"Raport SLO zawiera {fails} naruszeń krytycznych.")
        if breaches > 0 or warnings_count > 0:
            warnings.append("W raporcie SLO występują odchylenia od celu.")

        details.update(
            {
                "status_counts": slo_counts,
                "composites": composites_counts,
            }
        )
        if signature_info:
            details["signature"] = dict(signature_info)

        return self._build_section_payload(issues, warnings, details)

    def _evaluate_oem(self, config: Stage5OemAcceptanceConfig) -> Mapping[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        details: MutableMapping[str, Any] = {
            "summary_path": str(config.summary_path.expanduser()),
            "steps": [],
        }

        try:
            payload = json.loads(config.summary_path.expanduser().read_text(encoding="utf-8"))
        except FileNotFoundError:
            issues.append("Brak podsumowania akceptacji OEM.")
            return self._build_section_payload(issues, warnings, details)
        except json.JSONDecodeError:
            issues.append("Niepoprawny format podsumowania akceptacji OEM.")
            return self._build_section_payload(issues, warnings, details)

        if not isinstance(payload, Sequence):
            issues.append("Podsumowanie akceptacji OEM powinno być listą kroków.")
            return self._build_section_payload(issues, warnings, details)

        signature_info: MutableMapping[str, Any] | None = None
        if config.signature_path:
            signature_path = config.signature_path.expanduser()
            signature_info = {"path": str(signature_path)}
            if not signature_path.exists():
                issues.append("Brak pliku z podpisem podsumowania OEM.")
            else:
                try:
                    signature_payload = _load_json(signature_path)
                except json.JSONDecodeError:
                    issues.append("Niepoprawny plik podpisu podsumowania OEM.")
                else:
                    verified = verify_hmac_signature(payload, signature_payload, key=config.signing_key)
                    signature_info["verified"] = bool(verified) if config.signing_key else None
                    if config.signing_key and not verified:
                        issues.append("Podpis HMAC podsumowania OEM jest niepoprawny.")
                    if not config.signing_key:
                        warnings.append("Brak klucza do weryfikacji podpisu OEM.")
        elif config.require_signature:
            issues.append("Podsumowanie OEM wymaga podpisu HMAC.")

        observed_steps: set[str] = set()
        for entry in payload:
            if not isinstance(entry, Mapping):
                warnings.append("Pominięto wpis akceptacji OEM o nieprawidłowej strukturze.")
                continue
            step = str(entry.get("step", "")).strip()
            status = str(entry.get("status", "")).lower()
            details_entry = {
                "step": step,
                "status": status,
                "details": entry.get("details", {}),
            }
            details["steps"].append(details_entry)
            if step:
                observed_steps.add(step)
            if status == "failed":
                issues.append(f"Krok OEM '{step}' zakończył się błędem.")
            elif status == "warn":
                warnings.append(f"Krok OEM '{step}' zakończył się ostrzeżeniem.")

        missing = sorted(set(config.required_steps) - observed_steps)
        if missing:
            issues.append("Brakuje kroków OEM: " + ", ".join(missing))

        if signature_info:
            details["signature"] = dict(signature_info)

        return self._build_section_payload(issues, warnings, details)

    # ------------------------------------------------------------------
    # Narzędzia pomocnicze
    # ------------------------------------------------------------------
    def _build_section_payload(
        self,
        issues: Sequence[str],
        warnings: Sequence[str],
        details: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "status": _status_from_flags(issues, warnings),
            "issues": list(issues),
            "warnings": list(warnings),
            "details": dict(details),
        }


def _coerce_strings(sequence: Any, *, field_name: str) -> list[str]:
    if sequence is None:
        return []
    if isinstance(sequence, (list, tuple)):
        return [str(item) for item in sequence]
    raise ValueError(f"Pole '{field_name}' powinno być listą lub krotką")


def _collect_artifact_statuses(artifacts: Any) -> tuple[dict[str, str], list[str]]:
    statuses: dict[str, str] = {}
    issues: list[str] = []

    if artifacts is None:
        issues.append("Brak sekcji 'artifacts' w raporcie Stage5")
        return statuses, issues
    if not isinstance(artifacts, Mapping):
        issues.append("Sekcja 'artifacts' powinna być obiektem JSON")
        return statuses, issues

    for name, payload in artifacts.items():
        if not isinstance(payload, Mapping):
            issues.append(f"Artefakt '{name}' powinien być obiektem z polem 'status'")
            continue
        status = str(payload.get("status", "unknown"))
        statuses[str(name)] = status
        if status not in {"ok", "warn", "fail", "skipped"}:
            issues.append(f"Artefakt '{name}' posiada nieobsługiwany status '{status}'")
        elif status == "fail":
            issues.append(f"Artefakt '{name}' zakończył się statusem 'fail'")

    return statuses, issues


def verify_stage5_hypercare_summary(
    summary_path: Path,
    *,
    signature_path: Path | None = None,
    signing_key: bytes | None = None,
    require_signature: bool = False,
) -> Stage5HypercareVerificationResult:
    """Weryfikuje podpisany raport hypercare Stage5."""

    summary_path = summary_path.expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono raportu Stage5: {summary_path}")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_data, Mapping):  # pragma: no cover - defensywne
        raise ValueError("Raport Stage5 powinien być obiektem JSON")

    issues: list[str] = []
    warnings: list[str] = []

    summary_type = summary_data.get("type")
    if summary_type != "stage5_hypercare_summary":
        issues.append("Niepoprawny typ raportu Stage5 – oczekiwano 'stage5_hypercare_summary'")

    try:
        summary_issues = _coerce_strings(summary_data.get("issues"), field_name="issues")
    except ValueError as exc:
        issues.append(str(exc))
        summary_issues = []

    try:
        summary_warnings = _coerce_strings(summary_data.get("warnings"), field_name="warnings")
    except ValueError as exc:
        issues.append(str(exc))
        summary_warnings = []

    issues.extend(summary_issues)
    warnings.extend(summary_warnings)

    artifacts = summary_data.get("artifacts")
    artifact_statuses, artifact_issues = _collect_artifact_statuses(artifacts)
    issues.extend(artifact_issues)

    overall_status = str(summary_data.get("overall_status", "unknown"))
    if overall_status not in {"ok", "warn", "fail", "skipped", "unknown"}:
        issues.append(f"Nieobsługiwany status raportu Stage5: '{overall_status}'")

    if issues and overall_status == "ok":
        warnings.append("Status 'ok' raportu nie odzwierciedla zgłoszonych problemów")
    elif not issues:
        if any(status == "fail" for status in artifact_statuses.values()):
            if overall_status != "fail":
                warnings.append("Status raportu powinien być 'fail' przy błędach artefaktów")
        elif any(status == "warn" for status in artifact_statuses.values()):
            if overall_status == "ok":
                warnings.append("Status raportu może wymagać 'warn' przy ostrzeżeniach artefaktów")

    signature_doc: Mapping[str, Any] | None = None
    signature_used_path: Path | None = None
    signature_candidate = signature_path.expanduser() if signature_path else None
    if signature_candidate is None:
        candidate = summary_path.with_suffix(summary_path.suffix + ".sig")
        if candidate.exists():
            signature_candidate = candidate

    if signature_candidate:
        if signature_candidate.exists():
            signature_doc = json.loads(signature_candidate.read_text(encoding="utf-8"))
            if not isinstance(signature_doc, Mapping):  # pragma: no cover - defensywne
                issues.append("Podpis HMAC powinien być obiektem JSON")
                signature_doc = None
            else:
                signature_used_path = signature_candidate
        else:
            issues.append(f"Oczekiwany plik podpisu Stage5 nie istnieje: {signature_candidate}")

    signature_valid = False
    if signature_doc is not None:
        if signing_key:
            signature_valid = verify_hmac_signature(summary_data, signature_doc, key=signing_key)
            if not signature_valid:
                issues.append("Nieprawidłowy podpis HMAC dla raportu Stage5")
        else:
            warnings.append(
                "Dostarczono podpis hypercare Stage5, lecz nie przekazano klucza HMAC do weryfikacji"
            )
    else:
        if require_signature:
            issues.append("Wymagany podpis HMAC raportu Stage5 nie został dostarczony")
        elif signing_key is not None:
            warnings.append("Przekazano klucz HMAC, lecz nie znaleziono podpisu raportu Stage5")

    return Stage5HypercareVerificationResult(
        summary_path=summary_path,
        signature_path=signature_used_path,
        summary=dict(summary_data),
        signature_valid=signature_valid,
        overall_status=overall_status,
        issues=issues,
        warnings=warnings,
        artifact_statuses=dict(artifact_statuses),
    )


__all__ = [
    "Stage5HypercareCycle",
    "Stage5HypercareConfig",
    "Stage5HypercareResult",
    "Stage5TcoConfig",
    "Stage5RotationConfig",
    "Stage5ComplianceConfig",
    "Stage5TrainingConfig",
    "Stage5SloConfig",
    "Stage5OemAcceptanceConfig",
    "Stage5HypercareVerificationResult",
    "verify_stage5_hypercare_summary",
]
