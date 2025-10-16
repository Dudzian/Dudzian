"""Agregacja raportów Stage5 i Stage6 w jeden podpisany przegląd hypercare."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from bot_core.runtime.stage5_hypercare import (
    Stage5HypercareVerificationResult,
    verify_stage5_hypercare_summary,
)
from bot_core.runtime.stage6_hypercare import (
    Stage6HypercareVerificationResult,
    verify_stage6_hypercare_summary,
)
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


def _now_utc_iso() -> str:
    timestamp = datetime.now(timezone.utc)
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _status_from_flags(issues: list[str], warnings: list[str]) -> str:
    if issues:
        return "fail"
    if warnings:
        return "warn"
    return "ok"


def _path_or_none(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def _coerce_strings(value: Any, *, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError(f"Pole '{field_name}' powinno zawierać teksty")
            result.append(entry)
        return result
    raise ValueError(f"Pole '{field_name}' powinno być listą tekstów")


def _collect_component_statuses(components: Any) -> tuple[dict[str, str], list[str]]:
    statuses: dict[str, str] = {}
    issues: list[str] = []
    if components is None:
        issues.append("Brak sekcji 'components' w raporcie")
        return statuses, issues
    if not isinstance(components, Mapping):
        issues.append("Pole 'components' powinno być obiektem JSON")
        return statuses, issues

    for name, payload in components.items():
        if not isinstance(name, str):  # pragma: no cover - defensywne
            continue
        if not isinstance(payload, Mapping):
            issues.append(f"Komponent '{name}' powinien być obiektem JSON")
            continue
        status = str(payload.get("status", "unknown"))
        if status not in {"ok", "warn", "fail", "skipped", "unknown"}:
            issues.append(f"Nieobsługiwany status komponentu '{name}': '{status}'")
        statuses[name] = status
    return statuses, issues


@dataclass(slots=True)
class FullHypercareSummaryConfig:
    """Konfiguracja zbiorczego raportu hypercare."""

    stage5_summary_path: Path
    stage6_summary_path: Path
    output_path: Path
    stage5_signature_path: Path | None = None
    stage6_signature_path: Path | None = None
    stage5_signing_key: bytes | None = None
    stage6_signing_key: bytes | None = None
    stage5_require_signature: bool = False
    stage6_require_signature: bool = False
    signature_path: Path | None = None
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class FullHypercareSummaryResult:
    """Wynik agregacji raportów Stage5 i Stage6."""

    payload: Mapping[str, Any]
    output_path: Path
    signature_path: Path | None
    stage5: Stage5HypercareVerificationResult
    stage6: Stage6HypercareVerificationResult


@dataclass(slots=True)
class FullHypercareVerificationResult:
    """Wynik weryfikacji zbiorczego raportu hypercare."""

    summary_path: Path
    signature_path: Path | None
    summary: Mapping[str, Any]
    signature_valid: bool
    overall_status: str
    issues: list[str]
    warnings: list[str]
    component_statuses: Mapping[str, str]
    stage5: Stage5HypercareVerificationResult | None
    stage6: Stage6HypercareVerificationResult | None


class FullHypercareSummaryBuilder:
    """Buduje podpisany raport zbiorczy na bazie wyników Stage5 i Stage6."""

    def __init__(self, config: FullHypercareSummaryConfig) -> None:
        self._config = config

    def run(self) -> FullHypercareSummaryResult:
        stage5_result = verify_stage5_hypercare_summary(
            self._config.stage5_summary_path,
            signature_path=self._config.stage5_signature_path,
            signing_key=self._config.stage5_signing_key,
            require_signature=self._config.stage5_require_signature,
        )
        stage6_result = verify_stage6_hypercare_summary(
            self._config.stage6_summary_path,
            signature_path=self._config.stage6_signature_path,
            signing_key=self._config.stage6_signing_key,
            require_signature=self._config.stage6_require_signature,
        )

        issues: list[str] = []
        warnings: list[str] = []
        components: MutableMapping[str, Mapping[str, Any]] = {}

        issues.extend(stage5_result.issues)
        warnings.extend(stage5_result.warnings)
        components["stage5"] = {
            "status": stage5_result.overall_status,
            "summary_path": stage5_result.summary_path.as_posix(),
            "signature_path": _path_or_none(stage5_result.signature_path),
            "signature_valid": stage5_result.signature_valid,
            "issues": list(stage5_result.issues),
            "warnings": list(stage5_result.warnings),
            "artifacts": dict(stage5_result.artifact_statuses),
        }

        issues.extend(stage6_result.issues)
        warnings.extend(stage6_result.warnings)
        components["stage6"] = {
            "status": stage6_result.overall_status,
            "summary_path": stage6_result.summary_path.as_posix(),
            "signature_path": _path_or_none(stage6_result.signature_path),
            "signature_valid": stage6_result.signature_valid,
            "issues": list(stage6_result.issues),
            "warnings": list(stage6_result.warnings),
            "components": dict(stage6_result.component_statuses),
        }

        overall_status = _status_from_flags(issues, warnings)

        payload: MutableMapping[str, Any] = {
            "type": "full_hypercare_summary",
            "generated_at": _now_utc_iso(),
            "overall_status": overall_status,
            "issues": list(issues),
            "warnings": list(warnings),
            "components": dict(components),
        }
        if self._config.metadata:
            payload["metadata"] = dict(self._config.metadata)

        output_path = self._config.output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        signature_path: Path | None = None
        if self._config.signing_key:
            target_path = (
                self._config.signature_path.expanduser()
                if self._config.signature_path
                else output_path.with_suffix(output_path.suffix + ".sig")
            )
            signature = build_hmac_signature(
                payload,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(signature, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            signature_path = target_path

        return FullHypercareSummaryResult(
            payload=dict(payload),
            output_path=output_path,
            signature_path=signature_path,
            stage5=stage5_result,
            stage6=stage6_result,
        )


def verify_full_hypercare_summary(
    summary_path: Path,
    *,
    signature_path: Path | None = None,
    signing_key: bytes | None = None,
    require_signature: bool = False,
    revalidate_stage5: bool = False,
    revalidate_stage6: bool = False,
    stage5_signing_key: bytes | None = None,
    stage6_signing_key: bytes | None = None,
    stage5_require_signature: bool = False,
    stage6_require_signature: bool = False,
) -> FullHypercareVerificationResult:
    """Weryfikuje podpis zbiorczy oraz opcjonalnie ponownie waliduje Stage5/Stage6."""

    summary_path = summary_path.expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono raportu zbiorczego: {summary_path}")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_data, Mapping):
        raise ValueError("Raport zbiorczy hypercare powinien być obiektem JSON")

    issues: list[str] = []
    warnings: list[str] = []

    if summary_data.get("type") != "full_hypercare_summary":
        issues.append("Niepoprawny typ raportu – oczekiwano 'full_hypercare_summary'")

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

    component_statuses, component_issues = _collect_component_statuses(summary_data.get("components"))
    issues.extend(component_issues)

    overall_status = str(summary_data.get("overall_status", "unknown"))
    if overall_status not in {"ok", "warn", "fail", "skipped", "unknown"}:
        issues.append(f"Nieobsługiwany status raportu: '{overall_status}'")

    if issues and overall_status == "ok":
        warnings.append("Status 'ok' raportu nie odzwierciedla zgłoszonych problemów")
    elif not issues:
        if any(status == "fail" for status in component_statuses.values()) and overall_status != "fail":
            warnings.append("Status raportu powinien być 'fail' przy błędach komponentów")
        elif any(status == "warn" for status in component_statuses.values()) and overall_status == "ok":
            warnings.append("Status raportu może wymagać 'warn' przy ostrzeżeniach komponentów")

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
            issues.append(f"Oczekiwany plik podpisu nie istnieje: {signature_candidate}")

    signature_valid = False
    if signature_doc is not None:
        if signing_key:
            signature_valid = verify_hmac_signature(summary_data, signature_doc, key=signing_key)
            if not signature_valid:
                issues.append("Nieprawidłowy podpis HMAC raportu zbiorczego")
        else:
            warnings.append(
                "Dostarczono podpis raportu zbiorczego, lecz nie przekazano klucza HMAC do weryfikacji"
            )
    else:
        if require_signature:
            issues.append("Wymagany podpis HMAC raportu zbiorczego nie został dostarczony")
        elif signing_key is not None:
            warnings.append("Przekazano klucz HMAC, lecz nie znaleziono podpisu raportu zbiorczego")

    stage5_result: Stage5HypercareVerificationResult | None = None
    if revalidate_stage5:
        stage5_info = None
        components = summary_data.get("components")
        if isinstance(components, Mapping):
            raw = components.get("stage5")
            if isinstance(raw, Mapping):
                stage5_info = raw
        if stage5_info is None:
            issues.append("Brak komponentu 'stage5' w raporcie do ponownej weryfikacji")
        else:
            stage5_path_raw = stage5_info.get("summary_path")
            stage5_sig_raw = stage5_info.get("signature_path")
            if not stage5_path_raw:
                issues.append("Komponent 'stage5' nie zawiera ścieżki do raportu")
            else:
                stage5_path = Path(str(stage5_path_raw)).expanduser()
                stage5_signature = Path(str(stage5_sig_raw)).expanduser() if stage5_sig_raw else None
                stage5_result = verify_stage5_hypercare_summary(
                    stage5_path,
                    signature_path=stage5_signature,
                    signing_key=stage5_signing_key,
                    require_signature=stage5_require_signature,
                )
                issues.extend(stage5_result.issues)
                warnings.extend(stage5_result.warnings)

    stage6_result: Stage6HypercareVerificationResult | None = None
    if revalidate_stage6:
        stage6_info = None
        components = summary_data.get("components")
        if isinstance(components, Mapping):
            raw = components.get("stage6")
            if isinstance(raw, Mapping):
                stage6_info = raw
        if stage6_info is None:
            issues.append("Brak komponentu 'stage6' w raporcie do ponownej weryfikacji")
        else:
            stage6_path_raw = stage6_info.get("summary_path")
            stage6_sig_raw = stage6_info.get("signature_path")
            if not stage6_path_raw:
                issues.append("Komponent 'stage6' nie zawiera ścieżki do raportu")
            else:
                stage6_path = Path(str(stage6_path_raw)).expanduser()
                stage6_signature = Path(str(stage6_sig_raw)).expanduser() if stage6_sig_raw else None
                stage6_result = verify_stage6_hypercare_summary(
                    stage6_path,
                    signature_path=stage6_signature,
                    signing_key=stage6_signing_key,
                    require_signature=stage6_require_signature,
                )
                issues.extend(stage6_result.issues)
                warnings.extend(stage6_result.warnings)

    return FullHypercareVerificationResult(
        summary_path=summary_path,
        signature_path=signature_used_path,
        summary=dict(summary_data),
        signature_valid=signature_valid,
        overall_status=overall_status,
        issues=issues,
        warnings=warnings,
        component_statuses=component_statuses,
        stage5=stage5_result,
        stage6=stage6_result,
    )


__all__ = [
    "FullHypercareSummaryBuilder",
    "FullHypercareSummaryConfig",
    "FullHypercareSummaryResult",
    "FullHypercareVerificationResult",
    "verify_full_hypercare_summary",
]

