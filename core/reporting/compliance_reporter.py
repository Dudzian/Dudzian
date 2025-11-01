"""Generator raportów audytu zgodności KYC/AML."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from core.compliance import ComplianceAuditResult, ComplianceFinding

_ISO_FILENAME = "%Y%m%dT%H%M%S"


@dataclass(slots=True)
class ComplianceFindingRow:
    """Wiersz tabeli z naruszeniem wykrytym w audycie."""

    rule_id: str
    severity: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_finding(cls, finding: ComplianceFinding) -> "ComplianceFindingRow":
        return cls(
            rule_id=finding.rule_id,
            severity=finding.severity,
            message=finding.message,
            metadata=dict(finding.metadata),
        )

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ComplianceReport:
    """Raport audytu zgodności gotowy do zapisania jako Markdown/JSON."""

    generated_at: datetime
    passed: bool
    findings: Sequence[ComplianceFindingRow]
    context_summary: Mapping[str, object]
    config_path: Path | None
    recommendations: Sequence[str]

    @classmethod
    def from_audit(
        cls,
        result: ComplianceAuditResult,
        *,
        recommendations: Sequence[str] | None = None,
    ) -> "ComplianceReport":
        rows = tuple(ComplianceFindingRow.from_finding(item) for item in result.findings)
        summary = dict(result.context_summary)
        suggestions = tuple(recommendations or _default_recommendations(result, rows))
        return cls(
            generated_at=result.generated_at.astimezone(timezone.utc),
            passed=result.passed,
            findings=rows,
            context_summary=summary,
            config_path=result.config_path,
            recommendations=suggestions,
        )

    def to_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "passed": self.passed,
            "context_summary": dict(self.context_summary),
            "findings": [finding.to_dict() for finding in self.findings],
            "recommendations": list(self.recommendations),
        }
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        return payload

    def severity_counts(self) -> Mapping[str, int]:
        counter: dict[str, int] = {}
        for finding in self.findings:
            key = finding.severity.lower().strip() or "unknown"
            counter[key] = counter.get(key, 0) + 1
        return counter

    def to_markdown(self) -> str:
        timestamp = self.generated_at.astimezone(timezone.utc).isoformat()
        lines = [
            "# Raport audytu zgodności",
            "",
            f"Wygenerowano: {timestamp}",
        ]
        if self.config_path is not None:
            lines.append(f"Konfiguracja: {self.config_path}")
        lines.append("")
        lines.extend(self._summary_section())
        lines.extend(self._findings_section())
        lines.extend(self._recommendations_section())
        return "\n".join(lines).strip() + "\n"

    def _summary_section(self) -> list[str]:
        rows = ["## Podsumowanie", ""]
        rows.append(f"Status audytu: {'POZYTYWNY' if self.passed else 'NEGATYWNY'}")
        rows.append("")
        if self.context_summary:
            rows.append("**Kontekst:**")
            for key, value in self.context_summary.items():
                rows.append(f"- {key}: {value}")
            rows.append("")
        counts = self.severity_counts()
        if counts:
            rows.append("**Liczba naruszeń wg poziomu:**")
            for severity, count in sorted(counts.items(), key=lambda item: item[0]):
                rows.append(f"- {severity}: {count}")
            rows.append("")
        return rows

    def _findings_section(self) -> list[str]:
        rows = ["## Naruszenia", ""]
        if not self.findings:
            rows.append("Brak naruszeń – konfiguracja spełnia wymagania KYC/AML.")
            rows.append("")
            return rows
        rows.append(
            "| Reguła | Poziom | Opis | Metadane |"
        )
        rows.append("| --- | --- | --- | --- |")
        for finding in self.findings:
            metadata = ", ".join(f"{key}={value}" for key, value in finding.metadata.items()) or "—"
            rows.append(
                f"| {finding.rule_id} | {finding.severity} | {finding.message} | {metadata} |"
            )
        rows.append("")
        return rows

    def _recommendations_section(self) -> list[str]:
        rows = ["## Rekomendacje", ""]
        if not self.recommendations:
            rows.append("Brak dodatkowych zaleceń.")
        else:
            for item in self.recommendations:
                rows.append(f"- {item}")
        rows.append("")
        return rows

    def write_markdown(self, directory: Path | str) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"compliance_{self.generated_at.strftime(_ISO_FILENAME)}.md"
        path = destination / filename
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def write_json(self, directory: Path | str) -> Path:
        import json

        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"compliance_{self.generated_at.strftime(_ISO_FILENAME)}.json"
        path = destination / filename
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _default_recommendations(
    result: ComplianceAuditResult,
    findings: Sequence[ComplianceFindingRow],
) -> Sequence[str]:
    if not findings:
        return ("Brak naruszeń – można kontynuować przejście w tryb paper/live.",)
    recs: list[str] = []
    severities = {finding.severity.lower() for finding in findings}
    if any(level in severities for level in {"critical", "high"}):
        recs.append(
            "Skontaktuj się z zespołem zgodności przed dalszym uruchomieniem strategii."
        )
    if any(finding.rule_id == "KYC_MISSING_FIELDS" for finding in findings):
        recs.append("Uzupełnij brakujące pola profilu KYC wskazane w metadanych naruszenia.")
    if any(finding.rule_id.startswith("AML_") for finding in findings):
        recs.append("Zweryfikuj źródła danych i kontrahentów pod kątem sankcji/AML.")
    if any(finding.rule_id.startswith("TX_") for finding in findings):
        recs.append("Dostosuj limity transakcyjne lub zmodyfikuj parametry strategii.")
    if not recs:
        recs.append("Przeanalizuj naruszenia i zaktualizuj konfigurację zgodnie z polityką.")
    return tuple(dict.fromkeys(recs))


__all__ = ["ComplianceReport", "ComplianceFindingRow"]
