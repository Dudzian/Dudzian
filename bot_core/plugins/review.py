"""Review pluginów strategii przed publikacją."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .manifest import SignedStrategyPlugin
from .signing import PluginVerifier


class ReviewStatus(str, Enum):
    """Status końcowy procesu review."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


@dataclass(slots=True)
class PluginReviewFinding:
    """Pojedyncze spostrzeżenie z procesu review."""

    severity: str
    message: str


@dataclass(slots=True)
class PluginReviewResult:
    """Rezultat walidacji manifestu."""

    status: ReviewStatus
    findings: Sequence[PluginReviewFinding]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "findings": [
                {"severity": finding.severity, "message": finding.message}
                for finding in self.findings
            ],
        }


class PluginReviewBoard:
    """Zespół review oceniający pluginy strategii."""

    def __init__(self, verifier: PluginVerifier | None = None) -> None:
        self._verifier = verifier

    def evaluate(self, package: SignedStrategyPlugin) -> PluginReviewResult:
        findings: list[PluginReviewFinding] = []

        manifest = package.manifest
        identifier = manifest.identifier.strip()
        if not identifier:
            findings.append(PluginReviewFinding(severity="error", message="Manifest wymaga identyfikatora"))

        if not manifest.version.strip():
            findings.append(PluginReviewFinding(severity="error", message="Brak wersji manifestu"))

        if not manifest.strategies:
            findings.append(PluginReviewFinding(severity="error", message="Manifest musi definiować strategie"))

        if not manifest.capabilities:
            findings.append(
                PluginReviewFinding(
                    severity="warning",
                    message="Rekomendacja: zdefiniuj capabilities opisujące wymagania platformy",
                )
            )

        if self._verifier is not None and not self._verifier.verify(manifest, package.signature):
            findings.append(PluginReviewFinding(severity="error", message="Podpis manifestu jest nieprawidłowy"))

        for note in package.review_notes:
            findings.append(PluginReviewFinding(severity="info", message=f"note: {note}"))

        errors = [finding for finding in findings if finding.severity == "error"]
        if errors:
            status = ReviewStatus.REJECTED
        elif any(finding.severity == "warning" for finding in findings):
            status = ReviewStatus.NEEDS_CHANGES
        else:
            status = ReviewStatus.ACCEPTED

        return PluginReviewResult(status=status, findings=tuple(findings))


__all__ = [
    "ReviewStatus",
    "PluginReviewFinding",
    "PluginReviewResult",
    "PluginReviewBoard",
]

