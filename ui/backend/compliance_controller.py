"""Kontroler udostępniający wyniki audytu zgodności dla QML."""
from __future__ import annotations

from datetime import timezone
from typing import Callable, Mapping, MutableMapping, Sequence

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.compliance.compliance_auditor import (
    ComplianceAuditResult,
    ComplianceAuditor,
    ComplianceFinding,
)

StrategyProvider = Callable[[], Mapping[str, object]]
DataSourcesProvider = Callable[[], Sequence[str]]
TransactionsProvider = Callable[[], Sequence[Mapping[str, object]]]
KycProfileProvider = Callable[[], Mapping[str, object]]


def _default_strategy_provider() -> Mapping[str, object]:
    return {}


def _default_datasource_provider() -> Sequence[str]:
    return ()


def _default_transactions_provider() -> Sequence[Mapping[str, object]]:
    return ()


def _default_kyc_provider() -> Mapping[str, object]:
    return {}


class ComplianceController(QObject):
    """Uruchamia audyt zgodności i udostępnia jego wynik."""

    busyChanged = Signal()
    summaryChanged = Signal()
    findingsChanged = Signal()
    recommendationsChanged = Signal()
    contextSummaryChanged = Signal()
    lastUpdatedChanged = Signal()
    errorMessageChanged = Signal()
    telemetryProviderChanged = Signal()
    auditCompleted = Signal()

    def __init__(
        self,
        *,
        auditor: ComplianceAuditor | None = None,
        strategy_provider: StrategyProvider | None = None,
        datasource_provider: DataSourcesProvider | None = None,
        transactions_provider: TransactionsProvider | None = None,
        kyc_profile_provider: KycProfileProvider | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._auditor = auditor or ComplianceAuditor()
        self._strategy_provider = strategy_provider or _default_strategy_provider
        self._datasource_provider = datasource_provider or _default_datasource_provider
        self._transactions_provider = transactions_provider or _default_transactions_provider
        self._kyc_profile_provider = kyc_profile_provider or _default_kyc_provider

        self._busy = False
        self._summary: dict[str, object] = {
            "passed": True,
            "kycStatus": "",
            "amlStatus": "",
            "transactionStatus": "",
            "totalFindings": 0,
        }
        self._findings: list[dict[str, object]] = []
        self._recommendations: list[str] = []
        self._context_summary: dict[str, object] = {}
        self._last_updated: str = ""
        self._error_message: str = ""
        self._telemetry_provider: QObject | None = None

    # ------------------------------------------------------------------
    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:  # type: ignore[override]
        return self._busy

    @Property("QVariantMap", notify=summaryChanged)
    def summary(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._summary)

    @Property("QVariantList", notify=findingsChanged)
    def findings(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._findings)

    @Property("QStringList", notify=recommendationsChanged)
    def recommendations(self) -> list[str]:  # type: ignore[override]
        return list(self._recommendations)

    @Property("QVariantMap", notify=contextSummaryChanged)
    def contextSummary(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._context_summary)

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(QObject, notify=telemetryProviderChanged)
    def telemetryProvider(self) -> QObject | None:  # type: ignore[override]
        return self._telemetry_provider

    @telemetryProvider.setter
    def telemetryProvider(self, provider: QObject | None) -> None:  # type: ignore[override]
        if provider is self._telemetry_provider:
            return
        self._telemetry_provider = provider
        self.telemetryProviderChanged.emit()

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshAudit(self) -> bool:
        if self._busy:
            return False
        self._set_busy(True)
        try:
            result = self._execute_audit()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiska
            self._set_error(str(exc))
            self._set_busy(False)
            return False

        self._apply_result(result)
        self._set_error("")
        self._set_busy(False)
        self.auditCompleted.emit()
        return True

    # ------------------------------------------------------------------
    def _execute_audit(self) -> ComplianceAuditResult:
        strategy_config = self._strategy_provider() or {}
        data_sources = self._datasource_provider() or ()
        transactions = self._transactions_provider() or ()
        kyc_profile = self._kyc_profile_provider() or {}
        return self._auditor.audit(
            strategy_config=strategy_config,
            data_sources=data_sources,
            transactions=transactions,
            kyc_profile=kyc_profile,
            event_publisher=None,
        )

    def _apply_result(self, result: ComplianceAuditResult) -> None:
        self._summary = self._build_summary(result)
        self.summaryChanged.emit()

        self._findings = [self._finding_to_dict(item) for item in result.findings]
        self.findingsChanged.emit()

        self._recommendations = self._build_recommendations(result.findings)
        self.recommendationsChanged.emit()

        self._context_summary = dict(result.context_summary)
        self.contextSummaryChanged.emit()

        self._last_updated = result.generated_at.astimezone(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        self.lastUpdatedChanged.emit()

        self._publish_compliance_summary(result)

    def _publish_compliance_summary(self, result: ComplianceAuditResult) -> None:
        provider = self._telemetry_provider
        if provider is None:
            return
        callback = getattr(provider, "updateComplianceSummary", None)
        if callback is None:
            return
        payload: MutableMapping[str, object] = dict(self._summary)
        payload["contextSummary"] = dict(self._context_summary)
        payload["findings"] = [self._finding_to_dict(item) for item in result.findings]
        try:
            callback(payload)  # type: ignore[misc]
        except Exception:  # pragma: no cover - nie blokuj UI w przypadku błędu emitera
            pass

    def _build_summary(self, result: ComplianceAuditResult) -> dict[str, object]:
        findings = tuple(result.findings)
        return {
            "passed": result.passed,
            "kycStatus": self._severity_for_prefix(findings, "KYC"),
            "amlStatus": self._severity_for_prefix(findings, "AML"),
            "transactionStatus": self._severity_for_prefix(findings, "TX"),
            "totalFindings": len(findings),
        }

    def _finding_to_dict(self, finding: ComplianceFinding) -> dict[str, object]:
        payload: MutableMapping[str, object] = {
            "ruleId": finding.rule_id,
            "severity": finding.severity,
            "message": finding.message,
        }
        if finding.metadata:
            payload["metadata"] = dict(finding.metadata)
        return payload

    def _build_recommendations(self, findings: Sequence[ComplianceFinding]) -> list[str]:
        recommendations: list[str] = []
        for finding in findings:
            text = finding.message
            if finding.metadata:
                details = ", ".join(f"{key}: {value}" for key, value in sorted(finding.metadata.items()))
                if details:
                    text = f"{text} ({details})"
            recommendations.append(text)
        if not recommendations:
            recommendations.append("Brak zaleceń - konfiguracja zgodna")
        return recommendations

    def _severity_for_prefix(
        self, findings: Sequence[ComplianceFinding], prefix: str
    ) -> str:
        levels = {"critical": 4, "high": 3, "error": 3, "warning": 2, "info": 1, "ok": 0}
        best_level = 0
        best_name = "ok"
        for finding in findings:
            if not finding.rule_id.upper().startswith(prefix.upper()):
                continue
            severity = str(finding.severity or "warning").lower()
            numeric = levels.get(severity, 2)
            if numeric > best_level:
                best_level = numeric
                best_name = severity
        if best_level == 0:
            return "ok"
        return best_name

    def _set_busy(self, value: bool) -> None:
        if self._busy == value:
            return
        self._busy = value
        self.busyChanged.emit()

    def _set_error(self, message: str) -> None:
        if self._error_message == message:
            return
        self._error_message = message
        self.errorMessageChanged.emit()


__all__ = ["ComplianceController"]
