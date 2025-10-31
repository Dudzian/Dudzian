"""Kontroler udostępniający raport audytu licencji dla QML."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.security import generate_license_audit_report


class LicenseAuditController(QObject):
    """Zapewnia interfejs do przeglądania i eksportu raportów licencyjnych."""

    busyChanged = Signal()
    summaryChanged = Signal()
    activationsChanged = Signal()
    warningsChanged = Signal()
    statusDocumentChanged = Signal()
    lastUpdatedChanged = Signal()
    errorMessageChanged = Signal()
    statusPathChanged = Signal()
    auditLogPathChanged = Signal()
    exportCompleted = Signal(str, str)
    exportFailed = Signal(str)

    def __init__(
        self,
        *,
        status_path: str | Path | None = None,
        audit_log_path: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._busy = False
        self._summary: dict[str, object] = {}
        self._activations: list[dict[str, object]] = []
        self._warnings: list[str] = []
        self._status_document: dict[str, object] = {}
        self._last_updated = ""
        self._error_message = ""
        self._status_path = Path(status_path).expanduser() if status_path else Path("var/security/license_status.json")
        self._audit_log_path = Path(audit_log_path).expanduser() if audit_log_path else Path("logs/security_admin.log")

    # ------------------------------------------------------------------
    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:  # type: ignore[override]
        return self._busy

    @Property("QVariantMap", notify=summaryChanged)
    def summary(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._summary)

    @Property("QVariantList", notify=activationsChanged)
    def activations(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._activations)

    @Property("QStringList", notify=warningsChanged)
    def warnings(self) -> list[str]:  # type: ignore[override]
        return list(self._warnings)

    @Property("QVariantMap", notify=statusDocumentChanged)
    def statusDocument(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._status_document)

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(str, notify=statusPathChanged)
    def statusPath(self) -> str:  # type: ignore[override]
        return str(self._status_path)

    @statusPath.setter
    def statusPath(self, value: str) -> None:  # type: ignore[override]
        path = Path(value).expanduser()
        if path == self._status_path:
            return
        self._status_path = path
        self.statusPathChanged.emit()

    @Property(str, notify=auditLogPathChanged)
    def auditLogPath(self) -> str:  # type: ignore[override]
        return str(self._audit_log_path)

    @auditLogPath.setter
    def auditLogPath(self, value: str) -> None:  # type: ignore[override]
        path = Path(value).expanduser()
        if path == self._audit_log_path:
            return
        self._audit_log_path = path
        self.auditLogPathChanged.emit()

    # ------------------------------------------------------------------
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

    def _update_from_report(self, report) -> None:
        self._summary = dict(report.summary.to_dict())
        self._activations = [record.to_dict() for record in report.activations]
        self._warnings = list(report.warnings)
        self._status_document = dict(report.status_document) if report.status_document else {}
        self._last_updated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.summaryChanged.emit()
        self.activationsChanged.emit()
        self.warningsChanged.emit()
        self.statusDocumentChanged.emit()
        self.lastUpdatedChanged.emit()

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshReport(self) -> bool:
        if self._busy:
            return False
        self._set_busy(True)
        try:
            report = generate_license_audit_report(
                status_path=self._status_path,
                audit_log_path=self._audit_log_path,
            )
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            self._set_error(str(exc))
            self._set_busy(False)
            return False

        self._set_error("")
        self._update_from_report(report)
        self._set_busy(False)
        return True

    @Slot(str, str, result=bool)
    def exportReport(self, directory: str, basename: str) -> bool:
        if not directory:
            self._set_error("Nie wskazano katalogu eksportu")
            self.exportFailed.emit("Nie wskazano katalogu eksportu")
            return False

        target_dir = Path(directory).expanduser()
        basename_normalized = basename.strip() or "license_audit"

        if self._busy:
            return False

        # Zapewnij aktualne dane przed eksportem
        if not self._summary:
            if not self.refreshReport():
                self.exportFailed.emit(self._error_message or "Nie udało się odświeżyć raportu")
                return False

        self._set_busy(True)
        try:
            report = generate_license_audit_report(
                status_path=self._status_path,
                audit_log_path=self._audit_log_path,
            )

            target_dir.mkdir(parents=True, exist_ok=True)
            json_path = target_dir / f"{basename_normalized}.json"
            markdown_path = target_dir / f"{basename_normalized}.md"

            json_payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
            json_path.write_text(json_payload, encoding="utf-8")
            markdown_path.write_text(report.to_markdown(), encoding="utf-8")

            self._update_from_report(report)
            self._set_error("")
            self.exportCompleted.emit(str(json_path), str(markdown_path))
            return True
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            message = str(exc)
            self._set_error(message)
            self.exportFailed.emit(message)
            return False
        finally:
            self._set_busy(False)


__all__ = ["LicenseAuditController"]

