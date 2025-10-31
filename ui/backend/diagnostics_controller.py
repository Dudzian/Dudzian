"""Kontroler generowania paczek diagnostycznych na potrzeby zgłoszeń serwisowych."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.support.diagnostics import DiagnosticsError, DiagnosticsPackage, create_diagnostics_package
from .logging import get_support_logger

LOGGER = get_support_logger()


class DiagnosticsController(QObject):
    """Zapewnia QML możliwość eksportu paczki diagnostycznej wraz z opisem zgłoszenia."""

    busyChanged = Signal()
    statusMessageIdChanged = Signal()
    statusDetailsChanged = Signal()
    outputDirectoryChanged = Signal()
    descriptionChanged = Signal()
    lastArchiveChanged = Signal()
    baseDirectoryChanged = Signal()
    exportCompleted = Signal(str)
    exportFailed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._busy = False
        self._status_message_id = "ticketDialog.status.idle"
        self._status_details = ""
        self._output_directory = Path("logs/support/diagnostics")
        self._description = ""
        self._last_archive_path = ""
        self._base_directory = Path.cwd()

    # ------------------------------------------------------------------
    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:  # type: ignore[override]
        return self._busy

    @Property(str, notify=statusMessageIdChanged)
    def statusMessageId(self) -> str:  # type: ignore[override]
        return self._status_message_id

    @Property(str, notify=statusDetailsChanged)
    def statusDetails(self) -> str:  # type: ignore[override]
        return self._status_details

    @Property(str, notify=outputDirectoryChanged)
    def outputDirectory(self) -> str:  # type: ignore[override]
        return str(self._output_directory)

    @outputDirectory.setter
    def outputDirectory(self, value: str) -> None:  # type: ignore[override]
        candidate = Path(value).expanduser()
        if candidate == self._output_directory:
            return
        self._output_directory = candidate
        self.outputDirectoryChanged.emit()

    @Property(str, notify=descriptionChanged)
    def description(self) -> str:  # type: ignore[override]
        return self._description

    @description.setter
    def description(self, value: str) -> None:  # type: ignore[override]
        normalised = value.strip()
        if normalised == self._description:
            return
        self._description = normalised
        self.descriptionChanged.emit()

    @Property(str, notify=lastArchiveChanged)
    def lastArchivePath(self) -> str:  # type: ignore[override]
        return self._last_archive_path

    @Property(str, notify=baseDirectoryChanged)
    def baseDirectory(self) -> str:  # type: ignore[override]
        return str(self._base_directory)

    @baseDirectory.setter
    def baseDirectory(self, value: str) -> None:  # type: ignore[override]
        candidate = Path(value).expanduser()
        if candidate == self._base_directory:
            return
        self._base_directory = candidate
        self.baseDirectoryChanged.emit()

    # ------------------------------------------------------------------
    def _set_busy(self, value: bool) -> None:
        if self._busy == value:
            return
        self._busy = value
        self.busyChanged.emit()

    def _set_status(self, message_id: str, details: str = "") -> None:
        if self._status_message_id != message_id:
            self._status_message_id = message_id
            self.statusMessageIdChanged.emit()
        if self._status_details != details:
            self._status_details = details
            self.statusDetailsChanged.emit()

    def _set_last_archive(self, package: DiagnosticsPackage | None) -> None:
        path_value = package.archive_path.as_posix() if package else ""
        if path_value == self._last_archive_path:
            return
        self._last_archive_path = path_value
        self.lastArchiveChanged.emit()

    # ------------------------------------------------------------------
    @Slot(str)
    def setDescription(self, value: str) -> None:
        self.description = value

    @Slot(str)
    def setOutputDirectory(self, value: str) -> None:
        self.outputDirectory = value

    @Slot(str)
    def setBaseDirectory(self, value: str) -> None:
        self.baseDirectory = value

    @Slot(result=bool)
    def generateDiagnostics(self) -> bool:
        if self._busy:
            return False

        self._set_busy(True)
        self._set_status("ticketDialog.status.processing", "")
        try:
            package = create_diagnostics_package(
                self._output_directory,
                base_path=self._base_directory,
                metadata={
                    "description": self._description,
                    "source": "ui.ticket_dialog",
                },
            )
        except DiagnosticsError as exc:
            LOGGER.error("Błąd tworzenia paczki diagnostycznej: %s", exc)
            self._set_status("ticketDialog.status.error", str(exc))
            self._set_last_archive(None)
            self.exportFailed.emit(str(exc))
            return False
        except Exception as exc:  # pragma: no cover - defensywne
            LOGGER.exception("Nieoczekiwany błąd podczas generowania diagnostyki")
            self._set_status("ticketDialog.status.error", str(exc))
            self._set_last_archive(None)
            self.exportFailed.emit(str(exc))
            return False
        finally:
            self._set_busy(False)

        LOGGER.info("Wygenerowano paczkę diagnostyczną %s", package.archive_path)
        self._set_last_archive(package)
        self._set_status("ticketDialog.status.success", package.archive_path.as_posix())
        self.exportCompleted.emit(package.archive_path.as_posix())
        return True


__all__ = ["DiagnosticsController"]
