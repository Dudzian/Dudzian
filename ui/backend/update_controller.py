"""Kontroler importu pakietów `.kbot` na potrzeby dialogu aktualizacji."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.update.offline_updater import ImportedOfflinePackage, OfflinePackageError, import_kbot_package
from .logging import get_update_logger

LOGGER = get_update_logger()


class OfflineUpdateController(QObject):
    """Zapewnia QML funkcje importu pakietów aktualizacji."""

    busyChanged = Signal()
    statusMessageIdChanged = Signal()
    statusDetailsChanged = Signal()
    packagesDirectoryChanged = Signal()
    fingerprintChanged = Signal()
    lastPackageChanged = Signal()
    importCompleted = Signal(str)
    importFailed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._busy = False
        self._status_message_id = "updateDialog.status.idle"
        self._status_details = ""
        self._packages_directory = Path("var/updates/packages")
        self._fingerprint: str | None = None
        self._signing_key: bytes | None = None
        self._last_package_id = ""

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

    @Property(str, notify=packagesDirectoryChanged)
    def packagesDirectory(self) -> str:  # type: ignore[override]
        return str(self._packages_directory)

    @packagesDirectory.setter
    def packagesDirectory(self, value: str) -> None:  # type: ignore[override]
        path = Path(value).expanduser()
        if self._packages_directory == path:
            return
        self._packages_directory = path
        self.packagesDirectoryChanged.emit()

    @Property(str, notify=fingerprintChanged)
    def fingerprint(self) -> str:  # type: ignore[override]
        return self._fingerprint or ""

    @fingerprint.setter
    def fingerprint(self, value: str) -> None:  # type: ignore[override]
        normalized = value.strip() or None
        if normalized == self._fingerprint:
            return
        self._fingerprint = normalized
        self.fingerprintChanged.emit()

    @Property(str, notify=lastPackageChanged)
    def lastPackageId(self) -> str:  # type: ignore[override]
        return self._last_package_id

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

    def _set_last_package(self, package: ImportedOfflinePackage | None) -> None:
        identifier = ""
        if package is not None:
            identifier = f"{package.manifest.package_id}-{package.manifest.version}"
        if identifier != self._last_package_id:
            self._last_package_id = identifier
            self.lastPackageChanged.emit()

    # ------------------------------------------------------------------
    @Slot(str)
    def setSigningKey(self, key: str) -> None:
        """Zapisuje klucz HMAC jako bajty UTF-8."""

        encoded = key.encode("utf-8").strip() if key else b""
        self._signing_key = encoded or None

    @Slot(str, result=bool)
    def importPackage(self, package_path: str) -> bool:
        """Importuje wskazany pakiet `.kbot` do katalogu aktualizacji."""

        if self._busy:
            return False

        candidate = Path(package_path).expanduser()
        if not candidate.exists():
            self._set_status("updateDialog.error.io", f"Brak pliku {candidate}")
            self.importFailed.emit(f"Brak pliku {candidate}")
            return False

        self._set_busy(True)
        try:
            result = import_kbot_package(
                candidate,
                self._packages_directory,
                expected_fingerprint=self._fingerprint,
                hmac_key=self._signing_key,
            )
        except OfflinePackageError as exc:
            LOGGER.warning("Weryfikacja pakietu %s nie powiodła się: %s", candidate, exc)
            self._set_status("updateDialog.error.validation", str(exc))
            self.importFailed.emit(str(exc))
            self._set_last_package(None)
            return False
        except OSError as exc:  # pragma: no cover - operacje IO
            LOGGER.error("Błąd IO podczas importu pakietu %s: %s", candidate, exc)
            self._set_status("updateDialog.error.io", str(exc))
            self.importFailed.emit(str(exc))
            self._set_last_package(None)
            return False
        except Exception as exc:  # pragma: no cover - defensywnie
            LOGGER.exception("Nieoczekiwany błąd podczas importu pakietu %s", candidate)
            self._set_status("updateDialog.error.unexpected", str(exc))
            self.importFailed.emit(str(exc))
            self._set_last_package(None)
            return False
        finally:
            self._set_busy(False)

        self._set_last_package(result)
        details = f"Zainstalowano {result.manifest.package_id} {result.manifest.version}"
        self._set_status("updateDialog.status.success", details)
        self.importCompleted.emit(result.target_directory.as_posix())
        return True


__all__ = ["OfflineUpdateController"]
