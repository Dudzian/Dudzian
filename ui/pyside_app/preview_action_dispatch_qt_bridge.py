"""Thin PySide QtCore bridge for BLOK D paper action dispatch preview.

The bridge owns a local source-only provider and exposes only QVariant-compatible
plain snapshots for future QML binding.  It does not register itself in QML,
wire handlers, start runtime loops, dispatch commands, execute lifecycle
commands, generate or submit orders, read accounts or secrets, fetch
live/testnet data, export files, access cloud paths, read environment variables,
perform I/O, or perform network calls.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from PySide6.QtCore import QObject, Property, Signal, Slot

from ui.pyside_app.preview_action_dispatch_bridge_provider import (
    PaperRuntimeActionDispatchBridgeProvider,
)

QT_BRIDGE_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_qt_bridge.v1"
QT_BRIDGE_KIND: Final[str] = "block_d_thin_qtcore_action_dispatch_preview_bridge"


class PaperRuntimeActionDispatchQtBridge(QObject):
    """QObject adapter over the source-only paper dispatch preview provider."""

    snapshotChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._provider = PaperRuntimeActionDispatchBridgeProvider()

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self) -> dict[str, Any]:
        """Return a copy-safe plain provider snapshot for Qt/QML consumers."""

        return self._snapshot_payload(self._provider.snapshot())

    @Slot(str, result="QVariantMap")
    @Slot(str, bool, str, result="QVariantMap")
    def previewSelectAction(
        self,
        action: object,
        operatorConfirmation: bool = False,
        operatorNote: object = "",
    ) -> dict[str, Any]:
        """Preview-select an action without allowing or performing execution."""

        payload = self._provider.preview_select_action(
            action,
            operator_confirmation=bool(operatorConfirmation),
            operator_note=operatorNote,
        )
        self.snapshotChanged.emit()
        return self._snapshot_payload(payload)

    @Slot(str, bool, str, result="QVariantMap")
    def previewSelectSourceControl(
        self,
        sourceControl: object,
        operatorConfirmation: bool = False,
        operatorNote: object = "",
    ) -> dict[str, Any]:
        """Preview-select a source control without dispatching commands."""

        payload = self._provider.preview_select_source_control(
            sourceControl,
            operator_confirmation=bool(operatorConfirmation),
            operator_note=operatorNote,
        )
        self.snapshotChanged.emit()
        return self._snapshot_payload(payload)

    @Slot(result="QVariantMap")
    def resetPreviewSelection(self) -> dict[str, Any]:
        """Reset local preview state to no-selection and emit a local signal."""

        payload = self._provider.reset_preview_selection()
        self.snapshotChanged.emit()
        return self._snapshot_payload(payload)

    @staticmethod
    def _snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(snapshot)
        payload.update(
            {
                "qt_bridge_schema_version": QT_BRIDGE_SCHEMA_VERSION,
                "qt_bridge_kind": QT_BRIDGE_KIND,
                "qt_bridge_execution_allowed": False,
                "qt_bridge_execution_performed": False,
            }
        )
        return payload


__all__ = [
    "QT_BRIDGE_KIND",
    "QT_BRIDGE_SCHEMA_VERSION",
    "PaperRuntimeActionDispatchQtBridge",
]
