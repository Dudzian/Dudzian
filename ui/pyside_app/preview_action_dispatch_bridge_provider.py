"""Source-only BLOK D paper action dispatch bridge provider preflight.

This provider is a pure-Python local state holder for a future PySide/QML bridge.
It only refreshes QML-safe preview snapshots from the bridge snapshot helper; it
never imports PySide/QML, registers context properties, wires handlers, starts
runtime loops, dispatches commands, executes lifecycle actions, generates or
submits orders, reads accounts or secrets, fetches live/testnet data, exports
files, accesses cloud paths, reads environment variables, performs I/O, or
performs network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    NO_SELECTION_STATUS,
    build_paper_runtime_action_dispatch_bridge_snapshot,
)

PROVIDER_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_bridge_provider.v1"
PROVIDER_KIND: Final[str] = "block_d_source_only_action_dispatch_bridge_provider"
PROVIDER_STATUS_READY: Final[str] = "source_only_preview_ready"


@dataclass(slots=True)
class PaperRuntimeActionDispatchBridgeProvider:
    """Deterministic source-only provider for non-executing preview snapshots."""

    _last_requested_action_or_control: object = None
    _last_source_panel: object = ""
    _last_source_control: object = ""
    _last_operator_confirmation: bool = False
    _last_operator_note: object = ""
    _last_reason: object = ""

    def snapshot(self) -> dict[str, Any]:
        """Return the current QML-safe provider snapshot without execution."""

        return build_paper_runtime_action_dispatch_bridge_provider_snapshot(
            self._last_requested_action_or_control,
            source_panel=self._last_source_panel,
            source_control=self._last_source_control,
            operator_confirmation=self._last_operator_confirmation,
            operator_note=self._last_operator_note,
            reason=self._last_reason,
        )

    def preview_select_action(
        self,
        requested_action: object,
        *,
        source_panel: object = "",
        source_control: object = "",
        operator_confirmation: bool = False,
        operator_note: object = "",
        reason: object = "",
    ) -> dict[str, Any]:
        """Store a local preview action request and return a fresh snapshot."""

        self._last_requested_action_or_control = requested_action
        self._last_source_panel = source_panel
        self._last_source_control = source_control
        self._last_operator_confirmation = bool(operator_confirmation)
        self._last_operator_note = operator_note
        self._last_reason = reason
        return self.snapshot()

    def preview_select_source_control(
        self,
        source_control: object,
        *,
        source_panel: object = "",
        operator_confirmation: bool = False,
        operator_note: object = "",
        reason: object = "",
    ) -> dict[str, Any]:
        """Store a local source-control preview request and return a fresh snapshot."""

        self._last_requested_action_or_control = None
        self._last_source_panel = source_panel
        self._last_source_control = source_control
        self._last_operator_confirmation = bool(operator_confirmation)
        self._last_operator_note = operator_note
        self._last_reason = reason
        return self.snapshot()

    def reset_preview_selection(self) -> dict[str, Any]:
        """Reset local preview state to no-selection and return a fresh snapshot."""

        self._last_requested_action_or_control = None
        self._last_source_panel = ""
        self._last_source_control = ""
        self._last_operator_confirmation = False
        self._last_operator_note = ""
        self._last_reason = ""
        return self.snapshot()


def build_paper_runtime_action_dispatch_bridge_provider_snapshot(
    requested_action_or_control: object = None,
    *,
    source_panel: object = "",
    source_control: object = "",
    operator_confirmation: bool = False,
    operator_note: object = "",
    reason: object = "",
) -> dict[str, Any]:
    """Build a source-only provider snapshot around the QML-safe bridge payload."""

    bridge_snapshot = build_paper_runtime_action_dispatch_bridge_snapshot(
        requested_action_or_control,
        source_panel=source_panel,
        source_control=source_control,
        operator_confirmation=operator_confirmation,
        operator_note=operator_note,
        reason=reason,
    )
    selected_result = dict(bridge_snapshot["selected_result"])
    bridge_snapshot.update(
        {
            "provider_schema_version": PROVIDER_SCHEMA_VERSION,
            "provider_kind": PROVIDER_KIND,
            "provider_status": PROVIDER_STATUS_READY,
            "last_requested_action_or_control": selected_result["requested_action_or_control"],
            "last_result_status": selected_result["result_status"],
            "provider_execution_allowed": False,
            "provider_execution_performed": False,
        }
    )
    if bridge_snapshot["last_result_status"] == NO_SELECTION_STATUS:
        bridge_snapshot["last_requested_action_or_control"] = None
    return bridge_snapshot


__all__ = [
    "PROVIDER_KIND",
    "PROVIDER_SCHEMA_VERSION",
    "PROVIDER_STATUS_READY",
    "PaperRuntimeActionDispatchBridgeProvider",
    "build_paper_runtime_action_dispatch_bridge_provider_snapshot",
]
