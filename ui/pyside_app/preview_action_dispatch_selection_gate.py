"""Pure-Python BLOK E selection preview gate contract.

This helper describes the read-only gate that must be satisfied before a future
QML preview selection call can be enabled.  It returns deterministic QML-safe
plain containers only; it does not import PySide/QML, wire handlers, start
runtime loops, dispatch commands, execute lifecycle commands, generate or submit
orders, read accounts or secrets, fetch live/testnet data, export files, access
cloud paths, or perform I/O.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_action_dispatch_contract import RUNTIME_MODE

SELECTION_PREVIEW_GATE_SCHEMA_VERSION: Final[str] = (
    "paper_runtime_action_dispatch_selection_preview_gate.v1"
)
SELECTION_PREVIEW_GATE_KIND: Final[str] = "block_e_selection_preview_gate_contract"
SELECTION_PREVIEW_GATE_STATUS: Final[str] = "locked_read_only_method_calls_disabled"
_SELECTION_PREVIEW_GATE_OPERATOR_MESSAGE: Final[str] = (
    "Selection preview gate is locked/read-only for this step; next step may enable "
    "previewSelectAction only after tests, while current QML method calls and all "
    "execution/order/lifecycle paths remain disabled."
)

_ALLOWED_NEXT_QML_METHODS: Final[tuple[dict[str, str], ...]] = (
    {
        "method": "previewSelectAction",
        "availability": "next_step_only_not_active_now",
        "condition": "enable only after source guards, QML no-handler audit, and smoke tests pass",
    },
)
_BLOCKED_CURRENT_QML_METHODS: Final[tuple[str, ...]] = (
    "previewSelectAction",
    "previewSelectSourceControl",
    "resetPreviewSelection",
)
_REQUIRED_OPERATOR_GATE: Final[tuple[str, ...]] = (
    "operator can see locked/read-only gate state",
    "operator can see method calls allowed now are false",
    "operator can see execution, order submission, and lifecycle execution are false",
    "operator can see next-step previewSelectAction requires tests before activation",
)
_REQUIRED_SOURCE_GUARDS: Final[tuple[str, ...]] = (
    "QML surface remains diagnostic text only",
    "no Button/IconButton activation is added for this gate",
    "no click, mouse-area, connection, tap, key, or shortcut handlers are added",
    "no QML bridge method calls are added in this step",
    "no runtime loop, command dispatch, lifecycle execution, order submission, live/testnet, account, secrets, or export path is added",
)


def build_paper_runtime_action_dispatch_selection_preview_gate() -> dict[str, Any]:
    """Return the deterministic QML-safe selection preview gate payload."""

    return {
        "gate_schema_version": SELECTION_PREVIEW_GATE_SCHEMA_VERSION,
        "gate_kind": SELECTION_PREVIEW_GATE_KIND,
        "gate_status": SELECTION_PREVIEW_GATE_STATUS,
        "selection_preview_allowed_in_next_step": True,
        "qml_method_calls_allowed_now": False,
        "execution_allowed": False,
        "execution_performed": False,
        "order_submission_allowed": False,
        "lifecycle_execution_allowed": False,
        "live_mode_allowed": False,
        "testnet_mode_allowed": False,
        "account_fetch_allowed": False,
        "secrets_export_allowed": False,
        "cloud_export_allowed": False,
        "runtime_mode": RUNTIME_MODE,
        "paper_only": True,
        "local_only": True,
        "allowed_next_qml_methods": [dict(item) for item in _ALLOWED_NEXT_QML_METHODS],
        "blocked_current_qml_methods": list(_BLOCKED_CURRENT_QML_METHODS),
        "required_operator_gate": list(_REQUIRED_OPERATOR_GATE),
        "required_source_guards": list(_REQUIRED_SOURCE_GUARDS),
        "operator_message": _SELECTION_PREVIEW_GATE_OPERATOR_MESSAGE,
    }


__all__ = [
    "SELECTION_PREVIEW_GATE_KIND",
    "SELECTION_PREVIEW_GATE_SCHEMA_VERSION",
    "SELECTION_PREVIEW_GATE_STATUS",
    "build_paper_runtime_action_dispatch_selection_preview_gate",
]
