"""Controlled context-property registration preflight for the BLOK D Qt bridge.

The helper registers the thin QtCore preview bridge on a caller-provided
context-like object only.  It does not create an engine, wire UI handlers, start
runtime work, dispatch actions, generate orders, read environment variables,
perform I/O, or perform network calls.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_action_dispatch_contract import RUNTIME_MODE
from ui.pyside_app.preview_action_dispatch_qt_bridge import (
    QT_BRIDGE_KIND,
    PaperRuntimeActionDispatchQtBridge,
)

PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY: Final[str] = (
    "paperRuntimeActionDispatchBridge"
)
QT_BRIDGE_REGISTRATION_SCHEMA_VERSION: Final[str] = (
    "paper_runtime_action_dispatch_qt_bridge_registration.v1"
)
QT_BRIDGE_REGISTRATION_KIND: Final[str] = "block_d_controlled_qt_bridge_context_preflight"
_MISSING_SET_CONTEXT_PROPERTY: Final[str] = "missing_set_context_property"
_INVALID_CONTEXT_PROPERTY_NAME: Final[str] = "invalid_context_property_name"


def register_paper_runtime_action_dispatch_qt_bridge(
    context: object,
    *,
    property_name: object = PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY,
    bridge: PaperRuntimeActionDispatchQtBridge | None = None,
) -> dict[str, Any]:
    """Register the paper preview Qt bridge on a supplied context-like object.

    The registration is intentionally limited to a single
    ``context.setContextProperty(name, value)`` call when inputs are valid.  All
    returned evidence is plain, deterministic, copy-safe Python data.
    """

    normalized_property_name = _normalize_property_name(property_name)
    if not normalized_property_name:
        return _registration_evidence(
            context_property_name="",
            bridge=bridge,
            registered=False,
            registration_performed=False,
            blocked_reason=_INVALID_CONTEXT_PROPERTY_NAME,
        )

    set_context_property = getattr(context, "setContextProperty", None)
    if not callable(set_context_property):
        return _registration_evidence(
            context_property_name=normalized_property_name,
            bridge=bridge,
            registered=False,
            registration_performed=False,
            blocked_reason=_MISSING_SET_CONTEXT_PROPERTY,
        )

    bridge_to_register = bridge if bridge is not None else PaperRuntimeActionDispatchQtBridge()
    set_context_property(normalized_property_name, bridge_to_register)
    return _registration_evidence(
        context_property_name=normalized_property_name,
        bridge=bridge_to_register,
        registered=True,
        registration_performed=True,
        blocked_reason="",
    )


def _normalize_property_name(property_name: object) -> str:
    if not isinstance(property_name, str):
        return ""
    return property_name.strip()


def _registration_evidence(
    *,
    context_property_name: str,
    bridge: PaperRuntimeActionDispatchQtBridge | None,
    registered: bool,
    registration_performed: bool,
    blocked_reason: str,
) -> dict[str, Any]:
    snapshot = (
        bridge.snapshot if bridge is not None else PaperRuntimeActionDispatchQtBridge().snapshot
    )
    boundary_checks = {
        "context_property_name_valid": bool(context_property_name),
        "context_like_set_context_property_available": registered
        or not blocked_reason == _MISSING_SET_CONTEXT_PROPERTY,
        "registration_fail_closed": not registered,
        "registration_preview_only": True,
        "qml_engine_not_touched": True,
        "qml_files_not_changed": True,
        "execution_disabled": True,
        "execution_not_performed": True,
        "paper_only": True,
        "local_only": True,
    }
    if blocked_reason:
        boundary_checks[blocked_reason] = True

    return {
        "schema_version": QT_BRIDGE_REGISTRATION_SCHEMA_VERSION,
        "registration_kind": QT_BRIDGE_REGISTRATION_KIND,
        "context_property_name": context_property_name,
        "bridge_kind": QT_BRIDGE_KIND,
        "registered": bool(registered),
        "registration_performed": bool(registration_performed),
        "blocked_reason": blocked_reason,
        "qml_engine_touched": False,
        "qml_files_changed": False,
        "execution_allowed": False,
        "execution_performed": False,
        "runtime_mode": RUNTIME_MODE,
        "paper_only": True,
        "local_only": True,
        "snapshot": deepcopy(snapshot),
        "boundary_checks": boundary_checks,
    }


__all__ = [
    "PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY",
    "QT_BRIDGE_REGISTRATION_KIND",
    "QT_BRIDGE_REGISTRATION_SCHEMA_VERSION",
    "register_paper_runtime_action_dispatch_qt_bridge",
]
