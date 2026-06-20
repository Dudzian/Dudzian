"""Static-local BLOK D paper action dispatch catalog view model.

This module builds a deterministic, immutable catalog of paper runtime action
controls that a future UI binding may display.  It composes the existing action
intent contract and audit envelope only; it does not import PySide/QML, wire
handlers, start runtime loops, dispatch commands, execute lifecycle commands,
generate or submit orders, read accounts, fetch live/testnet data,
or access external paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from ui.pyside_app import preview_action_dispatch_audit as _audit
from ui.pyside_app.preview_action_dispatch_audit import (
    ACCEPTED_INTENT_NOT_EXECUTED,
    PaperRuntimeActionDispatchAuditEnvelope,
)
from ui.pyside_app.preview_action_dispatch_contract import (
    ALLOWED_PAPER_RUNTIME_ACTIONS,
    RUNTIME_MODE,
    FrozenMapping,
    build_paper_runtime_action_dispatch_contract,
)

CATALOG_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_catalog.v1"
CATALOG_KIND: Final[str] = "block_d_ui_safe_paper_runtime_action_catalog"
SOURCE_PANEL: Final[str] = "runtimeSessionControlPanel"

_ACTION_VIEW_METADATA: Final[Mapping[str, Mapping[str, str]]] = FrozenMapping(
    {
        "paper_runtime_start_requested": FrozenMapping(
            {
                "label": "Request paper preview start",
                "description": (
                    "Local paper preview intent only; records a non-executing start request."
                ),
                "source_control": "paper-runtime-start",
            }
        ),
        "paper_runtime_stop_requested": FrozenMapping(
            {
                "label": "Request paper preview stop",
                "description": (
                    "Local paper preview intent only; records a non-executing stop request."
                ),
                "source_control": "paper-runtime-stop",
            }
        ),
        "paper_runtime_pause_requested": FrozenMapping(
            {
                "label": "Request paper preview pause",
                "description": (
                    "Local paper preview intent only; records a non-executing pause request."
                ),
                "source_control": "paper-runtime-pause",
            }
        ),
        "paper_runtime_resume_requested": FrozenMapping(
            {
                "label": "Request paper preview resume",
                "description": (
                    "Local paper preview intent only; records a non-executing resume request."
                ),
                "source_control": "paper-runtime-resume",
            }
        ),
        "paper_runtime_snapshot_refresh_requested": FrozenMapping(
            {
                "label": "Request paper preview snapshot refresh",
                "description": (
                    "Local paper preview intent only; records a non-executing snapshot refresh request."
                ),
                "source_control": "paper-runtime-snapshot-refresh",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class PaperRuntimeActionCatalogItem:
    """Immutable UI-safe view model row for one paper action intent."""

    action: str
    normalized_action: str
    label: str
    description: str
    source_panel: str
    source_control: str
    requires_operator_confirmation: bool
    safe_to_bind_from_ui: bool
    execution_allowed: bool
    execution_performed: bool
    paper_only: bool
    local_only: bool
    runtime_mode: str
    blocked_reason: str
    refusal_reason: str
    audit_status: str
    audit_envelope: PaperRuntimeActionDispatchAuditEnvelope
    boundary_checks: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class PaperRuntimeActionDispatchCatalog:
    """Immutable UI-safe catalog for future paper action controls."""

    schema_version: str
    catalog_kind: str
    runtime_mode: str
    paper_only: bool
    local_only: bool
    execution_allowed: bool
    execution_performed: bool
    actions: tuple[PaperRuntimeActionCatalogItem, ...]
    action_count: int
    allowed_actions: tuple[str, ...]
    boundary_checks: Mapping[str, bool]
    safe_to_bind_from_ui: bool


def build_paper_runtime_action_dispatch_catalog() -> PaperRuntimeActionDispatchCatalog:
    """Build the deterministic paper action catalog without executing actions."""

    items = tuple(_build_catalog_item(action) for action in ALLOWED_PAPER_RUNTIME_ACTIONS)
    boundary_checks = _build_catalog_boundary_checks(items)
    return PaperRuntimeActionDispatchCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        catalog_kind=CATALOG_KIND,
        runtime_mode=RUNTIME_MODE,
        paper_only=True,
        local_only=True,
        execution_allowed=False,
        execution_performed=False,
        actions=items,
        action_count=len(items),
        allowed_actions=tuple(ALLOWED_PAPER_RUNTIME_ACTIONS),
        boundary_checks=boundary_checks,
        safe_to_bind_from_ui=all(item.safe_to_bind_from_ui for item in items),
    )


def _build_catalog_item(action: str) -> PaperRuntimeActionCatalogItem:
    dispatch_evidence = build_paper_runtime_action_dispatch_contract(action)
    metadata = _ACTION_VIEW_METADATA[action]
    audit_envelope = _audit.build_paper_runtime_action_dispatch_audit_envelope(
        dispatch_evidence,
        source_panel=SOURCE_PANEL,
        source_control=metadata["source_control"],
        operator_confirmation=dispatch_evidence.requires_operator_confirmation,
    )
    return PaperRuntimeActionCatalogItem(
        action=action,
        normalized_action=dispatch_evidence.normalized_action,
        label=metadata["label"],
        description=metadata["description"],
        source_panel=SOURCE_PANEL,
        source_control=metadata["source_control"],
        requires_operator_confirmation=dispatch_evidence.requires_operator_confirmation,
        safe_to_bind_from_ui=audit_envelope.safe_to_bind_from_ui,
        execution_allowed=False,
        execution_performed=False,
        paper_only=audit_envelope.paper_only,
        local_only=audit_envelope.local_only,
        runtime_mode=audit_envelope.runtime_mode,
        blocked_reason=audit_envelope.blocked_reason,
        refusal_reason=audit_envelope.refusal_reason,
        audit_status=audit_envelope.audit_status,
        audit_envelope=audit_envelope,
        boundary_checks=FrozenMapping(dict(audit_envelope.boundary_checks)),
    )


def _build_catalog_boundary_checks(
    items: tuple[PaperRuntimeActionCatalogItem, ...],
) -> Mapping[str, bool]:
    return FrozenMapping(
        {
            "allowed_actions_complete": tuple(item.action for item in items)
            == ALLOWED_PAPER_RUNTIME_ACTIONS,
            "accepted_intents_only": all(
                item.audit_status == ACCEPTED_INTENT_NOT_EXECUTED for item in items
            ),
            "paper_only": all(item.paper_only for item in items),
            "local_only": all(item.local_only for item in items),
            "execution_disabled": True,
            "execution_not_performed": True,
            "safe_to_bind_from_ui": all(item.safe_to_bind_from_ui for item in items),
            "runtime_mode_paper": all(item.runtime_mode == RUNTIME_MODE for item in items),
            "source_metadata_static": all(
                item.source_panel == SOURCE_PANEL and bool(item.source_control) for item in items
            ),
        }
    )


__all__ = [
    "CATALOG_KIND",
    "CATALOG_SCHEMA_VERSION",
    "SOURCE_PANEL",
    "PaperRuntimeActionCatalogItem",
    "PaperRuntimeActionDispatchCatalog",
    "build_paper_runtime_action_dispatch_catalog",
]
