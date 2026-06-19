"""Static-local BLOK D UI action dispatch intention contract.

This module is intentionally pure: it only classifies requested UI intentions for
future paper runtime binding.  It does not import PySide/QML, start runtime
loops, execute lifecycle commands, generate or submit orders, read accounts or
secrets, fetch live/testnet data, export files, or access cloud paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Iterator
from typing import Final, Mapping, TypeVar

SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_contract.v1"
DISPATCH_KIND: Final[str] = "block_d_paper_runtime_ui_action_intention"
RUNTIME_MODE: Final[str] = "paper"

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class FrozenMapping(Mapping[_Key, _Value]):
    """Small immutable mapping that remains compatible with dataclasses.asdict."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[_Key, _Value]) -> None:
        self._items = tuple(values.items())
        self._lookup = dict(self._items)

    def __getitem__(self, key: _Key) -> _Value:
        return self._lookup[key]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self.items())!r})"


ALLOWED_PAPER_RUNTIME_ACTIONS: Final[tuple[str, ...]] = (
    "paper_runtime_start_requested",
    "paper_runtime_stop_requested",
    "paper_runtime_pause_requested",
    "paper_runtime_resume_requested",
    "paper_runtime_snapshot_refresh_requested",
)

_REJECTED_ACTION_CATEGORIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "unknown": ("unknown",),
        "live_mode": ("live", "prod", "production", "real_trading"),
        "testnet_mode": ("testnet", "sandbox"),
        "order_generation_submission": (
            "order",
            "submit",
            "submission",
            "generate_order",
            "order_intent",
            "fill",
        ),
        "account_balance_fetch": ("account", "balance", "fetch", "wallet", "portfolio_fetch"),
        "export_cloud_secrets": (
            "export",
            "cloud",
            "secret",
            "secrets",
            "credential",
            "credentials",
            "api_key",
        ),
        "invalid": ("empty", "none", "non_string"),
    }
)

_BOUNDARY_CHECK_NAMES: Final[tuple[str, ...]] = (
    "paper_only",
    "local_only",
    "execution_disabled",
    "execution_not_performed",
    "order_generation_disabled",
    "order_submission_disabled",
    "live_mode_blocked",
    "testnet_mode_blocked",
    "account_fetch_blocked",
    "secrets_blocked",
    "export_cloud_blocked",
    "qml_handler_absent",
    "runtime_loop_absent",
    "lifecycle_execution_absent",
)


@dataclass(frozen=True, slots=True)
class PaperRuntimeActionDispatchEvidence:
    """Immutable evidence for one requested UI-to-paper-runtime intention."""

    schema_version: str
    dispatch_kind: str
    requested_action: object
    normalized_action: str
    runtime_mode: str
    paper_only: bool
    local_only: bool
    execution_allowed: bool
    execution_performed: bool
    order_generation_allowed: bool
    order_submission_allowed: bool
    live_mode_allowed: bool
    testnet_mode_allowed: bool
    requires_operator_confirmation: bool
    refusal_reason: str
    blocked_reason: str
    safe_to_bind_from_ui: bool
    boundary_checks: Mapping[str, bool]
    allowed_actions: tuple[str, ...]
    rejected_actions: Mapping[str, tuple[str, ...]]


def build_paper_runtime_action_dispatch_contract(
    requested_action: object,
) -> PaperRuntimeActionDispatchEvidence:
    """Classify a UI action request without executing it.

    Allowed paper actions are safe to bind later, but execution remains disabled
    in BLOK D because this is only the action-dispatch intention contract.
    """

    normalized_action = _normalize_action(requested_action)
    refusal_reason = _refusal_reason(requested_action, normalized_action)
    safe_to_bind = refusal_reason == ""
    boundary_checks = _build_boundary_checks(safe_to_bind)
    return PaperRuntimeActionDispatchEvidence(
        schema_version=SCHEMA_VERSION,
        dispatch_kind=DISPATCH_KIND,
        requested_action=requested_action,
        normalized_action=normalized_action,
        runtime_mode=RUNTIME_MODE,
        paper_only=True,
        local_only=True,
        execution_allowed=False,
        execution_performed=False,
        order_generation_allowed=False,
        order_submission_allowed=False,
        live_mode_allowed=False,
        testnet_mode_allowed=False,
        requires_operator_confirmation=True,
        refusal_reason=refusal_reason,
        blocked_reason=refusal_reason,
        safe_to_bind_from_ui=safe_to_bind,
        boundary_checks=boundary_checks,
        allowed_actions=tuple(ALLOWED_PAPER_RUNTIME_ACTIONS),
        rejected_actions=FrozenMapping(
            {key: tuple(value) for key, value in _REJECTED_ACTION_CATEGORIES.items()}
        ),
    )


def _normalize_action(requested_action: object) -> str:
    if not isinstance(requested_action, str):
        return ""
    return requested_action.strip().lower()


def _refusal_reason(requested_action: object, normalized_action: str) -> str:
    if not isinstance(requested_action, str):
        return "invalid_action_non_string"
    if not normalized_action:
        return "invalid_action_empty"
    if normalized_action in ALLOWED_PAPER_RUNTIME_ACTIONS:
        return ""
    category = _rejected_category(normalized_action)
    return f"blocked_{category}"


def _rejected_category(normalized_action: str) -> str:
    for category, tokens in _REJECTED_ACTION_CATEGORIES.items():
        if category in {"unknown", "invalid"}:
            continue
        if any(token in normalized_action for token in tokens):
            return category
    return "unknown_action"


def _build_boundary_checks(safe_to_bind: bool) -> Mapping[str, bool]:
    values = {name: True for name in _BOUNDARY_CHECK_NAMES}
    values["allowlisted_action"] = safe_to_bind
    values["fail_closed"] = not safe_to_bind
    return FrozenMapping(values)


__all__ = [
    "ALLOWED_PAPER_RUNTIME_ACTIONS",
    "DISPATCH_KIND",
    "RUNTIME_MODE",
    "SCHEMA_VERSION",
    "PaperRuntimeActionDispatchEvidence",
    "build_paper_runtime_action_dispatch_contract",
]
