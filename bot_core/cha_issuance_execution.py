"""Installed executable issuance routes, currently limited to local operations.

Runtime executes through this table rather than accepting a claimed capability
list. Installation of any additional route, or substitution of a local handler,
closes the local-only boundary. Historical SQLite proofs never consult it.
There is no transport implementation or caller-selected installation API.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from bot_core.cha_attempt_store import AttemptConflictError, SQLiteCHAAttemptStore
from bot_core.cha_root_proof_signing_custody import (
    LocalCHARequesterSigningCustody,
    LocalPreaccountClaimantAuthorizationCustody,
)

_LOCAL_OPERATIONS = MappingProxyType(
    {
        "RESERVE": (SQLiteCHAAttemptStore, SQLiteCHAAttemptStore.reserve_or_resolve_attempt_id),
        "PREPARE_SIGNATURE": (SQLiteCHAAttemptStore, SQLiteCHAAttemptStore.prepare_signature),
        "REQUESTER_SIGNATURE": (
            LocalCHARequesterSigningCustody,
            LocalCHARequesterSigningCustody.sign_issuance_request,
        ),
        "CLAIMANT_AUTHORIZATION": (
            LocalPreaccountClaimantAuthorizationCustody,
            LocalPreaccountClaimantAuthorizationCustody.authorize_entitlement_claim,
        ),
        "PERSIST_SIGNATURE": (SQLiteCHAAttemptStore, SQLiteCHAAttemptStore.persist_signature),
        "FINALIZE": (SQLiteCHAAttemptStore, SQLiteCHAAttemptStore.finalize_attempt),
    }
)
# This is the actual dispatch table consumed below, not an asserted grant list.
_INSTALLED_OPERATIONS = MappingProxyType(dict(_LOCAL_OPERATIONS))
_LOCAL_GRANT_BY_OPERATION = MappingProxyType(
    {
        "RESERVE": "RESERVE",
        "PREPARE_SIGNATURE": "SIGN",
        "REQUESTER_SIGNATURE": "SIGN",
        "CLAIMANT_AUTHORIZATION": "SIGN",
        "PERSIST_SIGNATURE": "SIGN",
        "FINALIZE": "FINALIZE",
    }
)


def _qualified_operations() -> dict:
    if type(_INSTALLED_OPERATIONS) not in {dict, type(_LOCAL_OPERATIONS)}:
        raise AttemptConflictError("PRE_SEND_LOCAL_ONLY_AUTHORITY_REQUIRED")
    installed = dict(_INSTALLED_OPERATIONS)
    if installed.keys() != _LOCAL_GRANT_BY_OPERATION.keys() or any(
        type(installed[name]) is not tuple
        or len(installed[name]) != 2
        or installed[name][0] is not expected[0]
        or installed[name][1] is not expected[1]
        for name, expected in _LOCAL_OPERATIONS.items()
    ):
        raise AttemptConflictError("PRE_SEND_LOCAL_ONLY_AUTHORITY_REQUIRED")
    return installed


def require_local_only_execution() -> frozenset[str]:
    return frozenset(_LOCAL_GRANT_BY_OPERATION[name] for name in _qualified_operations())


def execute_local_issuance(operation: str, receiver: object, *args: Any, **kwargs: Any) -> Any:
    """Dispatch only a qualified installed local handler to its exact receiver."""
    installed = _qualified_operations()
    if type(operation) is not str or operation not in installed:
        raise AttemptConflictError("LOCAL_ISSUANCE_OPERATION_REQUIRED")
    expected, handler = installed[operation]
    if type(receiver) is not expected:
        raise AttemptConflictError("EXACT_LOCAL_ISSUANCE_EXECUTOR_REQUIRED")
    return handler(receiver, *args, **kwargs)
