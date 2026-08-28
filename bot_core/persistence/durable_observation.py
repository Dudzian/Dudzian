"""Ephemeral observation derived from one verified durable StateStore snapshot.

The carrier in this module records local integrity facts only.  It is rebuildable,
is not persisted, and grants no domain, runtime, restore, or LIVE authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state_store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class DurableStateObservation:
    """Immutable local result of the complete S2C snapshot verification gates."""

    account_id: str
    device_installation_id: str
    state_store_schema_version: int
    state_store_identity_fingerprint_sha256: str
    environment: str
    protected_freshness_generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    history_tail_fingerprint_sha256: str
    durable_confirmed: bool
    authoritative_history_integrity: bool
    current_commit: bool


def observe_verified_durable_state(
    store: SQLiteStateStore,
) -> DurableStateObservation | None:
    """Derive an observation from exactly one coherent, verified snapshot.

    An empty store has no durable commit to observe.  Verification errors are
    deliberately allowed to propagate fail-closed from ``read_verified_snapshot``.
    """

    snapshot = store.read_verified_snapshot()
    if snapshot is None:
        return None
    metadata = snapshot.metadata
    return DurableStateObservation(
        account_id=metadata.account_id,
        device_installation_id=metadata.device_installation_id,
        state_store_schema_version=metadata.state_store_schema_version,
        state_store_identity_fingerprint_sha256=(metadata.state_store_identity_fingerprint_sha256),
        environment=metadata.environment,
        protected_freshness_generation=metadata.protected_freshness_generation,
        state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        transaction_fingerprint_sha256=metadata.transaction_fingerprint_sha256,
        history_tail_fingerprint_sha256=metadata.history_tail_fingerprint_sha256,
        durable_confirmed=True,
        authoritative_history_integrity=True,
        current_commit=True,
    )
