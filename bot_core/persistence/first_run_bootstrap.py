"""Durable M0.11 adapter for the pure M0.3 first-run bootstrap authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import sqlite3
from typing import cast

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    BootstrapTransitionResult,
    ConsumedBootstrapAuthority,
    CoreCurrentBootstrapState,
    FirstRunBootstrapAuthority,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningBoundary,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
    state_content_fingerprint,
    validate_bootstrap_state_transition,
)

from .lifecycle_records import persistence_record
from .state_store import SQLiteStateStore, StateStoreError, StateStoreMetadata


class DurableFirstRunBootstrapRegistry:
    """Resolve bootstrap acceptance and designation only from a verified StateStore."""

    def __init__(self, store: SQLiteStateStore) -> None:
        self._store = store

    @staticmethod
    def _decode(payload: object) -> CoreCurrentBootstrapState:
        if not isinstance(payload, Mapping):
            raise StateStoreError("bootstrap state payload is not a mapping")
        value = dict(payload)
        consumed = value.get("consumed_authorities")
        if not isinstance(consumed, (list, tuple)):
            raise StateStoreError("bootstrap consumed history is not an array")
        value["consumed_authorities"] = tuple(
            ConsumedBootstrapAuthority(**dict(item))
            for item in consumed
            if isinstance(item, Mapping)
        )
        if len(value["consumed_authorities"]) != len(consumed):
            raise StateStoreError("bootstrap consumed history item is not a mapping")
        state = CoreCurrentBootstrapState(**value)
        if state_content_fingerprint(state) != state.state_fingerprint_sha256:
            raise StateStoreError("bootstrap state content fingerprint mismatch")
        return state

    @staticmethod
    def _decode_consumed(payload: object) -> ConsumedBootstrapAuthority:
        if not isinstance(payload, Mapping):
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
        try:
            return ConsumedBootstrapAuthority(**dict(payload))
        except (TypeError, ValueError, FirstRunBootstrapError) as exc:
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT") from exc

    def _family(self) -> tuple[CoreCurrentBootstrapState, ...]:
        try:
            snapshot = self._store.read_verified_snapshot()
            if snapshot is None:
                raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
            states = tuple(
                sorted(
                    (
                        self._decode(record.payload)
                        for record in snapshot.current_records
                        if record.representation_name == "bootstrap consumed fence"
                    ),
                    key=lambda state: state.state_revision,
                )
            )
            history = tuple(
                self._decode_consumed(record.payload)
                for record in snapshot.immutable_history
                if record.representation_name == "bootstrap accepted/consumption history"
            )
            if not states:
                raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
            initial = states[0]
            if (
                initial.state_revision != 1
                or initial.consumed_authorities
                or initial.initial_security_lifecycle != "PRE_INITIAL_SECURITY"
                or initial.first_operator_presence != "ABSENT"
            ):
                raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
            appended: list[ConsumedBootstrapAuthority] = []
            terminal_seen = False
            for index, (pre, post) in enumerate(zip(states, states[1:], strict=False), 1):
                terminal = (
                    post.initial_security_lifecycle == "INITIAL_SECURITY_COMPLETED"
                    and post.first_operator_presence == "PRESENT"
                )
                if terminal:
                    if terminal_seen or index != len(states) - 1:
                        raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
                    terminal_seen = True
                    if (
                        pre.initial_security_lifecycle != "PRE_INITIAL_SECURITY"
                        or pre.first_operator_presence != "ABSENT"
                        or post.state_revision != pre.state_revision + 1
                        or post.account_id != pre.account_id
                        or post.device_installation_id != pre.device_installation_id
                        or post.intended_operator_id != pre.intended_operator_id
                        or post.startup_readiness != pre.startup_readiness
                        or post.startup_readiness != "SETUP_REQUIRED"
                        or post.expected_generation != pre.expected_generation
                        or post.expected_revision != pre.expected_revision
                        or post.consumed_authorities != pre.consumed_authorities
                        or len(pre.consumed_authorities) != 1
                    ):
                        raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
                else:
                    if terminal_seen or not post.consumed_authorities:
                        raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
                    consumed = post.consumed_authorities[-1]
                    validate_bootstrap_state_transition(pre, post, consumed)
                    appended.append(consumed)
            if len(history) != len(appended) or set(history) != set(appended):
                raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
            return states
        except FirstRunBootstrapError as exc:
            if exc.reason == "CONTRACT_INCONSISTENT":
                raise
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT") from exc
        except (TypeError, ValueError, StateStoreError, sqlite3.Error) as exc:
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT") from exc

    def resolve_accepted_state(self, state_reference: str) -> CoreCurrentBootstrapState:
        matches = tuple(
            state for state in self._family() if state.state_fingerprint_sha256 == state_reference
        )
        if len(matches) != 1:
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
        return matches[0]

    def current_state_reference(self, account_id: str, device_installation_id: str) -> str:
        family = self._family()
        scoped = tuple(
            state
            for state in family
            if (state.account_id, state.device_installation_id)
            == (account_id, device_installation_id)
        )
        if not scoped:
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
        if len(scoped) != len(family):
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT")
        current = family[-1]
        if (
            current.initial_security_lifecycle == "INITIAL_SECURITY_COMPLETED"
            or current.first_operator_presence == "PRESENT"
        ):
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
        return cast(str, current.state_fingerprint_sha256)

    def terminal_state(
        self, account_id: str, device_installation_id: str
    ) -> CoreCurrentBootstrapState:
        """Return the proved terminal state for audit/reconstruction, never bootstrap use."""
        family = self._family()
        current = family[-1]
        if (
            (current.account_id, current.device_installation_id)
            != (account_id, device_installation_id)
            or current.initial_security_lifecycle != "INITIAL_SECURITY_COMPLETED"
            or current.first_operator_presence != "PRESENT"
        ):
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
        return current


def _bootstrap_state_record(state: CoreCurrentBootstrapState):
    """Build the frozen M0.11 canonical accepted/current bootstrap carrier."""

    payload = asdict(state)
    payload["consumed_authorities"] = list(payload["consumed_authorities"])
    return persistence_record(
        "bootstrap consumed fence",
        f"bootstrap-current:{state.account_id}:{state.device_installation_id}:{state.state_revision}",
        payload,
    )


def _consumed_history_record(consumed: ConsumedBootstrapAuthority):
    payload = asdict(consumed)
    return persistence_record(
        "bootstrap accepted/consumption history",
        "bootstrap-history:"
        f"{consumed.account_id}:{consumed.device_installation_id}:"
        f"{consumed.bootstrap_generation}:{consumed.bootstrap_revision}:"
        f"{consumed.claim_fingerprint_sha256}",
        payload,
    )


class DurableFirstRunBootstrapCoordinator:
    """Compare, semantically consume, and atomically publish one bootstrap transition."""

    def __init__(self, store: SQLiteStateStore, provisioning: ProvisioningBoundary) -> None:
        self._store = store
        self._registry = DurableFirstRunBootstrapRegistry(store)
        self._authority = FirstRunBootstrapAuthority(provisioning, self._registry)
        self._provisioning = provisioning

    def _read_metadata(self) -> StateStoreMetadata:
        try:
            metadata = self._store.read_metadata()
        except (StateStoreError, sqlite3.Error) as exc:
            raise FirstRunBootstrapError("CONTRACT_INCONSISTENT") from exc
        if metadata is None:
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
        return metadata

    def materialize_initial_state(self, claim_reference: str) -> str:
        """Atomically materialize canonical PRE from accepted provisioning authority."""

        try:
            metadata = self._read_metadata()
            snapshot = self._store.read_verified_snapshot()
            if snapshot is None or any(
                record.representation_name
                in {"bootstrap consumed fence", "bootstrap accepted/consumption history"}
                for record in (*snapshot.current_records, *snapshot.immutable_history)
            ):
                raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
            claim = self._provisioning.resolve_accepted_claim(claim_reference)
            binding = self._provisioning.resolve_membership(claim_reference)
            if not isinstance(claim, FirstRunBootstrapClaim) or not isinstance(
                binding, ProvisioningMembershipBinding
            ):
                raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
            complete = claim_content_fingerprint(claim)
            if (
                complete != claim.claim_fingerprint_sha256
                or claim.claim_fingerprint_sha256 != claim_reference
                or binding.claim_fingerprint_sha256 != claim_reference
                or binding.complete_claim_content_fingerprint_sha256 != complete
                or binding.authority_source != AUTHORITY_SOURCE
                or binding.provisioning_context_fingerprint_sha256
                != claim.provisioning_context_fingerprint_sha256
                or (claim.account_id, claim.device_installation_id)
                != (metadata.account_id, metadata.device_installation_id)
            ):
                raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED")
            initial = CoreCurrentBootstrapState(
                state_fingerprint_sha256="0" * 64,
                account_id=claim.account_id,
                device_installation_id=claim.device_installation_id,
                intended_operator_id=claim.intended_operator_id,
                startup_readiness="SETUP_REQUIRED",
                initial_security_lifecycle="PRE_INITIAL_SECURITY",
                first_operator_presence="ABSENT",
                expected_generation=claim.bootstrap_generation,
                expected_revision=claim.bootstrap_revision,
                consumed_authorities=(),
                state_revision=1,
            )
            initial = CoreCurrentBootstrapState(
                **{
                    **asdict(initial),
                    "state_fingerprint_sha256": state_content_fingerprint(initial),
                }
            )
            record = _bootstrap_state_record(initial)
            target = StateStoreMetadata.from_mapping(
                {
                    **metadata.to_mapping(),
                    "protected_freshness_generation": metadata.protected_freshness_generation + 1,
                }
            )
            prepared = self._store.derive_prepared_metadata(
                target,
                current_records=(record,),
                expected_current_generation=metadata.protected_freshness_generation,
            )
            self._store.commit_prepared_state(
                prepared,
                current_records=(record,),
                immutable_history=(),
                expected_current_generation=metadata.protected_freshness_generation,
            )
            return cast(str, initial.state_fingerprint_sha256)
        except FirstRunBootstrapError:
            raise
        except (KeyError, TypeError, ValueError, StateStoreError, sqlite3.Error) as exc:
            raise FirstRunBootstrapError("BOOTSTRAP_AUTHORITY_DENIED") from exc

    def consume(
        self,
        claim: FirstRunBootstrapClaim,
        now_utc: str,
        purpose: str = INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    ) -> BootstrapTransitionResult:
        try:
            metadata = self._read_metadata()
            pre_reference = self._registry.current_state_reference(
                claim.account_id, claim.device_installation_id
            )
            result, post = self._authority.consume(claim, pre_reference, now_utc, purpose)
            current_record = _bootstrap_state_record(post)
            history_record = _consumed_history_record(result.consumed_authority)
            target = StateStoreMetadata.from_mapping(
                {
                    **metadata.to_mapping(),
                    "protected_freshness_generation": metadata.protected_freshness_generation + 1,
                }
            )
            prepared = self._store.derive_prepared_metadata(
                target,
                current_records=(current_record,),
                immutable_history=(history_record,),
                expected_current_generation=metadata.protected_freshness_generation,
            )
            # BEGIN IMMEDIATE plus the generation CAS is the transaction-time PRE fence.
            self._store.commit_prepared_state(
                prepared,
                current_records=(current_record,),
                immutable_history=(history_record,),
                expected_current_generation=metadata.protected_freshness_generation,
            )
            return result
        except FirstRunBootstrapError:
            raise
        except (KeyError, TypeError, ValueError, StateStoreError, sqlite3.Error) as exc:
            raise FirstRunBootstrapError("STALE_CORE_BOOTSTRAP_STATE") from exc


__all__ = [
    "DurableFirstRunBootstrapCoordinator",
    "DurableFirstRunBootstrapRegistry",
]
