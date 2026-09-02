"""Ordered consumer-side handoff to the external protected freshness owner.

This module deliberately supplies no external storage, provisioning, replacement,
retirement, backup, restore, or LIVE-authority implementation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from threading import RLock
from typing import Any, Protocol, TypeAlias

from .fingerprints import canonical_json_sha256
from .local_durable_evidence import (
    EvidenceScope,
    LocalDurableEvidenceRegistry,
    LocalDurableStateEvidence,
)
from .records import PersistenceRecord
from .state_store import SQLiteStateStore, StateStoreMetadata, StateStoreSnapshot

_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE = "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY"
EvidenceResolver: TypeAlias = Callable[[object], LocalDurableStateEvidence | None]


class ProtectedFreshnessHandoffError(RuntimeError):
    """Fail-closed implementation error for one ordinary S5 advance."""


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ProtectedFreshnessAuthorityRecord:
    """Exact frozen M0.3 protected membership projection read carrier."""

    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    lifecycle: str
    committed_generation: int | None
    committed_state_fingerprint_sha256: str | None
    prepared_generation: int | None
    prepared_state_fingerprint_sha256: str | None
    prepared_transaction_fingerprint_sha256: str | None
    authority_revision: int
    authority_source: str
    content_fingerprint_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProtectedFreshnessAuthorityRecord:
        expected = tuple(item.name for item in fields(cls))
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise ProtectedFreshnessHandoffError("external record has a non-exact field set")
        try:
            record = cls(**{name: value[name] for name in expected})
        except TypeError as exc:
            raise ProtectedFreshnessHandoffError("external record is malformed") from exc
        record.validate()
        return record

    def validate(self) -> None:
        if (
            not isinstance(self.account_id, str)
            or not _ID_RE.fullmatch(self.account_id)
            or not self.account_id.startswith("acct_")
        ):
            raise ProtectedFreshnessHandoffError("external account scope is noncanonical")
        if (
            not isinstance(self.device_installation_id, str)
            or not _ID_RE.fullmatch(self.device_installation_id)
            or not self.device_installation_id.startswith("dev_")
        ):
            raise ProtectedFreshnessHandoffError("external device scope is noncanonical")
        if not _sha(self.state_store_identity_fingerprint_sha256):
            raise ProtectedFreshnessHandoffError("external store scope is malformed")
        if not _positive(self.authority_revision) or self.authority_source != _SOURCE:
            raise ProtectedFreshnessHandoffError("external authority metadata is malformed")
        committed = (self.committed_generation, self.committed_state_fingerprint_sha256)
        prepared = (
            self.prepared_generation,
            self.prepared_state_fingerprint_sha256,
            self.prepared_transaction_fingerprint_sha256,
        )
        valid = False
        if self.lifecycle == "UNINITIALIZED":
            valid = committed == (None, None) and prepared == (None, None, None)
        elif self.lifecycle == "COMMITTED":
            valid = (
                _positive(committed[0]) and _sha(committed[1]) and prepared == (None, None, None)
            )
        elif self.lifecycle == "PREPARED":
            valid = _positive(prepared[0]) and _sha(prepared[1]) and _sha(prepared[2])
            valid = valid and (
                (committed == (None, None) and prepared[0] == 1)
                or (
                    _positive(committed[0])
                    and _sha(committed[1])
                    and prepared[0] == committed[0] + 1
                )
            )
        if not valid:
            raise ProtectedFreshnessHandoffError("external lifecycle field presence is malformed")
        projection = asdict(self)
        supplied = projection.pop("content_fingerprint_sha256")
        if not _sha(supplied) or canonical_json_sha256(projection) != supplied:
            raise ProtectedFreshnessHandoffError("external content fingerprint mismatch")

    @property
    def scope(self) -> EvidenceScope:
        return (
            self.account_id,
            self.device_installation_id,
            self.state_store_identity_fingerprint_sha256,
        )


class ProtectedFreshnessAuthorityPort(Protocol):
    """Trusted port implemented by the external protected boundary."""

    def resolve_current(self, scope: EvidenceScope) -> tuple[object, Mapping[str, Any]] | None: ...

    def prepare(
        self,
        current_ref: object,
        scope: EvidenceScope,
        *,
        expected_committed_generation: int | None,
        expected_committed_state_fingerprint_sha256: str | None,
        candidate_generation: int,
        candidate_state_fingerprint_sha256: str,
        candidate_transaction_fingerprint_sha256: str,
    ) -> None: ...

    def finalize(
        self,
        current_ref: object,
        scope: EvidenceScope,
        *,
        evidence_ref: object,
        evidence_resolver: EvidenceResolver,
    ) -> None: ...

    def abort(
        self,
        current_ref: object,
        scope: EvidenceScope,
        *,
        evidence_ref: object,
        evidence_resolver: EvidenceResolver,
    ) -> None: ...


class ProtectedFreshnessHandoffCoordinator:
    """Process-serialized ordered protocol for one ordinary protected advance."""

    def __init__(
        self,
        store: SQLiteStateStore,
        registry: LocalDurableEvidenceRegistry,
        authority: ProtectedFreshnessAuthorityPort,
    ) -> None:
        self._store, self._registry, self._authority = store, registry, authority
        self._lock = RLock()
        self._recovery_required = False
        self._recovery_scope: EvidenceScope | None = None

    def _bind_recovery(self, scope: EvidenceScope) -> None:
        if self._recovery_required and self._recovery_scope != scope:
            raise ProtectedFreshnessHandoffError(
                "recovery scope does not match active protected recovery"
            )
        self._recovery_required = True
        self._recovery_scope = scope

    def _clear_recovery(self) -> None:
        self._recovery_required = False
        self._recovery_scope = None

    def _assert_local_still_matches(self, expected: StateStoreMetadata | None) -> None:
        current = self._store.read_verified_snapshot()
        if (expected is None and current is not None) or (
            expected is not None and (current is None or current.metadata != expected)
        ):
            raise ProtectedFreshnessHandoffError(
                "local StateStore changed during protected handoff"
            )

    def _external(self, scope: EvidenceScope) -> tuple[object, ProtectedFreshnessAuthorityRecord]:
        resolved = self._authority.resolve_current(scope)
        if resolved is None:
            raise ProtectedFreshnessHandoffError("external current membership is missing")
        ref, mapping = resolved
        record = ProtectedFreshnessAuthorityRecord.from_mapping(mapping)
        if record.scope != scope:
            raise ProtectedFreshnessHandoffError("external scope mismatch")
        return ref, record

    @staticmethod
    def _validate_scope(scope: EvidenceScope) -> None:
        if (
            not isinstance(scope, tuple)
            or len(scope) != 3
            or not isinstance(scope[0], str)
            or _ID_RE.fullmatch(scope[0]) is None
            or not scope[0].startswith("acct_")
            or not isinstance(scope[1], str)
            or _ID_RE.fullmatch(scope[1]) is None
            or not scope[1].startswith("dev_")
            or not _sha(scope[2])
        ):
            raise ProtectedFreshnessHandoffError("recovery scope is malformed")

    @staticmethod
    def _is_terminal(
        ref: object,
        expected_ref: object,
        record: ProtectedFreshnessAuthorityRecord,
        generation: int,
        state: str,
        *,
        revision_after: int | None = None,
    ) -> bool:
        return (
            ref == expected_ref
            and record.lifecycle == "COMMITTED"
            and record.committed_generation == generation
            and record.committed_state_fingerprint_sha256 == state
            and (revision_after is None or record.authority_revision > revision_after)
        )

    def _store_bound_resolver(self, scope: EvidenceScope) -> EvidenceResolver:
        def resolve(ref: object) -> LocalDurableStateEvidence | None:
            evidence = self._registry.resolve_current(scope, ref)
            if evidence is None:
                return None
            snapshot = self._store.read_verified_snapshot()
            if snapshot is None:
                return None
            metadata = snapshot.metadata
            if (
                metadata.account_id,
                metadata.device_installation_id,
                metadata.state_store_identity_fingerprint_sha256,
            ) != scope:
                return None
            if (
                evidence.generation,
                evidence.state_fingerprint_sha256,
                evidence.transaction_fingerprint_sha256,
            ) != (
                metadata.protected_freshness_generation,
                metadata.state_fingerprint_sha256,
                metadata.transaction_fingerprint_sha256,
            ):
                return None
            # Fail closed if a same-generation publication replaced this ref
            # while the fresh SQLite snapshot was being verified.
            return self._registry.resolve_current(scope, ref)

        return resolve

    def recover_protected_state(self, scope: EvidenceScope) -> StateStoreMetadata | None:
        """Reconcile one existing protected/local pair without a business transition."""

        with self._lock:
            self._validate_scope(scope)
            self._bind_recovery(scope)
            ref, external = self._external(scope)
            snapshot = self._store.read_verified_snapshot()
            local = None if snapshot is None else snapshot.metadata
            if (
                local is not None
                and (
                    local.account_id,
                    local.device_installation_id,
                    local.state_store_identity_fingerprint_sha256,
                )
                != scope
            ):
                raise ProtectedFreshnessHandoffError("verified local scope mismatch")

            if external.lifecycle == "UNINITIALIZED":
                if local is not None:
                    raise ProtectedFreshnessHandoffError("local store is ahead of UNINITIALIZED")
                self._assert_local_still_matches(None)
                self._clear_recovery()
                return None

            if external.lifecycle == "COMMITTED":
                if local is None or (
                    local.protected_freshness_generation,
                    local.state_fingerprint_sha256,
                ) != (
                    external.committed_generation,
                    external.committed_state_fingerprint_sha256,
                ):
                    raise ProtectedFreshnessHandoffError(
                        "external COMMITTED and verified local state differ"
                    )
                self._assert_local_still_matches(local)
                self._clear_recovery()
                return local

            # A genesis PREPARED record can never be aborted: absence of local
            # state is not evidence that generation 1 was never durable.
            if external.committed_generation is None:
                if local is None:
                    raise ProtectedFreshnessHandoffError("GENESIS_PENDING_RECOVERY_REQUIRED")
                action = "finalize"
            else:
                if local is None:
                    raise ProtectedFreshnessHandoffError("normal pending has no local baseline")
                local_pair = (
                    local.protected_freshness_generation,
                    local.state_fingerprint_sha256,
                )
                committed_pair = (
                    external.committed_generation,
                    external.committed_state_fingerprint_sha256,
                )
                prepared_pair = (
                    external.prepared_generation,
                    external.prepared_state_fingerprint_sha256,
                )
                if local_pair == committed_pair:
                    action = "abort"
                elif local_pair == prepared_pair:
                    action = "finalize"
                else:
                    raise ProtectedFreshnessHandoffError(
                        "verified local state matches neither committed nor exact pending"
                    )

            if local is None:
                raise ProtectedFreshnessHandoffError("recovery action requires durable local state")
            if snapshot is None:
                raise ProtectedFreshnessHandoffError("verified recovery snapshot is unavailable")
            if action == "finalize":
                if (
                    local.protected_freshness_generation != external.prepared_generation
                    or local.state_fingerprint_sha256 != external.prepared_state_fingerprint_sha256
                    or local.transaction_fingerprint_sha256
                    != external.prepared_transaction_fingerprint_sha256
                ):
                    raise ProtectedFreshnessHandoffError(
                        "local state is not the exact pending state"
                    )
                descriptors = tuple(
                    descriptor
                    for descriptor in snapshot.transaction_descriptors
                    if descriptor.target_generation == local.protected_freshness_generation
                )
                if len(descriptors) != 1:
                    raise ProtectedFreshnessHandoffError("exact current descriptor is unavailable")
                descriptor = descriptors[0]
                if external.committed_generation is not None and (
                    descriptor.expected_current_generation != external.committed_generation
                    or descriptor.pre_state_fingerprint_sha256
                    != external.committed_state_fingerprint_sha256
                    or descriptor.post_state_fingerprint_sha256
                    != external.prepared_state_fingerprint_sha256
                    or descriptor.transaction_fingerprint_sha256
                    != external.prepared_transaction_fingerprint_sha256
                ):
                    raise ProtectedFreshnessHandoffError(
                        "pending does not descend from exact external baseline"
                    )

            evidence_ref = self._registry.publish_verified_state(self._store)
            resolver = self._store_bound_resolver(scope)
            evidence = resolver(evidence_ref)
            expected_generation = (
                external.committed_generation if action == "abort" else external.prepared_generation
            )
            expected_state = (
                external.committed_state_fingerprint_sha256
                if action == "abort"
                else external.prepared_state_fingerprint_sha256
            )
            if not _positive(expected_generation) or not _sha(expected_state):
                raise ProtectedFreshnessHandoffError("recovery terminal target is malformed")
            revision_after = external.authority_revision if action == "abort" else None
            if (
                evidence is None
                or evidence.generation != expected_generation
                or evidence.state_fingerprint_sha256 != expected_state
            ):
                raise ProtectedFreshnessHandoffError("fresh recovery evidence mismatch")
            try:
                if action == "abort":
                    self._authority.abort(
                        ref, scope, evidence_ref=evidence_ref, evidence_resolver=resolver
                    )
                else:
                    self._authority.finalize(
                        ref, scope, evidence_ref=evidence_ref, evidence_resolver=resolver
                    )
            except Exception as exc:
                try:
                    terminal_ref, terminal = self._external(scope)
                except Exception:
                    raise ProtectedFreshnessHandoffError(
                        f"external {action.upper()} outcome is unresolved"
                    ) from exc
                if not self._is_terminal(
                    terminal_ref,
                    ref,
                    terminal,
                    expected_generation,
                    expected_state,
                    revision_after=revision_after,
                ):
                    raise ProtectedFreshnessHandoffError(
                        f"external {action.upper()} outcome is unresolved"
                    ) from exc
            terminal_ref, terminal = self._external(scope)
            if not self._is_terminal(
                terminal_ref,
                ref,
                terminal,
                expected_generation,
                expected_state,
                revision_after=revision_after,
            ):
                raise ProtectedFreshnessHandoffError(
                    f"external {action.upper()} terminal state mismatch"
                )
            self._assert_local_still_matches(local)
            self._clear_recovery()
            return local

    def advance_protected_state(
        self,
        metadata: StateStoreMetadata,
        *,
        current_records: Iterable[PersistenceRecord] = (),
        immutable_history: Iterable[PersistenceRecord] = (),
        _expected_source: StateStoreMetadata | None = None,
    ) -> StateStoreMetadata:
        with self._lock:
            if self._recovery_required:
                raise ProtectedFreshnessHandoffError("ordinary protected advance requires recovery")
            scope = (
                metadata.account_id,
                metadata.device_installation_id,
                metadata.state_store_identity_fingerprint_sha256,
            )
            ref, external = self._external(scope)
            before = self._store.read_verified_snapshot()
            if _expected_source is not None and (
                before is None or before.metadata != _expected_source
            ):
                raise ProtectedFreshnessHandoffError(
                    "protected mutation source changed before advancement"
                )
            expected = None if before is None else before.metadata.protected_freshness_generation
            if external.lifecycle == "PREPARED":
                self._bind_recovery(scope)
                raise ProtectedFreshnessHandoffError("external PREPARED requires recovery")
            if before is None:
                if (
                    external.lifecycle != "UNINITIALIZED"
                    or metadata.protected_freshness_generation != 1
                ):
                    raise ProtectedFreshnessHandoffError("invalid protected genesis entry")
            else:
                local = before.metadata
                if (
                    external.lifecycle != "COMMITTED"
                    or external.committed_generation != expected
                    or external.committed_state_fingerprint_sha256 != local.state_fingerprint_sha256
                ):
                    raise ProtectedFreshnessHandoffError(
                        "external committed authority does not match local current"
                    )
                if metadata.protected_freshness_generation != expected + 1:
                    raise ProtectedFreshnessHandoffError("candidate generation is not local G+1")
                if (
                    metadata.account_id != local.account_id
                    or metadata.device_installation_id != local.device_installation_id
                    or metadata.state_store_identity_fingerprint_sha256
                    != local.state_store_identity_fingerprint_sha256
                    or metadata.state_store_schema_version != local.state_store_schema_version
                ):
                    raise ProtectedFreshnessHandoffError(
                        "candidate violates immutable local metadata preflight"
                    )
            current_delta, history_delta = tuple(current_records), tuple(immutable_history)
            candidate = self._store.derive_prepared_metadata(
                metadata,
                current_records=current_delta,
                immutable_history=history_delta,
                expected_current_generation=expected,
            )
            latest = self._store.read_verified_snapshot()
            if (before is None) != (latest is None) or (
                before is not None and latest is not None and latest.metadata != before.metadata
            ):
                raise ProtectedFreshnessHandoffError(
                    "local StateStore baseline changed during candidate derivation"
                )
            # From this point an external transition may have occurred.  Only
            # exact post-FINALIZE verification makes this instance ordinary-ready.
            self._bind_recovery(scope)
            self._authority.prepare(
                ref,
                scope,
                expected_committed_generation=external.committed_generation,
                expected_committed_state_fingerprint_sha256=external.committed_state_fingerprint_sha256,
                candidate_generation=candidate.protected_freshness_generation,
                candidate_state_fingerprint_sha256=candidate.state_fingerprint_sha256,
                candidate_transaction_fingerprint_sha256=candidate.transaction_fingerprint_sha256,
            )
            prepared_ref, prepared = self._external(scope)
            if (
                prepared_ref != ref
                or prepared.lifecycle != "PREPARED"
                or (
                    prepared.prepared_generation,
                    prepared.prepared_state_fingerprint_sha256,
                    prepared.prepared_transaction_fingerprint_sha256,
                )
                != (
                    candidate.protected_freshness_generation,
                    candidate.state_fingerprint_sha256,
                    candidate.transaction_fingerprint_sha256,
                )
                or (prepared.committed_generation, prepared.committed_state_fingerprint_sha256)
                != (external.committed_generation, external.committed_state_fingerprint_sha256)
            ):
                raise ProtectedFreshnessHandoffError("external PREPARE acknowledgement mismatch")
            self._store.commit_prepared_state(
                candidate,
                current_records=current_delta,
                immutable_history=history_delta,
                expected_current_generation=expected,
            )
            after = self._store.read_verified_snapshot()
            if after is None or (
                after.metadata.account_id,
                after.metadata.device_installation_id,
                after.metadata.state_store_identity_fingerprint_sha256,
                after.metadata.protected_freshness_generation,
                after.metadata.state_fingerprint_sha256,
                after.metadata.transaction_fingerprint_sha256,
            ) != (
                *scope,
                candidate.protected_freshness_generation,
                candidate.state_fingerprint_sha256,
                candidate.transaction_fingerprint_sha256,
            ):
                raise ProtectedFreshnessHandoffError("local commit verification mismatch")
            evidence_ref = self._registry.publish_verified_state(self._store)
            resolver = self._store_bound_resolver(scope)
            evidence = resolver(evidence_ref)
            if evidence is None or (
                evidence.generation,
                evidence.state_fingerprint_sha256,
                evidence.transaction_fingerprint_sha256,
            ) != (
                candidate.protected_freshness_generation,
                candidate.state_fingerprint_sha256,
                candidate.transaction_fingerprint_sha256,
            ):
                raise ProtectedFreshnessHandoffError("current local evidence mismatch")
            self._authority.finalize(
                ref, scope, evidence_ref=evidence_ref, evidence_resolver=resolver
            )
            committed_ref, committed = self._external(scope)
            if (
                committed_ref != ref
                or committed.lifecycle != "COMMITTED"
                or (committed.committed_generation, committed.committed_state_fingerprint_sha256)
                != (candidate.protected_freshness_generation, candidate.state_fingerprint_sha256)
            ):
                raise ProtectedFreshnessHandoffError("external FINALIZE acknowledgement mismatch")
            self._assert_local_still_matches(candidate)
            self._clear_recovery()
            return candidate

    def advance_protected_mutation(
        self,
        builder: Callable[
            [StateStoreSnapshot],
            tuple[
                StateStoreMetadata,
                tuple[PersistenceRecord, ...],
                tuple[PersistenceRecord, ...],
            ],
        ],
    ) -> StateStoreMetadata:
        """Build records from the precise verified pre-transition observation."""

        with self._lock:
            source = self._store.read_verified_snapshot()
            if source is None:
                raise ProtectedFreshnessHandoffError(
                    "protected semantic mutation requires initialized StateStore"
                )
            metadata, current, history = builder(source)
            return self.advance_protected_state(
                metadata,
                current_records=current,
                immutable_history=history,
                _expected_source=source.metadata,
            )
