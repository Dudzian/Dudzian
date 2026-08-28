"""Ordered consumer-side handoff to the external protected freshness owner.

This module deliberately supplies no external storage, provisioning, replacement,
retirement, abort, recovery, backup, restore, or LIVE-authority implementation.
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
from .state_store import SQLiteStateStore, StateStoreMetadata

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

    def _external(self, scope: EvidenceScope) -> tuple[object, ProtectedFreshnessAuthorityRecord]:
        resolved = self._authority.resolve_current(scope)
        if resolved is None:
            raise ProtectedFreshnessHandoffError("external current membership is missing")
        ref, mapping = resolved
        record = ProtectedFreshnessAuthorityRecord.from_mapping(mapping)
        if record.scope != scope:
            raise ProtectedFreshnessHandoffError("external scope mismatch")
        return ref, record

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

    def advance_protected_state(
        self,
        metadata: StateStoreMetadata,
        *,
        current_records: Iterable[PersistenceRecord] = (),
        immutable_history: Iterable[PersistenceRecord] = (),
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
            expected = None if before is None else before.metadata.protected_freshness_generation
            if external.lifecycle == "PREPARED":
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
            self._recovery_required = True
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
            self._recovery_required = False
            return candidate
