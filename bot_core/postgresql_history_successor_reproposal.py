"""Versioned isolated-test history proposals preserving the original Root Proof.

This internal port signs only a new history head through genuine ACTIVE custody.
Approval and final atomic commit use separate qualified direct principals. A
recovery locator permits reads; writes always identify one exact revision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryEventIdentity,
    _credential_material,
    _plain,
    build_record,
    attest_head,
    canonical_json_bytes,
    verify_head,
)
from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    SigningKeyLifecycle,
)
from bot_core.postgresql_atomic_root_proof_commit import (
    AtomicRootProofCommitConflict,
    AtomicRootProofCommitCorrupt,
    AtomicRootProofCommitDecision,
    AtomicRootProofCommitError,
    AtomicRootProofCommitFences,
    AtomicRootProofCommitIndeterminate,
    AtomicRootProofCommitLocator,
    AtomicRootProofCommitUnavailable,
    PostgreSQLAtomicRootProofCommit,
    PreparedAtomicRootProofCommit,
    _PREPARATION_LOCK,
    _PREPARATIONS,
    _TrustedPreparation,
    _base_package,
    _issuance_history_payload,
    _provider_pin,
    _trusted_preparation,
    atomic_history_fence_key,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_history_successor_reproposal_evidence import (
    requalify_original_history_successor_evidence,
)
from bot_core.postgresql_history_successor_reproposal_qualification import (
    PostgreSQLHistorySuccessorReproposalV1Deployment,
    qualify_history_successor_reproposal_v1,
)
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.postgresql_root_proof_preparation_readonly import (
    lock_root_proof_preparation_signer_v2,
)
from bot_core.postgresql_root_proof_preparation_v2 import (
    PostgreSQLAuthenticatedIssuerHistoryV2,
    PostgreSQLRootProofRetentionV2,
    PostgreSQLRootProofSigningIntentV2,
)
from bot_core.postgresql_root_proof_retention import _canonical
from bot_core.root_proof_atomic_attempt_consistency import _validate_providers, _validate_records

_CANDIDATE_ID = re.compile(r"hp1_[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class HistorySuccessorProposal:
    """Exact immutable public selector; its construction conveys no authority."""

    locator: AtomicRootProofCommitLocator
    candidate_id: str
    revision: int
    predecessor_candidate_id: str | None

    def __post_init__(self) -> None:
        if (
            type(self.locator) is not AtomicRootProofCommitLocator
            or type(self.candidate_id) is not str
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
            or type(self.revision) is not int
            or not 1 <= self.revision <= 9007199254740991
            or self.revision == 1
            and self.predecessor_candidate_id is not None
            or self.revision > 1
            and (
                type(self.predecessor_candidate_id) is not str
                or _CANDIDATE_ID.fullmatch(self.predecessor_candidate_id) is None
            )
        ):
            raise TypeError("EXACT_VERSIONED_HISTORY_PROPOSAL_REQUIRED")


def _invariant_package(package: dict[str, Any]) -> bytes:
    """Allow only the mathematically derived history successor coordinates."""
    detached = dict(package)
    record = dict(detached["history_record"])
    for key in ("sequence", "predecessor_authenticated_digest", "authenticated_digest"):
        del record[key]
    detached["history_record"] = record
    del detached["expected_history_digest"]
    del detached["history_record_hex"]
    return _canonical(detached)


def _persistence_cut(name: str, connection: psycopg.Connection[Any]) -> None:
    """Fault injection seam, without caller callbacks or runtime flags."""


def _reprepare_original(
    original: PreparedAtomicRootProofCommit,
    binding: object,
    signed_attempt: object,
    *,
    signing_intent: PostgreSQLRootProofSigningIntentV2,
    retention: PostgreSQLRootProofRetentionV2,
    issuer_authority: PostgreSQLRootProofIssuanceAuthorityV2,
    history: PostgreSQLAuthenticatedIssuerHistoryV2,
) -> PreparedAtomicRootProofCommit:
    """Full read-only preparation with the original historical signed evidence.

    A legacy authorization includes its former standalone BIND capability in
    the signed digest. The explicit evidence bridge verifies that historical
    grant removal through current V2 readers without reissuing authorization.
    RESERVED, STAGED, signatures, custody and every event fact are reverified.
    """
    _validate_providers(signing_intent, retention)
    if (
        signing_intent._authority is not issuer_authority
        or history._deployment is not signing_intent._deployment
        or issuer_authority.deployment is not signing_intent._deployment
    ):
        raise AtomicRootProofCommitConflict("EXACT_SHARED_V2_PREPARATION_DEPLOYMENT_REQUIRED")
    evidence = requalify_original_history_successor_evidence(
        original, binding, signed_attempt, issuer_authority=issuer_authority
    )
    auth, request = evidence.source.authorization, evidence.request
    with signing_intent._connect() as conn, conn.transaction():
        # The enrolled fence uses FOR SHARE; this transaction changes no rows.
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        signing_intent._qualify(conn)
        signing_intent._lock_attempt(conn, request.attempt_id)
        snapshot = signing_intent._snapshot()
        lock_root_proof_preparation_signer_v2(conn, signing_intent._deployment, snapshot)
        row = signing_intent._read_row(conn, request)
        reserved = None if row is None else signing_intent._restore(row, auth, request)
        if (
            snapshot != original.prepared.credential_snapshot
            or signing_intent._snapshot() != snapshot
        ):
            raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_ACTIVE_ROOT_SIGNER_REQUIRED")
    prepared = retention.read_prepared(authorization=auth, retained_request=request)
    _validate_records(auth, request, reserved, prepared, signing_intent)
    if reserved != original.reserved or prepared != original.prepared:
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_RESERVED_STAGED_REQUIRED")
    identity = evidence.current.identity
    if identity is None:
        raise AtomicRootProofCommitConflict("EXACT_SIGNED_ATTEMPT_IDENTITY_REQUIRED")
    stream = history._stream
    if (
        stream != original.history_record.stream
        or stream.environment != auth.environment
        or stream.trust_domain != auth.trust_domain
        or stream.product_scope != auth.product_scope
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_HISTORY_SCOPE_REQUIRED")
    event = HistoryEventIdentity(
        auth.logical_operation_id, request.attempt_id, prepared.root_proof_id
    )
    payload = _issuance_history_payload(
        auth, original.bound_state, evidence.predecessor, identity.digest_sha256
    )
    if event != original.history_record.event_identity or canonical_json_bytes(
        payload
    ) != _canonical(_plain(original.history_record.canonical_event_payload)):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_ISSUANCE_EVENT_REQUIRED")
    latest = history.current_record()
    record = build_record(
        stream,
        1 if latest is None else latest.sequence + 1,
        NO_PREDECESSOR if latest is None else latest.authenticated_digest,
        event,
        payload,
    )
    value = PreparedAtomicRootProofCommit(
        reserved,
        prepared,
        original.bound_state,
        record,
        original.locator,
        issuer_authority.requester_registry._current(auth.requester_principal_id),
        issuer_authority.claimant_registry._current(auth.provisioning_principal_id),
    )
    repeated = requalify_original_history_successor_evidence(
        original, binding, signed_attempt, issuer_authority=issuer_authority
    )
    if repeated != evidence or signing_intent._snapshot() != snapshot:
        raise AtomicRootProofCommitConflict("ORIGINAL_IDENTITIES_CHANGED_DURING_REPREPARATION")
    trusted = _TrustedPreparation(
        _canonical(_base_package(value)),
        (
            _provider_pin(issuer_authority.entitlement_registry, "_"),
            _provider_pin(signing_intent, "_"),
            _provider_pin(retention, "_"),
            _provider_pin(history, "_"),
            _provider_pin(issuer_authority.requester_registry, "_"),
            _provider_pin(issuer_authority.claimant_registry, "_"),
        ),
    )
    with _PREPARATION_LOCK:
        _PREPARATIONS[value] = trusted
    return value


class PostgreSQLHistorySuccessorReproposalV1:
    """Qualified versioned proposal and one-transaction atomic commit client."""

    __slots__ = ("__deployment", "__admin_connection")
    __deployment: PostgreSQLHistorySuccessorReproposalV1Deployment
    __admin_connection: PostgreSQLConnectionConfig | None

    def __init__(
        self,
        deployment: PostgreSQLHistorySuccessorReproposalV1Deployment,
        *,
        admin_connection: PostgreSQLConnectionConfig | None = None,
    ) -> None:
        if (
            type(deployment) is not PostgreSQLHistorySuccessorReproposalV1Deployment
            or admin_connection is not None
            and type(admin_connection) is not PostgreSQLConnectionConfig
        ):
            raise TypeError("EXACT_QUALIFIED_REPROPOSAL_DEPLOYMENT_REQUIRED")
        object.__setattr__(self, "_PostgreSQLHistorySuccessorReproposalV1__deployment", deployment)
        object.__setattr__(
            self, "_PostgreSQLHistorySuccessorReproposalV1__admin_connection", admin_connection
        )
        try:
            with self._connect() as conn, conn.transaction():
                qualify_history_successor_reproposal_v1(conn, deployment)
        except Exception:
            raise AtomicRootProofCommitUnavailable(
                "QUALIFIED_REPROPOSAL_COMPOSITION_UNAVAILABLE"
            ) from None

    def __init_subclass__(cls) -> None:
        raise TypeError("INTERNAL_REPROPOSAL_AUTHORITY_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("INTERNAL_REPROPOSAL_AUTHORITY_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return (
            "PostgreSQLHistorySuccessorReproposalV1(connection=<redacted>, production_ready=False)"
        )

    def _connect(self) -> psycopg.Connection[Any]:
        connection = self.__deployment.atomic_deployment.writer_connection
        return psycopg.connect(connection.dsn, autocommit=True)

    @property
    def _schema(self) -> str:
        return self.__deployment.atomic_deployment.provisioning.schema

    @staticmethod
    def _locator(
        value: PreparedAtomicRootProofCommit
        | AtomicRootProofCommitLocator
        | HistorySuccessorProposal,
    ) -> AtomicRootProofCommitLocator:
        if type(value) is HistorySuccessorProposal:
            return value.locator
        if type(value) is PreparedAtomicRootProofCommit:
            return value.locator
        if type(value) is AtomicRootProofCommitLocator:
            return value
        raise AtomicRootProofCommitConflict("EXACT_RECOVERY_LOCATOR_REQUIRED")

    @staticmethod
    def _proposal(value: object, locator: AtomicRootProofCommitLocator) -> HistorySuccessorProposal:
        if type(value) is not dict or type(value.get("package")) is not dict:
            raise AtomicRootProofCommitCorrupt("EXACT_RETAINED_PROPOSAL_REQUIRED")
        PostgreSQLAtomicRootProofCommit._require_identity(value["package"], locator)
        try:
            return HistorySuccessorProposal(
                locator,
                value["candidate_id"],
                value["revision"],
                value["predecessor_candidate_id"],
            )
        except (KeyError, TypeError):
            raise AtomicRootProofCommitCorrupt("RETAINED_PROPOSAL_IDENTITY_INVALID") from None

    def _read_proposal(
        self,
        conn: psycopg.Connection[Any],
        locator: AtomicRootProofCommitLocator,
        candidate_id: str | None,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            sql.SQL("SELECT {}.read_history_proposal(%s,%s)").format(sql.Identifier(self._schema)),
            (locator.attempt, candidate_id),
        ).fetchone()
        if row is None or len(row) != 1:
            raise AtomicRootProofCommitCorrupt("EXACT_PROPOSAL_READ_CARDINALITY_REQUIRED")
        if row[0] is None:
            return None
        self._proposal(row[0], locator)
        return cast(dict[str, Any], row[0])

    def read_current_proposal(
        self, value: PreparedAtomicRootProofCommit | AtomicRootProofCommitLocator
    ) -> HistorySuccessorProposal | None:
        """Protected read of the selector; this alone never resolves COMMIT absence."""
        locator = self._locator(value)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
                conn.execute("SET LOCAL search_path=pg_catalog")
                qualify_history_successor_reproposal_v1(conn, self.__deployment)
                result = self._read_proposal(conn, locator, None)
                return None if result is None else self._proposal(result, locator)
        except AtomicRootProofCommitError:
            raise
        except Exception:
            raise AtomicRootProofCommitUnavailable("QUALIFIED_PROPOSAL_READ_UNAVAILABLE") from None

    def _read_decision(
        self, conn: psycopg.Connection[Any], locator: AtomicRootProofCommitLocator
    ) -> AtomicRootProofCommitDecision | None:
        row = conn.execute(
            sql.SQL("SELECT {}.read_decision(%s,%s)").format(sql.Identifier(self._schema)),
            (locator.scope, locator.attempt),
        ).fetchone()
        if row is None or len(row) != 1:
            raise AtomicRootProofCommitCorrupt("EXACT_DECISION_READ_CARDINALITY_REQUIRED")
        return (
            None if row[0] is None else PostgreSQLAtomicRootProofCommit._decision(row[0], locator)
        )

    def repropose(
        self,
        original: PreparedAtomicRootProofCommit,
        binding: object,
        signed_attempt: object,
        *,
        signing_intent: PostgreSQLRootProofSigningIntentV2,
        retention: PostgreSQLRootProofRetentionV2,
        issuer_authority: PostgreSQLRootProofIssuanceAuthorityV2,
        history: PostgreSQLAuthenticatedIssuerHistoryV2,
        history_signer: LocalHistoryAttestationSigningProvider,
        fences: AtomicRootProofCommitFences,
        predecessor: HistorySuccessorProposal | None = None,
    ) -> HistorySuccessorProposal | AtomicRootProofCommitDecision:
        """Reverify original evidence, sign the actual successor, and admit via CAS.

        Positive existing decisions return unchanged. SQL serializes admission
        with every earlier final commit before checking absence and stale
        history. A disconnected approval leaves an INDETERMINATE result; an
        ordinary absent read never permits a replacement.
        """
        original_trust = _trusted_preparation(original)
        if self.__admin_connection is None:
            raise AtomicRootProofCommitConflict("SEPARATE_TEST_ADMIN_APPROVAL_PRINCIPAL_REQUIRED")
        if (
            type(signing_intent) is not PostgreSQLRootProofSigningIntentV2
            or type(retention) is not PostgreSQLRootProofRetentionV2
            or type(issuer_authority) is not PostgreSQLRootProofIssuanceAuthorityV2
            or type(history) is not PostgreSQLAuthenticatedIssuerHistoryV2
            or type(history_signer) is not LocalHistoryAttestationSigningProvider
            or type(fences) is not AtomicRootProofCommitFences
            or predecessor is not None
            and type(predecessor) is not HistorySuccessorProposal
            or predecessor is not None
            and predecessor.locator != original.locator
            or signing_intent._deployment is not self.__deployment.preparation_deployment
            or history._deployment is not self.__deployment.preparation_deployment
            or issuer_authority.deployment is not self.__deployment.preparation_deployment
        ):
            raise AtomicRootProofCommitConflict("EXACT_GENUINE_REPROPOSAL_AUTHORITIES_REQUIRED")
        attempted = False
        commit_sent = False
        try:
            # A positive marker can be recovered even after eligibility closes.
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
                conn.execute("SET LOCAL search_path=pg_catalog")
                qualify_history_successor_reproposal_v1(conn, self.__deployment)
                existing = self._read_decision(conn, original.locator)
                if existing is not None:
                    if (
                        existing.payload != original.prepared.canonical_payload
                        or existing.signature != original.prepared.signature
                        or existing.envelope != original.prepared.canonical_envelope
                    ):
                        raise AtomicRootProofCommitCorrupt("ORIGINAL_STAGED_BYTES_CHANGED")
                    return existing
                if predecessor is not None:
                    retained = self._read_proposal(conn, original.locator, predecessor.candidate_id)
                    if (
                        retained is None
                        or self._proposal(retained, original.locator) != predecessor
                    ):
                        raise AtomicRootProofCommitConflict(
                            "EXACT_RETAINED_PROPOSAL_PREDECESSOR_REQUIRED"
                        )
            before = history.authenticated_snapshot(history_signer)
            preparation = _reprepare_original(
                original,
                binding,
                signed_attempt,
                signing_intent=signing_intent,
                retention=retention,
                issuer_authority=issuer_authority,
                history=history,
            )
            trusted = _trusted_preparation(preparation)
            if _invariant_package(_base_package(original)) != _invariant_package(
                _base_package(preparation)
            ) or tuple(pin[1] for pin in original_trust.pins) != tuple(
                pin[1] for pin in trusted.pins
            ):
                raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_PROOF_AND_EVENT_REQUIRED")
            if history.authenticated_snapshot(history_signer) != before:
                raise AtomicRootProofCommitConflict(
                    "AUTHENTICATED_HISTORY_CHANGED_DURING_REPREPARATION"
                )
            record = preparation.history_record
            expected_sequence = 1 if not before.records else before.records[-1].sequence + 1
            expected_digest = (
                NO_PREDECESSOR if not before.records else before.records[-1].authenticated_digest
            )
            if (
                record.sequence != expected_sequence
                or record.predecessor_authenticated_digest != expected_digest
            ):
                raise AtomicRootProofCommitConflict(
                    "EXACT_AUTHENTICATED_HISTORY_SUCCESSOR_REQUIRED"
                )
            snapshot = history_signer.credential_snapshot()
            request = preparation.prepared.retained_request
            if request.requester is None or request.claimant is None:
                raise AtomicRootProofCommitCorrupt("EXACT_RETAINED_REQUEST_SIGNERS_REQUIRED")
            if (
                snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
                or snapshot.provider_identity.security
                != preparation.prepared.credential_snapshot.provider_identity.security
                or snapshot.key_material_identity
                in {
                    preparation.prepared.credential_snapshot.key_material_identity,
                    request.requester[0].key_material_identity,
                    request.claimant[0].key_material_identity,
                }
                or fences.root_key != preparation.root_fence_key
                or fences.root_generation
                != preparation.prepared.credential_snapshot.lifecycle_generation
                or fences.trust_key != preparation.trust_fence_key
            ):
                raise AtomicRootProofCommitConflict("EXACT_ACTIVE_SIGNER_FENCE_BINDING_REQUIRED")
            head = attest_head(record, history_signer)
            verify_head(head, history_signer, expected_stream=record.stream)
            if (
                snapshot.credential_identity != head.signing_credential_identity
                or fences.history_key != atomic_history_fence_key(head)
                or fences.history_generation != snapshot.lifecycle_generation
                or history_signer.credential_snapshot() != snapshot
            ):
                raise AtomicRootProofCommitConflict("EXACT_STABLE_HISTORY_SIGNER_FENCE_REQUIRED")
            _trusted_preparation(original)
            _trusted_preparation(preparation)
            package = _base_package(preparation)
            package.update(
                history_head={
                    "sequence": head.sequence,
                    "record_digest": head.record_digest,
                    "canonical_credential_hex": canonical_json_bytes(
                        _credential_material(head.signing_credential_identity)
                    ).hex(),
                },
                history_head_hex=head.canonical_attestation_bytes.hex(),
                history_signature_hex=head.signature.hex(),
                root_fence_key=fences.root_key,
                root_fence_generation=fences.root_generation,
                history_fence_key=fences.history_key,
                history_fence_generation=fences.history_generation,
                trust_fence_key=fences.trust_key,
                trust_fence_generation=fences.trust_generation,
            )
            atomic = self.__deployment.atomic_deployment
            config = atomic.provisioning
            schemas = tuple(
                getattr(config, component + "_schema")
                for component in (
                    "entitlement",
                    "intent",
                    "retention",
                    "history",
                    "requester",
                    "claimant",
                )
            )
            if tuple(pin[1] for pin in trusted.pins) != schemas:
                raise AtomicRootProofCommitConflict("EXACT_MIGRATED_AUTHORITY_PINS_REQUIRED")
            for connection, _ in trusted.pins:
                with psycopg.connect(connection.dsn, autocommit=True) as participant:
                    if (
                        read_atomic_root_proof_physical_identity(participant, config.instance_id)
                        != atomic.identity
                    ):
                        raise AtomicRootProofCommitConflict("ONE_PHYSICAL_DATABASE_REQUIRED")
            revision = 1 if predecessor is None else predecessor.revision + 1
            prior_id = None if predecessor is None else predecessor.candidate_id
            with psycopg.connect(self.__admin_connection.dsn, autocommit=True) as conn:
                conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
                conn.execute("SET LOCAL search_path=pg_catalog")
                qualify_history_successor_reproposal_v1(
                    conn, self.__deployment, expected_role="admin"
                )
                _persistence_cut("before_proposal_admission", conn)
                attempted = True
                row = conn.execute(
                    sql.SQL("SELECT {}.approve_history_proposal(%s,%s,%s)").format(
                        sql.Identifier(self._schema)
                    ),
                    (Jsonb(package), revision, prior_id),
                ).fetchone()
                if row is None or len(row) != 1 or type(row[0]) is not dict:
                    raise AtomicRootProofCommitCorrupt("EXACT_PROPOSAL_ADMISSION_REQUIRED")
                result: HistorySuccessorProposal | AtomicRootProofCommitDecision
                if "decision" in row[0]:
                    result = PostgreSQLAtomicRootProofCommit._decision(
                        row[0]["decision"], original.locator
                    )
                else:
                    result = self._proposal(row[0], original.locator)
                    if (
                        result.revision != revision
                        or result.predecessor_candidate_id != prior_id
                        or row[0]["package"] != package
                    ):
                        raise AtomicRootProofCommitCorrupt("EXACT_PROPOSAL_LINEAGE_REQUIRED")
                _persistence_cut("before_proposal_commit", conn)
                commit_sent = True
                conn.commit()
                _persistence_cut("after_proposal_commit", conn)
                return result
        except (
            psycopg.errors.SerializationFailure,
            psycopg.errors.UniqueViolation,
            psycopg.errors.CheckViolation,
            psycopg.errors.InvalidParameterValue,
            psycopg.errors.InsufficientPrivilege,
        ):
            if commit_sent:
                raise AtomicRootProofCommitIndeterminate(
                    "PROPOSAL_APPROVAL_OUTCOME_UNKNOWN"
                ) from None
            raise AtomicRootProofCommitConflict(
                "PROPOSAL_CAS_OR_LIVE_ELIGIBILITY_REJECTED"
            ) from None
        except AtomicRootProofCommitError:
            if commit_sent:
                raise AtomicRootProofCommitIndeterminate(
                    "PROPOSAL_APPROVAL_OUTCOME_UNKNOWN"
                ) from None
            raise
        except Exception:
            if attempted:
                raise AtomicRootProofCommitIndeterminate(
                    "PROPOSAL_APPROVAL_OUTCOME_UNKNOWN"
                ) from None
            raise AtomicRootProofCommitUnavailable("QUALIFIED_REPROPOSAL_UNAVAILABLE") from None

    def commit(self, proposal: HistorySuccessorProposal) -> AtomicRootProofCommitDecision:
        """Commit exactly this revision; never select a replacement by locator."""
        if type(proposal) is not HistorySuccessorProposal:
            raise AtomicRootProofCommitConflict("EXACT_VERSIONED_COMMIT_PROPOSAL_REQUIRED")
        attempted = False
        commit_sent = False
        conn: psycopg.Connection[Any] | None = None
        try:
            conn = self._connect()
            conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            qualify_history_successor_reproposal_v1(conn, self.__deployment)
            retained = self._read_proposal(conn, proposal.locator, proposal.candidate_id)
            if retained is None or self._proposal(retained, proposal.locator) != proposal:
                raise AtomicRootProofCommitConflict("EXACT_RETAINED_PROPOSAL_REQUIRED")
            attempted = True
            row = conn.execute(
                sql.SQL("SELECT {}.commit_history_proposal(%s,%s)").format(
                    sql.Identifier(self._schema)
                ),
                (proposal.locator.attempt, proposal.candidate_id),
            ).fetchone()
            if row is None or len(row) != 1:
                raise AtomicRootProofCommitCorrupt("EXACT_SQL_DECISION_CARDINALITY_REQUIRED")
            decision = PostgreSQLAtomicRootProofCommit._decision(row[0], proposal.locator)
            package = retained["package"]
            if (
                decision.payload.hex() != package["payload_hex"]
                or decision.signature.hex() != package["signature_hex"]
                or decision.envelope.hex() != package["staged_envelope_hex"]
            ):
                raise AtomicRootProofCommitCorrupt("ORIGINAL_STAGED_BYTES_CHANGED")
            _persistence_cut("before_commit", conn)
            commit_sent = True
            conn.commit()
            _persistence_cut("after_commit", conn)
            return decision
        except (
            psycopg.errors.SerializationFailure,
            psycopg.errors.UniqueViolation,
            psycopg.errors.CheckViolation,
            psycopg.errors.InvalidParameterValue,
            psycopg.errors.InsufficientPrivilege,
        ):
            if commit_sent:
                raise AtomicRootProofCommitIndeterminate(
                    "ATOMIC_COMMIT_PRIOR_OUTCOME_UNKNOWN"
                ) from None
            raise AtomicRootProofCommitConflict(
                "ATOMIC_PROPOSAL_CAS_OR_ELIGIBILITY_REJECTED"
            ) from None
        except AtomicRootProofCommitError:
            if commit_sent:
                raise AtomicRootProofCommitIndeterminate(
                    "ATOMIC_COMMIT_PRIOR_OUTCOME_UNKNOWN"
                ) from None
            raise
        except Exception:
            if attempted:
                raise AtomicRootProofCommitIndeterminate(
                    "ATOMIC_COMMIT_PRIOR_OUTCOME_UNKNOWN"
                ) from None
            raise AtomicRootProofCommitUnavailable(
                "QUALIFIED_ATOMIC_AUTHORITY_UNAVAILABLE"
            ) from None
        finally:
            if conn is not None:
                conn.close()

    def reconcile(
        self, value: HistorySuccessorProposal | AtomicRootProofCommitLocator
    ) -> AtomicRootProofCommitDecision:
        """Only a positive immutable decision resolves an unknown write outcome."""
        locator = self._locator(value)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
                conn.execute("SET LOCAL search_path=pg_catalog")
                qualify_history_successor_reproposal_v1(conn, self.__deployment)
                decision = self._read_decision(conn, locator)
                if decision is None:
                    raise AtomicRootProofCommitIndeterminate(
                        "DECISION_NOT_FOUND_PRIOR_OUTCOME_UNRESOLVED"
                    )
                return decision
        except AtomicRootProofCommitError:
            raise
        except Exception:
            raise AtomicRootProofCommitIndeterminate(
                "AUTHORITATIVE_RECONCILIATION_UNAVAILABLE"
            ) from None
