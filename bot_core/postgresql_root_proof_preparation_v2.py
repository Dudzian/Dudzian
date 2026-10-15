"""Qualified post-migration RESERVED/STAGED ports, without final issuance rights.

V2 changes catalog qualification and enrolled signer fencing, never the frozen
payload or retained envelopes. Preparation principals retain their original
single-operation SQL grants. Approval and atomic COMMIT remain separate ports.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, NoReturn, cast

import psycopg
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from psycopg import sql
from psycopg.errors import DataCorrupted, UniqueViolation

from bot_core.authenticated_issuer_history import (
    HistoryContractError,
    HistoryRecord,
    HistoryStreamIdentity,
    ReconciliationOutcome,
    _snapshot_stream,
    canonical_json_bytes,
    verify_head,
)
from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import AttemptAuthorization
from bot_core.licensing.canonical import parse_canonical
from bot_core.licensing.cha_account_reservation import VerifiedAccountGenesisInitialBinding
from bot_core.licensing.cha_root_proof_signed_attempt import (
    _read_issuer_preflight_source,
    _signed_attempt_lock_selector,
)
from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    LocalRootProofSigningProvider,
    SigningCredentialSnapshot,
    SigningKeyLifecycle,
)
from bot_core.postgresql_authenticated_issuer_history import (
    HistoryReadSnapshot,
    HistoryStorageUnavailable,
    _read_history_snapshot,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.postgresql_root_proof_preparation_readonly import (
    PostgreSQLRootProofPreparationV2Deployment,
    lock_root_proof_preparation_signer_v2,
    qualify_root_proof_preparation_v2,
)
from bot_core.postgresql_root_proof_retention import (
    ROOT_PROOF_SIGNING_DOMAIN,
    PreparedRootProof,
    RootProofRetentionConflict,
    RootProofRetentionCorrupt,
    RootProofRetentionError,
    RootProofRetentionIndeterminate,
    RootProofRetentionNotIssued,
    RootProofRetentionScope,
    RootProofRetentionUnavailable,
    _canonical,
    _envelope as _prepared_envelope,
    _exact_inputs,
    _historical_verification_unavailable,
    _scope_data,
    _snapshot_from_data,
    _validate_payload,
    _validate_snapshot,
)
from bot_core.postgresql_root_proof_signing_intent import (
    RootProofSigningIntent,
    RootProofSigningIntentConflict,
    RootProofSigningIntentCorrupt,
    RootProofSigningIntentError,
    RootProofSigningIntentIndeterminate,
    RootProofSigningIntentNotIssued,
    RootProofSigningIntentUnavailable,
    _envelope as _intent_envelope,
    _new_root_proof_id,
    _payload as _intent_payload,
    _stored_request,
)
from bot_core.root_proof_issuance_reconciliation import _failure
from bot_core.root_proof_issuer_runtime import RootProofIssuerRuntimeError, _verify_request
from bot_core.root_proof_issuer_substrate import SecurityProfile


def _persistence_cut(name: str) -> None:
    """Fault injection seam immediately around a preparation COMMIT."""


def _connect(connection: PostgreSQLConnectionConfig) -> psycopg.Connection[Any]:
    conn = psycopg.connect(connection.dsn, autocommit=True)
    try:
        conn.execute("SET search_path=pg_catalog")
    except Exception:
        conn.close()
        raise
    return conn


class PostgreSQLRootProofSigningIntentV2:
    """Exact genuine INITIAL_BINDING to immutable RESERVED after V2 migration."""

    __slots__ = ("_connection", "_schema", "_scope", "_signer", "_authority", "_deployment")
    _connection: PostgreSQLConnectionConfig
    _schema: str
    _scope: RootProofRetentionScope
    _signer: LocalRootProofSigningProvider
    _authority: PostgreSQLRootProofIssuanceAuthorityV2
    _deployment: PostgreSQLRootProofPreparationV2Deployment

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        deployment: PostgreSQLRootProofPreparationV2Deployment,
        scope: RootProofRetentionScope,
        verification_authority: LocalRootProofSigningProvider,
        issuer_authority: PostgreSQLRootProofIssuanceAuthorityV2,
    ) -> None:
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or type(deployment) is not PostgreSQLRootProofPreparationV2Deployment
            or type(schema) is not str
            or schema != deployment.provisioning.intent_schema
            or type(scope) is not RootProofRetentionScope
            or scope.trust_domain != deployment.provisioning.trust_domain
            or type(verification_authority) is not LocalRootProofSigningProvider
            or type(issuer_authority) is not PostgreSQLRootProofIssuanceAuthorityV2
        ):
            raise TypeError("EXACT_V2_SIGNING_INTENT_COMPOSITION_REQUIRED")
        for name, value in (
            ("connection", PostgreSQLConnectionConfig(connection.dsn)),
            ("schema", schema),
            ("scope", RootProofRetentionScope(**asdict(scope))),
            ("signer", verification_authority),
            ("authority", issuer_authority),
            ("deployment", deployment),
        ):
            object.__setattr__(self, "_" + name, value)
        try:
            subject = issuer_authority._entitlement_subject
            if (
                subject.environment != scope.environment
                or subject.trust_domain != scope.trust_domain
            ):
                raise ValueError("EXACT_ISSUER_SCOPE_REQUIRED")
            issuer_authority.requalify()
            _validate_snapshot(self._signer.credential_snapshot(), self._scope)
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
            if issuer_authority.deployment is not deployment:
                raise ValueError("EXACT_SHARED_V2_PUBLIC_DEPLOYMENT_REQUIRED")
        except Exception:
            raise RootProofSigningIntentUnavailable(
                "V2_SIGNING_INTENT_COMPOSITION_UNAVAILABLE"
            ) from None

    def __init_subclass__(cls) -> None:
        raise TypeError("V2_SIGNING_INTENT_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("V2_SIGNING_INTENT_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return f"PostgreSQLRootProofSigningIntentV2(connection=<redacted>, schema={self._schema!r})"

    def _connect(self) -> psycopg.Connection[Any]:
        return _connect(self._connection)

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        qualify_root_proof_preparation_v2(conn, self._deployment, component="intent")

    def _inputs(
        self, binding: object, signed_attempt: object
    ) -> tuple[AttemptAuthorization, RetainedIssuanceRequest]:
        try:
            if type(binding) is not VerifiedAccountGenesisInitialBinding:
                raise RootProofSigningIntentConflict("GENUINE_INITIAL_BINDING_REQUIRED")
            source, current, retained = _read_issuer_preflight_source(signed_attempt)
            if source.binding is not binding or source.provider is not self._authority:
                raise RootProofSigningIntentConflict("EXACT_PINNED_BINDING_AND_AUTHORITY_REQUIRED")
            self._authority.requalify()
            _verify_request(source, current, retained)
            if (
                source.authorization.environment != self._scope.environment
                or source.authorization.trust_domain != self._scope.trust_domain
                or source.authorization.product_scope != self._scope.product_scope
            ):
                raise RootProofSigningIntentConflict("EXACT_V2_SIGNING_INTENT_SCOPE_REQUIRED")
            return cast(
                tuple[AttemptAuthorization, RetainedIssuanceRequest],
                _exact_inputs(self._scope, source.authorization, retained),
            )
        except RootProofSigningIntentError:
            raise
        except RootProofIssuerRuntimeError as error:
            if str(error) in {
                "EXACT_PUBLIC_ISSUER_AUTHORITY_REQUIRED",
                "EXACT_CURRENT_SIGNED_ATTEMPT_REQUIRED",
                "EXACT_IMMUTABLE_REQUEST_REQUIRED",
                "EXACT_IMMUTABLE_ATTEMPT_IDENTITY_REQUIRED",
                "EXACT_ACTIVE_UNBOUND_ENTITLEMENT_REQUIRED",
                "EXACT_ACTIVE_PUBLIC_SIGNER_REQUIRED",
                "REQUESTER_CLAIMANT_ALIAS",
            }:
                raise RootProofSigningIntentConflict("V2_LIVE_ATTEMPT_IDENTITY_CONFLICT") from None
            raise RootProofSigningIntentIndeterminate(
                "V2_LIVE_ATTEMPT_REVERIFICATION_REFUSED"
            ) from None
        except Exception as error:
            outcome = _failure("V2_SIGNING_INTENT_INPUT", error).outcome
            error_type = {
                ReconciliationOutcome.CONFLICT: RootProofSigningIntentConflict,
                ReconciliationOutcome.CORRUPT: RootProofSigningIntentCorrupt,
                ReconciliationOutcome.INDETERMINATE: RootProofSigningIntentIndeterminate,
            }.get(outcome, RootProofSigningIntentUnavailable)
            raise error_type("GENUINE_V2_SIGNED_ATTEMPT_REVERIFICATION_REFUSED") from None

    def _selector(self, binding: object, signed_attempt: object) -> str:
        try:
            if type(binding) is not VerifiedAccountGenesisInitialBinding:
                raise TypeError("GENUINE_INITIAL_BINDING_REQUIRED")
            return cast(str, _signed_attempt_lock_selector(signed_attempt))
        except Exception:
            raise RootProofSigningIntentConflict("GENUINE_LOCK_SELECTOR_REQUIRED") from None

    def _lock_attempt(self, conn: psycopg.Connection[Any], selector: str) -> None:
        # Share the legacy mutex: the physical immutable rows remain V1 bytes.
        key = int.from_bytes(
            hashlib.sha256(b"RootProofSigningIntentV1\x00" + selector.encode()).digest()[:8],
            "big",
            signed=True,
        )
        conn.execute("SELECT pg_catalog.pg_advisory_xact_lock(%s)", (key,))

    def _snapshot(self) -> SigningCredentialSnapshot:
        try:
            snapshot = self._signer.credential_snapshot()
            _validate_snapshot(snapshot, self._scope)
        except Exception as error:
            if _historical_verification_unavailable(error):
                raise RootProofSigningIntentUnavailable("V2_PINNED_SIGNER_UNAVAILABLE") from None
            raise RootProofSigningIntentCorrupt("V2_PINNED_SIGNER_IDENTITY_REJECTED") from None
        if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
            raise RootProofSigningIntentConflict("V2_RESERVATION_REQUIRES_ACTIVE_SIGNER")
        return snapshot

    def _read_row(self, conn: psycopg.Connection[Any], request: RetainedIssuanceRequest) -> object:
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload FROM {}.read_intent(%s,%s)").format(
                sql.Identifier(self._schema)
            ),
            (_canonical(_scope_data(self._scope)), _canonical(request.attempt_id)),
        ).fetchall()
        if len(rows) > 1:
            raise RootProofSigningIntentCorrupt("V2_INTENT_CARDINALITY_INVALID")
        return None if not rows else rows[0]

    def _restore(
        self, row: object, auth: AttemptAuthorization, request: RetainedIssuanceRequest
    ) -> RootProofSigningIntent:
        try:
            if type(row) is not tuple or len(row) != 2 or any(type(v) is not bytes for v in row):
                raise ValueError("EXACT_STORED_BYTEA_REQUIRED")
            envelope, payload = row
            document = parse_canonical(envelope)
            scope = RootProofRetentionScope(
                **{**document["scope"], "security_profile": self._scope.security_profile}
            )
            stored_auth, stored_request = _exact_inputs(
                scope,
                AttemptAuthorization(**document["authorization"]),
                _stored_request(document["retained_request"]),
            )
            snapshot = _snapshot_from_data(document["credential_snapshot"])
            _validate_snapshot(snapshot, scope)
            root_id = document["root_proof_id"]
            if (
                snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
                or payload != _intent_payload(scope, stored_auth, stored_request, snapshot, root_id)
                or envelope
                != _intent_envelope(scope, stored_auth, stored_request, snapshot, root_id, payload)
            ):
                raise ValueError("EXACT_ORIGINAL_RESERVED_BYTES_REQUIRED")
        except Exception:
            raise RootProofSigningIntentCorrupt("V2_RETAINED_INTENT_REJECTED") from None
        if scope != self._scope or stored_auth != auth or stored_request != request:
            raise RootProofSigningIntentConflict("EXACT_RETRY_INPUT_IDENTITY_REQUIRED")
        return RootProofSigningIntent(scope, auth, request, envelope, payload, snapshot, root_id)

    def _read_retained(
        self, auth: AttemptAuthorization, request: RetainedIssuanceRequest
    ) -> RootProofSigningIntent | None:
        """Original immutable linkage for historical STAGED reads, without new grants."""
        auth, request = _exact_inputs(self._scope, auth, request)
        with self._connect() as conn, conn.transaction():
            self._qualify(conn)
            row = self._read_row(conn, request)
            return None if row is None else self._restore(row, auth, request)

    def read_intent(self, binding: object, signed_attempt: object) -> RootProofSigningIntent | None:
        selector = self._selector(binding, signed_attempt)
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                self._lock_attempt(conn, selector)
                auth, request = self._inputs(binding, signed_attempt)
                if request.attempt_id != selector:
                    raise RootProofSigningIntentConflict("VERIFIED_ATTEMPT_SELECTOR_REQUIRED")
                snapshot = self._snapshot()
                lock_root_proof_preparation_signer_v2(conn, self._deployment, snapshot)
                row = self._read_row(conn, request)
                value = None if row is None else self._restore(row, auth, request)
                if value is not None and value.credential_snapshot != snapshot:
                    raise RootProofSigningIntentConflict("EXACT_ORIGINAL_SIGNER_REQUIRED")
                if (
                    self._inputs(binding, signed_attempt) != (auth, request)
                    or self._snapshot() != snapshot
                ):
                    raise RootProofSigningIntentConflict("LIVE_IDENTITIES_CHANGED_DURING_READ")
                return value
        except RootProofSigningIntentError:
            raise
        except DataCorrupted:
            raise RootProofSigningIntentCorrupt("V2_INTENT_KEYS_REJECTED") from None
        except Exception:
            raise RootProofSigningIntentUnavailable("V2_INTENT_READ_UNAVAILABLE") from None

    def reserve(self, binding: object, signed_attempt: object) -> RootProofSigningIntent:
        selector = self._selector(binding, signed_attempt)
        write_attempted = False
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                self._lock_attempt(conn, selector)
                auth, request = self._inputs(binding, signed_attempt)
                if request.attempt_id != selector:
                    raise RootProofSigningIntentConflict("VERIFIED_ATTEMPT_SELECTOR_REQUIRED")
                snapshot = self._snapshot()
                lock_root_proof_preparation_signer_v2(conn, self._deployment, snapshot)
                row = self._read_row(conn, request)
                if row is not None:
                    value = self._restore(row, auth, request)
                    if value.credential_snapshot != snapshot:
                        raise RootProofSigningIntentConflict("EXACT_ORIGINAL_SIGNER_REQUIRED")
                    if (
                        self._inputs(binding, signed_attempt) != (auth, request)
                        or self._snapshot() != snapshot
                    ):
                        raise RootProofSigningIntentConflict("LIVE_ATTEMPT_CHANGED_DURING_RETRY")
                    return value
                root_id = _new_root_proof_id()
                payload = _intent_payload(self._scope, auth, request, snapshot, root_id)
                envelope = _intent_envelope(self._scope, auth, request, snapshot, root_id, payload)
                write_attempted = True
                rows = conn.execute(
                    sql.SQL(
                        "SELECT envelope,payload FROM {}.reserve_intent(%s,%s,%s,%s,%s,%s,%s)"
                    ).format(sql.Identifier(self._schema)),
                    (
                        _canonical(_scope_data(self._scope)),
                        _canonical(request.attempt_id),
                        _canonical(auth.logical_operation_id),
                        _canonical(
                            {
                                "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
                                "entitlement_generation": auth.entitlement_generation,
                            }
                        ),
                        _canonical(root_id),
                        envelope,
                        payload,
                    ),
                ).fetchall()
                if len(rows) != 1:
                    raise RootProofSigningIntentCorrupt("V2_INTENT_WRITE_CARDINALITY_INVALID")
                value = self._restore(rows[0], auth, request)
                if value.canonical_envelope != envelope or value.canonical_payload != payload:
                    raise RootProofSigningIntentConflict("EXACT_RETRY_BYTES_REQUIRED")
                if (
                    self._inputs(binding, signed_attempt) != (auth, request)
                    or self._snapshot() != snapshot
                ):
                    raise RootProofSigningIntentConflict("LIVE_IDENTITIES_CHANGED_BEFORE_COMMIT")
                _persistence_cut("RESERVED_BEFORE_COMMIT")
            _persistence_cut("RESERVED_AFTER_COMMIT")
            return value
        except RootProofSigningIntentError:
            raise
        except UniqueViolation:
            raise RootProofSigningIntentConflict("V2_INTENT_IDENTITY_ALREADY_FENCED") from None
        except DataCorrupted:
            raise RootProofSigningIntentCorrupt("V2_INTENT_KEYS_REJECTED") from None
        except Exception:
            if write_attempted:
                raise RootProofSigningIntentIndeterminate(
                    "V2_INTENT_WRITE_OUTCOME_UNKNOWN"
                ) from None
            raise RootProofSigningIntentUnavailable("V2_INTENT_WRITE_UNAVAILABLE") from None

    def require_issued(self, *args: object, **kwargs: object) -> NoReturn:
        raise RootProofSigningIntentNotIssued("RESERVED_INTENT_IS_NOT_ISSUED")


class PostgreSQLRootProofRetentionV2:
    """Original genuine RESERVED and signature to exactly linked immutable STAGED."""

    __slots__ = (
        "_connection",
        "_schema",
        "_scope",
        "_verification_authority",
        "_signing_intent",
        "_deployment",
    )
    _connection: PostgreSQLConnectionConfig
    _schema: str
    _scope: RootProofRetentionScope
    _verification_authority: LocalRootProofSigningProvider
    _signing_intent: PostgreSQLRootProofSigningIntentV2
    _deployment: PostgreSQLRootProofPreparationV2Deployment

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        deployment: PostgreSQLRootProofPreparationV2Deployment,
        scope: RootProofRetentionScope,
        verification_authority: LocalRootProofSigningProvider,
        signing_intent: PostgreSQLRootProofSigningIntentV2,
    ) -> None:
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or type(deployment) is not PostgreSQLRootProofPreparationV2Deployment
            or type(schema) is not str
            or schema != deployment.provisioning.retention_schema
            or type(scope) is not RootProofRetentionScope
            or type(verification_authority) is not LocalRootProofSigningProvider
            or type(signing_intent) is not PostgreSQLRootProofSigningIntentV2
            or signing_intent._signer is not verification_authority
            or signing_intent._scope != scope
            or signing_intent._deployment is not deployment
        ):
            raise TypeError("EXACT_V2_RETENTION_COMPOSITION_REQUIRED")
        for name, value in (
            ("connection", PostgreSQLConnectionConfig(connection.dsn)),
            ("schema", schema),
            ("scope", RootProofRetentionScope(**asdict(scope))),
            ("verification_authority", verification_authority),
            ("signing_intent", signing_intent),
            ("deployment", deployment),
        ):
            object.__setattr__(self, "_" + name, value)
        try:
            _validate_snapshot(verification_authority.credential_snapshot(), scope)
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
        except Exception:
            raise RootProofRetentionUnavailable("V2_RETENTION_COMPOSITION_UNAVAILABLE") from None

    def __init_subclass__(cls) -> None:
        raise TypeError("V2_RETENTION_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("V2_RETENTION_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return f"PostgreSQLRootProofRetentionV2(connection=<redacted>, schema={self._schema!r})"

    def _connect(self) -> psycopg.Connection[Any]:
        return _connect(self._connection)

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        qualify_root_proof_preparation_v2(conn, self._deployment, component="retention")

    def _read_row(self, conn: psycopg.Connection[Any], request: RetainedIssuanceRequest) -> object:
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload,signature FROM {}.read_prepared(%s,%s)").format(
                sql.Identifier(self._schema)
            ),
            (_canonical(_scope_data(self._scope)), _canonical(request.attempt_id)),
        ).fetchall()
        if len(rows) > 1:
            raise RootProofRetentionCorrupt("V2_RETENTION_CARDINALITY_INVALID")
        return None if not rows else rows[0]

    def _restore(
        self, row: object, auth: AttemptAuthorization, request: RetainedIssuanceRequest
    ) -> PreparedRootProof:
        try:
            if type(row) is not tuple or len(row) != 3 or any(type(v) is not bytes for v in row):
                raise ValueError("EXACT_STORED_BYTEA_REQUIRED")
            envelope, payload, signature = row
            document = parse_canonical(envelope)
            snapshot = _snapshot_from_data(document["credential_snapshot"])
            _validate_snapshot(snapshot, self._scope)
            fields = _validate_payload(payload, signature, self._scope, auth, request, snapshot)
            if (
                snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
                or envelope
                != _prepared_envelope(
                    self._scope, auth, request, payload, signature, snapshot, fields
                )
            ):
                raise ValueError("EXACT_ORIGINAL_STAGED_ENVELOPE_REQUIRED")
            Ed25519PublicKey.from_public_bytes(snapshot.public_key).verify(
                signature, ROOT_PROOF_SIGNING_DOMAIN + payload
            )
        except Exception:
            raise RootProofRetentionCorrupt("V2_RETAINED_ROOT_PROOF_EVIDENCE_REJECTED") from None
        try:
            if (
                self._verification_authority.verify_historical(
                    ROOT_PROOF_SIGNING_DOMAIN + payload, signature, snapshot
                )
                is not True
            ):
                raise ValueError("PINNED_ROOT_PROOF_SIGNATURE_REQUIRED")
        except Exception as error:
            if _historical_verification_unavailable(error):
                raise RootProofRetentionUnavailable("V2_RETAINED_CUSTODY_UNAVAILABLE") from None
            raise RootProofRetentionCorrupt("V2_RETAINED_ROOT_PROOF_EVIDENCE_REJECTED") from None
        reserved = self._signing_intent._read_retained(auth, request)
        if (
            reserved is None
            or reserved.canonical_payload != payload
            or reserved.credential_snapshot != snapshot
            or reserved.root_proof_id != fields["root_proof_id"]
        ):
            raise RootProofRetentionConflict("EXACT_ORIGINAL_RESERVED_STAGED_LINKAGE_REQUIRED")
        return PreparedRootProof(
            self._scope,
            auth,
            request,
            envelope,
            payload,
            signature,
            snapshot,
            fields["root_proof_id"],
        )

    def read_prepared(
        self, *, authorization: AttemptAuthorization, retained_request: RetainedIssuanceRequest
    ) -> PreparedRootProof | None:
        try:
            auth, request = _exact_inputs(self._scope, authorization, retained_request)
        except Exception:
            raise RootProofRetentionConflict("V2_RETENTION_INPUT_BINDING_REJECTED") from None
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                row = self._read_row(conn, request)
                return None if row is None else self._restore(row, auth, request)
        except RootProofRetentionError:
            raise
        except Exception:
            raise RootProofRetentionUnavailable("V2_RETENTION_READ_UNAVAILABLE") from None

    def retain_prepared(
        self,
        binding: object,
        signed_attempt: object,
        *,
        canonical_payload: bytes,
        signature: bytes,
        canonical_envelope: bytes | None = None,
    ) -> PreparedRootProof:
        write_attempted = False
        try:
            auth, request = self._signing_intent._inputs(binding, signed_attempt)
            reserved = self._signing_intent.read_intent(binding, signed_attempt)
            if reserved is None:
                raise RootProofRetentionConflict("GENUINE_RESERVED_REQUIRED")
            snapshot = reserved.credential_snapshot
            if (
                type(canonical_payload) is not bytes
                or canonical_payload != reserved.canonical_payload
            ):
                raise RootProofRetentionConflict("EXACT_RESERVED_PAYLOAD_REQUIRED")
            fields = _validate_payload(
                canonical_payload, signature, self._scope, auth, request, snapshot
            )
            envelope = _prepared_envelope(
                self._scope, auth, request, canonical_payload, signature, snapshot, fields
            )
            if canonical_envelope is not None and (
                type(canonical_envelope) is not bytes or canonical_envelope != envelope
            ):
                raise RootProofRetentionConflict("EXACT_ORIGINAL_STAGED_ENVELOPE_REQUIRED")
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                self._signing_intent._lock_attempt(conn, request.attempt_id)
                lock_root_proof_preparation_signer_v2(conn, self._deployment, snapshot)
                if self._verification_authority.credential_snapshot() != snapshot:
                    raise RootProofRetentionConflict("EXACT_ORIGINAL_SIGNER_REQUIRED")
                row = self._read_row(conn, request)
                if row is not None:
                    original = self._restore(row, auth, request)
                    if (
                        original.canonical_envelope != envelope
                        or original.canonical_payload != canonical_payload
                        or original.signature != signature
                    ):
                        raise RootProofRetentionConflict("EXACT_RETRY_BYTES_REQUIRED")
                    if (
                        self._signing_intent._inputs(binding, signed_attempt) != (auth, request)
                        or self._verification_authority.credential_snapshot() != snapshot
                    ):
                        raise RootProofRetentionConflict("LIVE_IDENTITIES_CHANGED_DURING_RETRY")
                    return original
                if (
                    self._verification_authority.verify_historical(
                        ROOT_PROOF_SIGNING_DOMAIN + canonical_payload, signature, snapshot
                    )
                    is not True
                ):
                    raise RootProofRetentionConflict("EXACT_ROOT_PROOF_SIGNATURE_REQUIRED")
                write_attempted = True
                rows = conn.execute(
                    sql.SQL(
                        "SELECT envelope,payload,signature FROM {}.retain_prepared(%s,%s,%s,%s,%s,%s,%s,%s)"
                    ).format(sql.Identifier(self._schema)),
                    (
                        _canonical(_scope_data(self._scope)),
                        _canonical(request.attempt_id),
                        _canonical(auth.logical_operation_id),
                        _canonical(
                            {
                                "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
                                "entitlement_generation": auth.entitlement_generation,
                            }
                        ),
                        _canonical(reserved.root_proof_id),
                        envelope,
                        canonical_payload,
                        signature,
                    ),
                ).fetchall()
                if len(rows) != 1:
                    raise RootProofRetentionCorrupt("V2_RETENTION_WRITE_CARDINALITY_INVALID")
                value = self._restore(rows[0], auth, request)
                if (
                    value.canonical_envelope != envelope
                    or value.canonical_payload != canonical_payload
                    or value.signature != signature
                ):
                    raise RootProofRetentionConflict("EXACT_RETRY_BYTES_REQUIRED")
                if (
                    self._signing_intent._inputs(binding, signed_attempt) != (auth, request)
                    or self._verification_authority.credential_snapshot() != snapshot
                ):
                    raise RootProofRetentionConflict("LIVE_IDENTITIES_CHANGED_BEFORE_COMMIT")
                _persistence_cut("STAGED_BEFORE_COMMIT")
            _persistence_cut("STAGED_AFTER_COMMIT")
            return value
        except RootProofRetentionError:
            raise
        except RootProofSigningIntentConflict:
            raise RootProofRetentionConflict("GENUINE_RESERVED_REVERIFICATION_REFUSED") from None
        except RootProofSigningIntentCorrupt:
            raise RootProofRetentionCorrupt("GENUINE_RESERVED_EVIDENCE_REJECTED") from None
        except RootProofSigningIntentIndeterminate:
            raise RootProofRetentionIndeterminate(
                "GENUINE_RESERVED_REVERIFICATION_INDETERMINATE"
            ) from None
        except RootProofSigningIntentUnavailable:
            raise RootProofRetentionUnavailable(
                "GENUINE_RESERVED_REVERIFICATION_UNAVAILABLE"
            ) from None
        except UniqueViolation:
            raise RootProofRetentionConflict("V2_RETENTION_IDENTITY_ALREADY_FENCED") from None
        except (ValueError, TypeError):
            if write_attempted:
                raise RootProofRetentionIndeterminate(
                    "V2_RETENTION_WRITE_OUTCOME_UNKNOWN"
                ) from None
            raise RootProofRetentionConflict("V2_ROOT_PROOF_CANDIDATE_REJECTED") from None
        except Exception:
            if write_attempted:
                raise RootProofRetentionIndeterminate(
                    "V2_RETENTION_WRITE_OUTCOME_UNKNOWN"
                ) from None
            raise RootProofRetentionUnavailable("V2_RETENTION_WRITE_UNAVAILABLE") from None

    def require_issued(self, **kwargs: Any) -> NoReturn:
        raise RootProofRetentionNotIssued("STAGED_ROOT_PROOF_IS_NOT_ISSUED")


class PostgreSQLAuthenticatedIssuerHistoryV2:
    """Qualified read-only frozen history projection for unsigned approval prep."""

    __slots__ = ("_connection", "_schema", "_stream", "_deployment")
    _connection: PostgreSQLConnectionConfig
    _schema: str
    _stream: HistoryStreamIdentity
    _deployment: PostgreSQLRootProofPreparationV2Deployment

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        deployment: PostgreSQLRootProofPreparationV2Deployment,
        stream: HistoryStreamIdentity,
    ) -> None:
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or type(deployment) is not PostgreSQLRootProofPreparationV2Deployment
            or type(schema) is not str
            or schema != deployment.provisioning.history_schema
        ):
            raise TypeError("EXACT_V2_HISTORY_READ_COMPOSITION_REQUIRED")
        selected = _snapshot_stream(stream)
        if (
            selected.security_profile != SecurityProfile.PRODUCTION_LOCAL.value
            or selected.trust_domain != deployment.provisioning.trust_domain
        ):
            raise HistoryContractError("EXACT_V2_HISTORY_SCOPE_REQUIRED")
        for name, value in (
            ("connection", PostgreSQLConnectionConfig(connection.dsn)),
            ("schema", schema),
            ("stream", selected),
            ("deployment", deployment),
        ):
            object.__setattr__(self, "_" + name, value)
        self.retained_history()

    def __init_subclass__(cls) -> None:
        raise TypeError("V2_HISTORY_READER_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("V2_HISTORY_READER_IS_IMMUTABLE")

    def _connect(self) -> psycopg.Connection[Any]:
        return _connect(self._connection)

    def retained_history(self) -> tuple[HistoryRecord, ...]:
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                qualify_root_proof_preparation_v2(conn, self._deployment, component="history")
                return _read_history_snapshot(
                    conn,
                    self._schema,
                    self._stream,
                    canonical_json_bytes(self._stream.material()),
                    lock=False,
                ).records
        except HistoryContractError:
            raise
        except Exception:
            raise HistoryStorageUnavailable("V2_QUALIFIED_HISTORY_READ_UNAVAILABLE") from None

    def current_record(self) -> HistoryRecord | None:
        records = self.retained_history()
        return records[-1] if records else None

    def authenticated_snapshot(
        self, history_signer: LocalHistoryAttestationSigningProvider
    ) -> HistoryReadSnapshot:
        """Authenticate the actual retained tail against genuine pinned custody.

        The decoded chain alone is not evidence of its signer's authority. This
        read verifies every retained public attestation and requires a signed
        tail, using public keys and lifecycle from the exact local provider.
        No caller-supplied head, key, credential or signature participates.
        """
        if type(history_signer) is not LocalHistoryAttestationSigningProvider:
            raise HistoryContractError("GENUINE_PINNED_HISTORY_SIGNER_REQUIRED")
        try:
            before = history_signer.credential_snapshot()
            if before.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                raise HistoryContractError("REPROPOSAL_REQUIRES_ACTIVE_HISTORY_SIGNER")
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                qualify_root_proof_preparation_v2(conn, self._deployment, component="history")
                snapshot = _read_history_snapshot(
                    conn,
                    self._schema,
                    self._stream,
                    canonical_json_bytes(self._stream.material()),
                    lock=False,
                )
                for head in snapshot.heads:
                    verify_head(head, history_signer, expected_stream=self._stream)
                if snapshot.records and not any(
                    head.sequence == snapshot.records[-1].sequence
                    and head.record_digest == snapshot.records[-1].authenticated_digest
                    for head in snapshot.heads
                ):
                    raise HistoryContractError("AUTHENTICATED_RETAINED_HISTORY_TAIL_REQUIRED")
                if history_signer.credential_snapshot() != before:
                    raise HistoryContractError("HISTORY_SIGNER_CHANGED_DURING_RETAINED_READ")
                return HistoryReadSnapshot(self._stream, snapshot.records, snapshot.heads)
        except HistoryContractError:
            raise
        except Exception:
            raise HistoryStorageUnavailable("V2_AUTHENTICATED_HISTORY_READ_UNAVAILABLE") from None
