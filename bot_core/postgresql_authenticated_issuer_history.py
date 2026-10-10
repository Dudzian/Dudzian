"""Durable PRODUCTION_LOCAL implementation of the frozen issuer-history contract.

Only a separately provisioned PostgreSQL stream is admitted.  Opaque history
events confer no permission to issue Root Proof, bind entitlement, or advance a
checkpoint.  Signing stays behind the existing history-attestation provider;
this database retains public attestations and never receives private keys.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

import psycopg
from psycopg import sql
from psycopg.types.json import Json

from bot_core.authenticated_issuer_history import (
    ATTESTATION_DOMAIN,
    GENESIS_SEQUENCE,
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoricalRevokedSignatureVerificationUnavailable,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryRecord,
    HistorySigningAuthorityUnavailable,
    HistoryStreamIdentity,
    LocalCheckpointProvider,
    ReconciliationOutcome,
    VerifiedHistoryHead,
    _compare_verified_checkpoint,
    _credential_material,
    _plain,
    _snapshot_event_identity,
    _snapshot_head,
    _snapshot_record,
    _snapshot_stream,
    build_record,
    canonical_json_bytes,
    verify_head,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_schema import (
    HistorySchemaQualificationError,
    qualify_history_schema,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    HistoryAttestationSigningProvider,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)

_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_T = TypeVar("_T")


class HistoryStorageUnavailable(HistoryContractError):
    """PostgreSQL could not supply authoritative history; never means absence."""


class HistoryOperationIndeterminate(HistoryContractError):
    """A write response was lost; resolve with exact identity replay, not issuance."""


def _persistence_cut(name: str) -> None:
    """Crash-test seam; production has no behavior at transaction boundaries."""


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise HistoryContractError(f"malformed retained {label}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryContractError("duplicate canonical JSON key")
        result[key] = value
    return result


def _reject_noncanonical_number(value: str) -> object:
    raise HistoryContractError("noncanonical retained JSON number")


def _parse_canonical(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_float=_reject_noncanonical_number,
            parse_constant=_reject_noncanonical_number,
        )
        if type(value) is not dict or canonical_json_bytes(value) != raw:
            raise HistoryContractError("noncanonical retained JSON bytes")
        return value
    except (ValueError, TypeError, UnicodeError) as exc:
        raise HistoryContractError("invalid retained canonical JSON") from exc


def _unhex(value: object) -> bytes:
    if type(value) is not str:
        raise HistoryContractError("retained bytes must have an exact hex envelope")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise HistoryContractError("malformed retained byte envelope") from exc
    if decoded.hex() != value:
        raise HistoryContractError("noncanonical retained byte envelope")
    return decoded


def _record_from_row(row: object, stream: HistoryStreamIdentity) -> HistoryRecord:
    envelope = _object(
        row,
        {
            "sequence",
            "predecessor_authenticated_digest",
            "event_identity_hex",
            "event_digest",
            "authenticated_digest",
            "canonical_record",
        },
        "record envelope",
    )
    raw = _unhex(envelope["canonical_record"])
    material = _object(
        _parse_canonical(raw),
        {
            "stream",
            "sequence",
            "predecessor_authenticated_digest",
            "event_identity",
            "canonical_event_payload",
            "event_digest",
        },
        "record",
    )
    stored_stream = HistoryStreamIdentity(**material["stream"])
    event = HistoryEventIdentity(**material["event_identity"])
    if (
        stored_stream != stream
        or canonical_json_bytes(event.material()) != _unhex(envelope["event_identity_hex"])
        or any(
            envelope[key] != material[key]
            for key in ("sequence", "predecessor_authenticated_digest", "event_digest")
        )
        or type(envelope["sequence"]) is not int
    ):
        raise HistoryContractError("physical history envelope contradicts canonical record")
    return HistoryRecord(
        stored_stream,
        material["sequence"],
        material["predecessor_authenticated_digest"],
        event,
        material["canonical_event_payload"],
        material["event_digest"],
        envelope["authenticated_digest"],
    )


def _head_from_row(row: object, stream: HistoryStreamIdentity) -> AttestedHistoryHead:
    envelope = _object(
        row,
        {"sequence", "record_digest", "canonical_credential_hex", "canonical_head", "signature"},
        "head envelope",
    )
    credential = _object(
        _parse_canonical(_unhex(envelope["canonical_credential_hex"])),
        {
            "semantic_role",
            "credential_identity",
            "provider_namespace",
            "key_handle_or_version",
            "custody_lifecycle_namespace",
            "key_material_identity",
        },
        "credential",
    )
    identity = CredentialRoleIdentity(
        **{**credential, "semantic_role": CredentialSemanticRole(credential["semantic_role"])}
    )
    canonical = _unhex(envelope["canonical_head"])
    expected = ATTESTATION_DOMAIN + canonical_json_bytes(
        {
            "stream": stream.material(),
            "sequence": envelope["sequence"],
            "record_digest": envelope["record_digest"],
            "signing_credential_identity": _credential_material(identity),
        }
    )
    if canonical != expected:
        raise HistoryContractError("physical head envelope contradicts canonical attestation")
    return AttestedHistoryHead(
        _snapshot_stream(stream),
        envelope["sequence"],
        envelope["record_digest"],
        identity,
        canonical,
        _unhex(envelope["signature"]),
    )


@dataclass(frozen=True, slots=True)
class _HistorySnapshot:
    records: tuple[HistoryRecord, ...]
    heads: tuple[AttestedHistoryHead, ...]


class PostgreSQLAuthenticatedIssuerHistory:
    """Reviewed runtime history port with database CAS and complete-chain reads.

    Connections must log in directly as the offline-installed runtime role.
    Every operation rechecks schema, principal, ACL and database durability.
    A database-wide privileged rollback to a valid prefix remains outside this
    store's detection boundary; it needs an independent durable checkpoint.
    """

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        stream: HistoryStreamIdentity,
    ) -> None:
        if type(connection) is not PostgreSQLConnectionConfig:
            raise TypeError("connection must be an exact PostgreSQLConnectionConfig")
        if type(schema) is not str or _SAFE_IDENTIFIER.fullmatch(schema) is None:
            raise ValueError("schema must be a safe lowercase PostgreSQL identifier")
        self._stream = _snapshot_stream(stream)
        if self._stream.security_profile != SecurityProfile.PRODUCTION_LOCAL.value:
            raise HistoryContractError("durable issuer history requires PRODUCTION_LOCAL")
        self._connection = connection
        self._schema = schema
        self._stream_bytes = canonical_json_bytes(self._stream.material())
        self._read(lambda snapshot: None)

    def __init_subclass__(cls) -> None:
        raise TypeError("production issuer history cannot be subclassed")

    def __repr__(self) -> str:
        return (
            "PostgreSQLAuthenticatedIssuerHistory(connection=<redacted>, "
            f"schema={self._schema!r}, stream={self._stream!r})"
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._connection.dsn, autocommit=True)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            ProviderRole.ISSUER_AUTHENTICATED_HISTORY,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, self._stream.trust_domain),
            f"postgresql:{self._schema}",
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        # These are implementation facts. Live qualification still runs at each
        # authority operation; a DTO does not authorize semantic issuance.
        return ProviderCapabilities(
            True, authoritative_reads=True, durable_state=True, compare_and_swap=True
        )

    def credential_identities(self) -> tuple[()]:
        return ()

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        try:
            qualify_history_schema(
                conn,
                self._schema,
                expected_role="runtime_role",
                trust_domain=self._stream.trust_domain,
            )
        except HistorySchemaQualificationError as exc:
            raise HistoryStorageUnavailable(
                "reviewed PostgreSQL history authority unavailable"
            ) from exc

    def _snapshot(self, conn: psycopg.Connection[Any], *, lock: bool) -> _HistorySnapshot:
        result = conn.execute(
            sql.SQL("SELECT {}.read_stream(%s,%s)").format(sql.Identifier(self._schema)),
            (self._stream_bytes, lock),
        ).fetchone()
        if result is None:
            raise HistoryContractError("provisioned history stream is unavailable")
        value = _object(
            result[0],
            {"canonical_identity", "head_sequence", "head_digest", "records", "attestations"},
            "stream snapshot",
        )
        if _unhex(value["canonical_identity"]) != self._stream_bytes:
            raise HistoryContractError("wrong stream/environment/trust/product/security epoch")
        if type(value["records"]) is not list or type(value["attestations"]) is not list:
            raise HistoryContractError("malformed retained history collection")
        records = tuple(_record_from_row(row, self._stream) for row in value["records"])
        predecessor = NO_PREDECESSOR
        events: set[HistoryEventIdentity] = set()
        for sequence, record in enumerate(records, GENESIS_SEQUENCE):
            if (
                record.sequence != sequence
                or record.predecessor_authenticated_digest != predecessor
                or record.event_identity in events
            ):
                raise HistoryContractError("gap, rewind, splice, fork, or inconsistent event index")
            events.add(record.event_identity)
            predecessor = record.authenticated_digest
        if (
            type(value["head_sequence"]) is not int
            or value["head_sequence"] != len(records)
            or value["head_digest"] != predecessor
        ):
            raise HistoryContractError("retained head/history tail mismatch")
        heads = tuple(_head_from_row(row, self._stream) for row in value["attestations"])
        head_sequences: set[int] = set()
        for head in heads:
            if (
                head.sequence > len(records)
                or head.sequence in head_sequences
                or head.record_digest
                != records[head.sequence - GENESIS_SEQUENCE].authenticated_digest
            ):
                raise HistoryContractError("retained attestation/history relation is corrupt")
            head_sequences.add(head.sequence)
        return _HistorySnapshot(records, heads)

    def _read(self, operation: Callable[[_HistorySnapshot], _T]) -> _T:
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                conn.execute("SET LOCAL search_path = pg_catalog")
                self._qualify(conn)
                return operation(self._snapshot(conn, lock=False))
        except (psycopg.errors.InvalidParameterValue, psycopg.errors.CheckViolation) as exc:
            raise HistoryContractError("PostgreSQL retained history or scope is corrupt") from exc
        except psycopg.Error as exc:
            raise HistoryStorageUnavailable(
                "PostgreSQL authoritative history read unavailable"
            ) from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HistoryContractError("invalid retained history") from exc

    def current_record(self) -> HistoryRecord | None:
        return self._read(lambda snapshot: snapshot.records[-1] if snapshot.records else None)

    def current_head(self) -> HistoryRecord | None:
        """The substrate port's head is the exact retained record, not a signer."""
        return self.current_record()

    def retained_history(self) -> tuple[HistoryRecord, ...]:
        return self._read(lambda snapshot: snapshot.records)

    def record_at(self, sequence: int) -> HistoryRecord | None:
        if type(sequence) is not int or sequence < GENESIS_SEQUENCE:
            raise TypeError("sequence must be an exact int >= 1")
        return self._read(
            lambda snapshot: (
                snapshot.records[sequence - GENESIS_SEQUENCE]
                if sequence <= len(snapshot.records)
                else None
            )
        )

    def verify(self) -> None:
        self._read(lambda snapshot: None)

    def append(
        self,
        *,
        expected_digest: str,
        event_identity: HistoryEventIdentity,
        payload: Mapping[str, object],
    ) -> tuple[HistoryRecord, bool]:
        admitted_event = _snapshot_event_identity(event_identity)
        # Freeze/validate before DB I/O. The locked predecessor below determines
        # the actual record, including replay's original sequence/predecessor.
        validated = build_record(
            self._stream, GENESIS_SEQUENCE, NO_PREDECESSOR, admitted_event, payload
        )
        try:
            with self._connect() as conn:
                with conn.transaction():
                    conn.execute("SET LOCAL search_path = pg_catalog")
                    self._qualify(conn)
                    snapshot = self._snapshot(conn, lock=True)
                    prior = next(
                        (
                            record
                            for record in snapshot.records
                            if record.event_identity == admitted_event
                        ),
                        None,
                    )
                    if prior is not None:
                        if validated.event_digest != prior.event_digest:
                            raise HistoryContractError("conflicting idempotency replay")
                        return prior, True
                    actual = (
                        snapshot.records[-1].authenticated_digest
                        if snapshot.records
                        else NO_PREDECESSOR
                    )
                    if expected_digest != actual:
                        raise HistoryContractError(
                            "stale predecessor CAS; append race/fork rejected"
                        )
                    record = build_record(
                        self._stream,
                        len(snapshot.records) + GENESIS_SEQUENCE,
                        actual,
                        admitted_event,
                        _plain(validated.canonical_event_payload),
                    )
                    _persistence_cut("before_append")
                    result = conn.execute(
                        sql.SQL("SELECT {}.append_record(%s,%s,%s,%s)").format(
                            sql.Identifier(self._schema)
                        ),
                        (
                            self._stream_bytes,
                            expected_digest,
                            Json(
                                {
                                    "sequence": record.sequence,
                                    "predecessor_authenticated_digest": actual,
                                    "canonical_event_identity_hex": canonical_json_bytes(
                                        admitted_event.material()
                                    ).hex(),
                                    "event_digest": record.event_digest,
                                    "authenticated_digest": record.authenticated_digest,
                                    "canonical_payload_hex": canonical_json_bytes(
                                        _plain(record.canonical_event_payload)
                                    ).hex(),
                                }
                            ),
                            canonical_json_bytes(record.unsigned_material()),
                        ),
                    ).fetchone()
                    if result != (False,):
                        raise HistoryContractError("database append returned an inexact CAS result")
                    retained = self._snapshot(conn, lock=True)
                    if retained.records != (*snapshot.records, record):
                        raise HistoryContractError("database append did not retain exact successor")
                    _persistence_cut("after_append_before_commit")
                _persistence_cut("after_commit")
            return _snapshot_record(record), False
        except (psycopg.errors.InvalidParameterValue, psycopg.errors.CheckViolation) as exc:
            raise HistoryContractError(
                "PostgreSQL rejected invalid or conflicting history"
            ) from exc
        except psycopg.errors.SerializationFailure as exc:
            raise HistoryContractError("stale predecessor CAS; append race/fork rejected") from exc
        except psycopg.Error as exc:
            # No automatic retry of an unknown commit outcome. An explicit exact
            # event replay requalifies and verifies the entire retained chain.
            raise HistoryOperationIndeterminate(
                "PostgreSQL history write outcome indeterminate"
            ) from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HistoryContractError("invalid retained history during append") from exc

    def append_exact_successor(
        self, expected_head: HistoryRecord | None, record: HistoryRecord
    ) -> tuple[HistoryRecord, bool]:
        if expected_head is not None and type(expected_head) is not HistoryRecord:
            raise TypeError("expected head must be an exact HistoryRecord or None")
        admitted = _snapshot_record(record)
        expected = None if expected_head is None else _snapshot_record(expected_head)
        predecessor = NO_PREDECESSOR if expected is None else expected.authenticated_digest
        if (
            admitted.stream != self._stream
            or (expected is not None and expected.stream != self._stream)
            or admitted.predecessor_authenticated_digest != predecessor
            or admitted.sequence
            != (GENESIS_SEQUENCE if expected is None else expected.sequence + 1)
        ):
            raise HistoryContractError("candidate is not the exact stream successor")
        return self.append(
            expected_digest=predecessor,
            event_identity=admitted.event_identity,
            payload=_plain(admitted.canonical_event_payload),
        )

    def _verify_head_in_snapshot(
        self,
        snapshot: _HistorySnapshot,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider | None,
        *,
        current: bool,
    ) -> VerifiedHistoryHead:
        copied = _snapshot_head(head)
        if (
            copied.stream != self._stream
            or not snapshot.records
            or copied.sequence > len(snapshot.records)
            or (current and copied.sequence != len(snapshot.records))
            or copied.record_digest
            != snapshot.records[copied.sequence - GENESIS_SEQUENCE].authenticated_digest
        ):
            raise HistoryContractError("attested head does not name its exact retained history")
        verify_head(
            copied,
            authority,
            expected_stream=self._stream,
            historical_evidence_authority=historical_evidence_authority,
        )
        return VerifiedHistoryHead(copied)

    def verify_attested_head(
        self,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider | None = None,
    ) -> VerifiedHistoryHead:
        return self._read(
            lambda snapshot: self._verify_head_in_snapshot(
                snapshot, head, authority, historical_evidence_authority, current=True
            )
        )

    def verify_attested_historical_head(
        self,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider,
    ) -> VerifiedHistoryHead:
        if type(historical_evidence_authority) is not LocalCheckpointProvider:
            raise TypeError("historical evidence authority must be exact")
        return self._read(
            lambda snapshot: self._verify_head_in_snapshot(
                snapshot, head, authority, historical_evidence_authority, current=False
            )
        )

    def retain_attested_head(
        self,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider | None = None,
    ) -> tuple[AttestedHistoryHead, bool]:
        admitted = _snapshot_head(head)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET LOCAL search_path = pg_catalog")
                self._qualify(conn)
                snapshot = self._snapshot(conn, lock=True)
                self._verify_head_in_snapshot(
                    snapshot, admitted, authority, historical_evidence_authority, current=False
                )
                prior = next(
                    (item for item in snapshot.heads if item.sequence == admitted.sequence), None
                )
                if prior is not None:
                    if prior != admitted:
                        raise HistoryContractError("conflicting retained head replay")
                    return prior, True
                result = conn.execute(
                    sql.SQL("SELECT {}.retain_head(%s,%s,%s,%s)").format(
                        sql.Identifier(self._schema)
                    ),
                    (
                        self._stream_bytes,
                        Json(
                            {
                                "sequence": admitted.sequence,
                                "record_digest": admitted.record_digest,
                                "canonical_credential_hex": canonical_json_bytes(
                                    _credential_material(admitted.signing_credential_identity)
                                ).hex(),
                            }
                        ),
                        admitted.canonical_attestation_bytes,
                        admitted.signature,
                    ),
                ).fetchone()
                if result != (False,):
                    raise HistoryContractError("database retained-head result is inexact")
                retained = self._snapshot(conn, lock=True)
                if admitted not in retained.heads:
                    raise HistoryContractError("database did not retain exact attested head")
                return admitted, False
        except (psycopg.errors.InvalidParameterValue, psycopg.errors.CheckViolation) as exc:
            raise HistoryContractError(
                "PostgreSQL rejected invalid or conflicting history head"
            ) from exc
        except psycopg.Error as exc:
            raise HistoryOperationIndeterminate(
                "PostgreSQL head retention outcome indeterminate"
            ) from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HistoryContractError("invalid retained history during head retention") from exc

    def retained_attested_head(
        self,
        sequence: int | None = None,
        *,
        verification_authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider | None = None,
    ) -> AttestedHistoryHead | None:
        if sequence is not None and (type(sequence) is not int or sequence < GENESIS_SEQUENCE):
            raise TypeError("sequence must be an exact int >= 1 or None")

        def read(snapshot: _HistorySnapshot) -> AttestedHistoryHead | None:
            selected = len(snapshot.records) if sequence is None else sequence
            head = next((item for item in snapshot.heads if item.sequence == selected), None)
            if head is None:
                return None
            return self._verify_head_in_snapshot(
                snapshot,
                head,
                verification_authority,
                historical_evidence_authority,
                current=sequence is None,
            ).head

        return self._read(read)

    def reconcile(
        self,
        *,
        head: AttestedHistoryHead | None,
        verification_authority: HistoryAttestationSigningProvider | None,
        checkpoint_authority: LocalCheckpointProvider,
    ) -> ReconciliationOutcome:
        """Read-only oracle classification; this does not advance a checkpoint."""
        if type(checkpoint_authority) is not LocalCheckpointProvider:
            raise TypeError("checkpoint authority must be the exact reference provider")

        def compare(snapshot: _HistorySnapshot) -> ReconciliationOutcome:
            if not snapshot.records:
                if head is not None:
                    return ReconciliationOutcome.CORRUPT
                return (
                    ReconciliationOutcome.NOT_FOUND
                    if checkpoint_authority.current_checkpoint() is None
                    else ReconciliationOutcome.CORRUPT
                )
            if head is None or verification_authority is None:
                return ReconciliationOutcome.CORRUPT
            verified = self._verify_head_in_snapshot(
                snapshot, head, verification_authority, checkpoint_authority, current=True
            )
            return _compare_verified_checkpoint(
                verified.head, checkpoint_authority.current_checkpoint()
            )

        try:
            return self._read(compare)
        except HistoricalRevokedSignatureVerificationUnavailable:
            raise
        except (HistoryStorageUnavailable, HistorySigningAuthorityUnavailable):
            return ReconciliationOutcome.UNAVAILABLE
        except (HistoryContractError, TypeError):
            return ReconciliationOutcome.CORRUPT
