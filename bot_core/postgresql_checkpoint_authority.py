"""Trwały lokalny checkpoint i historia akceptacji V2, bez emisji Root Proof.

Porty historii i custody wybiera zaufany composition root. Ich weryfikacja
kończy się przed blokadą wiersza checkpointu; pod blokadą porównywane są dokładne
wcześniej uwierzytelnione bajty. PostgreSQL nie jest niezależnym anti-rollback.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import psycopg
from psycopg import sql

from bot_core.authenticated_issuer_history import (
    ACCEPTANCE_DOMAIN,
    ATTESTATION_DOMAIN,
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoricalHeadAcceptanceEvidence,
    HistoricalRevokedSignatureVerificationUnavailable,
    HistoryContractError,
    HistorySigningAuthorityUnavailable,
    HistoryStreamIdentity,
    LocalCheckpoint,
    LocalCheckpointProvider,
    _acceptance_matches_head,
    _acceptance_material,
    _credential_material,
    _digest,
    _integer,
    _snapshot_acceptance,
    _snapshot_checkpoint,
    _snapshot_head,
    _snapshot_stream,
    _text,
    canonical_json_bytes,
    verify_head,
)
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.postgresql_authenticated_issuer_history import (
    PostgreSQLAuthenticatedIssuerHistory,
    _object,
    _parse_canonical,
    _unhex,
)
from bot_core.postgresql_checkpoint_schema import (
    CheckpointSchemaQualificationError,
    _identifier,
    _scalar,
    qualify_checkpoint_schema,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import (
    CheckpointCapabilities,
    CredentialRoleIdentity,
    CredentialSemanticRole,
    HistoryAttestationSigningProvider,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)


class CheckpointStorageUnavailable(HistoryContractError):
    """Brak autorytatywnego odczytu; nigdy nie oznacza nieobecności checkpointu."""


class CheckpointOperationIndeterminate(HistoryContractError):
    """Nieznany wynik zapisu; wymagany dokładny odczyt, bez nowej tożsamości."""


class CheckpointHistoryMismatchError(HistoryContractError):
    """Checkpoint nie odpowiada retained history; brak dowodu kierunku restore."""

    def __init__(
        self,
        *,
        reason: str,
        history_sequence: int,
        checkpoint_sequence: int,
        checkpoint_record_digest: str,
        observed_record_digest: str | None,
    ) -> None:
        super().__init__("checkpoint ahead of history or retained prefix rolled back")
        self.reason = reason
        self.history_sequence = history_sequence
        self.checkpoint_sequence = checkpoint_sequence
        self.checkpoint_record_digest = checkpoint_record_digest
        self.observed_record_digest = observed_record_digest


def _persistence_cut(name: str) -> None:
    """Punkt testowego przerwania procesu; bez działania w produkcji."""


def _credential(value: object) -> CredentialRoleIdentity:
    fields = _object(
        value,
        {
            "semantic_role",
            "credential_identity",
            "provider_namespace",
            "key_handle_or_version",
            "custody_lifecycle_namespace",
            "key_material_identity",
        },
        "checkpoint credential",
    )
    return CredentialRoleIdentity(
        **{**fields, "semantic_role": CredentialSemanticRole(fields["semantic_role"])}
    )


def _checkpoint_material(checkpoint: LocalCheckpoint) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "stream": checkpoint.stream.material(),
        "history_sequence": checkpoint.history_sequence,
        "authenticated_head_digest": checkpoint.authenticated_head_digest,
        "history_signing_credential_identity": _credential_material(
            checkpoint.history_signing_credential_identity
        ),
        "checkpoint_revision": checkpoint.checkpoint_revision,
    }


def _checkpoint_from_bytes(raw: bytes) -> LocalCheckpoint:
    value = _object(
        _parse_canonical(raw),
        {
            "checkpoint_id",
            "stream",
            "history_sequence",
            "authenticated_head_digest",
            "history_signing_credential_identity",
            "checkpoint_revision",
        },
        "checkpoint",
    )
    return LocalCheckpoint(
        **{
            **value,
            "stream": HistoryStreamIdentity(**value["stream"]),
            "history_signing_credential_identity": _credential(
                value["history_signing_credential_identity"]
            ),
        }
    )


def _acceptance_from_bytes(raw: bytes) -> HistoricalHeadAcceptanceEvidence:
    value = _object(
        _parse_canonical(raw),
        {
            "checkpoint_authority_identity",
            "checkpoint_revision",
            "stream",
            "accepted_history_sequence",
            "accepted_authenticated_head_digest",
            "history_signing_credential_identity",
            "lifecycle_generation_at_acceptance",
            "lifecycle_state_at_acceptance",
            "predecessor_acceptance_digest",
            "acceptance_digest",
        },
        "historical acceptance",
    )
    return HistoricalHeadAcceptanceEvidence(
        **{
            **value,
            "stream": HistoryStreamIdentity(**value["stream"]),
            "history_signing_credential_identity": _credential(
                value["history_signing_credential_identity"]
            ),
            "lifecycle_state_at_acceptance": SigningKeyLifecycle(
                value["lifecycle_state_at_acceptance"]
            ),
        }
    )


def _head_from_bytes(raw: bytes, signature: bytes) -> AttestedHistoryHead:
    if not raw.startswith(ATTESTATION_DOMAIN):
        raise HistoryContractError("checkpoint requires a V2 attestation")
    value = _object(
        _parse_canonical(raw[len(ATTESTATION_DOMAIN) :]),
        {"stream", "sequence", "record_digest", "signing_credential_identity"},
        "accepted head",
    )
    if len(signature) != 64:
        raise HistoryContractError("missing or malformed retained signature")
    return AttestedHistoryHead(
        HistoryStreamIdentity(**value["stream"]),
        value["sequence"],
        value["record_digest"],
        _credential(value["signing_credential_identity"]),
        raw,
        signature,
    )


@dataclass(frozen=True, slots=True)
class CheckpointReadSnapshot:
    """Odłączony raport jednego odczytu autorytetu; nie nadaje uprawnień."""

    authority_identity: str
    checkpoint_id: str
    stream: HistoryStreamIdentity
    checkpoints: tuple[LocalCheckpoint, ...]
    acceptances: tuple[HistoricalHeadAcceptanceEvidence, ...]
    heads: tuple[AttestedHistoryHead, ...]

    @property
    def current(self) -> LocalCheckpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None


@dataclass(frozen=True, slots=True)
class _CheckpointSnapshot:
    authority_identity: str
    checkpoints: tuple[LocalCheckpoint, ...]
    acceptances: tuple[HistoricalHeadAcceptanceEvidence, ...]
    heads: tuple[AttestedHistoryHead, ...]

    @property
    def current(self) -> LocalCheckpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None


class PostgreSQLCheckpointAuthority:
    """CheckpointAuthorityProvider wybrany przez zaufany composition root.

    Constructor nie instaluje SQL ani nie autoryzuje caller-selected portów.
    Pełna koperta każdej rewizji zachowuje evidence V2 oraz dokładny podpis.
    """

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        checkpoint_id: str,
        stream: HistoryStreamIdentity,
        history: PostgreSQLAuthenticatedIssuerHistory,
        verification_authorities: tuple[HistoryAttestationSigningProvider, ...],
    ) -> None:
        if type(connection) is not PostgreSQLConnectionConfig:
            raise TypeError("connection must be exact PostgreSQLConnectionConfig")
        if type(history) is not PostgreSQLAuthenticatedIssuerHistory:
            raise TypeError("history must be the exact configured PostgreSQL authority")
        if type(verification_authorities) is not tuple or not verification_authorities:
            raise TypeError("trusted verification authorities must be a nonempty exact tuple")
        self._schema = _identifier(schema)
        self._checkpoint_id = _text(checkpoint_id, "checkpoint_id")
        self._stream = _snapshot_stream(stream)
        if (
            self._stream.security_profile != SecurityProfile.PRODUCTION_LOCAL.value
            or history._stream != self._stream
        ):
            raise HistoryContractError("checkpoint requires its exact PRODUCTION_LOCAL history")
        self._connection = connection
        self._history = history
        self._authorities = verification_authorities
        self._id_bytes = _scalar(self._checkpoint_id)
        self._stream_bytes = canonical_json_bytes(self._stream.material())
        self._authority_identity: str | None = None
        self._observation_lock = Lock()
        self._highest_revision = 0
        self._highest_acceptance_digest = NO_PREDECESSOR
        state = self._read_authenticated()
        self._authority_identity = state.authority_identity

    def __init_subclass__(cls) -> None:
        raise TypeError("production checkpoint authority cannot be subclassed")

    def __repr__(self) -> str:
        return (
            "PostgreSQLCheckpointAuthority(connection=<redacted>, "
            f"schema={self._schema!r}, checkpoint_id={self._checkpoint_id!r})"
        )

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            ProviderRole.CHECKPOINT_AUTHORITY,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, self._stream.trust_domain),
            str(self._authority_identity),
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            True,
            authoritative_reads=True,
            durable_state=True,
            compare_and_swap=True,
            checkpoint=CheckpointCapabilities(
                authenticated=True,
                monotonic=True,
                exact_history_head_binding=True,
                independent_rollback_domain=False,
                independent_admin_or_security_domain=False,
                retained_authenticated_history=True,
                historical_lookup_or_recovery=True,
                rollback_direction_proof=True,
                wall_clock_arbitration_forbidden=True,
            ),
        )

    def credential_identities(self) -> tuple[()]:
        return ()

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._connection.dsn, autocommit=True)

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        try:
            qualify_checkpoint_schema(
                conn,
                self._schema,
                expected_role="runtime_role",
                trust_domain=self._stream.trust_domain,
            )
        except CheckpointSchemaQualificationError as exc:
            raise CheckpointStorageUnavailable("reviewed checkpoint authority unavailable") from exc

    def _oracle(self, state: _CheckpointSnapshot) -> LocalCheckpointProvider:
        # Prywatne odtworzenie walidatora oracle z autorytatywnych bajtów bazy.
        # Obiekt nigdy nie pochodzi od caller i nie opuszcza implementacji.
        oracle = LocalCheckpointProvider(self._checkpoint_id, self._stream)
        oracle._authority_identity = state.authority_identity
        oracle._acceptances = list(state.acceptances)
        oracle._current = state.current
        oracle._verify_authority_state_locked()
        return oracle

    def _snapshot(self, conn: psycopg.Connection[Any], *, lock: bool) -> _CheckpointSnapshot:
        row = conn.execute(
            sql.SQL("SELECT {}.read_checkpoint(%s,%s,%s)").format(sql.Identifier(self._schema)),
            (self._id_bytes, self._stream_bytes, lock),
        ).fetchone()
        if row is None:
            raise HistoryContractError("missing provisioned checkpoint authority")
        envelope = _object(
            row[0],
            {"checkpoint_id", "stream", "authority_identity", "revision", "entries"},
            "checkpoint authority",
        )
        identity = _text(envelope["authority_identity"], "checkpoint authority identity")
        if (
            _unhex(envelope["checkpoint_id"]) != self._id_bytes
            or _unhex(envelope["stream"]) != self._stream_bytes
            or (self._authority_identity is not None and identity != self._authority_identity)
            or type(envelope["entries"]) is not list
            or type(envelope["revision"]) is not int
            or envelope["revision"] != len(envelope["entries"])
        ):
            raise HistoryContractError("checkpoint authority identity or cardinality corrupt")
        checkpoints = []
        acceptances = []
        heads = []
        for revision, value in enumerate(envelope["entries"], 1):
            entry = _object(
                value, {"revision", "checkpoint", "acceptance", "head", "signature"}, "revision"
            )
            checkpoint = _checkpoint_from_bytes(_unhex(entry["checkpoint"]))
            acceptance = _acceptance_from_bytes(_unhex(entry["acceptance"]))
            head = _head_from_bytes(_unhex(entry["head"]), _unhex(entry["signature"]))
            if (
                type(entry["revision"]) is not int
                or entry["revision"] != revision
                or checkpoint.checkpoint_revision != revision
                or checkpoint.checkpoint_id != self._checkpoint_id
                or checkpoint.stream != self._stream
                or acceptance.checkpoint_revision != revision
                or not _acceptance_matches_head(acceptance, head)
                or (
                    checkpoint.history_sequence,
                    checkpoint.authenticated_head_digest,
                    checkpoint.history_signing_credential_identity,
                )
                != (head.sequence, head.record_digest, head.signing_credential_identity)
            ):
                raise HistoryContractError("checkpoint/evidence/retained signature binding corrupt")
            checkpoints.append(checkpoint)
            acceptances.append(acceptance)
            heads.append(head)
        state = _CheckpointSnapshot(identity, tuple(checkpoints), tuple(acceptances), tuple(heads))
        self._oracle(state)
        return state

    def _authority_for(self, head: AttestedHistoryHead) -> HistoryAttestationSigningProvider:
        matches = []
        for authority in self._authorities:
            try:
                if head.signing_credential_identity in authority.credential_identities():
                    matches.append(authority)
            except Exception as exc:
                raise HistorySigningAuthorityUnavailable(
                    "selected signing authority unavailable"
                ) from exc
        if len(matches) != 1:
            raise HistoryContractError(
                "head signer is not uniquely selected by trusted composition"
            )
        return matches[0]

    def _authenticate(self, state: _CheckpointSnapshot) -> None:
        # Wszystkie transakcje checkpointu są tu zakończone. Historia oraz
        # custody nigdy nie są wywoływane pod blokadą checkpoint row.
        records = self._history.retained_history()
        oracle = self._oracle(state)
        for head in state.heads:
            record_digest = (
                records[head.sequence - 1].authenticated_digest
                if head.sequence <= len(records)
                else None
            )
            if record_digest != head.record_digest:
                raise CheckpointHistoryMismatchError(
                    reason=(
                        "CHECKPOINT_AHEAD_OF_HISTORY"
                        if record_digest is None
                        else "RETAINED_PREFIX_MISMATCH"
                    ),
                    history_sequence=len(records),
                    checkpoint_sequence=head.sequence,
                    checkpoint_record_digest=head.record_digest,
                    observed_record_digest=record_digest,
                )
            verify_head(
                head,
                self._authority_for(head),
                expected_stream=self._stream,
                historical_evidence_authority=oracle,
            )

    def _read_authenticated(self) -> _CheckpointSnapshot:
        with self._observation_lock:
            minimum = (self._highest_revision, self._highest_acceptance_digest)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                self._qualify(conn)
                state = self._snapshot(conn, lock=False)
            self._authenticate(state)
            self._observe(state, minimum)
            return state
        except (
            psycopg.errors.InvalidParameterValue,
            psycopg.errors.CheckViolation,
            psycopg.errors.NumericValueOutOfRange,
        ) as exc:
            raise HistoryContractError("corrupt retained checkpoint authority") from exc
        except psycopg.Error as exc:
            raise CheckpointStorageUnavailable("authoritative checkpoint read unavailable") from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HistoryContractError("invalid retained checkpoint bytes") from exc

    def _observe(self, state: _CheckpointSnapshot, minimum: tuple[int, str]) -> None:
        # Lokalna pamięć wykrywa rewind obserwowanej instancji. Nie zastępuje
        # niezależnego trwałego świadka po restarcie/restore całego magazynu.
        # Ten krótki lock nigdy nie obejmuje historii, custody ani SQL.
        with self._observation_lock:
            revision = len(state.acceptances)
            floor_revision, floor_digest = minimum
            if revision < floor_revision or (
                floor_revision > 0
                and state.acceptances[floor_revision - 1].acceptance_digest != floor_digest
            ):
                raise HistoryContractError("observed checkpoint prefix rolled back or forked")
            if revision > self._highest_revision:
                self._highest_revision = revision
                self._highest_acceptance_digest = state.acceptances[-1].acceptance_digest

    def current_checkpoint(self) -> LocalCheckpoint | None:
        return self._read_authenticated().current

    def authenticated_read_snapshot(self) -> CheckpointReadSnapshot:
        """Zweryfikuj cały retained state bez zapisu lub globalnego snapshotu.

        Transakcja checkpointu kończy się przed odczytem historii i custody.
        Raport grupuje jeden snapshot checkpointu; nie dowodzi atomowości
        pomiędzy autorytetami i nie może zastąpić ich kolejnej weryfikacji.
        """
        state = self._read_authenticated()
        return CheckpointReadSnapshot(
            state.authority_identity,
            self._checkpoint_id,
            _snapshot_stream(self._stream),
            tuple(_snapshot_checkpoint(item) for item in state.checkpoints),
            tuple(_snapshot_acceptance(item) for item in state.acceptances),
            tuple(_snapshot_head(item) for item in state.heads),
        )

    def retained_acceptances(self) -> tuple[HistoricalHeadAcceptanceEvidence, ...]:
        return self._read_authenticated().acceptances

    def retained_attested_heads(self) -> tuple[AttestedHistoryHead, ...]:
        return self._read_authenticated().heads

    def authenticated_checkpoint_at(self, sequence: int) -> LocalCheckpoint | None:
        _integer(sequence, "sequence", minimum=1)
        return next(
            (
                item
                for item in self._read_authenticated().checkpoints
                if item.history_sequence == sequence
            ),
            None,
        )

    def historical_acceptance(
        self, head: AttestedHistoryHead
    ) -> HistoricalHeadAcceptanceEvidence | None:
        snapshot = _snapshot_head(head)
        return next(
            (
                item
                for item in self._read_authenticated().acceptances
                if _acceptance_matches_head(item, snapshot)
            ),
            None,
        )

    def verify_historical_acceptance(
        self, head: AttestedHistoryHead, *, current_lifecycle_generation: int
    ) -> None:
        _integer(current_lifecycle_generation, "current_lifecycle_generation", minimum=1)
        match = self.historical_acceptance(head)
        if match is None:
            raise HistoricalRevokedSignatureVerificationUnavailable(
                "no exact retained ACTIVE acceptance"
            )
        if match.lifecycle_generation_at_acceptance >= current_lifecycle_generation:
            raise HistoryContractError("invalid lifecycle ordering in historical acceptance")

    def _selected_authority(
        self, head: AttestedHistoryHead, supplied: HistoryAttestationSigningProvider | None
    ) -> HistoryAttestationSigningProvider:
        selected = self._authority_for(head)
        if supplied is not None and supplied is not selected:
            raise HistoryContractError(
                "caller-selected signing authority is not the configured authority"
            )
        return selected

    def advance(
        self,
        *,
        expected_revision: int,
        head: AttestedHistoryHead,
        verification_authority: HistoryAttestationSigningProvider | None = None,
    ) -> tuple[LocalCheckpoint, bool]:
        _integer(expected_revision, "expected_revision")
        admitted = _snapshot_head(head)
        authority = self._selected_authority(admitted, verification_authority)
        self._history.verify_attested_head(admitted, authority)
        before = self._read_authenticated()
        try:
            lifecycle_reader = getattr(authority, "lifecycle_state", None)
            if not callable(lifecycle_reader):
                raise HistorySigningAuthorityUnavailable("checkpoint lifecycle reader unavailable")
            generation = authority.lifecycle_generation()
            lifecycle = lifecycle_reader()
            confirmed = authority.lifecycle_generation()
        except Exception as exc:
            raise HistorySigningAuthorityUnavailable(
                "checkpoint acceptance lifecycle unavailable"
            ) from exc
        if (
            lifecycle is not SigningKeyLifecycle.ACTIVE
            or type(generation) is not int
            or generation < 1
            or type(confirmed) is not int
            or confirmed != generation
        ):
            raise HistoryContractError("checkpoint acceptance requires a stable ACTIVE credential")
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                locked = self._snapshot(conn, lock=True)
                if locked != before:
                    raise HistoryContractError(
                        "checkpoint changed after authentication / CAS conflict"
                    )
                current = locked.current
                revision = 0 if current is None else current.checkpoint_revision
                if expected_revision != revision:
                    raise HistoryContractError("checkpoint CAS conflict")
                if current is not None and admitted.sequence <= current.history_sequence:
                    if admitted.sequence < current.history_sequence:
                        raise HistoryContractError("checkpoint rewind / valid-prefix rollback")
                    if locked.heads[-1] != admitted:
                        raise HistoryContractError("same-sequence split brain")
                    return current, True
                successor = LocalCheckpoint(
                    self._checkpoint_id,
                    self._stream,
                    admitted.sequence,
                    admitted.record_digest,
                    admitted.signing_credential_identity,
                    revision + 1,
                )
                material = {
                    "checkpoint_authority_identity": locked.authority_identity,
                    "checkpoint_revision": revision + 1,
                    "stream": self._stream.material(),
                    "accepted_history_sequence": admitted.sequence,
                    "accepted_authenticated_head_digest": admitted.record_digest,
                    "history_signing_credential_identity": _credential_material(
                        admitted.signing_credential_identity
                    ),
                    "lifecycle_generation_at_acceptance": generation,
                    "lifecycle_state_at_acceptance": lifecycle.value,
                    "predecessor_acceptance_digest": locked.acceptances[-1].acceptance_digest
                    if locked.acceptances
                    else NO_PREDECESSOR,
                }
                acceptance = HistoricalHeadAcceptanceEvidence(
                    **{
                        **material,
                        "stream": self._stream,
                        "history_signing_credential_identity": admitted.signing_credential_identity,
                        "lifecycle_state_at_acceptance": lifecycle,
                        "acceptance_digest": _digest(ACCEPTANCE_DOMAIN, material),
                    }
                )
                result = conn.execute(
                    sql.SQL("SELECT {}.commit_checkpoint(%s,%s,%s,%s,%s,%s,%s)").format(
                        sql.Identifier(self._schema)
                    ),
                    (
                        self._id_bytes,
                        self._stream_bytes,
                        expected_revision,
                        canonical_json_bytes(_checkpoint_material(successor)),
                        canonical_json_bytes(
                            {
                                **_acceptance_material(acceptance),
                                "acceptance_digest": acceptance.acceptance_digest,
                            }
                        ),
                        admitted.canonical_attestation_bytes,
                        admitted.signature,
                    ),
                ).fetchone()
                if result != (False,):
                    raise HistoryContractError("inexact checkpoint commit result")
                committed = self._snapshot(conn, lock=True)
                if committed != _CheckpointSnapshot(
                    locked.authority_identity,
                    (*locked.checkpoints, successor),
                    (*locked.acceptances, acceptance),
                    (*locked.heads, admitted),
                ):
                    raise HistoryContractError(
                        "checkpoint and acceptance were not retained atomically"
                    )
                _persistence_cut("before_commit")
            _persistence_cut("after_commit")
            self._observe(
                committed,
                (
                    len(locked.acceptances),
                    locked.acceptances[-1].acceptance_digest
                    if locked.acceptances
                    else NO_PREDECESSOR,
                ),
            )
            return _snapshot_checkpoint(successor), False
        except (
            psycopg.errors.InvalidParameterValue,
            psycopg.errors.CheckViolation,
            psycopg.errors.SerializationFailure,
            psycopg.errors.NumericValueOutOfRange,
        ) as exc:
            raise HistoryContractError(
                "PostgreSQL rejected corrupt checkpoint or CAS conflict"
            ) from exc
        except psycopg.Error as exc:
            raise CheckpointOperationIndeterminate(
                "checkpoint write outcome indeterminate"
            ) from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HistoryContractError("invalid checkpoint commit bytes") from exc

    def resolve_advance(
        self, *, expected_revision: int, head: AttestedHistoryHead
    ) -> tuple[LocalCheckpoint, bool] | None:
        """Rozwiąż pierwotną operację przez retained revision, bez nowego CAS.

        Nieobecność rewizji można stwierdzić wyłącznie po autorytatywnym odczycie.
        Stare expected_revision nie staje się uprawnieniem do nowego zapisu.
        """
        _integer(expected_revision, "expected_revision")
        admitted = _snapshot_head(head)
        state = self._read_authenticated()
        if expected_revision >= len(state.checkpoints):
            return None
        if state.heads[expected_revision] != admitted:
            raise HistoryContractError("different operation at retained checkpoint revision")
        return state.checkpoints[expected_revision], True

    def retry_advance(
        self,
        *,
        expected_revision: int,
        head: AttestedHistoryHead,
        verification_authority: HistoryAttestationSigningProvider | None = None,
    ) -> tuple[LocalCheckpoint, bool]:
        admitted = _snapshot_head(head)
        self._selected_authority(admitted, verification_authority)
        resolved = self.resolve_advance(expected_revision=expected_revision, head=admitted)
        return (
            resolved
            if resolved is not None
            else self.advance(
                expected_revision=expected_revision,
                head=admitted,
                verification_authority=verification_authority,
            )
        )

    def advance_exact_successor(self, expected: object, successor: object) -> bool:
        """Facade portu; LocalCheckpoint jest żądaniem, nigdy capability."""
        if (expected is not None and type(expected) is not LocalCheckpoint) or type(
            successor
        ) is not LocalCheckpoint:
            raise TypeError("exact checkpoint request values required")
        candidate = _checkpoint_from_bytes(canonical_json_bytes(_checkpoint_material(successor)))
        current = self.current_checkpoint()
        if current != expected:
            raise HistoryContractError("checkpoint CAS conflict")
        revision = 0 if current is None else current.checkpoint_revision
        if (
            candidate.checkpoint_id != self._checkpoint_id
            or candidate.stream != self._stream
            or candidate.checkpoint_revision != revision + 1
        ):
            raise HistoryContractError("inexact checkpoint successor request")
        credential_heads = self._history._read(lambda snapshot: snapshot.heads)
        head = next(
            (item for item in credential_heads if item.sequence == candidate.history_sequence), None
        )
        if head is None:
            raise HistoryContractError("no authoritative retained attestation for successor")
        if (candidate.authenticated_head_digest, candidate.history_signing_credential_identity) != (
            head.record_digest,
            head.signing_credential_identity,
        ):
            raise HistoryContractError("successor differs from authoritative retained head")
        committed, replay = self.advance(expected_revision=revision, head=head)
        if committed != candidate:
            raise HistoryContractError("committed checkpoint differs from successor request")
        return not replay
