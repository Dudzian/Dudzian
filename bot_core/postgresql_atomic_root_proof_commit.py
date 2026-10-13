"""Internal PostgreSQL atomic commit client for an administratively isolated test DB.

Preparation verifies genuine INITIAL_BINDING/attempt authority before the explicit
V1 migration closes standalone writers. It never signs or mints a Root Proof.
Only the qualified test administrator can retain that verified preparation; the
issuer writer then invokes one closed SQL operation in one physical SERIALIZABLE
transaction. No public issuer composition imports this module.

The database fence here qualifies the isolated test protocol only. Coupling a
production custody administrative transition to that fence remains a blocker.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast
from weakref import WeakKeyDictionary

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoryEventIdentity,
    HistoryRecord,
    _credential_material,
    _freeze_json,
    _plain,
    _snapshot_head,
    build_record,
    canonical_json_bytes,
    verify_head,
)
from bot_core.cha_attempt_store import AttemptAuthorization
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    BoundBinding,
    EntitlementLifecycle,
    RegistryReadOutcome,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.licensing.cha_root_proof_signed_attempt import _read_issuer_preflight_source
from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    SigningKeyLifecycle,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofTestDeployment,
    qualify_atomic_root_proof_authority,
    read_atomic_root_proof_physical_identity,
    validate_atomic_root_proof_deployment,
)
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig, _state_json
from bot_core.postgresql_preaccount_credentials import CredentialGeneration
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.postgresql_root_proof_retention import (
    PostgreSQLRootProofRetention,
    PreparedRootProof,
    _canonical,
    _scope_data,
    _snapshot_data,
)
from bot_core.postgresql_root_proof_signing_intent import (
    PostgreSQLRootProofSigningIntent,
    RootProofSigningIntent,
)
from bot_core.root_proof_atomic_attempt_consistency import _validate_providers, _validate_records
from bot_core.root_proof_issuance_reconciliation import (
    _expected_issuance_history_payload,
    _plain as _report_plain,
)
from bot_core.root_proof_issuer_runtime import _verify_request


class AtomicRootProofCommitError(RuntimeError):
    """Fail-closed internal commit refusal without connection secrets."""


class AtomicRootProofCommitConflict(AtomicRootProofCommitError):
    """Exact identity, CAS predecessor or live eligibility was rejected."""


class AtomicRootProofCommitCorrupt(AtomicRootProofCommitError):
    """Original evidence or the retained decision is internally inconsistent."""


class AtomicRootProofCommitUnavailable(AtomicRootProofCommitError):
    """Qualified authority was unavailable before a write could occur."""


class AtomicRootProofCommitIndeterminate(AtomicRootProofCommitError):
    """A previous commit may exist; only positive reconciliation resolves it."""


@dataclass(frozen=True, slots=True)
class AtomicRootProofCommitLocator:
    """Exact recovery identity; its construction never authorizes a write."""

    scope: bytes
    attempt: bytes
    operation: bytes
    entitlement: bytes
    root_proof: bytes
    account: bytes

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not bytes for name in self.__dataclass_fields__):
            raise TypeError("EXACT_IMMUTABLE_COMMIT_LOCATOR_REQUIRED")


def _fence_key(role: str, material: bytes) -> str:
    return role + ":" + hashlib.sha256(b"CryptoHunter/AtomicTestFence/v1\0" + material).hexdigest()


def atomic_history_fence_key(head: AttestedHistoryHead) -> str:
    admitted = _snapshot_head(head)
    return _fence_key(
        "HISTORY_ATTESTATION_SIGNING",
        canonical_json_bytes(_credential_material(admitted.signing_credential_identity)),
    )


@dataclass(frozen=True, slots=True)
class AtomicRootProofCommitFences:
    root_key: str
    root_generation: int
    history_key: str
    history_generation: int
    trust_key: str
    trust_generation: int

    def __post_init__(self) -> None:
        for name in ("root_key", "history_key", "trust_key"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise TypeError("EXACT_LIFECYCLE_FENCE_IDENTITY_REQUIRED")
        for name in ("root_generation", "history_generation", "trust_generation"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TypeError("EXACT_POSITIVE_LIFECYCLE_GENERATION_REQUIRED")


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class PreparedAtomicRootProofCommit:
    """Detached pre-migration evidence; caller construction is never sufficient."""

    reserved: RootProofSigningIntent
    prepared: PreparedRootProof
    bound_state: AuthoritativeEntitlementState
    history_record: HistoryRecord
    locator: AtomicRootProofCommitLocator
    requester_generation: CredentialGeneration | None = None
    claimant_generation: CredentialGeneration | None = None

    @property
    def root_fence_key(self) -> str:
        return _fence_key(
            "ROOT_PROOF_SIGNING", _canonical(_snapshot_data(self.prepared.credential_snapshot))
        )

    @property
    def trust_fence_key(self) -> str:
        return _fence_key("TRUST_ACTIVATION", self.locator.scope)


@dataclass(frozen=True, slots=True)
class _TrustedPreparation:
    package: bytes
    pins: tuple[tuple[PostgreSQLConnectionConfig, str], ...]


_PREPARATIONS: WeakKeyDictionary[PreparedAtomicRootProofCommit, _TrustedPreparation] = (
    WeakKeyDictionary()
)
_PREPARATION_LOCK = threading.Lock()


def _provider_pin(provider: object, prefix: str = "") -> tuple[PostgreSQLConnectionConfig, str]:
    connection = getattr(provider, prefix + "connection")
    schema = getattr(provider, prefix + "schema")
    if type(connection) is not PostgreSQLConnectionConfig or type(schema) is not str:
        raise AtomicRootProofCommitConflict("GENUINE_PROVIDER_PIN_REQUIRED")
    return connection, schema


def _base_package(value: PreparedAtomicRootProofCommit) -> dict[str, Any]:
    record = value.history_record
    state = value.bound_state
    locator = value.locator
    return {
        "scope_hex": locator.scope.hex(),
        "attempt_hex": locator.attempt.hex(),
        "operation_hex": locator.operation.hex(),
        "entitlement_hex": locator.entitlement.hex(),
        "root_proof_hex": locator.root_proof.hex(),
        "account_hex": locator.account.hex(),
        "lookup_handle": state.subject.lookup_handle,
        "environment": state.subject.environment,
        "trust_domain": state.subject.trust_domain,
        "expected_revision": state.predecessor_revision,
        "entitlement_generation": state.identity.entitlement_generation,
        "reserved_envelope_hex": value.reserved.canonical_envelope.hex(),
        "staged_envelope_hex": value.prepared.canonical_envelope.hex(),
        "payload_hex": value.prepared.canonical_payload.hex(),
        "signature_hex": value.prepared.signature.hex(),
        "bound_state": _state_json(state),
        "stream_hex": canonical_json_bytes(record.stream.material()).hex(),
        "expected_history_digest": record.predecessor_authenticated_digest,
        "history_record": {
            "sequence": record.sequence,
            "predecessor_authenticated_digest": record.predecessor_authenticated_digest,
            "canonical_event_identity_hex": canonical_json_bytes(
                record.event_identity.material()
            ).hex(),
            "event_digest": record.event_digest,
            "authenticated_digest": record.authenticated_digest,
            "canonical_payload_hex": canonical_json_bytes(
                _plain(record.canonical_event_payload)
            ).hex(),
        },
        "history_record_hex": canonical_json_bytes(record.unsigned_material()).hex(),
        "requester_credential": _credential_generation_data(value.requester_generation),
        "claimant_credential": _credential_generation_data(value.claimant_generation),
    }


def _credential_generation_data(value: CredentialGeneration | None) -> dict[str, Any]:
    if type(value) is not CredentialGeneration:
        raise AtomicRootProofCommitConflict("EXACT_GENUINE_PUBLIC_CREDENTIAL_GENERATION_REQUIRED")
    return {
        "credential_id": value.credential_id,
        "principal_id": value.principal_id,
        "semantic_role": value.semantic_role.value,
        "credential_role": value.credential_role,
        "key_id": value.key_id,
        "key_version": value.key_version,
        "public_key_hex": value.public_key.hex(),
        "key_material_identity": value.key_material_identity,
        "environment": value.environment,
        "trust_domain": value.trust_domain,
        "lifecycle_generation": value.lifecycle_generation,
        "lifecycle": value.lifecycle.value,
        "registry_revision": value.registry_revision,
    }


def _issuance_history_payload(
    authorization: AttemptAuthorization,
    bound_state: AuthoritativeEntitlementState,
    predecessor: AuthoritativeEntitlementState,
    attempt_identity_digest: str,
) -> dict[str, object]:
    """Detach contract enums into the existing exact primitive event projection."""
    return cast(
        dict[str, object],
        _report_plain(
            _expected_issuance_history_payload(
                authorization, bound_state, predecessor, attempt_identity_digest
            )
        ),
    )


def prepare_atomic_root_proof_commit(
    binding: object,
    signed_attempt: object,
    *,
    signing_intent: PostgreSQLRootProofSigningIntent,
    retention: PostgreSQLRootProofRetention,
    issuer_authority: PostgreSQLRootProofIssuanceAuthority,
    history: PostgreSQLAuthenticatedIssuerHistory,
) -> PreparedAtomicRootProofCommit:
    """Verify exact genuine RESERVED/STAGED before the administrative migration.

    This read-only operation builds an unsigned history successor. No supplied
    authorization DTO, readiness flag, signing callback, or SQL names are used.
    """
    if (
        type(issuer_authority) is not PostgreSQLRootProofIssuanceAuthority
        or type(history) is not PostgreSQLAuthenticatedIssuerHistory
    ):
        raise AtomicRootProofCommitConflict("GENUINE_PREPARATION_AUTHORITIES_REQUIRED")
    try:
        _validate_providers(signing_intent, retention)
        source, current, request = _read_issuer_preflight_source(signed_attempt)
        if source.binding is not binding or source.provider is not issuer_authority:
            raise AtomicRootProofCommitConflict("EXACT_GENUINE_BINDING_AUTHORITY_REQUIRED")
        issuer_authority.requalify()
        _verify_request(source, current, request)
        requester_generation = issuer_authority.requester_registry._current(
            source.authorization.requester_principal_id
        )
        claimant_generation = issuer_authority.claimant_registry._current(
            source.authorization.provisioning_principal_id
        )
        reserved = signing_intent.read_intent(binding, signed_attempt)
        prepared = retention.read_prepared(
            authorization=source.authorization, retained_request=request
        )
        if reserved is None or prepared is None:
            raise AtomicRootProofCommitConflict("EXACT_RESERVED_AND_STAGED_REQUIRED")
        _validate_records(source.authorization, request, reserved, prepared, signing_intent)
        if (
            reserved.root_proof_id != prepared.root_proof_id
            or reserved.canonical_payload != prepared.canonical_payload
            or reserved.credential_snapshot != prepared.credential_snapshot
        ):
            raise AtomicRootProofCommitConflict("EXACT_RESERVED_STAGED_LINKAGE_REQUIRED")
        result = issuer_authority.entitlement_registry.authoritative_state(
            issuer_authority._entitlement_subject
        )
        validate_exact_snapshot(result)
        predecessor = result.state
        if (
            result.outcome is not RegistryReadOutcome.FOUND
            or type(predecessor) is not AuthoritativeEntitlementState
            or predecessor.lifecycle is not EntitlementLifecycle.ACTIVE
            or type(predecessor.binding) is not UnboundBinding
        ):
            raise AtomicRootProofCommitConflict("EXACT_ACTIVE_UNBOUND_PREDECESSOR_REQUIRED")
        auth = source.authorization
        snapshot = prepared.credential_snapshot
        attempted_binding = BoundBinding(
            auth.logical_operation_id,
            auth.account_id,
            auth.canonical_genesis_request_fingerprint_sha256,
            auth.entitlement_generation,
            request.attempt_id,
            auth.requester_principal_id,
            auth.requester_key_id,
            auth.requester_key_version,
            auth.provisioning_principal_id,
            auth.claimant_key_id,
            auth.claimant_key_version,
            request.digest,
            request.reference,
            prepared.root_proof_id,
            snapshot.credential_identity.credential_identity,
            snapshot.key_version,
        )
        bound = AuthoritativeEntitlementState(
            predecessor.subject,
            predecessor.identity,
            predecessor.provenance,
            predecessor.lifecycle,
            attempted_binding,
            predecessor.authoritative_state_revision + 1,
            predecessor.authoritative_state_revision,
        )
        if current.identity is None:
            raise AtomicRootProofCommitConflict("EXACT_SIGNED_ATTEMPT_IDENTITY_REQUIRED")
        latest = history.current_record()
        stream = history._stream
        if (
            stream.environment != auth.environment
            or stream.trust_domain != auth.trust_domain
            or stream.product_scope != auth.product_scope
        ):
            raise AtomicRootProofCommitConflict("EXACT_HISTORY_SCOPE_REQUIRED")
        issuance_payload = _issuance_history_payload(
            auth, bound, predecessor, current.identity.digest_sha256
        )
        # Preserve the existing exact event projection. Immutable SQL decision
        # linkage retains the original signed proof without changing the event.
        record = build_record(
            stream,
            1 if latest is None else latest.sequence + 1,
            NO_PREDECESSOR if latest is None else latest.authenticated_digest,
            HistoryEventIdentity(
                auth.logical_operation_id, request.attempt_id, prepared.root_proof_id
            ),
            issuance_payload,
        )
        locator = AtomicRootProofCommitLocator(
            _canonical(_scope_data(prepared.scope)),
            _canonical(request.attempt_id),
            _canonical(auth.logical_operation_id),
            _canonical(
                {
                    "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
                    "entitlement_generation": auth.entitlement_generation,
                }
            ),
            _canonical(prepared.root_proof_id),
            _canonical(auth.account_id),
        )
        value = PreparedAtomicRootProofCommit(
            reserved,
            prepared,
            bound,
            record,
            locator,
            requester_generation,
            claimant_generation,
        )
        # Each genuine provider owns its pin. Nothing comes from caller-owned
        # schema names; the migrated qualification later resolves these pins.
        trusted = _TrustedPreparation(
            _canonical(_base_package(value)),
            (
                _provider_pin(issuer_authority.entitlement_registry, "_"),
                _provider_pin(signing_intent, "_PostgreSQLRootProofSigningIntent__"),
                _provider_pin(retention, "_PostgreSQLRootProofRetention__"),
                _provider_pin(history, "_"),
                _provider_pin(issuer_authority.requester_registry, "_"),
                _provider_pin(issuer_authority.claimant_registry, "_"),
            ),
        )
        _, requester_revision, claimant_revision = _verify_request(source, current, request)
        if (
            requester_generation.registry_revision != requester_revision
            or claimant_generation.registry_revision != claimant_revision
        ):
            raise AtomicRootProofCommitConflict("EXACT_PUBLIC_CREDENTIAL_PREDECESSORS_REQUIRED")
        with _PREPARATION_LOCK:
            _PREPARATIONS[value] = trusted
        return value
    except AtomicRootProofCommitError:
        raise
    except Exception:
        raise AtomicRootProofCommitUnavailable(
            "GENUINE_PREPARATION_REVERIFICATION_FAILED"
        ) from None


def _trusted_preparation(value: PreparedAtomicRootProofCommit) -> _TrustedPreparation:
    if type(value) is not PreparedAtomicRootProofCommit:
        raise AtomicRootProofCommitConflict("GENUINE_PREPARATION_REQUIRED")
    with _PREPARATION_LOCK:
        trusted = _PREPARATIONS.get(value)
    if trusted is None or trusted.package != _canonical(_base_package(value)):
        raise AtomicRootProofCommitConflict("GENUINE_IMMUTABLE_PREPARATION_REQUIRED")
    return trusted


@dataclass(frozen=True, slots=True)
class AtomicRootProofCommitDecision:
    """Positive immutable database decision and exact original signed proof."""

    marker: Mapping[str, object]
    payload: bytes
    signature: bytes
    envelope: bytes
    state: Literal["COMMITTED"] = "COMMITTED"

    def __post_init__(self) -> None:
        if self.state != "COMMITTED":
            raise TypeError("EXACT_COMMITTED_DECISION_REQUIRED")
        object.__setattr__(self, "marker", _freeze_json(dict(self.marker)))


def _persistence_cut(name: str, connection: psycopg.Connection[Any]) -> None:
    """Test interruption seam; there are no runtime flags or callbacks."""


class PostgreSQLAtomicRootProofCommit:
    """Explicit internal isolated-test authority, never a production issuer port."""

    __slots__ = ("__deployment", "__admin_connection")
    __deployment: PostgreSQLAtomicRootProofTestDeployment
    __admin_connection: PostgreSQLConnectionConfig | None

    def __init__(
        self,
        deployment: PostgreSQLAtomicRootProofTestDeployment,
        *,
        admin_connection: PostgreSQLConnectionConfig | None = None,
    ) -> None:
        if (
            type(deployment) is not PostgreSQLAtomicRootProofTestDeployment
            or admin_connection is not None
            and type(admin_connection) is not PostgreSQLConnectionConfig
        ):
            raise TypeError("EXACT_QUALIFIED_INTERNAL_TEST_AUTHORITY_REQUIRED")
        validate_atomic_root_proof_deployment(deployment)
        object.__setattr__(self, "_PostgreSQLAtomicRootProofCommit__deployment", deployment)
        object.__setattr__(
            self, "_PostgreSQLAtomicRootProofCommit__admin_connection", admin_connection
        )
        with self._connect() as conn:
            qualify_atomic_root_proof_authority(conn, deployment)

    def __init_subclass__(cls) -> None:
        raise TypeError("INTERNAL_ATOMIC_AUTHORITY_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("INTERNAL_ATOMIC_AUTHORITY_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return "PostgreSQLAtomicRootProofCommit(connection=<redacted>, production_ready=False)"

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.__deployment.writer_connection.dsn, autocommit=True)

    def approve_prepared(
        self,
        preparation: PreparedAtomicRootProofCommit,
        head: AttestedHistoryHead,
        history_signer: LocalHistoryAttestationSigningProvider,
        *,
        fences: AtomicRootProofCommitFences,
    ) -> None:
        """Test administration retains only genuine, independently verified bytes.

        All custody verification finishes before opening the DB transaction.
        This is not a signer: an isolated fixture already owns the signature.
        """
        trusted = _trusted_preparation(preparation)
        if self.__admin_connection is None:
            raise AtomicRootProofCommitConflict("SEPARATE_TEST_ADMIN_APPROVAL_PRINCIPAL_REQUIRED")
        if (
            type(history_signer) is not LocalHistoryAttestationSigningProvider
            or type(fences) is not AtomicRootProofCommitFences
        ):
            raise AtomicRootProofCommitConflict("GENUINE_HISTORY_AUTHORITY_AND_FENCES_REQUIRED")
        admitted = _snapshot_head(head)
        record = preparation.history_record
        if (
            admitted.sequence != record.sequence
            or admitted.record_digest != record.authenticated_digest
        ):
            raise AtomicRootProofCommitConflict("EXACT_HISTORY_HEAD_SUCCESSOR_REQUIRED")
        snapshot = history_signer.credential_snapshot()
        request = preparation.prepared.retained_request
        requester = request.requester
        claimant = request.claimant
        if requester is None or claimant is None:
            raise AtomicRootProofCommitCorrupt("EXACT_RETAINED_REQUEST_SIGNERS_REQUIRED")
        if (
            snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
            or snapshot.credential_identity != admitted.signing_credential_identity
            or snapshot.provider_identity.security
            != preparation.prepared.credential_snapshot.provider_identity.security
            or snapshot.key_material_identity
            in {
                preparation.prepared.credential_snapshot.key_material_identity,
                requester[0].key_material_identity,
                claimant[0].key_material_identity,
            }
            or fences.root_key != preparation.root_fence_key
            or fences.root_generation
            != preparation.prepared.credential_snapshot.lifecycle_generation
            or fences.history_key != atomic_history_fence_key(admitted)
            or fences.history_generation != snapshot.lifecycle_generation
            or fences.trust_key != preparation.trust_fence_key
        ):
            raise AtomicRootProofCommitConflict("EXACT_ACTIVE_SIGNER_FENCE_BINDING_REQUIRED")
        verify_head(admitted, history_signer, expected_stream=record.stream)
        if history_signer.credential_snapshot() != snapshot:
            raise AtomicRootProofCommitConflict("HISTORY_SIGNER_CHANGED_DURING_VERIFICATION")
        package = _base_package(preparation)
        package.update(
            history_head={
                "sequence": admitted.sequence,
                "record_digest": admitted.record_digest,
                "canonical_credential_hex": canonical_json_bytes(
                    _credential_material(admitted.signing_credential_identity)
                ).hex(),
            },
            history_head_hex=admitted.canonical_attestation_bytes.hex(),
            history_signature_hex=admitted.signature.hex(),
            root_fence_key=fences.root_key,
            root_fence_generation=fences.root_generation,
            history_fence_key=fences.history_key,
            history_fence_generation=fences.history_generation,
            trust_fence_key=fences.trust_key,
            trust_fence_generation=fences.trust_generation,
        )
        provisioning = self.__deployment.provisioning
        selected_schemas = (
            provisioning.entitlement_schema,
            provisioning.intent_schema,
            provisioning.retention_schema,
            provisioning.history_schema,
            provisioning.requester_schema,
            provisioning.claimant_schema,
        )
        if tuple(pin[1] for pin in trusted.pins) != selected_schemas:
            raise AtomicRootProofCommitConflict("EXACT_MIGRATED_AUTHORITY_PINS_REQUIRED")
        try:
            for connection, _ in trusted.pins:
                with psycopg.connect(connection.dsn, autocommit=True) as participant:
                    identity = read_atomic_root_proof_physical_identity(
                        participant, provisioning.instance_id
                    )
                    if identity != self.__deployment.identity:
                        raise AtomicRootProofCommitConflict("ONE_PHYSICAL_DATABASE_REQUIRED")
            with psycopg.connect(self.__admin_connection.dsn, autocommit=True) as conn:
                qualify_atomic_root_proof_authority(
                    conn, self.__deployment, expected_role="test_admin"
                )
                with conn.transaction():
                    conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                    conn.execute("SET LOCAL search_path=pg_catalog")
                    conn.execute(
                        sql.SQL("SELECT {}.approve_commit(%s)").format(
                            sql.Identifier(provisioning.schema)
                        ),
                        (Jsonb(package),),
                    ).fetchone()
        except AtomicRootProofCommitError:
            raise
        except psycopg.errors.UniqueViolation:
            raise AtomicRootProofCommitConflict("IMMUTABLE_APPROVED_PACKAGE_CONFLICT") from None
        except Exception:
            raise AtomicRootProofCommitUnavailable("TEST_ADMIN_APPROVAL_UNAVAILABLE") from None

    @staticmethod
    def _locator(
        value: PreparedAtomicRootProofCommit | AtomicRootProofCommitLocator,
    ) -> AtomicRootProofCommitLocator:
        if type(value) is PreparedAtomicRootProofCommit:
            return value.locator
        if type(value) is AtomicRootProofCommitLocator:
            return value
        raise AtomicRootProofCommitConflict("EXACT_RECOVERY_LOCATOR_REQUIRED")

    @staticmethod
    def _decision(
        value: object, locator: AtomicRootProofCommitLocator
    ) -> AtomicRootProofCommitDecision:
        if value is None:
            # Even an authoritative fresh snapshot's absence cannot establish
            # the outcome of an earlier connection whose COMMIT response was lost.
            raise AtomicRootProofCommitIndeterminate("DECISION_NOT_FOUND_PRIOR_OUTCOME_UNRESOLVED")
        if type(value) is not dict:
            raise AtomicRootProofCommitCorrupt("EXACT_DATABASE_DECISION_REQUIRED")
        PostgreSQLAtomicRootProofCommit._require_identity(value, locator)
        try:
            payload = bytes.fromhex(value["payload_hex"])
            signature = bytes.fromhex(value["signature_hex"])
            envelope = bytes.fromhex(value["staged_envelope_hex"])
            if (
                any(
                    type(value[key]) is not str or raw.hex() != value[key]
                    for key, raw in (
                        ("payload_hex", payload),
                        ("signature_hex", signature),
                        ("staged_envelope_hex", envelope),
                    )
                )
                or len(signature) != 64
            ):
                raise ValueError("EXACT_ORIGINAL_BYTEA_REQUIRED")
        except (ValueError, TypeError, KeyError):
            raise AtomicRootProofCommitCorrupt("RETAINED_DECISION_BYTES_INVALID") from None
        return AtomicRootProofCommitDecision(
            cast(Mapping[str, object], value), payload, signature, envelope
        )

    @staticmethod
    def _require_identity(value: object, locator: AtomicRootProofCommitLocator) -> None:
        if type(value) is not dict:
            raise AtomicRootProofCommitConflict("PROTECTED_APPROVED_ATTEMPT_REQUIRED")
        expected = {
            "scope_hex": locator.scope.hex(),
            "attempt_hex": locator.attempt.hex(),
            "operation_hex": locator.operation.hex(),
            "entitlement_hex": locator.entitlement.hex(),
            "root_proof_hex": locator.root_proof.hex(),
            "account_hex": locator.account.hex(),
        }
        if any(value.get(key) != selected for key, selected in expected.items()):
            raise AtomicRootProofCommitConflict("EXACT_IMMUTABLE_DECISION_IDENTITY_REQUIRED")

    def commit(
        self, preparation: PreparedAtomicRootProofCommit | AtomicRootProofCommitLocator
    ) -> AtomicRootProofCommitDecision:
        """Submit one approved exact attempt in one physical SERIALIZABLE tx.

        There is no automatic write retry, replacement attempt, private-key
        operation, compensating UNBIND or checkpoint/CHA advancement.
        """
        locator = self._locator(preparation)
        if type(preparation) is PreparedAtomicRootProofCommit:
            _trusted_preparation(preparation)
        attempted = False
        commit_sent = False
        conn: psycopg.Connection[Any] | None = None
        try:
            conn = self._connect()
            conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            qualify_atomic_root_proof_authority(conn, self.__deployment)
            approval = conn.execute(
                sql.SQL("SELECT {}.read_approved_identity(%s)").format(
                    sql.Identifier(self.__deployment.provisioning.schema)
                ),
                (locator.attempt,),
            ).fetchone()
            if approval is None or len(approval) != 1:
                raise AtomicRootProofCommitConflict("PROTECTED_APPROVAL_READ_REQUIRED")
            self._require_identity(approval[0], locator)
            attempted = True
            row = conn.execute(
                sql.SQL("SELECT {}.commit_root_proof(%s)").format(
                    sql.Identifier(self.__deployment.provisioning.schema)
                ),
                (locator.attempt,),
            ).fetchone()
            if row is None or len(row) != 1:
                raise AtomicRootProofCommitCorrupt("EXACT_SQL_DECISION_CARDINALITY_REQUIRED")
            decision = self._decision(row[0], locator)
            if type(preparation) is PreparedAtomicRootProofCommit and (
                decision.payload != preparation.prepared.canonical_payload
                or decision.signature != preparation.prepared.signature
                or decision.envelope != preparation.prepared.canonical_envelope
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
            raise AtomicRootProofCommitConflict(
                "ATOMIC_SQL_IDENTITY_CAS_OR_ELIGIBILITY_REJECTED"
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
                # Closing an open transaction aborts its whole physical write
                # set. A failed close never changes an INDETERMINATE result.
                conn.close()

    def reconcile(
        self, value: PreparedAtomicRootProofCommit | AtomicRootProofCommitLocator
    ) -> AtomicRootProofCommitDecision:
        """Read the exact authoritative marker; absent rows stay INDETERMINATE.

        Recovery after client restart accepts only identity locators for reads.
        Its only positive result is the existing byte-identical decision.
        """
        locator = self._locator(value)
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
                conn.execute("SET LOCAL search_path=pg_catalog")
                qualify_atomic_root_proof_authority(conn, self.__deployment)
                row = conn.execute(
                    sql.SQL("SELECT {}.read_decision(%s,%s)").format(
                        sql.Identifier(self.__deployment.provisioning.schema)
                    ),
                    (locator.scope, locator.attempt),
                ).fetchone()
                if row is None or len(row) != 1:
                    raise AtomicRootProofCommitCorrupt("EXACT_DECISION_READ_CARDINALITY_REQUIRED")
                return self._decision(row[0], locator)
        except AtomicRootProofCommitError:
            raise
        except Exception:
            raise AtomicRootProofCommitIndeterminate(
                "AUTHORITATIVE_RECONCILIATION_UNAVAILABLE"
            ) from None
