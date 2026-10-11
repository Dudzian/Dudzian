"""Odczytowe porównanie autorytetów Stage 9, bez uprawnień do emisji.

Każdy raport opisuje dwa niezależne zestawy odczytów. Ich zgodność nie jest
dowodem atomowego snapshotu kilku baz ani RootProofIssuanceReconciliationEvidenceV1.
Wyłącznie zaufany composition root wybiera rzeczywiste porty PostgreSQL.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Callable, Literal, TypeVar, cast

from bot_core.authenticated_issuer_history import (
    HistoricalRevokedSignatureVerificationUnavailable,
    HistoryContractError,
    HistoryRecord,
    HistorySigningAuthorityUnavailable,
    ReconciliationOutcome,
)
from bot_core.cha_attempt_signatures import verify_final_identity
from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptConflictError,
    AttemptCorruptError,
    AttemptNotFoundError,
    AttemptSchemaUnsupportedError,
    AttemptState,
    AttemptStoreUnavailableError,
)
from bot_core.cha_issuance_request import IssuanceSigningRole
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    BoundBinding,
    ContractValidationError,
    EntitlementLifecycle,
    RegistryReadOutcome,
    RegistryReadResult,
    RetainedHistoryResult,
    UnboundBinding,
    binding_identity_digest,
    predecessor_for,
    validate_exact_snapshot,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.cha_account_reservation import (
    AccountInitialBindingError,
    _initial_binding_snapshot,
)
from bot_core.licensing.cha_root_proof_signed_attempt import (
    SignedIssuanceAttemptError,
    _IssuanceReconciliationSource,
    _read_issuance_reconciliation_source,
)
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.postgresql_authenticated_issuer_history import (
    HistoryOperationIndeterminate,
    HistoryReadSnapshot,
    HistoryStorageUnavailable,
    PostgreSQLAuthenticatedIssuerHistory,
)
from bot_core.postgresql_checkpoint_authority import (
    CheckpointHistoryMismatchError,
    CheckpointOperationIndeterminate,
    CheckpointReadSnapshot,
    CheckpointStorageUnavailable,
    PostgreSQLCheckpointAuthority,
)
from bot_core.postgresql_preaccount_credentials import (
    CredentialGeneration,
    CredentialLifecycle,
    CredentialResolutionError,
)
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    HistoryAttestationSigningProvider,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    public_key_material_identity,
)


class RootProofIssuanceReconciliationError(RuntimeError):
    """Brak wybranego, dokładnego zaufanego composition root."""


@dataclass(frozen=True, slots=True)
class _ReconciliationRuntime:
    issuer_authority: PostgreSQLRootProofIssuanceAuthority
    history: PostgreSQLAuthenticatedIssuerHistory
    checkpoint: PostgreSQLCheckpointAuthority
    verification_authorities: tuple[HistoryAttestationSigningProvider, ...]


# Instalacja należy wyłącznie do wewnętrznego composition root wdrożenia.
# Brak konfiguracji nie wybiera zastępczego providera ani pamięciowego logu.
_RUNTIME_CONFIGURATION: _ReconciliationRuntime | None = None


def _validate_runtime(runtime: _ReconciliationRuntime) -> None:
    if (
        type(runtime) is not _ReconciliationRuntime
        or type(runtime.issuer_authority) is not PostgreSQLRootProofIssuanceAuthority
        or type(runtime.history) is not PostgreSQLAuthenticatedIssuerHistory
        or type(runtime.checkpoint) is not PostgreSQLCheckpointAuthority
        or runtime.checkpoint._history is not runtime.history
        or runtime.checkpoint._authorities is not runtime.verification_authorities
        or type(runtime.verification_authorities) is not tuple
        or not runtime.verification_authorities
    ):
        raise RootProofIssuanceReconciliationError("EXACT_PINNED_AUTHORITIES_REQUIRED")
    subject = runtime.issuer_authority._entitlement_subject
    stream = runtime.history._stream
    if (
        runtime.checkpoint._stream != stream
        or stream.security_profile != SecurityProfile.PRODUCTION_LOCAL.value
        or stream.environment != subject.environment
        or stream.environment != "PRODUCTION"
        or stream.trust_domain != subject.trust_domain
    ):
        raise RootProofIssuanceReconciliationError("PINNED_AUTHORITY_SCOPE_MISMATCH")
    for signer in runtime.verification_authorities:
        identity = signer.identity
        if (
            type(identity) is not ProviderIdentity
            or identity.role is not ProviderRole.HISTORY_ATTESTATION_SIGNING
            or identity.security.profile is not SecurityProfile.PRODUCTION_LOCAL
            or identity.security.trust_domain != stream.trust_domain
        ):
            raise RootProofIssuanceReconciliationError("PINNED_HISTORY_SIGNER_SCOPE_MISMATCH")


def _compose_reconciliation_runtime(
    issuer_authority: PostgreSQLRootProofIssuanceAuthority,
    history: PostgreSQLAuthenticatedIssuerHistory,
    checkpoint: PostgreSQLCheckpointAuthority,
) -> _ReconciliationRuntime:
    """Wewnętrzna kompozycja; żaden port nie jest argumentem publicznego API."""
    if type(checkpoint) is not PostgreSQLCheckpointAuthority:
        raise TypeError("checkpoint must be the exact selected PostgreSQL authority")
    runtime = _ReconciliationRuntime(issuer_authority, history, checkpoint, checkpoint._authorities)
    _validate_runtime(runtime)
    return runtime


def _configured_reconciliation_runtime() -> _ReconciliationRuntime:
    runtime = _RUNTIME_CONFIGURATION
    if runtime is None:
        raise RootProofIssuanceReconciliationError("MISSING_RECONCILIATION_RUNTIME_CONFIGURATION")
    _validate_runtime(runtime)
    return runtime


def _cut(name: str) -> None:
    """Punkt testowego przerwania odczytu; nie przyznaje uprawnień."""


def _plain(value: object) -> object:
    """Odłącz wyłącznie znane wartości raportowe do danych pierwotnych."""
    if isinstance(value, Enum):
        return value.value
    if type(value) is bytes:
        return value.hex()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_plain(item) for item in cast(tuple[object, ...], value)]
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError("non-reporting object cannot enter an observation")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes({"observation": _plain(value)})).hexdigest()


@dataclass(frozen=True, slots=True)
class ReconciliationReadFailure:
    authority: str
    outcome: ReconciliationOutcome
    reason: str
    observed_revisions: tuple[int, ...] = ()
    observed_digests: tuple[str | None, ...] = ()


@dataclass(frozen=True, slots=True)
class ReconciliationObservation:
    """Odłączone identyfikatory i rewizje, bez portów, kluczy lub capability."""

    identifiers: tuple[tuple[str, str], ...] = ()
    registry_outcome: str | None = None
    registry_revision: int | None = None
    registry_predecessor_revision: int | None = None
    registry_binding_kind: str | None = None
    registry_binding_digest: str | None = None
    registry_state_digest: str | None = None
    registry_state_identifiers: tuple[tuple[str, str | int | None], ...] = ()
    registry_binding_identifiers: tuple[tuple[str, str | int | None], ...] = ()
    registry_retained_revisions: tuple[int, ...] = ()
    registry_retained_digest: str | None = None
    attempt_fence: int | None = None
    attempt_state: str | None = None
    attempt_identity_digest_sha256: str | None = None
    source_digest: str | None = None
    requester_registry_revision: int | None = None
    claimant_registry_revision: int | None = None
    requester_retained_revisions: tuple[int, ...] = ()
    claimant_retained_revisions: tuple[int, ...] = ()
    requester_history_digest: str | None = None
    claimant_history_digest: str | None = None
    history_sequence: int | None = None
    history_authenticated_digest: str | None = None
    history_stream_identifiers: tuple[tuple[str, str | int | None], ...] = ()
    history_record_observations: tuple[tuple[int, str, str, str, str], ...] = ()
    history_snapshot_digest: str | None = None
    history_head_observations: tuple[tuple[int, str, str, str], ...] = ()
    checkpoint_revision: int | None = None
    checkpoint_history_sequence: int | None = None
    checkpoint_authenticated_digest: str | None = None
    checkpoint_stream_identifiers: tuple[tuple[str, str | int | None], ...] = ()
    checkpoint_retained_revisions: tuple[int, ...] = ()
    checkpoint_acceptance_observations: tuple[tuple[int, int, str, str], ...] = ()
    checkpoint_snapshot_digest: str | None = None
    signer_observations: tuple[tuple[str, str, int, str], ...] = ()
    read_failures: tuple[ReconciliationReadFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class RootProofIssuanceReconciliationReport:
    outcome: ReconciliationOutcome
    disposition: Literal["PRE_ISSUANCE_OBSERVED", "ALIGNED_COMMITTED_OBSERVED", "REFUSED"]
    observations: tuple[ReconciliationObservation, ...]
    reasons: tuple[str, ...]
    atomic_global_snapshot: Literal[False] = False
    authorizes_operations: Literal[False] = False


def _failure(authority: str, error: Exception) -> ReconciliationReadFailure:
    # Nigdy nie umieszczaj tekstu wyjątków DB/custody w raporcie: może zawierać DSN.
    if isinstance(error, (HistoryOperationIndeterminate, CheckpointOperationIndeterminate)):
        outcome, reason = ReconciliationOutcome.INDETERMINATE, "AUTHORITY_OUTCOME_INDETERMINATE"
    elif isinstance(error, (AttemptNotFoundError, FileNotFoundError)):
        outcome, reason = ReconciliationOutcome.NOT_FOUND, "EXACT_DURABLE_ATTEMPT_NOT_FOUND"
    elif isinstance(error, (AttemptConflictError, AccountInitialBindingError)):
        outcome, reason = ReconciliationOutcome.CONFLICT, "EXACT_RETAINED_IDENTITY_REQUIRED"
    elif isinstance(error, CheckpointHistoryMismatchError):
        return ReconciliationReadFailure(
            authority,
            ReconciliationOutcome.CORRUPT,
            error.reason,
            (error.history_sequence, error.checkpoint_sequence),
            (error.checkpoint_record_digest, error.observed_record_digest),
        )
    elif isinstance(
        error,
        (
            HistoryStorageUnavailable,
            CheckpointStorageUnavailable,
            HistorySigningAuthorityUnavailable,
            HistoricalRevokedSignatureVerificationUnavailable,
            AttemptStoreUnavailableError,
        ),
    ):
        outcome, reason = ReconciliationOutcome.UNAVAILABLE, "AUTHORITATIVE_READ_UNAVAILABLE"
    elif isinstance(
        error,
        (
            HistoryContractError,
            AttemptCorruptError,
            AttemptSchemaUnsupportedError,
            ContractValidationError,
            CredentialResolutionError,
            ValueError,
            TypeError,
            KeyError,
        ),
    ):
        outcome, reason = ReconciliationOutcome.CORRUPT, "AUTHORITATIVE_DATA_CORRUPT"
    elif isinstance(error, SignedIssuanceAttemptError):
        outcome, reason = ReconciliationOutcome.CONFLICT, "EXACT_IMMUTABLE_ATTEMPT_REQUIRED"
    else:
        outcome, reason = ReconciliationOutcome.UNAVAILABLE, "AUTHORITATIVE_READ_UNAVAILABLE"
    return ReconciliationReadFailure(authority, outcome, reason)


T = TypeVar("T")


def _read(
    authority: str, operation: Callable[[], T], failures: list[ReconciliationReadFailure]
) -> T | None:
    try:
        return operation()
    except Exception as error:
        failures.append(_failure(authority, error))
        return None


@dataclass(frozen=True, slots=True)
class _Sample:
    source: _IssuanceReconciliationSource | None
    registry: RegistryReadResult | None
    registry_history: RetainedHistoryResult | None
    history: HistoryReadSnapshot | None
    checkpoint: CheckpointReadSnapshot | None
    requester_history: tuple[CredentialGeneration, ...] | None
    claimant_history: tuple[CredentialGeneration, ...] | None
    observation: ReconciliationObservation


def _selected_credential(
    history: tuple[CredentialGeneration, ...], key_id: str, version: int
) -> CredentialGeneration | None:
    selected = tuple(
        item
        for item in history
        if item.key_id == key_id
        and item.key_version == version
        and item.lifecycle is CredentialLifecycle.ACTIVE
    )
    return selected[0] if len(selected) == 1 else None


def _signer_observations(
    runtime: _ReconciliationRuntime,
) -> tuple[tuple[str, str, int, str], ...]:
    observations = []
    for signer in runtime.verification_authorities:
        lifecycle_reader = getattr(signer, "lifecycle_state", None)
        generation_reader = getattr(signer, "lifecycle_generation", None)
        if not callable(lifecycle_reader) or not callable(generation_reader):
            raise HistorySigningAuthorityUnavailable("selected signing lifecycle unavailable")
        lifecycle = lifecycle_reader()
        generation = generation_reader()
        if (
            type(lifecycle) is not SigningKeyLifecycle
            or type(generation) is not int
            or generation < 1
        ):
            raise HistoryContractError("invalid selected signer lifecycle")
        credentials = signer.credential_identities()
        if type(credentials) is not tuple or not credentials:
            raise HistorySigningAuthorityUnavailable("selected signing credentials unavailable")
        for credential in credentials:
            if (
                type(credential) is not CredentialRoleIdentity
                or credential.semantic_role
                is not CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING
                or credential.provider_namespace != signer.identity.provider_namespace
            ):
                raise HistoryContractError("invalid selected signer credential identity")
            public = signer.public_key(credential.credential_identity)
            if public_key_material_identity(public) != credential.key_material_identity:
                raise HistoryContractError("invalid selected signer public material")
            observations.append(
                (
                    credential.credential_identity,
                    lifecycle.value,
                    generation,
                    _digest((signer.identity, credential, public)),
                )
            )
    return tuple(observations)


def _identifiers(value: object, prefix: str = "") -> tuple[tuple[str, str | int | None], ...]:
    plain = _plain(value)
    if isinstance(plain, Mapping):
        return tuple(
            pair
            for key, item in plain.items()
            for pair in _identifiers(item, f"{prefix}.{key}" if prefix else str(key))
        )
    if plain is None or type(plain) in (str, int):
        return ((prefix, cast(str | int | None, plain)),)
    raise TypeError("exact primitive identity required")


def _capture(binding: object, runtime: _ReconciliationRuntime) -> _Sample:
    failures: list[ReconciliationReadFailure] = []
    source = _read(
        "CHA_ATTEMPT_STORE", lambda: _read_issuance_reconciliation_source(binding), failures
    )
    if source is None:
        return _Sample(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ReconciliationObservation(read_failures=tuple(failures)),
        )
    if source.authority is not runtime.issuer_authority:
        failures.append(
            ReconciliationReadFailure(
                "COMPOSITION_ROOT", ReconciliationOutcome.CONFLICT, "FOREIGN_ISSUER_AUTHORITY"
            )
        )
        return _Sample(
            source,
            None,
            None,
            None,
            None,
            None,
            None,
            ReconciliationObservation(read_failures=tuple(failures)),
        )
    auth = source.authorization
    registry_port = runtime.issuer_authority.entitlement_registry
    subject = runtime.issuer_authority._entitlement_subject
    registry = _read(
        "ENTITLEMENT_REGISTRY", lambda: registry_port.authoritative_state(subject), failures
    )
    registry_history = _read(
        "ENTITLEMENT_HISTORY", lambda: registry_port.retained_history(subject), failures
    )
    requester = _read(
        "REQUESTER_REGISTRY",
        lambda: runtime.issuer_authority.requester_registry.retained_history(
            auth.requester_principal_id
        ),
        failures,
    )
    claimant = _read(
        "CLAIMANT_REGISTRY",
        lambda: runtime.issuer_authority.claimant_registry.retained_history(
            auth.provisioning_principal_id
        ),
        failures,
    )
    history = _read(
        "ISSUER_HISTORY",
        lambda: runtime.history.authenticated_read_snapshot(
            verification_authorities=runtime.verification_authorities,
            historical_evidence_authority=runtime.checkpoint,
        ),
        failures,
    )
    checkpoint = _read(
        "CHECKPOINT_AUTHORITY", runtime.checkpoint.authenticated_read_snapshot, failures
    )
    signers = _read("HISTORY_SIGNER", lambda: _signer_observations(runtime), failures)
    state = registry.state if registry is not None else None
    tail = history.records[-1] if history is not None and history.records else None
    current = checkpoint.current if checkpoint is not None else None
    requester_selected = (
        _selected_credential(requester, auth.requester_key_id, auth.requester_key_version)
        if requester is not None
        else None
    )
    claimant_selected = (
        _selected_credential(claimant, auth.claimant_key_id, auth.claimant_key_version)
        if claimant is not None
        else None
    )
    identifiers = tuple(
        (name, str(getattr(auth, name)))
        for name in (
            "environment",
            "trust_domain",
            "product_scope",
            "logical_operation_id",
            "account_id",
            "bootstrap_entitlement_id",
            "entitlement_generation",
            "initial_binding_reference",
            "initial_binding_digest_sha256",
            "authorization_evidence_sha256",
            "requester_principal_id",
            "requester_key_id",
            "requester_key_version",
            "provisioning_principal_id",
            "claimant_key_id",
            "claimant_key_version",
            "reservation_identity",
            "reservation_relation",
        )
    ) + (
        ("issuance_attempt_id", source.current.reservation.issuance_attempt_id),
        ("request_reference", source.retained.reference if source.retained else "NOT_DEFINED"),
        ("request_digest_sha256", source.retained.digest if source.retained else "NOT_DEFINED"),
        ("issuer_authority_identity", runtime.history._stream.issuer_authority_identity),
        ("issuer_registry_identity", registry_port.identity.provider_namespace),
        ("entitlement_lookup_handle", subject.lookup_handle),
        ("history_stream_id", runtime.history._stream.stream_id),
        ("history_authority_identity", runtime.history.identity.provider_namespace),
        ("checkpoint_id", runtime.checkpoint._checkpoint_id),
        (
            "checkpoint_authority_identity",
            checkpoint.authority_identity if checkpoint else "UNAVAILABLE",
        ),
    )
    observation = ReconciliationObservation(
        identifiers=identifiers,
        registry_outcome=registry.outcome.value if registry else None,
        registry_revision=state.authoritative_state_revision if state else None,
        registry_predecessor_revision=state.predecessor_revision if state else None,
        registry_binding_kind=state.binding.kind.value if state else None,
        registry_binding_digest=binding_identity_digest(state.binding) if state else None,
        registry_state_digest=_digest(state) if state else None,
        registry_state_identifiers=_identifiers(
            {
                "subject": state.subject,
                "identity": state.identity,
                "provenance": state.provenance,
                "lifecycle": state.lifecycle,
            }
        )
        if state
        else (),
        registry_binding_identifiers=_identifiers(state.binding) if state else (),
        registry_retained_revisions=tuple(
            item.authoritative_state_revision for item in registry_history.states
        )
        if registry_history
        else (),
        registry_retained_digest=_digest(registry_history) if registry_history else None,
        attempt_fence=source.current.fence,
        attempt_state=source.current.state.value,
        attempt_identity_digest_sha256=source.current.immutable_attempt_digest_sha256,
        source_digest=_digest(
            (source.binding_state_raw, source.context_raw, source.current, source.retained)
        ),
        requester_registry_revision=requester_selected.registry_revision
        if requester_selected
        else None,
        claimant_registry_revision=claimant_selected.registry_revision
        if claimant_selected
        else None,
        requester_retained_revisions=tuple(item.registry_revision for item in requester)
        if requester is not None
        else (),
        claimant_retained_revisions=tuple(item.registry_revision for item in claimant)
        if claimant is not None
        else (),
        requester_history_digest=_digest(requester) if requester is not None else None,
        claimant_history_digest=_digest(claimant) if claimant is not None else None,
        history_sequence=tail.sequence if tail else None,
        history_authenticated_digest=tail.authenticated_digest if tail else None,
        history_stream_identifiers=_identifiers(history.stream) if history else (),
        history_record_observations=tuple(
            (
                item.sequence,
                item.authenticated_digest,
                item.event_identity.logical_operation_id,
                item.event_identity.issuance_attempt_id,
                item.event_identity.root_proof_id,
            )
            for item in history.records
        )
        if history
        else (),
        history_snapshot_digest=_digest(history) if history else None,
        history_head_observations=tuple(
            (
                item.sequence,
                item.record_digest,
                item.signing_credential_identity.credential_identity,
                _digest(item),
            )
            for item in history.heads
        )
        if history
        else (),
        checkpoint_revision=current.checkpoint_revision if current else None,
        checkpoint_history_sequence=current.history_sequence if current else None,
        checkpoint_authenticated_digest=current.authenticated_head_digest if current else None,
        checkpoint_stream_identifiers=_identifiers(checkpoint.stream) if checkpoint else (),
        checkpoint_retained_revisions=tuple(
            item.checkpoint_revision for item in checkpoint.checkpoints
        )
        if checkpoint
        else (),
        checkpoint_acceptance_observations=tuple(
            (
                item.checkpoint_revision,
                item.accepted_history_sequence,
                item.accepted_authenticated_head_digest,
                item.acceptance_digest,
            )
            for item in checkpoint.acceptances
        )
        if checkpoint
        else (),
        checkpoint_snapshot_digest=_digest(checkpoint) if checkpoint else None,
        signer_observations=signers or (),
        read_failures=tuple(failures),
    )
    return _Sample(
        source, registry, registry_history, history, checkpoint, requester, claimant, observation
    )


def _expected_issuance_history_payload(
    authorization: AttemptAuthorization,
    state: AuthoritativeEntitlementState,
    predecessor: AuthoritativeEntitlementState,
    attempt_identity_digest_sha256: str,
) -> dict[str, object]:
    """Lokalna projekcja porównawcza; nie jest frozen schema emisji historii.

    Generic history nie określa znaczenia payloadu. Ta funkcja wyłącznie porównuje
    istniejący zapis ze stanem registry i pełnym CAS, bez dopisywania zdarzeń.
    """
    return {
        "schema_version": "1",
        "observation_kind": "ROOT_PROOF_ISSUANCE_BINDING_OBSERVATION",
        "authorization": asdict(authorization),
        "entitlement": asdict(state),
        "predecessor": asdict(predecessor_for(predecessor)),
        "attempt_identity_digest_sha256": attempt_identity_digest_sha256,
    }


def _verify_retained_signatures(sample: _Sample) -> bool:
    source = sample.source
    if (
        source is None
        or source.retained is None
        or source.current.identity is None
        or sample.requester_history is None
        or sample.claimant_history is None
    ):
        return False
    auth, retained = source.authorization, source.retained
    verify_final_identity(source.current.identity, retained)
    publics: list[bytes] = []
    for history, checkpoint, key_id, version, principal, role, semantic in (
        (
            sample.requester_history,
            retained.requester,
            auth.requester_key_id,
            auth.requester_key_version,
            auth.requester_principal_id,
            IssuanceSigningRole.REQUESTER,
            CredentialSemanticRole.ROOT_PROOF_REQUESTER,
        ),
        (
            sample.claimant_history,
            retained.claimant,
            auth.claimant_key_id,
            auth.claimant_key_version,
            auth.provisioning_principal_id,
            IssuanceSigningRole.CLAIMANT,
            CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
        ),
    ):
        credential = _selected_credential(history, key_id, version)
        if credential is None or checkpoint is None:
            return False
        signer, signature = checkpoint
        signer.require_authorization(auth)
        if (
            credential.principal_id != principal
            or credential.semantic_role is not semantic
            or credential.environment != auth.environment
            or credential.trust_domain != auth.trust_domain
            or credential.key_material_identity
            != public_key_material_identity(credential.public_key)
            or signer.role is not role
            or signer.public_key_hex != credential.public_key.hex()
            or signer.key_material_identity != credential.key_material_identity
        ):
            return False
        signer.verify(retained.canonical_bytes, signature)
        publics.append(credential.public_key)
    return publics[0] != publics[1]


def _authorization_anchor_matches(
    state: AuthoritativeEntitlementState, authorization: AttemptAuthorization
) -> bool:
    return all(
        (
            state.identity.environment == authorization.environment,
            state.identity.trust_domain == authorization.trust_domain,
            state.identity.product_scope == authorization.product_scope,
            state.identity.bootstrap_entitlement_id == authorization.bootstrap_entitlement_id,
            state.identity.entitlement_generation == authorization.entitlement_generation,
            state.provenance.provisioning_principal_id == authorization.provisioning_principal_id,
            state.provenance.claimant_key_id == authorization.claimant_key_id,
            state.provenance.claimant_key_version == authorization.claimant_key_version,
        )
    )


def _classify(
    sample: _Sample, runtime: _ReconciliationRuntime
) -> tuple[ReconciliationOutcome, str, tuple[str, ...]]:
    failures = sample.observation.read_failures
    if failures:
        priorities = (
            ReconciliationOutcome.CORRUPT,
            ReconciliationOutcome.CONFLICT,
            ReconciliationOutcome.INDETERMINATE,
            ReconciliationOutcome.UNAVAILABLE,
            ReconciliationOutcome.NOT_FOUND,
        )
        outcome = next(item for item in priorities if any(f.outcome is item for f in failures))
        return outcome, "REFUSED", tuple(f"{f.authority}:{f.reason}" for f in failures)
    source, registry, retained = sample.source, sample.registry, sample.registry_history
    history, checkpoint = sample.history, sample.checkpoint
    if (
        source is None
        or registry is None
        or retained is None
        or history is None
        or checkpoint is None
    ):
        return ReconciliationOutcome.UNAVAILABLE, "REFUSED", ("INCOMPLETE_AUTHORITATIVE_READS",)
    validate_exact_snapshot(registry)
    validate_exact_snapshot(retained)
    if {registry.outcome, retained.outcome} == {
        RegistryReadOutcome.FOUND,
        RegistryReadOutcome.NOT_FOUND,
    }:
        return (
            ReconciliationOutcome.INDETERMINATE,
            "REFUSED",
            ("REGISTRY_READS_CHANGED_WITHIN_OBSERVATION",),
        )
    for result in (registry, retained):
        if result.outcome is not RegistryReadOutcome.FOUND:
            outcome = {
                RegistryReadOutcome.CORRUPT: ReconciliationOutcome.CORRUPT,
                RegistryReadOutcome.NOT_FOUND: ReconciliationOutcome.NOT_FOUND,
                RegistryReadOutcome.UNAVAILABLE: ReconciliationOutcome.UNAVAILABLE,
            }[result.outcome]
            return outcome, "REFUSED", ("AUTHORITATIVE_ENTITLEMENT_REQUIRED",)
    state = registry.state
    if state is None or not retained.states:
        return ReconciliationOutcome.CORRUPT, "REFUSED", ("INCOMPLETE_REGISTRY_LINEAGE",)
    if retained.states[-1] != state:
        return (
            ReconciliationOutcome.INDETERMINATE,
            "REFUSED",
            ("REGISTRY_READS_CHANGED_WITHIN_OBSERVATION",),
        )
    auth = source.authorization
    if (
        state.subject != runtime.issuer_authority._entitlement_subject
        or state.identity.environment != auth.environment
        or state.identity.trust_domain != auth.trust_domain
        or state.identity.product_scope != auth.product_scope
        or state.identity.bootstrap_entitlement_id != auth.bootstrap_entitlement_id
        or history.stream != runtime.history._stream
        or checkpoint.stream != history.stream
        or history.stream.product_scope != auth.product_scope
    ):
        return (
            ReconciliationOutcome.CONFLICT,
            "REFUSED",
            ("FULL_AUTHORITY_SCOPE_OR_ENTITLEMENT_MISMATCH",),
        )
    if source.current.identity is None:
        return ReconciliationOutcome.INDETERMINATE, "REFUSED", ("ATTEMPT_NOT_IMMUTABLE",)
    if source.retained is None:
        return ReconciliationOutcome.CORRUPT, "REFUSED", ("IMMUTABLE_REQUEST_MISSING",)
    if not _verify_retained_signatures(sample):
        return (
            ReconciliationOutcome.CORRUPT,
            "REFUSED",
            ("RETAINED_PUBLIC_CREDENTIAL_OR_SIGNATURE_MISMATCH",),
        )
    if not _authorization_anchor_matches(state, auth):
        superseded = tuple(
            item
            for item in retained.states
            if item.subject == state.subject
            and item.lifecycle is EntitlementLifecycle.SUPERSEDED
            and _authorization_anchor_matches(item, auth)
        )
        if state.identity.entitlement_generation > auth.entitlement_generation and superseded:
            return (
                ReconciliationOutcome.STALE,
                "REFUSED",
                ("ENTITLEMENT_GENERATION_SUPERSEDED",),
            )
        return (
            ReconciliationOutcome.CONFLICT,
            "REFUSED",
            ("FULL_AUTHORITY_SCOPE_OR_ENTITLEMENT_MISMATCH",),
        )
    binding = state.binding
    if (
        source.current.state is AttemptState.EXACT_BOUND_RECOVERED
        and type(binding) is UnboundBinding
    ):
        return (
            ReconciliationOutcome.CONFLICT,
            "REFUSED",
            ("CHA_RECOVERED_BOUND_REGISTRY_CONFLICT",),
        )
    if type(binding) is BoundBinding:
        expected_fields = {
            name: getattr(auth, name)
            for name in (
                "logical_operation_id",
                "account_id",
                "canonical_genesis_request_fingerprint_sha256",
                "entitlement_generation",
                "requester_principal_id",
                "requester_key_id",
                "requester_key_version",
                "provisioning_principal_id",
                "claimant_key_id",
                "claimant_key_version",
            )
        }
        expected_fields.update(
            issuance_attempt_id=source.retained.attempt_id,
            signed_request_payload_digest_sha256=source.retained.digest,
            signed_request_canonical_bytes_reference=source.retained.reference,
        )
        if any(getattr(binding, name) != value for name, value in expected_fields.items()):
            return (
                ReconciliationOutcome.CONFLICT,
                "REFUSED",
                ("FULL_BOUND_ATTEMPT_IDENTITY_MISMATCH",),
            )
    if state.lifecycle is not EntitlementLifecycle.ACTIVE:
        return ReconciliationOutcome.STALE, "REFUSED", ("ENTITLEMENT_LIFECYCLE_NOT_ACTIVE",)
    # Globalny retained tail porównujemy także przed pre-issuance observation.
    tail = history.records[-1] if history.records else None
    current = checkpoint.current
    if tail is not None:
        if current is None:
            return ReconciliationOutcome.NOT_FOUND, "REFUSED", ("EXACT_CHECKPOINT_NOT_FOUND",)
        if current.history_sequence < tail.sequence:
            return ReconciliationOutcome.STALE, "REFUSED", ("HISTORY_AHEAD_OF_CHECKPOINT",)
        if current.history_sequence > tail.sequence:
            return ReconciliationOutcome.CORRUPT, "REFUSED", ("CHECKPOINT_AHEAD_OF_HISTORY",)
        if current.authenticated_head_digest != tail.authenticated_digest:
            return (
                ReconciliationOutcome.CONFLICT,
                "REFUSED",
                ("CHECKPOINT_CURRENT_DIGEST_MISMATCH",),
            )
        tail_head = next((head for head in history.heads if head.sequence == tail.sequence), None)
        if tail_head is None:
            return (
                ReconciliationOutcome.INDETERMINATE,
                "REFUSED",
                ("EXACT_HISTORY_TAIL_ATTESTATION_MISSING",),
            )
        if tail_head.signing_credential_identity != current.history_signing_credential_identity:
            return (
                ReconciliationOutcome.CONFLICT,
                "REFUSED",
                ("CHECKPOINT_CURRENT_SIGNER_MISMATCH",),
            )
        retained_tail_heads = tuple(
            head for head in checkpoint.heads if head.sequence == tail.sequence
        )
        if len(retained_tail_heads) != 1 or retained_tail_heads[0] != tail_head:
            return (
                ReconciliationOutcome.CONFLICT,
                "REFUSED",
                ("CHECKPOINT_CURRENT_ATTESTATION_MISMATCH",),
            )
    elif current is not None:
        return ReconciliationOutcome.CORRUPT, "REFUSED", ("CHECKPOINT_AHEAD_OF_HISTORY",)
    # Dowolny częściowo dopasowany operation/attempt/root-proof stanowi konflikt,
    # a nie nieobecność konkretnego zdarzenia.
    matching = tuple(
        record
        for record in history.records
        if (
            record.event_identity.logical_operation_id == auth.logical_operation_id
            or record.event_identity.issuance_attempt_id == source.retained.attempt_id
            or (
                type(binding) is BoundBinding
                and record.event_identity.root_proof_id == binding.root_proof_id
            )
        )
    )
    if not matching:
        if (
            type(binding) is UnboundBinding
            and source.current.state is AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        ):
            return (
                ReconciliationOutcome.NOT_FOUND,
                "PRE_ISSUANCE_OBSERVED",
                ("EXACT_ISSUANCE_HISTORY_NOT_FOUND",),
            )
        if source.current.state is AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN:
            return ReconciliationOutcome.INDETERMINATE, "REFUSED", ("CHA_SENT_OUTCOME_UNKNOWN",)
        return ReconciliationOutcome.NOT_FOUND, "REFUSED", ("EXACT_ISSUANCE_HISTORY_NOT_FOUND",)
    if len(matching) != 1 or type(binding) is not BoundBinding:
        return (
            ReconciliationOutcome.CONFLICT,
            "REFUSED",
            ("ISSUANCE_HISTORY_REGISTRY_BINDING_CONFLICT",),
        )
    record: HistoryRecord = matching[0]
    if (
        record.event_identity.logical_operation_id != auth.logical_operation_id
        or record.event_identity.issuance_attempt_id != source.retained.attempt_id
        or record.event_identity.root_proof_id != binding.root_proof_id
    ):
        return ReconciliationOutcome.CONFLICT, "REFUSED", ("HISTORY_EVENT_IDENTITY_MISMATCH",)
    # Pierwszy BIND tej generacji pozostaje niezmienny w complete registry lineage.
    bound_index = next(
        (
            index
            for index, item in enumerate(retained.states)
            if type(item.binding) is BoundBinding
            and item.identity.entitlement_generation == auth.entitlement_generation
            and item.binding == binding
        ),
        None,
    )
    if bound_index is None or bound_index == 0 or source.current.identity is None:
        return ReconciliationOutcome.CORRUPT, "REFUSED", ("BOUND_PREDECESSOR_CAS_MISSING",)
    bound_state, predecessor = retained.states[bound_index], retained.states[bound_index - 1]
    expected_payload = _expected_issuance_history_payload(
        auth, bound_state, predecessor, source.current.identity.digest_sha256
    )
    if _plain(record.canonical_event_payload) != _plain(expected_payload):
        return (
            ReconciliationOutcome.CONFLICT,
            "REFUSED",
            ("FULL_HISTORY_PAYLOAD_OR_PREDECESSOR_CAS_MISMATCH",),
        )
    head = next((item for item in history.heads if item.sequence == record.sequence), None)
    if head is None:
        return (
            ReconciliationOutcome.INDETERMINATE,
            "REFUSED",
            ("EXACT_ISSUANCE_ATTESTATION_MISSING",),
        )
    current = checkpoint.current
    if current is None:
        return ReconciliationOutcome.NOT_FOUND, "REFUSED", ("EXACT_CHECKPOINT_NOT_FOUND",)
    tail = history.records[-1]
    if current.history_sequence < tail.sequence:
        return ReconciliationOutcome.STALE, "REFUSED", ("HISTORY_AHEAD_OF_CHECKPOINT",)
    if current.history_sequence > tail.sequence:
        return ReconciliationOutcome.CORRUPT, "REFUSED", ("CHECKPOINT_AHEAD_OF_HISTORY",)
    if current.authenticated_head_digest != tail.authenticated_digest:
        return ReconciliationOutcome.CONFLICT, "REFUSED", ("CHECKPOINT_CURRENT_DIGEST_MISMATCH",)
    accepted = tuple(
        item for item in checkpoint.acceptances if item.accepted_history_sequence == record.sequence
    )
    accepted_heads = tuple(item for item in checkpoint.heads if item.sequence == record.sequence)
    if (
        len(accepted) != 1
        or len(accepted_heads) != 1
        or accepted[0].accepted_authenticated_head_digest != record.authenticated_digest
        or accepted[0].history_signing_credential_identity != head.signing_credential_identity
        or accepted_heads[0] != head
    ):
        return ReconciliationOutcome.CONFLICT, "REFUSED", ("EXACT_RETAINED_V2_ACCEPTANCE_MISMATCH",)
    return ReconciliationOutcome.EXACT_COMMITTED, "ALIGNED_COMMITTED_OBSERVED", ()


def reconcile_root_proof_issuance_attempt(binding: object) -> RootProofIssuanceReconciliationReport:
    """Porównaj genuine InitialBinding z wybranymi autorytetami, bez mutacji.

    Brak danych i każdy wynik odmowy pozostają bez uprawnień. Nawet zgodne dwa
    odczyty nie autoryzują emisji, podpisu, BIND, transportu ani checkpoint CAS.
    """
    try:
        _initial_binding_snapshot(binding)
    except Exception as error:
        failure = _failure("INITIAL_BINDING", error)
        return RootProofIssuanceReconciliationReport(
            failure.outcome, "REFUSED", (), (failure.reason,)
        )
    try:
        runtime = _configured_reconciliation_runtime()
        _validate_runtime(runtime)
    except Exception:
        return RootProofIssuanceReconciliationReport(
            ReconciliationOutcome.UNAVAILABLE,
            "REFUSED",
            (),
            ("PINNED_RECONCILIATION_AUTHORITIES_UNAVAILABLE",),
        )
    return _reconcile_selected_authorities(binding, runtime)


def _reconcile_selected_authorities(
    binding: object, runtime: _ReconciliationRuntime
) -> RootProofIssuanceReconciliationReport:
    """Wewnętrzny silnik po genuine guard i wyborze composition root.

    To nie jest alternatywne publiczne wejście dla DTO lub caller-selected
    portów. Publiczny wrapper zawsze weryfikuje InitialBinding przed odczytem.
    """
    observations: tuple[ReconciliationObservation, ...] = ()
    try:
        first = _capture(binding, runtime)
        observations = (first.observation,)
        _cut("FIRST_OBSERVATION_READ")
        selected = _configured_reconciliation_runtime()
        _validate_runtime(selected)
        repeated = _capture(binding, selected)
        observations = (first.observation, repeated.observation)
        if (
            selected.issuer_authority is not runtime.issuer_authority
            or selected.history is not runtime.history
            or selected.checkpoint is not runtime.checkpoint
            or selected.verification_authorities is not runtime.verification_authorities
            or first.observation != repeated.observation
        ):
            return RootProofIssuanceReconciliationReport(
                ReconciliationOutcome.INDETERMINATE,
                "REFUSED",
                observations,
                ("AUTHORITATIVE_OBSERVATIONS_CHANGED",)
                + tuple(
                    "CHANGED:" + field.name
                    for field in fields(ReconciliationObservation)
                    if getattr(first.observation, field.name)
                    != getattr(repeated.observation, field.name)
                ),
            )
        outcome, disposition, reasons = _classify(repeated, selected)
        return RootProofIssuanceReconciliationReport(
            outcome,
            cast(
                Literal["PRE_ISSUANCE_OBSERVED", "ALIGNED_COMMITTED_OBSERVED", "REFUSED"],
                disposition,
            ),
            observations,
            reasons,
        )
    except Exception as error:
        failure = _failure("RECONCILIATION", error)
        if len(observations) == 1:
            observations += (ReconciliationObservation(read_failures=(failure,)),)
            return RootProofIssuanceReconciliationReport(
                ReconciliationOutcome.INDETERMINATE,
                "REFUSED",
                observations,
                ("REPEATED_AUTHORITATIVE_OBSERVATION_UNAVAILABLE", failure.reason),
            )
        return RootProofIssuanceReconciliationReport(
            failure.outcome, "REFUSED", observations, (failure.reason,)
        )
