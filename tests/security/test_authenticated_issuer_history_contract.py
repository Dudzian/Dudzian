from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import MappingProxyType

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import bot_core.authenticated_issuer_history as history_contract
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoryContractError,
    HistoricalRevokedSignatureVerificationUnavailable,
    HistoryEventIdentity,
    HistoryRecord,
    HistoryStreamIdentity,
    LocalCheckpoint,
    LocalCheckpointProvider,
    ReconciliationOutcome,
    ReferenceAuthenticatedHistory,
    VerifiedHistoryHead,
    attest_head,
    build_record,
    canonical_json_bytes,
    reconcile_checkpoint,
    verify_head,
)
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)


class Signer:
    def __init__(
        self,
        *,
        role=CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
        provider_role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
        credential_id="history-credential",
        version="7",
        key=None,
        advertised_key=None,
        material_identity=None,
        lifecycle=SigningKeyLifecycle.ACTIVE,
        generation=1,
        provider_namespace="history-provider",
        lifecycle_namespace="lifecycle",
    ):
        self.key = key or Ed25519PrivateKey.generate()
        self.role = role
        self.provider_role = provider_role
        self.credential_id = credential_id
        self.version = version
        self.advertised_key = advertised_key or self.public()
        self.material_identity = material_identity or public_key_material_identity(
            self.advertised_key
        )
        self.lifecycle = lifecycle
        self.generation = generation
        self.provider_namespace = provider_namespace
        self.lifecycle_namespace = lifecycle_namespace

    @property
    def identity(self):
        return ProviderIdentity(
            self.provider_role,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "tenant-a"),
            self.provider_namespace,
        )

    @property
    def capabilities(self):
        return None

    def credential_identities(self):
        return (
            CredentialRoleIdentity(
                self.role,
                self.credential_id,
                self.provider_namespace,
                self.version,
                self.lifecycle_namespace,
                self.material_identity,
            ),
        )

    def active_credential_identity(self):
        if self.lifecycle is not SigningKeyLifecycle.ACTIVE:
            raise RuntimeError("not ACTIVE")
        return self.credential_identities()[0]

    def lifecycle_generation(self):
        return self.generation

    def sign_history_head(self, payload):
        if self.lifecycle is not SigningKeyLifecycle.ACTIVE:
            raise RuntimeError("not ACTIVE")
        return self.key.sign(payload)

    def public_key(self, credential_identity):
        if credential_identity != self.credential_id:
            raise RuntimeError("unknown credential")
        return self.advertised_key

    def lifecycle_state(self):
        return self.lifecycle

    def public(self):
        return self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )


@pytest.fixture
def stream():
    return HistoryStreamIdentity(
        "account-genesis-history",
        "IndependentAccountGenesisRootProofIssuer",
        "PRODUCTION_LOCAL",
        "production",
        "tenant-a",
        "product-a",
        1,
    )


def identity(n=1):
    return HistoryEventIdentity(f"operation-{n}", f"attempt-{n}", f"proof-{n}")


def committed(stream, count=1):
    history = ReferenceAuthenticatedHistory(stream)
    records = []
    expected = NO_PREDECESSOR
    for n in range(1, count + 1):
        record, replay = history.append(
            expected_digest=expected,
            event_identity=identity(n),
            payload={"account_id": "ą", "generation": n, "enabled": True, "optional": None},
        )
        assert not replay
        records.append(record)
        expected = record.authenticated_digest
    return history, records


def signed_history(stream, count=1, signer=None):
    signer = signer or Signer()
    history, records = committed(stream, count)
    head = attest_head(records[-1], signer)
    evidence = history.verify_attested_head(head, signer)
    return history, records, signer, head, evidence


def reconcile(history, head, signer, checkpoint_authority):
    return reconcile_checkpoint(
        history=history,
        history_head=head,
        verification_authority=signer,
        checkpoint_authority=checkpoint_authority,
    )


def checkpoint_authority(stream, history, head, signer):
    authority = LocalCheckpointProvider("cp", stream)
    authority.advance(
        expected_revision=0,
        history=history,
        head=head,
        verification_authority=signer,
    )
    return authority


def test_canonicalization_is_order_independent_and_exact_typed():
    assert canonical_json_bytes({"b": 1, "a": "ą"}) == canonical_json_bytes({"a": "ą", "b": 1})
    for invalid in ({"bad": 1.0}, {"bad": {1: "not text"}}, {"bad": bytearray(b"x")}):
        with pytest.raises(TypeError):
            canonical_json_bytes(invalid)


def test_build_record_rejects_mapping_subclass_and_snapshots_nested_payload(stream):
    class HostileDict(dict):
        pass

    with pytest.raises(TypeError, match="exact canonical JSON dict"):
        build_record(stream, 1, NO_PREDECESSOR, identity(), HostileDict(x=1))
    nested = {"items": [{"value": 1}]}
    record = build_record(stream, 1, NO_PREDECESSOR, identity(), nested)
    nested["items"][0]["value"] = 2
    assert record.canonical_event_payload["items"][0]["value"] == 1


def test_models_reject_subclasses_bool_int_and_bytes_subclasses(stream):
    class StreamSubclass(HistoryStreamIdentity):
        pass

    subclass = StreamSubclass(*stream.material().values())
    with pytest.raises(TypeError):
        build_record(subclass, 1, NO_PREDECESSOR, identity(), {})
    with pytest.raises(TypeError):
        build_record(stream, True, NO_PREDECESSOR, identity(), {})

    class BytesSubclass(bytes):
        pass

    with pytest.raises(TypeError):
        AttestedHistoryHead(
            stream, 1, "d", Signer().credential_identities()[0], BytesSubclass(b"x"), b"s"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "test"),
        ("trust_domain", "tenant-b"),
        ("product_scope", "product-b"),
        ("stream_id", "other-history"),
        ("security_epoch", 2),
    ],
)
def test_cross_boundary_head_replay_fails(stream, field, value):
    _, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    with pytest.raises(HistoryContractError, match="wrong stream"):
        verify_head(head, signer, expected_stream=replace(stream, **{field: value}))


def test_exact_append_replay_and_lost_response(stream):
    history, records = committed(stream)
    replay, was_replay = history.append(
        expected_digest="ignored-on-replay",
        event_identity=identity(),
        payload={"optional": None, "enabled": True, "generation": 1, "account_id": "ą"},
    )
    assert was_replay and replay == records[0]


def test_conflicting_append_replay(stream):
    history, records = committed(stream)
    with pytest.raises(HistoryContractError, match="conflicting"):
        history.append(
            expected_digest=records[0].authenticated_digest,
            event_identity=identity(),
            payload={"generation": 2},
        )


def test_stale_predecessor_and_duplicate_successor_fail_closed(stream):
    history, _ = committed(stream)
    with pytest.raises(HistoryContractError, match="stale predecessor"):
        history.append(expected_digest=NO_PREDECESSOR, event_identity=identity(2), payload={"x": 1})


def test_payload_tamper_detected(stream):
    history, records = committed(stream, 2)
    object.__setattr__(history._records[0], "event_digest", "sha256:" + "0" * 64)
    with pytest.raises(HistoryContractError):
        history.verify()


def test_predecessor_digest_tamper_detected(stream):
    history, records = committed(stream, 2)
    object.__setattr__(
        history._records[1], "predecessor_authenticated_digest", "sha256:" + "1" * 64
    )
    with pytest.raises(HistoryContractError):
        history.verify()


def test_sequence_gap_and_head_rewind_detected(stream):
    history, records = committed(stream, 2)
    object.__setattr__(history._records[1], "sequence", 4)
    with pytest.raises(HistoryContractError, match="gap, rewind"):
        history.verify()


def test_cross_stream_record_splice_detected(stream):
    history, records = committed(stream)
    other = replace(stream, trust_domain="tenant-b")
    object.__setattr__(history._records[0], "stream", other)
    with pytest.raises(HistoryContractError, match="splice"):
        history.verify()


def test_trusted_authority_verifies_signature(stream):
    _, _, signer, head, _ = signed_history(stream)
    verify_head(head, signer, expected_stream=stream)


def test_signature_tamper_fails(stream):
    _, _, signer, head, _ = signed_history(stream)
    with pytest.raises(HistoryContractError, match="signature"):
        verify_head(replace(head, signature=b"x" * 64), signer, expected_stream=stream)


def test_valid_signature_under_attacker_key_cannot_launder_trusted_identity(stream):
    _, records = committed(stream)
    trusted = Signer()
    legitimate = attest_head(records[0], trusted)
    attacker = Ed25519PrivateKey.generate()
    forged = replace(legitimate, signature=attacker.sign(legitimate.canonical_attestation_bytes))
    with pytest.raises(HistoryContractError, match="signature"):
        verify_head(forged, trusted, expected_stream=stream)
    attacker_public = attacker.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    with pytest.raises(HistoryContractError, match="authority"):
        verify_head(forged, attacker_public, expected_stream=stream)  # type: ignore[arg-type]


def test_key_material_identity_mismatch_fails_before_signature_acceptance(stream):
    _, records = committed(stream)
    key = Ed25519PrivateKey.generate()
    other = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    signer = Signer(key=key, material_identity=public_key_material_identity(other))
    head = attest_head(records[0], signer)
    with pytest.raises(HistoryContractError, match="material identity mismatch"):
        verify_head(head, signer, expected_stream=stream)


@pytest.mark.parametrize(
    "field,value", [("credential_identity", "wrong"), ("key_handle_or_version", "8")]
)
def test_wrong_declared_credential_or_version_with_correct_key_fails(stream, field, value):
    _, _, signer, head, _ = signed_history(stream)
    forged = replace(
        head,
        signing_credential_identity=replace(head.signing_credential_identity, **{field: value}),
    )
    with pytest.raises(HistoryContractError):
        verify_head(forged, signer, expected_stream=stream)


def test_root_proof_role_cannot_sign_or_verify_history(stream):
    _, records = committed(stream)
    root = Signer(
        role=CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        provider_role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    with pytest.raises(HistoryContractError, match="HISTORY"):
        attest_head(records[0], root)
    history_signer = Signer(key=root.key)
    head = attest_head(records[0], history_signer)
    with pytest.raises(HistoryContractError, match="authority"):
        verify_head(head, root, expected_stream=stream)


def test_cross_role_key_alias_is_rejected_by_history_authority_role(stream):
    _, records = committed(stream)
    shared = Ed25519PrivateKey.generate()
    history = Signer(key=shared)
    head = attest_head(records[0], history)
    root = Signer(
        key=shared,
        role=CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        provider_role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    with pytest.raises(HistoryContractError):
        verify_head(head, root, expected_stream=stream)


def test_verify_only_cannot_sign_but_retained_key_verifies(stream):
    _, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    verify_only = Signer(key=active.key, lifecycle=SigningKeyLifecycle.VERIFY_ONLY)
    with pytest.raises(RuntimeError, match="not ACTIVE"):
        attest_head(records[0], verify_only)
    verify_head(head, verify_only, expected_stream=stream)


def test_revoked_signer_cannot_create_new_head(stream):
    _, records = committed(stream)
    with pytest.raises(RuntimeError, match="not ACTIVE"):
        attest_head(records[0], Signer(lifecycle=SigningKeyLifecycle.REVOKED))


def test_active_verify_only_revoked_historical_verification_matrix(stream):
    _, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    verify_head(head, active, expected_stream=stream)
    verify_head(
        head,
        Signer(key=active.key, lifecycle=SigningKeyLifecycle.VERIFY_ONLY),
        expected_stream=stream,
    )
    with pytest.raises(HistoricalRevokedSignatureVerificationUnavailable):
        verify_head(
            head,
            Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED),
            expected_stream=stream,
        )


def test_checkpoint_reverifies_authority_and_rejects_forged_higher_head(stream):
    history, _, signer, head, _ = signed_history(stream)
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    checkpoints.advance(
        expected_revision=0,
        history=history,
        head=head,
        verification_authority=signer,
    )
    forged = replace(
        head,
        sequence=99,
        record_digest="sha256:" + "f" * 64,
        signing_credential_identity=replace(
            head.signing_credential_identity,
            credential_identity="arbitrary",
            key_handle_or_version="999",
        ),
    )
    with pytest.raises(HistoryContractError, match="current verified history"):
        checkpoints.advance(
            expected_revision=1,
            history=history,
            head=forged,
            verification_authority=signer,
        )


def test_copied_seal_and_recomputed_material_exploit_has_no_authority_path(stream):
    history, _, signer, valid_head, legitimate = signed_history(stream)
    forged_head = replace(
        valid_head,
        sequence=valid_head.sequence + 100,
        record_digest="sha256:" + "f" * 64,
    )
    fake = object.__new__(VerifiedHistoryHead)
    object.__setattr__(fake, "head", forged_head)
    recomputed_material = (
        forged_head.stream,
        forged_head.sequence,
        forged_head.record_digest,
        forged_head.signing_credential_identity,
        forged_head.canonical_attestation_bytes,
        forged_head.signature,
    )
    # The fixed value has no transferable _bound_material or _seal slots.
    with pytest.raises(AttributeError):
        object.__setattr__(fake, "_bound_material", recomputed_material)
    with pytest.raises(AttributeError):
        object.__setattr__(fake, "_seal", getattr(legitimate, "_seal"))
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    with pytest.raises(TypeError, match="verified_head"):
        checkpoints.advance(expected_revision=0, verified_head=fake)  # type: ignore[call-arg]
    with pytest.raises(HistoryContractError):
        checkpoints.advance(
            expected_revision=0,
            history=history,
            head=forged_head,
            verification_authority=signer,
        )


def test_mutated_legitimate_evidence_is_not_an_authorization_capability(stream):
    history, _, signer, head, evidence = signed_history(stream)
    object.__setattr__(
        evidence,
        "head",
        replace(head, sequence=101, record_digest="sha256:" + "e" * 64),
    )
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    with pytest.raises(TypeError, match="verified_head"):
        checkpoints.advance(expected_revision=0, verified_head=evidence)  # type: ignore[call-arg]
    checkpoint, replay = checkpoints.advance(
        expected_revision=0,
        history=history,
        head=head,
        verification_authority=signer,
    )
    assert not replay and checkpoint.history_sequence == head.sequence


def test_checkpoint_replay_behind_ahead_rewind_and_split_brain(stream):
    history, records = committed(stream, 2)
    signer = Signer()
    h1, h2 = attest_head(records[0], signer), attest_head(records[1], signer)
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix._records, prefix._events = [records[0]], {records[0].event_identity: records[0]}
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    cp1, replay = checkpoints.advance(
        expected_revision=0,
        history=prefix,
        head=h1,
        verification_authority=signer,
    )
    assert not replay and reconcile(history, h2, signer, checkpoints) is ReconciliationOutcome.STALE
    cp1_again, replay = checkpoints.advance(
        expected_revision=1,
        history=prefix,
        head=h1,
        verification_authority=signer,
    )
    assert replay and cp1_again == cp1
    cp2, _ = checkpoints.advance(
        expected_revision=1,
        history=history,
        head=h2,
        verification_authority=signer,
    )
    assert reconcile(prefix, h1, signer, checkpoints) is ReconciliationOutcome.CORRUPT
    with pytest.raises(HistoryContractError, match="rewind"):
        checkpoints.advance(
            expected_revision=2,
            history=prefix,
            head=h1,
            verification_authority=signer,
        )


def checkpoint_for(head, stream, **changes):
    values = dict(
        checkpoint_id="cp",
        stream=stream,
        history_sequence=head.sequence,
        authenticated_head_digest=head.record_digest,
        history_signing_credential_identity=head.signing_credential_identity,
        checkpoint_revision=1,
    )
    values.update(changes)
    return LocalCheckpoint(**values)


def test_fabricated_checkpoint_cannot_authorize_exact_committed(stream):
    history, _, signer, head, _ = signed_history(stream)
    fabricated = checkpoint_for(head, stream)
    authority = LocalCheckpointProvider("cp", stream)
    assert authority.current_checkpoint() is None
    assert reconcile(history, head, signer, authority) is ReconciliationOutcome.STALE
    with pytest.raises(TypeError):
        reconcile_checkpoint(
            history=history,
            history_head=head,
            verification_authority=signer,
            checkpoint=fabricated,  # type: ignore[call-arg]
        )


def test_raw_forged_head_cannot_produce_exact_or_stale(stream):
    history, _, signer, head, _ = signed_history(stream)
    authority = checkpoint_authority(stream, history, head, signer)
    forged_exact = replace(head, canonical_attestation_bytes=b"forged", signature=b"forged")
    assert reconcile(history, forged_exact, signer, authority) is ReconciliationOutcome.CORRUPT
    forged_higher = replace(
        head,
        sequence=head.sequence + 1,
        record_digest="sha256:" + "a" * 64,
        canonical_attestation_bytes=b"forged",
        signature=b"forged",
    )
    assert reconcile(history, forged_higher, signer, authority) is ReconciliationOutcome.CORRUPT


def test_authority_paths_reject_fabricated_subclasses(stream):
    history, _, signer, head, _ = signed_history(stream)

    class HeadSubclass(AttestedHistoryHead):
        pass

    class CheckpointProviderSubclass(LocalCheckpointProvider):
        pass

    class HistorySubclass(ReferenceAuthenticatedHistory):
        pass

    class StreamSubclass(HistoryStreamIdentity):
        pass

    subclass_head = HeadSubclass(
        head.stream,
        head.sequence,
        head.record_digest,
        head.signing_credential_identity,
        head.canonical_attestation_bytes,
        head.signature,
    )
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    with pytest.raises((TypeError, HistoryContractError)):
        checkpoints.advance(
            expected_revision=0,
            history=history,
            head=subclass_head,
            verification_authority=signer,
        )
    with pytest.raises(TypeError):
        reconcile(history, head, signer, CheckpointProviderSubclass("cp", stream))
    with pytest.raises(TypeError):
        reconcile_checkpoint(
            history=HistorySubclass(stream),
            history_head=None,
            verification_authority=None,
            checkpoint_authority=checkpoints,
        )
    subclass_stream = StreamSubclass(*stream.material().values())
    with pytest.raises(TypeError):
        LocalCheckpoint("cp", subclass_stream, 1, "digest", signer.credential_identities()[0], 1)


def test_reconcile_exact_uses_authoritative_committed_checkpoint(stream):
    history, _, signer, head, _ = signed_history(stream)
    authority = checkpoint_authority(stream, history, head, signer)
    assert reconcile(history, head, signer, authority) is ReconciliationOutcome.EXACT_COMMITTED


@pytest.mark.parametrize("difference", ["provider", "material", "lifecycle"])
def test_same_local_id_and_version_different_exact_identity_is_not_committed(
    stream,
    difference,
):
    history, records = committed(stream)
    signer_a = Signer(
        credential_id="same-local-id",
        version="same-local-version",
        provider_namespace="provider-A",
        lifecycle_namespace="lifecycle-A",
    )
    head_a = attest_head(records[0], signer_a)
    checkpoints = checkpoint_authority(stream, history, head_a, signer_a)
    options = {
        "key": signer_a.key,
        "credential_id": "same-local-id",
        "version": "same-local-version",
        "provider_namespace": "provider-A",
        "lifecycle_namespace": "lifecycle-A",
    }
    if difference == "provider":
        options["provider_namespace"] = "provider-B"
    elif difference == "material":
        options["key"] = Ed25519PrivateKey.generate()
    else:
        options["lifecycle_namespace"] = "lifecycle-B"
    signer_b = Signer(**options)
    head_b = attest_head(records[0], signer_b)

    assert reconcile(history, head_b, signer_b, checkpoints) is ReconciliationOutcome.CONFLICT


def test_revoked_acceptance_for_a_cannot_authorize_namespaced_replacement_b(stream):
    history, records = committed(stream)
    signer_a = Signer(
        credential_id="same-local-id",
        version="same-local-version",
        provider_namespace="provider-A",
        lifecycle_namespace="lifecycle-A",
    )
    head_a = attest_head(records[0], signer_a)
    checkpoints = checkpoint_authority(stream, history, head_a, signer_a)
    signer_b = Signer(
        credential_id="same-local-id",
        version="same-local-version",
        provider_namespace="provider-B",
        lifecycle_namespace="lifecycle-B",
    )
    head_b = attest_head(records[0], signer_b)
    revoked_b = Signer(
        key=signer_b.key,
        credential_id="same-local-id",
        version="same-local-version",
        provider_namespace="provider-B",
        lifecycle_namespace="lifecycle-B",
        lifecycle=SigningKeyLifecycle.REVOKED,
        generation=2,
    )

    with pytest.raises(HistoricalRevokedSignatureVerificationUnavailable):
        history.verify_attested_historical_head(head_b, revoked_b, checkpoints)


def test_v1_weak_head_material_is_rejected(stream):
    history, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    weak = {
        "stream": stream.material(),
        "sequence": head.sequence,
        "record_digest": head.record_digest,
        "signing_role": "HISTORY_ATTESTATION_SIGNING",
        "signing_credential_id": signer.credential_id,
        "signing_key_version": signer.version,
    }
    canonical = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryHead/v1\0" + canonical_json_bytes(
        weak
    )
    old_head = replace(
        head,
        canonical_attestation_bytes=canonical,
        signature=signer.key.sign(canonical),
    )
    with pytest.raises(HistoryContractError, match="non-canonical"):
        history.verify_attested_head(old_head, signer)


def test_checkpoint_behind_is_stale_for_verified_history(stream):
    history, records = committed(stream, 2)
    signer = Signer()
    h1, h2 = attest_head(records[0], signer), attest_head(records[1], signer)
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(1),
        payload={"account_id": "ą", "generation": 1, "enabled": True, "optional": None},
    )
    authority = checkpoint_authority(stream, prefix, h1, signer)
    assert reconcile(history, h2, signer, authority) is ReconciliationOutcome.STALE


def test_absence_outcomes_do_not_conflate_corruption(stream):
    history, _, signer, head, _ = signed_history(stream)
    absent = LocalCheckpointProvider("cp", stream)
    assert reconcile(history, head, signer, absent) is ReconciliationOutcome.STALE
    empty = ReferenceAuthenticatedHistory(stream)
    assert reconcile(empty, None, None, absent) is ReconciliationOutcome.NOT_FOUND
    committed_authority = checkpoint_authority(stream, history, head, signer)
    assert reconcile(empty, None, None, committed_authority) is ReconciliationOutcome.CORRUPT


def test_restart_identity_and_verified_chain_are_value_based(stream):
    _, records, signer, head, _ = signed_history(stream)
    restarted = ReferenceAuthenticatedHistory(replace(stream))
    restarted._records = list(records)
    restarted._events = {records[0].event_identity: records[0]}
    restarted.verify_attested_head(head, signer)


def _run_race(*operations):
    barrier = Barrier(len(operations))

    def run(operation):
        barrier.wait()
        try:
            return ("ok", operation())
        except Exception as exc:  # captured for exact concurrent outcome assertions
            return ("error", exc)

    with ThreadPoolExecutor(max_workers=len(operations)) as executor:
        return tuple(executor.map(run, operations))


def test_concurrent_append_append_has_exactly_one_cas_winner(stream):
    history = ReferenceAuthenticatedHistory(stream)
    results = _run_race(
        lambda: history.append(
            expected_digest=NO_PREDECESSOR, event_identity=identity(1), payload={"winner": "a"}
        ),
        lambda: history.append(
            expected_digest=NO_PREDECESSOR, event_identity=identity(2), payload={"winner": "b"}
        ),
    )
    assert [status for status, _ in results].count("ok") == 1
    error = next(value for status, value in results if status == "error")
    assert isinstance(error, HistoryContractError) and "stale predecessor" in str(error)
    assert len(history._records) == 1
    history.verify()


def test_concurrent_same_event_exact_retry_inserts_once(stream):
    history = ReferenceAuthenticatedHistory(stream)
    operation = lambda: history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(),
        payload={"same": True},
    )
    results = _run_race(operation, operation)
    assert all(status == "ok" for status, _ in results)
    values = [value for _, value in results]
    assert sorted(replay for _, replay in values) == [False, True]
    assert values[0][0] == values[1][0]
    assert values[0][0] is not values[1][0]
    assert len(history._records) == 1


def test_concurrent_same_event_conflicting_payload_has_one_winner(stream):
    history = ReferenceAuthenticatedHistory(stream)
    results = _run_race(
        lambda: history.append(
            expected_digest=NO_PREDECESSOR, event_identity=identity(), payload={"value": 1}
        ),
        lambda: history.append(
            expected_digest=NO_PREDECESSOR, event_identity=identity(), payload={"value": 2}
        ),
    )
    assert [status for status, _ in results].count("ok") == 1
    error = next(value for status, value in results if status == "error")
    assert isinstance(error, HistoryContractError) and "conflicting" in str(error)
    assert len(history._records) == 1 == len(history._events)


def test_concurrent_exact_retry_and_new_append_preserve_identity(stream):
    history, records = committed(stream)
    results = _run_race(
        lambda: history.append(
            expected_digest="ignored",
            event_identity=identity(),
            payload={"account_id": "ą", "generation": 1, "enabled": True, "optional": None},
        ),
        lambda: history.append(
            expected_digest=records[0].authenticated_digest,
            event_identity=identity(2),
            payload={"new": True},
        ),
    )
    assert all(status == "ok" for status, _ in results)
    retry, successor = (value for _, value in results)
    assert retry == (records[0], True)
    assert successor[0].event_identity == identity(2) and successor[1] is False
    assert len(history._records) == 2 == len(history._events)


class BlockingSigner(Signer):
    def __init__(self, entered, release, **kwargs):
        super().__init__(**kwargs)
        self.entered = entered
        self.release = release

    def public_key(self, credential_identity):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().public_key(credential_identity)


def test_verify_during_append_uses_one_coherent_history_snapshot(stream):
    base = Signer()
    history, records = committed(stream)
    head = attest_head(records[0], base)
    entered, release = Event(), Event()
    blocking = BlockingSigner(entered, release, key=base.key)
    with ThreadPoolExecutor(max_workers=2) as executor:
        verifying = executor.submit(history.verify_attested_head, head, blocking)
        assert entered.wait(timeout=5)
        appending = executor.submit(
            history.append,
            expected_digest=records[0].authenticated_digest,
            event_identity=identity(2),
            payload={"new": True},
        )
        assert not appending.done()
        release.set()
        assert verifying.result(timeout=5).head == head
        appended, replay = appending.result(timeout=5)
    assert not replay and appended.sequence == 2 and len(history._records) == 2


def _candidate(stream, signer, number):
    history = ReferenceAuthenticatedHistory(stream)
    record, _ = history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(number),
        payload={"candidate": number},
    )
    return history, attest_head(record, signer)


def test_concurrent_checkpoint_same_revision_has_one_winner(stream):
    verification_barrier = Barrier(2)

    class BarrierSigner(Signer):
        def public_key(self, credential_identity):
            verification_barrier.wait(timeout=5)
            return super().public_key(credential_identity)

    signer = BarrierSigner()
    history_a, head_a = _candidate(stream, signer, 1)
    history_b, head_b = _candidate(stream, signer, 2)
    checkpoints = LocalCheckpointProvider("cp", stream)
    results = _run_race(
        lambda: checkpoints.advance(
            expected_revision=0, history=history_a, head=head_a, verification_authority=signer
        ),
        lambda: checkpoints.advance(
            expected_revision=0, history=history_b, head=head_b, verification_authority=signer
        ),
    )
    assert [status for status, _ in results].count("ok") == 1
    error = next(value for status, value in results if status == "error")
    assert isinstance(error, HistoryContractError) and "CAS conflict" in str(error)
    assert checkpoints.current_checkpoint().checkpoint_revision == 1


def test_concurrent_checkpoint_exact_replay_has_one_mutation(stream):
    history, _, signer, head, _ = signed_history(stream)
    checkpoints = LocalCheckpointProvider("cp", stream)
    results = _run_race(
        lambda: checkpoints.advance(
            expected_revision=0, history=history, head=head, verification_authority=signer
        ),
        lambda: checkpoints.advance(
            expected_revision=0, history=history, head=head, verification_authority=signer
        ),
    )
    assert [status for status, _ in results].count("ok") == 1
    assert [status for status, _ in results].count("error") == 1
    assert checkpoints.current_checkpoint().checkpoint_revision == 1


def test_checkpoint_head_mutation_toctou_uses_verified_snapshot(stream):
    base = Signer()
    history, records = committed(stream)
    head = attest_head(records[0], base)
    entered, release = Event(), Event()
    blocking = BlockingSigner(entered, release, key=base.key)
    checkpoints = LocalCheckpointProvider("cp", stream)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            checkpoints.advance,
            expected_revision=0,
            history=history,
            head=head,
            verification_authority=blocking,
        )
        assert entered.wait(timeout=5)
        object.__setattr__(head, "sequence", 999)
        object.__setattr__(head, "record_digest", "sha256:" + "f" * 64)
        release.set()
        checkpoint, replay = future.result(timeout=5)
    assert not replay
    assert checkpoint.history_sequence == 1
    assert checkpoint.authenticated_head_digest == records[0].authenticated_digest


def test_reconciliation_head_mutation_toctou_uses_verified_snapshot(stream):
    base = Signer()
    history, records = committed(stream)
    head = attest_head(records[0], base)
    checkpoints = checkpoint_authority(stream, history, head, base)
    entered, release = Event(), Event()
    blocking = BlockingSigner(entered, release, key=base.key)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            reconcile_checkpoint,
            history=history,
            history_head=head,
            verification_authority=blocking,
            checkpoint_authority=checkpoints,
        )
        assert entered.wait(timeout=5)
        object.__setattr__(head, "sequence", 999)
        object.__setattr__(head, "record_digest", "sha256:" + "f" * 64)
        release.set()
        outcome = future.result(timeout=5)
    assert outcome is ReconciliationOutcome.EXACT_COMMITTED


def test_reconciliation_authority_unavailability_is_not_corruption(stream):
    history, _, signer, head, _ = signed_history(stream)
    checkpoints = checkpoint_authority(stream, history, head, signer)

    class UnavailableKeySigner(Signer):
        def public_key(self, credential_identity):
            raise RuntimeError("provider unavailable")

    unavailable_key = UnavailableKeySigner(key=signer.key)
    assert (
        reconcile(history, head, unavailable_key, checkpoints) is ReconciliationOutcome.UNAVAILABLE
    )

    class UnavailableLifecycleSigner(Signer):
        def lifecycle_state(self):
            raise RuntimeError("lifecycle unavailable")

    unavailable_lifecycle = UnavailableLifecycleSigner(key=signer.key)
    assert (
        reconcile(history, head, unavailable_lifecycle, checkpoints)
        is ReconciliationOutcome.UNAVAILABLE
    )


def test_returned_history_record_mutation_cannot_change_authority(stream):
    history = ReferenceAuthenticatedHistory(stream)
    record, _ = history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(),
        payload={"value": 1},
    )
    original_digest = record.authenticated_digest
    attacker_digest = "sha256:" + "f" * 64
    object.__setattr__(record, "authenticated_digest", attacker_digest)
    assert history.current_record().authenticated_digest == original_digest
    with pytest.raises(HistoryContractError, match="stale predecessor"):
        history.append(
            expected_digest=attacker_digest,
            event_identity=identity(2),
            payload={"value": 2},
        )


def test_current_record_is_detached(stream):
    history, records = committed(stream)
    public_head = history.current_record()
    object.__setattr__(public_head, "sequence", 999)
    assert history.current_record() == records[0]


def test_event_identity_is_snapshotted_on_admission(stream):
    history = ReferenceAuthenticatedHistory(stream)
    event = identity()
    record, _ = history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=event,
        payload={"value": 1},
    )
    object.__setattr__(event, "logical_operation_id", "attacker")
    assert history.current_record().event_identity.logical_operation_id == "operation-1"
    replay, was_replay = history.append(
        expected_digest="ignored",
        event_identity=identity(),
        payload={"value": 1},
    )
    assert was_replay and replay == record and len(history._records) == 1


def test_stream_identity_is_snapshotted_by_history_and_checkpoint(stream):
    history = ReferenceAuthenticatedHistory(stream)
    checkpoints = LocalCheckpointProvider("cp", stream)
    object.__setattr__(stream, "trust_domain", "attacker-domain")
    record, _ = history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(),
        payload={"value": 1},
    )
    assert record.stream.trust_domain == "tenant-a"
    assert checkpoints._stream.trust_domain == "tenant-a"


def test_internal_history_corruption_blocks_new_append(stream):
    history, records = committed(stream)
    object.__setattr__(history._records[-1], "authenticated_digest", "sha256:" + "f" * 64)
    with pytest.raises(HistoryContractError):
        history.append(
            expected_digest=records[0].authenticated_digest,
            event_identity=identity(2),
            payload={"value": 2},
        )


def test_returned_and_current_checkpoint_snapshots_are_detached(stream):
    history, _, signer, head, _ = signed_history(stream)
    checkpoints = LocalCheckpointProvider("cp", stream)
    returned, _ = checkpoints.advance(
        expected_revision=0,
        history=history,
        head=head,
        verification_authority=signer,
    )
    object.__setattr__(returned, "checkpoint_revision", 0)
    object.__setattr__(returned.stream, "trust_domain", "attacker")
    current = checkpoints.current_checkpoint()
    assert current.checkpoint_revision == 1
    assert current.stream.trust_domain == "tenant-a"
    object.__setattr__(current, "history_sequence", 999)
    assert checkpoints.current_checkpoint().history_sequence == 1


def test_returned_checkpoint_mutation_cannot_reuse_revision(stream):
    history, records = committed(stream, 2)
    signer = Signer()
    h1, h2 = attest_head(records[0], signer), attest_head(records[1], signer)
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(1),
        payload={"account_id": "ą", "generation": 1, "enabled": True, "optional": None},
    )
    checkpoints = LocalCheckpointProvider("cp", stream)
    returned, _ = checkpoints.advance(
        expected_revision=0,
        history=prefix,
        head=h1,
        verification_authority=signer,
    )
    object.__setattr__(returned, "checkpoint_revision", 0)
    with pytest.raises(HistoryContractError, match="CAS conflict"):
        checkpoints.advance(
            expected_revision=0,
            history=history,
            head=h2,
            verification_authority=signer,
        )
    assert checkpoints.current_checkpoint().checkpoint_revision == 1


def test_reconciliation_uses_one_detached_checkpoint_snapshot_during_advance(stream):
    history, records = committed(stream, 2)
    signer = Signer()
    h1, h2 = attest_head(records[0], signer), attest_head(records[1], signer)
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=identity(1),
        payload={"account_id": "ą", "generation": 1, "enabled": True, "optional": None},
    )
    checkpoints = checkpoint_authority(stream, prefix, h1, signer)
    results = _run_race(
        lambda: reconcile(history, h2, signer, checkpoints),
        lambda: checkpoints.advance(
            expected_revision=1,
            history=history,
            head=h2,
            verification_authority=signer,
        ),
    )
    assert all(status == "ok" for status, _ in results)
    assert results[0][1] in (ReconciliationOutcome.STALE, ReconciliationOutcome.EXACT_COMMITTED)
    assert checkpoints.current_checkpoint().history_sequence == 2


def test_revoked_evidence_exact_acceptance_enables_historical_verification(stream):
    history, records = committed(stream)
    active = Signer(generation=1)
    head = attest_head(records[0], active)
    checkpoints = checkpoint_authority(stream, history, head, active)
    revoked = Signer(
        key=active.key,
        credential_id=active.credential_id,
        lifecycle=SigningKeyLifecycle.REVOKED,
        generation=2,
    )
    assert history.verify_attested_historical_head(head, revoked, checkpoints).head == head


def test_revoked_evidence_missing_or_wrong_authority_is_unavailable(stream):
    history, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    revoked = Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED, generation=2)
    with pytest.raises(HistoricalRevokedSignatureVerificationUnavailable):
        history.verify_attested_historical_head(
            head,
            revoked,
            LocalCheckpointProvider("cp", stream),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted_history_sequence", 2),
        ("accepted_authenticated_head_digest", "sha256:" + "9" * 64),
        ("lifecycle_generation_at_acceptance", 9),
    ],
)
def test_revoked_evidence_wrong_retained_binding_fails_closed(stream, field, value):
    history, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    checkpoints = checkpoint_authority(stream, history, head, active)
    object.__setattr__(checkpoints._acceptances[0], field, value)
    revoked = Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED, generation=2)
    with pytest.raises(HistoryContractError):
        history.verify_attested_historical_head(head, revoked, checkpoints)


def test_revoked_evidence_detached_value_mutation_does_not_change_authority(stream):
    history, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    checkpoints = checkpoint_authority(stream, history, head, active)
    detached = checkpoints.historical_acceptance(head)
    object.__setattr__(detached, "accepted_history_sequence", 99)
    revoked = Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED, generation=2)
    history.verify_attested_historical_head(head, revoked, checkpoints)


def test_revoked_key_cannot_commit_post_revocation_acceptance(stream):
    history, records = committed(stream)
    active = Signer()
    head = attest_head(records[0], active)
    revoked = Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED, generation=2)
    checkpoints = LocalCheckpointProvider("cp", stream)
    with pytest.raises(HistoricalRevokedSignatureVerificationUnavailable):
        checkpoints.advance(
            expected_revision=0,
            history=history,
            head=head,
            verification_authority=revoked,
        )
    assert checkpoints.historical_acceptance(head) is None


def test_revoked_forged_post_revocation_head_is_not_self_corroborating(stream):
    history, records = committed(stream)
    active = Signer()
    accepted = attest_head(records[0], active)
    checkpoints = checkpoint_authority(stream, history, accepted, active)
    forged_record = replace(records[0], authenticated_digest=records[0].authenticated_digest)
    forged = attest_head(forged_record, active)
    forged = replace(forged, record_digest="sha256:" + "8" * 64)
    material = {
        "stream": stream.material(),
        "sequence": forged.sequence,
        "record_digest": forged.record_digest,
        "signing_credential_identity": history_contract._credential_material(
            forged.signing_credential_identity
        ),
    }
    canonical = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryHead/v1\0" + canonical_json_bytes(
        material
    )
    forged = replace(
        forged, canonical_attestation_bytes=canonical, signature=active.key.sign(canonical)
    )
    revoked = Signer(key=active.key, lifecycle=SigningKeyLifecycle.REVOKED, generation=2)
    with pytest.raises(HistoryContractError):
        history.verify_attested_historical_head(forged, revoked, checkpoints)


def test_revoked_evidence_survives_rotation_and_later_checkpoint_advance(stream):
    history, records = committed(stream, 2)
    signer_a = Signer(credential_id="credential-a", version="1")
    head_a = attest_head(records[0], signer_a)
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix._records = [records[0]]
    prefix._events = {records[0].event_identity: records[0]}
    checkpoints = checkpoint_authority(stream, prefix, head_a, signer_a)
    signer_b = Signer(credential_id="credential-b", version="2")
    head_b = attest_head(records[1], signer_b)
    checkpoints.advance(
        expected_revision=1,
        history=history,
        head=head_b,
        verification_authority=signer_b,
    )
    revoked_a = Signer(
        key=signer_a.key,
        credential_id="credential-a",
        version="1",
        lifecycle=SigningKeyLifecycle.REVOKED,
        generation=3,
    )
    history.verify_attested_historical_head(head_a, revoked_a, checkpoints)
    assert (
        checkpoints.current_checkpoint().history_signing_credential_identity.credential_identity
        == "credential-b"
    )


def test_concurrent_revocation_or_evidence_commit_never_accepts_revoked_state(stream):
    history, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    checkpoints = LocalCheckpointProvider("cp", stream)

    def revoke():
        signer.lifecycle = SigningKeyLifecycle.REVOKED
        signer.generation = 2

    results = _run_race(
        revoke,
        lambda: checkpoints.advance(
            expected_revision=0,
            history=history,
            head=head,
            verification_authority=signer,
        ),
    )
    acceptance = checkpoints.historical_acceptance(head)
    if acceptance is not None:
        assert acceptance.lifecycle_state_at_acceptance is SigningKeyLifecycle.ACTIVE
        assert acceptance.lifecycle_generation_at_acceptance == 1
    else:
        assert any(status == "error" for status, _ in results)


def _history_prefix(stream, records, count):
    prefix = ReferenceAuthenticatedHistory(stream)
    prefix._records = list(records[:count])
    prefix._events = {record.event_identity: record for record in records[:count]}
    return prefix


def _redigest_acceptance(evidence, **changes):
    candidate = object.__new__(type(evidence))
    for name in evidence.__dataclass_fields__:
        if name != "acceptance_digest":
            object.__setattr__(candidate, name, changes.get(name, getattr(evidence, name)))
    object.__setattr__(
        candidate,
        "acceptance_digest",
        history_contract._acceptance_digest(candidate),
    )
    return candidate


def _two_head_checkpoint_state(stream):
    history, records = committed(stream, 3)
    signer = Signer()
    heads = tuple(attest_head(record, signer) for record in records)
    checkpoints = LocalCheckpointProvider("cp", stream)
    checkpoint1, _ = checkpoints.advance(
        expected_revision=0,
        history=_history_prefix(stream, records, 1),
        head=heads[0],
        verification_authority=signer,
    )
    return history, records, signer, heads, checkpoints, checkpoint1


def _advance_second(history, records, signer, heads, checkpoints):
    return checkpoints.advance(
        expected_revision=1,
        history=_history_prefix(history._stream, records, 2),
        head=heads[1],
        verification_authority=signer,
    )


def test_corrupted_acceptance_blocks_next_advance_without_mutation(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    object.__setattr__(checkpoints._acceptances[0], "accepted_history_sequence", 999)
    with pytest.raises(HistoryContractError, match="acceptance"):
        _advance_second(history, records, signer, heads, checkpoints)
    assert checkpoints._current.checkpoint_revision == 1
    assert len(checkpoints._acceptances) == 1


def test_corrupt_acceptance_digest_blocks_next_advance(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    object.__setattr__(
        checkpoints._acceptances[-1],
        "acceptance_digest",
        "sha256:" + "f" * 64,
    )
    with pytest.raises(HistoryContractError, match="acceptance chain"):
        _advance_second(history, records, signer, heads, checkpoints)
    assert checkpoints._current.checkpoint_revision == len(checkpoints._acceptances) == 1


def test_missing_acceptance_with_current_checkpoint_blocks_advance(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    checkpoints._acceptances.clear()
    with pytest.raises(HistoryContractError, match="cardinality"):
        _advance_second(history, records, signer, heads, checkpoints)
    assert checkpoints._current.checkpoint_revision == 1
    assert checkpoints._acceptances == []


def test_current_checkpoint_rollback_with_newer_evidence_blocks_advance(stream):
    history, records, signer, heads, checkpoints, checkpoint1 = _two_head_checkpoint_state(stream)
    _advance_second(history, records, signer, heads, checkpoints)
    checkpoints._current = checkpoint1
    with pytest.raises(HistoryContractError, match="cardinality"):
        checkpoints.advance(
            expected_revision=1,
            history=history,
            head=heads[2],
            verification_authority=signer,
        )
    assert checkpoints._current.checkpoint_revision == 1
    assert len(checkpoints._acceptances) == 2


def test_extra_trailing_acceptance_blocks_next_authority_operation(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    original = checkpoints._acceptances[0]
    trailing = _redigest_acceptance(
        original,
        checkpoint_revision=2,
        accepted_history_sequence=2,
        predecessor_acceptance_digest=original.acceptance_digest,
    )
    checkpoints._acceptances.append(trailing)
    with pytest.raises(HistoryContractError, match="cardinality"):
        _advance_second(history, records, signer, heads, checkpoints)
    assert checkpoints._current.checkpoint_revision == 1
    assert len(checkpoints._acceptances) == 2


def test_exact_checkpoint_replay_rejects_corrupt_evidence(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    object.__setattr__(checkpoints._acceptances[0], "acceptance_digest", "broken")
    with pytest.raises(HistoryContractError, match="acceptance chain"):
        checkpoints.advance(
            expected_revision=1,
            history=_history_prefix(stream, records, 1),
            head=heads[0],
            verification_authority=signer,
        )


def test_current_checkpoint_last_evidence_mismatch_is_detected(stream):
    _, _, _, _, checkpoints, _ = _two_head_checkpoint_state(stream)
    evidence = checkpoints._acceptances[0]
    checkpoints._acceptances[0] = _redigest_acceptance(
        evidence,
        history_signing_credential_identity=replace(
            evidence.history_signing_credential_identity,
            credential_identity="different-credential",
        ),
    )
    with pytest.raises(HistoryContractError, match="binding"):
        checkpoints.current_checkpoint()


def test_reconciliation_maps_corrupt_checkpoint_authority_to_corrupt(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    object.__setattr__(checkpoints._acceptances[0], "acceptance_digest", "broken")
    assert reconcile(history, heads[2], signer, checkpoints) is ReconciliationOutcome.CORRUPT


def test_correctly_digested_verify_only_acceptance_is_semantically_rejected(stream):
    _, _, _, _, checkpoints, _ = _two_head_checkpoint_state(stream)
    evidence = checkpoints._acceptances[0]
    checkpoints._acceptances[0] = _redigest_acceptance(
        evidence,
        lifecycle_state_at_acceptance=SigningKeyLifecycle.VERIFY_ONLY,
    )
    with pytest.raises(HistoryContractError, match="chain corrupt"):
        checkpoints.current_checkpoint()


def test_correctly_digested_non_monotonic_acceptance_sequence_is_rejected(stream):
    history, records, signer, heads, checkpoints, _ = _two_head_checkpoint_state(stream)
    _advance_second(history, records, signer, heads, checkpoints)
    second = checkpoints._acceptances[1]
    checkpoints._acceptances[1] = _redigest_acceptance(
        second,
        accepted_history_sequence=1,
    )
    with pytest.raises(HistoryContractError, match="chain corrupt"):
        checkpoints.current_checkpoint()


def test_concurrent_advance_then_rollback_is_detected_before_next_advance(stream):
    history, records, signer, heads, checkpoints, checkpoint1 = _two_head_checkpoint_state(stream)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _advance_second,
            history,
            records,
            signer,
            heads,
            checkpoints,
        )
        assert future.result(timeout=5)[0].checkpoint_revision == 2
    checkpoints._current = checkpoint1
    with pytest.raises(HistoryContractError, match="cardinality"):
        checkpoints.advance(
            expected_revision=1,
            history=history,
            head=heads[2],
            verification_authority=signer,
        )
    assert len(checkpoints._acceptances) == 2
