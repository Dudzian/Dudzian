"""Fast diagnostic tests; mocked input ports do not issue genuine capabilities.

The retained reader and Ed25519 verification run unchanged. A test-only public
custody verifier substitutes the native metadata I/O; real PostgreSQL/genuine
binding coverage lives in the atomic boundary integration suite.
"""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from bot_core import (
    local_signing_custody as custody,
    postgresql_root_proof_retention as retention,
    postgresql_root_proof_signing_intent as intent,
    root_proof_atomic_attempt_consistency as implementation,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)
from tests.licensing.test_cha_attempt_store_stage9 import authorization
from tests.security.test_postgresql_root_proof_retention import _retained

ROOT_ID = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab"
OTHER_ROOT_ID = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ac"
Outcome = implementation.RootProofAtomicAttemptConsistencyOutcome


def _reject_mutation(*args, **kwargs):
    pytest.fail("read-only RESERVED/STAGED consistency invoked a mutation")


@pytest.fixture
def boundary(monkeypatch):
    auth = authorization(bootstrap_entitlement_id="ent_018f3e70-7b5a-7c21-8b9a-0123456789ab")
    request = _retained(auth, (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()))
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, auth.trust_domain)
    scope = retention.RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL,
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        "IndependentAccountGenesisRootProofIssuer",
    )
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    material = public_key_material_identity(public)
    snapshot = custody.SigningCredentialSnapshot(
        ProviderIdentity(ProviderRole.ROOT_PROOF_SIGNING, security, "TEST_ROOT_CUSTODY"),
        CredentialRoleIdentity(
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
            "TEST_ROOT_KEY",
            "TEST_ROOT_CUSTODY",
            "TEST_ROOT_HANDLE",
            "TEST_ROOT_LIFECYCLE",
            material,
        ),
        1,
        1,
        custody.SigningKeyLifecycle.ACTIVE,
        public,
        material,
    )
    signer = object.__new__(custody.LocalRootProofSigningProvider)
    signing_intent = object.__new__(intent.PostgreSQLRootProofSigningIntent)
    prepared_retention = object.__new__(retention.PostgreSQLRootProofRetention)
    object.__setattr__(signing_intent, "_PostgreSQLRootProofSigningIntent__signer", signer)
    object.__setattr__(signing_intent, "_PostgreSQLRootProofSigningIntent__scope", scope)
    object.__setattr__(prepared_retention, "_PostgreSQLRootProofRetention__scope", scope)
    object.__setattr__(
        prepared_retention, "_PostgreSQLRootProofRetention__verification_authority", signer
    )
    payload = intent._payload(scope, auth, request, snapshot, ROOT_ID)
    reserved = intent.RootProofSigningIntent(
        scope,
        auth,
        request,
        intent._envelope(scope, auth, request, snapshot, ROOT_ID, payload),
        payload,
        snapshot,
        ROOT_ID,
    )
    value = SimpleNamespace(
        auth=auth,
        request=request,
        scope=scope,
        snapshot=snapshot,
        private=private,
        signer=signer,
        signing_intent=signing_intent,
        retention=prepared_retention,
        reserved=reserved,
        row=None,
        binding=object(),
        signed_attempt=object(),
        reads=[],
        historical=[],
    )

    def inputs(self, binding, signed_attempt):
        assert self is signing_intent
        assert binding is value.binding and signed_attempt is value.signed_attempt
        return auth, request

    def read_intent(self, binding, signed_attempt):
        value.reads.append("RESERVED")
        return value.reserved

    @contextmanager
    def connect(self):
        yield SimpleNamespace(transaction=lambda: transaction())

    @contextmanager
    def transaction():
        yield

    def read_row(self, conn, retained_request):
        assert retained_request == request
        value.reads.append("STAGED")
        return value.row

    def historical(self, domain_payload, signature, original):
        assert self is signer and original == snapshot
        value.historical.append((domain_payload, signature, original))
        Ed25519PublicKey.from_public_bytes(original.public_key).verify(signature, domain_payload)
        return True

    monkeypatch.setattr(intent.PostgreSQLRootProofSigningIntent, "_inputs", inputs)
    monkeypatch.setattr(intent.PostgreSQLRootProofSigningIntent, "read_intent", read_intent)
    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "_connect", connect)
    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "_qualify", lambda *args: None)
    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "_read_row", read_row)
    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", historical)
    for target, method in (
        (intent.PostgreSQLRootProofSigningIntent, "reserve"),
        (retention.PostgreSQLRootProofRetention, "retain_prepared"),
        (custody.LocalRootProofSigningProvider, "sign_root_proof"),
        (custody._LocalSigningProviderBase, "_sign"),
        (custody._LocalSigningProviderBase, "_sign_result"),
    ):
        monkeypatch.setattr(target, method, _reject_mutation)
    return value


def _stage(value, *, root_id=ROOT_ID, snapshot=None):
    snapshot = value.snapshot if snapshot is None else snapshot
    payload = intent._payload(value.scope, value.auth, value.request, snapshot, root_id)
    signature = value.private.sign(retention.ROOT_PROOF_SIGNING_DOMAIN + payload)
    fields = retention._validate_payload(
        payload, signature, value.scope, value.auth, value.request, snapshot
    )
    envelope = retention._envelope(
        value.scope, value.auth, value.request, payload, signature, snapshot, fields
    )
    value.row = envelope, payload, signature


def _check(value):
    report = implementation.validate_root_proof_atomic_attempt_consistency(
        value.signing_intent,
        value.retention,
        binding=value.binding,
        signed_attempt=value.signed_attempt,
    )
    assert report.disposition == "NOT_READY"
    assert report.authorizes_operations is False
    assert report.root_proof_issuance_available is False
    assert report.atomic_global_snapshot is False
    assert "REGISTRY_BIND" in report.blocked_operations
    assert "ACCOUNT_GENESIS_MUTATION" in report.blocked_operations
    assert not hasattr(report, "signature")
    assert not hasattr(report, "canonical_payload")
    return report


def test_reserved_without_staged_is_explicitly_unsigned_and_not_issued(boundary):
    report = _check(boundary)
    assert report.outcome is Outcome.RESERVED_WITHOUT_STAGED
    assert report.root_proof_id == ROOT_ID
    assert boundary.reads == ["RESERVED", "STAGED"] * 2
    assert boundary.historical == []
    with pytest.raises(FrozenInstanceError):
        report.authorizes_operations = True


def test_exact_original_bytes_require_existing_cryptographic_and_historical_reader(boundary):
    _stage(boundary)
    before = boundary.row
    report = _check(boundary)
    assert report.outcome is Outcome.EXACT_MATCH
    assert report.root_proof_id == ROOT_ID
    assert report.observations[0] == report.observations[1]
    assert boundary.row == before
    assert len(boundary.historical) == 2
    assert boundary.historical[0][0] == boundary.reserved.bytes_to_sign


def test_valid_staged_other_root_is_identity_conflict_not_missing(boundary):
    _stage(boundary, root_id=OTHER_ROOT_ID)
    report = _check(boundary)
    assert report.outcome is Outcome.CONFLICT
    assert report.root_proof_id is None
    assert report.observations[0].staged_root_proof_id == OTHER_ROOT_ID
    assert len(boundary.historical) == 2


@pytest.mark.parametrize(
    "field,other",
    [
        ("logical_operation_id", "ago_018f3e70-7b5a-7c21-8b9a-0123456789ac"),
        ("account_id", "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"),
        ("bootstrap_entitlement_id", "ent_018f3e70-7b5a-7c21-8b9a-0123456789ac"),
        ("entitlement_generation", 2),
        ("canonical_genesis_request_fingerprint_sha256", "c" * 64),
        ("reservation_identity", "ibr_" + "c" * 64),
        ("initial_binding_sha256", "c" * 64),
        ("authorization_evidence_sha256", "c" * 64),
    ],
)
def test_retained_tuple_difference_cannot_replace_trusted_initial_binding(boundary, field, other):
    changed = replace(boundary.auth, **{field: other})
    boundary.reserved = replace(boundary.reserved, authorization=changed)
    report = _check(boundary)
    assert report.outcome is Outcome.CONFLICT
    assert report.root_proof_id is None


def test_different_signed_request_attempt_cannot_reinterpret_reserved(boundary):
    changed = _retained(
        boundary.auth,
        (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()),
        attempt_id="rpa_018f3e70-7b5d-7c21-8b9a-0123456789ac",
    )
    boundary.reserved = replace(boundary.reserved, retained_request=changed)
    assert _check(boundary).outcome is Outcome.CONFLICT


def test_different_issuer_scope_is_not_an_equivalent_reserved_identity(boundary):
    boundary.reserved = replace(
        boundary.reserved, scope=replace(boundary.scope, issuer_id="OTHER_ISSUER")
    )
    assert _check(boundary).outcome is Outcome.CONFLICT


def test_payload_bytes_difference_with_same_root_identifier_is_corrupt(boundary):
    boundary.reserved = replace(
        boundary.reserved, canonical_payload=boundary.reserved.canonical_payload + b" "
    )
    assert _check(boundary).outcome is Outcome.CORRUPT


def test_valid_staged_changed_key_version_conflicts_with_reserved_snapshot(boundary):
    _stage(boundary, snapshot=replace(boundary.snapshot, key_version=2))
    report = _check(boundary)
    assert report.outcome is Outcome.CORRUPT
    # Historical custody rejects even an otherwise valid signature when its
    # original credential snapshot is not the pinned version.
    assert boundary.historical == []


def test_historically_accepted_other_key_version_still_conflicts_with_reserved(
    boundary, monkeypatch
):
    _stage(boundary, snapshot=replace(boundary.snapshot, key_version=2))

    def historical(self, domain_payload, signature, original):
        Ed25519PublicKey.from_public_bytes(original.public_key).verify(signature, domain_payload)
        boundary.historical.append(original)
        return True

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", historical)
    report = _check(boundary)
    assert report.outcome is Outcome.CONFLICT
    assert len(boundary.historical) == 2
    assert all(snapshot.key_version == 2 for snapshot in boundary.historical)


@pytest.mark.parametrize(
    "reserved,staged,expected", [(False, False, Outcome.NOT_FOUND), (False, True, Outcome.CONFLICT)]
)
def test_missing_reserved_never_launders_staged_or_absence(boundary, reserved, staged, expected):
    boundary.reserved = None
    if staged:
        _stage(boundary)
    report = _check(boundary)
    assert report.outcome is expected
    assert report.root_proof_id is None


@pytest.mark.parametrize("column", [0, 1, 2])
def test_corrupted_original_retained_bytes_fail_closed_before_historical_trust(boundary, column):
    _stage(boundary)
    row = list(boundary.row)
    row[column] = row[column][:-1] + bytes([row[column][-1] ^ 1])
    boundary.row = tuple(row)
    report = _check(boundary)
    assert report.outcome is Outcome.CORRUPT
    assert report.root_proof_id is None
    assert boundary.historical == []


def test_invalid_signature_cannot_be_reinterpreted_as_missing_staged(boundary):
    _stage(boundary)
    signature = b"x" * 64
    envelope = retention._envelope(
        boundary.scope,
        boundary.auth,
        boundary.request,
        boundary.row[1],
        signature,
        boundary.snapshot,
        retention._validate_payload(
            boundary.row[1],
            signature,
            boundary.scope,
            boundary.auth,
            boundary.request,
            boundary.snapshot,
        ),
    )
    boundary.row = envelope, boundary.row[1], signature
    assert _check(boundary).outcome is Outcome.CORRUPT
    assert boundary.historical == []


@pytest.mark.parametrize(
    "error,expected",
    [
        (retention.RootProofRetentionUnavailable("hidden DSN"), Outcome.UNAVAILABLE),
        (retention.RootProofRetentionIndeterminate("hidden DSN"), Outcome.INDETERMINATE),
        (retention.RootProofRetentionConflict("hidden DSN"), Outcome.CONFLICT),
        (retention.RootProofRetentionCorrupt("hidden DSN"), Outcome.CORRUPT),
        (RuntimeError("hidden DSN"), Outcome.INDETERMINATE),
    ],
)
def test_refusal_classification_does_not_expose_exception_or_prove_absence(
    boundary, monkeypatch, error, expected
):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "read_prepared", fail)
    report = _check(boundary)
    assert report.outcome is expected
    assert report.root_proof_id is None
    assert "hidden DSN" not in repr(report)


def test_second_read_outage_is_unknown_not_authoritative_absence(boundary, monkeypatch):
    original = retention.PostgreSQLRootProofRetention.read_prepared
    count = 0

    def fail_second(self, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise retention.RootProofRetentionUnavailable("hidden DSN")
        return original(self, **kwargs)

    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "read_prepared", fail_second)
    report = _check(boundary)
    assert report.outcome is Outcome.INDETERMINATE
    assert len(report.observations) == 1


def test_staged_appearing_between_independent_reads_is_unknown(boundary, monkeypatch):
    monkeypatch.setattr(implementation, "_cut", lambda name: _stage(boundary))
    report = _check(boundary)
    assert report.outcome is Outcome.INDETERMINATE
    assert len(report.observations) == 2
    assert report.observations[0] != report.observations[1]


def test_changed_signed_initial_binding_identity_with_absent_rows_is_unknown(boundary, monkeypatch):
    boundary.reserved = None
    second_auth = replace(
        boundary.auth,
        initial_binding_reference="initial-binding-v1:" + "b" * 64,
        initial_binding_digest_sha256="b" * 64,
    )
    second_request = _retained(
        second_auth, (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    )
    count = 0

    def inputs(self, binding, signed_attempt):
        nonlocal count
        count += 1
        return (boundary.auth, boundary.request) if count == 1 else (second_auth, second_request)

    monkeypatch.setattr(intent.PostgreSQLRootProofSigningIntent, "_inputs", inputs)
    monkeypatch.setattr(retention.PostgreSQLRootProofRetention, "_read_row", lambda *args: None)
    report = _check(boundary)
    assert report.outcome is Outcome.INDETERMINATE
    # The full signed authorization/request comparison detects the change even
    # though the detached presence/digest observations have no retained rows.
    assert report.observations[0] == report.observations[1]


@pytest.mark.parametrize("lifecycle", ["VERIFY_ONLY", "REVOKED"])
def test_signing_intent_active_lifecycle_refusal_remains_conflict(boundary, monkeypatch, lifecycle):
    _stage(boundary)

    def refused(*args, **kwargs):
        raise intent.RootProofSigningIntentConflict(
            "SIGNING_INTENT_REQUIRES_EXACT_ACTIVE_SIGNER:" + lifecycle
        )

    monkeypatch.setattr(intent.PostgreSQLRootProofSigningIntent, "read_intent", refused)
    report = _check(boundary)
    assert report.outcome is Outcome.CONFLICT
    assert boundary.historical == []


def test_unpinned_custody_provider_is_refused_before_any_read(boundary):
    other = object.__new__(custody.LocalRootProofSigningProvider)
    object.__setattr__(
        boundary.retention, "_PostgreSQLRootProofRetention__verification_authority", other
    )
    assert _check(boundary).outcome is Outcome.CONFLICT
    assert boundary.reads == []


@pytest.mark.parametrize("port", ["signing_intent", "retention"])
def test_provider_dto_cannot_claim_trusted_origin(boundary, port):
    setattr(
        boundary,
        port,
        SimpleNamespace(read_intent=_reject_mutation, read_prepared=_reject_mutation),
    )
    assert _check(boundary).outcome is Outcome.CONFLICT
    assert boundary.reads == []


def test_fabricated_binding_and_attempt_are_refused_by_real_input_guard(boundary, monkeypatch):
    # Restore the unchanged input guard instead of the diagnostic test port.
    monkeypatch.undo()
    report = implementation.validate_root_proof_atomic_attempt_consistency(
        boundary.signing_intent,
        boundary.retention,
        binding=boundary.binding,
        signed_attempt=boundary.signed_attempt,
    )
    assert report.outcome is Outcome.CONFLICT
    assert report.authorizes_operations is False
    assert boundary.reads == []


def test_changed_signing_domain_is_corrupt_and_never_reaches_ports(boundary, monkeypatch):
    monkeypatch.setattr(implementation, "ROOT_PROOF_SIGNING_DOMAIN", b"OTHER_DOMAIN\x00")
    assert _check(boundary).outcome is Outcome.CORRUPT
    assert boundary.reads == []
