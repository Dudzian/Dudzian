from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.external_provisioning import (
    PACKAGE_SCHEMA,
    MEMBERSHIP_DOMAIN,
    P256_ORDER,
    ConflictError,
    ProductionProvisioningPackageVerifier,
    ProductionMembershipSignerUnavailable,
    ProductionVerifierUnavailable,
    PDSAAuthorizationRequestV1,
    PDSAChallengeV1,
    ProvisioningError,
    ProvisioningRepository,
    SagaState,
    Stage9ProvisioningService,
    TestOnlyProvisioningAuthority,
    TestOnlyProvisioningPackageVerifier,
    TestOnlyMembershipSigner,
)

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
DEVICE = hashlib.sha256(b"TEST_ONLY device P-256 public key").hexdigest()


def payload(*, reference: str = "auth-test-1", subject: str = "psub_test-1", device: str = DEVICE):
    h = hashlib.sha256
    return {
        "schema_version": PACKAGE_SCHEMA,
        "environment": "TEST_ONLY",
        "pdsa_trust_domain": "TEST_ONLY_2_OF_3_ED25519",
        "provisioning_subject_id": subject,
        "enrollment_reference": reference,
        "pdsa_challenge_id": "challenge-test-1",
        "pdsa_challenge_digest_sha256": h(b"challenge").hexdigest(),
        "pre_enrollment_request_digest_sha256": h(b"request").hexdigest(),
        "verified_tpm_exchange_reference": "TEST_ONLY_exchange",
        "verified_tpm_public_projection_id": "TEST_ONLY_projection",
        "target_tpm_ek_public_digest": h(b"ek").hexdigest(),
        "target_tpm_ak_public_digest": h(b"ak").hexdigest(),
        "pre_enrollment_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "pre_enrollment_public_key_fingerprint_sha256": device,
        "authorization_generation": 1,
        "authorization_version": 1,
        "issued_at_utc": "2026-01-02T03:04:04Z",
        "expires_at_utc": "2026-01-02T04:04:05Z",
        "release_policy_digest_sha256": h(b"release").hexdigest(),
        "release_policy_generation": 1,
        "product_profile": "TEST_ONLY",
        "predecessor_package_digest_or_null": None,
        "lineage_generation": 1,
    }


@pytest.fixture
def authority():
    return TestOnlyProvisioningAuthority()


@pytest.fixture
def runtime(tmp_path, authority):
    repository = ProvisioningRepository(tmp_path / "provisioning.db")
    verifier = TestOnlyProvisioningPackageVerifier(authority.public_keys)
    return repository, Stage9ProvisioningService(repository, verifier, TestOnlyMembershipSigner())


def counts(repository):
    with repository._connect() as db:
        return tuple(
            db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in (
                "provisioning_operations",
                "cha_genesis",
                "first_device_memberships",
                "consumed_authorizations",
            )
        )


def test_complete_saga_and_ten_exact_retries_are_one_outcome(runtime, authority):
    repository, service = runtime
    package = authority.issue(payload())
    outcomes = [service.provision(package, expected_device_key=DEVICE, now=NOW) for _ in range(10)]
    assert len(set(outcomes)) == 1
    assert outcomes[0].state is SagaState.CONSUMED
    assert outcomes[0].provisioning_operation_id.startswith("prvop_")
    assert outcomes[0].account_id.startswith("acct_")
    assert counts(repository) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "point",
    [
        "AFTER_PRVOP_BEFORE_CHA",
        "AFTER_CHA_BEFORE_ACCOUNT_OBSERVED",
        "AFTER_ACCOUNT_BEFORE_MEMBERSHIP",
        "AFTER_MEMBERSHIP_BEFORE_COMPLETED",
    ],
)
def test_crash_recovery_at_every_required_boundary(tmp_path, authority, point):
    repository = ProvisioningRepository(tmp_path / "crash.db")
    verifier = TestOnlyProvisioningPackageVerifier(authority.public_keys)
    fired = False

    def crash(name):
        nonlocal fired
        if name == point and not fired:
            fired = True
            raise RuntimeError("simulated process death")

    package = authority.issue(payload())
    with pytest.raises(RuntimeError, match="simulated"):
        Stage9ProvisioningService(
            repository, verifier, TestOnlyMembershipSigner(), crash_hook=crash
        ).provision(package, expected_device_key=DEVICE, now=NOW)
    recovered = Stage9ProvisioningService(
        repository, verifier, TestOnlyMembershipSigner()
    ).provision(package, expected_device_key=DEVICE, now=NOW)
    assert recovered.state is SagaState.CONSUMED
    assert counts(repository) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "mutation", ["signature", "unknown_signer", "expired", "device", "unknown_field"]
)
def test_rejected_package_has_zero_durable_side_effects(runtime, authority, mutation):
    repository, service = runtime
    package = json.loads(authority.issue(payload()))
    if mutation == "signature":
        package["signatures"][0]["signature_hex"] = "00" * 64
    elif mutation == "unknown_signer":
        package["signatures"][0]["key_id"] = "TEST_ONLY_UNKNOWN"
    elif mutation == "expired":
        package["payload"]["expires_at_utc"] = "2026-01-02T03:04:05Z"
        package = json.loads(authority.issue(package["payload"]))
    elif mutation == "device":
        package["payload"]["pre_enrollment_public_key_fingerprint_sha256"] = "0" * 64
        package = json.loads(authority.issue(package["payload"]))
    else:
        package["payload"]["provisioning_operation_id"] = "caller-controlled"
    with pytest.raises(ProvisioningError):
        service.provision(canonical_json_bytes(package), expected_device_key=DEVICE, now=NOW)
    assert counts(repository) == (0, 0, 0, 0)


def test_single_use_authorization_is_consumed_by_logical_operation(runtime, authority):
    repository, service = runtime
    first = service.provision(authority.issue(payload()), expected_device_key=DEVICE, now=NOW)
    changed = payload(subject="psub_test-2")
    with pytest.raises(ConflictError, match="AUTHORIZATION_ALREADY_CONSUMED"):
        service.provision(authority.issue(changed), expected_device_key=DEVICE, now=NOW)
    assert (
        service.provision(authority.issue(payload()), expected_device_key=DEVICE, now=NOW) == first
    )
    assert counts(repository) == (1, 1, 1, 1)


def test_same_subject_new_authorization_creates_new_operation(runtime, authority):
    repository, service = runtime
    one = service.provision(authority.issue(payload()), expected_device_key=DEVICE, now=NOW)
    two = service.provision(
        authority.issue(payload(reference="auth-test-2")), expected_device_key=DEVICE, now=NOW
    )
    assert one.provisioning_operation_id != two.provisioning_operation_id
    assert one.account_id != two.account_id
    assert counts(repository) == (2, 2, 2, 2)


def test_storage_constraints_reject_duplicate_cha_and_membership(runtime, authority):
    repository, service = runtime
    outcome = service.provision(authority.issue(payload()), expected_device_key=DEVICE, now=NOW)
    with repository._connect() as db:
        cha = db.execute("SELECT * FROM cha_genesis").fetchone()
        member = db.execute("SELECT * FROM first_device_memberships").fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO cha_genesis VALUES(?,?,?,?)",
                (
                    "ago_other",
                    outcome.provisioning_operation_id,
                    cha["binding_digest"],
                    "acct_other",
                ),
            )
        values = tuple(member)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO first_device_memberships VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("prvop_other",) + values[1:],
            )
    assert counts(repository) == (1, 1, 1, 1)


def test_illegal_state_transition_fails_closed(runtime, authority):
    repository, service = runtime
    result = service.provision(authority.issue(payload()), expected_device_key=DEVICE, now=NOW)
    with pytest.raises(ConflictError, match="ILLEGAL_SAGA_TRANSITION"):
        repository.transition(
            result.provisioning_operation_id,
            SagaState.PREPARED,
            SagaState.MEMBERSHIP_COMMITTED,
            "later",
        )


def test_production_verifier_is_unconditionally_fail_closed(authority):
    with pytest.raises(ProductionVerifierUnavailable, match="CEREMONY_NOT_COMPLETED"):
        ProductionProvisioningPackageVerifier().verify(
            authority.issue(payload()), expected_device_key=DEVICE, now=NOW
        )
    signer = ProductionMembershipSignerUnavailable()
    with pytest.raises(ProvisioningError, match="LEGAL_ENROLLMENT_NOT_COMPLETED"):
        signer.sign(b"forbidden")


def test_test_membership_signer_cannot_be_composed_with_production_verifier(tmp_path):
    with pytest.raises(ValueError, match="TEST_ONLY membership signer forbidden"):
        Stage9ProvisioningService(
            ProvisioningRepository(tmp_path / "production.db"),
            ProductionProvisioningPackageVerifier(),
            TestOnlyMembershipSigner(),
        )


def test_noncanonical_and_altered_decoded_payload_are_rejected(runtime, authority):
    repository, service = runtime
    package = authority.issue(payload())
    with pytest.raises(ProvisioningError, match="NONCANONICAL"):
        service.provision(package + b"\n", expected_device_key=DEVICE, now=NOW)
    decoded = json.loads(package)
    decoded["payload"]["product_profile"] = "altered"
    with pytest.raises(ProvisioningError):
        service.provision(canonical_json_bytes(decoded), expected_device_key=DEVICE, now=NOW)
    assert counts(repository) == (0, 0, 0, 0)


def test_pdsa_package_has_no_operation_or_account_identity(authority):
    document = json.loads(authority.issue(payload()))
    assert "provisioning_operation_id" not in document["payload"]
    assert "account_id" not in document["payload"]
    assert {item["algorithm"] for item in document["signatures"]} == {"Ed25519"}


def test_challenge_and_request_contracts_are_strict_and_canonical():
    challenge_value = {
        "schema_version": "PDSAChallengeV1",
        "challenge_id": "challenge-1",
        "nonce_hex": "a" * 64,
        "pdsa_trust_domain": "TEST_ONLY",
        "issued_at_utc": "2026-01-02T03:04:04Z",
        "expires_at_utc": "2026-01-02T04:04:04Z",
    }
    challenge = PDSAChallengeV1.from_mapping(challenge_value)
    assert json.loads(challenge.canonical_bytes()) == challenge_value
    with pytest.raises(ProvisioningError, match="SCHEMA"):
        PDSAChallengeV1.from_mapping({**challenge_value, "ignored": "forbidden"})
    request_value = {
        "schema_version": "PDSAAuthorizationRequestV1",
        "pdsa_challenge_id": "challenge-1",
        "pdsa_challenge_digest_sha256": "1" * 64,
        "pre_enrollment_request_digest_sha256": "2" * 64,
        "verified_tpm_exchange_reference": "TEST_ONLY_exchange",
        "pre_enrollment_public_key_fingerprint_sha256": DEVICE,
    }
    request = PDSAAuthorizationRequestV1.from_mapping(request_value)
    assert json.loads(request.canonical_bytes()) == request_value
    with pytest.raises(ProvisioningError, match="SCHEMA"):
        PDSAAuthorizationRequestV1.from_mapping({**request_value, "prvop": "forbidden"})


def test_concurrent_exact_retries_are_constraint_idempotent(runtime, authority):
    repository, service = runtime
    package = authority.issue(payload())
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda _: service.provision(package, expected_device_key=DEVICE, now=NOW), range(16)
            )
        )
    assert len(set(outcomes)) == 1
    assert counts(repository) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("environment", {"name": "TEST_ONLY"}),
        ("environment", "PRODUCTION"),
        ("pdsa_trust_domain", "PRODUCTION_2_OF_3_ED25519"),
        ("product_profile", "PRODUCTION"),
        ("authorization_generation", "1"),
        ("authorization_generation", True),
        ("authorization_generation", 0),
        ("authorization_version", -1),
        ("release_policy_generation", "1"),
        ("lineage_generation", 0),
        ("verified_tpm_exchange_reference", ""),
        ("verified_tpm_public_projection_id", ["projection"]),
        ("predecessor_package_digest_or_null", "garbage"),
        ("issued_at_utc", "2026-01-02T04:04:05Z"),
    ],
)
def test_every_package_field_has_strict_semantics_and_zero_side_effects(
    runtime, authority, field, invalid
):
    repository, service = runtime
    changed = payload()
    changed[field] = invalid
    package = authority.issue(changed)
    with pytest.raises(ProvisioningError):
        service.provision(package, expected_device_key=DEVICE, now=NOW)
    assert counts(repository) == (0, 0, 0, 0)


def _provision_and_membership(runtime, authority):
    repository, service = runtime
    package = authority.issue(payload())
    outcome = service.provision(package, expected_device_key=DEVICE, now=NOW)
    return repository, service, package, outcome


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("signature", b"garbage"),
        ("payload", b'{"tampered":true}'),
        ("membership_digest", "f" * 64),
        ("signer_identity", "TEST_ONLY_wrong-identity"),
        ("signer_profile", "TEST_ONLY_wrong-profile"),
        ("account_id", "acct_tampered"),
        ("subject_id", "psub_tampered"),
        ("package_digest", "a" * 64),
        ("enrollment_reference", "auth-tampered"),
        ("device_key", "b" * 64),
        ("logical_operation_id", "ago_tampered"),
    ],
)
def test_membership_storage_tamper_is_rejected_on_exact_replay(runtime, authority, column, value):
    repository, service, package, outcome = _provision_and_membership(runtime, authority)
    with repository._connect() as db:
        db.execute(
            f"UPDATE first_device_memberships SET {column}=? WHERE prvop=?",
            (value, outcome.provisioning_operation_id),
        )
    with pytest.raises(ProvisioningError):
        service.provision(package, expected_device_key=DEVICE, now=NOW)
    assert counts(repository) == (1, 1, 1, 1)


def test_valid_signed_membership_cannot_be_misbound_to_another_account(runtime, authority):
    repository, service, package, outcome = _provision_and_membership(runtime, authority)
    with repository._connect() as db:
        db.execute(
            "UPDATE first_device_memberships SET account_id=? WHERE prvop=?",
            ("acct_account-b", outcome.provisioning_operation_id),
        )
    with pytest.raises(ConflictError, match="MEMBERSHIP_AUTHENTICATION_BINDING_CONFLICT"):
        service.provision(package, expected_device_key=DEVICE, now=NOW)


def test_valid_signed_membership_cannot_be_misbound_to_another_device(runtime, authority):
    repository, service, package, outcome = _provision_and_membership(runtime, authority)
    with repository._connect() as db:
        db.execute(
            "UPDATE first_device_memberships SET device_key=? WHERE prvop=?",
            ("c" * 64, outcome.provisioning_operation_id),
        )
    with pytest.raises(ConflictError, match="MEMBERSHIP_AUTHENTICATION_BINDING_CONFLICT"):
        service.provision(package, expected_device_key=DEVICE, now=NOW)


def test_membership_high_s_signature_is_rejected_on_replay(runtime, authority):
    repository, service, package, outcome = _provision_and_membership(runtime, authority)
    with repository._connect() as db:
        signature = db.execute(
            "SELECT signature FROM first_device_memberships WHERE prvop=?",
            (outcome.provisioning_operation_id,),
        ).fetchone()[0]
        r, s = decode_dss_signature(signature)
        high_s = encode_dss_signature(r, P256_ORDER - s)
        db.execute(
            "UPDATE first_device_memberships SET signature=? WHERE prvop=?",
            (high_s, outcome.provisioning_operation_id),
        )
    with pytest.raises(ProvisioningError, match="LOW_S"):
        service.provision(package, expected_device_key=DEVICE, now=NOW)


def test_membership_wrong_valid_p256_key_is_rejected_on_replay(runtime, authority):
    repository, service, package, outcome = _provision_and_membership(runtime, authority)
    with repository._connect() as db:
        digest = db.execute(
            "SELECT membership_digest FROM first_device_memberships WHERE prvop=?",
            (outcome.provisioning_operation_id,),
        ).fetchone()[0]
        wrong_key = ec.derive_private_key(2, ec.SECP256R1())
        signature = wrong_key.sign(
            MEMBERSHIP_DOMAIN + bytes.fromhex(digest), ec.ECDSA(hashes.SHA256())
        )
        r, s = decode_dss_signature(signature)
        signature = encode_dss_signature(r, min(s, P256_ORDER - s))
        db.execute(
            "UPDATE first_device_memberships SET signature=? WHERE prvop=?",
            (signature, outcome.provisioning_operation_id),
        )
    with pytest.raises(ProvisioningError, match="SIGNATURE_INVALID"):
        service.provision(package, expected_device_key=DEVICE, now=NOW)
