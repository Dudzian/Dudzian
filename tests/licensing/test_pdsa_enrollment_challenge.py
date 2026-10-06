"""TEST_ONLY quorum/loader harness; no production authority or TPM is created."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

import bot_core.licensing.pdsa_enrollment_challenge as challenge
import deployment.production_enrollment_issuer as issuer
import deployment.windows_stage9_production_trust as trust
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.pre_enrollment import (
    ALGORITHM_PROFILE,
    PAYLOAD_FIELDS,
    PDSA_TRUST_DOMAIN,
    PreEnrollmentRequestV1,
)

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # The canonical loader's package verifier is explicitly mocked. These private
    # TEST_ONLY software fixtures never enter installed production composition.
    keys = {
        f"TEST_ONLY_CHALLENGE_{index}": ed25519.Ed25519PrivateKey.from_private_bytes(
            bytes([index]) * 32
        )
        for index in range(1, 4)
    }
    release = SimpleNamespace(
        payload_digest=trust.RELEASE_PAYLOAD_DIGEST,
        release_version=1,
        pinned_root=SimpleNamespace(
            key_set_digest=trust.ROOT_KEY_SET_DIGEST, environment="PRODUCTION"
        ),
        pdsa_key_set_digest=trust.PDSA_KEY_SET_DIGEST,
        recovery_public_digest=trust.RECOVERY_PUBLIC_DIGEST,
        purpose="PRODUCTION",
        pdsa_threshold=2,
        pdsa_keys=keys,
    )
    ceremony = SimpleNamespace(ceremony_id=trust.CEREMONY_ID, verified_release=release)
    monkeypatch.setattr(trust, "verify_final_package", lambda *args, **kwargs: ceremony)

    def public_document(path: Path):
        if path.name == "pdsa_public_bundle.json":
            return {
                "threshold": 2,
                "keys": [
                    {
                        "key_id": key_id,
                        "public_key_hex": key.public_key()
                        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
                        .hex(),
                    }
                    for key_id, key in keys.items()
                ],
            }
        return {"status": "PRODUCTION_ROOT_OF_TRUST_FROZEN"}

    monkeypatch.setattr(trust, "_canonical_document", public_document)
    context = trust.verify_production_trust_for_audit(tmp_path, verification_time=NOW)
    # Explicit TEST_ONLY current-release harness. Production never substitutes
    # audit provenance for the current loader/reverification predicate.
    monkeypatch.setattr(
        challenge,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        issuer,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    # Substitute only the installed service configuration boundary. The genuine
    # factory and private issuer/store registries remain active in this TEST_ONLY
    # harness; arbitrary path constructors never receive production authority.
    monkeypatch.setattr(
        issuer,
        "_installed_service_configuration",
        lambda: issuer._InstalledIssuerConfiguration(
            state_directory=tmp_path / "TEST_ONLY_ISSUER", trust=context
        ),
    )
    installed = issuer.open_installed_production_enrollment_issuer()

    def sign(message: bytes):
        return [
            {"key_id": key_id, "algorithm": "Ed25519", "signature_hex": key.sign(message).hex()}
            for key_id, key in list(keys.items())[:2]
        ]

    return SimpleNamespace(
        context=context,
        keys=keys,
        issuer=installed,
        store=installed.pdsa_store,
        sign=sign,
        reopen=issuer.open_installed_production_enrollment_issuer,
    )


def _request(raw: bytes, *, nonce: str = "ab" * 32) -> PreEnrollmentRequestV1:
    public = (
        ec.derive_private_key(1, ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    artifact = challenge.PDSAEnrollmentChallengeV1.from_canonical_bytes(raw)
    payload = artifact.document["payload"]
    value = {field: "ab" * 32 for field in PAYLOAD_FIELDS}
    value.update(
        schema_version="PreEnrollmentRequestV1",
        environment="PRODUCTION",
        product="CryptoHunter",
        product_profile="CryptoHunter",
        pdsa_trust_domain=PDSA_TRUST_DOMAIN,
        pdsa_challenge_id=payload["challenge_id"],
        pdsa_challenge_digest_sha256=artifact.digest_sha256,
        pdsa_challenge_nonce_digest_sha256=artifact.nonce_digest_sha256,
        release_policy_digest_sha256=payload["release_policy_digest_sha256"],
        release_policy_generation=payload["release_policy_generation"],
        pre_enrollment_public_key_algorithm_profile=ALGORITHM_PROFILE,
        pre_enrollment_public_key_canonical_bytes=public.hex(),
        pre_enrollment_public_key_fingerprint_sha256=hashlib.sha256(public).hexdigest(),
        request_nonce_hex=nonce,
    )
    return PreEnrollmentRequestV1.from_mapping(value)


def _rows(store):
    with sqlite3.connect(store.path) as db:
        db.row_factory = sqlite3.Row
        return db.execute("SELECT * FROM pdsa_challenges").fetchall()


def _mechanics_capability(monkeypatch, request_raw, challenge_raw, context):
    # Bypass ONLY the independent authentication guard in this explicit mechanics
    # harness. Production consume always imports the genuine provenance predicate.
    import bot_core.licensing.production_pre_enrollment as production

    marker = object()

    def accept(value, **kwargs):
        assert value is marker
        return SimpleNamespace(
            request_raw=request_raw, challenge_raw=challenge_raw, context=context
        )

    monkeypatch.setattr(production, "require_authenticated_pre_enrollment", accept)
    return marker


def test_issuer_generated_signed_canonical_challenge_retained_before_return(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    rows = _rows(harness.store)
    assert len(rows) == 1 and rows[0]["state"] == "ISSUED"
    assert rows[0]["challenge_raw"] == raw
    artifact = challenge.PDSAEnrollmentChallengeV1.from_canonical_bytes(raw)
    assert rows[0]["challenge_digest"] == hashlib.sha256(raw).hexdigest()
    assert artifact.digest_sha256 == hashlib.sha256(raw).hexdigest()
    payload = artifact.document["payload"]
    assert (
        artifact.nonce_digest_sha256
        == hashlib.sha256(bytes.fromhex(payload["nonce_hex"])).hexdigest()
    )
    assert set(payload) == challenge.PAYLOAD_FIELDS
    assert payload["issued_at_utc"] == "2026-10-06T12:00:00Z"
    assert payload["expires_at_utc"] == "2026-10-13T12:00:00Z"
    verified = harness.store.verify_issued(raw, harness.context)
    assert challenge.require_verified_issued_challenge(verified) is verified
    verified.require_request_binding(_request(raw))
    with pytest.raises(TypeError, match="immutable"):
        verified.canonical_bytes = b"changed"
    reopened = harness.reopen()
    assert reopened.pdsa_store.verify_issued(raw, harness.context)
    assert issuer.require_production_pdsa_store(reopened.pdsa_store) is reopened


def test_issuer_freshness_is_independent_and_not_caller_selectable(harness):
    first = json.loads(harness.store.issue(harness.context, harness.sign))["payload"]
    second = json.loads(harness.store.issue(harness.context, harness.sign))["payload"]
    assert first["challenge_id"] != second["challenge_id"]
    assert first["nonce_hex"] != second["nonce_hex"]
    with pytest.raises(TypeError):
        harness.store.issue(harness.context, harness.sign, nonce_hex="00" * 32)


@pytest.mark.parametrize("context", [None, object(), SimpleNamespace(environment="PRODUCTION")])
def test_arbitrary_caller_selected_trust_keys_rejected(harness, context):
    with pytest.raises((trust.ProductionTrustUnavailable, issuer.ProductionEnrollmentIssuerError)):
        harness.store.issue(context, harness.sign)
    raw = harness.store.issue(harness.context, harness.sign)
    with pytest.raises((trust.ProductionTrustUnavailable, issuer.ProductionEnrollmentIssuerError)):
        harness.store.verify_issued(raw, context)


def test_audit_only_context_cannot_issue_through_real_current_guard(harness, monkeypatch):
    monkeypatch.setattr(
        challenge,
        "require_current_production_trust_context",
        trust.require_current_production_trust_context,
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="CURRENT_RUNTIME"):
        harness.store.issue(harness.context, harness.sign)
    assert _rows(harness.store) == []


def test_client_can_verify_signed_challenge_without_issuer_retention_authority(
    harness, monkeypatch
):
    raw = harness.store.issue(harness.context, harness.sign)
    public = challenge.verify_signed_production_pdsa_challenge(raw, harness.context)
    assert type(public) is challenge.PDSAEnrollmentChallengeV1
    assert public.digest_sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(public)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW - timedelta(seconds=1))
    with pytest.raises(challenge.PDSAChallengeError, match="NOT_YET_VALID"):
        challenge.verify_signed_production_pdsa_challenge(raw, harness.context)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        challenge.verify_signed_production_pdsa_challenge(raw, harness.context)
    # A client has no issuer-store mutation authority; only issuer verification
    # durably marks an unconsumed challenge EXPIRED.
    assert _rows(harness.store)[0]["state"] == "ISSUED"


def test_client_signed_verification_rejects_real_audit_only_context(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    monkeypatch.setattr(
        challenge,
        "require_current_production_trust_context",
        trust.require_current_production_trust_context,
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="CURRENT_RUNTIME"):
        challenge.verify_signed_production_pdsa_challenge(raw, harness.context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", "TEST_ONLY"),
        ("product", "OtherProduct"),
        ("product_profile", "TEST_ONLY"),
        ("pdsa_trust_domain", "TEST_ONLY_PDSA"),
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", 2),
        ("release_policy_generation", True),
        ("pdsa_key_set_digest", "00" * 32),
        ("nonce_hex", "00" * 32),
        ("signature_algorithm_profile", "Ed25519"),
        ("expires_at_utc", "2026-10-14T12:00:00Z"),
        ("issued_at_utc", "2026-10-06T12:00:00+00:00"),
        ("issued_at_utc", "2026-10-06T12:00:00.000Z"),
        ("challenge_id", "pchal_01930000-0000-7000-8000-000000000001"),
    ],
)
def test_changed_signed_payload_rejected(harness, field, value):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    document["payload"][field] = value
    with pytest.raises(challenge.PDSAChallengeError):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)
    assert _rows(harness.store)[0]["state"] == "ISSUED"


def test_changed_challenge_uuid_random_bits_rejected(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    old = document["payload"]["challenge_id"]
    document["payload"]["challenge_id"] = old[:-1] + ("0" if old[-1] != "0" else "1")
    with pytest.raises(challenge.PDSAChallengeError, match="SIGNATURE"):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


@pytest.mark.parametrize("field", sorted(challenge.PAYLOAD_FIELDS))
def test_missing_payload_fields_rejected(harness, field):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    del document["payload"][field]
    with pytest.raises(challenge.PDSAChallengeError, match="SCHEMA"):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


@pytest.mark.parametrize("scope", ["envelope", "payload", "signature"])
def test_unknown_fields_rejected(harness, scope):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    target = document if scope == "envelope" else document["payload"]
    if scope == "signature":
        target = document["signatures"][0]
    target["extra"] = "forbidden"
    with pytest.raises(challenge.PDSAChallengeError, match="SCHEMA"):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


@pytest.mark.parametrize("bad", ["00" * 64, "AA" * 64, "00" * 63, "not-hex", None, True])
def test_wrong_or_malformed_signature_rejected(harness, bad):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    document["signatures"][0]["signature_hex"] = bad
    with pytest.raises(challenge.PDSAChallengeError, match="SIGNATURE"):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "reversed", "one", "algorithm"])
def test_threshold_key_identity_and_order_rejected(harness, mutation):
    raw = harness.store.issue(harness.context, harness.sign)
    document = json.loads(raw)
    records = document["signatures"]
    if mutation == "unknown":
        records[0]["key_id"] = "OTHER_CALLER_SELECTED_KEY"
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    elif mutation == "reversed":
        records.reverse()
    elif mutation == "one":
        records.pop()
    else:
        records[0]["algorithm"] = "TEST_ONLY_Ed25519"
    with pytest.raises(challenge.PDSAChallengeError):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


def test_three_valid_signatures_accepted_but_bad_third_signature_rejects_whole(harness):
    def sign_three(message):
        return [
            {"key_id": key_id, "algorithm": "Ed25519", "signature_hex": key.sign(message).hex()}
            for key_id, key in harness.keys.items()
        ]

    raw = harness.store.issue(harness.context, sign_three)
    assert harness.store.verify_issued(raw, harness.context)
    changed = json.loads(raw)
    changed["signatures"][2]["signature_hex"] = "00" * 64
    with pytest.raises(challenge.PDSAChallengeError, match="SIGNATURE"):
        harness.store.verify_issued(canonical_json_bytes(changed), harness.context)


def test_wrong_key_signature_cannot_be_retained(harness):
    outsider = ed25519.Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)

    def foreign_signatures(message):
        records = harness.sign(message)
        records[0]["signature_hex"] = outsider.sign(message).hex()
        return records

    with pytest.raises(challenge.PDSAChallengeError, match="SIGNATURE"):
        harness.store.issue(harness.context, foreign_signatures)
    assert _rows(harness.store) == []


def test_lost_signer_response_never_publishes_issued_state(harness):
    def lost_response(message):
        harness.sign(message)
        raise TimeoutError("TEST_ONLY lost signing response")

    with pytest.raises(TimeoutError):
        harness.store.issue(harness.context, lost_response)
    assert _rows(harness.store) == []


def test_identifier_collision_does_not_overwrite_retained_identity(harness, monkeypatch):
    monkeypatch.setattr(challenge.secrets, "randbits", lambda bits: 1)
    raw = harness.store.issue(harness.context, harness.sign)
    with pytest.raises(challenge.PDSAChallengeError, match="ISSUANCE_CONFLICT"):
        harness.store.issue(harness.context, harness.sign)
    row = _rows(harness.store)[0]
    assert row["challenge_raw"] == raw and row["state"] == "ISSUED"


@pytest.mark.parametrize("raw", [None, "text", b"[]", b"x" * 16385])
def test_public_artifact_rejects_bad_bytes(raw):
    with pytest.raises(challenge.PDSAChallengeError):
        challenge.PDSAEnrollmentChallengeV1.from_canonical_bytes(raw)


def test_noncanonical_and_changed_envelope_bytes_rejected(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    for bad in (raw + b"\n", b" " + raw, json.dumps(json.loads(raw)).encode(), b"[]", b"{bad}"):
        with pytest.raises(challenge.PDSAChallengeError):
            harness.store.verify_issued(bad, harness.context)
    # A different valid envelope over the same signed payload is still different
    # challenge bytes. Retained equality includes the exact signature block.
    document = json.loads(raw)
    message = (
        challenge.SIGNATURE_DOMAIN
        + hashlib.sha256(canonical_json_bytes(document["payload"])).digest()
    )
    key_id, key = list(harness.keys.items())[2]
    document["signatures"].append(
        {"key_id": key_id, "algorithm": "Ed25519", "signature_hex": key.sign(message).hex()}
    )
    with pytest.raises(challenge.PDSAChallengeError, match="BYTES_MISMATCH"):
        harness.store.verify_issued(canonical_json_bytes(document), harness.context)


def test_unknown_retained_id_rejected(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    with harness.store._connect() as db:
        db.execute("DELETE FROM pdsa_challenges")
    with pytest.raises(challenge.PDSAChallengeError, match="UNKNOWN_RETAINED"):
        harness.store.verify_issued(raw, harness.context)


def test_expiry_is_current_clock_terminal_and_invalidates_existing_capability(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        challenge.require_verified_issued_challenge(verified)
    assert _rows(harness.store)[0]["state"] == "EXPIRED"
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW)
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        harness.store.verify_issued(raw, harness.context)
    harness.issuer.close()
    restarted = harness.reopen()
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        restarted.pdsa_store.verify_issued(raw, harness.context)


def test_future_challenge_rejected_and_slow_signer_cannot_publish_expired(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW - timedelta(seconds=1))
    with pytest.raises(challenge.PDSAChallengeError, match="NOT_YET_VALID"):
        harness.store.verify_issued(raw, harness.context)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW)

    def slow_sign(message):
        monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
        return harness.sign(message)

    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        harness.store.issue(harness.context, slow_sign)
    assert len(_rows(harness.store)) == 1


def test_issuance_rechecks_expiry_after_waiting_for_database_lock(harness, monkeypatch):
    original_connect = challenge.PDSAChallengeStore._connect

    @contextmanager
    def delayed_database(store):
        with original_connect(store) as db:
            if store is harness.store:
                monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
            yield db

    monkeypatch.setattr(challenge.PDSAChallengeStore, "_connect", delayed_database)
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        harness.store.issue(harness.context, harness.sign)
    assert _rows(harness.store) == []


@pytest.mark.parametrize("operation", ["issue", "consume"])
def test_current_trust_rechecked_after_database_lock_wait(harness, monkeypatch, operation):
    raw = harness.store.issue(harness.context, harness.sign) if operation == "consume" else None
    marker = None
    if raw is not None:
        marker = _mechanics_capability(
            monkeypatch, _request(raw).canonical_bytes, raw, harness.context
        )
    original_connect = challenge.PDSAChallengeStore._connect
    original_guard = challenge.require_current_production_trust_context
    lock_acquired = False

    @contextmanager
    def delayed_database(store):
        nonlocal lock_acquired
        with original_connect(store) as db:
            if store is harness.store:
                lock_acquired = True
            yield db

    def changed_current_trust(context):
        if lock_acquired:
            raise trust.ProductionTrustUnavailable("TEST_ONLY release expired while waiting")
        return original_guard(context)

    monkeypatch.setattr(challenge.PDSAChallengeStore, "_connect", delayed_database)
    monkeypatch.setattr(
        challenge, "require_current_production_trust_context", changed_current_trust
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="release expired"):
        if operation == "issue":
            harness.store.issue(harness.context, harness.sign)
        else:
            harness.store.consume_authenticated_request(marker)
    rows = _rows(harness.store)
    if operation == "issue":
        assert rows == []
    else:
        assert rows[0]["state"] == "ISSUED" and rows[0]["request_raw"] is None


def test_verified_fields_cannot_transfer_provenance_or_change_store(harness, tmp_path):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    forged = object.__new__(challenge.VerifiedIssuedPDSAChallenge)
    for field in ("canonical_bytes", "digest_sha256", "nonce_digest_sha256"):
        object.__setattr__(forged, field, getattr(verified, field))
    with pytest.raises(TypeError, match="immutable"):
        copy.copy(verified)
    for bad in (None, object(), forged):
        with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
            challenge.require_verified_issued_challenge(bad)
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(
            verified, store=challenge.PDSAChallengeStore(tmp_path / "other.sqlite")
        )
    object.__setattr__(verified, "digest_sha256", "00" * 32)
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(verified)


@pytest.mark.parametrize(
    "field", ["pdsa_challenge_digest_sha256", "pdsa_challenge_nonce_digest_sha256"]
)
def test_exact_request_challenge_bindings_required(harness, field):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    document = _request(raw).document
    document[field] = "00" * 32
    with pytest.raises(challenge.PDSAChallengeError, match="BINDING_MISMATCH"):
        verified.require_request_binding(PreEnrollmentRequestV1.from_mapping(document))


def test_authenticated_consume_retains_exact_request_receipt_and_retry(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    request = _request(raw)
    marker = _mechanics_capability(monkeypatch, request.canonical_bytes, raw, harness.context)
    receipt = harness.store.consume_authenticated_request(marker)
    row = _rows(harness.store)[0]
    assert row["state"] == "CONSUMED"
    assert row["request_raw"] == request.canonical_bytes
    assert row["request_digest"] == request.digest_sha256
    assert row["receipt_raw"] == receipt
    assert harness.store.consume_authenticated_request(marker) == receipt
    assert json.loads(receipt)["legal_enrollment"] == "NOT_PERFORMED"
    harness.issuer.close()
    restarted = harness.reopen()
    reopened = restarted.pdsa_store
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=8))
    assert (
        reopened.retry_exact_accepted(request_raw=request.canonical_bytes, challenge_raw=raw)
        == receipt
    )
    with pytest.raises(challenge.PDSAChallengeError, match="REPLAY_CONFLICT"):
        reopened.retry_exact_accepted(
            request_raw=_request(raw, nonce="00" * 32).canonical_bytes, challenge_raw=raw
        )
    with pytest.raises(challenge.PDSAChallengeError, match="ALREADY_CONSUMED"):
        reopened.verify_issued(raw, harness.context)
    assert _rows(harness.store)[0]["state"] == "CONSUMED"


def test_consumed_different_request_cannot_reactivate_challenge(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    first = _request(raw)
    marker = _mechanics_capability(monkeypatch, first.canonical_bytes, raw, harness.context)
    receipt = harness.store.consume_authenticated_request(marker)
    changed = _request(raw, nonce="00" * 32)
    different = _mechanics_capability(monkeypatch, changed.canonical_bytes, raw, harness.context)
    with pytest.raises(challenge.PDSAChallengeError, match="REPLAY_CONFLICT"):
        harness.store.consume_authenticated_request(different)
    row = _rows(harness.store)[0]
    assert row["state"] == "CONSUMED"
    assert row["request_raw"] == first.canonical_bytes
    assert row["receipt_raw"] == receipt


def test_consumption_checks_expiry_again_and_no_authority_from_raw_digest(harness, monkeypatch):
    raw = harness.store.issue(harness.context, harness.sign)
    request = _request(raw)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        harness.store.consume_authenticated_request(request.digest_sha256)
    assert _rows(harness.store)[0]["state"] == "ISSUED"
    marker = _mechanics_capability(monkeypatch, request.canonical_bytes, raw, harness.context)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        harness.store.consume_authenticated_request(marker)
    row = _rows(harness.store)[0]
    assert row["state"] == "EXPIRED"
    assert row["request_raw"] is None and row["receipt_raw"] is None


def test_issued_retry_does_not_mint_result(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    assert (
        harness.store.retry_exact_accepted(
            request_raw=_request(raw).canonical_bytes, challenge_raw=raw
        )
        is None
    )
    assert _rows(harness.store)[0]["state"] == "ISSUED"


def test_connection_requires_full_synchronization_and_closes(harness):
    with harness.store._connect() as db:
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")


_TEST_ONLY_LEGACY_AUTHORIZATION_SCHEMA = """CREATE TABLE pdsa_authorization_issuances (
    pdsa_challenge_id TEXT PRIMARY KEY,
    pdsa_challenge_digest_sha256 TEXT NOT NULL,
    pre_enrollment_request_digest_sha256 TEXT NOT NULL UNIQUE,
    request_raw BLOB NOT NULL,
    payload_raw BLOB NOT NULL,
    provisioning_subject_id TEXT NOT NULL UNIQUE,
    enrollment_reference TEXT NOT NULL UNIQUE,
    issued_at_utc TEXT NOT NULL,
    expires_at_utc TEXT NOT NULL,
    signer_ids_raw BLOB NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('RESERVED','SIGNED','COMMITTED')),
    package_raw BLOB,
    pdsa_package_digest_sha256 TEXT UNIQUE,
    CHECK((state='RESERVED' AND package_raw IS NULL
        AND pdsa_package_digest_sha256 IS NULL)
        OR (state IN ('SIGNED','COMMITTED') AND package_raw IS NOT NULL
        AND pdsa_package_digest_sha256 IS NOT NULL)),
    FOREIGN KEY(pdsa_challenge_id) REFERENCES pdsa_challenges(challenge_id)
)"""


def _test_only_legacy_authorization_database(tmp_path: Path, *, state: str | None) -> Path:
    path = tmp_path / "TEST_ONLY_LEGACY_AUTHORIZATION.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(_TEST_ONLY_LEGACY_AUTHORIZATION_SCHEMA)
        if state is not None:
            db.execute(
                "INSERT INTO pdsa_authorization_issuances VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "TEST_ONLY_CHALLENGE",
                    "ab" * 32,
                    "cd" * 32,
                    b"TEST_ONLY_EXACT_REQUEST",
                    b"TEST_ONLY_EXACT_PAYLOAD",
                    "TEST_ONLY_DURABLE_SUBJECT",
                    "TEST_ONLY_DURABLE_REFERENCE",
                    "2026-10-06T12:00:00Z",
                    "2026-10-06T13:00:00Z",
                    canonical_json_bytes(["TEST_ONLY_K1", "TEST_ONLY_K2"]),
                    state,
                    None if state == "RESERVED" else b"TEST_ONLY_EXACT_SIGNED_PACKAGE",
                    None if state == "RESERVED" else "ef" * 32,
                ),
            )
    return path


def test_empty_legacy_authorization_schema_upgrades_without_inventing_quorum(tmp_path):
    path = _test_only_legacy_authorization_database(tmp_path, state=None)
    store = challenge.PDSAChallengeStore(path)
    with store._connect() as db:
        columns = {
            row["name"]: row
            for row in db.execute("PRAGMA table_info(pdsa_authorization_issuances)")
        }
        assert columns["production_trust_raw"]["notnull"] == 1
        assert columns["authorized_signer_ids_raw"]["notnull"] == 1
        assert columns["required_threshold"]["notnull"] == 1
        assert columns["signer_ids_raw"]["notnull"] == 0
        assert db.execute("SELECT * FROM pdsa_authorization_issuances").fetchall() == []
    # Reopening the current schema is idempotent and does not create a reservation.
    challenge.PDSAChallengeStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM pdsa_authorization_issuances").fetchall() == []


@pytest.mark.parametrize("state", ["RESERVED", "SIGNED", "COMMITTED"])
def test_populated_legacy_authorization_schema_fails_closed_and_preserves_exact_rows(
    tmp_path, state
):
    path = _test_only_legacy_authorization_database(tmp_path, state=state)
    with sqlite3.connect(path) as db:
        schema_before = db.execute(
            "SELECT sql,rootpage FROM sqlite_master WHERE name='pdsa_authorization_issuances'"
        ).fetchone()
        rows_before = db.execute("SELECT * FROM pdsa_authorization_issuances").fetchall()
    with pytest.raises(
        challenge.PDSAChallengeError, match="PDSA_AUTHORIZATION_ISSUANCE_SCHEMA_MISMATCH"
    ):
        challenge.PDSAChallengeStore(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute(
                "SELECT sql,rootpage FROM sqlite_master WHERE name='pdsa_authorization_issuances'"
            ).fetchone()
            == schema_before
        )
        assert db.execute("SELECT * FROM pdsa_authorization_issuances").fetchall() == rows_before
        assert "production_trust_raw" not in {
            row[1] for row in db.execute("PRAGMA table_info(pdsa_authorization_issuances)")
        }


@pytest.mark.parametrize(
    "malformed_schema",
    [
        "CREATE TABLE pdsa_authorization_issuances (pdsa_challenge_id TEXT PRIMARY KEY)",
        _TEST_ONLY_LEGACY_AUTHORIZATION_SCHEMA.replace(
            "signer_ids_raw BLOB NOT NULL,",
            "production_trust_raw BLOB NOT NULL,authorized_signer_ids_raw BLOB NOT NULL,"
            "required_threshold INTEGER NOT NULL,signer_ids_raw BLOB NOT NULL,",
        ),
        _TEST_ONLY_LEGACY_AUTHORIZATION_SCHEMA.replace(
            "signer_ids_raw BLOB NOT NULL,",
            "production_trust_raw BLOB,authorized_signer_ids_raw BLOB NOT NULL,"
            "required_threshold INTEGER NOT NULL,signer_ids_raw BLOB,",
        ),
    ],
    ids=["missing_quorum_columns", "selected_pair_required_while_reserved", "trust_not_required"],
)
def test_partial_authorization_schema_fails_closed_without_replacement(tmp_path, malformed_schema):
    path = tmp_path / "TEST_ONLY_PARTIAL_AUTHORIZATION.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(malformed_schema)
        before = db.execute(
            "SELECT sql,rootpage FROM sqlite_master WHERE name='pdsa_authorization_issuances'"
        ).fetchone()
    with pytest.raises(
        challenge.PDSAChallengeError, match="PDSA_AUTHORIZATION_ISSUANCE_SCHEMA_MISMATCH"
    ):
        challenge.PDSAChallengeStore(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute(
                "SELECT sql,rootpage FROM sqlite_master WHERE name='pdsa_authorization_issuances'"
            ).fetchone()
            == before
        )


def test_arbitrary_path_store_cannot_issue_or_verify_even_with_signed_issuer_bytes(
    harness, tmp_path
):
    raw = harness.store.issue(harness.context, harness.sign)
    mechanics = challenge.PDSAChallengeStore(tmp_path / "CALLER_SELECTED.sqlite")
    called = False

    def forbidden_signer(message):
        nonlocal called
        called = True
        return harness.sign(message)

    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.issue(harness.context, forbidden_signer)
    assert called is False
    assert _rows(mechanics) == []
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.verify_issued(raw, harness.context)
    # Provenance is checked before parsing, authentication, or receipt retrieval.
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.verify_issued(b"malformed", harness.context)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.retry_exact_accepted(request_raw=b"malformed", challenge_raw=b"malformed")
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.consume_authenticated_request(object())


def test_manually_reconstructed_schema_and_valid_issued_row_have_no_provenance(harness, tmp_path):
    raw = harness.store.issue(harness.context, harness.sign)
    path = tmp_path / "RECONSTRUCTED_ISSUED.sqlite"
    with harness.store._connect() as source:
        schema = source.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pdsa_challenges'"
        ).fetchone()[0]
        row = source.execute("SELECT * FROM pdsa_challenges").fetchone()
    with sqlite3.connect(path) as reconstructed:
        reconstructed.execute(schema)
        reconstructed.execute("INSERT INTO pdsa_challenges VALUES (?,?,?,?,?,?,?,?)", tuple(row))
    mechanics = challenge.PDSAChallengeStore(path)
    assert _rows(mechanics)[0]["challenge_raw"] == raw
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        mechanics.verify_issued(raw, harness.context)


def test_byte_for_byte_sqlite_copy_cannot_mint_verified_issued_capability(harness, tmp_path):
    raw = harness.store.issue(harness.context, harness.sign)
    with harness.store._connect() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = tmp_path / "BYTE_COPY_ISSUED.sqlite"
    shutil.copyfile(harness.store.path, path)
    assert path.read_bytes() == harness.store.path.read_bytes()
    copied = challenge.PDSAChallengeStore(path)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        copied.verify_issued(raw, harness.context)


def test_preconsumption_issued_snapshot_cannot_replay_after_canonical_consume(
    harness, tmp_path, monkeypatch
):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    with harness.store._connect() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = tmp_path / "STALE_ISSUED_SNAPSHOT.sqlite"
    shutil.copyfile(harness.store.path, path)
    copied = challenge.PDSAChallengeStore(path)
    request = _request(raw)
    marker = _mechanics_capability(monkeypatch, request.canonical_bytes, raw, harness.context)
    receipt = harness.store.consume_authenticated_request(marker)
    assert _rows(copied)[0]["state"] == "ISSUED"
    assert _rows(harness.store)[0]["state"] == "CONSUMED"
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        copied.verify_issued(raw, harness.context)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        copied.consume_authenticated_request(marker)
    with pytest.raises(challenge.PDSAChallengeError, match="ALREADY_CONSUMED"):
        challenge.require_verified_issued_challenge(verified)
    harness.issuer.close()
    restarted = harness.reopen()
    reopened = restarted.pdsa_store
    assert (
        reopened.retry_exact_accepted(request_raw=request.canonical_bytes, challenge_raw=raw)
        == receipt
    )
    with pytest.raises(challenge.PDSAChallengeError, match="ALREADY_CONSUMED"):
        reopened.verify_issued(raw, harness.context)


def test_store_python_fields_and_configured_path_do_not_transfer_factory_provenance(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    forged = object.__new__(challenge.PDSAChallengeStore)
    object.__setattr__(forged, "_path", harness.store.path)
    for store in (forged, challenge.PDSAChallengeStore(harness.store.path)):
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
            store.verify_issued(raw, harness.context)
    with pytest.raises(TypeError, match="immutable"):
        copy.copy(harness.store)
    with pytest.raises(TypeError, match="immutable"):
        harness.store.path = harness.store.path


def test_registered_store_configuration_mutation_invalidates_store_and_capability(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    object.__setattr__(harness.store, "_path", harness.store.path.with_name("MUTATED.sqlite"))
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        harness.store.verify_issued(raw, harness.context)
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(verified)


def test_verified_issued_capability_is_bound_to_exact_issuer_context(harness):
    raw = harness.store.issue(harness.context, harness.sign)
    verified = harness.store.verify_issued(raw, harness.context)
    reopened = harness.reopen()
    assert (
        challenge.require_verified_issued_challenge(
            verified, store=harness.store, context=harness.context, issuer=harness.issuer
        )
        is verified
    )
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(verified, issuer=reopened)
    with pytest.raises(challenge.PDSAChallengeError, match="VERIFIED_ISSUED"):
        challenge.require_verified_issued_challenge(verified, store=reopened.pdsa_store)
