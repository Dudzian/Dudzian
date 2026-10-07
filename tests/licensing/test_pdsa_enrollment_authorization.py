"""TEST_ONLY issuer simulations with genuine #3067 cryptographic verification.

The installed trust/configuration and NCrypt boundaries are explicitly mocked
by the reused pre-enrollment fixture. The only additional substitute is the
off-host signing service, using the fixture's labelled TEST_ONLY Ed25519 keys.
No production private signing material or physical TPM qualification is used.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from bot_core.licensing import (
    pdsa_enrollment_authorization as authorization,
    pdsa_enrollment_challenge as challenge,
    production_pre_enrollment as production,
    production_tpm_custody as custody,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.external_provisioning import (
    PACKAGE_FIELDS,
    PDSA_DOMAIN,
    ProductionProvisioningPackageVerifier,
    ProvisioningError,
)
from bot_core.licensing.pre_enrollment import PreEnrollmentRequestV1
from deployment import (
    production_enrollment_issuer as issuer,
    windows_stage9_production_trust as trust,
)
from tests.licensing import test_production_pre_enrollment as pre_enrollment_tests
from tests.licensing.test_production_pre_enrollment import (
    NOW,
    _authenticate,
    _changed_request,
    _copy_database,
)
from tests.licensing.test_production_tpm_custody import _attest, _signature

challenge_harness = pre_enrollment_tests.challenge_harness
integration = pre_enrollment_tests.integration


class TestOnlyCrash(RuntimeError):
    __test__ = False


@pytest.fixture
def issuance(integration, monkeypatch):
    item = integration
    monkeypatch.setattr(
        authorization,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    monkeypatch.setattr(authorization, "_utc_now", lambda: NOW)
    calls = []
    item.available_signer_ids = set(item.authority.keys)

    def test_only_sign(service, payload_raw):
        # This monkeypatch is confined to the fixture: the public production
        # issuance API has no signer argument, service endpoint or key input.
        issuer.require_production_enrollment_issuer(service)
        authorization._require_signing_reservation(service, payload_raw)
        payload = json.loads(payload_raw)
        assert canonical_json_bytes(payload) == payload_raw
        assert set(payload) == PACKAGE_FIELDS
        calls.append(payload_raw)
        selected = sorted(item.available_signer_ids & item.authority.keys.keys())[:2]
        if len(selected) != 2:
            raise issuer.ProductionEnrollmentIssuerError("TEST_ONLY_QUORUM_UNAVAILABLE")
        message = PDSA_DOMAIN + hashlib.sha256(payload_raw).digest()
        return [
            {
                "key_id": key_id,
                "algorithm": "Ed25519",
                "signature_hex": item.authority.keys[key_id].sign(message).hex(),
            }
            for key_id in selected
        ]

    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext,
        "sign_enrollment_authorization",
        test_only_sign,
    )
    item.sign_calls = calls
    item.test_only_sign = test_only_sign
    return item


def _issue(item, accepted=None):
    if accepted is None:
        accepted = _authenticate(item)
    return authorization.issue_production_pdsa_enrollment_authorization(accepted)


def _retry(item, service=None, **changes):
    return authorization.retry_production_pdsa_enrollment_authorization(
        item.authority.issuer if service is None else service,
        **({"request_raw": item.request.canonical_bytes, "challenge_raw": item.pdsa_raw} | changes),
    )


def _rows(item):
    with sqlite3.connect(item.authority.store.path) as db:
        db.row_factory = sqlite3.Row
        challenges = db.execute("SELECT * FROM pdsa_challenges").fetchall()
        records = db.execute("SELECT * FROM pdsa_authorization_issuances").fetchall()
    return challenges, records


def _assert_pending(item, state):
    challenges, records = _rows(item)
    assert len(challenges) == 1 and challenges[0]["state"] == "ISSUED"
    assert challenges[0]["request_raw"] is None
    assert challenges[0]["request_digest"] is None
    assert challenges[0]["receipt_raw"] is None
    assert len(records) == 1 and records[0]["state"] == state
    assert records[0]["request_raw"] == item.request.canonical_bytes
    assert records[0]["pre_enrollment_request_digest_sha256"] == item.request.digest_sha256
    assert records[0]["provisioning_subject_id"]
    assert records[0]["enrollment_reference"]
    assert records[0]["authorized_signer_ids_raw"] == canonical_json_bytes(
        sorted(item.authority.context.pdsa_keys)
    )
    assert records[0]["required_threshold"] == 2
    if state == "RESERVED":
        assert records[0]["signer_ids_raw"] is None
    else:
        assert records[0]["signer_ids_raw"] == canonical_json_bytes(
            [entry["key_id"] for entry in json.loads(records[0]["package_raw"])["signatures"]]
        )
    assert _retry(item) is None
    assert (
        authorization.lookup_production_pdsa_enrollment_authorization(
            item.authority.issuer, enrollment_reference=records[0]["enrollment_reference"]
        )
        is None
    )
    return records[0]


def _assert_committed(item, raw):
    value = json.loads(raw)
    payload = value["payload"]
    digest = hashlib.sha256(raw).hexdigest()
    assert canonical_json_bytes(value) == raw
    challenges, records = _rows(item)
    assert len(challenges) == len(records) == 1
    retained_challenge, record = challenges[0], records[0]
    assert retained_challenge["state"] == "CONSUMED"
    assert retained_challenge["request_raw"] == item.request.canonical_bytes
    assert retained_challenge["request_digest"] == item.request.digest_sha256
    assert record["state"] == "COMMITTED"
    assert record["pdsa_challenge_id"] == item.pdsa.document["payload"]["challenge_id"]
    assert record["pdsa_challenge_digest_sha256"] == item.pdsa.digest_sha256
    assert record["request_raw"] == item.request.canonical_bytes
    assert record["pre_enrollment_request_digest_sha256"] == item.request.digest_sha256
    assert record["payload_raw"] == canonical_json_bytes(payload)
    assert record["package_raw"] == raw
    assert record["pdsa_package_digest_sha256"] == digest
    assert record["authorized_signer_ids_raw"] == canonical_json_bytes(
        sorted(item.authority.context.pdsa_keys)
    )
    assert record["required_threshold"] == 2
    assert record["signer_ids_raw"] == canonical_json_bytes(
        [entry["key_id"] for entry in value["signatures"]]
    )
    for name in (
        "provisioning_subject_id",
        "enrollment_reference",
        "issued_at_utc",
        "expires_at_utc",
    ):
        assert record[name] == payload[name]
    assert _retry(item) == raw
    assert (
        authorization.lookup_production_pdsa_enrollment_authorization(
            item.authority.issuer, enrollment_reference=payload["enrollment_reference"]
        )
        == raw
    )
    return payload


def _restart(item):
    old = item.authority.issuer
    old.close()
    fresh = item.authority.reopen()
    assert fresh is not old
    assert fresh.pdsa_store is not item.authority.store
    assert fresh.tpm_store is not item.pending
    item.authority.issuer = fresh
    item.authority.store = fresh.pdsa_store
    item.pending = fresh.tpm_store
    item.arguments["challenge_store"] = fresh.pdsa_store
    item.arguments["pending"] = fresh.tpm_store
    return fresh


def _another_authenticated_request(item):
    # New request PoP and AK CertifyCreation are generated: this is a genuinely
    # authenticated conflicting request, rather than a copied JSON capability.
    changed = _changed_request(item, request_nonce_hex="ef" * 32)
    attest = _attest(
        item.names["pre_enrollment"],
        b"\x66" * 32,
        custody.pre_enrollment_custody_qualifying_data(changed),
    )
    evidence = custody.make_pre_enrollment_custody_evidence(
        request=changed,
        target_tpm_projection=item.projection,
        subject_tpmt_public=item.publics["pre_enrollment"],
        subject_name=item.names["pre_enrollment"],
        creation_hash=b"\x66" * 32,
        attest=attest,
        signature=_signature(item.keys["ak"], attest),
    )
    accepted = _authenticate(
        item,
        request_raw=changed.canonical_bytes,
        signature=item.key.sign_request(changed, production_trust_context=item.authority.context),
        custody_evidence_raw=evidence.canonical_bytes,
    )
    return changed, accepted


def test_real_crypto_exact_frozen_payload_and_atomic_terminal_record(issuance):
    item = issuance
    accepted = _authenticate(item)
    raw = _issue(item, accepted)
    payload = _assert_committed(item, raw)
    assert len(payload) == 23 and set(payload) == PACKAGE_FIELDS
    request = item.request.document
    for package_field, request_field in (
        ("pdsa_challenge_id", "pdsa_challenge_id"),
        ("pdsa_challenge_digest_sha256", "pdsa_challenge_digest_sha256"),
        ("verified_tpm_exchange_reference", "verified_tpm_exchange_reference"),
        ("verified_tpm_public_projection_id", "verified_tpm_public_projection_id"),
        ("target_tpm_ek_public_digest", "ek_public_digest"),
        ("target_tpm_ak_public_digest", "ak_public_digest"),
        (
            "pre_enrollment_public_key_algorithm_profile",
            "pre_enrollment_public_key_algorithm_profile",
        ),
        (
            "pre_enrollment_public_key_fingerprint_sha256",
            "pre_enrollment_public_key_fingerprint_sha256",
        ),
        ("release_policy_digest_sha256", "release_policy_digest_sha256"),
        ("release_policy_generation", "release_policy_generation"),
        ("product_profile", "product_profile"),
    ):
        assert payload[package_field] == request[request_field]
    assert payload["pre_enrollment_request_digest_sha256"] == item.request.digest_sha256
    assert payload["environment"] == "PRODUCTION"
    assert payload["pdsa_trust_domain"] == request["pdsa_trust_domain"]
    assert payload["authorization_generation"] == 1
    assert payload["authorization_version"] == 1
    assert payload["lineage_generation"] == 1
    assert payload["predecessor_package_digest_or_null"] is None
    assert payload["issued_at_utc"] == "2026-10-06T12:00:00Z"
    assert payload["expires_at_utc"] == "2026-10-07T12:00:00Z"
    assert not {"provisioning_operation_id", "account_id", "membership", "successor_key"} & set(
        payload
    )
    signatures = json.loads(raw)["signatures"]
    assert len(signatures) == 2
    assert [entry["key_id"] for entry in signatures] == sorted(item.authority.keys)[:2]
    message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
    for signature in signatures:
        item.authority.context.pdsa_keys[signature["key_id"]].verify(
            bytes.fromhex(signature["signature_hex"]), message
        )
    assert authorization.issue_production_pdsa_enrollment_authorization(accepted) == raw
    assert len(item.sign_calls) == 1


def test_psub_and_reference_are_independent_cs_random_uuidv7(issuance, monkeypatch):
    item = issuance
    draws = []
    original = authorization.secrets.randbits

    def observe(bits):
        value = original(bits)
        draws.append((bits, value))
        return value

    monkeypatch.setattr(authorization.secrets, "randbits", observe)
    raw = _issue(item)
    payload = _assert_committed(item, raw)
    assert [bits for bits, _ in draws] == [12, 62, 12, 62]
    for field, prefix in (("provisioning_subject_id", "psub_"), ("enrollment_reference", "penr_")):
        value = payload[field]
        assert value.startswith(prefix)
        parsed = uuid.UUID(value[len(prefix) :])
        assert parsed.version == 7 and parsed.variant == uuid.RFC_4122
        assert str(parsed) == value[len(prefix) :]
        assert parsed.int >> 80 == int(NOW.timestamp() * 1000)
    assert payload["provisioning_subject_id"][5:] != payload["enrollment_reference"][5:]
    assert payload["provisioning_subject_id"] not in item.request.canonical_bytes.decode()
    assert _retry(item) == raw
    assert len(draws) == 4


@pytest.mark.parametrize("microsecond", [789123, 999999])
def test_uuidv7_retains_reservation_milliseconds_while_wire_time_uses_seconds(
    issuance, monkeypatch, microsecond
):
    item = issuance
    reservation_now = NOW.replace(microsecond=microsecond)
    expected_ms = 1_791_288_000_000 + microsecond // 1000
    monkeypatch.setattr(authorization, "_utc_now", lambda: reservation_now)
    payload = _assert_committed(item, _issue(item))
    assert payload["issued_at_utc"] == "2026-10-06T12:00:00Z"
    for field in ("provisioning_subject_id", "enrollment_reference"):
        assert uuid.UUID(payload[field][5:]).int >> 80 == expected_ms


def test_both_uuidv7_ids_capture_one_instant_across_second_rollover(issuance, monkeypatch):
    item = issuance
    captured = NOW.replace(microsecond=999999)
    clock_reads = []
    minted = []
    original = authorization._mint_uuidv7

    def clock():
        value = captured if not minted else NOW + timedelta(seconds=1)
        clock_reads.append(value)
        return value

    def mint(prefix, timestamp):
        value = original(prefix, timestamp)
        minted.append(value)
        return value

    monkeypatch.setattr(authorization, "_utc_now", clock)
    monkeypatch.setattr(authorization, "_mint_uuidv7", mint)
    payload = _assert_committed(item, _issue(item))
    assert clock_reads[-1] == NOW + timedelta(seconds=1)
    assert payload["issued_at_utc"] == "2026-10-06T12:00:00Z"
    assert len(minted) == 2
    assert {uuid.UUID(value[5:]).int >> 80 for value in minted} == {1_791_288_000_999}


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)], ids=["K1+K2", "K1+K3", "K2+K3"])
def test_any_two_authorized_signers_issue_and_verify(issuance, monkeypatch, pair):
    item = issuance
    keys = sorted(item.authority.keys)
    item.available_signer_ids = {keys[index] for index in pair}
    accepted = _authenticate(item)
    raw = _issue(item, accepted)
    assert [entry["key_id"] for entry in json.loads(raw)["signatures"]] == sorted(
        item.available_signer_ids
    )
    monkeypatch.setattr(
        trust,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    verifier = ProductionProvisioningPackageVerifier(item.authority.context)
    assert (
        verifier.verify(
            raw,
            expected_device_key=item.request.document[
                "pre_enrollment_public_key_fingerprint_sha256"
            ],
            now=NOW,
        ).canonical_package
        == raw
    )
    _assert_committed(item, raw)


def test_only_one_available_signer_keeps_reservation_unpublished(issuance):
    item = issuance
    item.available_signer_ids = {sorted(item.authority.keys)[2]}
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="QUORUM_UNAVAILABLE"):
        _issue(item)
    _assert_pending(item, "RESERVED")


def test_signed_pair_survives_issuer_restart_and_restored_signer(issuance, monkeypatch):
    item = issuance
    keys = sorted(item.authority.keys)
    item.available_signer_ids = set(keys[1:])
    with monkeypatch.context() as patch:

        def crash_before_final_commit(value):
            raise TestOnlyCrash("SIGNED before final commit")

        patch.setattr(authorization, "_finalize_issuance", crash_before_final_commit)
        with pytest.raises(TestOnlyCrash):
            _issue(item)
    retained = _assert_pending(item, "SIGNED")["package_raw"]
    assert [entry["key_id"] for entry in json.loads(retained)["signatures"]] == keys[1:]
    _restart(item)
    item.available_signer_ids = set(keys)
    assert _issue(item) == retained
    assert len(item.sign_calls) == 1
    _assert_committed(item, retained)


def test_other_valid_quorum_cannot_replace_pair_after_signed_retention(issuance):
    item = issuance
    accepted = _authenticate(item)
    reservation = authorization._reserve_issuance(accepted)
    first = canonical_json_bytes(
        {
            "payload": json.loads(reservation.payload_raw),
            "signatures": item.test_only_sign(item.authority.issuer, reservation.payload_raw),
        }
    )
    authorization._persist_signed_package(accepted, first)
    _assert_pending(item, "SIGNED")
    item.available_signer_ids = set(sorted(item.authority.keys)[1:])
    alternate = canonical_json_bytes(
        {
            "payload": json.loads(reservation.payload_raw),
            "signatures": item.test_only_sign(item.authority.issuer, reservation.payload_raw),
        }
    )
    assert alternate != first
    authorization._verify_package_signatures(alternate, item.authority.context)
    with pytest.raises(ValueError, match="SIGNATURE_RETRY_CONFLICT"):
        authorization._persist_signed_package(accepted, alternate)
    assert _assert_pending(item, "SIGNED")["package_raw"] == first
    assert _issue(item, accepted) == first


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        (
            "authorized_signer_ids_raw",
            canonical_json_bytes(["UNKNOWN_1", "UNKNOWN_2", "UNKNOWN_3"]),
        ),
        ("production_trust_raw", canonical_json_bytes({"ceremony_id": "DIFFERENT"})),
    ],
)
def test_frozen_reservation_authority_tampering_blocks_signing(issuance, column, replacement):
    item = issuance
    accepted = _authenticate(item)
    authorization._reserve_issuance(accepted)
    with sqlite3.connect(item.authority.store.path) as db:
        # Identifiers are restricted to the two literal test cases above.
        statement = {
            "authorized_signer_ids_raw": "UPDATE pdsa_authorization_issuances SET authorized_signer_ids_raw=?",
            "production_trust_raw": "UPDATE pdsa_authorization_issuances SET production_trust_raw=?",
        }[column]
        db.execute(statement, (replacement,))
    with pytest.raises(ValueError):
        item.authority.issuer.sign_enrollment_authorization(_rows(item)[1][0]["payload_raw"])
    assert item.sign_calls == []
    assert _rows(item)[0][0]["state"] == "ISSUED"
    assert _rows(item)[1][0]["state"] == "RESERVED"


@pytest.mark.parametrize("changed_field", ["public_key", "ceremony_id", "release_digest"])
def test_well_formed_reservation_trust_mismatch_fails_closed(issuance, changed_field):
    item = issuance
    accepted = _authenticate(item)
    reservation = authorization._reserve_issuance(accepted)
    record = _assert_pending(item, "RESERVED")
    snapshot = json.loads(record["production_trust_raw"])
    if changed_field == "public_key":
        # The exact same eligible IDs/ceremony/release cannot hide key rotation.
        snapshot["pdsa_public_keys"][0]["public_key_hex"] = "12" * 32
    elif changed_field == "ceremony_id":
        snapshot["ceremony_id"] += "-different"
    else:
        snapshot["release_policy_digest_sha256"] = "12" * 32
    with sqlite3.connect(item.authority.store.path) as db:
        db.execute(
            "UPDATE pdsa_authorization_issuances SET production_trust_raw=?",
            (canonical_json_bytes(snapshot),),
        )
    with pytest.raises(ValueError, match="EXACT_SIGNING_RESERVATION_REQUIRED"):
        authorization._require_signing_reservation(item.authority.issuer, reservation.payload_raw)
    with pytest.raises(ValueError, match="TARGET_MISMATCH"):
        _issue(item, accepted)
    assert item.sign_calls == []
    assert _rows(item)[0][0]["state"] == "ISSUED"


@pytest.mark.parametrize("millis", [-1, 1 << 48, 1.0, True, None])
def test_uuidv7_invalid_timestamp_fails_before_randomness(monkeypatch, millis):
    def unexpected_randomness(bits):
        raise AssertionError("invalid UUID timestamp must fail before CSPRNG")

    monkeypatch.setattr(authorization.secrets, "randbits", unexpected_randomness)
    with pytest.raises(authorization.PDSAAuthorizationError, match="INVALID_PACKAGE_TIMESTAMP"):
        authorization._mint_uuidv7("psub_", millis)


@pytest.mark.parametrize(
    "name",
    [
        "provisioning_subject_id",
        "enrollment_reference",
        "context",
        "signer",
        "signatures",
        "keys",
        "signer_ids",
        "threshold",
        "now",
        "endpoint",
        "callback",
    ],
)
def test_transport_cannot_choose_identity_trust_or_signing_authority(issuance, name):
    accepted = _authenticate(issuance)
    assert tuple(
        inspect.signature(authorization.issue_production_pdsa_enrollment_authorization).parameters
    ) == ("value",)
    with pytest.raises(TypeError):
        authorization.issue_production_pdsa_enrollment_authorization(accepted, **{name: "CALLER"})
    assert _rows(issuance)[1] == []


@pytest.mark.parametrize("kind", ["plain_json", "shape", "clone", "new", "subclass"])
def test_only_real_authenticated_capability_can_reserve(issuance, kind):
    item = issuance
    accepted = _authenticate(item)
    if kind == "plain_json":
        fake = json.loads(item.request.canonical_bytes)
    elif kind == "shape":
        fake = SimpleNamespace(
            request_raw=accepted.request_raw,
            challenge_raw=accepted.challenge_raw,
            context=accepted.context,
        )
    elif kind == "clone":
        fake = copy.copy(accepted)
    elif kind == "new":
        fake = object.__new__(production.AuthenticatedProductionPreEnrollment)
    else:

        class Forged(production.AuthenticatedProductionPreEnrollment):
            pass

        fake = object.__new__(Forged)
    with pytest.raises(
        production.ProductionPreEnrollmentError,
        match="AUTHENTICATED_PRODUCTION_PRE_ENROLLMENT_REQUIRED",
    ):
        authorization.issue_production_pdsa_enrollment_authorization(fake)
    assert _rows(item)[1] == []
    assert item.sign_calls == []


def test_exact_lost_response_retry_after_new_factory_needs_no_consumed_capability(issuance):
    item = issuance
    accepted = _authenticate(item)
    raw = _issue(item, accepted)
    payload = json.loads(raw)["payload"]
    service = _restart(item)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER"):
        authorization.issue_production_pdsa_enrollment_authorization(accepted)
    with pytest.raises(challenge.PDSAChallengeError, match="CONSUMED"):
        _authenticate(item)
    assert _retry(item, service) == raw
    assert (
        authorization.lookup_production_pdsa_enrollment_authorization(
            service, enrollment_reference=payload["enrollment_reference"]
        )
        == raw
    )
    assert _assert_committed(item, raw) == payload
    assert len(item.sign_calls) == 1


def test_restart_expired_terminal_retry_retrieves_exact_bytes_without_renewing_authority(
    issuance, monkeypatch
):
    item = issuance
    raw = _issue(item)
    payload = json.loads(raw)["payload"]
    service = _restart(item)
    expired = NOW + timedelta(days=8)
    for module in (authorization, challenge, custody):
        monkeypatch.setattr(module, "_utc_now", lambda: expired)

    def forbid_live_authority(*args, **kwargs):
        raise AssertionError("terminal retrieval must not authenticate, mint, or sign")

    with monkeypatch.context() as patch:
        for module in (authorization, challenge, production, issuer, trust):
            patch.setattr(module, "require_current_production_trust_context", forbid_live_authority)
        patch.setattr(production, "require_authenticated_pre_enrollment", forbid_live_authority)
        patch.setattr(authorization, "_mint_uuidv7", forbid_live_authority)
        patch.setattr(
            issuer.ProductionEnrollmentIssuerContext,
            "sign_enrollment_authorization",
            forbid_live_authority,
        )
        assert _retry(item, service) == raw
        assert (
            authorization.lookup_production_pdsa_enrollment_authorization(
                service, enrollment_reference=payload["enrollment_reference"]
            )
            == raw
        )
    assert len(item.sign_calls) == 1
    assert _assert_committed(item, raw) == payload
    # TEST_ONLY current-release boundary only; the actual production verifier
    # still rejects the retained package's expired authority at the caller time.
    monkeypatch.setattr(
        trust,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    verifier = ProductionProvisioningPackageVerifier(item.authority.context)
    with pytest.raises(ProvisioningError, match="PACKAGE_EXPIRED"):
        verifier.verify(
            raw,
            expected_device_key=item.request.document[
                "pre_enrollment_public_key_fingerprint_sha256"
            ],
            now=expired,
        )


def test_different_authenticated_request_cannot_steal_reserved_or_committed_identity(
    issuance, monkeypatch
):
    item = issuance
    first = _authenticate(item)
    other, conflicting = _another_authenticated_request(item)
    original = authorization._reserve_issuance

    def crash_after_reservation(accepted):
        original(accepted)
        raise TestOnlyCrash("after reservation")

    with monkeypatch.context() as patch:
        patch.setattr(authorization, "_reserve_issuance", crash_after_reservation)
        with pytest.raises(TestOnlyCrash):
            _issue(item, first)
    reservation = _assert_pending(item, "RESERVED")
    with pytest.raises(ValueError, match="REPLAY_CONFLICT"):
        _issue(item, conflicting)
    assert _rows(item)[1][0]["provisioning_subject_id"] == reservation["provisioning_subject_id"]
    raw = _issue(item, first)
    assert (
        json.loads(raw)["payload"]["provisioning_subject_id"]
        == reservation["provisioning_subject_id"]
    )
    with pytest.raises(ValueError, match="REPLAY_CONFLICT"):
        _retry(item, request_raw=other.canonical_bytes)
    _assert_committed(item, raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_nonce_hex", "cd" * 32),
        ("pre_enrollment_public_key_fingerprint_sha256", "00" * 32),
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", 2),
        ("verified_tpm_exchange_reference", "00" * 32),
        ("verified_tpm_public_projection_id", "00" * 32),
        ("ek_public_digest", "00" * 32),
        ("ak_public_digest", "00" * 32),
    ],
)
def test_exact_retry_rejects_changed_raw_key_release_or_tpm_targets(issuance, field, value):
    item = issuance
    raw = _issue(item)
    changed = item.request.document | {field: value}
    if field == "pre_enrollment_public_key_fingerprint_sha256":
        public = (
            ec.generate_private_key(ec.SECP256R1())
            .public_key()
            .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        )
        changed["pre_enrollment_public_key_canonical_bytes"] = public.hex()
        changed[field] = hashlib.sha256(public).hexdigest()
    request = PreEnrollmentRequestV1.from_mapping(changed)
    with pytest.raises(ValueError, match="REPLAY_CONFLICT"):
        _retry(item, request_raw=request.canonical_bytes)
    _assert_committed(item, raw)


@pytest.mark.parametrize("field", ["request_raw", "challenge_raw"])
def test_retry_requires_exact_canonical_raw_bytes(issuance, field):
    item = issuance
    raw = _issue(item)
    source = item.request.canonical_bytes if field == "request_raw" else item.pdsa_raw
    with pytest.raises(ValueError):
        _retry(item, **{field: source + b" "})
    _assert_committed(item, raw)


@pytest.mark.parametrize(
    "cut",
    [
        "after_reservation",
        "during_signing",
        "after_signing",
        "before_commit",
        "after_commit_before_response",
    ],
)
def test_durable_crash_cut_points_resume_exact_identity_after_restart(issuance, monkeypatch, cut):
    item = issuance
    accepted = _authenticate(item)
    retained_result = []
    with monkeypatch.context() as patch:
        if cut == "after_reservation":
            original = authorization._reserve_issuance

            def interrupt(value):
                original(value)
                raise TestOnlyCrash(cut)

            patch.setattr(authorization, "_reserve_issuance", interrupt)
        elif cut == "during_signing":

            def interrupt(service, payload_raw):
                item.test_only_sign(service, payload_raw)
                raise TestOnlyCrash(cut)

            patch.setattr(
                issuer.ProductionEnrollmentIssuerContext, "sign_enrollment_authorization", interrupt
            )
        elif cut == "after_signing":

            def interrupt(value, package_raw):
                retained_result.append(package_raw)
                raise TestOnlyCrash(cut)

            patch.setattr(authorization, "_persist_signed_package", interrupt)
        elif cut == "before_commit":

            def interrupt(value):
                raise TestOnlyCrash(cut)

            patch.setattr(authorization, "_finalize_issuance", interrupt)
        else:
            original = authorization._finalize_issuance

            def interrupt(value):
                retained_result.append(original(value))
                raise TestOnlyCrash(cut)

            patch.setattr(authorization, "_finalize_issuance", interrupt)
        with pytest.raises(TestOnlyCrash):
            _issue(item, accepted)
    if cut == "after_commit_before_response":
        payload = _assert_committed(item, retained_result[0])
    else:
        record = _assert_pending(item, "SIGNED" if cut == "before_commit" else "RESERVED")
        payload = json.loads(record["payload_raw"])
    service = _restart(item)
    if cut == "after_commit_before_response":
        raw = _retry(item, service)
        assert raw == retained_result[0]
    else:
        assert _retry(item, service) is None

        def reject_remint(*args, **kwargs):
            raise AssertionError("durably reserved identity must never be reminted")

        with monkeypatch.context() as patch:
            patch.setattr(authorization, "_mint_uuidv7", reject_remint)
            raw = _issue(item)
    assert _assert_committed(item, raw) == payload
    if retained_result:
        assert raw == retained_result[0]


@pytest.mark.parametrize("cut", ["before_psub_mint", "after_psub_mint", "after_reference_mint"])
def test_partial_identity_mint_never_publishes_or_consumes_challenge(issuance, monkeypatch, cut):
    item = issuance
    accepted = _authenticate(item)
    original = authorization._mint_uuidv7
    minted = []

    def interrupt(prefix, issued):
        if cut == "before_psub_mint":
            raise TestOnlyCrash(cut)
        value = original(prefix, issued)
        minted.append(value)
        if (cut == "after_psub_mint" and len(minted) == 1) or (
            cut == "after_reference_mint" and len(minted) == 2
        ):
            raise TestOnlyCrash(cut)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(authorization, "_mint_uuidv7", interrupt)
        with pytest.raises(TestOnlyCrash):
            _issue(item, accepted)
    challenges, records = _rows(item)
    assert challenges[0]["state"] == "ISSUED"
    assert records == []
    assert item.sign_calls == []
    _restart(item)
    _assert_committed(item, _issue(item))


def test_external_signing_never_holds_sqlite_write_lock(issuance, monkeypatch):
    item = issuance

    def test_only_sign_without_lock(service, payload_raw):
        with sqlite3.connect(service.pdsa_store.path, timeout=0) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM pdsa_authorization_issuances").fetchone()
            assert row == ("RESERVED",)
            assert db.execute("SELECT state FROM pdsa_challenges").fetchone() == ("ISSUED",)
            db.rollback()
        return item.test_only_sign(service, payload_raw)

    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext,
        "sign_enrollment_authorization",
        test_only_sign_without_lock,
    )
    _assert_committed(item, _issue(item))


def test_concurrent_authenticated_exact_requests_share_one_durable_identity_and_result(issuance):
    item = issuance
    accepted = [_authenticate(item), _authenticate(item)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(authorization.issue_production_pdsa_enrollment_authorization, accepted)
        )
    assert results[0] == results[1]
    _assert_committed(item, results[0])


def test_concurrent_commit_between_initial_retry_and_live_challenge_guard_returns_winner(
    issuance, monkeypatch
):
    item = issuance
    accepted = _authenticate(item)
    original = authorization.require_verified_issued_challenge
    competing_result = []
    racing = False

    def competing_commit_then_guard(*args, **kwargs):
        nonlocal racing
        if not racing:
            racing = True
            competing_result.append(_issue(item, accepted))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        authorization, "require_verified_issued_challenge", competing_commit_then_guard
    )
    raw = _issue(item, accepted)
    assert competing_result == [raw]
    _assert_committed(item, raw)
    assert len(item.sign_calls) == 1


def test_concurrent_commit_before_external_signer_reservation_guard_returns_winner(
    issuance, monkeypatch
):
    item = issuance
    accepted = _authenticate(item)
    competing_result = []
    racing = False

    def competing_commit_then_sign(service, payload_raw):
        nonlocal racing
        if not racing:
            racing = True
            competing_result.append(_issue(item, accepted))
        # The genuine internal signing guard rejects signing a COMMITTED row.
        # The issuance wrapper must safely retrieve the already-retained winner.
        return item.test_only_sign(service, payload_raw)

    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext,
        "sign_enrollment_authorization",
        competing_commit_then_sign,
    )
    raw = _issue(item, accepted)
    assert competing_result == [raw]
    _assert_committed(item, raw)
    assert len(item.sign_calls) == 1


def test_consumed_acceptance_receipt_cannot_be_silently_promoted_to_package(issuance):
    item = issuance
    accepted = _authenticate(item)
    receipt = item.authority.store.consume_authenticated_request(accepted)
    with pytest.raises(ValueError):
        _issue(item, accepted)
    assert (
        item.authority.store.retry_exact_accepted(
            request_raw=item.request.canonical_bytes, challenge_raw=item.pdsa_raw
        )
        == receipt
    )
    assert _rows(item)[1] == []
    assert item.sign_calls == []


def test_reserved_package_cannot_be_consumed_as_legacy_acceptance_receipt(issuance, monkeypatch):
    item = issuance
    accepted = _authenticate(item)
    original = authorization._reserve_issuance

    def interrupt(value):
        original(value)
        raise TestOnlyCrash("reserved")

    with monkeypatch.context() as patch:
        patch.setattr(authorization, "_reserve_issuance", interrupt)
        with pytest.raises(TestOnlyCrash):
            _issue(item, accepted)
    with pytest.raises(ValueError):
        item.authority.store.consume_authenticated_request(accepted)
    _assert_pending(item, "RESERVED")
    _assert_committed(item, _issue(item, accepted))


@pytest.mark.parametrize("kind", ["copied", "reconstructed"])
def test_transplanted_database_does_not_transfer_package_issuer_authority(issuance, tmp_path, kind):
    item = issuance
    raw = _issue(item)
    target = tmp_path / f"TEST_ONLY_{kind}.sqlite3"
    if kind == "copied":
        _copy_database(item.authority.store.path, target)
    else:
        with (
            sqlite3.connect(item.authority.store.path) as source,
            sqlite3.connect(target) as destination,
        ):
            for statement in source.iterdump():
                destination.execute(statement)
    mechanics = challenge.PDSAChallengeStore(target)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER"):
        authorization.retry_production_pdsa_enrollment_authorization(
            mechanics, request_raw=item.request.canonical_bytes, challenge_raw=item.pdsa_raw
        )
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER"):
        authorization.lookup_production_pdsa_enrollment_authorization(
            mechanics, enrollment_reference=json.loads(raw)["payload"]["enrollment_reference"]
        )
    _assert_committed(item, raw)


def test_closed_or_forged_issuer_context_cannot_retrieve_retained_packages(issuance):
    item = issuance
    raw = _issue(item)
    fake = object.__new__(issuer.ProductionEnrollmentIssuerContext)
    with pytest.raises(
        issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER_CONTEXT_REQUIRED"
    ):
        _retry(item, fake)
    item.authority.issuer.close()
    with pytest.raises(
        issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER_CONTEXT_REQUIRED"
    ):
        _retry(item)
    service = _restart(item)
    assert _retry(item, service) == raw


def test_different_factory_bundle_cannot_issue_original_acceptance(issuance):
    item = issuance
    accepted = _authenticate(item)
    other = item.authority.reopen()
    assert other.pdsa_store is not item.authority.store
    assert other.trust is item.authority.context
    item.authority.issuer.close()
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PRODUCTION_ISSUER"):
        _issue(item, accepted)
    assert _rows(item)[1] == []
    other.close()


def test_live_tpm_endorsement_expiry_during_external_signing_prevents_final_consumption(
    issuance, monkeypatch
):
    item = issuance

    def sign_then_expire(service, payload_raw):
        result = item.test_only_sign(service, payload_raw)
        expired = NOW + timedelta(days=1)
        monkeypatch.setattr(authorization, "_utc_now", lambda: expired)
        monkeypatch.setattr(challenge, "_utc_now", lambda: expired)
        monkeypatch.setattr(custody, "_utc_now", lambda: expired)
        return result

    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext, "sign_enrollment_authorization", sign_then_expire
    )
    with pytest.raises(ValueError, match="EXPIRED"):
        _issue(item)
    challenges, records = _rows(item)
    assert challenges[0]["state"] == "ISSUED"
    assert records[0]["state"] != "COMMITTED"
    assert _retry(item) is None


def test_final_transaction_rolls_back_both_consumption_and_package_publication(
    issuance, monkeypatch
):
    item = issuance
    original = challenge.PDSAChallengeStore._connect
    connections = []

    @contextmanager
    def test_only_rollback_connection(store):
        with original(store) as db:
            connections.append(db)
            # Commit is interrupted only in the final transaction, recognized
            # through its already-durable signed reservation.
            state = db.execute("SELECT state FROM pdsa_authorization_issuances").fetchone()
            if state is not None and state[0] == "SIGNED":

                class CommitCrash:
                    def execute(self, *args, **kwargs):
                        return db.execute(*args, **kwargs)

                    def commit(self):
                        raise TestOnlyCrash("before SQLite commit")

                    def __getattr__(self, name):
                        return getattr(db, name)

                try:
                    yield CommitCrash()
                finally:
                    db.rollback()
            else:
                yield db

    with monkeypatch.context() as patch:
        patch.setattr(challenge.PDSAChallengeStore, "_connect", test_only_rollback_connection)
        with pytest.raises(TestOnlyCrash):
            _issue(item)
    assert connections
    record = _assert_pending(item, "SIGNED")
    signed_raw = record["package_raw"]
    _restart(item)
    raw = _issue(item)
    assert raw == signed_raw
    _assert_committed(item, raw)


@pytest.mark.parametrize(
    "kind",
    [
        "one_signature",
        "three_signatures",
        "duplicate_signer",
        "wrong_signature",
        "foreign_signer",
        "unordered_signers",
    ],
)
def test_invalid_quorum_service_output_never_consumes_or_publishes(issuance, monkeypatch, kind):
    item = issuance

    def faulty_test_only_sign(service, payload_raw):
        signatures = item.test_only_sign(service, payload_raw)
        if kind == "one_signature":
            return signatures[:1]
        if kind == "duplicate_signer":
            return [signatures[0], signatures[0]]
        if kind == "wrong_signature":
            signatures[0]["signature_hex"] = "00" * 64
            return signatures
        if kind == "foreign_signer":
            signatures[0]["key_id"] = "TEST_ONLY_FOREIGN_AUTHORITY"
            return signatures
        if kind == "unordered_signers":
            return list(reversed(signatures))
        key_id = sorted(item.authority.keys)[2]
        third = {
            "key_id": key_id,
            "algorithm": "Ed25519",
            "signature_hex": item.authority.keys[key_id]
            .sign(PDSA_DOMAIN + hashlib.sha256(payload_raw).digest())
            .hex(),
        }
        return signatures + [third]

    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext,
        "sign_enrollment_authorization",
        faulty_test_only_sign,
    )
    with pytest.raises(ValueError):
        _issue(item)
    _assert_pending(item, "RESERVED")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pdsa_challenge_id", "pchal_019ba13c-5c00-7000-8000-000000000001"),
        ("pdsa_challenge_digest_sha256", "00" * 32),
        ("pre_enrollment_request_digest_sha256", "00" * 32),
        ("verified_tpm_exchange_reference", "00" * 32),
        ("verified_tpm_public_projection_id", "00" * 32),
        ("target_tpm_ek_public_digest", "00" * 32),
        ("target_tpm_ak_public_digest", "00" * 32),
        ("pre_enrollment_public_key_algorithm_profile", "UNSUPPORTED_PROFILE"),
        ("pre_enrollment_public_key_fingerprint_sha256", "00" * 32),
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", 2),
    ],
)
def test_reserved_target_tampering_is_rechecked_against_live_capability(
    issuance, monkeypatch, field, value
):
    item = issuance
    accepted = _authenticate(item)
    original = authorization._reserve_issuance

    def interrupt(verified):
        original(verified)
        raise TestOnlyCrash("after reservation")

    with monkeypatch.context() as patch:
        patch.setattr(authorization, "_reserve_issuance", interrupt)
        with pytest.raises(TestOnlyCrash):
            _issue(item, accepted)
    record = _assert_pending(item, "RESERVED")
    payload = json.loads(record["payload_raw"]) | {field: value}
    with sqlite3.connect(item.authority.store.path) as db:
        db.execute(
            "UPDATE pdsa_authorization_issuances SET payload_raw=?",
            (canonical_json_bytes(payload),),
        )
    with pytest.raises(ValueError, match="TARGET_MISMATCH"):
        _issue(item, accepted)
    challenges, records = _rows(item)
    assert challenges[0]["state"] == "ISSUED"
    assert records[0]["state"] == "RESERVED"
    assert records[0]["provisioning_subject_id"] == record["provisioning_subject_id"]
    assert item.sign_calls == []


@pytest.mark.parametrize(
    "column",
    [
        "package_raw",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
    ],
)
def test_retained_terminal_package_conflict_fails_closed(issuance, column):
    item = issuance
    raw = _issue(item)
    replacement = {
        "package_raw": raw + b" ",
        "pdsa_package_digest_sha256": "00" * 32,
        "provisioning_subject_id": "psub_019ba13c-5c00-7000-8000-000000000001",
        "enrollment_reference": "penr_019ba13c-5c00-7000-8000-000000000001",
    }[column]
    statement = {
        "package_raw": "UPDATE pdsa_authorization_issuances SET package_raw=?",
        "pdsa_package_digest_sha256": (
            "UPDATE pdsa_authorization_issuances SET pdsa_package_digest_sha256=?"
        ),
        "provisioning_subject_id": (
            "UPDATE pdsa_authorization_issuances SET provisioning_subject_id=?"
        ),
        "enrollment_reference": ("UPDATE pdsa_authorization_issuances SET enrollment_reference=?"),
    }[column]
    with sqlite3.connect(item.authority.store.path) as db:
        db.execute(statement, (replacement,))
    with pytest.raises(ValueError):
        _retry(item)
    assert len(item.sign_calls) == 1


def test_real_current_runtime_trust_guard_blocks_audit_only_package_issuance(issuance, monkeypatch):
    item = issuance
    accepted = _authenticate(item)
    monkeypatch.setattr(
        authorization,
        "require_current_production_trust_context",
        trust.require_current_production_trust_context,
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="CURRENT_RUNTIME"):
        _issue(item, accepted)
    challenges, records = _rows(item)
    assert challenges[0]["state"] == "ISSUED"
    assert not records or records[0]["state"] == "RESERVED"


def test_deployment_b_and_trust_b_cannot_retrieve_deployment_a_package(
    issuance, monkeypatch, tmp_path
):
    item = issuance
    raw = _issue(item)
    context_b = trust.verify_production_trust_for_audit(
        tmp_path / "TEST_ONLY_TRUST_B", verification_time=NOW
    )
    assert context_b is not item.authority.context
    with monkeypatch.context() as patch:
        patch.setattr(
            issuer,
            "_installed_service_configuration",
            lambda: issuer._InstalledIssuerConfiguration(
                state_directory=tmp_path / "TEST_ONLY_DEPLOYMENT_B", trust=context_b
            ),
        )
        service_b = issuer.open_installed_production_enrollment_issuer()
    with pytest.raises(challenge.PDSAChallengeError, match="UNKNOWN|RETAINED"):
        _retry(item, service_b)
    reference = json.loads(raw)["payload"]["enrollment_reference"]
    assert (
        authorization.lookup_production_pdsa_enrollment_authorization(
            service_b, enrollment_reference=reference
        )
        is None
    )
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        issuer.require_production_pdsa_store(
            item.authority.store, context=context_b, issuer=service_b
        )
    _assert_committed(item, raw)
    service_b.close()


def test_copy_replacing_original_database_cannot_retain_issuer_authority(issuance, tmp_path):
    item = issuance
    issued = _issue(item)
    original = item.authority.store.path
    replacement = tmp_path / "TEST_ONLY_REPLACEMENT.sqlite3"
    _copy_database(original, replacement)
    replacement.chmod(0o600)
    try:
        replacement.replace(original)
    except PermissionError:
        # Windows may refuse replacing a live SQLite database outright. That is
        # already a fail-closed outcome for this attack: the original database
        # remains in place and exact retry must still return the retained package.
        assert original.exists()
        assert replacement.exists()
        assert _retry(item) == issued
        return
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        _retry(item)


def test_shorter_tpm_expiry_is_rechecked_after_final_sqlite_writer_lock(issuance, monkeypatch):
    item = issuance
    original = challenge.PDSAChallengeStore._connect

    @contextmanager
    def delayed_final_transaction(store):
        with original(store) as db:
            state = db.execute("SELECT state FROM pdsa_authorization_issuances").fetchone()
            if state is not None and state[0] == "SIGNED":
                expired = NOW + timedelta(days=1)
                monkeypatch.setattr(authorization, "_utc_now", lambda: expired)
                monkeypatch.setattr(challenge, "_utc_now", lambda: expired)
                monkeypatch.setattr(custody, "_utc_now", lambda: expired)
            yield db

    monkeypatch.setattr(challenge.PDSAChallengeStore, "_connect", delayed_final_transaction)
    with pytest.raises(ValueError, match="EXPIRED"):
        _issue(item)
    challenges, records = _rows(item)
    assert challenges[0]["state"] == "ISSUED"
    assert challenges[0]["request_raw"] is None
    assert records[0]["state"] == "SIGNED"
    assert records[0]["package_raw"] is not None
    assert _retry(item) is None
