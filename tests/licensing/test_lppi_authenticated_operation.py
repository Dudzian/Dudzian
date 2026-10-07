"""TEST_ONLY native ABI simulation with real ACTIVE lineage and ECDSA verification.

The inherited #3071 fixture substitutes Windows NCrypt/TBS and current trust.
All production package, custody, lifecycle, durable reservation and verifier
guards remain active. These tests are not physical Windows qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import (
    lppi_authenticated_operation as operation,
    lppi_authority_key as authority_binding,
    pdsa_enrollment_authorization as authorization,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.external_provisioning import (
    ProductionMembershipSignerUnavailable,
    TestOnlyMembershipSigner,
)
from bot_core.licensing.pre_enrollment import P256_ORDER, PDSA_TRUST_DOMAIN
from deployment import (
    windows_production_lppi_authority as lifecycle,
    windows_production_lppi_operation as installed,
)
from tests.deployment.test_windows_cng_pre_enrollment import wide
from tests.licensing import test_lppi_authority_lifecycle as authority_tests
from tests.licensing.test_production_pre_enrollment import NOW

challenge_harness = authority_tests.challenge_harness
integration = authority_tests.integration
package = authority_tests.package
lppi = authority_tests.lppi

RESERVATION_NOW = NOW.replace(microsecond=123456)


def _document():
    millis = authorization._reservation_epoch_milliseconds(RESERVATION_NOW)
    value = (millis << 80) | (7 << 76) | (2 << 62) | 1
    return {
        "schema_version": 1,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "pdsa_package_digest_sha256": "ab" * 32,
        "provisioning_subject_id": "psub_019ba13c-5c00-7000-8000-000000000001",
        "enrollment_reference": "penr_019ba13c-5c00-7000-8000-000000000002",
        "provisioning_operation_id": "prvop_" + str(uuid.UUID(int=value)),
        "binding_generation": 1,
        "created_at_utc": RESERVATION_NOW.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }


def _state():
    return installed._read(installed._state_path())


def _low_s(private, message):
    signature = private.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(signature)
    return utils.encode_dss_signature(r, min(s, P256_ORDER - s))


def _nonminimal_der(signature):
    # Add a redundant zero to the first INTEGER while retaining the same r/s.
    assert signature[:3] == bytes((0x30, len(signature) - 2, 0x02))
    length = signature[3]
    return bytes((0x30, len(signature) - 1, 0x02, length + 1, 0)) + signature[4:]


@pytest.fixture
def active(lppi, monkeypatch):
    monkeypatch.setattr(installed, "_utc_now", lambda: RESERVATION_NOW)
    result = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    yield result
    result.close()


@pytest.fixture
def committed(active, lppi):
    value = installed.establish_installed_lppi_authenticated_operation(active)
    return SimpleNamespace(value=value, active=active, lppi=lppi)


def test_canonical_nine_fields_and_exact_domain_bytes():
    binding = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(_document())
    assert set(binding.document) == operation.BINDING_FIELDS
    assert len(binding.document) == 9
    assert binding.canonical_bytes == canonical_json_bytes(_document())
    assert binding.digest_sha256 == hashlib.sha256(binding.canonical_bytes).hexdigest()
    assert operation.DOMAIN == (
        b"CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1\x00"
    )
    assert operation.PURPOSE_DOMAIN == "CryptoHunter.Stage9.ProvisioningOperation.v1"
    assert operation.authenticated_operation_signed_bytes(binding.canonical_bytes) == (
        operation.DOMAIN + hashlib.sha256(binding.canonical_bytes).digest()
    )


@pytest.mark.parametrize("field", sorted(operation.BINDING_FIELDS))
def test_every_missing_field_is_rejected(field):
    document = _document()
    del document[field]
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(document)


@pytest.mark.parametrize("extra", ["caller_extension", "logical_operation_id", "account_id"])
def test_unknown_fields_are_rejected(extra):
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
            _document() | {extra: "caller-controlled"}
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 0),
        ("schema_version", "1"),
        ("schema_version", True),
        ("environment", "TEST_ONLY"),
        ("environment", "LPPI_TEST_ONLY"),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519"),
        ("pdsa_package_digest_sha256", "AB" * 32),
        ("pdsa_package_digest_sha256", "ab" * 31),
        ("pdsa_package_digest_sha256", "ab" * 32 + "\n"),
        ("provisioning_subject_id", "psub_caller"),
        ("enrollment_reference", "penr_caller"),
        ("provisioning_operation_id", "prvop_caller"),
        ("provisioning_operation_id", "prvop_019ba13c-5c00-4000-8000-000000000001"),
        ("provisioning_operation_id", "prvop_019BA13C-5c00-7000-8000-000000000001"),
        ("provisioning_operation_id", "prvop_019ba13c-5c00-7000-0000-000000000001"),
        ("binding_generation", 0),
        ("binding_generation", 2),
        ("binding_generation", True),
        ("binding_generation", "1"),
        ("created_at_utc", "2026-10-06T12:00:00Z"),
        ("created_at_utc", "2026-10-06T12:00:00.12Z"),
        ("created_at_utc", "2026-10-06T12:00:00.1234Z"),
        ("created_at_utc", "2026-10-06T12:00:00.123+00:00"),
        ("created_at_utc", "2026-10-06T12:00:00.123Z\n"),
        ("created_at_utc", "2026-02-30T12:00:00.123Z"),
        ("created_at_utc", "2026-10-06T12:00:00.124Z"),
        ("created_at_utc", None),
    ],
)
def test_wrong_schema_profile_identity_generation_or_timestamp_is_rejected(field, replacement):
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
            _document() | {field: replacement}
        )


@pytest.mark.parametrize("kind", ["space", "order", "duplicate", "newline", "array", "utf8"])
def test_noncanonical_json_is_rejected(kind):
    document = _document()
    canonical = canonical_json_bytes(document)
    raw = {
        "space": json.dumps(document).encode(),
        "order": json.dumps(document, separators=(",", ":")).encode(),
        "duplicate": b'{"schema_version":1,' + canonical[1:],
        "newline": canonical + b"\n",
        "array": b"[]",
        "utf8": b"\xff",
    }[kind]
    assert raw != canonical
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(raw)


@pytest.mark.parametrize("raw", [None, "{}", bytearray(b"{}"), b"", b" " * 8193])
def test_invalid_transport_type_empty_or_oversized_is_rejected(raw):
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(raw)


def test_reserved_uuidv7_uses_one_real_millisecond_instant_and_csprng(active, monkeypatch):
    captured, random_calls = [], []
    original = authorization.secrets.randbits

    def clock():
        captured.append(RESERVATION_NOW)
        return RESERVATION_NOW

    def randomness(bits):
        result = original(bits)
        random_calls.append((bits, result))
        return result

    monkeypatch.setattr(installed, "_utc_now", clock)
    monkeypatch.setattr(authorization.secrets, "randbits", randomness)
    value = installed.establish_installed_lppi_authenticated_operation(active)
    payload = value.binding.document
    identity = uuid.UUID(value.provisioning_operation_id.removeprefix("prvop_"))
    millis = authorization._reservation_epoch_milliseconds(RESERVATION_NOW)
    assert identity.version == 7 and identity.variant == uuid.RFC_4122
    assert value.provisioning_operation_id == "prvop_" + str(identity)
    assert identity.int >> 80 == millis
    assert millis % 1000 == 123
    assert payload["created_at_utc"].endswith(".123Z")
    assert captured == [RESERVATION_NOW]
    assert [bits for bits, _ in random_calls] == [12, 62]
    assert (identity.int >> 64) & ((1 << 12) - 1) == random_calls[0][1]
    assert identity.int & ((1 << 62) - 1) == random_calls[1][1]
    assert _state()["provisioning_operation_id"] == value.provisioning_operation_id


@pytest.mark.parametrize("field", ["prvop", "provisioning_operation_id", "uuid", "timestamp"])
def test_caller_cannot_choose_operation_identity_or_time(active, field):
    with pytest.raises(TypeError):
        installed.establish_installed_lppi_authenticated_operation(
            active, **{field: "caller-controlled"}
        )
    assert _state() is None


@pytest.mark.parametrize("source", [None, {}, "fingerprint", "cng-name", "TEST_ONLY-key"])
def test_raw_sources_cannot_start_operation(lppi, source):
    with pytest.raises((ValueError, RuntimeError)):
        installed.establish_installed_lppi_authenticated_operation(source)
    assert _state() is None


def test_authority_closed_after_current_guard_cannot_reserve_or_acquire_operation_locks(
    active, monkeypatch
):
    original = lifecycle.require_verified_active_lppi_authority_key

    def guard_then_close(value):
        verified = original(value)
        verified.close()
        return verified

    monkeypatch.setattr(lifecycle, "require_verified_active_lppi_authority_key", guard_then_close)
    with pytest.raises(
        operation.LPPIAuthenticatedOperationError,
        match="VERIFIED_ACTIVE_LPPI_AUTHORITY_KEY_REQUIRED",
    ):
        installed.establish_installed_lppi_authenticated_operation(active)
    path = installed._state_path()
    assert not path.exists()
    assert not path.with_name("initial-authenticated-operation.lock").exists()
    assert not path.with_name("initial-authenticated-operation-signing.lock").exists()


def test_complete_commit_retains_exact_sources_signature_and_immutable_result(committed):
    value, active = committed.value, committed.active
    raw, signature = value.binding.canonical_bytes, value.signature
    payload, state = value.binding.document, _state()
    source = operation.operation_authority_tuple(active)
    assert type(value) is operation.VerifiedLPPIAuthenticatedProvisioningOperation
    assert operation.require_verified_lppi_authenticated_operation(value) is value
    assert state["status"] == "AUTHENTICATED_OPERATION_COMMITTED"
    assert state["history"] == list(installed.STATUSES)
    assert state["purpose_domain"] == operation.PURPOSE_DOMAIN
    assert state["reservation_digest_sha256"] == operation.reservation_digest(source)
    assert state["active_authority_binding_raw_hex"] == active.binding.canonical_bytes.hex()
    assert state["binding_raw_hex"] == raw.hex()
    assert state["binding_digest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert state["signature_hex"] == signature.hex()
    for field, expected in source.items():
        assert state[field] == expected
        if field in operation.BINDING_FIELDS:
            assert payload[field] == expected
    for field in ("provisioning_operation_id", "binding_generation", "created_at_utc"):
        assert state[field] == payload[field]
    verified = operation.verify_authenticated_operation_binding(raw, signature, active)
    assert verified.canonical_bytes == raw
    committed.lppi.successor.private.public_key().verify(
        signature, operation.authenticated_operation_signed_bytes(raw), ec.ECDSA(hashes.SHA256())
    )
    r, s = utils.decode_dss_signature(signature)
    assert 0 < s <= P256_ORDER // 2 and utils.encode_dss_signature(r, s) == signature
    document = value.binding.document
    document["provisioning_operation_id"] = "caller-controlled"
    assert value.binding.canonical_bytes == raw
    with pytest.raises(TypeError):
        value.signature = b"caller-controlled"


@pytest.mark.parametrize("method", ["copy", "deepcopy", "new", "subclass", "transport", "record"])
def test_copy_reconstructed_or_subclassed_operation_never_grants_authority(committed, method):
    value = committed.value
    forged = {
        "copy": lambda: copy.copy(value),
        "deepcopy": lambda: copy.deepcopy(value),
        "new": lambda: object.__new__(type(value)),
        "subclass": lambda: object.__new__(type("TestOnlyForgedOperation", (type(value),), {})),
        "transport": lambda: operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
            value.binding.document
        ),
        "record": _state,
    }[method]()
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.require_verified_lppi_authenticated_operation(forged)
    assert operation.require_verified_lppi_authenticated_operation(value) is value


def test_public_capability_constructor_and_transport_verifier_cannot_issue_authority(committed):
    with pytest.raises(TypeError):
        operation.VerifiedLPPIAuthenticatedProvisioningOperation()
    parsed = operation.verify_authenticated_operation_binding(
        committed.value.binding.canonical_bytes, committed.value.signature, committed.active
    )
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.require_verified_lppi_authenticated_operation(parsed)


@pytest.mark.parametrize(
    "mutation",
    [
        "pre_enrollment_key",
        "second_p256_key",
        "test_only_key",
        "wrong_domain",
        "high_s",
        "nonminimal",
        "trailing",
        "empty",
    ],
)
def test_signature_requires_exact_active_key_domain_and_strict_low_s_der(committed, mutation):
    value, active, lppi = committed.value, committed.active, committed.lppi
    raw, signature = value.binding.canonical_bytes, value.signature
    message = operation.authenticated_operation_signed_bytes(raw)
    if mutation == "pre_enrollment_key":
        signature = _low_s(lppi.item.dll.private, message)
    elif mutation == "second_p256_key":
        signature = _low_s(ec.derive_private_key(89, ec.SECP256R1()), message)
    elif mutation == "test_only_key":
        signer = TestOnlyMembershipSigner()
        signature = signer.sign(message)
    elif mutation == "wrong_domain":
        signature = _low_s(
            lppi.successor.private,
            authority_binding.AUTHORITY_POP_DOMAIN + hashlib.sha256(raw).digest(),
        )
    elif mutation == "high_s":
        r, s = utils.decode_dss_signature(signature)
        signature = utils.encode_dss_signature(r, P256_ORDER - s)
    elif mutation == "nonminimal":
        signature = _nonminimal_der(signature)
    elif mutation == "trailing":
        signature += b"\x00"
    else:
        signature = b""
    with pytest.raises((ValueError, RuntimeError)):
        operation.verify_authenticated_operation_binding(raw, signature, active)


@pytest.mark.parametrize(
    "field",
    [
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "provisioning_operation_id",
    ],
)
def test_changed_valid_payload_byte_or_source_tuple_rejected(committed, field):
    value = committed.value
    changed = value.binding.document
    old = changed[field]
    changed[field] = old[:-1] + ("1" if old[-1] != "1" else "2")
    raw = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
        changed
    ).canonical_bytes
    with pytest.raises((ValueError, RuntimeError)):
        operation.verify_authenticated_operation_binding(raw, value.signature, committed.active)


def test_exact_retry_and_load_never_resign_or_remint(committed, monkeypatch):
    value, active = committed.value, committed.active
    raw, signature, identity = (
        value.binding.canonical_bytes,
        value.signature,
        value.provisioning_operation_id,
    )
    retained = installed._state_path().read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("committed operation retry attempted new randomness or native signing")

    monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", forbidden)
    monkeypatch.setattr(authorization.secrets, "randbits", forbidden)
    monkeypatch.setattr(installed, "_utc_now", forbidden)
    for retry in (
        installed.establish_installed_lppi_authenticated_operation(active),
        installed.load_installed_lppi_authenticated_operation(active),
    ):
        assert retry.provisioning_operation_id == identity
        assert retry.binding.canonical_bytes == raw and retry.signature == signature
    assert installed._state_path().read_bytes() == retained


def test_restart_clears_process_registry_and_reverifies_current_active(committed, monkeypatch):
    value, active = committed.value, committed.active
    raw, signature, identity = (
        value.binding.canonical_bytes,
        value.signature,
        value.provisioning_operation_id,
    )
    active.close()
    operation._ISSUED.clear()
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.require_verified_lppi_authenticated_operation(value)
    fresh = lifecycle.establish_installed_lppi_authority_key(committed.lppi.accept())

    def forbidden(*args, **kwargs):
        pytest.fail("restart attempted another operation signature or ID")

    monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", forbidden)
    monkeypatch.setattr(authorization.secrets, "randbits", forbidden)
    loaded = installed.load_installed_lppi_authenticated_operation(fresh)
    assert loaded.provisioning_operation_id == identity
    assert loaded.binding.canonical_bytes == raw and loaded.signature == signature
    assert operation.require_verified_lppi_authenticated_operation(loaded) is loaded
    fresh.close()


@pytest.mark.parametrize(
    "cut",
    [
        "before_reserve",
        "after_reserve",
        "before_sign",
        "during_sign",
        "after_sign",
        "before_commit",
        "after_commit",
    ],
)
def test_crash_cut_points_resume_same_durable_identity_and_retain_committed_signature(
    active, monkeypatch, cut
):
    original_write = installed._write
    original_sign = installed.sign_active_lppi_authenticated_operation_binding
    crashed, signed, snapshots = [], [], []

    def crash():
        crashed.append(True)
        raise RuntimeError("TEST_ONLY_OPERATION_CRASH")

    def write(path, state):
        reserved = state["status"] == "PRVOP_RESERVED"
        if (reserved and cut == "before_reserve") or (not reserved and cut == "before_commit"):
            crash()
        original_write(path, state)
        snapshots.append(dict(state))
        if (reserved and cut == "after_reserve") or (not reserved and cut == "after_commit"):
            crash()

    def sign(*args, **kwargs):
        if cut in {"before_sign", "during_sign"}:
            crash()
        signature = original_sign(*args, **kwargs)
        signed.append(signature)
        if cut == "after_sign":
            crash()
        return signature

    monkeypatch.setattr(installed, "_write", write)
    monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", sign)
    with pytest.raises(RuntimeError, match="TEST_ONLY_OPERATION_CRASH"):
        installed.establish_installed_lppi_authenticated_operation(active)
    assert crashed == [True]
    retained = _state()
    if cut == "before_reserve":
        assert retained is None
    else:
        assert retained is not None
    monkeypatch.setattr(installed, "_write", original_write)
    monkeypatch.setattr(
        installed, "sign_active_lppi_authenticated_operation_binding", original_sign
    )
    if retained is not None:

        def forbid_remint(*args, **kwargs):
            pytest.fail("durable reservation crash attempted second operation UUID")

        monkeypatch.setattr(authorization.secrets, "randbits", forbid_remint)
    if cut == "after_commit":

        def forbid_resign(*args, **kwargs):
            pytest.fail("lost response after durable commit attempted another signature")

        monkeypatch.setattr(
            installed, "sign_active_lppi_authenticated_operation_binding", forbid_resign
        )
    result = installed.establish_installed_lppi_authenticated_operation(active)
    if retained is not None:
        assert result.provisioning_operation_id == retained["provisioning_operation_id"]
        assert result.binding.canonical_bytes.hex() == retained["binding_raw_hex"]
        assert result.binding.document["created_at_utc"] == retained["created_at_utc"]
    if cut == "after_commit":
        assert signed and result.signature == signed[0]
        assert _state() == retained
    assert len({state["provisioning_operation_id"] for state in snapshots}) <= 1


@pytest.mark.parametrize(
    "field",
    [
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "provisioning_operation_id",
        "binding_generation",
        "created_at_utc",
        "signature_hex",
        "binding_digest_sha256",
        "lppi_authority_public_key_fingerprint_sha256",
        "custody_profile",
        "purpose_domain",
        "reservation_digest_sha256",
    ],
)
def test_retained_tuple_signature_and_signer_snapshot_mutation_fail_closed(committed, field):
    retained = _state()
    if field == "binding_generation":
        retained[field] = 2
    elif field == "signature_hex":
        retained[field] = "00"
    else:
        retained[field] = "changed"
    installed._state_path().write_bytes(canonical_json_bytes(retained))
    for consume in (
        lambda: operation.require_verified_lppi_authenticated_operation(committed.value),
        lambda: installed.load_installed_lppi_authenticated_operation(committed.active),
        lambda: installed.establish_installed_lppi_authenticated_operation(committed.active),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            consume()


@pytest.mark.parametrize(
    "mutation", ["fingerprint", "custody", "binding_digest", "unique_name", "tpmt_public"]
)
def test_current_active_requalification_blocks_committed_operation_after_tampering(
    committed, mutation
):
    lppi = committed.lppi
    if mutation in {"fingerprint", "custody", "binding_digest"}:
        state = lifecycle._read(lifecycle._state_path())
        document = committed.active.binding.document
        field = {
            "fingerprint": "lppi_authority_public_key_fingerprint_sha256",
            "custody": "custody_profile",
            "binding_digest": "tpm_creation_attestation_sha256",
        }[mutation]
        document[field] = "aa" * 32 if mutation != "custody" else "TEST_ONLY"
        state["binding_raw_hex"] = canonical_json_bytes(document).hex()
        lifecycle._write(lifecycle._state_path(), state)
    elif mutation == "unique_name":
        lppi.successor.properties[(22, "Unique Name")] = wide("TEST_ONLY-foreign-unique")
    else:
        lppi.successor_tbs.subject = bytes.fromhex("00" * 64)
    with pytest.raises((ValueError, RuntimeError)):
        operation.require_verified_lppi_authenticated_operation(committed.value)
    with pytest.raises((ValueError, RuntimeError)):
        installed.load_installed_lppi_authenticated_operation(committed.active)


def test_closed_active_revokes_committed_operation(committed):
    committed.active.close()
    with pytest.raises((ValueError, RuntimeError)):
        operation.require_verified_lppi_authenticated_operation(committed.value)


def test_production_membership_remains_blocked():
    with pytest.raises((ValueError, RuntimeError)):
        ProductionMembershipSignerUnavailable().sign(b"operation-must-not-issue-membership")


@pytest.mark.parametrize("field", sorted(installed._SOURCE_FIELDS))
def test_different_package_subject_enrollment_authority_or_trust_domain_has_stable_conflict(
    committed, field
):
    source = installed._source(committed.active)
    changed = source.document
    changed[field] = "another-authorized-lifecycle-identity"
    conflicting_source = replace(source, tuple_raw=canonical_json_bytes(changed))
    retained = installed._state_path().read_bytes()
    with installed._locked_state() as path:
        with pytest.raises(
            installed.LPPIAuthenticatedOperationError,
            match="^LPPI_AUTHENTICATED_OPERATION_CONFLICT$",
        ):
            installed._reserve(path, conflicting_source)
    assert installed._state_path().read_bytes() == retained


def test_committed_write_rejects_another_valid_signature_and_new_identity(committed):
    value = committed.value
    retained = _state()
    message = operation.authenticated_operation_signed_bytes(value.binding.canonical_bytes)
    replacement = _low_s(committed.lppi.successor.private, message)
    assert replacement != value.signature
    with pytest.raises(
        installed.LPPIAuthenticatedOperationError, match="^LPPI_AUTHENTICATED_OPERATION_CONFLICT$"
    ):
        installed._write(installed._state_path(), retained | {"signature_hex": replacement.hex()})
    assert _state() == retained


def test_durable_owner_rejects_semantically_valid_second_operation(committed):
    retained = _state()
    changed = dict(retained)
    document = committed.value.binding.document
    old = uuid.UUID(document["provisioning_operation_id"].removeprefix("prvop_"))
    document["provisioning_operation_id"] = "prvop_" + str(uuid.UUID(int=old.int ^ 1))
    binding = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(document)
    changed.update(
        provisioning_operation_id=document["provisioning_operation_id"],
        binding_raw_hex=binding.canonical_bytes.hex(),
        binding_digest_sha256=binding.digest_sha256,
    )
    with pytest.raises(
        installed.LPPIAuthenticatedOperationError, match="^LPPI_AUTHENTICATED_OPERATION_CONFLICT$"
    ):
        installed._write(installed._state_path(), changed)
    assert _state() == retained


def test_loader_never_mints_or_publishes_reserved_operation(active, monkeypatch):
    original_randomness = authorization.secrets.randbits

    def forbid_mint(*args, **kwargs):
        pytest.fail("trusted loader minted an operation identity")

    monkeypatch.setattr(authorization.secrets, "randbits", forbid_mint)
    with pytest.raises(
        installed.LPPIAuthenticatedOperationError,
        match="LPPI_AUTHENTICATED_OPERATION_NOT_COMMITTED",
    ):
        installed.load_installed_lppi_authenticated_operation(active)
    assert _state() is None
    monkeypatch.setattr(authorization.secrets, "randbits", original_randomness)
    source = installed._source(active)
    with installed._locked_state() as path:
        reserved = installed._reserve(path, source)
    assert reserved["status"] == "PRVOP_RESERVED" and reserved["signature_hex"] is None
    with pytest.raises(
        installed.LPPIAuthenticatedOperationError,
        match="LPPI_AUTHENTICATED_OPERATION_NOT_COMMITTED",
    ):
        installed.load_installed_lppi_authenticated_operation(active)
    assert _state() == reserved


def test_native_sign_and_live_requalification_do_not_hold_state_write_lock(active, monkeypatch):
    original_source = installed._source
    original_sign = installed.sign_active_lppi_authenticated_operation_binding
    observations = []

    def source(*args):
        with installed._locked_state():
            observations.append("source")
        return original_source(*args)

    def sign(*args, **kwargs):
        with installed._locked_state():
            observations.append("sign")
        return original_sign(*args, **kwargs)

    monkeypatch.setattr(installed, "_source", source)
    monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", sign)
    installed.establish_installed_lppi_authenticated_operation(active)
    assert observations.count("sign") == 1
    assert observations.count("source") >= 2


@pytest.mark.parametrize(
    "kind", ["missing", "noncanonical", "schema", "oversized", "hardlink", "symlink"]
)
def test_retained_file_boundary_rejects_unsafe_or_malformed_record(lppi, tmp_path, kind):
    path = installed._state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "missing":
        assert installed._read(path) is None
        return
    if kind == "noncanonical":
        path.write_bytes(b"{}\n")
    elif kind == "schema":
        path.write_bytes(b"{}")
    elif kind == "oversized":
        path.write_bytes(b" " * 1_048_577)
    elif kind == "hardlink":
        path.write_bytes(b"{}")
        os.link(path, tmp_path / "TEST_ONLY-hardlink")
    else:
        target = tmp_path / "TEST_ONLY-symlink-target"
        target.write_bytes(b"{}")
        path.symlink_to(target)
    with pytest.raises((ValueError, RuntimeError)):
        installed._read(path)


@pytest.mark.parametrize(
    "instant",
    [
        datetime(1969, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc),
        datetime(2026, 10, 6, 12, 0, 0),
        datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_invalid_reservation_clock_fails_before_durable_identity_or_native_sign(
    active, monkeypatch, instant
):
    monkeypatch.setattr(installed, "_utc_now", lambda: instant)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid reservation instant reached native signing")

    monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        installed.establish_installed_lppi_authenticated_operation(active)
    assert _state() is None


@pytest.mark.skipif(os.name == "nt", reason="TEST_ONLY inherited native ABI needs fork")
def test_two_processes_have_one_reservation_and_nonblocking_busy(active, monkeypatch):
    # Covered current-ACTIVE qualification can take longer under parallel CI.
    # Keep that setup budget separate from the real signing lock's BUSY deadline.
    requalification_timeout, busy_timeout, process_exit_timeout = 120, 5, 30
    context = multiprocessing.get_context("fork")
    released = context.Event()
    contender_lock_attempt = context.Event()
    first_reader, first_writer = context.Pipe(duplex=False)
    second_reader, second_writer = context.Pipe(duplex=False)
    original = installed.sign_active_lppi_authenticated_operation_binding
    original_locked_signing = installed._locked_signing

    @contextmanager
    def contender_locked_signing():
        contender_lock_attempt.set()
        with original_locked_signing() as path:
            yield path

    def slow_native_sign(*args, **kwargs):
        retained = _state()
        first_writer.send(("reserved", retained["provisioning_operation_id"]))
        if not released.wait(requalification_timeout + busy_timeout + process_exit_timeout):
            raise RuntimeError("TEST_ONLY_CONCURRENT_SIGN_RELEASE_TIMEOUT")
        return original(*args, **kwargs)

    def establish(pipe, contender=False):
        try:
            if contender:
                monkeypatch.setattr(installed, "_locked_signing", contender_locked_signing)
            value = installed.establish_installed_lppi_authenticated_operation(active)
            pipe.send(
                (
                    "ok",
                    value.provisioning_operation_id,
                    value.binding.canonical_bytes,
                    value.signature,
                )
            )
        except (ValueError, RuntimeError) as exc:
            pipe.send(("error", str(exc)))
        finally:
            pipe.close()

    monkeypatch.setattr(
        installed, "sign_active_lppi_authenticated_operation_binding", slow_native_sign
    )
    first = context.Process(target=establish, args=(first_writer,))
    second = context.Process(target=establish, args=(second_writer, True))
    first.start()
    try:
        assert first_reader.poll(requalification_timeout), (
            "first process did not durably reserve before signing"
        )
        status, identity = first_reader.recv()
        assert status == "reserved"
        second.start()
        assert contender_lock_attempt.wait(requalification_timeout), (
            "second process did not finish current-ACTIVE qualification before its lock attempt"
        )
        assert second_reader.poll(busy_timeout), (
            "second process blocked at the signing lock boundary instead of returning busy"
        )
        assert second_reader.recv() == ("error", "LPPI_AUTHENTICATED_OPERATION_BUSY")
        released.set()
        # The response includes three independent current-ACTIVE capability reads.
        # Allow native requalification under coverage; the busy deadline stays short.
        assert first_reader.poll(requalification_timeout), (
            "first process did not commit after signing was released"
        )
        status, committed_id, raw, signature = first_reader.recv()
        assert status == "ok" and committed_id == identity
        first.join(process_exit_timeout)
        second.join(process_exit_timeout)
        assert first.exitcode == second.exitcode == 0
        monkeypatch.setattr(installed, "sign_active_lppi_authenticated_operation_binding", original)
        retry = installed.establish_installed_lppi_authenticated_operation(active)
        assert retry.provisioning_operation_id == identity
        assert retry.binding.canonical_bytes == raw and retry.signature == signature
        assert _state()["provisioning_operation_id"] == identity
    finally:
        released.set()
        for process in (first, second):
            if process.pid is not None:
                process.join(process_exit_timeout)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
        first_reader.close()
        second_reader.close()
