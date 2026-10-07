"""TEST_ONLY native ABI checks for the production operation signing boundary."""

from __future__ import annotations

import copy
import hashlib
import inspect
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import lppi_authenticated_operation as operation
from bot_core.licensing import lppi_authority_key as authority_binding
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.pre_enrollment import P256_ORDER
from deployment import (
    windows_lppi_authority_key as native_key,
    windows_production_lppi_authority as lifecycle,
    windows_production_lppi_operation as operation_store,
)
from tests.licensing import test_lppi_authority_lifecycle as lifecycle_tests
from tests.deployment.test_windows_cng_pre_enrollment import wide

challenge_harness = lifecycle_tests.challenge_harness
integration = lifecycle_tests.integration
package = lifecycle_tests.package
lppi = lifecycle_tests.lppi


@pytest.fixture
def reserved(lppi, request):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    source = operation_store._source(active)
    with operation_store._locked_state() as path:
        state = operation_store._reserve(path, source)
    raw = bytes.fromhex(state["binding_raw_hex"])
    result = SimpleNamespace(
        active=active,
        key=lifecycle._ACTIVE[active].key,
        raw=raw,
        lppi=lppi,
        digest=hashlib.sha256(operation.authenticated_operation_signed_bytes(raw)).digest(),
    )
    try:
        if getattr(request, "param", "locked") == "unlocked":
            yield result
        else:
            with operation_store._locked_signing():
                yield result
    finally:
        active.close()


def _operation_signs(item):
    return [call for call in item.lppi.successor.calls if call == ("sign", item.digest)]


def test_native_operation_api_has_only_exact_active_and_binding_inputs():
    method = inspect.signature(
        native_key.WindowsLPPIAuthorityKey.sign_authenticated_operation_binding
    )
    assert set(method.parameters) == {"self", "binding_raw", "active_authority"}
    assert method.parameters["active_authority"].kind is inspect.Parameter.KEYWORD_ONLY
    entrypoint = inspect.signature(native_key.sign_active_lppi_authenticated_operation_binding)
    assert set(entrypoint.parameters) == {"active_authority", "binding_raw"}


def test_reserved_exact_active_successor_signs_only_operation_domain_and_normalizes_low_s(reserved):
    item = reserved
    signature = native_key.sign_active_lppi_authenticated_operation_binding(item.active, item.raw)
    signed = operation.authenticated_operation_signed_bytes(item.raw)
    assert signed == (
        b"CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1\x00"
        + hashlib.sha256(item.raw).digest()
    )
    authority_binding.verify_low_s_signature(
        item.key.public_key_bytes, signature, signed, error="invalid TEST_ONLY operation signature"
    )
    r, s = utils.decode_dss_signature(signature)
    assert utils.encode_dss_signature(r, s) == signature
    assert 0 < s <= P256_ORDER // 2
    assert len(_operation_signs(item)) == 1
    for domain in (
        authority_binding.AUTHORITY_POP_DOMAIN,
        authority_binding.CONTINUITY_DOMAIN,
        b"CryptoHunter.Stage9.ProvisioningMembershipBinding.v1\x00",
        b"CryptoHunter.Stage9.LPPI.PackageAcceptancePoP.v1\x00",
    ):
        with pytest.raises(authority_binding.LPPIAuthorityKeyError):
            authority_binding.verify_low_s_signature(
                item.key.public_key_bytes,
                signature,
                domain + hashlib.sha256(item.raw).digest(),
                error="invalid TEST_ONLY operation signature",
            )
    assert not hasattr(item.lppi.item.key, "sign_authenticated_operation_binding")


def test_native_direct_sign_requires_durable_reserved_exact_payload(reserved):
    item = reserved
    operation_store._state_path().unlink()
    with pytest.raises((RuntimeError, ValueError)):
        item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)
    assert not _operation_signs(item)


@pytest.mark.parametrize("reserved", ["unlocked"], indirect=True)
def test_native_direct_sign_rejects_already_committed_operation(reserved):
    item = reserved
    operation_store.establish_installed_lppi_authenticated_operation(item.active)
    assert len(_operation_signs(item)) == 1
    with operation_store._locked_signing():
        with pytest.raises((RuntimeError, ValueError), match="ALREADY_COMMITTED"):
            item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)
    assert len(_operation_signs(item)) == 1


@pytest.mark.parametrize("reserved", ["unlocked"], indirect=True)
def test_native_direct_sign_requires_current_signing_owner_even_for_durable_reservation(reserved):
    item = reserved
    assert operation_store._read(operation_store._state_path())["status"] == "PRVOP_RESERVED"
    with pytest.raises(RuntimeError, match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED"):
        item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)
    assert not _operation_signs(item)


@pytest.mark.parametrize("method", ["copy", "new", "subclass", "record", "native", "test_only"])
def test_native_operation_rejects_unissued_or_non_active_authority(reserved, method):
    item = reserved
    if method == "copy":
        forged = copy.copy(item.active)
    elif method == "new":
        forged = object.__new__(lifecycle.VerifiedActiveLPPIAuthorityKey)
    elif method == "subclass":
        forged = object.__new__(
            type("TestOnlyForgedActive", (lifecycle.VerifiedActiveLPPIAuthorityKey,), {})
        )
    elif method == "record":
        forged = item.active.active_key_record
    elif method == "native":
        forged = item.key
    else:
        forged = item.lppi.successor.private
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="REJECT_NON_ACTIVE"):
        native_key.sign_active_lppi_authenticated_operation_binding(forged, item.raw)
    assert not _operation_signs(item)


@pytest.mark.parametrize("status", lifecycle.STATUSES[:-1])
def test_native_operation_rejects_every_non_active_successor_lifecycle(reserved, status):
    item = reserved
    state = lifecycle._read(lifecycle._state_path())
    state["status"] = status
    state["history"] = list(lifecycle.STATUSES[: lifecycle.STATUSES.index(status) + 1])
    lifecycle._write(lifecycle._state_path(), state)
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="REJECT_NON_ACTIVE"):
        item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)
    assert not _operation_signs(item)


@pytest.mark.parametrize("method", ["copy", "new", "subclass", "pre_enrollment"])
def test_native_operation_rejects_every_key_other_than_exact_active_successor(reserved, method):
    item = reserved
    if method == "copy":
        forged = copy.copy(item.key)
    elif method == "subclass":
        forged = object.__new__(type("TestOnlyOtherKey", (native_key.WindowsLPPIAuthorityKey,), {}))
    elif method == "pre_enrollment":
        forged = item.lppi.item.key
    else:
        forged = object.__new__(native_key.WindowsLPPIAuthorityKey)
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="REJECT_NON_ACTIVE"):
        native_key.WindowsLPPIAuthorityKey.sign_authenticated_operation_binding(
            forged, item.raw, active_authority=item.active
        )
    assert not _operation_signs(item)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pdsa_package_digest_sha256", "ff" * 32),
        ("provisioning_subject_id", "psub_019019c2-0000-7000-8000-000000000002"),
        ("enrollment_reference", "penr_019019c2-0000-7000-8000-000000000002"),
    ],
)
def test_native_operation_rejects_different_exact_package_enrollment_source(reserved, field, value):
    item = reserved
    payload = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
        item.raw
    ).document
    payload[field] = value
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="SOURCE_CONFLICT"):
        item.key.sign_authenticated_operation_binding(
            canonical_json_bytes(payload), active_authority=item.active
        )
    assert not _operation_signs(item)


def test_native_operation_rejects_caller_selected_prvop_even_for_exact_active_enrollment(reserved):
    item = reserved
    payload = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
        item.raw
    ).document
    current = payload["provisioning_operation_id"]
    payload["provisioning_operation_id"] = current[:-1] + ("0" if current[-1] != "0" else "1")
    with pytest.raises((RuntimeError, ValueError)):
        item.key.sign_authenticated_operation_binding(
            canonical_json_bytes(payload), active_authority=item.active
        )
    assert not _operation_signs(item)


@pytest.mark.parametrize("malformation", ["wrong_key", "high_s", "trailing", "empty"])
def test_native_operation_validates_returned_signature_before_publication(
    reserved, monkeypatch, malformation
):
    item = reserved
    original = native_key._AuthorityNCryptAPI.sign_digest
    wrong_key = ec.derive_private_key(99, ec.SECP256R1())

    def sign(native, handle, digest):
        if digest != item.digest:
            return original(native, handle, digest)
        signature = wrong_key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        r, s = utils.decode_dss_signature(signature)
        if malformation == "high_s":
            return utils.encode_dss_signature(r, max(s, P256_ORDER - s))
        if malformation == "trailing":
            return signature + b"\x00"
        if malformation == "empty":
            return b""
        return utils.encode_dss_signature(r, min(s, P256_ORDER - s))

    monkeypatch.setattr(native_key._AuthorityNCryptAPI, "sign_digest", sign)
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="SIGNING_FAILED"):
        item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)


@pytest.mark.parametrize("mutation", ["unique_name", "active_record", "tpmt_public"])
def test_native_operation_requalifies_active_after_native_signature(
    reserved, monkeypatch, mutation
):
    item = reserved
    original = native_key._AuthorityNCryptAPI.sign_digest

    def sign_then_mutate(native, handle, digest):
        signature = original(native, handle, digest)
        if digest == item.digest:
            if mutation == "unique_name":
                item.lppi.successor.properties[(22, "Unique Name")] = wide("TEST_ONLY-changed")
            elif mutation == "active_record":
                state = lifecycle._read(lifecycle._state_path())
                state["authority_unique_name"] = "TEST_ONLY-changed"
                lifecycle._write(lifecycle._state_path(), state)
            else:
                item.lppi.successor_tbs.subject = lifecycle_tests._public(
                    ec.derive_private_key(91, ec.SECP256R1()), "pre_enrollment"
                )
        return signature

    monkeypatch.setattr(native_key._AuthorityNCryptAPI, "sign_digest", sign_then_mutate)
    with pytest.raises(native_key.WindowsLPPIAuthorityKeyError, match="REJECT_NON_ACTIVE"):
        item.key.sign_authenticated_operation_binding(item.raw, active_authority=item.active)
    state = operation_store._read(operation_store._state_path())
    assert state["status"] == "PRVOP_RESERVED"
    assert state["signature_hex"] is None
    assert len(_operation_signs(item)) == 1
