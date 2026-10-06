"""TEST_ONLY native ABI simulation with real package, AK and ECDSA signatures."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import struct
from datetime import timedelta
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import (
    lppi_authority_custody as custody,
    lppi_authority_key as binding,
    lppi_package_acceptance as acceptance,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.pre_enrollment import P256_ORDER
from deployment import (
    windows_cng_custody_bridge as bridge,
    windows_cng_pre_enrollment as cng,
    windows_lppi_authority_key as native_key,
    windows_production_lppi_authority as lifecycle,
    windows_stage9_production_trust as trust,
)
from tests.deployment.test_windows_cng_custody_bridge import TestOnlyTBSDLL, two_b
from tests.deployment.test_windows_cng_pre_enrollment import (
    TestOnlyFunction,
    TestOnlyNCryptDLL,
    set_number,
    wide,
)
from tests.licensing import test_production_pdsa_verifier as pdsa_tests
from tests.licensing.test_production_pre_enrollment import NOW
from tests.licensing.test_production_tpm_custody import _public

_signed = pdsa_tests._signed
challenge_harness = pdsa_tests.challenge_harness
integration = pdsa_tests.integration
package = pdsa_tests.package


def _native_properties(dll, tbs, *, creation_hash, key_name):
    dll.properties.update(
        {
            (11, "PCP_PLATFORMHANDLE"): tbs.context.to_bytes(
                ctypes.sizeof(ctypes.c_void_p), "little"
            ),
            (22, "Name"): wide(key_name),
            (22, "PCP_PLATFORMHANDLE"): struct.pack("<I", 0x80000001),
            (22, "PCP_KEY_CREATIONHASH"): creation_hash,
            (22, "PCP_KEY_CREATIONTICKET"): struct.pack(">HI", 0x8021, 0x40000001)
            + two_b(b"t" * 32),
            (33, "PCP_PLATFORMHANDLE"): struct.pack("<I", 0x80000002),
            (33, "Name"): wide("Stage9.Retained.AK"),
            (33, "Key Type"): struct.pack("<I", cng.MACHINE_KEY),
            (33, "PCP_KEY_USAGE_POLICY"): struct.pack("<I", 8),
            (33, "Export Policy"): bytes(4),
            (33, "PCP_EXPORT_ALLOWED"): b"\x00",
            (33, "PCP_PASSWORD_REQUIRED"): b"\x00",
        }
    )
    original = dll.NCryptOpenKey.function

    def open_ak(provider, output, name, legacy, flags):
        if name != "Stage9.Retained.AK":
            return original(provider, output, name, legacy, flags)
        dll.calls.append(("open_ak", name))
        set_number(output, 33, ctypes.c_size_t)
        return 0

    dll.NCryptOpenKey.function = open_ak


@pytest.fixture
def lppi(integration, package, monkeypatch, tmp_path):
    item = integration
    monkeypatch.setattr(
        acceptance,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    monkeypatch.setattr(acceptance, "_utc_now", lambda: NOW)
    monkeypatch.setattr(lifecycle, "_utc_now", lambda: NOW)
    pre_tbs = TestOnlyTBSDLL(item.publics["pre_enrollment"], item.publics["ak"], item.keys["ak"])
    _native_properties(item.dll, pre_tbs, creation_hash=b"\x66" * 32, key_name=cng.KEY_NAME)
    successor = TestOnlyNCryptDLL()
    successor.private = ec.derive_private_key(73, ec.SECP256R1())
    successor.properties[(22, "Unique Name")] = wide("TEST_ONLY-successor-unique")
    successor_tbs = TestOnlyTBSDLL(
        _public(successor.private, "pre_enrollment"), item.publics["ak"], item.keys["ak"]
    )
    successor_tbs.context = 0x87654321
    _native_properties(
        successor, successor_tbs, creation_hash=b"c" * 32, key_name=native_key.KEY_NAME
    )
    successor_native = object.__new__(native_key._AuthorityNCryptAPI)
    successor_native.dll = successor
    monkeypatch.setattr(native_key, "_load_native", lambda: successor_native)

    class TestOnlyRouter:
        def __init__(self):
            self.Tbsi_GetDeviceInfo = pre_tbs.Tbsi_GetDeviceInfo
            self.Tbsip_Submit_Command = TestOnlyFunction(self.submit)

        def submit(self, context, *arguments):
            target = pre_tbs if context == pre_tbs.context else successor_tbs
            return target.Tbsip_Submit_Command(context, *arguments)

    tbs_native = object.__new__(bridge._TBSCustodyAPI)
    tbs_native.dll = TestOnlyRouter()
    monkeypatch.setattr(bridge, "_load_tbs_native", lambda: tbs_native)
    monkeypatch.setattr(
        lifecycle, "resolve_paths", lambda: SimpleNamespace(state=tmp_path / "TEST_ONLY_MACHINE")
    )
    arguments = {
        **{
            name: value
            for name, value in item.arguments.items()
            if name not in {"challenge_store", "pending", "signature"}
        },
        "key": item.key,
        "request_signature": item.arguments["signature"],
        "ak_key_name": "Stage9.Retained.AK",
    }
    result = SimpleNamespace(
        item=item,
        package=package,
        successor=successor,
        pre_tbs=pre_tbs,
        successor_tbs=successor_tbs,
        arguments=arguments,
    )
    result.accept = lambda raw=None, **changes: acceptance.accept_production_lppi_package(
        package.raw if raw is None else raw, **(arguments | changes)
    )
    return result


def _state():
    return lifecycle._read(lifecycle._state_path())


def test_complete_real_crypto_lifecycle_exact_binding_and_restart(lppi):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    record, state = active.active_key_record, _state()
    assert record["status"] == "ACTIVE"
    assert state["history"] == list(lifecycle.STATUSES)
    document = active.binding.document
    assert len(document) == 22 and set(document) == binding.BINDING_FIELDS
    assert (
        document["lppi_authority_public_key_fingerprint_sha256"]
        == hashlib.sha256(lppi.successor_tbs.subject).hexdigest()
    )
    assert (
        document["lppi_authority_key_tpm_name"]
        == "000b" + document["lppi_authority_public_key_fingerprint_sha256"]
    )
    assert (
        document["pre_enrollment_public_key_fingerprint_sha256"]
        == hashlib.sha256(lppi.item.key.public_key_bytes).hexdigest()
    )
    retained = active.binding.canonical_bytes
    active.close()
    fresh = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert fresh.binding.canonical_bytes == retained
    assert _state() == state
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1
    assert lppi.item.dll.persisted
    fresh.close()


@pytest.mark.parametrize("field", sorted(binding.BINDING_FIELDS))
def test_binding_unknown_missing_or_changed_fields_rejected(field):
    from tests.architecture.test_stage9_lppi_authority_key_binding_contract import contract

    properties = contract()["binding_artifact"]["json_schema"]["properties"]
    document = {name: profile.get("const", "ab" * 32) for name, profile in properties.items()}
    document.update(
        cng_key_unique_name="TEST_ONLY-successor-unique",
        lppi_authority_key_tpm_name="000b" + "ab" * 32,
        provisioning_subject_id="psub_019ba13c-5c00-7000-8000-000000000001",
        enrollment_reference="penr_019ba13c-5c00-7000-8000-000000000002",
        created_at_utc="2026-10-06T12:00:00Z",
    )
    valid = binding.LPPIAuthorityKeyBindingV1.from_mapping(document)
    document.pop(field)
    with pytest.raises(binding.LPPIAuthorityKeyError):
        binding.LPPIAuthorityKeyBindingV1.from_mapping(document)
    with pytest.raises(binding.LPPIAuthorityKeyError):
        binding.LPPIAuthorityKeyBindingV1.from_mapping(valid.document | {"caller_extension": True})


@pytest.mark.parametrize(
    "change", ["quorum", "expired", "target", "signature", "exchange", "projection"]
)
def test_invalid_acceptance_creates_no_successor(lppi, monkeypatch, change):
    raw, arguments = lppi.package.raw, {}
    if change == "quorum":
        value = json.loads(raw)
        value["signatures"] = value["signatures"][:1]
        raw = canonical_json_bytes(value)
    elif change == "expired":
        monkeypatch.setattr(acceptance, "_utc_now", lambda: NOW + timedelta(days=2))
    elif change == "target":
        payload = dict(lppi.package.payload)
        payload["pre_enrollment_public_key_fingerprint_sha256"] = "aa" * 32
        raw = _signed(lppi.package, payload)
    elif change == "signature":
        arguments["request_signature"] = b"bad"
    elif change == "exchange":
        arguments["tpm_response_raw"] = b"{}"
    else:
        lppi.pre_tbs.ak = lppi.successor_tbs.subject
    with pytest.raises((ValueError, RuntimeError)):
        lppi.accept(raw, **arguments)
    assert not lppi.successor.persisted
    assert _state() is None
    assert not any(call[0] == "create" for call in lppi.successor.calls)


def test_fresh_package_bound_acceptance_nontransferable(lppi):
    first, second = lppi.accept(), lppi.accept()
    assert first.pop_challenge_raw != second.pop_challenge_raw
    challenge = acceptance.validate_package_acceptance_challenge(first.pop_challenge_raw)
    assert challenge["pdsa_package_digest_sha256"] == first.package.package_digest_sha256
    for field in acceptance.PACKAGE_ACCEPTANCE_FIELDS:
        changed = dict(challenge)
        changed[field] = "ff" * 32
        with pytest.raises(acceptance.LPPIPackageAcceptanceError):
            acceptance._verify_low_s(
                first.key.public_key_bytes,
                first.pop_signature,
                acceptance.PACKAGE_ACCEPTANCE_DOMAIN
                + hashlib.sha256(canonical_json_bytes(changed)).digest(),
            )


@pytest.mark.parametrize("kind", ["acceptance", "active", "native", "custody"])
@pytest.mark.parametrize("method", ["copy", "new", "subclass"])
def test_forged_or_copied_capability_never_establishes_authority(lppi, kind, method):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    verified_custody = custody.verify_lppi_authority_key_custody(
        bytes.fromhex(_state()["custody_raw_hex"]), accepted=accepted, key=snapshot.key
    )
    value, guard = {
        "acceptance": (accepted, acceptance.require_verified_lppi_package_acceptance),
        "active": (active, lifecycle.require_verified_active_lppi_authority_key),
        "native": (snapshot.key, native_key.require_verified_production_lppi_authority_key),
        "custody": (verified_custody, custody.require_verified_lppi_authority_key_custody),
    }[kind]
    if method == "copy":
        forged = copy.copy(value)
    elif method == "new":
        forged = object.__new__(type(value))
    else:
        forged = object.__new__(type("TestOnlyForged", (type(value),), {}))
    with pytest.raises((ValueError, RuntimeError)):
        guard(forged)
    active.close()


@pytest.mark.parametrize("gate", list(lifecycle.STATUSES))
def test_crash_after_each_durable_gate_restores_exact_candidate(lppi, monkeypatch, gate):
    original = lifecycle._write
    cut = {"done": False}

    def write_then_crash(path, state):
        original(path, state)
        if state["status"] == gate and not cut["done"]:
            cut["done"] = True
            raise RuntimeError("TEST_ONLY_CRASH")

    monkeypatch.setattr(lifecycle, "_write", write_then_crash)
    with pytest.raises(RuntimeError, match="TEST_ONLY_CRASH"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    retained = _state()
    monkeypatch.setattr(lifecycle, "_write", original)
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    state = _state()
    assert state["created_at_utc"] == retained["created_at_utc"]
    for field in (
        "authority_sec1_hex",
        "authority_unique_name",
        "binding_raw_hex",
        "continuity_signature_hex",
        "authority_pop_signature_hex",
    ):
        if retained[field] is not None:
            assert state[field] == retained[field]
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1
    active.close()


def test_ambiguous_create_reconciles_existing_without_remint(lppi):
    lppi.successor.finalize_error = 0x80090020
    lppi.successor.finalize_lost_response = True
    with pytest.raises(RuntimeError):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert _state()["status"] == "CREATION_RESERVED" and lppi.successor.persisted
    lppi.successor.finalize_error = 0
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert active.active_key_record["status"] == "ACTIVE"
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1
    active.close()


def test_missing_ambiguous_reserved_identity_never_reminted(lppi):
    lppi.successor.create_error = 0x80090020
    with pytest.raises(RuntimeError):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    lppi.successor.create_error = 0
    with pytest.raises(RuntimeError, match="RETAINED_AUTHORITY_KEY_MISSING"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1


def test_preexisting_unreserved_identity_not_adopted(lppi):
    lppi.successor.persisted = True
    with pytest.raises(RuntimeError, match="PREEXISTING_UNRESERVED_LPPI_AUTHORITY_IDENTITY"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert _state() is None


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "history",
        "binding_raw_hex",
        "authority_unique_name",
        "package_raw_hex",
        "projection_raw_hex",
        "provisioning_subject_id",
        "created_at_utc",
    ],
)
def test_active_record_mutation_rejects_runtime_and_restart(lppi, field):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    state = _state()
    if field == "history":
        state[field] = ["CREATION_RESERVED", "ACTIVE"]
    elif field == "status":
        state[field] = "CANDIDATE"
    elif field == "binding_raw_hex":
        document = active.binding.document
        document["lppi_authority_key_tpm_name"] = "000b" + "aa" * 32
        state[field] = canonical_json_bytes(document).hex()
    else:
        state[field] = "changed"
    lifecycle._write(lifecycle._state_path(), state)
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.require_verified_active_lppi_authority_key(active)
    active.close()
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())


@pytest.mark.parametrize(
    "property_name",
    [
        "Name",
        "Unique Name",
        "Algorithm Name",
        "Algorithm Group",
        "Length",
        "Key Type",
        "Key Usage",
        "Export Policy",
        "PCP_EXPORT_ALLOWED",
    ],
)
def test_cng_successor_properties_requalified_after_restart(lppi, property_name):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    active.close()
    old = lppi.successor.properties[(22, property_name)]
    lppi.successor.properties[(22, property_name)] = (
        wide("wrong")
        if property_name in {"Name", "Unique Name", "Algorithm Name", "Algorithm Group"}
        else b"\x01"
        if property_name == "PCP_EXPORT_ALLOWED"
        else struct.pack("<I", 1)
    )
    with pytest.raises(RuntimeError):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    lppi.successor.properties[(22, property_name)] = old


@pytest.mark.parametrize("proof", ["continuity", "pop"])
@pytest.mark.parametrize(
    "mutation", ["wrong_key", "high_s", "trailing", "missing", "changed_binding"]
)
def test_signature_gates_require_exact_key_and_strict_low_s(lppi, proof, mutation):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    state = _state()
    document = active.binding
    evidence = custody.LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(
        bytes.fromhex(state["custody_raw_hex"])
    )
    signature = bytes.fromhex(
        state[
            "continuity_signature_hex" if proof == "continuity" else "authority_pop_signature_hex"
        ]
    )
    message = (
        binding.continuity_signed_bytes(document.canonical_bytes)
        if proof == "continuity"
        else binding.authority_pop_signed_bytes(document.canonical_bytes)
    )
    if mutation == "wrong_key":
        other = ec.derive_private_key(79, ec.SECP256R1())
        signature = other.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(signature)
        signature = utils.encode_dss_signature(r, min(s, P256_ORDER - s))
    elif mutation == "high_s":
        r, s = utils.decode_dss_signature(signature)
        signature = utils.encode_dss_signature(r, P256_ORDER - s)
    elif mutation == "trailing":
        signature += b"\0"
    elif mutation == "missing":
        signature = b""
    else:
        changed = document.document
        changed["pdsa_package_digest_sha256"] = "aa" * 32
        document = binding.LPPIAuthorityKeyBindingV1.from_mapping(changed)
    with pytest.raises(binding.LPPIAuthorityKeyError):
        if proof == "continuity":
            binding.verify_authority_key_continuity(document, signature, accepted=accepted)
        else:
            binding.verify_authority_key_pop(
                document, signature, key=snapshot.key, evidence=evidence
            )
    active.close()


@pytest.mark.parametrize(
    "field",
    [
        "subject_tpmt_public_hex",
        "subject_tpm_name",
        "subject_creation_hash",
        "subject_creation_ticket_hex",
        "certify_creation_attest_hex",
        "certify_creation_signature_hex",
        "retained_ak_tpmt_public_hex",
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "cng_provider_name",
        "custody_profile",
    ],
)
def test_custody_tampering_rejected_independently(lppi, field):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    state = _state()
    value = json.loads(bytes.fromhex(state["custody_raw_hex"]))
    value[field] = (
        "wrong"
        if field in {"cng_provider_name", "custody_profile"}
        else bytes([bytes.fromhex(value[field])[0] ^ 1]).hex() + value[field][2:]
    )
    with pytest.raises((ValueError, RuntimeError)):
        custody.verify_lppi_authority_key_custody(
            canonical_json_bytes(value), accepted=accepted, key=snapshot.key
        )
    active.close()


@pytest.mark.parametrize(
    "offset,raw",
    [
        (0, b"\x00\x01"),
        (2, b"\x00\x04"),
        (4, b"\x00\x00\x00\x00"),
        (12, b"\x00\x10"),
        (16, b"\x00\x04"),
    ],
)
def test_tpm_public_wrong_type_name_algorithm_origin_scheme_curve_rejected(lppi, offset, raw):
    public = lppi.successor_tbs.subject
    changed = public[:offset] + raw + public[offset + len(raw) :]
    with pytest.raises(ValueError):
        custody.parse_lppi_authority_public(changed)


def test_open_only_pre_enrollment_missing_state_never_creates(lppi, tmp_path):
    calls = sum(call[0] == "create" for call in lppi.item.dll.calls)
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="RETAINED_PRE_ENROLLMENT_IDENTITY_REQUIRED"
    ):
        cng.WindowsCNGPreEnrollmentKey.open_existing(tmp_path / "absent")
    assert sum(call[0] == "create" for call in lppi.item.dll.calls) == calls


@pytest.mark.parametrize(
    "field",
    [
        "provisioning_subject_id",
        "enrollment_reference",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "pre_enrollment_request_digest_sha256",
        "pdsa_trust_domain",
    ],
)
def test_cross_context_package_cannot_replace_retained_lifecycle(lppi, field):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    retained = _state()
    payload = dict(lppi.package.payload)
    payload[field] = payload[field][:-1] + ("3" if payload[field][-1] != "3" else "4")
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.establish_installed_lppi_authority_key(
            lppi.accept(_signed(lppi.package, payload))
        )
    assert _state() == retained
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1
    active.close()


def test_same_pre_enrollment_key_rejected_as_successor(lppi):
    lppi.successor.private = lppi.item.dll.private
    with pytest.raises(RuntimeError, match="SUCCESSOR_AUTHORITY_KEY_MUST_BE_DISTINCT"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert _state()["status"] == "CREATION_RESERVED"


@pytest.mark.parametrize("profile", ["software", "wrong_provider", "exportable", "migratable"])
def test_production_successor_rejects_nonproduction_custody(lppi, profile):
    if profile == "software":
        lppi.successor.properties[(11, "Impl Type")] = struct.pack("<I", cng.SOFTWARE)
    elif profile == "wrong_provider":
        lppi.successor.properties[(11, "Name")] = wide("Microsoft Software Key Storage Provider")
    elif profile == "exportable":
        lppi.successor.properties[(22, "PCP_EXPORT_ALLOWED")] = b"\x01"
    else:
        raw = lppi.successor_tbs.subject
        lppi.successor_tbs.subject = raw[:4] + struct.pack(">I", 0x40070) + raw[8:]
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert _state()["status"] == "CREATION_RESERVED"


def test_active_live_tpm_public_change_rejected_even_with_same_cng_properties(lppi):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    lppi.successor_tbs.subject = _public(
        ec.derive_private_key(97, ec.SECP256R1()), "pre_enrollment"
    )
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.require_verified_active_lppi_authority_key(active)
    active.close()


def test_missing_active_key_never_reminted(lppi):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    active.close()
    lppi.successor.persisted = False
    with pytest.raises(RuntimeError, match="RETAINED_AUTHORITY_KEY_MISSING"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert sum(call[0] == "create" for call in lppi.successor.calls) == 1


def test_public_active_rows_or_duplicate_records_do_not_construct_capability(lppi):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    record = active.active_key_record
    with pytest.raises(RuntimeError, match="VERIFIED_ACTIVE_LPPI_AUTHORITY_KEY_REQUIRED"):
        lifecycle.require_verified_active_lppi_authority_key(record)
    path = lifecycle._state_path()
    retained = path.read_bytes()
    path.write_bytes(b"[" + retained + b"," + retained + b"]")
    with pytest.raises(RuntimeError):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    active.close()


def test_successor_native_abi_has_fixed_machine_key_name(monkeypatch):
    dll = TestOnlyNCryptDLL()
    dll.properties[(22, "Name")] = wide(native_key.KEY_NAME)
    monkeypatch.setattr(cng.sys, "platform", "win32")
    monkeypatch.setattr(cng.ctypes, "WinDLL", lambda *args, **kwargs: dll, raising=False)
    native = native_key._AuthorityNCryptAPI()
    assert dll.NCryptCreatePersistedKey.argtypes[0] is cng._HANDLE
    assert dll.NCryptSignHash.restype is cng._STATUS
    provider = native.open_provider()
    native.create_key(provider)
    assert ("create", cng.ALGORITHM, native_key.KEY_NAME, cng.MACHINE_KEY) in dll.calls
    assert not any(call[0] == "create" and call[2] == cng.KEY_NAME for call in dll.calls)


def test_installed_entrypoint_opens_retained_key_only_and_fixed_trust(lppi, monkeypatch):
    from deployment.platforms import windows

    directory = lifecycle.resolve_paths().state / "PreEnrollment"
    directory.mkdir(parents=True)
    retained = lppi.item.key._state_directory / "pre-enrollment-key-identity.json"
    (directory / retained.name).write_bytes(retained.read_bytes())
    monkeypatch.setattr(
        windows, "production_trust_package_path", lambda ceremony: directory / ceremony
    )
    monkeypatch.setattr(trust, "load_production_trust", lambda path: lppi.item.authority.context)
    arguments = {
        name: value for name, value in lppi.arguments.items() if name not in {"key", "context"}
    }
    before = sum(call[0] == "create" for call in lppi.item.dll.calls)
    active = lifecycle.establish_installed_lppi_authority_from_artifacts(
        lppi.package.raw, **arguments
    )
    assert active.active_key_record["status"] == "ACTIVE"
    assert sum(call[0] == "create" for call in lppi.item.dll.calls) == before
    assert active in lifecycle._OWNED_PRE_ENROLLMENT
    active.close()


def test_installed_entrypoint_missing_prekey_state_creates_no_authority(lppi):
    arguments = {
        name: value for name, value in lppi.arguments.items() if name not in {"key", "context"}
    }
    with pytest.raises(cng.WindowsCNGPreEnrollmentError):
        lifecycle.establish_installed_lppi_authority_from_artifacts(lppi.package.raw, **arguments)
    assert not lppi.successor.persisted and _state() is None


def test_wrong_ak_signature_with_valid_tpm_signature_encoding_rejected(lppi):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    evidence = json.loads(bytes.fromhex(_state()["custody_raw_hex"]))
    signature = bytes.fromhex(evidence["certify_creation_signature_hex"])
    evidence["certify_creation_signature_hex"] = (signature[:-1] + bytes([signature[-1] ^ 1])).hex()
    with pytest.raises(ValueError, match="INVALID_PRODUCTION_CERTIFY_SIGNATURE"):
        custody.verify_lppi_authority_key_custody(
            canonical_json_bytes(evidence), accepted=accepted, key=snapshot.key
        )
    active.close()


def test_live_wrong_ak_signer_cannot_accept_package(lppi):
    lppi.pre_tbs.private = ec.derive_private_key(101, ec.SECP256R1())
    with pytest.raises(ValueError, match="INVALID_PRODUCTION_CERTIFY_SIGNATURE"):
        lppi.accept()
    assert not lppi.successor.persisted and _state() is None


def test_crash_after_creation_attempt_marker_before_native_effect_has_no_authority(
    lppi, monkeypatch
):
    original = lifecycle._write

    def marker_then_crash(path, state):
        original(path, state)
        if state["creation_attempted"]:
            raise RuntimeError("TEST_ONLY_CRASH_BEFORE_CNG_CREATE")

    monkeypatch.setattr(lifecycle, "_write", marker_then_crash)
    with pytest.raises(RuntimeError, match="TEST_ONLY_CRASH_BEFORE_CNG_CREATE"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert not lppi.successor.persisted
    monkeypatch.setattr(lifecycle, "_write", original)
    with pytest.raises(RuntimeError, match="RETAINED_AUTHORITY_KEY_MISSING"):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    assert not any(call[0] == "create" for call in lppi.successor.calls)


@pytest.mark.parametrize(
    "field",
    [
        "acceptance_challenge_raw_hex",
        "acceptance_signature_hex",
        "acceptance_live_attestation_hex",
        "acceptance_live_signature_hex",
    ],
)
def test_retained_acceptance_history_corruption_rejected(lppi, field):
    active = lifecycle.establish_installed_lppi_authority_key(lppi.accept())
    state = _state()
    active.close()
    raw = bytes.fromhex(state[field])
    state[field] = (raw[:-1] + bytes([raw[-1] ^ 1])).hex()
    lifecycle._write(lifecycle._state_path(), state)
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle.establish_installed_lppi_authority_key(lppi.accept())


def test_both_signature_domains_bind_every_package_request_device_and_subject_field(lppi):
    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    state = _state()
    evidence = custody.LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(
        bytes.fromhex(state["custody_raw_hex"])
    )
    for field in (
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "pre_enrollment_public_key_fingerprint_sha256",
        "lppi_authority_public_key_fingerprint_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
    ):
        document = active.binding.document
        changed = document[field][:-1] + ("3" if document[field][-1] != "3" else "4")
        document[field] = changed
        if field == "lppi_authority_public_key_fingerprint_sha256":
            document["lppi_authority_tpmt_public_sha256"] = changed
            document["lppi_authority_key_tpm_name"] = "000b" + changed
        candidate = binding.LPPIAuthorityKeyBindingV1.from_mapping(document)
        with pytest.raises(binding.LPPIAuthorityKeyError):
            binding.verify_authority_key_continuity(
                candidate, bytes.fromhex(state["continuity_signature_hex"]), accepted=accepted
            )
        with pytest.raises(binding.LPPIAuthorityKeyError):
            binding.verify_authority_key_pop(
                candidate,
                bytes.fromhex(state["authority_pop_signature_hex"]),
                key=snapshot.key,
                evidence=evidence,
            )
    active.close()


def test_valid_ak_signature_cannot_substitute_another_creation_ticket(lppi):
    from tests.licensing.test_production_tpm_custody import _attest, _signature

    accepted = lppi.accept()
    active = lifecycle.establish_installed_lppi_authority_key(accepted)
    snapshot = lifecycle._ACTIVE[active]
    document = json.loads(bytes.fromhex(_state()["custody_raw_hex"]))
    ticket = bytes.fromhex(document["subject_creation_ticket_hex"])
    document["subject_creation_ticket_hex"] = (ticket[:-1] + bytes([ticket[-1] ^ 1])).hex()
    qualifier = custody.lppi_authority_custody_qualifying_data(
        {field: document[field] for field in custody.BINDING_FIELDS}
    )
    attest = _attest(
        bytes.fromhex(document["subject_tpm_name"]),
        bytes.fromhex(document["subject_creation_hash"]),
        qualifier,
    )
    # Preserve exact retained AK Qualified Name while generating a real signature.
    attest = attest[:8] + bytes.fromhex(document["retained_ak_qualified_name"]) + attest[42:]
    document["certify_creation_attest_hex"] = attest.hex()
    document["certify_creation_signature_hex"] = _signature(lppi.item.keys["ak"], attest).hex()
    document["tpm_creation_attestation_sha256"] = hashlib.sha256(attest).hexdigest()
    custody.LPPIAuthorityKeyCustodyEvidenceV1.from_mapping(document)
    with pytest.raises(
        custody.LPPIAuthorityCustodyError, match="LPPI_AUTHORITY_NATIVE_CREATION_BINDING_MISMATCH"
    ):
        custody.verify_lppi_authority_key_custody(
            canonical_json_bytes(document), accepted=accepted, key=snapshot.key
        )
    active.close()
